"""Databricks layer, exercised through LocalLake / LocalModelStore / a fake Vector Search client.
Spark (DeltaLake), MLflow (MlflowModelStore) and the real Vector Search service are NOT exercised here."""
import os, re, warnings; warnings.filterwarnings("ignore")
import pandas as pd, pytest
from procintel import Copilot, PackageSpec
from procintel.lakehouse import LocalLake
from procintel.mlops import LocalModelStore
from procintel.pipeline import MODEL_NAME, alerts, bronze_ingest, learning_cycle, load_platform, silver_gold, train
from procintel.synth import generate, write_dataset
from procintel.vectorstore import VectorSearchRetriever


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("lh"); base = generate(seed=3, n_projects=60)
    write_dataset(base, str(d / "base"))
    lake, store = LocalLake(str(d / "lake")), LocalModelStore(str(d / "models"))
    bronze_ingest(lake, str(d / "base")); sg = silver_gold(lake); v = train(lake, store)
    return dict(d=d, lake=lake, store=store, base=base, sg=sg, v=v)


def test_merge_is_an_upsert():
    import tempfile
    l = LocalLake(tempfile.mkdtemp()); l.write("t", pd.DataFrame(dict(k=[1, 2], v=["a", "b"])))
    l.merge("t", pd.DataFrame(dict(k=[2, 3], v=["B", "c"])), ["k"])
    assert l.read("t").sort_values("k").v.tolist() == ["a", "B", "c"]


def test_medallion_tables_and_idempotency(env):
    lake = env["lake"]
    for t in ["bronze_contracts", "bronze_docs", "silver_contracts", "silver_chunks", "gold_contract_features", "gold_chunks", "gold_kg_edges", "gold_conflicts"]:
        assert lake.exists(t), t
    assert "EVIDENCE" in set(lake.read("gold_kg_edges").rel)
    assert 0 < len(lake.read("gold_chunks_vs")) < len(lake.read("gold_chunks"))     # Vector Search embeds only the subset
    n = len(lake.read("silver_chunks"))
    assert bronze_ingest(lake, str(env["d"] / "base"))["docs_new"] == 0          # same content hash => no new bronze docs
    assert silver_gold(lake)["docs_chunked"] == 0 and len(lake.read("silver_chunks")) == n


def test_registry_champion_alias(env):
    store = env["store"]
    assert env["v"][1] and store.has(MODEL_NAME) and "mae_days" in store.metrics(MODEL_NAME)


def test_platform_from_lake_matches_in_memory_behaviour(env):
    plat = load_platform(env["lake"], env["store"])
    rep = Copilot(plat).analyze(PackageSpec("Rehabilitation of 50 km rural roads", "Civil Works", "Transport", "KEN", 12.0, "ICB"))
    assert [f for f in rep["findings"] if f["kind"] == "risk"] and not rep["rejected"]
    a = alerts(env["lake"], plat, "2025-06-30"); assert {"project_id", "risk_score", "flags"} <= set(a.columns)


class FakeIndex:
    """Mimics VectorSearchIndex.similarity_search: IN-filters on columns, returns {'result': {'data_array': [[id, score]]}}."""
    def __init__(self, chunks): self.chunks, self.calls = chunks, []

    def similarity_search(self, query_text, columns, filters=None, num_results=10, query_type="ANN"):
        self.calls.append(dict(filters=filters, n=num_results, query_type=query_type))
        q = set(re.findall(r"[a-z]{4,}", query_text.lower()))
        rows = []
        for c in self.chunks:
            if all(getattr(c, k) in v for k, v in (filters or {}).items()):
                s = len(q & set(re.findall(r"[a-z]{4,}", c.text.lower())))
                if s:
                    rows.append([c.chunk_id, s / 100])
        return {"result": {"data_array": sorted(rows, key=lambda r: -r[1])[:num_results]}}


def test_vector_search_backend_drives_the_copilot(env):
    holder = {}
    def factory(chunks):
        holder["idx"] = FakeIndex(chunks); return VectorSearchRetriever(holder["idx"], chunks)
    plat = load_platform(env["lake"], env["store"], factory)
    rep = Copilot(plat).analyze(PackageSpec("Rehabilitation of 50 km rural roads", "Civil Works", "Transport", "KEN", 12.0, "ICB"))
    risks = [f for f in rep["findings"] if f["kind"] == "risk"]
    assert risks and not rep["rejected"]
    assert holder["idx"].calls[0]["filters"] == {"doc_type": ["ICR", "PPAR"], "section_kind": ["lessons"]}
    assert all(c["query_type"] == "HYBRID" for c in holder["idx"].calls)



def test_vector_search_request_never_exceeds_the_hybrid_limit(env):
    """Hybrid Vector Search rejects num_results > 200 (HTTP 400), which silently dropped the app to in-memory search."""
    chunks = load_platform(env["lake"], env["store"]).index.chunks[:50]
    idx = FakeIndex(chunks)
    VectorSearchRetriever(idx, chunks).search("road works delays", k=48)
    assert idx.calls[-1]["n"] <= 200


def test_learning_cycle_detects_drift_and_promotes(env):
    new = generate(seed=99, n_projects=40, id_start=900000, approval_years=(2016, 2020), cut="2026-06-30", duration_shift=1.35, gov=env["base"].governance)
    write_dataset(new, str(env["d"] / "b1"))
    r = learning_cycle(env["lake"], env["store"], str(env["d"] / "b1"))
    assert r["new_contracts"] > 20 and r["monitor"]["retrain"] and r["retrain"]["promoted"]
    assert env["store"].reg[MODEL_NAME]["aliases"]["champion"] >= 2
    assert env["store"].metrics(MODEL_NAME)["mae_days"] == pytest.approx(r["retrain"]["mae_challenger"], abs=0.1)


def test_serving_snapshot_reproduces_copilot_output(env, tmp_path):
    import json
    from procintel.serving import export_snapshot, load_copilot, report_to_json
    path = export_snapshot(env["lake"], env["store"], str(tmp_path / "snap"))
    spec = PackageSpec("Rehabilitation of 50 km rural roads", "Civil Works", "Transport", "KEN", 12.0, "ICB")
    live = Copilot(load_platform(env["lake"], env["store"])).analyze(spec)
    served = json.loads(report_to_json(load_copilot(path).analyze(spec)))
    assert [f["claim"] for f in served["findings"]] == [f["claim"] for f in live["findings"]]
    assert served["schedule"]["plan_days"] == live["schedule"]["plan_days"]


def test_mlflow_store_register_alias_and_load(env, tmp_path, monkeypatch):
    """MlflowModelStore on a local sqlite registry: same alias API as Unity Catalog; signature is required there."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path); db = "mlflow.db"                    # relative: absolute Windows paths break sqlite URIs
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    from procintel.mlops import MlflowModelStore, register_champion
    store = MlflowModelStore("t", experiment="procintel_test", registry_uri=f"sqlite:///{db}")
    models = env["store"].load(MODEL_NAME)
    v, champ = register_champion(store, MODEL_NAME, models)
    assert champ and store.has(MODEL_NAME, "champion") and store.metrics(MODEL_NAME)["mae_days"] > 0
    mv = mlflow.MlflowClient().get_model_version_by_alias(f"t.{MODEL_NAME}", "champion")
    pf = mlflow.pyfunc.load_model(f"models:/t.{MODEL_NAME}@champion")
    assert pf.metadata.signature is not None and len(pf.predict(models.example_)) == 5
    assert store.load(MODEL_NAME).predict_df(models.example_).shape[0] == 5 and int(mv.version) == v


def test_app_serves_from_lakehouse_snapshot(env, tmp_path):
    """What the Databricks App does: snapshot exported by notebook 05 -> copied elsewhere (Volume -> app disk) -> UI state."""
    import shutil, app
    from procintel.serving import export_snapshot
    src = export_snapshot(env["lake"], env["store"], str(tmp_path / "vol"))
    moved = shutil.copytree(src, tmp_path / "app_disk")                      # model registry paths must not be absolute
    import os; os.environ["PROCINTEL_OFFLINE"] = "1"; os.environ["PROCINTEL_EVENTS_DIR"] = str(tmp_path / "events")
    app.build_state(None, str(moved))
    assert app.STATE["status"] == "ready", app.STATE.get("error")
    m = app.meta()
    assert m["source"].startswith("Databricks lakehouse snapshot") and m["summary"]["passages"] > 100
    assert app.insights()["model"]["mae_days"] > 0


def test_chunker_version_bump_rechunks_without_stale_chunks(env, monkeypatch):
    import procintel.pipeline as P
    lake = env["lake"]
    monkeypatch.setattr(P, "CHUNKER_VERSION", 999)
    from procintel.ingest import chunk_document as real
    monkeypatch.setattr(P, "chunk_document", lambda d: real(d)[:1])          # new "chunker": one chunk per doc
    r = silver_gold(lake)
    s, g = lake.read("silver_chunks"), lake.read("gold_chunks")
    assert r["docs_chunked"] == s.doc_id.nunique() == len(s) == len(g) and set(s.chunk_id) == set(g.chunk_id)


def test_snapshot_pickles_carry_no_producer_objects(tmp_path):
    """Spark Connect's toPandas() stores query metrics in df.attrs; pickling them made the App (no pyspark) crash on load."""
    import pickle
    df = pd.DataFrame(dict(a=[1])); df.attrs["metrics"] = object()
    LocalLake(str(tmp_path)).write("t", df)
    assert pickle.load(open(tmp_path / "t.pkl", "rb")).attrs == {}


def test_schema_change_promotes_even_with_worse_metric(tmp_path):
    """A champion pickled from an older model class cannot be served by new code; its metrics also come from another evaluation."""
    from procintel.mlops import register_champion
    store = LocalModelStore(str(tmp_path))
    from types import SimpleNamespace as M
    assert register_champion(store, "m", M(metrics=dict(mae_days=50.0, model_version=1)))[1]
    assert register_champion(store, "m", M(metrics=dict(mae_days=90.0, model_version=2)))[1]      # schema change: promoted
    assert not register_champion(store, "m", M(metrics=dict(mae_days=95.0, model_version=2)))[1]  # same schema, worse: not


def test_incremental_batch_keeps_other_projects_open_tenders(tmp_path):
    """A one-country batch must replace open tenders only for ITS projects (workspace bug with a Rwanda batch)."""
    from procintel.pipeline import _open_snapshot
    lake = LocalLake(str(tmp_path))
    t1, t2 = pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-03")
    lake.write("bronze_open_tenders", pd.DataFrame(dict(project_id=["P1", "P1", "P2"], tender_id=["a", "b", "c"], _ingest_ts=t1, _source="base")), "append")
    lake.write("bronze_open_tenders", pd.DataFrame(dict(project_id=["P1", "P9"], tender_id=["b", "z"], _ingest_ts=t2, _source="incoming")), "append")
    snap = _open_snapshot(lake)
    assert sorted(snap.tender_id) == ["b", "c", "z"]      # P1 refreshed (a signed since), P2 kept, P9 new


def test_dropped_duplicate_documents_lose_their_gold_chunks(tmp_path):
    """Chunks of a doc that is no longer in the dataset (a duplicate copy chunked before de-duplication existed) are deleted
    from silver AND gold, or they stay searchable in the app and in Vector Search."""
    from procintel.lakehouse import LocalLake
    from procintel.pipeline import bronze_ingest, silver_gold
    from procintel.synth import generate, write_dataset
    raw = tmp_path / "raw"; write_dataset(generate(seed=3, n_projects=30), str(raw))
    lake = LocalLake(str(tmp_path / "lake")); bronze_ingest(lake, str(raw)); silver_gold(lake)
    for t in ("silver_chunks", "gold_chunks"):                 # an old copy's chunks, left over from an earlier run
        df = lake.read(t); ghost = df[df.doc_id == df.doc_id.iloc[0]].copy()
        ghost["doc_id"] = "ISR_P000000_GHOST"; ghost["chunk_id"] = ghost.chunk_id + "-ghost"
        lake.write(t, __import__("pandas").concat([df, ghost], ignore_index=True))
    silver_gold(lake)
    assert "ISR_P000000_GHOST" not in set(lake.read("silver_chunks").doc_id) | set(lake.read("gold_chunks").doc_id)


def test_new_batches_skips_ingested_and_incomplete_folders(env, tmp_path):
    from procintel.pipeline import new_batches
    inc = tmp_path / "incoming"
    done, fresh, partial = inc / "2026-11-02", inc / "2026-11-09", inc / "2026-11-16"
    for d in (done, fresh, partial):
        (d / "docs").mkdir(parents=True)
    (done / "projects.json").write_text("[]"); (fresh / "projects.json").write_text("[]")    # partial: projects.json not yet copied
    lake = LocalLake(str(tmp_path / "lake")); lake.write("bronze_projects", pd.DataFrame(dict(project_id=["P123456"], _source=[str(done)])))
    assert new_batches(lake, str(inc)) == [str(fresh)]
    assert new_batches(lake, str(tmp_path / "missing")) == []


def test_publish_copies_projects_json_last_and_skips_unchanged(tmp_path, monkeypatch):
    import shutil
    from procintel import live_wb
    src, dest = tmp_path / "staging", tmp_path / "base"
    (src / "docs").mkdir(parents=True)
    for f, t in (("contracts.csv", "a"), ("docs/ICR_P123456_1.txt", "text"), ("projects.json", "[]"), ("plan_activities.csv", "p")):
        (src / f).write_text(t)
    order = []
    real = shutil.copyfile
    monkeypatch.setattr(shutil, "copyfile", lambda a, b: (order.append(os.path.basename(b)), real(a, b))[1])
    assert live_wb.publish(str(src), str(dest)) == 4 and order[-1] == "projects.json"
    assert (dest / "docs" / "ICR_P123456_1.txt").read_text() == "text"
    order.clear()
    assert live_wb.publish(str(src), str(dest)) == 1 and order == ["projects.json"]     # unchanged files are not copied again
    (src / "projects.json").unlink()
    with pytest.raises(FileNotFoundError):
        live_wb.publish(str(src), str(tmp_path / "x"))                                     # an unfinished pull is never published


def test_fetch_settings_are_one_set_and_notebooks_fetch_plans():
    import yaml
    from procintel import live_wb
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v = yaml.safe_load(open(os.path.join(root, "databricks.yml"), encoding="utf-8"))["variables"]
    assert v["countries"]["default"] == live_wb.DEFAULT_COUNTRIES
    assert (int(v["max_projects"]["default"]), int(v["max_docs"]["default"])) == (live_wb.MAX_PROJECTS, live_wb.MAX_DOCS)
    for nb in ("01_bronze_ingest.py", "00_fetch_live.py"):          # fetch_all = build + fetch_plans: plans are never skipped
        src = open(os.path.join(root, "databricks", "notebooks", nb), encoding="utf-8").read()
        assert "fetch_all(" in src and "publish(" in src and "build(" not in src, nb


def test_pdf_links_come_from_document_metadata():
    from procintel import doc_links
    from procintel.core import Platform
    from procintel.ingest import Dataset
    doc_links._CACHE = {}
    try:
        d = dict(doc_id="ICR_P123456_99", doc_type="ICR", project_id="P123456", filename="x.txt", text="t",
                 meta=dict(Country="Kenya", pdf_url="https://documents.worldbank.org/y.pdf"))
        Platform(Dataset(*[pd.DataFrame()] * 5, [d], {}))
        assert doc_links.page_link("ICR_P123456_99", 4) == "https://documents.worldbank.org/y.pdf#page=4"
    finally:
        doc_links._CACHE = None


def test_snapshot_carries_a_version_the_app_can_poll(env, tmp_path):
    from procintel.serving import export_snapshot
    path = export_snapshot(env["lake"], env["store"], str(tmp_path / "snap"))
    v = open(os.path.join(path, "_version.txt")).read()
    assert re.fullmatch(r"\d{8}T\d{6}", v)


def test_learning_cycle_ingests_each_folder_once(env):
    from procintel.pipeline import new_batches
    b1 = str(env["d"] / "b1")                                   # ingested by test_learning_cycle_detects_drift_and_promotes
    if not os.path.isfile(os.path.join(b1, "projects.json")):
        pytest.skip("depends on the drift test's batch")
    n = len(env["lake"].read("bronze_projects"))
    assert b1 not in new_batches(env["lake"], b1, include_root=True)
    assert learning_cycle(env["lake"], env["store"], b1)["status"] == "nothing new to ingest"
    assert len(env["lake"].read("bronze_projects")) == n


def test_bronze_reingest_adds_no_copies_but_keeps_snapshots_whole(tmp_path):
    from procintel.pipeline import new_batches
    base = generate(seed=5, n_projects=20); d = tmp_path / "raw"; write_dataset(base, str(d))
    lake = LocalLake(str(tmp_path / "lake"))
    bronze_ingest(lake, str(d)); n = {t: len(lake.read(t)) for t in ("bronze_contracts", "bronze_projects", "bronze_docs")}
    r = bronze_ingest(lake, str(d))                               # the same files again (full rebuild, trigger firing twice)
    assert all(v == 0 for v in r["rows_added"].values()), r["rows_added"]
    assert {t: len(lake.read(t)) for t in n} == n
    assert str(d) not in new_batches(lake, str(d), include_root=True)     # recorded although nothing was added
    # a project's open-tender snapshot: one tender signed, the other still open -> the whole new snapshot is appended
    o = pd.DataFrame(dict(tender_id=["T1", "T2"], project_id=["P123456"] * 2, notice_date=["2026-01-01"] * 2))
    lake.write("bronze_open_tenders", o.assign(_ingest_ts=pd.Timestamp("2026-01-02"), _source="a"))
    from procintel.pipeline import _append_new, _open_snapshot
    assert _append_new(lake, "bronze_open_tenders", o.iloc[1:], "b", pd.Timestamp("2026-02-01"), snapshot_by="project_id") == 1
    assert _open_snapshot(lake).tender_id.tolist() == ["T2"]
    assert _append_new(lake, "bronze_open_tenders", o.iloc[1:], "c", pd.Timestamp("2026-03-01"), snapshot_by="project_id") == 0
