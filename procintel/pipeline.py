"""Backend-agnostic medallion pipeline (runs on LocalLake in tests and on DeltaLake in Databricks notebooks/jobs).

  bronze_ingest   raw feeds/docs -> bronze_* (append-only, _ingest_ts/_source; docs de-duplicated by content hash)
  silver_gold     bronze -> clean entities (ER, as-of governance join) -> incremental chunks -> gold feature/index/graph/conflict tables
  train           gold features -> RiskModels -> registry (champion/challenger aliases)
  alerts          early-warning table for active projects
  learning_cycle  new landing files -> pipeline -> drift check -> challenger -> promote, all in one idempotent job
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import asdict, fields
import pandas as pd
from .core import Platform, icr_ratings_from_docs
from .graph import KnowledgeGraph
from .guardrails import optimism_bias, resolve_conflict
from .ingest import CHUNKER_VERSION, Chunk, build_dataset, chunk_document, read_raw
from .lakehouse import Lake
from .learning import ContinuousLearner
from .mlops import register_champion
from .models import FEATS

MODEL_NAME = "procintel_risk"
FEATURE_COLS = ["contract_id", "project_id", "supplier_id", "notice_date", "award_date", "tender_days", "rebid", "amendments"] + FEATS


def _now(): return pd.Timestamp.now().floor("s")


VS_SECTIONS = {"lessons", "procurement", "ratings", "context"}


def vs_subset(chunks: pd.DataFrame) -> pd.Series:
    """Rows the Copilot actually retrieves through Vector Search: evaluation/appraisal lessons, procurement, ratings, context
    (ISR narrative excluded - its ratings are structured data already)."""
    return chunks.section_kind.isin(VS_SECTIONS) & (chunks.doc_type.isin(["ICR", "PPAR", "PAD"]) | (chunks.section_kind == "ratings"))


META = ["_ingest_ts", "_source", "_row_hash"]


def _row_hash(df: pd.DataFrame) -> pd.Series:
    """Fingerprint of a row's values (bronze metadata excluded), stored with the row so the next ingest can tell repeats."""
    vals = df.drop(columns=[c for c in META if c in df]).astype(str)
    return pd.util.hash_pandas_object(vals, index=False).astype(str)


def _append_new(lake: Lake, name: str, df: pd.DataFrame, raw_dir: str, ts, snapshot_by=None) -> int:
    """Append only what bronze does not hold yet, so re-running an ingest (a full rebuild, a trigger firing again) adds no
    copies. Row tables (silver keeps the latest row per key): rows already stored are skipped. Snapshot tables (silver keeps
    the latest pull per `snapshot_by`, e.g. a project's open tenders): a project's rows are skipped only when they equal its
    latest stored snapshot exactly, otherwise the whole new snapshot is appended (skipping part of it would lose rows)."""
    df = df.copy(); df["_row_hash"] = _row_hash(df)
    if lake.exists(name) and len(df):
        old = lake.read(name)
        h = old["_row_hash"] if "_row_hash" in old else pd.Series(None, index=old.index, dtype=object)
        old = old.assign(_row_hash=h.where(h.notna(), _row_hash(old)))   # rows stored before fingerprints existed
        if snapshot_by is None:
            df = df[~df._row_hash.isin(set(old._row_hash))]
        else:
            last = old[old._ingest_ts == old.groupby(snapshot_by)._ingest_ts.transform("max")]
            stored = last.groupby(snapshot_by)._row_hash.apply(frozenset).to_dict()
            new = df.groupby(snapshot_by)._row_hash.apply(frozenset).to_dict()
            df = df[[stored.get(g) != new[g] for g in df[snapshot_by]]]
    if len(df):
        df["_ingest_ts"] = ts; df["_source"] = raw_dir
        lake.write(name, df, "append")
    return len(df)


def bronze_ingest(lake: Lake, raw_dir: str) -> dict:
    raw, ts = read_raw(raw_dir), _now()
    added = {}
    for k in ("contracts", "projects", "governance", "disbursements", "isr"):
        added[k] = _append_new(lake, f"bronze_{k}", raw[k], raw_dir, ts)
    if len(raw.get("plan", [])):                                  # Procurement Plan activities (parsed public STEP exports)
        added["plan"] = _append_new(lake, "bronze_plan_activities", raw["plan"].astype(str), raw_dir, ts, snapshot_by="project_id")
    if len(raw.get("notices", [])):                               # procurement notices as fetched (STEP activity join)
        added["notices"] = _append_new(lake, "bronze_notices", raw["notices"].astype(str), raw_dir, ts)
    if len(raw.get("open_tenders", [])):                         # a snapshot: what was still open on the day of this pull
        added["open_tenders"] = _append_new(lake, "bronze_open_tenders", raw["open_tenders"], raw_dir, ts, snapshot_by="project_id")
    docs = pd.DataFrame([dict(doc_id=d["doc_id"], doc_type=d["doc_type"], project_id=d["project_id"], filename=d["filename"], text=d["text"],
                              meta_json=json.dumps(d.get("meta") or {}), content_hash=hashlib.md5(d["text"].encode()).hexdigest(),
                              _ingest_ts=ts, _source=raw_dir) for d in raw["docs"]])
    if lake.exists("bronze_docs") and len(docs):
        seen = set(zip(lake.read("bronze_docs").doc_id, lake.read("bronze_docs").content_hash))
        docs = docs[[(a, b) not in seen for a, b in zip(docs.doc_id, docs.content_hash)]]
    if len(docs):
        lake.write("bronze_docs", docs, "append")
    added["docs"] = len(docs)
    lake.write("bronze_ingest_log", pd.DataFrame([dict(_source=raw_dir, _ingest_ts=ts, rows_added=json.dumps(added))]), "append")
    return dict(raw_dir=raw_dir, docs_new=len(docs), contracts=len(raw["contracts"]), rows_added=added)


def _latest(lake, name, key):
    df = lake.read(name).sort_values("_ingest_ts").drop_duplicates(key, keep="last")
    return df.drop(columns=[c for c in META if c in df]).reset_index(drop=True)


def _docs(lake):
    d = lake.read("bronze_docs").sort_values("_ingest_ts").drop_duplicates("doc_id", keep="last")
    return [dict(doc_id=r.doc_id, doc_type=r.doc_type, project_id=r.project_id, filename=r.filename, text=r.text,
                 meta=json.loads(r.meta_json or "{}"), content_hash=r.content_hash) for r in d.itertuples()]


def _plan_latest(lake):
    """The newest plan per project (a later pull supersedes the earlier plan as a whole), typed back from bronze strings."""
    if not lake.exists("bronze_plan_activities"):
        return pd.DataFrame()
    b = lake.read("bronze_plan_activities")
    latest = b.groupby("project_id")._ingest_ts.transform("max")
    b = b[b._ingest_ts == latest].drop(columns=[c for c in META if c in b]).replace({"None": None, "nan": None, "NaN": None})
    for c in ("estimated_usd", "actual_usd", "n_dates", "page", "plan_versions"):
        if c in b:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.reset_index(drop=True)


def _open_snapshot(lake):
    """Per project, only the LATEST pull's open tenders: one open last month may have signed since (then it is a contract).
    Per project, not per pull: an incremental batch (one new country) covers only its own projects, and taking the latest
    pull overall dropped every other country's open tenders (seen on the workspace with a Rwanda batch)."""
    if not lake.exists("bronze_open_tenders"):
        return pd.DataFrame()
    b = lake.read("bronze_open_tenders")
    latest = b.groupby("project_id")._ingest_ts.transform("max")
    return b[b._ingest_ts == latest].drop(columns=[c for c in META if c in b]).reset_index(drop=True)


def silver_gold(lake: Lake) -> dict:
    raw = dict(contracts=_latest(lake, "bronze_contracts", "contract_id"), projects=_latest(lake, "bronze_projects", "project_id"),
               governance=_latest(lake, "bronze_governance", ["country_code", "year"]),
               disbursements=_latest(lake, "bronze_disbursements", ["project_id", "date"]),
               isr=_latest(lake, "bronze_isr", "isr_id"), docs=_docs(lake), open_tenders=_open_snapshot(lake),
               notices=_latest(lake, "bronze_notices", "id").replace({"None": None, "nan": None}) if lake.exists("bronze_notices") else pd.DataFrame(),
               plan=_plan_latest(lake))
    ds = build_dataset(raw)
    lake.write("silver_open_tenders", ds.open_tenders if len(ds.open_tenders) else pd.DataFrame(columns=["tender_id", "project_id", "notice_date", "censored_at"]))
    if len(ds.activities):                                       # one row per STEP procurement activity
        lake.write("silver_procurement_activity", ds.activities)
    if len(ds.plan):                                             # Procurement Plan activities with planned / actual key dates
        lake.write("silver_procurement_plan", ds.plan.drop(columns=["code"], errors="ignore").astype({"description": str}))
    # ---- silver: cleaned entity tables (full recompute keeps supplier ids stable) + incremental chunks
    lake.write("silver_contracts", ds.contracts); lake.write("silver_projects", ds.projects); lake.write("silver_governance", ds.governance)
    lake.write("silver_disbursements", ds.disbursements); lake.write("silver_isr", ds.isr)
    # a doc is (re)chunked when its text OR the chunker version changes; its old chunks are deleted first so no stale ids remain
    key = lambda d: f"{d['content_hash']}:v{CHUNKER_VERSION}"
    have = set(zip(lake.read("silver_chunks").doc_id, lake.read("silver_chunks").doc_hash)) if lake.exists("silver_chunks") else set()
    todo = [d for d in ds.docs if (d["doc_id"], key(d)) not in have]
    rows = [dict(asdict(c), doc_hash=key(d)) for d in todo for c in chunk_document(d)]
    stale = [d["doc_id"] for d in todo]
    if stale and lake.exists("silver_chunks"):
        lake.delete_in("silver_chunks", "doc_id", stale)
    if rows:
        lake.merge("silver_chunks", pd.DataFrame(rows), ["chunk_id"])
    # documents no longer in the dataset (e.g. a duplicate copy dropped by build_dataset) lose their chunks too
    if lake.exists("silver_chunks"):
        gone = sorted(set(lake.read("silver_chunks").doc_id) - {d["doc_id"] for d in ds.docs})
        if gone:
            lake.delete_in("silver_chunks", "doc_id", gone)
    chunks = lake.read("silver_chunks")
    # ---- gold: ML features, index source (CDF on => Vector Search delta-sync), KG edges, ISR/ICR conflicts
    lake.write("gold_contract_features", ds.contracts[[c for c in FEATURE_COLS if c in ds.contracts]])
    cols = [f.name for f in fields(Chunk)]
    # gold_chunks = full corpus (app / in-memory hybrid index); gold_chunks_vs = the decision-relevant subset that Vector Search
    # embeds (managed embedding runs ~70 chunks/min on Free Edition: 29k chunks ~7 h, the subset of ~3.3k ~45 min)
    for table, sub in (("gold_chunks", chunks), ("gold_chunks_vs", chunks[vs_subset(chunks)])):
        if lake.exists(table):
            # re-chunked docs, plus docs no longer in the dataset (dropped duplicate copies): CDF carries the deletes to Vector Search
            drop = sorted(set(stale) | (set(lake.read(table).doc_id) - {d["doc_id"] for d in ds.docs}))
            if drop:
                lake.delete_in(table, "doc_id", drop)
            new = sub[sub.doc_id.isin(stale)]                    # only touched docs: re-merging unchanged rows would re-embed them
        else:
            new = sub
        if len(new):
            lake.merge(table, new[cols], ["chunk_id"])
        elif not lake.exists(table):
            lake.write(table, sub[cols])
        lake.enable_cdf(table)
    chunk_objs = [Chunk(**r) for r in chunks[cols].to_dict("records")]
    lake.write("gold_kg_edges", KnowledgeGraph().build(ds, chunk_objs, None).edges_df())
    icr = icr_ratings_from_docs(ds.docs); bias = optimism_bias(ds.isr, icr)
    conf = [c for c in (resolve_conflict(p, ds.isr, r, bias=bias) for p, r in icr.items()) if c]
    cc = ["project_id", "isr_last", "icr", "gap_notches", "rule_applied", "adopted_source", "adopted_rating", "severity", "note",
          "confidence", "needs_human_review"]
    conf_df = pd.DataFrame([{**c, "isr_last": json.dumps(c["isr_last"]), "icr": json.dumps(c["icr"])} for c in conf])
    lake.write("gold_conflicts", conf_df if len(conf_df) else pd.DataFrame(columns=cc))    # Delta rejects a column-less table
    return dict(contracts=len(ds.contracts), chunks=len(chunks), docs_chunked=len(todo), conflicts=len(conf), quality=ds.quality)


def train(lake: Lake, store, min_gain=0.0):
    """Fit on the gold feature table and register; returns (version, became_champion)."""
    from .models import RiskModels
    open_t = lake.read("silver_open_tenders") if lake.exists("silver_open_tenders") else None
    m = RiskModels().fit(lake.read("gold_contract_features"), open_tenders=open_t)
    return register_champion(store, MODEL_NAME, m, dict(n_train=len(lake.read("gold_contract_features"))), min_gain)


def load_platform(lake: Lake, store=None, retriever_factory=None) -> Platform:
    return Platform.from_lake(lake, store, retriever_factory)


def alerts(lake: Lake, plat: Platform, as_of: str) -> pd.DataFrame:
    from .monitoring import project_warnings
    act = plat.ds.projects[plat.ds.projects.status == "Active"].project_id
    rows = [project_warnings(plat, p, as_of) for p in act]
    df = pd.DataFrame([{**r, "flags": json.dumps(r["flags"])} for r in rows])
    lake.write("gold_alerts", df); return df


def new_batches(lake: Lake, landing_dir: str, include_root=False) -> list[str]:
    """Folders not ingested yet (bronze_ingest_log records every folder ingested) that are complete (projects.json,
    written last, is there): the batch folders under landing_dir (one per scheduled pull: incoming/<date>/) and, with
    include_root, files dropped straight into landing_dir. A folder is ingested once: bronze stays free of repeated copies."""
    done = set()                     # every folder ever ingested: the ingest log (a repeat batch may add no rows at all) and,
    for t in ("bronze_ingest_log", "bronze_projects"):           # for ingests before the log existed, the projects' _source
        if lake.exists(t):
            done |= set(lake.read(t)._source)
    subs = sorted(os.path.join(landing_dir, d) for d in os.listdir(landing_dir)) if os.path.isdir(landing_dir) else []
    return [d for d in ([landing_dir] if include_root else []) + subs
            if os.path.isfile(os.path.join(d, "projects.json")) and d not in done]


def learning_cycle(lake: Lake, store, landing_dir: str, min_labelled=20) -> dict:
    before = set(lake.read("gold_contract_features").contract_id)
    dirs = new_batches(lake, landing_dir, include_root=True)       # each folder once (the trigger fires on every new file)
    if not dirs:
        return dict(status="nothing new to ingest")
    ing = [bronze_ingest(lake, d) for d in dirs]; sg = silver_gold(lake)
    plat = Platform.from_lake(lake, store)
    new_c = plat.ds.contracts[~plat.ds.contracts.contract_id.isin(before)]
    out = dict(ingest=ing, silver_gold={k: v for k, v in sg.items() if k != "quality"}, new_contracts=len(new_c))
    if new_c.tender_days.notna().sum() < min_labelled:
        return dict(out, status="too few labelled outcomes to evaluate drift")
    L = ContinuousLearner(plat)
    out["monitor"] = L.monitor(new_c)
    out["retrain"] = L.retrain(new_c, store=store, model_name=MODEL_NAME) if out["monitor"]["retrain"] else None
    return out
