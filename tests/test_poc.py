import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pytest
from procintel import Copilot, PackageSpec, Platform
from procintel.guardrails import resolve_conflict, verify_findings
from procintel.ingest import chunk_document, norm_project_id, resolve_suppliers
from procintel.learning import ContinuousLearner, eval_retrieval, psi
from procintel.synth import generate, write_dataset
from procintel.themes import THEMES


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("d"); base = generate(seed=3, n_projects=60)
    write_dataset(base, str(d / "base"))
    return Platform.build(str(d / "base")), base, d


def test_supplier_entity_resolution():
    m = resolve_suppliers(["Acme Construction Ltd", "ACME CONSTRUCTION LIMITED", "Acme Constr. Ltd.", "Baraka Engineering Inc"])
    assert len({m["Acme Construction Ltd"], m["ACME CONSTRUCTION LIMITED"], m["Acme Constr. Ltd."]}) == 1
    assert m["Baraka Engineering Inc"] != m["Acme Construction Ltd"]


def test_project_id_normalisation():
    assert norm_project_id(" p150001 ") == "P150001"


def test_chunking_keeps_hierarchy_tables_and_quotability(world):
    plat, *_ = world
    doc = next(d for d in plat.ds.docs if d["doc_type"] == "ICR")
    ch = chunk_document(doc)
    assert any(c.chunk_type == "table" for c in ch) and any(c.section_kind == "lessons" for c in ch)
    assert all(c.text in doc["text"].replace("\n", " ") or c.text.split("\n")[0] in doc["text"] for c in ch)


def test_no_governance_lookahead(world):
    plat, *_ = world
    c = plat.ds.contracts.iloc[0]
    g = plat.gov_asof(c.country_code, c.notice_date)
    assert g.available_from <= c.notice_date and g.wgi_gov_effectiveness == c.wgi_gov_effectiveness


def test_graph_links_contract_to_chunk_with_confidence(world):
    plat, *_ = world
    cid = plat.ds.contracts[plat.ds.contracts.project_id.isin(plat.icr)].contract_id.iloc[0]
    ev = plat.graph.evidence_for_contract(cid)
    assert ev and ev[0][2] == 1.0 or all(e[1] == "same_project_section" for e in ev)


def test_method_baseline_and_c_index():
    from procintel.models import c_index, method_baseline
    tr = pd.DataFrame(dict(category=["Goods"] * 6 + ["Works"] * 2, procurement_method=["RFB"] * 5 + ["RFQ"] + ["RFB"] * 2,
                           tender_days=[100, 110, 120, 130, 140, 30, 300, 400]))
    te = pd.DataFrame(dict(category=["Goods", "Goods", "Works"], procurement_method=["RFB", "RFQ", "RFB"]))
    b = method_baseline(te, tr, pd.Series([1.0, 2.0, 3.0], index=te.index))
    assert list(b) == [120.0, 2.0, 3.0]                  # Goods+RFB has 5 tenders; the rare pairs fall back
    assert c_index([10, 20, 30], [1, 2, 3]) == 1.0 and c_index([10, 20, 30], [3, 2, 1]) == 0.0
    assert c_index([10, 20], [5, 5]) == 0.5


def test_model_beats_baseline_and_handles_missing(world):
    plat, *_ = world
    m = plat.models.metrics
    assert m["mae_days"] < m["mae_baseline_category_median"]
    row = plat.ds.contracts.head(3).copy(); row["log_value"] = np.nan
    p = plat.models.predict_df(row)
    assert p.notna().all().all() and (p.p10_days <= p.p50_days).all() and (p.p50_days <= p.p90_days).all()


def test_copilot_every_finding_is_grounded(world):
    plat, *_ = world
    rep = Copilot(plat).analyze(PackageSpec("Rehabilitation of 50 km rural roads", "Civil Works", "Transport", "KEN", 12.0, "ICB"))
    cb = plat.chunks_by_id
    risks = [f for f in rep["findings"] if f["kind"] == "risk"]
    assert risks
    for f in risks:
        for e in f["evidence"]:
            assert " ".join(e["quote"].split()).lower() in " ".join(cb[e["chunk_id"]].text.split()).lower()


def test_guardrails_reject_fabrication(world):
    plat, *_ = world
    cid = plat.index.chunks[0].chunk_id
    bad = [dict(kind="risk", claim="x", facts=[], evidence=[]),
           dict(kind="risk", claim="Delays of 77 months occurred.", facts=[], evidence=[dict(chunk_id=cid, quote=plat.index.chunks[0].text[:30])]),
           dict(kind="risk", claim="Totally invented quote.", facts=[], evidence=[dict(chunk_id=cid, quote="this text does not exist")])]
    ok, rej = verify_findings(bad, plat.chunks_by_id, {}, {})
    assert not ok and len(rej) == 3


def test_conflict_resolution_prefers_icr():
    isr = pd.DataFrame(dict(project_id=["P1"] * 4, date=pd.date_range("2020-01-01", periods=4, freq="6MS"), dev_objective_rating=[5, 5, 5, 5]))
    r = resolve_conflict("P1", isr, 2)
    assert r["adopted_source"] == "ICR" and r["gap_notches"] == 3
    minor = resolve_conflict("P1", isr, 4)                  # 1 notch: R4 - reported with both sources, flagged minor
    assert minor["rule_applied"] == "R4" and minor["severity"] == "minor" and not minor["needs_human_review"]
    assert resolve_conflict("P1", isr, 5) is None          # agreement


def test_retrieval_quality_and_incremental_update(world):
    plat, base, d = world
    k = eval_retrieval(plat, base.eval_labels, THEMES)
    assert k["mrr"] > 0.5
    new = generate(seed=5, n_projects=15, id_start=800000, approval_years=(2016, 2019), cut="2026-06-30", gov=base.governance)
    write_dataset(new, str(d / "b1"))
    L = ContinuousLearner(plat); n_before = len(plat.index.chunks)
    r = L.ingest_batch(str(d / "b1")); assert len(plat.index.chunks) == n_before + r["new_chunks"]
    assert L.ingest_batch(str(d / "b1"))["index"]["added"] == 0      # idempotent
    L.feedback(["x"], True); assert plat.index.boost["x"] > 0


def test_psi_detects_shift():
    a = np.random.default_rng(0).normal(0, 1, 500)
    assert psi(a, a) < .05 < .25 < psi(a, a + 2)


def test_survival_targets_censor_as_of_without_look_ahead():
    from procintel.models import survival_targets
    c = pd.DataFrame(dict(notice_date=pd.to_datetime(["2020-01-01", "2020-01-01"]), award_date=pd.to_datetime(["2020-03-01", "2021-01-01"]),
                          tender_days=[60.0, 366.0], category="Goods"))
    o = pd.DataFrame(dict(notice_date=pd.to_datetime(["2020-02-01"]), censored_at=pd.to_datetime(["2022-01-01"]), category="Goods"))
    t = survival_targets(c, o, "2020-07-01").set_index("lower")
    assert t.loc[60.0, "upper"] == 60.0                                   # signed before the cut: exact
    assert np.isinf(t.loc[182.0, "upper"])                                # signed AFTER the cut: censored at the cut, not 366
    assert np.isinf(t.loc[151.0, "upper"])                                # open tender: censored at the cut, not at 2022


def test_open_tenders_lengthen_the_forecast(tmp_path):
    """Dropping slow, still-open tenders biases durations short; keeping them as censored rows must raise the median."""
    from procintel.ingest import load_dataset
    from procintel.models import RiskModels
    write_dataset(generate(seed=3, n_projects=60), str(tmp_path)); ds = load_dataset(str(tmp_path))
    c = ds.contracts
    late = c.notice_date.max()
    o = c.sample(80, random_state=0).assign(notice_date=late - pd.Timedelta(days=700), censored_at=late)   # open ~700 days
    a = RiskModels().fit(c, validate=False).predict_df(c.head(200)).p50_days.median()
    b = RiskModels().fit(c, validate=False, open_tenders=o).predict_df(c.head(200)).p50_days.median()
    assert b > a


def test_missing_value_does_not_mean_still_open(tmp_path):
    """Open tenders have no contract value yet. If the model could see 'value missing', it would learn 'missing => still open
    => very long' and forecast decades for any package entered without a value (seen on real data: P50 of 20,387 days)."""
    from procintel.ingest import load_dataset
    from procintel.models import RiskModels
    write_dataset(generate(seed=3, n_projects=60), str(tmp_path)); ds = load_dataset(str(tmp_path))
    c = ds.contracts; late = c.notice_date.max()
    o = c.sample(120, random_state=0).assign(notice_date=late - pd.Timedelta(days=900), censored_at=late, log_value=np.nan)
    m = RiskModels().fit(c, validate=False, open_tenders=o)
    row = c.head(50).copy()
    with_value = m.predict_df(row).p50_days.median()
    without = m.predict_df(row.assign(log_value=np.nan)).p50_days.median()
    assert without < 2 * with_value                       # missing value -> typical value for the category, not "open"


def test_backtest_is_stable_across_seeds(tmp_path):
    """A model whose backtest swings with the random seed cannot be trusted (seen on real data: 116 to 1,038 days MAE)."""
    from procintel.ingest import load_dataset
    from procintel.models import RiskModels
    write_dataset(generate(seed=5, n_projects=80), str(tmp_path)); ds = load_dataset(str(tmp_path))
    maes = [RiskModels(seed=s).fit(ds.contracts).metrics["mae_days"] for s in range(3)]
    assert max(maes) - min(maes) < 0.25 * np.mean(maes), maes


def test_page_numbers_are_pdf_pages_even_when_a_heading_starts_the_page(tmp_path):
    """WB text renditions keep one form feed per PDF page; heading promotion used to drop the form feed on heading lines
    (and splitlines() dropped all of them), so pages were guessed at ~3000 chars/page: page 31 for a passage on PDF page 28."""
    from procintel.ingest import chunk_document, parse_document
    body = "Intro paragraph on the first page.\n\n\fSECOND PAGE HEADING\n\nText on page two.\n\n\f" + "Text on page three.\n\n" * 3
    p = tmp_path / "ICR_P000001_123456.txt"; p.write_text(body, encoding="utf-8")
    d = parse_document(str(p))
    assert d["text"].count("\f") == 2
    pages = {c.text.split()[-1].strip("."): c.page for c in chunk_document(d)}
    assert pages["page"] == 1 and pages["two"] == 2 and pages["three"] == 3


def test_a_document_without_metadata_gets_country_and_sector_from_its_project():
    import pandas as pd
    from procintel.ingest import _fill_doc_meta
    docs = [dict(doc_id="ISR_P178487_1", project_id="P178487", text="ISR\nSeq No: 6 | Archived on 29-Jun-2026 | ISR08449\n", meta=None)]
    projects = pd.DataFrame([dict(project_id="P178487", country_code="BD", sector="Education")])
    gov = pd.DataFrame([dict(country_code="BD", country="Bangladesh")])
    assert _fill_doc_meta(docs, projects, gov)[0]["meta"] == {"Country": "Bangladesh", "Sector": "Education", "Date": "2026-06-29"}


def test_plain_language_durations_and_grades():
    from procintel.monitoring import grade_gap, plain_duration
    assert [plain_duration(d) for d in (1, 43, 59, 60, 189, 359, 547, 730, 900)] == \
        ["1 day", "43 days", "59 days", "2 months", "6 months", "12 months", "18 months", "2 years", "2.5 years"]
    assert grade_gap(0.62) == "about half a grade higher" and grade_gap(-1.1) == "about one grade lower"


def test_duplicate_copies_of_a_report_are_kept_once_per_project():
    import pandas as pd
    from procintel.ingest import _dedupe_docs, _dedupe_isr
    docs = [dict(doc_id=f"ISR_P1_{i}", project_id="P1", text="Same  report\ntext.") for i in (3, 1, 2)] + \
           [dict(doc_id="ICR_P2_9", project_id="P2", text="Shared ICR."), dict(doc_id="ICR_P3_9", project_id="P3", text="Shared ICR.")]
    assert [d["doc_id"] for d in _dedupe_docs(docs)] == ["ICR_P2_9", "ICR_P3_9", "ISR_P1_1"]       # one ISR copy; the ICR stays with both projects
    isr = pd.DataFrame(dict(isr_id=["P1-3", "P1-1", "P1-2"], project_id="P1", dev_objective_rating=5,
                            date=pd.to_datetime(["2025-05-14", "2025-05-14", "2025-11-25"])))
    assert list(_dedupe_isr(isr).isr_id) == ["P1-1", "P1-2"]
