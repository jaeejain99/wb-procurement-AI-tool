"""Offline tests of the live-data adapter using records shaped like the real API responses."""
import json, numpy as np, pandas as pd
from procintel import Platform
from procintel.live_wb import map_contract, map_project, map_wgi_rows, match_notices, parse_icr_rating, parse_isr_ratings
from procintel.ingest import chunk_document, parse_document

PROJ = {"id": "P508408", "project_name": "Bulsho", "countrycode": ["SO"], "sector1": {"Name": "", "Percent": 0},
        "mjtheme_namecode": [{"name": "Social inclusion", "code": "2"}], "totalcommamt": "200,000,000",
        "boardapprovaldate": "2024-12-20T00:00:00Z", "closingdate": "12/20/2029 12:00:00 AM", "status": "Active"}
CON = {"contr_id": "1898907", "contr_sgn_date": "05-Sep-2025", "projectid": "P508408", "contr_desc": "Financial Management Specialist - PCU",
       "total_contr_amnt": "70200.00", "contr_refnum": "SO-MOIFAR-498737-CS-INDV", "procurement_group_desc": "Consultant Services",
       "supp_name": ["INDIVIDUAL CONSULTANT"], "procu_meth_text": "Individual Consultant Selection", "rvw_type": "Prior"}
NOT = [{"id": "OP1", "notice_type": "Request for Expressions of Interest", "noticedate": "01-Jun-2025", "project_id": "P508408",
        "bid_reference_no": "SO-MOIFAR-498737-CS-INDV", "bid_description": "Financial Management Specialist", "notice_status": "Published"},
       {"id": "OP2", "notice_type": "Contract Award", "noticedate": "10-Sep-2025", "project_id": "P508408", "bid_reference_no": "x", "bid_description": "x"},
       {"id": "OP3", "notice_type": "Request for Bids", "noticedate": "20-Jun-2025", "project_id": "P508408", "bid_reference_no": "N.A.",
        "bid_description": "Financial Management Specialist - PCU", "notice_status": "Revised"}]


def test_project_and_contract_mapping():
    p = map_project(PROJ)
    assert p["country_code"] == "SO" and p["commitment_usd_m"] == 200 and p["sector"] == "Social inclusion" and p["closing_date"].year == 2029
    c = map_contract(CON, {"P508408": p})
    assert c["category"] == "Consultancies" and abs(c["value_usd_m"] - 0.0702) < 1e-9 and c["country_code"] == "SO"


def test_notice_matching_gives_duration_proxy_and_ignores_awards():
    c = map_contract(CON, {"P508408": map_project(PROJ)})
    m = match_notices(c, NOT)
    assert m["notice_date"] == pd.Timestamp("2025-06-01") and (c["sign_date"] - m["notice_date"]).days == 96
    # STEP activity match: only notices of the same activity count (OP3 has no reference, so it no longer adds an amendment)
    assert m["match_rule"] == "step_activity" and m["match_score"] == 1.0 and m["amendments"] == 0 and m["rebid"] == 0
    assert match_notices(c, [])["notice_date"] is pd.NaT


def test_step_reference_parsing_tolerates_formatting():
    from procintel.live_wb import parse_step_ref
    assert parse_step_ref("SO-MOIFAR-498737-CS-INDV")["activity"] == "SO-498737"
    assert parse_step_ref("ke - sdya - 533926 - cs - qcbs") == dict(activity="KE-533926", country="KE", number="533926", group="CS", method="QCBS")
    assert parse_step_ref("NP-DOR-123456-CW-RFB/LOT1")["activity"] == "NP-123456"
    assert parse_step_ref("IDA/73550") is None and parse_step_ref(None) is None and parse_step_ref("KE-X-533926-XX-CDS") is None


def test_activity_number_links_feeds_that_write_the_reference_differently():
    c = map_contract(dict(CON, contr_refnum="SO-MOIFAR-498737-CS-INDV"), {"P508408": map_project(PROJ)})
    notices = [dict(NOT[0], id="A", bid_reference_no="SO-MOF-498737-CS-IC", bid_description="something else entirely"),
               dict(NOT[0], id="B", noticedate="20-Jun-2025", bid_reference_no="SO-MOF-498737-CS-IC", notice_status="Revised"),
               dict(NOT[0], id="C", noticedate="01-Jul-2025", bid_reference_no="SO-MOF-498737-CS-IC")]   # advertised again
    m = match_notices(c, notices)
    assert m["match_rule"] == "step_activity" and m["notice_date"] == pd.Timestamp("2025-06-01")
    assert m["amendments"] == 1 and m["rebid"] == 1


def test_description_is_only_the_last_resort():
    c = map_contract(dict(CON, contr_refnum=""), {"P508408": map_project(PROJ)})
    m = match_notices(c, [dict(NOT[0], bid_reference_no="N.A.")])
    assert m["match_rule"] == "description" and 0.55 <= m["match_score"] < 1


def test_open_tenders_are_per_activity_and_closed_by_any_contract_or_award():
    from procintel.live_wb import open_tenders
    pull = pd.Timestamp("2026-11-01")
    ns = [dict(NOT[0], id="E1", bid_reference_no="SO-A-111111-CS-QCBS", noticedate="01-Jan-2026"),
          dict(NOT[0], id="E2", bid_reference_no="SO-B-111111-CS-QCBS", noticedate="01-Feb-2026"),   # same activity: one row
          dict(NOT[0], id="E3", bid_reference_no="SO-A-222222-GO-RFB", noticedate="01-Jan-2026"),    # signed (contract carries it)
          dict(NOT[0], id="E4", bid_reference_no="SO-A-333333-CW-RFB", noticedate="01-Jan-2026"),
          dict(NOT[1], id="E5", bid_reference_no="SO-X-333333-CW-RFB", noticedate="01-Mar-2026")]    # award notice for E4's activity
    out = open_tenders(ns, set(), pull, {"P508408": {"country_code": "SO", "sector": "Social"}}, contract_refs=["SO-Z-222222-GO-RFQ"])
    assert [o["step_activity"] for o in out] == ["SO-111111"]


def test_wgi_publication_lag_and_nan_cpia():
    rows = {"wgi_gov_effectiveness": [{"country": {"id": "KE", "value": "Kenya"}, "date": "2022", "value": -0.2}, {"country": {"id": "KE", "value": "Kenya"}, "date": "2023", "value": None}]}
    g = map_wgi_rows(rows)
    assert len(g) == 1 and g.available_from.iloc[0] == pd.Timestamp("2023-09-30") and np.isnan(g.cpia.iloc[0])


def test_rating_parsers_on_plain_text():
    assert parse_icr_rating("Overall Outcome:  Moderately Unsatisfactory\nRisk") == "Moderately Unsatisfactory"
    r = parse_isr_ratings("Progress towards achievement of PDO   Satisfactory\nOverall Implementation Progress (IP)  Moderately Satisfactory")
    assert r == {"dev_objective_rating": 5, "implementation_progress_rating": 4}


def test_rating_parsers_on_real_document_layout():
    # excerpts of P120370 ISR seq 16 and ICR (documents.worldbank.org txturl), fetched 2026-11-02
    isr = ("Progress towards achievement of PDO" + " " * 42 + "Moderately Unsatisfactory                  Moderately Satisfactory\r\n"
           " Overall Implementation Progress (IP)" + " " * 41 + "Satisfactory                  Moderately Satisfactory\r\n")
    assert parse_isr_ratings(isr) == {"dev_objective_rating": 4, "implementation_progress_rating": 4}   # current = last column
    icr = "Outcome                                 Bank Performance                          M&E Quality\r\nModerately Unsatisfactory   Moderately Satisfactory"
    assert parse_icr_rating(icr) == "Moderately Unsatisfactory"
    assert parse_icr_rating("Ratings were as follows: outcome was moderately unsatisfactory, Bank") == "Moderately Unsatisfactory"
    # DPO / newer ICR data sheets: other column headers sit between 'Outcome' and its value (31/31 real ICRs parse)
    assert parse_icr_rating("Outcome  Relevance of Prior Actions  Achievement of Objectives (Efficacy)\n\nSatisfactory  Moderately Satisfactory") == "Satisfactory"
    assert parse_icr_rating("OUTCOME RATING\n\n       Rating: Moderately Satisfactory\n") == "Moderately Satisfactory"
    assert parse_icr_rating("The overall Outcome rating for the project is Satisfactory based on") == "Satisfactory"


def test_real_project_fields_sector_and_es_risk():
    p = map_project(dict(PROJ, mjsector_namecode=[{"name": "Transportation", "code": "TX"}], envassesmentcategorycode="A"))
    assert p["sector"] == "Transportation" and p["es_risk"] == "High"
    assert map_project(dict(PROJ, esrc_ovrl_risk_rate="Substantial"))["es_risk"] == "Substantial"
    assert map_contract(dict(CON, procurement_group="NC"), {})["category"] == "Non-Consulting Services"


def test_plain_text_report_gets_headings_and_sidecar_metadata(tmp_path):
    f = tmp_path / "ICR_P1_1.txt"
    f.write_text("INTRODUCTION\n\nThe project did things.\n\nLESSONS LEARNED\n\nEstimates should be checked against market rates.\n")
    d = parse_document(str(f)); d["meta"] = {"Country": "Kenya", "Sector": "Transport", "Date": "2020-01-01"}
    ch = chunk_document(d)
    assert any(c.section_kind == "lessons" and c.country == "Kenya" for c in ch)


def test_activity_table_joins_both_feeds_on_the_step_number():
    from procintel.ingest import build_activities
    notices = pd.DataFrame([
        dict(id="1", project_id="P1", step_activity="KE-111", notice_type="Request for Bids", notice_status="Published", noticedate="01-Jan-2026",
             submission_deadline_date="2026-02-01T00:00:00Z"),
        dict(id="2", project_id="P1", step_activity="KE-111", notice_type="Request for Bids", notice_status="Revised", noticedate="10-Jan-2026",
             submission_deadline_date="2026-02-15T00:00:00Z"),
        dict(id="3", project_id="P1", step_activity="KE-222", notice_type="Request for Bids", notice_status="Published", noticedate="01-Mar-2026",
             submission_deadline_date=None),
        dict(id="4", project_id="P1", step_activity="KE-222", notice_type="Contract Award", notice_status="Published", noticedate="01-Jun-2026",
             submission_deadline_date=None)])
    contracts = pd.DataFrame([dict(contract_id="C1", project_id="P1", step_activity="KE-111", award_date=pd.Timestamp("2026-05-01"))])
    a = build_activities(contracts, notices).set_index("step_activity")
    assert a.loc["KE-111", "status"] == "signed" and a.loc["KE-111", "n_revised"] == 1 and a.loc["KE-111", "n_tender_notices"] == 1
    assert a.loc["KE-111", "submission_deadline"] == pd.Timestamp("2026-02-15") and a.loc["KE-111", "signed_date"] == pd.Timestamp("2026-05-01")
    assert a.loc["KE-222", "status"] == "awarded" and a.loc["KE-222", "award_notice_date"] == pd.Timestamp("2026-06-01")


def test_a_tender_still_being_advertised_is_never_running_late(tmp_path):
    from procintel import Platform
    from procintel.monitoring import overdue_tenders
    from procintel.synth import generate, write_dataset
    write_dataset(generate(seed=3, n_projects=40), str(tmp_path)); plat = Platform.build(str(tmp_path))
    pid = plat.ds.projects.project_id.iloc[0]
    row = dict(tender_id="T1", project_id=pid, notice_date=pd.Timestamp("2020-01-01"), censored_at=pd.Timestamp("2026-11-01"), category="Goods",
               procurement_method="Request for Bids", description="Long advertised goods", country_code=plat.ds.projects.country_code.iloc[0],
               sector=plat.ds.projects.sector.iloc[0], log_value=np.nan, notice_month=1, review_type=np.nan, step_ref="KE-X-111-GO-RFB")
    for deadline, late in ((pd.Timestamp("2027-01-01"), False), (pd.NaT, True)):
        plat.ds.open_tenders = pd.DataFrame([dict(row, submission_deadline=deadline)])
        t = next(x for x in overdue_tenders(plat, pid, pd.Timestamp("2026-11-01")) if x["tender_id"] == "T1")
        assert t["overdue"] is late and t["advertising"] is (not late) and t["step_ref"] == "KE-X-111-GO-RFB"


def _plan(**kw):
    base = dict(project_id="P123456", step_ref="KE-A-111111-CS-QCBS", step_activity="KE-111111", description="Survey", status="Under Implementation",
                actual_usd=None, text_check=True, plan_date="2026-10-01", notice_planned=None, notice_actual=None, signed_planned=None,
                signed_actual=None, first_actual=None)
    return dict(base, **kw)


def test_contracts_link_to_their_plan_activity_and_take_its_notice_date():
    from procintel.ingest import _plan_table, link_plan
    plan = _plan_table(pd.DataFrame([
        _plan(),                                                                                     # not linked: no contract
        _plan(step_ref="CWSIP/W11", step_activity=None, notice_actual="2025-01-10"),                 # agency code
        _plan(step_ref="KE-A-222222-GO-RFB", step_activity="KE-222222", actual_usd=1_000_000, signed_actual="2025-06-01",
              notice_actual="2025-02-01"),                                                          # amount + date
        _plan(step_ref="KE-A-333333-CW-RFB", step_activity="KE-333333", notice_actual="2025-03-01", text_check=False)]))  # contradicted
    contracts = pd.DataFrame([
        dict(contract_id="C1", project_id="P123456", ref="CWSIP/W11", step_activity=None, value_usd_m=2.0, award_date="2025-05-01",
             notice_date="2025-05-01", match_rule="none", match_score=0.0, description="works"),
        dict(contract_id="C2", project_id="P123456", ref="BD/LGED/77", step_activity=None, value_usd_m=1.005, award_date="2025-06-08",
             notice_date="2025-06-08", match_rule="description", match_score=0.6, description="goods"),
        dict(contract_id="C3", project_id="P123456", ref="KE-X-333333-CW-RFB", step_activity="KE-333333", value_usd_m=3.0, award_date="2025-07-01",
             notice_date="2025-07-01", match_rule="none", match_score=0.0, description="works")])
    c, st = link_plan(contracts, plan)
    c = c.set_index("contract_id")
    assert c.loc["C1", "plan_link"] == "plan_code" and str(c.loc["C1", "notice_date"])[:10] == "2025-01-10" and c.loc["C1", "match_score"] == 1.0
    assert c.loc["C2", "plan_link"] == "plan_amount_date" and c.loc["C2", "match_rule"] == "plan_amount_date" and c.loc["C2", "match_score"] == 0.9
    assert pd.isna(c.loc["C3", "plan_link"])                                   # its plan row was dropped: dates contradicted by the text
    assert st["plan_start_dates_added"] == 2


def test_behind_plan_uses_recent_plans_only_and_skips_signed_or_flagged_packages():
    from procintel.ingest import _plan_table
    from procintel.monitoring import behind_plan
    plan = _plan_table(pd.DataFrame([
        _plan(step_ref="KE-A-1-CS-QCBS", step_activity="KE-1", signed_planned="2026-03-01"),                       # 8 months late
        _plan(step_ref="KE-A-2-CS-QCBS", step_activity="KE-2", signed_planned="2026-10-20"),                       # inside the grace period
        _plan(step_ref="KE-A-3-CS-QCBS", step_activity="KE-3", signed_planned="2026-01-01", status="Signed"),      # done
        _plan(step_ref="KE-A-4-CS-QCBS", step_activity="KE-4", signed_planned="2026-01-01", signed_actual="2026-02-01"),
        _plan(step_ref="KE-A-5-CS-QCBS", step_activity="KE-5", signed_planned="2026-01-01"),                       # already a late tender
        _plan(step_ref="KE-A-6-CS-QCBS", step_activity="KE-6", signed_planned="2024-01-01", plan_date="2024-06-01")]))  # stale plan
    late = behind_plan(plan, "2026-11-01", skip_activities={"KE-5"})
    assert [x["step_ref"] for x in late] == ["KE-A-1-CS-QCBS"] and late[0]["days_late"] == 245



def test_behind_plan_drops_reference_wrapped_across_lines():
    """The plan PDF wraps '.../082/83' over two lines; the parser yields '.../082' + '83 / desc' as well: count it once."""
    from procintel.ingest import _plan_table
    from procintel.monitoring import behind_plan
    plan = _plan_table(pd.DataFrame([
        _plan(step_ref="NEHEP/UGC/CS/CQS-01/082", description="83 / Procurement of Survey", signed_planned="2026-03-17"),
        _plan(step_ref="NEHEP/UGC/CS/CQS-01/082/83", description="Procurement of Survey", signed_planned="2026-03-17"),
        _plan(step_ref="NEHEP/UGC/CS/CQS-02/082", description="83 / Procurement of Review", signed_planned="2026-03-18")]))
    late = behind_plan(plan, "2026-11-01")
    assert sorted(x["step_ref"] for x in late) == ["NEHEP/UGC/CS/CQS-01/082/83", "NEHEP/UGC/CS/CQS-02/082"]
    assert all(not x["description"].startswith("83") for x in late)

def test_stage_names_map_to_canonical_keys():
    from procintel.procplan import compact, stage_key
    assert stage_key("Expression of Interest Notice") == "eoi_notice" and stage_key("Specific Procurement Notice / Invitation") == "notice"
    assert stage_key("Combined Evaluation Report and Draft Negotiated Contract") == "combined_eval" and stage_key("Signed Contract") == "signed"
    assert compact("C onsultancy Provi sion") == compact("Consultancy Provision")
