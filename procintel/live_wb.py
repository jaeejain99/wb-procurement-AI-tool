"""Adapter for the REAL World Bank sources -> the same folder layout the PoC already ingests.

    python -m procintel.live_wb --out data/live --countries KE,GH,NP,BD,VN,PE,MZ,JO --max-projects 150 --max-docs 200
    python run_live.py data/live

STATUS: run against the live services on 2026-11-02. The query parameters below were confirmed to filter server-side, and the
rating parsers are tested on excerpts of real ISR / ICR text (tests/test_live_mapping.py). Fixes found on that run: CPIA keys
countries by ISO3 (WGI by ISO2), WDS returns a 'facets' entry beside documents, sector1 is often blank, and Python's default CA
bundle cannot verify the WB certificates on some Windows hosts (certifi is used when installed).

Honest data caveats this adapter makes explicit instead of hiding:
  * contractdata has only a SIGNING date. Tender duration = signing date - date of the matched tender notice (procnotices),
    so it is a notice->signature proxy and only exists where a notice can be matched (coverage is reported).
  * re-bid / amendment labels are WEAK proxies (>=2 matching tender notices; notices flagged 'Revised').
  * E&S risk: esrc_ovrl_risk_rate (ESF projects), else legacy env category A/B/C -> High/Moderate/Low, else 'Unknown'.
    Sector: sector1, else first major sector, else first major theme.
  * ISR ratings are not a feed; they are parsed from ISR text (regex, best effort). Disbursements: no URL given -> empty.
"""
from __future__ import annotations
import argparse, json, os, re, ssl, time, urllib.parse, urllib.request
from difflib import SequenceMatcher
import numpy as np
import pandas as pd

SEARCH = "https://search.worldbank.org/api"
WBAPI = "https://api.worldbank.org/v2"
PROJECTS_COUNTRY_PARAM = "countrycode_exact"   # all four verified live 2026-11-02
CONTRACT_PROJECT_PARAM = "projectid"
NOTICE_PROJECT_PARAM = "project_id"
WDS_PROJECT_PARAM = "projectid"
PROJECT_FIELDS = ("id,project_name,countrycode,sector1,mjsector_namecode,mjtheme_namecode,totalcommamt,boardapprovaldate,"
                  "closingdate,status,impagency,envassesmentcategorycode,esrc_ovrl_risk_rate")
ENV_CAT = {"A": "High", "B": "Moderate", "C": "Low"}
GROUP = {"CW": "Civil Works", "GO": "Goods", "CS": "Consultancies", "NC": "Non-Consulting Services"}
WGI = {"wgi_rule_of_law": "GOV_WGI_RL.EST", "wgi_gov_effectiveness": "GOV_WGI_GE.EST", "wgi_reg_quality": "GOV_WGI_RQ.EST"}
CPIA = "IQ.CPA.IRAI.XQ"
RATINGS = ["Highly Unsatisfactory", "Moderately Unsatisfactory", "Moderately Satisfactory", "Highly Satisfactory", "Unsatisfactory", "Satisfactory"]
RATING_NUM = {"Highly Satisfactory": 6, "Satisfactory": 5, "Moderately Satisfactory": 4, "Moderately Unsatisfactory": 3, "Unsatisfactory": 2, "Highly Unsatisfactory": 1}
_RAT = "|".join(RATINGS)


# ------------------------------------------------------------------ HTTP
def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_ctx()


def get_json(url, params=None, retries=5, pause=0.3):
    q = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers={"User-Agent": "procintel-poc"}), timeout=60, context=SSL_CTX) as r:
                time.sleep(pause)
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def get_text(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "procintel-poc"}), timeout=120, context=SSL_CTX) as r:
        return r.read().decode("utf-8", errors="ignore")


def paged(endpoint, key, params, rows=100, limit=1000):
    out, os_ = [], 0
    while len(out) < limit:
        d = get_json(f"{SEARCH}/{endpoint}", dict(params, format="json", rows=rows, os=os_))
        batch = d.get(key) or []
        batch = list(batch.values()) if isinstance(batch, dict) else batch
        batch = [b for b in batch if b]                                   # WDS adds an empty 'facets' entry
        if not batch:
            break
        out += batch; os_ += rows
        if os_ >= int(d.get("total", 0) or 0):
            break
    return out[:limit]


# ------------------------------------------------------------------ pure mapping functions (unit-tested)
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def _dt(x, fmts=("%d-%b-%Y", "%Y-%m-%dT%H:%M:%SZ", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d")):
    for f in fmts:
        try:
            return pd.to_datetime(x, format=f)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def map_project(p: dict) -> dict:
    sec = (next((t.get("name") for t in p.get("mjsector_namecode") or [] if t.get("name")), None) or (p.get("sector1") or {}).get("Name")
           or next((t.get("name") for t in p.get("mjtheme_namecode") or [] if t.get("name")), None) or "Unknown")
    es = p.get("esrc_ovrl_risk_rate") or ENV_CAT.get(p.get("envassesmentcategorycode") or "", "Unknown")
    return dict(project_id=p["id"], project_name=p.get("project_name", ""), country_code=(p.get("countrycode") or [""])[0],
                sector=sec, commitment_usd_m=_num(p.get("totalcommamt")) / 1e6, approval_date=_dt(p.get("boardapprovaldate")),
                closing_date=_dt(p.get("closingdate")), status=p.get("status", ""), es_risk=es,
                implementing_agency=p.get("impagency", ""))


def _category(desc: str, code: str = "") -> str:
    if code in GROUP:
        return GROUP[code]
    d = (desc or "").lower()
    return ("Non-Consulting Services" if "non" in d and "consult" in d else "Consultancies" if "consult" in d else
            "Civil Works" if "work" in d else "Goods" if "good" in d else "Non-Consulting Services")


# STEP procurement reference: <country>-<agency>-<activity no.>-<CW|GO|CS|NC>-<method>, e.g. KE-SDYA-533926-CS-CDS. The two public
# feeds often write the agency or method part differently for the same package, but the activity number is the same (sample of
# 6 projects, 2026-11-09: exact full reference links 28% of contracts, the activity number 69%).
_STEP = re.compile(r"(?<![A-Z0-9])([A-Z]{2})-(?:[A-Z0-9&.()]+-)*?(\d{4,8})-(CW|GO|CS|NC)(?:-([A-Z0-9]+))?(?![0-9])")


def parse_step_ref(ref) -> dict | None:
    """'KE-SDYA-533926-CS-CDS' -> {'activity': 'KE-533926', 'country': 'KE', 'number': '533926', 'group': 'CS', 'method': 'CDS'}.
    Tolerates case, spaces and stray characters around hyphens; None if there is no STEP-style reference."""
    t = re.sub(r"\s*-\s*", "-", str(ref or "").upper()).replace("_", "-")
    m = _STEP.search(t)
    if not m:
        return None
    return dict(activity=f"{m.group(1)}-{m.group(2)}", country=m.group(1), number=m.group(2), group=m.group(3), method=m.group(4) or "")


def _activity(ref) -> str | None:
    p = parse_step_ref(ref)
    return p["activity"] if p else None


def map_contract(c: dict, project_lookup: dict) -> dict:
    pid = c.get("projectid")
    pr = project_lookup.get(pid, {})
    ref = c.get("contr_refnum", "") or ""
    return dict(contract_id=f"{pid}-{c.get('contr_id')}", project_id=pid, description=c.get("contr_desc", ""), ref=ref,
                step_activity=_activity(ref),
                category=_category(c.get("procurement_group_desc"), c.get("procurement_group", "")), procurement_method=c.get("procu_meth_text", ""),
                review_type=c.get("rvw_type", ""), value_usd_m=_num(c.get("total_contr_amnt")) / 1e6,
                supplier_raw=(c.get("supp_name") or ["UNKNOWN"])[0], country_code=pr.get("country_code", ""),
                sector=pr.get("sector", "Unknown"), sign_date=_dt(c.get("contr_sgn_date")), no_objection_date=_dt(c.get("contr_no_obj_dat")))


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _tender_type(n) -> bool:
    t = str(n.get("notice_type", "")).lower()
    return not ("award" in t or "general" in t)


def _cancelled(n) -> bool:
    return "cancel" in str(n.get("notice_status", "")).lower() or "cancel" in str(n.get("notice_type", "")).lower()


def match_notices(contract: dict, notices: list[dict], min_score=0.55, max_days=1500) -> dict:
    """The tender notices of a contract, in priority order:
      1 step_activity - same project and same STEP activity number (the shared key of both feeds)
      2 exact_ref     - same full reference string
      3 description   - description similarity >= min_score (weak: durations from matches < 0.8 are not used for training)
    The earliest matched notice is the tender start. Labels: re-tender = the package was advertised again (the same notice type
    published twice, not counting revisions) or a notice was cancelled; amendments = 'Revised' notices."""
    sign = contract["sign_date"]
    pool = []
    for n in notices:
        if n.get("project_id") != contract["project_id"] or not _tender_type(n):
            continue
        nd = _dt(n.get("noticedate"))
        if pd.isna(nd) or pd.isna(sign) or not (7 <= (sign - nd).days <= max_days):
            continue
        pool.append((nd, n))
    act = contract.get("step_activity")
    cands, rule = ([(nd, 1.0, n) for nd, n in pool if act and _activity(n.get("bid_reference_no")) == act], "step_activity")
    if not cands:
        ref = _norm(contract.get("ref", ""))
        cands, rule = ([(nd, 1.0, n) for nd, n in pool if ref not in ("", "n a", "na") and ref == _norm(n.get("bid_reference_no", ""))],
                       "exact_ref")
    if not cands:
        rule, cands = "description", []
        for nd, n in pool:
            s = SequenceMatcher(None, _norm(contract["description"])[:120], _norm(n.get("bid_description", ""))[:120]).ratio()
            if s >= min_score:
                cands.append((nd, s, n))
    if not cands:
        return dict(notice_date=pd.NaT, rebid=0, amendments=0, match_score=0.0, match_rule="none", notice_ids=[])
    cands.sort(key=lambda t: t[0])
    first = [n for *_, n in cands if str(n.get("notice_status")) != "Revised"]
    types = [str(n.get("notice_type", "")).lower() for n in first]
    if rule == "description":                          # unrelated packages can look alike: keep the old, weaker proxy
        rebid = int(len({n.get("id") for n in first}) >= 2)
    else:
        rebid = int(len(types) != len(set(types)) or any(_cancelled(n) for *_, n in cands))
    return dict(notice_date=cands[0][0], match_score=round(max(s for _, s, _ in cands), 2), match_rule=rule,
                notice_ids=[n.get("id") for *_, n in cands], rebid=rebid,
                amendments=int(sum(str(n.get("notice_status")) == "Revised" for *_, n in cands)))


def open_tenders(notices: list[dict], matched_ids: set, pull_date, lookup: dict, max_age_days=730, contract_refs=()) -> list[dict]:
    """Tenders that, as far as the public record shows, have NOT been signed yet -> right-censored durations. One row per STEP
    activity (or per reference / description when a notice has no STEP reference). A package is NOT open if any contract of the
    project carries its activity number or reference, a Contract Award notice does, it was cancelled, or a matched contract
    used the notice. Only 7..max_age_days old."""
    closed_acts = {_activity(r) for r in contract_refs} | {_activity(n.get("bid_reference_no")) for n in notices if not _tender_type(n)}
    closed_acts.discard(None)
    closed_refs = {_norm(r) for r in contract_refs} | {_norm(n.get("bid_reference_no", "")) for n in notices if not _tender_type(n)}
    closed_refs -= {"", "n a", "na"}
    cancelled = {(_activity(n.get("bid_reference_no")) or _norm(n.get("bid_reference_no", ""))) for n in notices if _cancelled(n)}
    out, seen = [], set()
    for n in sorted(notices, key=lambda n: str(_dt(n.get("noticedate")))):
        if not _tender_type(n) or n.get("id") in matched_ids or _cancelled(n):
            continue
        nd, ref, act = _dt(n.get("noticedate")), _norm(n.get("bid_reference_no", "")), _activity(n.get("bid_reference_no"))
        if pd.isna(nd) or not (7 <= (pull_date - nd).days <= max_age_days):
            continue
        if (act and act in closed_acts) or (ref and ref in closed_refs) or (act or ref) in cancelled:
            continue
        key = act or (ref if ref not in ("", "n a", "na") else _norm(n.get("bid_description", ""))[:80])
        if key in seen:                                              # REOI -> RFP -> revision of one package: keep the first
            continue
        seen.add(key)
        pr = lookup.get(n.get("project_id"), {})
        out.append(dict(tender_id=n.get("id"), project_id=n.get("project_id"), notice_date=nd, censored_at=pull_date,
                        category=_category(n.get("procurement_group_desc") or n.get("bid_description"), n.get("procurement_group", "")),
                        procurement_method=n.get("procurement_method_name", ""), description=str(n.get("bid_description", ""))[:200],
                        country_code=pr.get("country_code", ""), sector=pr.get("sector", "Unknown"),
                        step_ref=str(n.get("bid_reference_no") or ""), step_activity=act,
                        submission_deadline=_dt(n.get("submission_deadline_date"))))
    return out


NOTICE_FIELDS = ("id", "project_id", "notice_type", "notice_status", "noticedate", "submission_deadline_date", "bid_reference_no",
                 "bid_description", "procurement_method_name", "procurement_group", "notice_lang_name")


def fetch_procurement(pids, lookup, pull_date):
    """Contract awards + procurement notices for the given projects -> (contracts rows, open tenders rows, notices rows)."""
    rows, opened, saved = [], [], []
    for pid in pids:
        raw = paged("contractdata", "contract", {CONTRACT_PROJECT_PARAM: pid}, limit=300)
        notices = paged("procnotices", "procnotices", {NOTICE_PROJECT_PARAM: pid}, limit=500)
        saved += [{k: n.get(k) for k in NOTICE_FIELDS} | {"step_activity": _activity(n.get("bid_reference_no"))} for n in notices]
        matched, refs = set(), []
        for c in raw:
            m = map_contract(c, lookup)
            m.update(match_notices(m, notices))
            matched.update(m.pop("notice_ids"))
            refs.append(m["ref"])
            rows.append(m)
        opened += open_tenders(notices, matched, pull_date, lookup, contract_refs=refs)
    return rows, opened, saved


def write_procurement(out, rows, opened, saved):
    pd.DataFrame(opened, columns=["tender_id", "project_id", "notice_date", "censored_at", "category", "procurement_method",
                                  "description", "country_code", "sector", "step_ref", "step_activity", "submission_deadline"]
                 ).to_csv(f"{out}/open_tenders.csv", index=False)
    json.dump(saved, open(f"{out}/notices.json", "w", encoding="utf-8"), default=str)
    ct = pd.DataFrame(rows)
    ct["award_date"] = ct.sign_date
    ct["notice_date"] = ct.notice_date.fillna(ct.award_date)          # unmatched: tender_days becomes NaN downstream (<=0)
    ct.drop(columns=["sign_date", "no_objection_date"]).to_csv(f"{out}/contracts.csv", index=False)
    rules = ct.match_rule.value_counts().to_dict()
    print(f"contracts {len(ct)}; linked to a tender notice {100 * (ct.match_score > 0).mean():.0f}% {rules}; "
          f"open tenders (right-censored) {len(opened)}; notices {len(saved)}")


def get_bytes(url, retries=4):
    """A document download; the server sometimes answers with an empty body under load, so empty bodies are retried too."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "procintel-poc"}), timeout=180, context=SSL_CTX) as r:
                b = r.read()
            if b and (not url.lower().endswith(".pdf") or b[:4] == b"%PDF"):
                return b
        except Exception:
            if i == retries - 1:
                raise
        time.sleep(2 * 2 ** i)
    raise IOError(f"empty or invalid download: {url}")


def fetch_plans(out, pids=None):
    """The newest Procurement Plan of every project (public STEP export): parse the PDF (procplan.parse_plan), check every
    activity's dates against the text rendition, and write plan_activities.csv (one row per activity, with key planned/actual
    dates) and plan_stages.csv (one row per stage date). PDFs are cached in <out>/plans."""
    from .procplan import NOTICE_STAGES, fetch_latest, parse_plan, validate_against_text
    if pids is None:
        pids = pd.read_json(f"{out}/projects.json").project_id.tolist()
    acts, stages, summary = [], [], dict(projects=len(pids), with_plan=0, activities=0, text_checked=0, text_exact=0, failed=[])
    for pid in pids:
        try:
            m = fetch_latest(pid, f"{out}/plans", get_json, get_bytes)
        except Exception as e:
            summary["failed"].append(f"{pid}: {e}"[:120]); continue
        if not m:
            continue
        try:
            a, s = parse_plan(m["pdf"])
        except Exception as e:
            summary["failed"].append(f"{pid}: parse {e}"[:120]); continue
        v = dict(bad_refs=set(), checked_refs=set())
        if m["txt"]:
            v = validate_against_text(a, open(m["txt"], encoding="utf-8", errors="ignore").read())
            summary["text_checked"] += v["checked"]; summary["text_exact"] += v["exact"]
        summary["with_plan"] += 1; summary["activities"] += len(a)
        for x in a:
            ds = [r for r in s if r["step_ref"] == x["step_ref"]]
            get = lambda stage_set, kind: next((r["date"] for r in ds if r["stage"] in stage_set and r["kind"] == kind), None)
            acts.append(dict({k: v for k, v in x.items() if k != "dates_in_order"}, project_id=pid, plan_doc_id=m["plan_doc_id"],
                             plan_date=m["plan_date"], plan_pdf=m["pdf_url"], plan_versions=m["n_versions"],
                             # dates confirmed by the text rendition (True), contradicted (False: not used), or not checkable (None)
                             text_check=(x["step_ref"] not in v["bad_refs"]) if x["step_ref"] in v["checked_refs"] else None,
                             notice_planned=get(NOTICE_STAGES, "planned"), notice_actual=get(NOTICE_STAGES, "actual"),
                             signed_planned=get(("signed",), "planned"), signed_actual=get(("signed",), "actual"),
                             first_actual=min((r["date"] for r in ds if r["kind"] == "actual"), default=None)))
            stages += [dict(r, project_id=pid, plan_doc_id=m["plan_doc_id"]) for r in ds]
    pd.DataFrame(acts).to_csv(f"{out}/plan_activities.csv", index=False)
    pd.DataFrame(stages).to_csv(f"{out}/plan_stages.csv", index=False)
    print(f"procurement plans: {summary['with_plan']} of {summary['projects']} projects, {summary['activities']} activities; dates equal to "
          f"the text rendition for {summary['text_exact']} of {summary['text_checked']}; failed {len(summary['failed'])}")
    return summary


def refresh_procurement(out):
    """Re-fetch ONLY contract awards and procurement notices for the projects already in `out` (projects, documents and
    governance stay as they are), e.g. after a change to the notice matching."""
    pdf = pd.read_json(f"{out}/projects.json")
    lookup = pdf.set_index("project_id")[["country_code", "sector"]].to_dict("index")
    write_procurement(out, *fetch_procurement(pdf.project_id, lookup, pd.Timestamp.today().normalize()))


def _canon(word):
    return next((r for r in RATINGS if r.lower() == re.sub(r"\s+", " ", word).lower()), None)


def parse_isr_ratings(text: str) -> dict:
    """ISR rating table: '<label>   [glyph]Previous rating   [glyph]Current rating' on one line -> take the LAST rating on the line."""
    def grab(label):
        for m in re.finditer(label + r"[^\n]*", text, re.I):
            found = re.findall("(" + _RAT + ")", m.group(0), re.I)
            if found:
                return RATING_NUM[_canon(found[-1])]
        return np.nan
    return dict(dev_objective_rating=grab(r"(?:Progress towards achievement of PDO|Development Objective Rating)"),
                implementation_progress_rating=grab(r"Overall Implementation Progress"))


def parse_icr_rating(text: str):
    """ICR data sheet is a table: 'Outcome' then other column headers (Bank Performance, M&E Quality, Relevance of Prior Actions,
    Efficacy), then the values - the first value is Outcome. Else 'OUTCOME RATING  Rating: X', 'Outcome: X' or 'outcome was X'."""
    hdr = r"(?:Bank Performance|M&E Quality|Relevance of Prior Actions|Achievement of Objectives \(Efficacy\)|Rating:?)"
    for pat in (r"Outcome(?:\s+Rating)?\s+(?:" + hdr + r"\s+)*(" + _RAT + ")",
                r"(?:Overall\s+)?Outcome(?:\s+Rating)?[^A-Za-z\n]{0,60}(" + _RAT + ")",
                r"outcome(?: rating)?(?: (?:of|for) the project)? (?:was|is) (?:rated )?\W?(" + _RAT + ")"):
        m = re.search(pat, text, re.I)
        if m:
            return _canon(m.group(1))
    return None


def map_wgi_rows(rows_by_col: dict[str, list], publish_month=9, lag_years=1) -> pd.DataFrame:
    """rows_by_col: column -> list of API rows {country:{id,value}, date, value}. Keyed on ISO2 country id; availability lagged."""
    recs = {}
    for col, rows in rows_by_col.items():
        for r in rows or []:
            if r.get("value") is None or not r.get("country", {}).get("id"):
                continue
            k = (r["country"]["id"], int(r["date"]))
            recs.setdefault(k, dict(country_code=k[0], country=r["country"]["value"], region="", year=k[1],
                                    available_from=pd.Timestamp(k[1] + lag_years, publish_month, 30)))[col] = float(r["value"])
    df = pd.DataFrame(recs.values())
    for c in list(WGI) + ["cpia"]:
        if c not in df:
            df[c] = np.nan
    return df.sort_values(["country_code", "year"]).reset_index(drop=True)


# ------------------------------------------------------------------ orchestration (network; VERIFY)
def fetch_indicator(ind, source):
    out, page = [], 1
    while True:
        d = get_json(f"{WBAPI}/country/all/indicator/{ind}", dict(source=source, format="json", per_page=20000, page=page))
        out += d[1] or []
        if page >= d[0]["pages"]:
            return out
        page += 1


def iso3_to_iso2():
    d = get_json(f"{WBAPI}/country", dict(format="json", per_page=400))
    return {c["id"]: c["iso2Code"] for c in d[1]}


# One set of fetch defaults, used by the CLI and the Databricks notebooks (databricks.yml's `countries` variable must match:
# tests/test_databricks_layer.py checks it). They are the settings of the pull the published model metrics were measured on.
DEFAULT_COUNTRIES = "KE,GH,NP,BD,VN"
MAX_PROJECTS, MAX_DOCS = 100, 300


def build(out, countries, max_projects=MAX_PROJECTS, max_docs=MAX_DOCS):
    os.makedirs(f"{out}/docs", exist_ok=True)
    cs = {c.strip().upper() for c in countries}
    gov = {col: [r for r in fetch_indicator(ind, 3) if r["country"]["id"] in cs] for col, ind in WGI.items()}
    to2 = iso3_to_iso2()                                                  # CPIA (source 31) keys countries by ISO3, WGI by ISO2
    cpia = [dict(r, country=dict(r["country"], id=to2.get(r["country"]["id"], r["country"]["id"]))) for r in fetch_indicator(CPIA, 31)]
    gov["cpia"] = [r for r in cpia if r["country"]["id"] in cs]
    g = map_wgi_rows(gov)
    g.loc[g.available_from.notna()] .to_csv(f"{out}/governance.csv", index=False)
    projs = []
    for c in cs:   # the API lists newest first; half closed projects so ICRs / lessons exist, half active for monitoring
        for status in ("Closed", "Active"):
            q = {PROJECTS_COUNTRY_PARAM: c, "status_exact": status, "strdate": "2008-01-01", "fl": PROJECT_FIELDS}
            projs += [map_project(p) for p in paged("v2/projects", "projects", q, limit=max_projects // (2 * len(cs)) + 1)]
    pdf = pd.DataFrame(projs).dropna(subset=["approval_date"])
    pdf.to_json(f"{out}/projects.json", orient="records", date_format="iso")
    lookup = pdf.set_index("project_id")[["country_code", "sector"]].to_dict("index")
    write_procurement(out, *fetch_procurement(pdf.project_id, lookup, pd.Timestamp.today().normalize()))
    pd.DataFrame(columns=["project_id", "date", "project_age_years", "cum_disbursed_pct"]).to_csv(f"{out}/disbursements.csv", index=False)
    # documents + ISR ratings. Knowledge docs (ICR/PPAR/PAD) first across ALL projects, then ISRs, projects with an ICR first
    # (ISR-vs-ICR pairs feed the conflict check); otherwise the per-project ISR series eats the whole budget.
    names = g.drop_duplicates("country_code").set_index("country_code").country.to_dict()   # chunk country must match governance names
    meta, isr, n = {}, [], 0

    def fetch(pid, docty, tag, limit):
        nonlocal n
        got = 0
        try:
            listing = paged("v3/wds", "documents", {"docty_exact": docty, WDS_PROJECT_PARAM: pid}, limit=limit)
        except Exception as e:                                            # transient outage: skip this listing, keep the run
            print(f"skip {tag} listing for {pid}: {e}")
            return 0
        for d in listing:
            if n >= max_docs or not d.get("txturl"):
                continue
            fn = f"{tag}_{pid}_{d['id']}.txt"
            path = f"{out}/docs/{fn}"
            try:                                                          # reuse earlier downloads -> a rerun resumes
                text = open(path, encoding="utf-8").read() if os.path.exists(path) else get_text(d["txturl"])
            except Exception:
                continue
            open(path, "w", encoding="utf-8").write(text)
            pr = lookup.get(pid, {})
            pdf = (d.get("pdfurl") or d.get("url") or "").replace("http://", "https://")
            meta[fn] = dict(doc_id=f"{tag}_{pid}_{d['id']}", doc_type=tag, project_id=pid,     # pdf_url -> bronze_docs.meta_json -> app
                            meta=dict(Country=names.get(pr.get("country_code"), d.get("count", "")), Sector=pr.get("sector", ""),
                                      Date=str(d.get("docdt", ""))[:10], pdf_url=pdf))
            if tag == "ISR":
                isr.append(dict(isr_id=f"{pid}-{d['id']}", project_id=pid, date=str(d.get("docdt", ""))[:10], **parse_isr_ratings(text)))
            n += 1; got += 1
        return got

    with_icr = set()
    for docty, tag in (("Implementation Completion and Results Report", "ICR"), ("Project Performance Assessment Report", "PPAR"),
                       ("Project Appraisal Document", "PAD")):
        for pid in pdf.project_id:
            if fetch(pid, docty, tag, 1) and tag == "ICR":
                with_icr.add(pid)
    for pid in sorted(pdf.project_id, key=lambda p: p not in with_icr):
        fetch(pid, "Implementation Status and Results Report", "ISR", 6)
    json.dump(meta, open(f"{out}/docs/_meta.json", "w"))
    pd.DataFrame(isr).dropna(subset=["dev_objective_rating"]).to_json(f"{out}/isr.json", orient="records") if isr else open(f"{out}/isr.json", "w").write("[]")
    print(f"docs {n}; ISR rows {len(isr)}")


def fetch_all(out, countries=DEFAULT_COUNTRIES, max_projects=MAX_PROJECTS, max_docs=MAX_DOCS):
    """Everything a pull needs: feeds + documents (build) and the Procurement Plans (fetch_plans). Files already in `out` are
    reused, so `out` doubles as the download cache."""
    cs = countries.split(",") if isinstance(countries, str) else list(countries)
    build(out, cs, max_projects, max_docs)
    fetch_plans(out)


LAST = "projects.json"         # published last: a reader that sees it knows the batch is complete


def publish(src, dest):
    """Copy a finished pull from a staging folder to where it is ingested (landing/base, or a new landing/incoming batch).
    projects.json goes last, so the file-arrival trigger or bronze never reads a half-copied batch; unchanged files are skipped.
    Returns the number of files copied."""
    import shutil
    files = sorted((os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs),
                   key=lambda f: f == LAST)
    if LAST not in files:
        raise FileNotFoundError(f"{src} has no {LAST}: the pull did not finish")
    n = 0
    for f in files:
        a, b = os.path.join(src, f), os.path.join(dest, f)
        if f != LAST and os.path.exists(b) and os.path.getsize(a) == os.path.getsize(b):
            continue
        os.makedirs(os.path.dirname(b), exist_ok=True)
        shutil.copyfile(a, b); n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live"); ap.add_argument("--countries", default=DEFAULT_COUNTRIES)
    ap.add_argument("--max-projects", type=int, default=MAX_PROJECTS); ap.add_argument("--max-docs", type=int, default=MAX_DOCS)
    ap.add_argument("--procurement-only", action="store_true", help="re-fetch only contracts + notices for the projects in --out")
    ap.add_argument("--plans-only", action="store_true", help="fetch + parse only the Procurement Plans of the projects in --out")
    a = ap.parse_args()
    if a.plans_only:
        fetch_plans(a.out)
    elif a.procurement_only:
        refresh_procurement(a.out)
    else:
        fetch_all(a.out, a.countries, a.max_projects, a.max_docs)
