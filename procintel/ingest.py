"""Module 1: ETL, entity resolution, document parsing and structure-aware chunking."""
from __future__ import annotations
import bisect, glob, hashlib, json, os, re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ entity resolution
_LEGAL = {"ltd", "limited", "llc", "inc", "co", "company", "corp", "corporation", "sa", "pvt", "plc", "gmbh", "jv", "the"}
_ABBR = {"constr": "construction", "intl": "international", "eng": "engineering", "infra": "infrastructure"}
PID_RE = re.compile(r"P\d{6}")
CID_RE = re.compile(r"P\d{6}-C\d{2}")


def norm_project_id(x) -> str | None:
    m = PID_RE.search(str(x).upper().replace(" ", ""))
    return m.group(0) if m else None


def norm_supplier(name: str) -> str:
    toks = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split()
    return " ".join(_ABBR.get(t, t) for t in toks if t not in _LEGAL)


def resolve_suppliers(raw_names, threshold=0.88) -> dict:
    """Blocking on first token + string similarity. Returns raw -> canonical supplier id.
    Production: add address/tax-id/debarment-list features and a learned pairwise matcher."""
    norm = {r: norm_supplier(r) for r in set(raw_names)}
    freq = pd.Series([norm[r] for r in raw_names]).value_counts()
    canon: dict[str, str] = {}     # normalized -> canonical normalized
    blocks: dict[str, list] = {}
    for n in freq.index:           # most frequent first => becomes the canonical form
        first = n.split()[0] if n else ""
        hit = next((c for c in blocks.get(first, []) if SequenceMatcher(None, n, c).ratio() >= threshold), None)
        if hit is None:
            blocks.setdefault(first, []).append(n)
            hit = n
        canon[n] = hit
    ids = {c: "S" + hashlib.md5(c.encode()).hexdigest()[:8] for c in set(canon.values())}
    return {r: ids[canon[norm[r]]] for r in norm}


# ------------------------------------------------------------------ structured ETL
@dataclass
class Dataset:
    projects: pd.DataFrame
    contracts: pd.DataFrame
    governance: pd.DataFrame
    disbursements: pd.DataFrame
    isr: pd.DataFrame
    docs: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    open_tenders: pd.DataFrame = field(default_factory=pd.DataFrame)   # right-censored: noticed, no contract yet
    activities: pd.DataFrame = field(default_factory=pd.DataFrame)     # one row per STEP procurement activity
    plan: pd.DataFrame = field(default_factory=pd.DataFrame)           # Procurement Plan activities (planned / actual dates)


REQUIRED = {"contracts": ["contract_id", "project_id", "category", "procurement_method", "notice_date", "country_code", "sector"],
            "projects": ["project_id", "country_code", "sector", "approval_date", "closing_date", "status"]}


def _check(df, name):
    miss = [c for c in REQUIRED[name] if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing required columns {miss}")


def _squash_ws(s: pd.Series) -> pd.Series:
    """'Consultant Qualification  Selection' and '... Qualification Selection' are one method."""
    return s.where(s.isna(), s.astype(str).str.split().str.join(" "))


def enrich_contracts(contracts: pd.DataFrame, governance: pd.DataFrame, supplier_map: dict | None = None) -> pd.DataFrame:
    """Normalize ids, derive tender_days, as-of join governance (publication-lagged => no look-ahead leakage)."""
    c = contracts.copy()
    c["project_id"] = c.project_id.map(norm_project_id)
    c = c.drop_duplicates("contract_id").dropna(subset=["project_id"])
    for col in ("notice_date", "award_date"):
        c[col] = pd.to_datetime(c[col], errors="coerce")
    c["procurement_method"] = _squash_ws(c.procurement_method)
    c["tender_days"] = (c.award_date - c.notice_date).dt.days
    c.loc[c.tender_days <= 0, "tender_days"] = pd.NA
    if "match_score" in c:   # live data: duration is a notice->signing proxy
        # fuzzy notice matches < 0.8 run ~2x longer than exact-reference matches => mostly the wrong notice; drop the label
        c.loc[c.match_score < 0.8, "tender_days"] = pd.NA
    if "match_rule" in c:   # re-tender / amendment labels exist only where the notices of THIS package are known: a STEP activity
        # or exact-reference link. Description links labelled 58% of contracts 're-tendered' (vs <1% for reliable links): look-alike
        # notices of other packages, not re-advertisements. Elsewhere the label is unknown, not 0.
        unreliable = ~c.match_rule.isin(["step_activity", "exact_ref"])
        c["rebid"] = c.rebid.astype(float); c["amendments"] = c.amendments.astype(float)
        c.loc[unreliable, ["rebid", "amendments"]] = np.nan
        # recent notices are visible only if they signed fast: handled by the survival model (open tenders = censored rows)
    c["tender_days"] = c.tender_days.astype(float)
    v = pd.to_numeric(c.value_usd_m, errors="coerce")
    c["log_value"] = np.log(v.where(v > 0))
    c["notice_month"] = c.notice_date.dt.month
    if supplier_map is None:
        supplier_map = resolve_suppliers(c.supplier_raw.fillna("UNKNOWN").tolist())
    c["supplier_id"] = c.supplier_raw.fillna("UNKNOWN").map(supplier_map)
    g = governance.copy()
    g["available_from"] = pd.to_datetime(g.available_from)
    c = c.sort_values("notice_date")
    parts = []
    for code, grp in c.groupby("country_code"):
        gg = g[g.country_code == code].sort_values("available_from")[["available_from", "wgi_rule_of_law", "wgi_gov_effectiveness", "wgi_reg_quality", "cpia"]]
        parts.append(pd.merge_asof(grp, gg, left_on="notice_date", right_on="available_from", direction="backward"))
    c = pd.concat(parts).drop(columns="available_from")
    return c.reset_index(drop=True)


def _asof_governance(df, governance, date_col="notice_date"):
    g = governance.copy(); g["available_from"] = pd.to_datetime(g.available_from)
    cols = ["available_from", "wgi_rule_of_law", "wgi_gov_effectiveness", "wgi_reg_quality", "cpia"]
    parts = []
    for code, grp in df.sort_values(date_col).groupby("country_code"):
        parts.append(pd.merge_asof(grp, g[g.country_code == code].sort_values("available_from")[cols],
                                   left_on=date_col, right_on="available_from", direction="backward"))
    return (pd.concat(parts).drop(columns="available_from") if parts else df.reindex(columns=list(df.columns) + cols[1:])).reset_index(drop=True)


def enrich_open_tenders(open_tenders: pd.DataFrame, governance: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Same feature view as contracts (value unknown until award -> NaN, the model handles missing)."""
    if open_tenders is None or not len(open_tenders):
        return pd.DataFrame()
    o = open_tenders.copy()
    o["project_id"] = o.project_id.map(norm_project_id)
    for col in ("notice_date", "censored_at"):
        o[col] = pd.to_datetime(o[col], errors="coerce")
    if "submission_deadline" in o:                               # bids/EOIs due: before this the tender is still being advertised
        o["submission_deadline"] = pd.to_datetime(o.submission_deadline, errors="coerce")
    o = o.dropna(subset=["notice_date", "censored_at", "project_id"])
    o["procurement_method"] = _squash_ws(o.procurement_method)
    o["log_value"] = np.nan; o["notice_month"] = o.notice_date.dt.month; o["review_type"] = np.nan
    o = _asof_governance(o, governance)
    return o.merge(projects[["project_id", "es_risk"]], on="project_id", how="left")


def read_raw(path: str) -> dict:
    """Read CSV/JSON feeds + documents from a folder (what lands in the bronze layer)."""
    contracts = pd.read_csv(os.path.join(path, "contracts.csv"))
    projects = pd.read_json(os.path.join(path, "projects.json"), convert_dates=["approval_date", "closing_date"])
    gov = pd.read_csv(os.path.join(path, "governance.csv"), parse_dates=["available_from"])
    disb = pd.read_csv(os.path.join(path, "disbursements.csv"), parse_dates=["date"])
    isr = pd.read_json(os.path.join(path, "isr.json"), convert_dates=False)
    for col in ("isr_id", "project_id", "date", "dev_objective_rating", "implementation_progress_rating"):
        if col not in isr:
            isr[col] = pd.Series(dtype="object")
    isr["date"] = pd.to_datetime(isr["date"], errors="coerce")
    docs = [parse_document(f) for f in sorted(glob.glob(os.path.join(path, "docs", "*"))) if not os.path.basename(f).startswith("_")
            and f.lower().endswith((".md", ".txt", ".pdf"))]
    mp = os.path.join(path, "docs", "_meta.json")              # sidecar metadata for fetched (live) documents
    if os.path.exists(mp):
        sidecar = json.load(open(mp))
        for d in docs:
            d.update(sidecar.get(d["filename"], {}))
    op = os.path.join(path, "open_tenders.csv")
    open_t = pd.read_csv(op) if os.path.exists(op) else pd.DataFrame()
    np_ = os.path.join(path, "notices.json")                   # procurement notices as fetched (live data only)
    notices = pd.DataFrame(json.load(open(np_, encoding="utf-8"))) if os.path.exists(np_) else pd.DataFrame()
    pp = os.path.join(path, "plan_activities.csv")             # Procurement Plans (public STEP exports), parsed at fetch time
    plan = pd.read_csv(pp) if os.path.exists(pp) and os.path.getsize(pp) > 2 else pd.DataFrame()
    return dict(contracts=contracts, projects=projects, governance=gov, disbursements=disb, isr=isr, docs=docs, open_tenders=open_t,
                notices=notices, plan=plan)


_DOC_DATE = re.compile(r"(?:Archived on|Date:)\s*(\d{1,2}-[A-Z][a-z]{2}-\d{4}|\d{4}-\d{2}-\d{2})")


def _fill_doc_meta(docs, projects, gov):
    """A document without sidecar metadata (e.g. a file kept from an earlier download run: 55 ISRs on the live pull) still
    gets Country/Sector from its project record and a Date from its text, so it is labelled and country filters find it."""
    names = gov.drop_duplicates("country_code").set_index("country_code").country.to_dict() if "country" in gov else {}
    proj = projects.drop_duplicates("project_id").set_index("project_id")
    out = []
    for d in docs:
        meta = dict(d.get("meta") or {})
        has = lambda k: bool(meta.get(k)) or bool(re.search(rf"^{k}:\s*\S", d["text"], re.M))
        if d.get("project_id") in proj.index and not (has("Country") and has("Sector")):
            p = proj.loc[d["project_id"]]
            meta.setdefault("Country", names.get(p.get("country_code"), "") or "")
            meta.setdefault("Sector", p.get("sector", "") or "")
            if not has("Date"):
                m = _DOC_DATE.search(d["text"][:3000])
                if m:
                    dt = pd.to_datetime(m.group(1), errors="coerce")
                    meta["Date"] = "" if pd.isna(dt) else dt.strftime("%Y-%m-%d")
            d = {**d, "meta": {k: v for k, v in meta.items() if v}}
        out.append(d)
    return out


def _dedupe_docs(docs):
    """The World Bank publishes some reports several times under different document ids (up to 5 identical copies of one ISR
    on the live pull). Keep one copy per project and text, so passages are not repeated in search and answers. A report that
    covers two projects (one ICR, two project ids) stays with each project."""
    import hashlib
    seen, out = set(), []
    for d in sorted(docs, key=lambda d: d["doc_id"]):
        key = (d.get("project_id"), hashlib.md5(" ".join(d["text"].split()).encode()).hexdigest())
        if key not in seen:
            seen.add(key); out.append(d)
    return out


def _dedupe_isr(isr):
    """One status-report rating per project and date (the copies above parse to identical rows); otherwise the copies show as
    repeated reports and shift the 'rating fell over the last two reports' comparison."""
    if not len(isr):
        return isr
    return isr.sort_values(["project_id", "date", "isr_id"]).drop_duplicates(["project_id", "date"], keep="first").reset_index(drop=True)


def build_activities(contracts: pd.DataFrame, notices, open_tenders=None) -> pd.DataFrame:
    """One row per STEP procurement activity (country + activity number), joining the two feeds on that key: first notice,
    submission deadline, award-notice date, contract signing, notice counts, revisions, cancellation, status."""
    cols = ["step_activity", "project_id", "first_notice_date", "submission_deadline", "award_notice_date", "signed_date", "n_contracts",
            "n_tender_notices", "n_revised", "cancelled", "status"]
    if notices is None or not len(notices) or "step_activity" not in notices:
        return pd.DataFrame(columns=cols)
    n = notices.dropna(subset=["step_activity"]).copy()
    n["date"] = pd.to_datetime(n.noticedate, format="%d-%b-%Y", errors="coerce")
    n["deadline"] = pd.to_datetime(n.get("submission_deadline_date"), errors="coerce", utc=True).dt.tz_localize(None)
    typ = n.notice_type.fillna("").str.lower()
    n["award"] = typ.str.contains("award"); n["tender"] = ~n.award & ~typ.str.contains("general")
    n["revised"] = n.notice_status.fillna("").eq("Revised")
    n["cancel"] = n.notice_status.fillna("").str.lower().str.contains("cancel") | typ.str.contains("cancel")
    g = n.groupby(["step_activity", "project_id"])
    act = pd.DataFrame(dict(
        first_notice_date=g.apply(lambda x: x.loc[x.tender, "date"].min()), submission_deadline=g.apply(lambda x: x.loc[x.tender, "deadline"].max()),
        award_notice_date=g.apply(lambda x: x.loc[x.award, "date"].min()), n_tender_notices=g.apply(lambda x: int((x.tender & ~x.revised).sum())),
        n_revised=g.revised.sum().astype(int), cancelled=g.cancel.any())).reset_index()
    c = contracts.dropna(subset=["step_activity"]) if "step_activity" in contracts else pd.DataFrame(columns=["step_activity", "award_date"])
    signed = c.groupby("step_activity").agg(signed_date=("award_date", "min"), n_contracts=("contract_id", "count")).reset_index()
    act = act.merge(signed, on="step_activity", how="left")
    act["n_contracts"] = act.n_contracts.fillna(0).astype(int)
    open_ids = set(open_tenders.step_activity.dropna()) if open_tenders is not None and len(open_tenders) and "step_activity" in open_tenders else set()
    act["status"] = np.select([act.n_contracts > 0, act.award_notice_date.notna(), act.cancelled, act.step_activity.isin(open_ids)],
                              ["signed", "awarded", "cancelled", "open"], "other")
    return act[cols]


def _plan_table(plan) -> pd.DataFrame:
    """Plan activities with parsed dates; activities whose dates the text rendition contradicts (text_check False) are dropped."""
    if plan is None or not len(plan):
        return pd.DataFrame()
    p = plan.copy()
    p = p[p.text_check.astype(str).str.lower() != "false"] if "text_check" in p else p
    for c in ("notice_planned", "notice_actual", "signed_planned", "signed_actual", "first_actual", "plan_date"):
        if c in p:
            p[c] = pd.to_datetime(p[c], errors="coerce")
    for c in ("estimated_usd", "actual_usd"):
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    p["project_id"] = p.project_id.map(norm_project_id)
    p["code"] = p.step_ref.map(_compact)
    return p.reset_index(drop=True)


def _compact(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def link_plan(contracts: pd.DataFrame, plan: pd.DataFrame):
    """Link contracts to their Procurement Plan activity, in order: same STEP activity number; same activity code (agencies' own
    codes such as 'CWSIP/W11' appear in both the plan and the award data); same actual amount (within 1%) and signing date
    (within 14 days). A contract with no reliable notice link (none, or a description guess) takes its tender start from the
    plan's ACTUAL notice date (expression of interest / specific procurement notice): the duration means the same as before,
    notice -> signing, now for packages the notices feed never linked."""
    stats = dict(plan_activities=int(len(plan)))
    if not len(plan) or not len(contracts) or "match_rule" not in contracts:
        return contracts, stats
    c = contracts.copy()
    c["_award"] = pd.to_datetime(c.award_date, errors="coerce")
    by_act = {(r.project_id, r.step_activity): r for r in plan.dropna(subset=["step_activity"]).itertuples()}
    by_code = {(r.project_id, r.code): r for r in plan.itertuples() if r.code}
    amt = plan.dropna(subset=["actual_usd", "signed_actual"])
    amt = amt[amt.actual_usd > 0]
    plan_ref, plan_rule, used = [], [], 0
    for i, r in c.iterrows():
        pid = norm_project_id(r.project_id)
        hit, how = None, None
        if isinstance(r.get("step_activity"), str) and (pid, r.step_activity) in by_act:
            hit, how = by_act[(pid, r.step_activity)], "plan_step"
        elif (pid, _compact(r.get("ref"))) in by_code and _compact(r.get("ref")):
            hit, how = by_code[(pid, _compact(r.get("ref")))], "plan_code"
        elif pd.notna(r.value_usd_m) and pd.notna(r._award):
            cand = amt[(amt.project_id == pid) & ((amt.actual_usd - r.value_usd_m * 1e6).abs() <= 0.01 * r.value_usd_m * 1e6)
                       & ((amt.signed_actual - r._award).dt.days.abs() <= 14)]
            if len(cand) == 1:
                hit, how = next(cand.itertuples()), "plan_amount_date"
        plan_ref.append(hit.step_ref if hit is not None else None); plan_rule.append(how)
        reliable = r.match_rule in ("step_activity", "exact_ref")
        # Direct Selection has no competitive tender (the 'invitation' goes to one chosen firm): its plan dates describe a
        # different process, and adding them made the duration backtest worse (92 vs 79 days), so they give no start date
        direct = "direct" in str(r.get("procurement_method", "")).lower()
        if hit is not None and not reliable and not direct and pd.notna(hit.notice_actual) and pd.notna(r._award):
            days = (r._award - hit.notice_actual).days
            if 7 <= days <= 1500:
                c.at[i, "notice_date"] = hit.notice_actual
                c.at[i, "match_rule"], c.at[i, "match_score"] = how, 1.0 if how != "plan_amount_date" else 0.9
                used += 1
    c["plan_ref"], c["plan_link"] = plan_ref, plan_rule
    stats.update(plan_linked_contracts=int(c.plan_link.notna().sum()),
                 plan_link_by_rule={k: int(v) for k, v in c.plan_link.value_counts().items()},
                 plan_start_dates_added=used)
    return c.drop(columns=["_award"]), stats


def build_dataset(raw: dict) -> Dataset:
    """Silver transformation: validate, normalise ids, entity-resolve, as-of join governance."""
    contracts, projects, gov = raw["contracts"], raw["projects"].copy(), raw["governance"]
    _check(contracts, "contracts"); _check(projects, "projects")
    projects["project_id"] = projects.project_id.map(norm_project_id)
    plan = _plan_table(raw.get("plan"))
    contracts, plan_stats = link_plan(contracts, plan)
    contracts = enrich_contracts(contracts, gov)
    contracts = contracts.merge(projects[["project_id", "es_risk"]], on="project_id", how="left")
    docs = _dedupe_docs(_fill_doc_meta(raw["docs"], projects, gov))
    isr = _dedupe_isr(raw["isr"])
    q = {"contracts": len(contracts), "missing_value_pct": float(round(100 * contracts.value_usd_m.isna().mean(), 1)),
         "missing_award_date_pct": float(round(100 * contracts.award_date.isna().mean(), 1)),
         "suppliers_raw": contracts.supplier_raw.nunique(), "suppliers_resolved": contracts.supplier_id.nunique(),
         "orphan_contracts": int((~contracts.project_id.isin(projects.project_id)).sum()), "docs": len(docs)}
    open_t = enrich_open_tenders(raw.get("open_tenders"), gov, projects)
    q["open_tenders"] = len(open_t)
    if "match_rule" in contracts:                              # how contracts were linked to their tender notice (live data)
        q["notice_link_by_rule"] = {k: int(v) for k, v in contracts.match_rule.fillna("none").value_counts().items()}
        q["notice_link_pct"] = float(round(100 * (contracts.match_rule.fillna("none") != "none").mean(), 1))
    activities = build_activities(contracts, raw.get("notices"), open_t)
    q["step_activities"] = len(activities)
    q.update(plan_stats)
    return Dataset(projects, contracts, gov, raw["disbursements"], isr, docs, q, open_t, activities, plan)


def load_dataset(path: str) -> Dataset:
    return build_dataset(read_raw(path))


# ------------------------------------------------------------------ documents
def _promote_headings(text: str) -> str:
    """World Bank .txt reports have no markup: promote ALL-CAPS lines and 'I. / 1.' numbered titles to headings."""
    out = []
    for ln in text.split("\n"):            # not splitlines(): it also splits on, and drops, the \f page breaks page numbers rely on
        t = ln.strip()
        if (t.isupper() and 4 <= len(t) <= 100 and sum(c.isalpha() for c in t) >= 4) or re.match(r"^([IVX]+|\d{1,2})\.\s+[A-Z][A-Za-z ,&/\-]{3,80}$", t):
            out.append("\f" * ln.count("\f") + "\n## " + t.title() + "\n")     # a heading at the top of a page keeps its page break
        else:
            out.append(ln)
    return "\n".join(out)


def parse_document(path: str) -> dict:
    """md/txt natively; PDF via pypdf with a heading heuristic (page breaks kept as \\f).
    Production: layout-aware parser (e.g. Docling/Unstructured) for scanned pages, multi-column text and complex tables."""
    base = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(path).pages]
        lines = []
        for pg in pages:
            for ln in pg.splitlines():
                s = ln.strip()
                if re.match(r"^(\d+(\.\d+)*\.?|Annex \d+\.?)\s+[A-Z][^.]{2,80}$", s):
                    s = "## " + s
                lines.append(s)
            lines.append("\f")
        text = "\n".join(lines)
    else:
        text = open(path, encoding="utf-8").read()
    if not re.search(r"^#{1,3}\s", text, re.M):
        text = _promote_headings(text)
    stem = os.path.splitext(base)[0]
    dtype = stem.split("_")[0] if stem.split("_")[0] in ("ICR", "PAD", "PPAR", "ISR") else ("ISR" if "-ISR" in stem else "DOC")
    m = re.search(r"Project:\s*(P\d{6})", text)
    return dict(doc_id=stem, doc_type=dtype, project_id=m.group(1) if m else norm_project_id(stem), filename=base, text=text)


CHUNKER_VERSION = 5          # bump when chunking or doc metadata changes => silver_gold re-chunks every doc (v5: country from the project when a doc lacks metadata)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    project_id: str
    section_path: str
    section_kind: str      # lessons | procurement | ratings | context | annex | other
    chunk_type: str        # text | table
    page: int
    text: str              # verbatim, quotable
    embed_text: str        # contextualised for embedding
    country: str
    sector: str
    date: str
    content_hash: str = ""


def _kind(path):
    p = path.lower()
    return ("lessons" if "lesson" in p else "procurement" if ("procure" in p or "fiduciary" in p) else "ratings" if "rating" in p else
            "annex" if "annex" in p else "context" if ("context" in p or "objective" in p) else "other")


def _split_sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def chunk_document(doc: dict, max_words=110) -> list[Chunk]:
    """Heading-hierarchy aware chunking: paragraphs/bullets are atomic, consecutive short paragraphs of the same section are
    packed up to max_words (real WB .txt files put a blank line after nearly every line - without packing a 150-page ICR became
    ~1,500 one-line chunks), long paragraphs split on sentences with one-sentence overlap, tables are kept as separate table
    chunks (header repeated every 12 rows), and each chunk carries a context prefix."""
    text = doc["text"]
    meta = {k: (re.search(rf"^{k}:\s*(.+)$", text, re.M).group(1).strip() if re.search(rf"^{k}:\s*(.+)$", text, re.M) else "")
            for k in ("Country", "Sector", "Date")}
    meta.update({k: v for k, v in (doc.get("meta") or {}).items() if v})
    title = (re.search(r"^#\s+(.+)$", text, re.M) or [None, doc["doc_id"]])[1]
    chunks, stack, buf, tbl, pack, n = [], [], [], [], [], 0
    char_pos = 0

    breaks = [m.start() for m in re.finditer("\f", text)]

    def page_of(pos):  # \f-delimited PDF pages (WB text renditions keep one per PDF page), else ~3000 chars/page
        return bisect.bisect_right(breaks, pos) + 1 if breaks else pos // 3000 + 1

    def emit(body, ctype, pos):
        nonlocal n
        path = " > ".join(stack) or title
        prefix = f"[{doc['doc_type']} {doc['project_id']} | {meta['Country']} | {meta['Sector']} | {path}]"
        c = Chunk(f"{doc['doc_id']}#{n:04d}", doc["doc_id"], doc["doc_type"], doc["project_id"], path, _kind(path), ctype,
                  page_of(pos), body, f"{prefix} {body}", meta["Country"], meta["Sector"], meta["Date"],
                  hashlib.md5(body.encode()).hexdigest())
        chunks.append(c); n += 1

    def flush_pack():
        if pack:
            emit("\n".join(p for p, _ in pack), "text", pack[0][1]); pack.clear()

    def flush_para(_pos=None):
        """buf holds (line, char position); a chunk's page is where its text STARTS, not where the paragraph was closed
        (the blank line closing a paragraph at the foot of a page already lies on the next page)."""
        if not buf:
            return
        lines = list(buf); buf.clear()
        para = " ".join(t for t, _ in lines)
        starts, off = [], 0
        for t, p in lines:
            starts.append((off, p)); off += len(t) + 1
        at = lambda o: next(p for st, p in reversed(starts) if st <= o)
        w = len(para.split())
        if w <= max_words:
            if pack and sum(len(x.split()) for x, _ in pack) + w > max_words:
                flush_pack()
            pack.append((para, lines[0][1])); return
        flush_pack()
        sents, cur, cursor, first = _split_sentences(para), [], 0, None
        for sn in sents:
            o = para.find(sn, cursor); o = cursor if o < 0 else o; cursor = o + len(sn)
            if not cur:
                first = o
            cur.append((sn, o))
            if sum(len(x.split()) for x, _ in cur) >= max_words:
                emit(" ".join(x for x, _ in cur), "text", at(first)); cur = cur[-1:]; first = cur[0][1]
        if len(cur) > 1 or (cur and not chunks[-1].text.endswith(cur[-1][0])):
            emit(" ".join(x for x, _ in cur), "text", at(first))

    def flush_tbl(_pos=None):
        if not tbl:
            return
        flush_pack()
        head, rows = tbl[:2], tbl[2:]
        for i in range(0, max(len(rows), 1), 12):                # page of the block's first row
            emit("\n".join(t for t, _ in head + rows[i:i + 12]), "table", (rows[i] if rows else head[0])[1])
        tbl.clear()

    for line in text.splitlines(keepends=True):
        s = line.strip(); pos = char_pos; char_pos += len(line)
        if "\f" in line:
            flush_pack()                                        # never pack across a page break (a sentence may still run over it)
        if s.startswith("|"):
            flush_para(pos); tbl.append((s, pos)); continue
        flush_tbl(pos)
        if s.startswith("#"):
            flush_para(pos); flush_pack()                       # never pack across a section boundary
            lvl = len(s) - len(s.lstrip("#"))
            if lvl == 1:
                continue
            stack[:] = stack[:lvl - 2] + [s.lstrip("# ").strip()]
        elif not s:
            # hard-wrapped WB text has blank lines INSIDE sentences: only a sentence end (or a runaway buffer) closes a paragraph
            if not buf or re.search(r"[.!?:;]['\")\]]*$", buf[-1][0]) or sum(len(b.split()) for b, _ in buf) > 3 * max_words:
                flush_para(pos)
        elif re.match(r"^(Country|Sector|Project|Date):", s):
            continue
        elif re.fullmatch(r"(?:Page\s+)?\d{1,3}(?:\s+of\s+\d{1,3})?", s):
            continue                                            # a printed page number (footer/header), not report text
        elif s.startswith("- "):
            flush_para(pos); buf.append((s[2:], pos)); flush_para(pos)
        else:
            buf.append((s, pos))
    flush_para(char_pos); flush_pack(); flush_tbl(char_pos)
    return chunks
