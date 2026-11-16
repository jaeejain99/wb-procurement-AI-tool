"""Application services behind the web UI: one function per screen, no HTTP here (app.py is a thin FastAPI layer).

Package workspace   analyze_package  - schedule + risks (fixed pipeline), market-approach options, lesson-derived mitigations
Portfolio monitor   portfolio, project_page
Ask                 the conversational agent (agent.py)
Evidence library    search, passage
Insights            insights - models, benchmark, usage, what is new, data quality"""
from __future__ import annotations
import glob, json, os, re, threading
from functools import lru_cache
import numpy as np
import pandas as pd
from .guardrails import RATING_NAME, resolve_conflict
from .monitoring import as_of_date, overdue_tenders, project_warnings
from .rag import PackageSpec
from .serving import report_to_json
from . import doc_links

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ helpers
def _vals(s):
    return sorted(x for x in s.dropna().unique().tolist() if str(x))


def options(plat) -> dict:
    c, names = plat.ds.contracts, plat.country_names
    return dict(countries=[dict(code=k, name=names.get(k, k)) for k in _vals(c.country_code)], sectors=_vals(c.sector),
                categories=_vals(c.category), methods=_vals(c.procurement_method), review_types=_vals(c.review_type) or ["Prior", "Post"],
                es_risks=_vals(c.es_risk) or ["Moderate"], doc_types=sorted({ch.doc_type for ch in plat.index.chunks}),
                as_of=str(as_of_date(plat).date()))


def make_spec(body: dict, plat) -> PackageSpec:
    o = options(plat)
    for k, allowed in (("category", o["categories"]), ("sector", o["sectors"]), ("method", o["methods"])):
        if body.get(k) not in allowed:
            raise ValueError(f"{k} must be one of the values in the data")
    if body.get("country_code") not in {c["code"] for c in o["countries"]}:
        raise ValueError("country_code is not in the data")
    val = body.get("value_usd_m")
    val = float(val) if val not in (None, "") else None
    if val is not None and not (0 < val < 1e5):
        raise ValueError("value must be between 0 and 100,000 (USD m)")
    nd = str(body.get("notice_date") or as_of_date(plat).date())
    pd.Timestamp(nd)
    return PackageSpec(str(body.get("description", ""))[:300], body["category"], body["sector"], body["country_code"], val,
                       body["method"], body.get("review_type") or "Prior", nd, body.get("es_risk") or "Moderate")


# ------------------------------------------------------------------ package workspace
def market_options(plat, cop, spec: PackageSpec, top=4) -> dict:
    """Model-based comparison of procurement methods (and lotting) for the same package. Advisory: procurement rules and
    thresholds still decide which methods are allowed."""
    c = plat.ds.contracts
    MIN_PAST = 10            # below this, the model would extrapolate (seen: 2,805 days for a method with 0 signed contracts)
    signed = c[(c.category == spec.category) & c.tender_days.notna()]
    counts = signed.procurement_method.value_counts()
    methods = [m for m in counts.index if counts[m] >= MIN_PAST][:top]
    if spec.method not in methods:
        methods = [spec.method] + methods[:top - 1]
    rows = []
    for m in methods:
        s = PackageSpec(spec.description, spec.category, spec.sector, spec.country_code, spec.value_usd_m, m, spec.review_type,
                        spec.notice_date, spec.es_risk)
        row = cop.feature_row(s); pr = plat.models.predict_df(row).iloc[0]
        comp = c[(c.category == spec.category) & (c.procurement_method == m)].dropna(subset=["tender_days"])
        rows.append(dict(method=m, chosen=m == spec.method, thin=len(comp) < MIN_PAST, p10_days=int(round(pr.p10_days)), p50_days=int(round(pr.p50_days)),
                         p90_days=int(round(pr.p90_days)), rebid_pct=round(100 * float(pr.p_rebid), 1),
                         past_contracts=int(len(comp)), past_median_days=int(comp.tender_days.median()) if len(comp) else None,
                         past_rebid_pct=round(100 * float(comp.rebid.mean()), 1) if len(comp) else None))
    solid = [r for r in rows if not r["thin"]]
    if len(solid) > 1:                                        # badges only between well-supported methods
        min(solid, key=lambda r: r["p50_days"])["fastest"] = True
        min(solid, key=lambda r: r["rebid_pct"])["lowest_rebid"] = True
    # No lot-size comparison (removed 10 Nov 2026): the duration model's value effect is coarse (depth-2 trees) and small next
    # to the forecast range, so 1 / 2 / 4 lots came out within days of each other, sometimes the wrong way round. Bring it back
    # when a model separates lot sizes on held-out data.
    return dict(methods=rows,
                min_past_contracts=MIN_PAST,
                note="These are estimates from past World Bank contracts, to compare options. The Procurement Regulations, thresholds "
                     "and review requirements decide which methods this package may actually use.")


MITIGATION_PROMPT = """You advise a procurement team. From the lesson below ONLY, write ONE concrete mitigation for the risk "{theme}"
in a new {category} package, in at most 30 words, starting with a verb. Do not add facts, numbers, countries or names that the
lesson does not contain. Lesson: "{quote}" """


def lesson_mitigations(findings, client=None, judge=None):
    """Replace each risk's standard mitigation with one written from its own cited lesson, kept only if the judge agrees the
    lesson supports it. Without a model (offline) the standard text stays, labelled as such."""
    from .llm import JUDGE_MODEL, LLMUnavailable, text_of
    for f in findings:
        f["mitigation_source"] = "standard practice"
    risks = [f for f in findings if f.get("kind") == "risk" and f.get("evidence")]
    if client is None or not risks:
        return findings
    drafts = []
    for f in risks:
        try:
            msg = client.chat(JUDGE_MODEL, [{"role": "user", "content": MITIGATION_PROMPT.format(
                theme=str(f.get("theme", "")).replace("_", " "), category=f.get("category", ""), quote=f["evidence"][0]["quote"][:900])}],
                max_tokens=120)
            drafts.append(text_of(msg).strip().strip('"'))
        except LLMUnavailable:
            return findings
    verdicts = judge([(d, [f["evidence"][0]["quote"]]) for d, f in zip(drafts, risks)]) if judge else None
    for i, (d, f) in enumerate(zip(drafts, risks)):
        if d and verdicts and verdicts[i][0]:
            f["standard_recommendation"], f["recommendation"], f["mitigation_source"] = f.get("recommendation"), d, "from the cited lesson"
    return findings


def analyze_package(state, body: dict) -> dict:
    plat, cop = state["plat"], state["cop"]
    spec = make_spec(body, plat)
    rep = json.loads(report_to_json(cop.analyze(spec)))
    for f in rep["findings"]:
        f["category"] = spec.category
    lesson_mitigations(rep["findings"], None if state.get("offline") else state["agent"].client, cop.judge)
    rep["market"] = market_options(plat, cop, spec)
    rep["spec"] = dict(category=spec.category, sector=spec.sector, country_code=spec.country_code, value_usd_m=spec.value_usd_m,
                       method=spec.method, review_type=spec.review_type, description=spec.description,
                       country=plat.country_names.get(spec.country_code, spec.country_code))
    return rep


# ------------------------------------------------------------------ portfolio monitor
def portfolio(state, country: str | None = None, as_of: str | None = None, top: int = 50) -> dict:
    plat = state["plat"]
    as_of = pd.Timestamp(as_of) if as_of else as_of_date(plat)
    pr = plat.ds.projects[plat.ds.projects.status == "Active"]
    if country:
        pr = pr[pr.country_code == country]
    bias = plat.bias()
    rows = []
    for r in pr.itertuples():
        w = project_warnings(plat, r.project_id, str(as_of.date()))
        isr = plat.ds.isr[(plat.ds.isr.project_id == r.project_id) & (plat.ds.isr.date <= as_of)].sort_values("date")
        conf = resolve_conflict(r.project_id, plat.ds.isr, plat.icr.get(r.project_id), bias=bias)
        rows.append(dict(project_id=r.project_id, name=r.project_name, country=plat.country_names.get(r.country_code, r.country_code),
                         country_code=r.country_code, sector=r.sector, risk_score=w["risk_score"], flags=w["flags"], alerts=w["alerts"],
                         open_tenders=w["open_tenders"], overdue_tenders=w["overdue_tenders"],
                         last_isr=RATING_NAME.get(int(isr.dev_objective_rating.iloc[-1])) if len(isr) else None,
                         rating_conflict=bool(conf)))
    rows.sort(key=lambda x: (-x["risk_score"], -x["overdue_tenders"], x["project_id"]))
    return dict(as_of=str(as_of.date()), projects=rows[:top], total_active=len(rows),
                flagged=sum(1 for x in rows if x["flags"]), overdue_tenders=sum(x["overdue_tenders"] for x in rows),
                isr_optimism=bias)


def project_page(state, pid: str) -> dict:
    plat = state["plat"]; ds = plat.ds; pid = str(pid).upper().strip()
    pr = ds.projects[ds.projects.project_id == pid]
    if not len(pr):
        raise KeyError(f"project {pid} is not in the data")
    r = pr.iloc[0]; as_of = as_of_date(plat)
    isr = ds.isr[ds.isr.project_id == pid].sort_values("date")
    c = ds.contracts[ds.contracts.project_id == pid]
    conf = resolve_conflict(pid, ds.isr, plat.icr.get(pid), bias=plat.bias())
    by_doc = {}
    for ch in plat.index.chunks:
        if ch.project_id == pid:
            by_doc.setdefault(ch.doc_id, ch)
    lessons = [ch for ch in plat.index.chunks if ch.project_id == pid and ch.section_kind in ("lessons", "procurement")][:6]
    dt = lambda x: str(pd.Timestamp(x).date()) if pd.notna(x) else None
    return dict(
        project_id=pid, name=r.project_name, country=plat.country_names.get(r.country_code, r.country_code), sector=r.sector,
        status=r.status, approval_date=dt(r.approval_date), closing_date=dt(r.closing_date),
        commitment_usd_m=None if pd.isna(r.commitment_usd_m) else round(float(r.commitment_usd_m), 1),
        implementing_agency=getattr(r, "implementing_agency", None), es_risk=getattr(r, "es_risk", None),
        warnings=project_warnings(plat, pid, str(as_of.date())) if r.status == "Active" else None,
        isr=[dict(date=dt(x.date), dev_objective=RATING_NAME.get(int(x.dev_objective_rating)), score=int(x.dev_objective_rating))
             for x in isr.itertuples() if pd.notna(x.dev_objective_rating)],
        icr_outcome=RATING_NAME.get(plat.icr.get(pid)), conflict=conf,
        contracts=dict(n=int(len(c)), by_category=c.category.value_counts().to_dict(),
                       value_usd_m=round(float(pd.to_numeric(c.value_usd_m, errors="coerce").sum()), 1),
                       median_tender_days=int(c.tender_days.median()) if c.tender_days.notna().any() else None),
        open_tenders=overdue_tenders(plat, pid, as_of),
        documents=[dict(doc_id=d, doc_type=ch.doc_type, date=ch.date) for d, ch in sorted(by_doc.items(), key=lambda kv: str(kv[1].date))],
        lessons=[dict(chunk_id=ch.chunk_id, doc_type=ch.doc_type, section=ch.section_path, page=ch.page, text=ch.text[:600]) for ch in lessons])


# ------------------------------------------------------------------ evidence library
_STOP = set("the and for are was were with that this from into than then they them their there what which when where while who whom "
             "whose why how about after before over under between because during does did done have has had been being also only very "
             "more most such some any each other these those your our its not but can could would should will may might must shall far "
             "led lead all one two get got say says said show shows".split())
MIN_MATCH = 0.5        # share of the query's key-term weight a passage must contain to be shown


_SUFFIX = re.compile(r"(ies|es|s|ing|ed|ions|ion)$")
_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=200_000)
def stem(w: str) -> str:
    """Light suffix stemmer, the same rule the UI uses to highlight: delays/delayed -> delay, prices/price -> pric,
    contracts -> contract (but contractor stays apart), acquisitions -> acquisit."""
    if len(w) > 5:
        s = _SUFFIX.sub("", w)
        w = s if len(s) >= 4 else w                  # 'rating' stays 'rating', not 'rat'
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]                                   # bids -> bid, isrs -> isr
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


_TOKEN = re.compile(r"[a-z0-9]+(?:[-‐‑][a-z0-9]+)*")
_ok = lambda w: (len(w) >= 3 or w.isdigit()) and w not in _STOP


def token_terms(tok: str, query: bool = False) -> set:
    """Terms of one (possibly hyphenated) token. A passage's 're-tendered' counts as 'retender' AND as 'tender'. In a query, a
    compound with a short prefix ('re-tendering', 'e-procurement', 'co-financing') is ONE term, so plain 'tendering' does not
    match 're-tendering'; a compound of full words ('cost-benefit') counts as its words."""
    parts = re.split(r"[-‐‑]", tok)
    if len(parts) == 1:
        return {stem(tok)} if _ok(tok) else set()
    joined, words = stem("".join(parts)), {stem(p) for p in parts if _ok(p)}
    if query:
        return {joined} if any(len(p) < 3 and not p.isdigit() for p in parts) else words
    return words | {joined}


def key_terms(text: str, query: bool = False) -> set:
    return {t for tok in set(_TOKEN.findall(text.lower())) for t in token_terms(tok, query)}


def snippet(text: str, q_terms: set, width: int = 800) -> str:
    """Long passages: show the part around the first matching word instead of the first `width` characters (where the match
    may not be at all, so nothing would be highlighted)."""
    if len(text) <= width:
        return text
    low = text.lower()
    first = next((m.start() for m in _TOKEN.finditer(low) if token_terms(m.group(0)) & q_terms), 0)
    start = 0 if first < width - 200 else max(0, low.rfind(" ", 0, first - 200) + 1)
    end = min(len(text), start + width)
    end = end if end == len(text) else max(start + 1, text.rfind(" ", start, end))
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


_TERMS_LOCK = threading.Lock()


def _term_stats(state):
    """Key terms of every passage and their document frequency, built once per loaded corpus (warmed in the background at
    start-up by app.build_state; rebuilt when the pipeline adds passages)."""
    with _TERMS_LOCK:
        plat = state["plat"]
        cache = state.get("_terms")
        if cache is None or cache[0] is not plat or cache[1] != len(plat.index.chunks):
            from collections import Counter
            terms = {c.chunk_id: key_terms(c.text) for c in plat.index.chunks}
            df = Counter(t for ts in terms.values() for t in ts)
            postings = {}
            for cid, ts in terms.items():
                for t in ts:
                    postings.setdefault(t, []).append(cid)
            cache = state["_terms"] = (plat, len(plat.index.chunks), terms, df, postings)
        return cache[2], cache[3], cache[1]


def _postings(state):
    _term_stats(state)
    return state["_terms"][4]


SCAN_MAX_DF = 3000     # term lookup only for terms in fewer passages than this (rare terms are where ranking misses things)


def match_score(q_terms: set, passage_terms: set, df, n) -> float:
    """IDF-weighted coverage of the query's key terms: rare words ('football', 'resettlement') count far more than words in
    every report ('world', 'results', 'project'); a word the corpus never uses gets the highest weight."""
    import math
    w = {t: math.log((n + 1) / (df.get(t, 0) + 1)) for t in q_terms}
    total = sum(w.values())
    return sum(v for t, v in w.items() if t in passage_terms) / total if total else 0.0


def search(state, q: str, k: int = 10, doc_type: str = "", section: str = "", country: str = "") -> dict:
    """Library search with a relevance threshold. Retrieval scores are rank-fused (RRF) and so not comparable across queries;
    the match score is: the IDF-weighted share of the query's key terms found in the passage. Hits below MIN_MATCH are dropped,
    so an off-topic or nonsense query returns nothing instead of the 'least bad' passages."""
    plat = state["plat"]
    flt = {}
    if doc_type:
        flt["doc_type"] = {doc_type}
    if section:
        flt["section_kind"] = {section}
    if country:                                   # a hard filter: the library promises "only this country", unlike the agent's preference
        flt["country"] = {country}
    qt = key_terms(q, query=True)
    hits = plat.index.search(q, k=k * 4, filters=flt) if qt else []          # over-fetch, then keep only hits that pass
    terms, df, n = _term_stats(state)
    # recall: the engines tokenise 're-tendering' as 're' + 'tendering' and may rank an exact passage below the cut-off, so
    # passages that contain the query's RARE key terms are checked too (after the ranked hits, best match first)
    post = _postings(state)
    ranked = {c.chunk_id for c, _, _ in hits}
    extra = {cid for t in qt if df.get(t, 0) <= SCAN_MAX_DF for cid in post.get(t, ())} - ranked
    by_id = plat.chunks_by_id
    extra = [by_id[cid] for cid in extra if cid in by_id and all(getattr(by_id[cid], f) in v for f, v in flt.items())]
    extra = sorted(((c, 0.0, None) for c in extra), key=lambda h: -match_score(qt, terms[h[0].chunk_id], df, n))
    results, seen = [], set()
    for c, s, _ in [*hits, *extra]:
        match = match_score(qt, terms.get(c.chunk_id) or key_terms(c.text), df, n)
        dup = (c.doc_id, " ".join(c.text.split()).lower())          # reports repeat some text word for word: show it once
        if match >= MIN_MATCH and dup not in seen:
            seen.add(dup)
            results.append(dict(chunk_id=c.chunk_id, doc_id=c.doc_id, doc_type=c.doc_type, project_id=c.project_id, country=c.country,
                                sector=c.sector, section=c.section_path, section_kind=c.section_kind, page=c.page, text=snippet(c.text, qt),
                                score=round(float(s), 2), match=round(match, 2), pdf_page_url=doc_links.page_link(c.doc_id, c.page)))
            if len(results) >= k:
                break
    return dict(backend=getattr(plat.index, "last_backend", "in-memory"), threshold=MIN_MATCH, considered=len(hits) + len(extra),
                results=results)


def passage(state, chunk_id: str) -> dict:
    plat = state["plat"]
    ch = plat.chunks_by_id.get(chunk_id)
    if ch is None:
        raise KeyError(f"passage {chunk_id} not found")
    same = [c for c in plat.index.chunks if c.doc_id == ch.doc_id]
    i = next(k for k, c in enumerate(same) if c.chunk_id == chunk_id)
    ctx = lambda c: dict(chunk_id=c.chunk_id, section=c.section_path, page=c.page, text=c.text)
    return dict(chunk_id=ch.chunk_id, doc_id=ch.doc_id, doc_type=ch.doc_type, project_id=ch.project_id, country=ch.country,
                sector=ch.sector, section=ch.section_path, page=ch.page, date=ch.date, text=ch.text,
                pdf_url=doc_links.pdf_url(ch.doc_id), pdf_page_url=doc_links.page_link(ch.doc_id, ch.page),
                before=ctx(same[i - 1]) if i > 0 else None, after=ctx(same[i + 1]) if i + 1 < len(same) else None)


def document(state, doc_id: str) -> dict:
    """A whole report as extracted, in reading order, grouped by PDF page: the viewer scrolls to one passage inside it."""
    plat = state["plat"]
    parts = [c for c in plat.index.chunks if c.doc_id == doc_id]
    if not parts:
        raise KeyError(f"document {doc_id} not found")
    first = parts[0]
    pages = []
    for c in parts:
        if not pages or pages[-1]["page"] != c.page:
            pages.append(dict(page=c.page, pdf_page_url=doc_links.page_link(doc_id, c.page), passages=[]))
        pages[-1]["passages"].append(dict(chunk_id=c.chunk_id, section=c.section_path, kind=c.chunk_type, text=c.text))
    return dict(doc_id=doc_id, doc_type=first.doc_type, project_id=first.project_id, country=first.country, sector=first.sector,
                date=first.date, pdf_url=doc_links.pdf_url(doc_id), pages=pages, passages=len(parts))


# ------------------------------------------------------------------ insights
def _benchmark():
    for f in ("results_databricks.json", "results.json"):
        p = os.path.join(HERE, "evals", f)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            return dict(source=d.get("source") or f, summary=d.get("summary"),
                        failures=[dict(id=r["id"], why=r["why"]) for r in d.get("items", []) if not r.get("passed")])
    return None


def insights(state) -> dict:
    from .events import load_events, usage_kpis
    plat = state["plat"]; ds = plat.ds; m = plat.models.metrics
    docs = pd.DataFrame([dict(doc_id=d["doc_id"], doc_type=d["doc_type"], project_id=d["project_id"],
                              date=(d.get("meta") or {}).get("Date", ""), ingested=d.get("ingested")) for d in ds.docs])
    order = "ingested" if "ingested" in docs and docs.ingested.notna().any() else "date"
    recent = docs.sort_values(order, ascending=False).head(8) if len(docs) else docs
    lesson_counts = pd.Series([c.doc_id for c in plat.index.chunks if c.section_kind == "lessons"]).value_counts()
    events = load_events(state["events"].local_dir) if state.get("events") else []
    return dict(
        model=dict(name="XGBoost survival (AFT) + calibrated re-bid / amendment classifiers", version=int(m.get("model_version", 1)),
                   mae_days=round(m.get("mae_days", np.nan), 1), baseline_mae_days=round(m.get("mae_baseline_category_median", np.nan), 1),
                   baseline_method_mae_days=round(m.get("mae_baseline_category_method_median", np.nan), 1),
                   c_index=round(m["c_index"], 3) if "c_index" in m else None,
                   c_index_baseline_method=round(m["c_index_baseline_category_method"], 3) if "c_index_baseline_category_method" in m else None,
                   auc_signed_6m=round(m["auc_signed_6m"], 3) if "auc_signed_6m" in m else None,
                   auc_signed_6m_baseline_method=round(m["auc_signed_6m_baseline_category_method"], 3) if "auc_signed_6m_baseline_category_method" in m else None,
                   interval_coverage=round(m.get("interval_coverage_p10_p90", np.nan), 3),
                   auc_rebid=round(m["auc_rebid"], 3) if "auc_rebid" in m else None,      # None: too few real re-tenders to measure
                   rebid_test_positives=m.get("positives_test_rebid"),
                   trained_on=dict(signed=int(m.get("n_final_exact", 0)), still_open=int(m.get("n_final_censored", 0))),
                   feature_set=getattr(plat.models, "dur_set", None)),
        benchmark=_benchmark(), usage=usage_kpis(events),
        whats_new=[dict(doc_id=r.doc_id, doc_type=r.doc_type, project_id=r.project_id, date=r.date,
                        ingested=str(r.ingested)[:19] if order == "ingested" else None, lessons=int(lesson_counts.get(r.doc_id, 0)))
                   for r in recent.itertuples()],
        data=dict(contracts=len(ds.contracts), open_tenders=len(getattr(ds, "open_tenders", [])), projects=len(ds.projects),
                  documents=len(ds.docs), passages=len(plat.index.chunks), countries=len(set(ds.contracts.country_code)),
                  quality=ds.quality, graph=plat.graph.stats()),
        retrieval=state.get("retrieval"), isr_optimism=plat.bias(),
        conflicts=sum(1 for p, r in plat.icr.items() if resolve_conflict(p, ds.isr, r)), as_of=str(as_of_date(plat).date()))
