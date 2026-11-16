"""Module 2b: agentic retrieval for free-form questions, all on Databricks.

Flow (one generator, Agent.stream; Agent.ask collects it):
  1 policy screen                      refusals before any model call
  2 route (planner.py)                 the tools this kind of question NEEDS - run in parallel on the server, so they always run
  3 evidence pack (planner.py)         compact, deduplicated, budgeted context instead of raw tool dumps
  4 model on Databricks Model Serving  streams its answer; it may ask for ONE extra round of tools (also run in parallel)
  5 StreamVerifier (guardrails.py)     each sentence is shown only after citation, ids, numbers, country scope and the judge pass
Context engineering cut the planning question from 6 model calls and 4 sequential tool rounds to 1-2 model calls.

Refusals: a deterministic policy screen runs before any model call; the system prompt adds the same policy for the model.
If the model endpoint is unavailable the agent degrades to the fixed path: the most relevant passages, no generated text."""
from __future__ import annotations
import json, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .guardrails import LLMJudge, RATING_NAME, StreamVerifier, resolve_conflict
from .llm import AGENT_MODEL, DatabricksChat, LLMUnavailable, chat_stream, text_of
from .planner import Plan, evidence_pack, route
from .monitoring import plain_duration
from .rag import Copilot, PackageSpec

MAX_EXTRA_ROUNDS = 1           # required tools run up front; the model may add one round of its own
ANSWER_TOKENS = 1500           # gpt-oss counts its hidden reasoning against this; length is set in the prompt
CATEGORIES = ["Civil Works", "Goods", "Consultancies", "Non-Consulting Services"]

# ------------------------------------------------------------------ refusal policy (screened before any model call)
POLICY = [
    ("supplier_scoring", r"\b(rank|rate|score|blacklist|debar|trust(?:worthy)?|reliable|best|worst)\b.{0,40}\b(supplier|contractor|bidder|firm|vendor|consultant)s?\b"
                         r"|\b(supplier|contractor|bidder|firm|vendor)s?\b.{0,40}\b(blacklist|debar|exclude|should (?:we|i) (?:avoid|choose|pick)|is the best|is the worst|to avoid)\b"),
    ("open_tender_outcome", r"\b(who|which (?:firm|bidder|supplier|company))\b.{0,40}\b(will|going to|likely to)\b.{0,15}\b(win|be awarded|get the contract)\b"),
    ("bid_manipulation", r"\b(rig|collude|collusion|split (?:the )?(?:contract|package)s? to avoid|avoid (?:prior )?review|bypass|kickback|bribe)\b"
                         r"|\b(discredit|dig up dirt|smear)\b|\bdisqualify (?:them|him|her|the (?:lowest|cheapest|winning|best) (?:bid(?:der)?|firm|offer))\b"
                         r"|\bfind (?:something|anything|a reason)\b.{0,40}\b(against|to (?:disqualify|reject|exclude))\b"),
    ("personal_data", r"\b(email|phone|mobile|home address|personal (?:details|contact))\b.{0,30}\b(of|for)\b"),
    ("legal_advice", r"\b(is it legal|legally binding|sue|lawsuit|legal advice|liable)\b"),
]
REFUSAL_TEXT = {
    "supplier_scoring": "I don't score, rank or blacklist named suppliers: that is a due-diligence and debarment decision for accredited staff, and public records are not enough to make it fairly.",
    "open_tender_outcome": "I don't predict who will win a tender: that would prejudge an evaluation that is still open.",
    "bid_manipulation": "I can't help with splitting packages, avoiding review or anything that undermines procurement integrity.",
    "personal_data": "I don't provide personal contact details; procurement notices are stripped of them at ingestion.",
    "legal_advice": "I can't give legal advice; please consult the project's legal counsel.",
    "out_of_scope": "That is outside what I can answer from World Bank procurement records and project documents.",
}


_FOLLOW_UP_ACTION = re.compile(r"\b(them|those|these|that one|which of|avoid|exclude|blacklist|rank|pick|choose|win|award)\b", re.I)


def policy_screen(question: str, previous: str | None = None):
    """Screen the question; a short follow-up ("should we exclude them?") is also screened together with the previous question,
    so a request split across two turns cannot slip past."""
    q = question.lower()
    hit = next((name for name, pat in POLICY if re.search(pat, q)), None)
    if hit or not previous or not _FOLLOW_UP_ACTION.search(q):
        return hit
    p = previous.lower()
    # "which contractors underperformed?" -> "should we exclude them?": the target is in one turn, the verb in the other
    if re.search(r"\b(supplier|contractor|bidder|firm|vendor|consultant|compan(?:y|ies))s?\b", p) and \
            re.search(r"\b(avoid|exclude|blacklist|debar|rank|pick|choose|trust|reliable|best|worst)\b", q):
        return "supplier_scoring"
    return next((name for name, pat in POLICY if re.search(pat, f"{p} {q}")), None)


MAX_TURNS = 4                       # conversation memory: the last 4 verified turns


def history_messages(history):
    """Earlier turns as chat messages: the question, and ONLY the verified answer (never drafts or removed sentences), with a
    short note of each cited source so follow-ups like "tell me more about the second lesson" can be resolved."""
    msgs = []
    for t in (history or [])[-MAX_TURNS:]:
        msgs.append({"role": "user", "content": t["question"]})
        if t.get("refused"):
            msgs.append({"role": "assistant", "content": "REFUSE: " + " ".join(a["text"] for a in t.get("answer", []))})
            continue
        lines = [f"{a['text']} [{', '.join(a['ids'])}]" if a.get("ids") else a["text"] for a in t.get("answer", [])]
        notes = [f"[{i}] {s.get('label', '')} | {s.get('country', '')} | {s['text'][:240]}" for i, s in (t.get("carry") or {}).items()]
        msgs.append({"role": "assistant", "content": "\n".join(lines) or "(no verified answer)"})
        if notes:
            msgs.append({"role": "user", "content": "(Context, not a question: sources cited above, still citable by id)\n" + "\n".join(notes)})
            msgs.append({"role": "assistant", "content": "Noted."})
    return msgs


SYSTEM = """You are Tenderwise, a procurement assistant for World Bank-financed projects. Answer ONLY from the EVIDENCE given
and any tool results. The required tools have already been run for you; call a tool only if something essential is missing.
Rules: at most ~150 words, short sentences or bullets, no tables. End EVERY factual sentence (including "none were found")
with its source ids in square brackets exactly as given, e.g. [ICR_P160162_34360386#0141] [forecast:KE-Civil Works].
Copy numbers exactly (no rounding, no conversions, no arithmetic). Name a country only if the source's country says so.
If the evidence is thin, say so. Advice ("what should the team do") must come from the lessons in the EVIDENCE: say what a
past project learned or did and cite it; never add generic advice of your own. When listing projects, give one bullet per
project that has warning signs, starting with its name and id, then why in plain words. Refuse (reply "REFUSE: <why>") requests to rank or blacklist suppliers, predict tender
winners, avoid procurement rules, give personal contact details or legal advice, or anything unrelated to procurement."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Hybrid search over passages of World Bank project documents: ICR (completion reports, incl. lessons), PAD "
                       "(appraisal documents), ISR (status reports), PPAR (IEG evaluations). Returns passages with ids to cite.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "specific words to look for"},
            "doc_types": {"type": "array", "items": {"type": "string", "enum": ["ICR", "PAD", "ISR", "PPAR"]}},
            "section": {"type": "string", "enum": ["lessons", "procurement", "ratings", "context", "any"],
                        "description": "lessons = lessons learned; procurement = procurement / fiduciary sections"},
            "country": {"type": "string", "description": "optional country name to prefer (soft)"},
            "project_id": {"type": "string", "description": "optional: only passages from this project"},
            "k": {"type": "integer", "minimum": 1, "maximum": 6}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "forecast_package",
        "description": "Survival-model forecast of tender-to-award days (P10/P50/P90), re-bid / amendment probability, the main "
                       "drivers, and statistics of comparable past contracts, for a planned procurement package.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "country_code": {"type": "string", "description": "ISO2, e.g. KE"},
            "sector": {"type": "string"}, "procurement_method": {"type": "string"},
            "value_usd_m": {"type": "number"}, "review_type": {"type": "string", "enum": ["Prior", "Post"]}},
            "required": ["category", "country_code"]}}},
    {"type": "function", "function": {
        "name": "get_project",
        "description": "A project's record: name, country, sector, status, dates, commitment, contracts count, last ISR rating, "
                       "ICR outcome rating, and any ISR-vs-ICR rating conflict with the rule that resolves it.",
        "parameters": {"type": "object", "properties": {"project_id": {"type": "string", "description": "e.g. P160162"}},
                       "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "early_warnings",
        "description": "Active projects with early-warning flags (falling ISR ratings, bias-adjusted ratings, overdue tenders).",
        "parameters": {"type": "object", "properties": {"country_code": {"type": "string"}, "project_id": {"type": "string"},
                                                        "top": {"type": "integer", "minimum": 1, "maximum": 8}}}}},
    {"type": "function", "function": {
        "name": "list_values",
        "description": "Valid values for categories, sectors, procurement methods and country codes in the data.",
        "parameters": {"type": "object", "properties": {}}}},
]


def risk_label(score) -> str:
    """Same levels as the Portfolio monitor (web/src/components.jsx riskLevel)."""
    return "High risk" if score >= 0.6 else "Watch" if score >= 0.3 else "Minor" if score > 0 else "No warning signs"


def _warning_display(w) -> str:
    head = f"{w.get('name') or w['project_id']} ({w['project_id']}), {w.get('country', '')}: {w['risk']} as of {w['as_of']}"
    lines = [f"- {x['title']}. {x['detail']}" for x in w.get("alerts", [])] or ["- No warning signs."]
    tend = f"{w.get('open_tenders', 0)} open tender(s), {w.get('overdue_tenders', 0)} running late."
    return "\n".join([head, *lines, tend])


def _project_display(r) -> str:
    out = [f"{r['name']} ({r['project_id']}), {r['country']}, {r['sector']}, {r['status']}",
           f"Approved {r.get('approval_date') or '-'}, closes {r.get('closing_date') or '-'}"
           + (f"; commitment USD {r['commitment_usd_m']}m" if r.get("commitment_usd_m") is not None else "") + f"; {r['n_contracts']} contracts"]
    if r.get("last_isr_rating"):
        out.append(f"Latest progress rating: {r['last_isr_rating']} ({r['last_isr_date']})")
    if r.get("icr_outcome_rating"):
        out.append(f"Final outcome rating (completion report): {r['icr_outcome_rating']}")
    if r.get("rating_conflict"):
        out.append("The status reports and the completion report disagree; the completion report's rating is used.")
    return "\n".join(out)


def _closest(value, known):
    """Map free text ('Transport', 'QCBS') onto a value the data uses ('Transportation', 'Quality And Cost-Based Selection')."""
    from difflib import SequenceMatcher
    v = str(value).lower().strip()
    known = [str(k) for k in known]
    hit = [k for k in known if k.lower() == v] or [k for k in known if k.lower().startswith(v) or v in k.lower()]
    if hit:
        return min(hit, key=len)
    acr = {"".join(w[0] for w in k.replace("-", " ").split() if w[0].isalpha() and w.lower() not in ("and", "of", "for", "the")).lower(): k
           for k in known}
    if v.replace(" ", "") in acr:                                   # QCBS -> Quality And Cost-Based Selection
        return acr[v.replace(" ", "")]
    best = max(known, key=lambda k: SequenceMatcher(None, v, k.lower()).ratio(), default=None)
    return best if best and SequenceMatcher(None, v, best.lower()).ratio() >= .6 else None


class Agent:
    def __init__(self, plat, client=None, model=None, judge=None, use_judge=True):
        self.p, self.cop = plat, Copilot(plat)
        from .llm import shared_client
        self.client = client or shared_client()
        self.model = model or AGENT_MODEL
        self.judge = judge if judge is not None else (LLMJudge(self.client) if use_judge else None)

    # ---------------------------------------------------------------- tools (each registers what it returns as citable sources)
    def _country_name(self, code):
        return self.p.country_names.get(code, code)

    def _t_search_documents(self, src, query, doc_types=None, section="any", country=None, k=5, project_id=None):
        if isinstance(doc_types, str):                      # models sometimes send "ICR" instead of ["ICR"]
            doc_types = [d.strip() for d in doc_types.split(",")]
        flt = {"doc_type": set(doc_types or ["ICR", "PAD", "ISR", "PPAR"])}
        if section and section != "any":
            flt["section_kind"] = {section}
        if project_id:
            flt["project_id"] = {str(project_id).upper()}
        hits = self.p.index.search(str(query)[:300], k=int(min(max(k or 5, 1), 6)), filters=flt, prefer={"country": country} if country else {})
        out = []
        for c, s, comp in hits:
            src[c.chunk_id] = dict(text=c.text, country=c.country, label=f"{c.doc_type} {c.project_id}, p.{c.page}, {c.section_path}",
                                   doc_id=c.doc_id, page=c.page, kind="passage")
            out.append(dict(id=c.chunk_id, doc_type=c.doc_type, project_id=c.project_id, country=c.country, sector=c.sector,
                            section=c.section_path, page=c.page, text=c.text[:700]))
        return out or {"result": "no passages matched these filters"}

    def _t_forecast_package(self, src, category, country_code, sector=None, procurement_method=None, value_usd_m=None, review_type="Prior"):
        c = self.p.ds.contracts
        if category not in CATEGORIES:
            return {"error": f"category must be one of {CATEGORIES}"}
        cc = str(country_code).upper()
        sector = _closest(sector, c.sector.dropna().unique()) if sector else None
        method = _closest(procurement_method, c.procurement_method.dropna().unique()) if procurement_method else None
        sector = sector or (c[c.country_code == cc].sector.mode().iloc[0] if (c.country_code == cc).any() else c.sector.mode().iloc[0])
        method = method or c[c.category == category].procurement_method.mode().iloc[0]
        spec = PackageSpec("", category, sector, cc, float(value_usd_m) if value_usd_m else None, method, review_type or "Prior",
                           str(pd.Timestamp.today().date()))
        row = self.cop.feature_row(spec)
        pred = self.p.models.predict_df(row).iloc[0]
        an = self.cop.analogues(row)
        fid, cid = f"forecast:{cc}-{category}", f"comparables:{cc}-{category}"
        fc = dict(p10_days=round(float(pred.p10_days)), p50_days=round(float(pred.p50_days)), p90_days=round(float(pred.p90_days)),
                  rebid_probability_pct=round(100 * float(pred.p_rebid), 1), amendment_probability_pct=round(100 * float(pred.p_amend), 1),
                  model="XGBoost survival (AFT) + calibrated classifiers", sector_used=sector, method_used=method,
                  drivers=[f"{e['feature']}={e['value']}" for e in self.p.models.explain(row, top=3)])
        cp = dict(n_comparable=len(an), median_days=round(float(an.tender_days.median())) if len(an) else None,
                  p80_days=round(float(an.tender_days.quantile(.8))) if len(an) else None,
                  rebid_pct=round(100 * float(an.rebid.mean()), 1) if len(an) else None)
        where = f"{category.lower()} in {self._country_name(cc)} ({method}, {sector})"
        src[fid] = dict(text=json.dumps(fc), country=self._country_name(cc), label="model forecast (champion)", kind="model",
                        display=f"Forecast for {where}: usually {plain_duration(fc['p50_days'])} from tender to signed contract; "
                                f"fast case {plain_duration(fc['p10_days'])}; 9 in 10 within {plain_duration(fc['p90_days'])}.\n"
                                f"Chance of re-tendering {fc['rebid_probability_pct']}%; chance of a contract amendment {fc['amendment_probability_pct']}%.")
        src[cid] = dict(text=json.dumps(cp), country=self._country_name(cc), label=f"{len(an)} comparable past contracts", kind="data",
                        display=(f"{len(an)} similar signed contracts: typically {plain_duration(cp['median_days'])}; 8 in 10 within "
                                 f"{plain_duration(cp['p80_days'])}; {cp['rebid_pct']}% were re-tendered.") if len(an) else "No similar signed contracts found.")
        for k in (fid, cid):          # the readable summary is part of the source, so plain wording ("9 in 10 within 21 months") verifies too
            src[k]["text"] += "\n" + src[k]["display"]
        return {"forecast": {"id": fid, **fc}, "comparables": {"id": cid, **cp}}

    def _t_get_project(self, src, project_id):
        pid = str(project_id).upper().strip()
        pr = self.p.ds.projects[self.p.ds.projects.project_id == pid]
        if not len(pr):
            return {"error": f"project {pid} is not in the data"}
        r = pr.iloc[0]
        isr = self.p.ds.isr[self.p.ds.isr.project_id == pid].sort_values("date")
        conf = resolve_conflict(pid, self.p.ds.isr, self.p.icr.get(pid), bias=self.p.bias())
        rec = dict(project_id=pid, name=r.project_name, country=self._country_name(r.country_code), sector=r.sector, status=r.status,
                   approval_date=str(pd.Timestamp(r.approval_date).date()) if pd.notna(r.approval_date) else None,
                   closing_date=str(pd.Timestamp(r.closing_date).date()) if pd.notna(r.closing_date) else None,
                   commitment_usd_m=None if pd.isna(r.commitment_usd_m) else round(float(r.commitment_usd_m), 1),
                   n_contracts=int((self.p.ds.contracts.project_id == pid).sum()),
                   last_isr_rating=RATING_NAME.get(int(isr.dev_objective_rating.iloc[-1])) if len(isr) else None,
                   last_isr_date=str(isr.date.iloc[-1].date()) if len(isr) else None,
                   icr_outcome_rating=RATING_NAME.get(self.p.icr.get(pid)),
                   rating_conflict=None if not conf else dict(rule=conf["rule_applied"], adopted=conf["adopted_source"], note=conf["note"]))
        sid = f"project:{pid}"
        disp = _project_display(rec)
        src[sid] = dict(text=json.dumps(rec) + "\n" + disp, country=rec["country"], label=f"project record {pid}", kind="data", display=disp)
        return {"id": sid, **rec}

    def _t_early_warnings(self, src, country_code=None, top=5, project_id=None):
        from .monitoring import as_of_date, project_warnings
        pr = self.p.ds.projects[self.p.ds.projects.status == "Active"]
        if project_id:                                     # one project: why is it flagged (or not)?
            pr = pr[pr.project_id == str(project_id).upper().strip()]
            if not len(pr):
                return {"result": f"project {project_id} is not an active project, so it has no early warnings"}
        if country_code:
            pr = pr[pr.country_code == str(country_code).upper()]
        as_of = str(as_of_date(self.p).date())             # the same 'today' as the Portfolio monitor
        allw = sorted((project_warnings(self.p, pid, as_of) for pid in pr.project_id), key=lambda r: -r["risk_score"])
        names = pr.set_index("project_id").project_name.to_dict(); ccs = pr.set_index("project_id").country_code.to_dict()
        flagged = [w for w in allw if w["flags"]]
        # one project is always returned (why it is or is not flagged); a list names only projects that have warning signs
        rows = allw if project_id else flagged[:int(top or 5)]
        out = []
        where = self._country_name(str(country_code).upper()) if country_code else "the portfolio"
        if not project_id:
            sid = f"warnings:{str(country_code).upper() if country_code else 'all'}"
            summ = dict(summary=True, checked=len(allw), flagged=len(flagged), as_of=as_of, where=where)
            disp = (f"{len(allw)} active projects in {where} checked on {as_of}: {len(flagged)} with warning signs"
                    + (f", the {len(rows)} highest-risk listed." if len(flagged) > len(rows) else "."))
            src[sid] = dict(text=json.dumps(summ) + " " + disp, country=where if country_code else None,
                            label=f"warning-sign check, {where}", kind="data", display=disp)
            out.append({"id": sid, **summ})
        for w in rows:
            sid = f"warning:{w['project_id']}"
            c = self._country_name(ccs.get(w["project_id"], ""))
            rec = dict(w, name=names.get(w["project_id"], ""), country=c, risk=risk_label(w["risk_score"]))
            disp = _warning_display(rec)
            src[sid] = dict(text=json.dumps(rec) + "\n" + disp, country=c, label=f"warning signs: {rec['name'] or w['project_id']}",
                            kind="data", display=disp)
            out.append({"id": sid, **rec})
        return out or {"result": "no active projects match"}

    def _derived_calls(self, results) -> list:
        """A flagged project needs evidence on what to DO about its flags: lessons from past projects on exactly those problems
        (late tenders of the flagged categories, falling progress ratings). Chosen from the warning text, so it always runs."""
        calls = []
        for name, _, res, _, _ in results:
            if name != "early_warnings" or not isinstance(res, list):
                continue
            for w in res[:2]:
                flags = " ".join(w.get("flags") or [])
                if not flags:
                    continue
                pr = self.p.ds.projects.set_index("project_id")
                country = self._country_name(pr.country_code.get(w["project_id"], "")) if w["project_id"] in pr.index else None
                cats = sorted(set(re.findall(r"tender \S+ \(([^)]+)\)", flags)))
                if cats:
                    calls.append(("search_documents", dict(query=f"delays in {' and '.join(c.lower() for c in cats)} procurement, slow selection "
                                                                 f"and contract award: causes and how projects fixed them",
                                                           section="lessons", doc_types=["ICR", "PPAR", "PAD"], country=country, k=4)))
                if "ISR" in flags:
                    calls.append(("search_documents", dict(query="falling implementation progress ratings: causes and remedies, restructuring, "
                                                                 "implementation support", section="lessons", doc_types=["ICR", "PPAR"],
                                                           country=country, k=3)))
        return calls[:3]

    def _t_list_values(self, src):
        c = self.p.ds.contracts
        return dict(categories=CATEGORIES, sectors=sorted(c.sector.dropna().unique().tolist()),
                    procurement_methods=c.procurement_method.value_counts().head(12).index.tolist(),
                    countries={k: self._country_name(k) for k in sorted(c.country_code.dropna().unique())})

    def _run_tool(self, name, args, src):
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(src, **args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    # ---------------------------------------------------------------- the loop
    def _parallel(self, calls, src):
        """Run tool calls concurrently; returns [(name, args, result, seconds, backend)] in call order."""
        def one(c):
            name, args = c
            t = time.time()
            res = self._run_tool(name, dict(args), src)
            return name, args, res, round(time.time() - t, 2), getattr(self.p.index, "last_backend", "") if name == "search_documents" else ""
        if not calls:
            return []
        with ThreadPoolExecutor(max_workers=min(6, len(calls))) as ex:
            return list(ex.map(one, calls))

    @staticmethod
    def _summary(name, res):
        if isinstance(res, list):
            return f"{len(res)} result(s)"
        if isinstance(res, dict) and res.get("error"):
            return "error: " + str(res["error"])[:80]
        if isinstance(res, dict) and res.get("result"):
            return str(res["result"])[:80]
        return "done"

    def stream(self, question: str, history: list | None = None, context: dict | None = None):
        """Yield events: status, tool, sentence (verified only), removed, done (the full result, as ask() returns)."""
        t0, trace = time.time(), []
        history = (history or [])[-MAX_TURNS:]
        src = {i: s for t in history for i, s in (t.get("carry") or {}).items()}          # earlier citations stay citable
        q = str(question).strip()[:1000]
        pol = policy_screen(q, history[-1]["question"] if history else None)
        if pol:
            yield {"type": "done", "result": dict(question=q, refused=True, refusal_reason=pol, answer=[dict(text=REFUSAL_TEXT[pol], ids=[], meta=True)],
                   removed=[], sources={}, trace=["policy screen: refused before any model call"], mode="policy", seconds=round(time.time() - t0, 2))}
            return
        prev = history[-1].get("plan") if history else None
        plan = route(q, self.p.country_names, sorted(set(self.p.ds.contracts.sector.dropna())),
                     Plan(**prev) if isinstance(prev, dict) else None, context)
        trace.append(f"plan: {', '.join(plan.intents)} -> {len(plan.calls)} required tool call(s) run in parallel" + (f" ({plan.note})" if plan.note else ""))
        if history:
            trace.append(f"conversation: {len(history)} earlier turn(s) in context, {len(src)} earlier source(s) citable")
        yield {"type": "status", "stage": "tools", "text": f"Gathering evidence: {len(plan.calls)} tool call(s) in parallel"}
        results = self._parallel(plan.calls, src)
        for name, args, res, secs, via in results:
            trace.append(f"round 1: {name}({json.dumps({k: v for k, v in args.items() if k != 'doc_types'}, ensure_ascii=False)[:150]})"
                         + (f" via {via}" if via else "") + f" [{secs}s]")
            yield {"type": "tool", "name": name, "args": {k: v for k, v in args.items() if k != "doc_types"}, "summary": self._summary(name, res), "seconds": secs}
        derived = self._derived_calls(results)
        if derived:
            more = self._parallel(derived, src)
            for name, args, res, secs, via in more:
                trace.append(f"round 1b (from the warnings found): {name}({json.dumps({k: v for k, v in args.items() if k != 'doc_types'}, ensure_ascii=False)[:150]}) [{secs}s]")
                yield {"type": "tool", "name": name, "args": {k: v for k, v in args.items() if k != "doc_types"}, "summary": self._summary(name, res), "seconds": secs}
            results = results + more
        pack = evidence_pack([(n, a, r) for n, a, r, _, _ in results], src, q)
        user = f"{q}\n\nEVIDENCE (already retrieved for this question; cite ids exactly):\n{pack}"
        msgs = [{"role": "system", "content": SYSTEM}, *history_messages(history), {"role": "user", "content": user}]
        asked = " ".join([t["question"] for t in history] + [q] + [json.dumps(context or {})])     # numbers the user gave
        verifier = StreamVerifier(src, sorted(set(self.p.country_names.values())), re.findall(r"\d+(?:\.\d+)?", asked), self.judge,
                                  batch=1)                         # one sentence per judge call, run in parallel: earliest release
        draft, released, refused = "", 0, None
        yield {"type": "status", "stage": "drafting", "text": "Writing the answer; each sentence is shown once it is verified"}
        try:
            for rnd in range(MAX_EXTRA_ROUNDS + 1):
                tools = TOOLS if rnd < MAX_EXTRA_ROUNDS else None
                head, end = "", None
                for kind, payload in self._call(msgs, tools):
                    if kind == "end":
                        end = payload; break
                    if refused is None and len((head + payload).strip()) < 8:          # hold the first characters: a refusal?
                        head += payload; continue
                    if refused is None:
                        payload, head = head + payload, ""
                        refused = payload.lstrip().upper().startswith("REFUSE")
                    draft += payload
                    if refused:
                        continue
                    for ev_kind, rec in verifier.feed(payload):
                        released += ev_kind == "sentence"
                        yield self._event(ev_kind, rec, src)
                if head:                                                                  # a very short answer
                    draft += head; refused = head.lstrip().upper().startswith("REFUSE")
                    if not refused:
                        for ev_kind, rec in verifier.feed(head):
                            yield self._event(ev_kind, rec, src)
                calls = (end or {}).get("tool_calls") or []
                if calls and not draft.strip() and rnd < MAX_EXTRA_ROUNDS:
                    extra = []
                    for c in calls:
                        try:
                            extra.append((c["function"]["name"], json.loads(c["function"].get("arguments") or "{}")))
                        except ValueError:
                            extra.append((c["function"]["name"], {}))
                    yield {"type": "status", "stage": "tools", "text": f"The model asked for {len(extra)} more tool call(s)"}
                    more = self._parallel(extra, src)
                    msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
                    for c, (name, args, res, secs, via) in zip(calls, more):
                        trace.append(f"round 2: {name}({json.dumps(args, ensure_ascii=False)[:150]})" + (f" via {via}" if via else "") + f" [{secs}s]")
                        yield {"type": "tool", "name": name, "args": args, "summary": self._summary(name, res), "seconds": secs}
                        msgs.append({"role": "tool", "tool_call_id": c.get("id") or name,
                                     "content": evidence_pack([(name, args, res)], src, q, budget=2500)})
                    continue
                break
            if not draft.strip():                                                         # some models return only reasoning
                msgs.append({"role": "user", "content": "Your answer was empty. Answer now from the evidence, citing ids."})
                for kind, payload in self._call(msgs, None):
                    if kind == "text":
                        draft += payload
                        for ev_kind, rec in verifier.feed(payload):
                            yield self._event(ev_kind, rec, src)
        except LLMUnavailable as e:
            yield {"type": "done", "result": self._fallback(q, src, trace + [f"model unavailable: {e}"], t0, plan)}
            return
        if refused:
            reason = draft.split(":", 1)[-1].strip() or REFUSAL_TEXT["out_of_scope"]
            yield {"type": "done", "result": dict(question=q, refused=True, refusal_reason="model", answer=[dict(text=reason, ids=[], meta=True)],
                   removed=[], sources={}, trace=trace + ["model refused"], mode="agent", seconds=round(time.time() - t0, 2),
                   plan=plan.__dict__)}
            return
        for ev_kind, rec in verifier.finish_iter():                     # each sentence as soon as ITS judge call returns
            yield self._event(ev_kind, rec, src)
        kept, removed = verifier.results()
        factual = [k for k in kept if not k.get("meta")]
        cited = {i for k in kept for i in k["ids"]}
        yield {"type": "done", "result": dict(
            question=q, refused=False, answer=kept, removed=removed, held_back=not factual,
            sources={i: {k: v for k, v in src[i].items() if k != "text"} | {"text": src[i]["text"][:700]} for i in cited if i in src},
            trace=trace + [f"verifier: kept {len(factual)} sentence(s), removed {len(removed)}"], mode="agent", model=self.model,
            seconds=round(time.time() - t0, 2), draft=draft, carry={i: src[i] for i in cited if i in src}, plan=plan.__dict__,
            tools_run=[n for n, *_ in results])}

    def _call(self, msgs, tools):
        """Stream from Databricks Model Serving; a client without streaming (tests, other backends) gives one chunk."""
        if isinstance(self.client, DatabricksChat):
            yield from chat_stream(self.client, self.model, msgs, tools=tools, max_tokens=ANSWER_TOKENS,
                                   reasoning_effort="low" if "gpt-oss" in self.model else None)
            return
        msg = self.client.chat(self.model, msgs, tools=tools, max_tokens=ANSWER_TOKENS)
        content = text_of(msg) or ""
        if content:
            yield ("text", content)
        yield ("end", {"content": content, "tool_calls": msg.get("tool_calls") or []})

    @staticmethod
    def _event(kind, rec, src):
        if kind == "sentence":
            return {"type": "sentence", "text": rec["text"], "ids": rec["ids"], "meta": bool(rec.get("meta")),
                    "sources": {i: {k: v for k, v in src[i].items() if k != "text"} | {"text": src[i]["text"][:700]} for i in rec["ids"] if i in src}}
        return {"type": "removed", "text": rec["text"], "ids": rec["ids"], "reasons": rec["reasons"]}

    def ask(self, question: str, history: list | None = None, context: dict | None = None) -> dict:
        """Non-streaming: the same pipeline as stream(), collected."""
        result = None
        for ev in self.stream(question, history, context):
            if ev["type"] == "done":
                result = ev["result"]
        return result

    def _fallback(self, q, src, trace, t0, plan=None):
        hits = self._t_search_documents(src, q, section="any", k=5)
        return dict(question=q, refused=False, answer=[], removed=[], held_back=True, passages=hits if isinstance(hits, list) else [],
                    sources={}, trace=trace + ["fallback: showing the most relevant passages, no generated text"],
                    mode="fallback", seconds=round(time.time() - t0, 2), plan=plan.__dict__ if plan else None)
