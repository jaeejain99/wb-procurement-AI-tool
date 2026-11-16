"""Context engineering for the agent: decide the REQUIRED tools up front and give the model a compact evidence pack.

route()          question (+ package context, + the previous turn's plan for follow-ups) -> the tool calls every answer of
                 this kind needs. They run in parallel on the server BEFORE the first model call, so they always run: the
                 model may add one round of extra calls, but it can never skip a required one.
evidence_pack()  tool results -> one line per source ("[id] doc . country . section | the sentences that match the question"),
                 deduplicated and inside a character budget. The verifier still checks against the FULL source text.
Measured before this change (Kenya roads planning question): 6 model calls, 4 sequential tool rounds, 17k characters of
context in the last call, 24-32 s end to end."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

CATEGORY_WORDS = [
    ("Non-Consulting Services", r"\bnon[- ]consult\w*"),
    ("Consultancies", r"\bconsult\w*|\bqcbs\b|\bqbs\b|\bterms of reference\b|\btor\b|\bsupervision firm\b|\btechnical assistance\b|\bindividual consultant"),
    ("Goods", r"\bgoods\b|\bequipment\b|\bsupply\b|\bsupplies\b|\bvehicles?\b|\bmedicines?\b|\bit hardware\b|\bcomputers?\b"),
    ("Civil Works", r"\bworks?\b|\bconstruction\b|\broads?\b|\bbridges?\b|\brehabilitation\b|\bcivil\b|\bbuilding\b"),
]
METHOD_WORDS = [("Request for Bids", r"\brfb\b|\brequest for bids\b|\bicb\b|\bncb\b|\bopen (?:international |national )?competition"),
                ("Quality And Cost-Based Selection", r"\bqcbs\b|quality and cost"), ("Request for Quotations", r"\brfq\b|request for quotations?"),
                ("Individual Consultant Selection", r"individual consultant"), ("Direct Selection", r"\bdirect (?:selection|contracting)\b")]
SECTOR_WORDS = [("Transportation", r"\broads?\b|\btransport\w*|\bhighways?\b|\bbridges?\b"),
                ("Water, Sanitation and Waste Management", r"\bwater\b|\bsanitation\b|\bwaste\b"), ("Health", r"\bhealth\b|\bhospitals?\b|\bclinics?\b"),
                ("Education", r"\beducation\b|\bschools?\b"), ("Energy and Extractives", r"\benergy\b|\bpower\b|\belectric\w*|\bsolar\b"),
                ("Agriculture, Fishing and Forestry", r"\bagricultur\w*|\birrigation\b|\bfarm\w*")]
PLAN_WORDS = r"\b(tender|procur\w*|plan\w*|package|prepare|launch|advertis\w*|bid\w*|award\w*)\b"
TIME_WORDS = r"\b(how long|timeline|duration|days|months|schedule|re-?bid\w*|re-?tender\w*|delay(?:ed)? award)\b"
LESSON_WORDS = r"\b(lessons?|learn\w*|slow\w*|risks?|problems?|issues?|challenges?|went wrong|mitigat\w*|experience|precedent\w*|why|delays?|bottleneck\w*|fail\w*|success\w*|recommend\w*|say about|what do .* say)\b"
WARN_WORDS = r"\b(early[- ]warning\w*|at risk|flag\w*|watch\w*|attention|struggl\w*|behind schedule|overdue|worry|concern\w*|off track)\b"
FOLLOW_UP = r"^\s*(what about|how about|and\b|same\b|for\b|in\b|what if|then\b|now\b|ok\b)|\b(that|those|these|it|them|this package)\b"
SYNONYMS = {"few bidders": "limited competition single bid low participation re-tendering",
            "land acquisition": "land acquisition resettlement right of way",
            "cost estimate": "cost estimate bid prices above estimate price escalation",
            "capacity": "procurement capacity implementing agency staff training",
            "contractor": "contractor performance supervision quality defects"}
THEMES = {"Civil Works": ["cost estimate bid prices above the estimate", "land acquisition right of way resettlement delays",
                          "contractor performance and supervision of works"],
          "Goods": ["specifications and delivery delays of goods", "few bidders limited competition re-tendering",
                    "cost estimate bid prices above the estimate"],
          "Consultancies": ["consultant selection delays and terms of reference quality", "implementing agency procurement capacity",
                            "evaluation and contract negotiation delays"],
          "Non-Consulting Services": ["few bidders limited competition re-tendering", "contract management and performance of service providers",
                                      "implementing agency procurement capacity"]}


@dataclass
class Plan:
    intents: list
    params: dict
    calls: list = field(default_factory=list)            # [(tool_name, args)] - all REQUIRED, executed in parallel
    note: str = ""


def _countries(text, country_names):
    found, low, squash = [], text.lower(), re.sub(r"[^a-z]", "", text.lower())
    for code, name in country_names.items():
        n = str(name).lower()
        if n and (n in low or re.sub(r"[^a-z]", "", n) in squash or re.search(rf"\b{re.escape(n)}(n|ian|ese|i)\b", low)):
            found.append(code)
        elif re.search(rf"(?<![A-Za-z]){re.escape(code)}(?![A-Za-z])", text) and len(code) == 2:
            found.append(code)
    return list(dict.fromkeys(found))


def extract(text, country_names, sectors=()) -> dict:
    t = text.lower()
    p = {}
    cs = _countries(text, country_names)
    if cs:
        p["country_code"] = cs[0]
    cat = next((c for c, pat in CATEGORY_WORDS if re.search(pat, t)), None)
    if cat:
        p["category"] = cat
    meth = next((m for m, pat in METHOD_WORDS if re.search(pat, t)), None)
    if meth:
        p["method"] = meth
    sec = next((s for s, pat in SECTOR_WORDS if re.search(pat, t) and (not sectors or s in sectors)), None)
    if sec:
        p["sector"] = sec
    m = re.search(r"(?:usd|us\$|\$)\s*([\d][\d.,]*)\s*(bn|billion|m|mn|million|k)?|([\d][\d.,]*)\s*(million|m|mn|bn|billion)\s*(?:usd|us\$|dollars)", t)
    if m:
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            v = float(num.replace(",", ""))
            p["value_usd_m"] = v * 1000 if unit in ("bn", "billion") else v / 1000 if unit == "k" else v if unit else v / 1e6 if v > 1e4 else v
        except ValueError:
            pass
    pids = list(dict.fromkeys(re.findall(r"\bP\d{6}\b", text.upper())))
    if pids:
        p["project_ids"] = pids[:3]
    return p


def _expand(q):
    extra = [v for k, v in SYNONYMS.items() if k in q.lower()]
    return (q + " " + " ".join(extra)).strip()


def route(question: str, country_names: dict, sectors=(), prev_plan: Plan | None = None, context: dict | None = None) -> Plan:
    q = question.strip(); t = q.lower()
    params = extract(q, country_names, sectors)
    follow = prev_plan is not None and (re.search(FOLLOW_UP, t) and len(q.split()) <= 14)
    if context:                                            # the package from the workspace
        for k in ("category", "sector", "country_code", "value_usd_m", "method"):
            if context.get(k) not in (None, ""):
                params.setdefault(k, context[k])
    if follow:                                             # "What about Ghana?" -> the previous package, Ghana
        for k, v in prev_plan.params.items():
            params.setdefault(k, v)
    intents = []
    if params.get("project_ids"):
        intents.append("project")
    if re.search(WARN_WORDS, t) and not params.get("project_ids"):
        intents.append("warnings")
    package = bool(params.get("category")) and (re.search(PLAN_WORDS, t) or params.get("value_usd_m") or context)
    if package or (follow and "package" in (prev_plan.intents if prev_plan else [])):
        intents.append("package")
    elif re.search(TIME_WORDS, t) and params.get("category"):
        intents.append("timeline")
    if re.search(LESSON_WORDS, t) or "package" in intents or not intents:
        intents.append("lessons")
    if follow and not set(intents) - {"lessons"}:
        intents = list(dict.fromkeys(prev_plan.intents + intents))

    country = country_names.get(params.get("country_code"), None)
    calls = []
    if "project" in intents:
        for pid in params["project_ids"]:
            calls.append(("get_project", dict(project_id=pid)))
            calls.append(("early_warnings", dict(project_id=pid)))             # why it is (or is not) flagged on the Portfolio monitor
            calls.append(("search_documents", dict(query=_expand(q), project_id=pid, k=4)))
    if "warnings" in intents:
        calls.append(("early_warnings", {"country_code": params.get("country_code"), "top": 6} if params.get("country_code") else {"top": 6}))
    if ("package" in intents or "timeline" in intents) and params.get("category") and params.get("country_code"):
        calls.append(("forecast_package", {k: v for k, v in dict(category=params["category"], country_code=params["country_code"],
                      sector=params.get("sector"), procurement_method=params.get("method"), value_usd_m=params.get("value_usd_m")).items() if v}))
    if "lessons" in intents and "project" not in intents:
        base = dict(doc_types=["ICR", "PAD", "PPAR"], country=country) if country else dict(doc_types=["ICR", "PAD", "PPAR"])
        calls.append(("search_documents", dict(base, query=_expand(q), section="lessons", k=5)))
        if "package" in intents:
            for theme in THEMES.get(params["category"], [])[:2]:
                calls.append(("search_documents", dict(base, query=f"{theme} {params.get('sector', '')}".strip(), section="lessons", k=3)))
        calls.append(("search_documents", dict(base, query=_expand(q), section="procurement", k=3)))
    note = ("package needs a category and a country for the forecast" if ("package" in intents or "timeline" in intents)
            and not (params.get("category") and params.get("country_code")) else "")
    return Plan(list(dict.fromkeys(intents)), params, calls, note)


# ------------------------------------------------------------------ evidence pack
def _stems(s):
    return {w[:5] for w in re.findall(r"[a-z]{4,}", s.lower())}


def snippet(text, query, max_chars=360):
    """The 1-3 sentences of a passage that best match the question, in their original order (numbers kept verbatim)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", re.sub(r"\s+", " ", text)) if len(s.strip()) > 20]
    if not sents:
        return text[:max_chars]
    q = _stems(query)
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & _stems(sents[i])))
    keep, total = [], 0
    for i in ranked:
        if total + len(sents[i]) > max_chars and keep:
            break
        keep.append(i); total += len(sents[i])
        if len(keep) == 3:
            break
    return " ".join(sents[i] for i in sorted(keep))[:max_chars + 80]


def evidence_pack(results: list, src: dict, question: str, budget=5200) -> str:
    """results: [(tool_name, args, returned)] in plan order -> compact, deduplicated, budgeted evidence lines."""
    lines, seen, used = [], set(), 0
    for name, args, res in results:
        rows = res if isinstance(res, list) else [res]
        for r in rows:
            if not isinstance(r, dict):
                continue
            if name == "search_documents" and "id" in r:
                if r["id"] in seen:
                    continue
                line = f"[{r['id']}] {r['doc_type']} {r['project_id']} · {r.get('country') or '?'} · {r['section'][:60]} | " \
                       f"{snippet(src.get(r['id'], {}).get('text', r.get('text', '')), args.get('query', question))}"
            elif name == "forecast_package" and "forecast" in r:
                f, c = r["forecast"], r["comparables"]
                line = (f"[{f['id']}] survival-model forecast ({f['method_used']}, {f['sector_used']}): p10_days {f['p10_days']}, "
                        f"p50_days {f['p50_days']}, p90_days {f['p90_days']}, rebid_probability_pct {f['rebid_probability_pct']}, "
                        f"amendment_probability_pct {f['amendment_probability_pct']}; drivers: {', '.join(f['drivers'])}\n"
                        f"[{c['id']}] {c['n_comparable']} comparable signed contracts: median_days {c['median_days']}, "
                        f"p80_days {c['p80_days']}, rebid_pct {c['rebid_pct']}")
            elif name == "get_project" and "id" in r:
                line = f"[{r['id']}] project record: " + "; ".join(f"{k} {v}" for k, v in r.items() if k not in ("id",) and v not in (None, ""))
            elif name == "early_warnings" and "id" in r and r.get("summary"):
                line = (f"[{r['id']}] {r['checked']} active projects in {r['where']} checked on {r['as_of']}; "
                        f"{r['flagged']} {'has' if r['flagged'] == 1 else 'have'} warning signs (only those are listed)")
            elif name == "early_warnings" and "id" in r:
                signs = "; ".join(f"{x['title']}: {x['detail']}" for x in r.get("alerts", [])) or "no warning signs"
                line = (f"[{r['id']}] {r.get('name') or r['project_id']} ({r['project_id']}), {r.get('country', '')}: {r.get('risk', '')}; "
                        f"{signs}; {r.get('open_tenders', 0)} open tenders, {r.get('overdue_tenders', 0)} running late")
            else:
                continue
            key = line.split("]")[0]
            if key in seen:
                continue
            if used + len(line) > budget and lines:
                break
            seen.add(key); lines.append(line); used += len(line)
    return "\n".join(lines) or "(no evidence found)"
