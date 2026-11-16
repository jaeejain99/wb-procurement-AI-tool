"""Module 3: faithfulness guardrails and evidence-conflict resolution.

Validation layers applied to EVERY candidate finding (from the extractive composer or from an LLM):
  L1 citation required         - no evidence => rejected
  L2 quote verification        - each quote must be a verbatim (whitespace-normalised) span of the cited chunk
  L3 numeric grounding         - every number in the claim must occur in a cited quote or in the computed-facts table
  L4 support check             - an LLM judge on Databricks Model Serving (a different model family from the agent) decides
                                 whether the cited sources support the claim; lexical overlap is the offline fallback
  L5 scope labelling           - evidence from another country/sector is kept but labelled 'analogue'
  L6 abstention                - too little accepted evidence => the system says so instead of guessing
Agent answers are checked sentence by sentence with the same layers (verify_answer): citation required, cited ids must have
come back from a tool in this session, numbers grounded in the cited sources, judge-checked support, and country scope.
"""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd

_NUM = re.compile(r"\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")
RATING_NAME = {6: "Highly Satisfactory", 5: "Satisfactory", 4: "Moderately Satisfactory",
               3: "Moderately Unsatisfactory", 2: "Unsatisfactory", 1: "Highly Unsatisfactory"}
RATING_NUM = {v: k for k, v in RATING_NAME.items()}


def _norm(s):
    return _WS.sub(" ", s).strip().lower()


def _tokens(s):
    return set(re.findall(r"[a-z]{4,}", s.lower()))


JUDGE_PROMPT = """You check whether sources support claims, for a procurement audit. For each item, decide if the claim is
SUPPORTED by its sources: every factual element of the claim (numbers, places, causes, timing, who did what) must be stated in
or directly implied by the sources. Paraphrase is fine. A recommendation ("plan for...", "consider...") is supported when the
risk or practice it rests on is in the sources. If the claim adds a country, cause, number or generalisation the sources do not
give, it is NOT supported. Answer ONLY a JSON array: [{"i": <item number>, "supported": true|false, "reason": "<= 15 words"}]."""


class LLMJudge:
    """L4 on Databricks Model Serving. Returns None when the endpoint is unavailable so callers fall back to the lexical check."""

    def __init__(self, client=None, model=None, batch=6):
        from .llm import JUDGE_MODEL, shared_client
        self.client, self.model, self.batch = client or shared_client(), model or JUDGE_MODEL, batch
        self.calls = 0

    def __call__(self, items):
        """items: [(claim, [source_text, ...])] -> [(supported: bool, reason)] or None."""
        from .llm import LLMUnavailable, text_of
        out = []
        for k in range(0, len(items), self.batch):
            chunk = items[k:k + self.batch]
            payload = [dict(i=i, claim=c, sources=[t[:1500] for t in srcs][:4]) for i, (c, srcs) in enumerate(chunk)]
            try:
                msg = self.client.chat(self.model, [{"role": "system", "content": JUDGE_PROMPT},
                                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], max_tokens=600)
                self.calls += 1
                m = re.search(r"\[.*\]", text_of(msg), re.S)
                verdict = {int(v["i"]): (bool(v["supported"]), str(v.get("reason", ""))) for v in json.loads(m.group(0))}
            except (LLMUnavailable, AttributeError, ValueError, KeyError, TypeError):
                return None
            out += [verdict.get(i, (False, "judge gave no verdict")) for i in range(len(chunk))]
        return out


def _lexical_support(claim, quotes):
    return len(_tokens(claim) & _tokens(quotes)) / max(1, len(_tokens(claim)))


def verify_findings(findings, chunks_by_id, facts: dict, ctx: dict, min_support=0.5, judge=None):
    """findings: [{claim, kind, evidence:[{chunk_id, quote}], facts:[fact_id,...]}]. Returns (accepted, rejected).
    judge: optional LLMJudge for L4; without one (or if it is unavailable) L4 is the lexical overlap check."""
    ok, bad, pending = [], [], []
    for f in findings:
        reasons = []
        ev = f.get("evidence", [])
        if not ev and not f.get("facts"):
            reasons.append("L1 no citation")
        for e in ev:
            ch = chunks_by_id.get(e["chunk_id"])
            if ch is None:
                reasons.append(f"L1 unknown chunk {e['chunk_id']}")
            elif _norm(e["quote"]) not in _norm(ch.text):
                reasons.append(f"L2 quote not found in {e['chunk_id']}")
            else:
                e["page"], e["section"], e["doc_id"], e["project_id"] = ch.page, ch.section_path, ch.doc_id, ch.project_id
                same_c, same_s = ch.country == ctx.get("country"), ch.sector == ctx.get("sector")
                e["scope"] = "same country+sector" if same_c and same_s else "same sector" if same_s else "same country" if same_c else "analogue (different context)"
        quotes = " ".join(e["quote"] for e in ev)
        own = {f"{facts[k]:g}" for k in f.get("facts", []) if k in facts and isinstance(facts[k], (int, float))}
        for n in _NUM.findall(re.sub(r"\bP\d+\b", "", f["claim"])):      # 'P90' is a label, not a quantity
            if f"{float(n):g}" not in own and f"{float(n):g}" not in {f"{float(q):g}" for q in _NUM.findall(quotes)}:
                reasons.append(f"L3 number {n} not grounded")
        if ev and not reasons:
            pending.append((f, quotes))                      # L4 runs once, batched, on findings that passed L1-L3
            continue
        (bad if reasons else ok).append({**f, "rejected_because": reasons} if reasons else f)
    verdicts = judge([(f["claim"], [q]) for f, q in pending]) if (judge and pending) else None
    for n, (f, quotes) in enumerate(pending):
        if verdicts is not None:
            sup, why = verdicts[n]
            f["support_check"] = "llm judge"
            reasons = [] if sup else [f"L4 judge: not supported ({why})"]
        else:
            sup = _lexical_support(f["claim"], quotes)
            f["support"], f["support_check"] = round(sup, 2), "lexical"
            reasons = [] if sup >= min_support else [f"L4 low support {sup:.2f}"]
        (bad if reasons else ok).append({**f, "rejected_because": reasons} if reasons else f)
    return ok, bad


# ------------------------------------------------------------------ agent answers, sentence by sentence
_CITE = re.compile(r"\[([^\[\]]+)\]")
_META = re.compile(r"\b(not enough evidence|no (?:relevant |direct )?evidence|could not find|did not find|cannot answer|can't answer|"
                   r"outside (?:my|the) scope|unable to|cannot be (?:calculated|determined|estimated))\b", re.I)
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def norm_id(i: str) -> str:
    """Models sometimes write ids with typographic hyphens (forecast:NP‑Goods) or stray spaces; ids are ASCII."""
    return re.sub(r"\s+", " ", str(i).translate(_DASHES)).strip()


def split_cited_sentences(text: str):
    """Agent text -> [(sentence_without_ids, [ids])]. Accepts [id], 【id】 and [id1, id2]; a citation after the full stop stays
    with its sentence; bullets and markdown emphasis are stripped."""
    t = text.replace("【", "[").replace("】", "]").replace("**", "")
    t = re.sub(r"([.!?])[ \t]*((?:\[[^\[\]]+\][ \t]*)+)", r" \2\1 ", t)    # "... works. [id]" -> "... works [id]." (same line only)
    out = []
    for line in t.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        parts = []
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", line):
            s = s.strip()
            if len(s) < 3:
                continue
            ids = [norm_id(i) for m in _CITE.findall(s) for i in re.split(r"[,;]", m) if i.strip()]
            clean = re.sub(r"\s+([.!?,;:])", r"\1", _WS.sub(" ", _CITE.sub("", s))).strip().rstrip(" ,;")
            parts.append([clean, ids])
        # a citation at the end of a bullet or paragraph covers its earlier sentences too (each is still verified against it)
        nxt = []
        for part in reversed(parts):
            if part[1]:
                nxt = part[1]
            elif nxt and re.search(r"[.!?]$", part[0]):
                part[1] = list(nxt)
        out.extend((c, i) for c, i in parts)
    return out


def _nums(text):
    """'1,273', '1 273' (incl. thin / no-break spaces) are ONE number, as are '20 387' and '20387'. A group that is part of a
    label ('p50 213 days', 'P90 540') is not a thousands prefix, so it is never joined to the next number."""
    t = re.sub(r"(?<![\w.])\d{1,3}(?:[,\u202f\u00a0\u2009 ]\d{3})+(?![\d.,]\d)", lambda m: re.sub(r"\D", "", m.group()), str(text))
    return {f"{float(n):g}" for n in _NUM.findall(t)}


def _squash(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


class StreamVerifier:
    """Sentence-by-sentence verification of agent text, usable on a token stream: feed() text as it arrives and get back the
    sentences that are settled, in their original order. A sentence is released only after every check has passed:
      L1 a factual sentence must cite, and every cited id must be one a tool returned in this session
      L3 every number must occur in the cited sources (or in the user's own question)
      L5 a country the sentence names must be the country of a cited source
      L4 the judge model agrees the cited sources support it; if the judge is unavailable, lexical support must be high,
         otherwise the sentence is removed - nothing reaches the user unchecked.
    Headings are released only when a verified claim follows them; "not enough evidence" sentences make no claim and pass."""

    def __init__(self, sources: dict, country_names=(), allowed_numbers=(), judge=None, batch=2, min_lexical=0.5):
        from concurrent.futures import ThreadPoolExecutor
        self.sources, self.country_names, self.judge, self.batch, self.min_lexical = sources, list(country_names), judge, batch, min_lexical
        self.allowed = {f"{float(n):g}" for n in allowed_numbers}
        self.buf, self.n_seen, self.recs, self.emitted = "", 0, [], 0
        self.pending, self.futures = [], []
        self.pool = ThreadPoolExecutor(max_workers=8) if judge else None
        self.held_heading = None

    # ---- the cheap checks (L1, L3, L5)
    def _static(self, sent, ids):
        if not ids:
            heading = len(sent.split()) <= 8 and not _NUM.search(sent) and not re.search(r"[.!?]$", sent)
            if _META.search(sent):
                return "meta", []
            if heading:
                return "heading", []
            return "removed", ["L1 no citation"]
        reasons = []
        unknown = [i for i in ids if i not in self.sources]
        if unknown:
            reasons.append(f"L1 cites {', '.join(unknown)}, which no tool returned")
        cited = [self.sources[i] for i in ids if i in self.sources]
        if cited:
            have = set().union(*[_nums(c["text"]) for c in cited]) | self.allowed
            have |= {f"{float(n) * 100:g}" for n in have} | {f"{float(n) / 100:g}" for n in have}   # 0.13 <-> 13 (%)
            # ids and codes are not quantities: P159238, OP00355746, ISR08449, CW4B, FY2024, P90
            for n in sorted(_nums(re.sub(r"\b[A-Za-z]{1,4}[-_]?\d[\w-]*\b", "", sent)) - have):
                reasons.append(f"L3 number {n} is not in the cited sources")
            src_countries = {_squash(c.get("country", "")) for c in cited}
            for name in self.country_names:
                if name and _squash(name) in _squash(sent) and _squash(name) not in src_countries:
                    reasons.append(f"L5 scope: names {name}, but the cited sources are from "
                                   f"{', '.join(sorted({c.get('country') or '?' for c in cited}))}")
        return ("removed", reasons) if reasons else ("judge", [])

    # ---- L4
    @staticmethod
    def _meta(c):            # the judge sees each source's metadata too: "this passage is from a Ghana project" lives there
        return f"[source: {c.get('label', '')}; country: {c.get('country') or 'unknown'}] {c['text']}"

    def _judge_batch(self, recs):
        items = [(r["text"], [self._meta(self.sources[i]) for i in r["ids"] if i in self.sources]) for r in recs]
        verdicts = self.judge(items) if self.judge else None
        for k, (r, (claim, srcs)) in enumerate(zip(recs, items)):
            if verdicts is not None:
                ok, why = verdicts[k]
                r.update(state="ok" if ok else "removed", support_check="llm judge",
                         reasons=[] if ok else [f"L4 judge: not supported ({why})"])
            else:
                sup = _lexical_support(claim, " ".join(srcs))
                ok = sup >= self.min_lexical
                r.update(state="ok" if ok else "removed", support_check=f"lexical {sup:.2f} (judge unavailable)",
                         reasons=[] if ok else [f"L4 could not be checked by the judge and lexical support is low ({sup:.2f})"])

    def _submit(self, force=False):
        while self.pending and (force or len(self.pending) >= self.batch):
            take, self.pending = self.pending[:self.batch], self.pending[self.batch:]
            if self.pool:
                self.futures.append(self.pool.submit(self._judge_batch, take))
            else:
                self._judge_batch(take)

    # ---- release settled records in order
    def _flush(self):
        out = []
        while self.emitted < len(self.recs) and self.recs[self.emitted]["state"] != "pending":
            r = self.recs[self.emitted]; self.emitted += 1
            if r["state"] == "heading":
                self.held_heading = r                         # shown only if a verified claim follows
            elif r["state"] == "ok":
                if self.held_heading:
                    out.append(("sentence", self.held_heading)); self.held_heading = None
                out.append(("sentence", r))
            elif r["state"] == "meta":
                out.append(("sentence", r))
            elif r["state"] == "removed":
                out.append(("removed", r))
        return out

    def _take(self, parts):
        for sent, ids in parts:
            kind, reasons = self._static(sent, ids)
            rec = dict(text=sent, ids=ids, state="pending" if kind == "judge" else kind, reasons=reasons)
            if kind in ("meta", "heading"):
                rec["meta"] = True
            self.recs.append(rec)
            if kind == "judge":
                self.pending.append(rec)

    def feed(self, delta: str):
        self.buf += delta
        parts = split_cited_sentences(self.buf)
        complete = parts[:-1] if parts else []               # the last part may still be growing
        self._take(complete[self.n_seen:]); self.n_seen = max(self.n_seen, len(complete))
        self._submit()
        return self._flush()

    def finish_iter(self):
        """End of text: yield each settled record as soon as possible (in order), instead of after every judge call."""
        from concurrent.futures import as_completed
        parts = split_cited_sentences(self.buf)
        self._take(parts[self.n_seen:]); self.n_seen = len(parts)
        self._submit(force=True)
        yield from self._flush()
        for f in as_completed(self.futures):
            f.result()
            yield from self._flush()
        if self.pool:
            self.pool.shutdown(wait=True)
        yield from self._flush()

    def finish(self):
        parts = split_cited_sentences(self.buf)
        self._take(parts[self.n_seen:]); self.n_seen = len(parts)
        self._submit(force=True)
        for f in self.futures:
            f.result()
        if self.pool:
            self.pool.shutdown(wait=True)
        return self._flush()

    def results(self):
        """(kept, removed) in original order, the same shape verify_answer always returned."""
        kept, removed = [], []
        for i, r in enumerate(self.recs):
            st = r["state"]
            if st == "ok":
                kept.append(dict(text=r["text"], ids=r["ids"], support_check=r.get("support_check")))
            elif st == "meta" or (st == "heading" and any(x["state"] == "ok" for x in self.recs[i + 1:])):
                kept.append(dict(text=r["text"], ids=[], meta=True))
            elif st == "removed":
                removed.append(dict(text=r["text"], ids=r["ids"], reasons=r["reasons"]))
        return kept, removed


def verify_answer(text: str, sources: dict, ctx: dict, judge=None, country_names=(), allowed_numbers=()):
    """Whole-text verification (the same StreamVerifier, fed in one go). Returns (kept, removed)."""
    v = StreamVerifier(sources, country_names, allowed_numbers, judge, batch=6)
    v.feed(text); v.finish()
    return v.results()


# ------------------------------------------------------------------ conflicting evidence
SOURCE_RELIABILITY = {"PPAR": 1.0, "ICR": 0.85, "ISR": 0.55}   # independent review > post-hoc self-eval > contemporaneous self-report


def optimism_bias(isr: pd.DataFrame, icr_ratings: dict) -> dict:
    """Empirical gap between the last ISR rating and the ICR outcome rating on closed projects (used to de-bias ISR-based alerts)."""
    last = isr.sort_values("date").groupby("project_id").dev_objective_rating.last()
    gaps = [last[p] - r for p, r in icr_ratings.items() if p in last.index]
    return dict(mean_gap=float(np.mean(gaps)) if gaps else 0.0, share_isr_higher=float(np.mean([g > 0 for g in gaps])) if gaps else 0.0, n=len(gaps))


def resolve_conflict(project_id, isr: pd.DataFrame, icr_rating: int | None, ppar_rating: int | None = None, bias=None) -> dict | None:
    """Rules, in order:
       R1 Independent review (PPAR) beats ICR beats last ISR for OUTCOME claims.
       R2 ISR wins for contemporaneous EVENT dating (what was known when) - not handled here, flagged in output.
       R3 If a conflict is >=2 notches and the ISR trend was declining before closure, treat as late-recognised deterioration.
       R4 Otherwise (<=1 notch, or missing ICR) report both, flag 'needs_human_review' only when reliability weights tie.
       Always: surface both sources with dates; never silently pick."""
    s = isr[isr.project_id == project_id].sort_values("date")
    if s.empty or icr_rating is None:
        return None
    last = int(s.dev_objective_rating.iloc[-1]); gap = last - icr_rating
    if gap == 0:
        return None
    trend = int(s.dev_objective_rating.iloc[-1] - s.dev_objective_rating.iloc[max(0, len(s) - 4)])
    winner = "PPAR" if ppar_rating is not None else "ICR"
    rule = "R4" if abs(gap) == 1 else "R1"           # R4: 1-notch disagreement -> both shown, outcome source still adopted
    adopted = ppar_rating if ppar_rating is not None else icr_rating
    if gap < 0:
        why = "ICR rated the outcome higher than the last ISR (late recovery or conservative supervision ratings)"
    elif trend < 0:
        why = "ISR trend was declining before closure (late-recognised deterioration, R3)"
    else:
        why = "ISR ratings were stable while ICR was lower: consistent with systematic ISR optimism"
    return dict(project_id=project_id, isr_last=dict(rating=RATING_NAME[last], date=str(s.date.iloc[-1].date())),
                icr=dict(rating=RATING_NAME[icr_rating]), gap_notches=gap, rule_applied=rule, adopted_source=winner,
                adopted_rating=RATING_NAME[adopted], severity="minor" if abs(gap) == 1 else "major",
                note=why + (f"; portfolio mean ISR-ICR gap {bias['mean_gap']:+.2f} over {bias['n']} closed projects" if bias else ""),
                confidence=round(SOURCE_RELIABILITY[winner] / (SOURCE_RELIABILITY[winner] + SOURCE_RELIABILITY["ISR"]), 2),
                # ISR improving yet ICR >=2 notches lower: contradictory, escalate
                needs_human_review=bool(gap >= 2 and trend > 0 and ppar_rating is None))
