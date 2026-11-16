"""Module 2b: grounded retrieval for a new procurement package - the FIXED pipeline.

The LLM agent (agent.py) chooses tools and queries itself; this module is the deterministic path it falls back to, and the one
the structured form uses. Steps (every step is logged in report['trace']):
  1 parse package -> feature row (governance as-of notice date)
  2 predict timeline / re-bid / amendment risk (+ ablation explanation)
  3 structured analogues (kNN over past contracts) -> benchmark facts
  4 theme-conditioned retrieval over ICR/PPAR: one lessons chunk per theme, metadata-filtered & boosted
  5 graph expansion: cited chunk -> contract -> supplier/project; project -> ICR rating vs ISR (conflict resolution)
  6 compose findings (extractive by default; optional LLM) and push ALL of them through the guardrails
"""
from __future__ import annotations
import json, os, re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .guardrails import RATING_NAME, resolve_conflict, verify_findings
from .ingest import CID_RE
from .themes import THEMES


@dataclass
class PackageSpec:
    description: str
    category: str
    sector: str
    country_code: str
    value_usd_m: float | None
    method: str
    review_type: str = "Prior"
    notice_date: str = "2026-03-01"
    es_risk: str = "Moderate"


def _theme_priors(row, pred):
    ge = row.wgi_gov_effectiveness.iloc[0]; rq = row.wgi_reg_quality.iloc[0]
    clip = lambda x: float(np.clip(x, 0, 1))
    return {"estimate_gap": clip(.4 + pred.p_rebid), "spec_ambiguity": clip(.4 + pred.p_rebid + .3 * pred.p_amend),
            "gov_delay": clip(.5 - ge / 2), "agency_capacity": clip(.5 - ge / 2), "market_thin": clip(.5 - rq / 3),
            "supplier_performance": .55 if row.category.iloc[0] in ("Civil Works", "Goods") else .4}


def _stems(s):
    return {w[:5] for w in re.findall(r"[a-z]{4,}", s.lower())}


def _relevant(comp, z_min=2.5):
    """Lexical relevance gate. In-memory index: the hit's TF-IDF score must stand >= z_min SDs above the mean over ALL chunks
    ("how unlike a random chunk"), which is scale-free; an absolute cosine cut-off tuned on a 3k-chunk synthetic corpus rejected
    every good hit on 26k real chunks. Vector Search hits carry only a normalised score, so they keep the absolute rule."""
    return comp["sparse_z"] >= z_min if "sparse_z" in comp else comp["sparse"] >= 0.05


def _best_sentence(text, query):
    """Return (sentence, overlap) - sentence-level lexical overlap with the theme query (prefix stems)."""
    sents = [re.sub(r"^Contract P\d{6}-C\d{2}:\s*", "", s) for s in re.split(r"(?<=[.!?])\s+", text)]
    q = _stems(query)
    best = max(sents, key=lambda s: len(q & _stems(s)))
    return best, len(q & _stems(best))


class Copilot:
    def __init__(self, plat, llm=False, judge=None):
        """judge: optional guardrails.LLMJudge for L4 (Databricks-served); None or unavailable -> lexical L4."""
        self.p, self.llm, self.judge = plat, llm and bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("DATABRICKS_LLM_ENDPOINT")), judge

    # ---------------------------------------------------------------- analogues
    def analogues(self, row, k=15):
        c = self.p.ds.contracts.dropna(subset=["tender_days"])
        pool = c[(c.category == row.category.iloc[0]) & (c.sector == row.sector.iloc[0])]
        if len(pool) < 8:
            pool = c[c.category == row.category.iloc[0]]
        lv = row.log_value.iloc[0]
        dist = (pool.log_value.sub(lv).abs().fillna(1.0) + 2 * pool.wgi_gov_effectiveness.sub(row.wgi_gov_effectiveness.iloc[0]).abs().fillna(.5)
                + .5 * (pool.procurement_method != row.procurement_method.iloc[0]))
        return pool.loc[dist.nsmallest(k).index]

    def feature_row(self, spec: PackageSpec) -> pd.DataFrame:
        """Package -> model feature row, with governance as published on the notice date."""
        g = self.p.gov_asof(spec.country_code, spec.notice_date)
        return pd.DataFrame([dict(category=spec.category, procurement_method=spec.method, review_type=spec.review_type, sector=spec.sector,
                                  country_code=spec.country_code, es_risk=spec.es_risk,
                                  log_value=np.log(spec.value_usd_m) if spec.value_usd_m else np.nan,
                                  wgi_rule_of_law=g.wgi_rule_of_law, wgi_gov_effectiveness=g.wgi_gov_effectiveness,
                                  wgi_reg_quality=g.wgi_reg_quality, cpia=g.cpia, notice_month=pd.Timestamp(spec.notice_date).month)])

    # ---------------------------------------------------------------- main entry
    def analyze(self, spec: PackageSpec, themes_k=4):
        p, trace = self.p, []
        row = self.feature_row(spec)
        pred = p.models.predict_df(row).iloc[0]
        expl = p.models.explain(row)
        trace.append("predict: survival-model timeline (P10/P50/P90) + re-bid/amendment probabilities (calibrated GBMs)")
        an = self.analogues(row)
        facts = {"n_comparable": len(an), "bench_median_days": round(float(an.tender_days.median())),
                 "bench_p80_days": round(float(an.tender_days.quantile(.8))), "bench_rebid_pct": round(100 * float(an.rebid.mean()), 1),
                 "bench_amend_pct": round(100 * float((an.amendments > 0).mean()), 1),
                 "model_p10_days": round(float(pred.p10_days)), "model_p50_days": round(float(pred.p50_days)),
                 "model_p90_days": round(float(pred.p90_days)), "model_rebid_pct": round(100 * float(pred.p_rebid), 1),
                 "model_amend_pct": round(100 * float(pred.p_amend), 1)}
        trace.append(f"analogues: {len(an)} comparable contracts ({spec.category}, {spec.sector}, nearest by value & governance)")
        cands = [
            dict(kind="model", facts=["model_p10_days", "model_p50_days", "model_p90_days"],
                 claim=f"Planning forecast for tender-to-award: median {facts['model_p50_days']} days (P10 {facts['model_p10_days']}, P90 {facts['model_p90_days']}).", evidence=[]),
            dict(kind="model", facts=["model_rebid_pct", "model_amend_pct"],
                 claim=f"Model risk: {facts['model_rebid_pct']}% probability of re-bid and {facts['model_amend_pct']}% probability of at least one amendment.", evidence=[]),
            dict(kind="benchmark", facts=["n_comparable", "bench_median_days", "bench_p80_days", "bench_rebid_pct"],
                 claim=f"Among {facts['n_comparable']} comparable past contracts, median tender-to-award was {facts['bench_median_days']} days (P80 {facts['bench_p80_days']}) and {facts['bench_rebid_pct']}% were re-bid.", evidence=[]),
        ]
        if len(an) < 8:
            cands[-1]["claim"] += " (small sample: treat as indicative)"
        # theme retrieval
        priors = _theme_priors(row, pred)
        prefer = dict(sector=spec.sector, country=p.country_names.get(spec.country_code))
        themed = []
        for th, t in THEMES.items():
            q = f"{t['query']} {spec.description}"
            les = p.index.search(q, k=3, filters={"doc_type": {"ICR", "PPAR"}, "section_kind": {"lessons"}}, prefer=prefer)
            if not les:
                continue
            lc, ls, comp = les[0]
            if not _relevant(comp):
                continue
            lq, lo = _best_sentence(lc.text, t["query"])
            if lo < 1:
                continue
            # lesson only: the 'precedent' (a sentence from an ICR procurement/fiduciary section) was removed - those sections
            # are mostly financial-management and safeguards text, so the precedent was often off-topic or missing
            ev = [dict(chunk_id=lc.chunk_id, quote=lq, role="lesson")]
            themed.append((ls * priors[th], dict(kind="risk", theme=th, claim=ev[0]["quote"], evidence=ev, facts=[],
                                                  recommendation=t["mitigation"], prior=round(priors[th], 2), retrieval_score=round(ls, 2))))
        themed.sort(key=lambda x: -x[0])
        cands += [f for _, f in themed[:themes_k]]
        trace.append(f"retrieval: {len(themed)} themes with evidence; top {themes_k} kept, ranked by retrieval score x model-derived prior")
        if self.llm:
            cands += self._llm_findings(spec, [f for _, f in themed[:themes_k]]); trace.append("llm: added LLM-composed candidates (verified identically)")
        cb = p.chunks_by_id
        ok, bad = verify_findings(cands, cb, facts, dict(country=p.country_names.get(spec.country_code), sector=spec.sector), judge=self.judge)
        trace.append(f"guardrails: {len(ok)} accepted, {len(bad)} rejected")
        # graph expansion + conflicts
        conflicts, seen = [], set()
        for f in ok:
            for e in f["evidence"]:
                pid = e["project_id"]
                cids = CID_RE.findall(cb[e["chunk_id"]].text)
                if cids:
                    e["contract"] = cids[0]
                    e["supplier_id"] = next((s[1] for _, s, a in p.graph.g.out_edges(("Contract", cids[0]), data=True) if a["type"] == "AWARDED_TO"), None)
                if pid not in seen:
                    seen.add(pid)
                    r = resolve_conflict(pid, p.ds.isr, p.icr.get(pid), bias=p.bias())
                    if r:
                        conflicts.append(r)
        flagged = {c["project_id"] for c in conflicts}
        for f in ok:
            if any(e["project_id"] in flagged for e in f["evidence"]):
                f["caution"] = "cited project has an ISR/ICR rating conflict; ICR adopted (see conflicts)"
        abstain = not any(f["kind"] == "risk" for f in ok)
        return dict(spec=spec, facts=facts, explanation=expl, findings=ok, rejected=bad, conflicts=conflicts,
                    abstained=abstain, trace=trace,
                    schedule=dict(plan_days=facts["model_p50_days"], contingency_days=facts["model_p90_days"],
                                  add_rebid_buffer=bool(pred.p_rebid > .25)))

    # ---------------------------------------------------------------- optional LLM composer (needs ANTHROPIC_API_KEY; not exercised in tests)
    def _llm_findings(self, spec, themed):
        ev = [dict(id=e["chunk_id"], text=self.p.chunks_by_id[e["chunk_id"]].text) for f in themed for e in f["evidence"]]
        prompt = ("You advise a procurement team on: " + spec.description + ". Using ONLY the evidence blocks below, return JSON list of "
                  "{claim, evidence:[{chunk_id, quote}]}. Each quote must be copied verbatim from its block. No numbers that are not in the quotes.\n"
                  + json.dumps(ev))
        if os.getenv("DATABRICKS_LLM_ENDPOINT"):      # Mosaic AI Model Serving / Foundation Model API (VERIFY on a workspace)
            from mlflow.deployments import get_deploy_client
            r = get_deploy_client("databricks").predict(endpoint=os.environ["DATABRICKS_LLM_ENDPOINT"],
                                                        inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 1500})
            text = r["choices"][0]["message"]["content"]
        else:
            import anthropic
            text = anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", max_tokens=1500,
                                                         messages=[{"role": "user", "content": prompt}]).content[0].text
        try:
            return [dict(kind="risk", facts=[], **x) for x in json.loads(text)]
        except Exception:
            return []


def render(rep) -> str:
    s, f, L = rep["spec"], rep["facts"], []
    L += [f"PACKAGE: {s.description} | {s.category} | {s.sector} | {s.country_code} | USD {s.value_usd_m}m | {s.method} ({s.review_type} review)", ""]
    L += [f"SCHEDULE: plan {rep['schedule']['plan_days']} days to award; hold contingency to {rep['schedule']['contingency_days']} days"
          + ("; add a re-bid buffer" if rep["schedule"]["add_rebid_buffer"] else ""), ""]
    L += ["TOP DRIVERS (ablation vs reference): " + ", ".join(f"{e['feature']}={e['value'] if not isinstance(e['value'], float) else round(e['value'], 2)} ({e['d_days']:+.0f}d, {100*e['d_rebid']:+.1f}pp rebid)" for e in rep["explanation"]), ""]
    L += ["FINDINGS (each verified against source)"]
    for i, x in enumerate(rep["findings"], 1):
        L.append(f" {i}. [{x['kind']}{'/' + x['theme'] if 'theme' in x else ''}] {x['claim']}")
        if "recommendation" in x:
            L.append(f"      -> Recommend: {x['recommendation']}")
        for e in x["evidence"]:
            L.append(f"      source: {e['doc_id']} p.{e['page']} \"{e['section']}\" [{e['scope']}]" + (f" contract {e['contract']}" if e.get("contract") else "") + f"\n        \"{e['quote'][:150]}\"")
        if x.get("caution"):
            L.append(f"      ! {x['caution']}")
    if rep["abstained"]:
        L.append("  (no grounded historical risk evidence found: abstaining rather than guessing)")
    if rep["conflicts"]:
        L += ["", "EVIDENCE CONFLICTS"]
        for c in rep["conflicts"]:
            L.append(f" - {c['project_id']}: ISR {c['isr_last']['rating']} ({c['isr_last']['date']}) vs ICR {c['icr']['rating']} -> adopt {c['adopted_source']} ({c['rule_applied']}), conf {c['confidence']}"
                     + (" [HUMAN REVIEW]" if c["needs_human_review"] else "") + f"\n     {c['note']}")
    L += ["", f"GUARDRAILS: {len(rep['findings'])} accepted / {len(rep['rejected'])} rejected"]
    return "\n".join(L)
