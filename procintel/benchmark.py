"""Benchmark the agent (and the fixed pipeline on package questions) against evals/benchmark.json.

    python -m procintel.benchmark --data data/live [--out evals/results.json]

Per item: answer -> expected tools called + a verified sentence citing each expected kind of source; abstain -> no verified
sentence makes the forbidden claim; refuse -> declined (policy screen or model). Also reports how much of the agent's own
draft the verifier removed, by layer - the hallucination the guardrails caught before anyone saw it."""
from __future__ import annotations
import argparse, collections, json, os, statistics, time, warnings

KIND = lambda i: ("passage" if "#" in i else "model" if i.startswith("forecast:") else "comparables" if i.startswith("comparables:")
                  else i.split(":", 1)[0])


def _fill(q, plat):
    from .guardrails import resolve_conflict
    icr = sorted(plat.icr)
    conf = [p for p in icr if resolve_conflict(p, plat.ds.isr, plat.icr[p])] or icr
    return (q.replace("{ICR_PROJECT_1}", icr[0] if icr else "P000000").replace("{ICR_PROJECT_2}", icr[1] if len(icr) > 1 else "P000000")
             .replace("{CONFLICT_PROJECT}", conf[0] if conf else "P000000"))


def score(item, r):
    tools = [t.split(":", 1)[1].split("(")[0].strip() for t in r.get("trace", []) if t.startswith("round ")]
    factual = [a for a in r.get("answer", []) if not a.get("meta")]
    kinds = {KIND(i) for a in factual for i in a["ids"]}
    why = []
    if item["type"] == "refuse":
        return (bool(r.get("refused")), [] if r.get("refused") else ["did not refuse"])
    if r.get("refused"):
        return False, ["refused an in-scope question"]
    if item["type"] == "abstain":
        import re
        from .guardrails import _META                   # "no evidence of X was found" / "none were for X" mention X without claiming it
        negated = lambda t: _META.search(t) or re.search(r"\b(no|none|not|cannot|without|do not|does not|did not)\b", t, re.I)
        bad = [a["text"] for a in factual if any(f.lower() in a["text"].lower() for f in item["forbid"]) and not negated(a["text"])]
        return (not bad, [f"verified sentence makes the forbidden claim: {bad[0][:80]}"] if bad else [])
    for t in item.get("tools_all", []):
        if t not in tools:
            why.append(f"did not call {t}")
    if item.get("tools_any") and not set(item["tools_any"]) & set(tools):
        why.append(f"called none of {item['tools_any']}")
    for frag in item.get("trace_contains", []):
        if not any(frag in t for t in r.get("trace", [])):
            why.append(f"no tool call with {frag}")
    for k in item.get("cite_kinds", []):
        if k not in kinds:
            why.append(f"no verified sentence cites a {k}")
    if r.get("mode") == "fallback":
        why.append("model unavailable (fallback)")
    return not why, why


def run(data, out, judge=True, limit=None):
    from .agent import Agent
    from .core import Platform
    from .guardrails import LLMJudge
    from .rag import Copilot, PackageSpec
    warnings.filterwarnings("ignore")
    plat = Platform.build(data)
    agent = Agent(plat, use_judge=judge)
    fixed = Copilot(plat, judge=LLMJudge(agent.client) if judge else None)
    items = json.load(open(os.path.join(os.path.dirname(__file__), "..", "evals", "benchmark.json")))["items"][:limit]
    rows = []
    for it in items:
        turns = [_fill(t, plat) for t in it.get("turns", [it.get("question", "")])]
        q = " -> ".join(turns)
        t0, history = time.time(), []
        for t in turns:                                  # multi-turn items: earlier turns build the memory, the last is scored
            r = agent.ask(t, history=history)
            history = (history + [dict(question=r["question"], answer=r["answer"], refused=r.get("refused"), carry=r.get("carry", {}),
                                       plan=r.get("plan"))])[-4:]
        ok, why = score(it, r)
        row = dict(id=it["id"], group=it["group"], type=it["type"], question=q, passed=ok, why=why, mode=r.get("mode"),
                   refused=r.get("refused"), seconds=round(time.time() - t0, 1),
                   kept=[dict(text=a["text"], ids=a["ids"]) for a in r.get("answer", []) if not a.get("meta")],
                   removed=r.get("removed", []), tools=[t for t in r.get("trace", []) if t.startswith("round ")])
        if it.get("spec"):                              # same package through the fixed pipeline, for comparison
            s = it["spec"]
            rep = fixed.analyze(PackageSpec("", s["category"], s["sector"], s["country_code"], s["value_usd_m"], s["method"]))
            row["fixed_risk_findings"] = sum(f["kind"] == "risk" for f in rep["findings"])
            row["agent_passage_sentences"] = sum(any("#" in i for i in a["ids"]) for a in row["kept"])
        rows.append(row)
        print(f"{'PASS' if ok else 'FAIL'}  {it['id']:22s} {row['seconds']:5.1f}s  kept={len(row['kept'])} removed={len(row['removed'])} {'; '.join(why)}")
    summary = summarise(rows)
    json.dump(dict(summary=summary, items=rows, agent_model=agent.model, judge_model=getattr(agent.judge, "model", None),
                   data=data, run_at=time.strftime("%Y-%m-%d %H:%M")), open(out, "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1))
    return summary


def summarise(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r["passed"])
    answered = [r for r in rows if r["type"] != "refuse" and not r["refused"]]
    kept = sum(len(r["kept"]) for r in answered); removed = sum(len(r["removed"]) for r in answered)
    layers = collections.Counter(reason.split(" ")[0] for r in answered for x in r["removed"] for reason in x["reasons"][:1])
    pk = [r for r in rows if "fixed_risk_findings" in r]
    return dict(
        passed=sum(r["passed"] for r in rows), total=len(rows),
        answer_items=f"{sum(r['passed'] for r in rows if r['type'] == 'answer')}/{sum(r['type'] == 'answer' for r in rows)}",
        abstain_items=f"{sum(r['passed'] for r in rows if r['type'] == 'abstain')}/{sum(r['type'] == 'abstain' for r in rows)}",
        must_refuse=f"{sum(r['passed'] for r in rows if r['type'] == 'refuse')}/{sum(r['type'] == 'refuse' for r in rows)}",
        refused_by_policy_screen=sum(r["mode"] == "policy" for r in rows if r["type"] == "refuse"),
        by_group={g: f"{sum(v)}/{len(v)}" for g, v in by.items()},
        draft_sentences=kept + removed, verified_sentences=kept, removed_sentences=removed,
        removed_share=round(removed / max(1, kept + removed), 3), removed_by_layer=dict(layers),
        answers_held_back=sum(1 for r in answered if not r["kept"]),
        median_seconds=statistics.median([r["seconds"] for r in rows if r["mode"] == "agent"] or [0]),
        package_questions_with_a_grounded_lesson=dict(agent=sum(r["agent_passage_sentences"] > 0 for r in pk),
                                                      fixed_pipeline=sum(r["fixed_risk_findings"] > 0 for r in pk), of=len(pk)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/live"); ap.add_argument("--out", default="evals/results.json")
    ap.add_argument("--no-judge", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.data, a.out, judge=not a.no_judge, limit=a.limit)
