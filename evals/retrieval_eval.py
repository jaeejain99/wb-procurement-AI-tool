"""Evidence-library retrieval evaluation on the LIVE corpus: can search put relevant passages in the top 5 and top 10?

    PYTHONPATH=. python evals/retrieval_eval.py --snapshot DIR [--vs-index CAT.SCHEMA.chunks_index] [--judge MODEL]

DIR is the lakehouse snapshot the app loads (copy of /Volumes/<cat>/<schema>/landing/snapshot). Four ways of searching are
compared, each asked for its top 10:
    memory    in-memory hybrid index (TF-IDF + LSA, rank-fused)            plat.index (HybridIndex)
    vector    Mosaic AI Vector Search, HYBRID query                        VectorSearchRetriever
    fused     both, reciprocal-rank fused (what the agent retrieves from)  FusedRetriever
    library   fused + rare-term scan + MIN_MATCH threshold + de-duplication (what the Evidence library shows)  services.search

Labels (pooled relevance judgement, the TREC method): the top 10 of every system are pooled per question and each pooled
passage is graded once by an LLM that the app does not use (default Llama 4 Maverick):
    2 relevant   gives evidence that answers the question (and is about the named project/country, if one is named)
    1 partial    on the topic but generic, about a different project, or only a passing mention
    0 not relevant
Grades are cached in evals/retrieval_labels.json, so a re-run only judges new passages; edit a grade there to override the
judge (set "by": "human"). Metrics use grade 2 as relevant; nDCG uses the grades.
    hit@k        share of questions with at least one relevant passage in the top k
    precision@k  relevant passages in the top k / k (an empty slot counts as not relevant)
    recall@10    relevant passages in the top 10 / min(relevant in the pool, 10)
    mrr@10       1 / rank of the first relevant passage (0 if none in the top 10)
    ndcg@10      graded gain (2, 1, 0), normalised by the best ordering of the pool
Off-topic questions (no answer in the corpus) are scored separately: the share for which a system returns nothing.
Results: evals/retrieval_results.json, with 95% bootstrap intervals over questions."""
import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

K = 10
SYSTEMS = ("memory", "vector", "fused", "library")
JUDGE_PROMPT = """You grade search results for a World Bank procurement analyst.

Question: {q}

Passage (from a {doc_type} report, project {project_id} "{project_name}", {country}):
\"\"\"{text}\"\"\"

Grade how useful this passage is as evidence for the question:
2 = relevant: it directly gives evidence that answers the question (if the question names a project, country or place, the passage must be about it)
1 = partial: on the topic but generic, about a different project/country than the one named, or only a passing mention
0 = not relevant

Reply with JSON only: {{"grade": 0|1|2, "reason": "<at most 15 words>"}}"""


def load_platform(snapshot):
    from procintel import Platform
    from procintel.lakehouse import LocalLake
    return Platform.from_lake(LocalLake(snapshot), None)        # models are refitted; only the passages and index matter here


def run_systems(plat, vs_index, queries):
    from procintel import services
    from procintel.vectorstore import FusedRetriever, RestVectorIndex, VectorSearchRetriever
    mem = plat.index
    vs = VectorSearchRetriever(RestVectorIndex(vs_index), mem.chunks) if vs_index else None
    fused = FusedRetriever(mem, vs)
    state = {"plat": plat}
    out = {}
    for qd in queries:
        q, res = qd["q"], {}
        res["memory"] = [c.chunk_id for c, *_ in mem.search(q, k=K)]
        res["vector"] = [c.chunk_id for c, *_ in vs.search(q, k=K)] if vs else []
        res["fused"] = [c.chunk_id for c, *_ in fused.search(q, k=K)]
        if vs and fused.last_backend != "vector search + in-memory":
            print(f"  warning: {qd['id']} fused search fell back to memory only")
        plat.index = fused
        res["library"] = [r["chunk_id"] for r in services.search(state, q, k=K)["results"]]
        plat.index = mem
        out[qd["id"]] = res
        print(f"  searched {qd['id']}: " + ", ".join(f"{s} {len(v)}" for s, v in res.items()), flush=True)
    return out


def judge_all(plat, queries, runs, labels, model, save_to):
    from procintel.llm import shared_client
    client, by_id = shared_client(), plat.chunks_by_id
    names = plat.ds.projects.set_index("project_id").project_name.to_dict()
    todo = [(qd, cid) for qd in queries if qd["kind"] != "offtopic"
            for cid in dict.fromkeys(c for s in SYSTEMS for c in runs[qd["id"]][s]) if f"{qd['id']}|{cid}" not in labels]
    print(f"judging {len(todo)} new (question, passage) pairs with {model}")

    def one(item):
        qd, cid = item
        c = by_id[cid]
        text = " ".join(c.text.split())[:2500]
        msg = JUDGE_PROMPT.format(q=qd["q"], doc_type=c.doc_type, project_id=c.project_id, project_name=names.get(c.project_id, ""),
                                  country=c.country, text=text)
        for attempt in range(3):
            try:
                content = client.chat(model, [{"role": "user", "content": msg}], max_tokens=120, temperature=0.0).get("content") or ""
                m = re.search(r"\{.*?\}", content, re.S)                # the first object (the model sometimes adds a second)
                d = json.loads(m.group(0))
                g = int(d["grade"])
                if g in (0, 1, 2):
                    return f"{qd['id']}|{cid}", dict(grade=g, reason=str(d.get("reason", ""))[:200], by=model)
            except Exception as e:                                  # malformed reply or transient error: retry
                err = e
                time.sleep(1 + attempt)
        print(f"  could not judge {qd['id']} {cid}: {err}")
        return None

    t = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:          # pay-per-token endpoints throttle; 4 in flight stays under the limit
        for n, r in enumerate(pool.map(one, todo), 1):
            if r:
                labels[r[0]] = r[1]
            if n % 50 == 0 or n == len(todo):
                save_labels(labels, save_to)                # a stopped run resumes from here
                rate = n / (time.time() - t)
                print(f"  {n}/{len(todo)} judged, {rate:.1f}/s, ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)
    return labels


def save_labels(labels, path):
    json.dump(dict(about="Relevance grades, key 'query_id|chunk_id'. 2 relevant, 1 partial, 0 not relevant. 'by' is the judge model "
                         "or 'human' for a hand-checked grade.", labels=dict(sorted(labels.items()))),
              open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def per_query(ranked, grades):
    """grades: chunk_id -> 0/1/2 for the whole pool of this question."""
    rel = {c for c, g in grades.items() if g == 2}
    top = ranked[:K]
    g = [grades.get(c, 0) for c in top]
    first = next((i for i, c in enumerate(top) if c in rel), None)
    dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g))
    ideal = sorted(grades.values(), reverse=True)[:K]
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal))
    return dict(hit_at_5=float(any(c in rel for c in top[:5])), hit_at_10=float(first is not None),
                precision_at_5=sum(c in rel for c in top[:5]) / 5, precision_at_10=sum(c in rel for c in top) / K,
                recall_at_10=sum(c in rel for c in top) / min(len(rel), K) if rel else None,
                mrr_at_10=1 / (first + 1) if first is not None else 0.0, ndcg_at_10=dcg / idcg if idcg else None,
                partial_hit_at_5=float(any(grades.get(c, 0) >= 1 for c in top[:5])), returned=len(top))


def summarise(rows, seed=7, n_boot=2000):
    rng = np.random.default_rng(seed)
    out = {}
    for m in rows[0]:
        v = np.array([r[m] for r in rows if r[m] is not None], dtype=float)
        if not len(v):
            continue
        boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
        out[m] = dict(mean=round(float(v.mean()), 3), ci95=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                      n=int(len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--vs-index", default=os.getenv("PROCINTEL_VS_INDEX", "workspace.dev_jaeejain99_procintel.chunks_index"))
    ap.add_argument("--judge", default="databricks-llama-4-maverick")
    ap.add_argument("--queries", default=os.path.join(HERE, "retrieval_queries.json"))
    ap.add_argument("--labels", default=os.path.join(HERE, "retrieval_labels.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "retrieval_results.json"))
    ap.add_argument("--only", default="", help="comma-separated question ids to run ('offtopic' adds the off-topic questions)")
    a = ap.parse_args()

    queries = json.load(open(a.queries, encoding="utf-8"))["queries"]
    if a.only:                                                   # a quick run on a subset of the questions
        keep = set(a.only.split(","))
        queries = [q for q in queries if q["id"] in keep or (q["kind"] == "offtopic" and "offtopic" in keep)]
    labels = json.load(open(a.labels, encoding="utf-8"))["labels"] if os.path.exists(a.labels) else {}
    t = time.time()
    plat = load_platform(a.snapshot)
    print(f"loaded {len(plat.index.chunks)} passages in {time.time() - t:.0f} s")
    runs = run_systems(plat, a.vs_index, queries)
    labels = judge_all(plat, queries, runs, labels, a.judge, a.labels)

    answerable = [q for q in queries if q["kind"] != "offtopic"]
    result = dict(run_at=time.strftime("%Y-%m-%d %H:%M"), snapshot_version=open(os.path.join(a.snapshot, "_version.txt")).read().strip()
                  if os.path.exists(os.path.join(a.snapshot, "_version.txt")) else None,
                  corpus=dict(passages=len(plat.index.chunks), documents=len({c.doc_id for c in plat.index.chunks})),
                  vector_index=a.vs_index, judge=a.judge, k=K, questions=len(answerable), offtopic_questions=len(queries) - len(answerable),
                  pooled_judgements=sum(1 for k in labels if k.split("|")[0] in {q["id"] for q in answerable}), systems={}, by_kind={},
                  no_relevant_in_pool=[q["id"] for q in answerable
                                       if not any(labels.get(f"{q['id']}|{c}", {}).get("grade") == 2 for s in SYSTEMS for c in runs[q["id"]][s])])
    for s in SYSTEMS:
        rows, kinds = [], {}
        for q in answerable:
            grades = {k.split("|", 1)[1]: v["grade"] for k, v in labels.items() if k.startswith(q["id"] + "|")}
            r = per_query(runs[q["id"]][s], grades)
            rows.append(r); kinds.setdefault(q["kind"], []).append(r)
        result["systems"][s] = summarise(rows)
        result["by_kind"][s] = {k: {m: v["mean"] for m, v in summarise(r).items()} for k, r in kinds.items()}
        off = [q for q in queries if q["kind"] == "offtopic"]
        result["systems"][s]["offtopic_returns_nothing"] = round(sum(not runs[q["id"]][s] for q in off) / len(off), 3) if off else None
    result["runs"] = runs
    json.dump(result, open(a.out, "w", encoding="utf-8"), indent=1)

    print(f"\n{len(answerable)} questions, {result['pooled_judgements']} graded passages, judge {a.judge}")
    print(f"{'metric':<18}" + "".join(f"{s:>12}" for s in SYSTEMS))
    for m in ("hit_at_5", "hit_at_10", "precision_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "partial_hit_at_5"):
        print(f"{m:<18}" + "".join(f"{result['systems'][s].get(m, {}).get('mean', float('nan')):>12.3f}" for s in SYSTEMS))
    print(f"{'offtopic: nothing':<18}" + "".join(f"{result['systems'][s]['offtopic_returns_nothing']:>12.2f}" for s in SYSTEMS))
    print(f"no relevant passage found by any system: {result['no_relevant_in_pool']}")


if __name__ == "__main__":
    main()
