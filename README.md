# Procurement Intelligence Platform: working PoC

Runs on real data pulled from the World Bank APIs on Databricks.
`pip install -r requirements.txt && pytest -q`
Proposal and phased rollout plan with KPI targets: [docs/PROPOSAL.md](docs/PROPOSAL.md). Agent benchmark: `python -m procintel.benchmark --data data/live` (needs a Databricks login for the models).

| Module | Where | What it does |
|---|---|---|
| 1 Ingestion & KG | `ingest.py`, `graph.py`, `index.py` | CSV/JSON/MD(/PDF) ETL, schema checks, publication-lagged as-of join of WGI/CPIA (no look-ahead), supplier entity resolution (normalise, block, fuzzy match), heading-aware chunking (tables kept separate, context prefix, page + section path), NetworkX KG with typed, confidence-weighted `EVIDENCE` edges (`id_mention` 1.0, `same_project_section` 0.8, lazy `sector+country+time` ≤0.5), hybrid TF-IDF + LSA retrieval with RRF, metadata filters and soft boosts |
| 2a Prediction | `models.py` | XGBoost survival model (`survival:aft`) for tender-to-award days: signed contracts exact, still-open tenders right-censored (as of each cut-off, so no look-ahead); P10-P90 from residual spread + split-conformal; feature set by nested temporal validation + one-SE rule; calibrated GBM classifiers for re-bid and amendment risk; ablation explanations |
| 2b Agentic retrieval | `agent.py`, `llm.py`, `rag.py` | Conversational agent on Databricks Model Serving (`gpt-oss-120b`): a deterministic router (`planner.py`) decides the tools every question needs (document search, forecast + comparables, project records, early warnings) and runs them in parallel before the model is called; the model gets a compact evidence pack (top sentences per source, ~5k chars), may add one round of tools, then answers citing tool ids. Answers stream over SSE (`/api/ask/stream`), one sentence at a time and only after that sentence passes the guardrails, so no unverified text reaches the browser; remembers the last 4 verified turns for follow-ups; deterministic policy screen for refusals; the fixed pipeline (`rag.py`: predict, analogues, six risk themes, graph expansion) is the fallback and the form path |
| 3 Guardrails | `guardrails.py` | Per sentence/finding: L1 citation (and cited ids must come from this session), L2 verbatim quotes, L3 numbers in the cited sources (unrounded), L4 LLM judge on Model Serving (`llama-3.3-70b`, lexical fallback), L5 country scope, L6 abstention; ISR-vs-ICR rules R1-R4 (1-notch = minor) plus measured ISR optimism |
| Benchmark | `benchmark.py`, `evals/benchmark.json` | 34 items (answer / no-evidence / must-refuse, incl. multi-turn follow-ups); pass rate, verifier removals by layer, agent vs fixed pipeline; runs in notebook 07, logs to MLflow + `gold_benchmark`, and fails the job below 90% or on any answered must-refuse |
| Monitoring | `monitoring.py` | Bias-adjusted ISR trend, disbursement vs historical P25, open tenders beyond predicted P90 |
| 4 Learning/KPIs | `learning.py` | Idempotent batch upsert, incremental index update with OOV/growth-triggered refit, PSI + error drift, champion/challenger with recency weights, feedback priors, P@5 / R@10 / MRR |

## Reference run, synthetic data (seed 7)
- Entity resolution: 191 raw supplier strings → 16 entities. Index: 2,679 chunks; KG: 7,237 evidence edges.
- Duration MAE 51 d vs 68 d category-median baseline; P10-P90 coverage 89% (nominal 80%). Re-bid AUC is only 0.60 (weak, honestly reported).
- Retrieval on planted themes: P@5 0.68, R@10 0.66, MRR 0.82. After the first batch update, P@5 0.95, R@10 0.93, MRR 1.00 (larger corpus, so not a like-for-like gain).
- A drifted batch (+35% tender times): champion MAE 109 d → drift flagged → challenger 94 d on holdout → promoted → the schedule recommendation shifts.
- Adversarial fabricated findings are all rejected by the guardrails.

## Real World Bank data (Kenya, Ghana, Nepal, Bangladesh, Vietnam; pulled 2026-11-02)
`python -m procintel.live_wb --out data/live --countries KE,GH,NP,BD,VN --max-projects 100 --max-docs 300` then `python run_live.py data/live`
(or `python app.py --data data/live`). Sources: Projects API, `contractdata`, `procnotices`, WDS (ICR / ISR / PAD text), WGI (source 3), CPIA (source 31).

| What | Result |
|---|---|
| Records | 2,815 contracts, 1,674 supplier strings → 1,604 entities, 300 documents (31 ICR, 46 PAD, 223 ISR; no PPARs exist for these projects) |
| Parsing | ISR ratings from 98% of ISRs; ICR outcome rating from 31/31 ICRs (three real layouts, tested in `tests/test_live_mapping.py`) |
| Chunks | 28,907 (median 94 words). Hard-wrapped WB text is re-joined into sentences/paragraphs before chunking |
| Duration model, XGBoost survival (holdout = latest 25% of notices) | Lakehouse, 6 countries, model v4 (STEP activity matching + Procurement Plans + 5-member ensemble, 9 Nov 2026): **MAE 76.6 d vs 99.0 d** category-median baseline (23% better; a tie, 108 vs 107, before STEP matching); P10-P90 coverage 78% (nominal 80%); stable across random seeds (spread under 1.5 d); trained on 785 signed contracts + 292 open tenders (censored) |
| Re-bid model | AUC 0.65, Brier 0.246 vs 0.262 base rate. The amendment label is all-zero in this sample, so it is a constant base rate, not a model |
| Agent benchmark (34 items) | Databricks job run 4 Nov 2026: 34/34 passed, must-refuse 10/10, median answer 4.2 s; the verifier removed 25% of draft sentences (35 of 141) before anyone saw them. See `evals/results_databricks.json` |
| ISR vs ICR | 21 projects have both: 17 agree, 4 differ by one notch (reported as minor R4 conflicts) |

What the real data forced, each now in the code:
- **Labels are proxies.** contractdata has only a signing date, so duration = signing − matched tender notice. Contracts are linked to their notices by, in order, the STEP activity number in both feeds' references, the exact reference, then description similarity (`match_notices`). Still only 30% of contracts link: 1,463 carry an agency package number instead of a STEP reference, and 418 were never publicly advertised (direct selection, individual consultants, small RFQs). Re-tender / amendment labels are kept only for STEP-activity or exact-reference links: description links had labelled 58% of contracts 're-tendered' against under 1% for reliable links, so the re-tender model now has too few real cases (about 5) to report an AUC.
- **Procurement Plans** (`procplan.py`): the public STEP export of each project, parsed from the PDF by word position (the text rendition drops empty cells). 53 of 108 projects publish one (55 never did, mostly older closed projects); 2,801 activities with planned and actual dates per stage. Each activity's dates are checked against the text rendition: 89% match exactly, the rest are not used. Against the notice feed, the plan's actual notice date was identical for 206 of 206 contracts that have both. Plans add start dates for 99 contracts the notices never linked (mostly Requests for Quotations; Direct Selection is excluded, it is not a tender) and the Portfolio monitor's 'behind its own plan' warning (planned signing passed by 30 days to 2 years, plan shows it in progress, plans published in the last 12 months). Fuzzy matches below 0.8 ran about 2× longer than exact-reference matches (wrong notice), so they are dropped; notices from the last 12 months are dropped too (right-censoring: only fast tenders have signed yet).
- **Small data needs a small model.** Depth-3 trees with small leaves made the backtest swing from 116 to 1,038 days MAE with the random seed alone; depth 2 with 20-contract leaves is stable (108-109 d on every seed) and a test guards it.
- **Open tenders are censored, not dropped.** The survival model keeps tenders that have not signed yet. They lack a contract value and review type, which only an award reveals; left missing, that gap taught the model "missing => decades" (a 56-year forecast, caught by the benchmark), so they are filled from random contracts of the same category.
- **Feature set chosen by nested temporal validation with the one-standard-error rule.** On ~550 labelled contracts from ~60 projects, sector/country/governance features identify individual projects; the full model scored 199 d MAE against 102 d for the simple one. The rule costs 10 d on the synthetic set (40 → 50 d) and is what makes the real model beat its baseline.
- **Relevance gate is corpus-relative** (z-score against all chunks). An absolute cut-off tuned on 3k synthetic chunks rejected every good lesson on 29k real chunks.
- Live API quirks: CPIA keys countries by ISO3 (WGI by ISO2), WDS adds a `facets` entry, `sector1` is often blank, certifi is needed for TLS on some Windows hosts, and a transient DNS failure no longer aborts a pull (reruns reuse downloads).

## Honest limits
- Local retrieval is TF-IDF + LSA, i.e. lexical. On real ICRs it finds relevant lessons for some themes and abstains on others; by design it prefers silence to an off-topic citation. On Databricks, retrieval uses Vector Search with `databricks-gte-large-en` embeddings; a cross-encoder reranker is the next step.
- The sample is small (5 countries, ~550 usable duration labels). Governance features (WGI/CPIA) are in the pipeline, but the selection rule drops them here; they need more countries to carry signal.
- Disbursement feed: no URL was provided, so it is empty and the disbursement early warning is skipped. Amendment and re-bid labels are weak proxies from notice history.
- The extractive composer is the default. The optional LLM composer (`--llm`, `ANTHROPIC_API_KEY` or `DATABRICKS_LLM_ENDPOINT`) is untested, but its output goes through the same guardrails. The L4 support check is lexical; use an NLI model for paraphrases.
- PDF parsing (`pypdf`) is implemented, but the real pull uses the WDS text renditions; scanned PDFs would need a layout-aware parser.

## Databricks
The bundle is at the project root (`databricks.yml`), uses serverless notebooks, and works on Free Edition (default catalog `workspace`).
```
databricks bundle deploy
databricks fs cp -r data/live dbfs:/Volumes/workspace/<schema>/landing/base     # dev-mode schema: dev_<user>_procintel
databricks bundle run procintel_pipeline
databricks bundle run procintel_ui                                              # starts the Databricks App
```
```
UC Volume landing/base ──► bronze_* (Delta, append, _ingest_ts, doc content-hash de-dupe)                         01
   └► silver_* (ER'd contracts, as-of WGI/CPIA, ISR, projects) + silver_chunks (re-chunked on text/chunker change) 02
        └► gold_contract_features ──► MLflow (with signature) ──► UC registry @champion/@challenger            03
        └► gold_chunks (full corpus) + gold_chunks_vs (decision-relevant subset, CDF) ──► Vector Search index   04
        └► gold_kg_edges, gold_conflicts, gold_alerts                                                       02/05
   notebook 05 exports a snapshot (tables + champion model) to landing/snapshot ──► Databricks App (app.py)
Job procintel-continuous-learning: file arrival in landing/incoming ──► bronze→gold ──► drift check ──► challenger ──► promote ──► index sync   06
```
| Concern | Databricks feature | Code |
|---|---|---|
| Storage, lineage, access | Unity Catalog Delta tables, Volumes | `lakehouse.DeltaLake` (explicit Spark schema, so empty tables work) |
| Incremental, idempotent loads | Delta `MERGE` / `DELETE`, content hash + `CHUNKER_VERSION` | `pipeline.bronze_ingest / silver_gold` |
| Retrieval | Vector Search delta-sync index on `gold_chunks_vs` via Change Data Feed | `vectorstore.py` |
| Models | MLflow + UC registry aliases | `mlops.MlflowModelStore`, `learning.retrain(store=)` |
| Orchestration, closed loop | Asset Bundle jobs, file-arrival trigger | `databricks.yml` |
| UI | Databricks App reading the Volume snapshot with its service principal | `app.py`, `apps:` in `databricks.yml` |

**Verified on a real workspace (Free Edition, 2026-11-02/03):** bundle validate and deploy (schema, volume, 2 jobs, app); the full `procintel-pipeline` job (bronze, silver/gold, MLflow training + Unity Catalog registration with the champion alias, Vector Search index on `gold_chunks_vs` synced to 3,313 rows, alerts + snapshot export); the Databricks App serving the snapshot (checked through its API: 5 countries, 28,907 chunks, champion metrics, a cited risk finding, 4 conflicts).
**Verified locally:** `MlflowModelStore` against a real MLflow registry (sqlite: signature, alias, pyfunc load), the snapshot → app hand-off, and the whole medallion flow on `LocalLake` (`tests/test_databricks_layer.py`).
**Not yet verified:** the continuous-learning job (file-arrival trigger) on the workspace; notebook 05 ran its Copilot smoke test through Vector Search, but those answers were not reviewed for quality.
**Deployment gotchas found and fixed:** Spark Connect's `toPandas()` puts pyspark objects in `DataFrame.attrs` (stripped before pickling); the App image ships numpy 1.x while serverless notebooks use 2.x, so numpy/pandas/scikit-learn are pinned to identical versions on both sides; a new delta-sync index rejects `sync()` until its first build ends.
**Free Edition constraints met in practice:** one Vector Search endpoint (the bundle reuses the existing one via `vs_endpoint`); managed embedding runs at about 70 chunks/min, hence the `gold_chunks_vs` subset (~3.3k chunks; the full 29k corpus would take ~7 h); serverless only (dependencies via `%pip`, package imported from the synced bundle files, no wheel).
At full scale: Spark-native entity resolution (e.g. Splink), Lakeflow Declarative Pipelines for silver, GraphFrames for the KG.

## Web UI (React + FastAPI)
`python app.py --data data/live` then open http://127.0.0.1:8000. The FastAPI backend (`app.py`, logic in `procintel/services.py`) loads data and models in the background; `/api/health` reports loading / ready / error and the UI waits on it. API docs: `/api/docs`.
The React UI lives in `web/` (Vite). It is pre-built into `web/dist`, which FastAPI serves, so the Databricks App needs only Python. After UI changes: `cd web && npm install && npm run build` (or `npm run dev` with `python app.py` running, for hot reload).

| Tab | For | What it does |
|---|---|---|
| Home | Everyone | What each tab is for and how to use it |
| 1 Package workspace | Procurement specialists | Schedule range (survival model), market-approach options (procurement methods compared), risks with mitigations written from the cited lesson |
| 2 Portfolio monitor | Task team leaders, managers | Active projects ranked by risk with the reason for every flag; project panel with ISR history, open tenders against their predicted P90, rating conflicts, lessons |
| 3 Ask | Anyone | Conversational agent with verified, cited answers and follow-ups; can start from a package |
| 4 Evidence library | Reviewers, specialists | Search report passages (quotes only; filters for document type, section and country; a relevance threshold so off-topic searches return "No results found"). "Read in context" opens the whole report page by page at the passage, highlighted with the search words marked; "Original PDF, page N" opens the official World Bank PDF at that page (`procintel/doc_links.json`, from the WDS API). Page numbers are real PDF pages, checked against the PDFs (59 of 59 sampled passages). Every citation elsewhere opens here |

Robustness: pydantic-validated requests with clear 400/422 errors, concurrent reads (only conversation and feedback state is locked), JSON errors everywhere, usage events that never fail a request, and an end-to-end browser walk-through of every tab (headless Chrome) before deploy.
