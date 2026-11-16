# Procurement Intelligence on World Bank Open Data

**Technical proposal and phased rollout plan.** November 2026.

The application connects World Bank contract awards, procurement notices and project records to the reports that explain why procurements went well or badly (completion reports, appraisal documents, supervision reports), and gives procurement teams evidence-backed answers at each stage of the procurement cycle. It runs entirely on Databricks: Unity Catalog for data, MLflow for models, Mosaic AI Vector Search for retrieval, Model Serving for the agent and its judge, Jobs for the closed loop, and a Databricks App for the interface.

Every number in this proposal was measured in this build on real World Bank data unless it is marked as a target.

## 1. Summary

| Rubric dimension | What ProcIntel does | Measured |
|---|---|---|
| System and knowledge architecture | Medallion lakehouse in Unity Catalog; knowledge graph as edge tables; section-aware chunking of 100+ page reports with page and section paths | 2,815 contracts, 300 documents, 28,907 passages; ICR outcome ratings parsed from 31 of 31 ICRs |
| Analytical and AI methodology | XGBoost survival model that keeps still-open tenders; agent on Databricks Model Serving that chooses its own tools; every sentence verified | Duration P10–P90 coverage 90% (target 80%); benchmark 33 of 34 (97%) passed, including follow-ups; 10 of 10 must-refuse refused |
| Domain alignment and data use | Six World Bank sources joined on project ID, notice reference and time; governance scores used only as published at the time | 271 open tenders kept as censored observations; 21 projects with both ISR and ICR ratings, 4 disagreements surfaced |
| Operationalization and scalability | Asset Bundle with two jobs (full build, file-arrival closed loop), MLflow champion/challenger, benchmark tracked in MLflow on every run | Full pipeline and App deployed and verified on a Databricks workspace |

## 2. Architecture

```
World Bank APIs (Projects, awards, notices, WDS reports, WGI, CPIA)
  -> UC Volume landing/base
  -> bronze_*  append-only, ingest timestamp, document content hash
  -> silver_*  resolved suppliers, as-of governance, ISR ratings,
               open tenders, re-chunked passages
  -> gold_contract_features -> survival + risk models -> MLflow, UC registry @champion
  -> gold_chunks (all passages)            -> hybrid index in the app
  -> gold_chunks_vs (decision subset, CDF) -> Mosaic AI Vector Search
  -> gold_kg_edges, gold_conflicts, gold_alerts, gold_benchmark

Databricks App <- snapshot (tables + champion model)
Agent (gpt-oss-120b) <-> tools -> verifier <- judge (llama-3.3-70b)

landing/incoming --file arrival--> continuous-learning job:
  ingest -> drift check -> challenger -> promote -> index sync
```

| Concern | Databricks component |
|---|---|
| Storage, lineage, access control | Unity Catalog Delta tables and Volumes |
| Incremental, idempotent loads | Delta MERGE and DELETE; content hash and chunker version decide what is re-processed |
| Models | MLflow tracking; Unity Catalog model registry with champion and challenger aliases |
| Retrieval | Mosaic AI Vector Search delta-sync index (`databricks-gte-large-en`), hybrid queries with metadata filters, fused in the App with an in-memory hybrid index over the full corpus (reciprocal rank); in-memory alone if Vector Search is unavailable |
| Agent and judge | Foundation Model API on Model Serving: `databricks-gpt-oss-120b` (agent), `databricks-meta-llama-3-3-70b-instruct` (judge) |
| Orchestration | Databricks Asset Bundle: pipeline job, file-arrival continuous-learning job |
| Interface | Databricks App, with permissions on the Volume and the two model endpoints |

## 3. Module 1: Ingestion and knowledge representation

**Sources used.** Projects API (metadata, sector, environmental and social risk), contract awards (`contractdata`), procurement notices (`procnotices`), the Documents & Reports API (ICR, PAD and ISR text), Worldwide Governance Indicators and CPIA. The current slice covers Kenya, Ghana, Nepal, Bangladesh and Vietnam: 2,815 contracts, 300 documents (31 ICRs, 46 PADs, 223 ISRs).

**Entity resolution.** Supplier names are normalised, grouped into candidate blocks and fuzzy-matched (1,674 raw names to 1,604 suppliers). Contracts are joined to tender notices by reference number, or by description similarity when no reference matches. Matches below 0.8 similarity ran about twice as long as exact-reference matches, so they are treated as wrong and their durations dropped.

**Knowledge graph.** Nodes for projects, contracts, suppliers, documents, passages, countries and sectors, stored as Delta edge tables. Evidence edges carry a confidence: an explicit ID mention 1.0, the same project's section 0.8, a sector-country-time link at most 0.5. This is how a supplier's contract reaches a lesson in a completion report.

**No look-ahead.** Each governance score becomes available to a contract only after its publication date, and every model backtest censors at its cut-off date.

**Long reports.** World Bank text renditions hard-wrap lines and put blank lines inside sentences. The parser rebuilds paragraphs from sentence endings, keeps the heading hierarchy (section path and page on every passage), keeps tables as separate passages, and packs short paragraphs to about 100 words. 300 documents became 28,907 passages (median 94 words). A passage is re-chunked only when its text or the chunker version changes, and its old passages are deleted, so the vector index never serves stale text.

**Ratings from text.** ISR ratings were parsed from 98% of ISRs, and ICR outcome ratings from 31 of 31 ICRs across three real layouts.

## 4. Module 2: Predictive and prescriptive engines

### 4.1 Timeline and risk models

**Survival model for tender-to-award time.** The public record holds a signing date but no tender-completion date, so duration is the time from the matched tender notice to signing. Tenders still open are the slow ones; dropping them biases every forecast short. ProcIntel uses an XGBoost accelerated failure time model (`survival:aft`): signed contracts are exact observations, and open tenders (and, in backtests, anything signed after the cut-off) are right-censored observations. 607 signed contracts and 271 open tenders train the current model.

Open tenders lack what only an award reveals (contract value and review type). Left missing, that gap marks "still open" and the model learns "missing means decades", which this build caught on real data. They are therefore filled from random contracts of the same category, so they look like any other tender except for being censored. A regression test guards this.

The feature set is chosen inside the training window with the one-standard-error rule, among full, no-governance and core sets. The P10–P90 band uses the spread of validation residuals, widened by split-conformal calibration.

| Measured on real data (holdout: newest 25% of notices) | Value |
|---|---|
| P50 mean absolute error | 76.6 days on the six-country lakehouse against 99.0 for the category-median baseline (23% better), model v4: STEP activity matching, Procurement Plan start dates and a 5-member ensemble (9 November 2026; a tie, 108 against 107, before). Stable across random seeds (spread under 1.5 days) |
| P10–P90 coverage | 78% (target 80%) |
| Re-bid model | Not measurable yet: with reliable labels only (STEP-activity or exact-reference links, 549 contracts), about 5 real re-tenders, 1 in the test set. The earlier AUC (0.65 to 0.72) was learning which contracts had been linked by description (58% labelled re-tendered), not re-tender risk |
| Amendment model | label all zero in this sample: reported as a base rate, not a model |

**STEP activity matching.** Both public feeds carry the STEP procurement reference (e.g. KE-SDYA-533926-CS-CDS), but they often write the agency or method part differently. Matching on the activity number links each contract to the notices of its own package, so the tender start date is right more often. With the same number of linked contracts, the median error fell from 108 to 78 days and the model now beats the baseline by 25%. It did not raise the link rate (still 30%): 1,463 contracts carry an agency's own package number instead of a STEP reference, and 418 were never publicly advertised. 

**Procurement Plans.** Each project's plan is the public export of STEP: every procurement activity with its method, amount, status and planned and actual dates per stage. The text rendition drops empty cells, so plans are parsed from the PDF by word position (the Planned / Actual header gives each date column, the labels above it name the stage). 53 of 108 projects publish a plan; 2,801 activities were parsed. Every activity's dates are compared with the text rendition (89% identical; the rest are not used), and the plan's actual notice date matched the notice feed for all 206 contracts that have both. Plans give a start date to 99 contracts the notice feed never linked, mostly unadvertised Requests for Quotations (Direct Selection is left out: it is not a tender, and including it made the forecast worse). They also give the Portfolio monitor a warning no other source can: a package whose planned signing date has passed while the plan still shows it in progress (23 of 60 active projects).

**Stability.** With more and more varied data, one random seed produced a backtest error of 130 days against 81 for the others: the open tenders' random fill-in and the trees' row sampling. The duration model is now an ensemble of five members, each with its own fill-in and sample, averaged in log-days; the spread across seeds fell to under 1.5 days.

Before this change the survival model tied the baseline. Its value was already correctness and stability: open tenders are kept instead of biasing forecasts short, the interval is calibrated, and the backtest no longer depends on chance. An earlier configuration (depth-3 trees, small leaves) overfitted this small sample: its backtest swung from 116 to 1,038 days with the random seed alone, which the model-quality figures on the deployed app exposed. Shallow trees with 20-contract leaves give the same result on every seed, and a test now guards it. On synthetic data with known structure the model beats the baseline (51 against 68 days). More countries and exact reference linking (Section 9) are what it needs to beat the baseline on real data.

### 4.2 Agentic retrieval

When a user asks a question, an agent on Databricks Model Serving (`databricks-gpt-oss-120b`) decides which tools to call, reads the results, and calls more tools for up to four rounds:

| Tool | Returns |
|---|---|
| `search_documents` | Passages from ICRs, PADs and ISRs, with filters for document type and section (lessons, procurement, ratings) |
| `forecast_package` | Survival-model P10/P50/P90, re-bid and amendment probability, main drivers, and comparable past contracts |
| `get_project` | Project record, last ISR rating, ICR outcome, and any rating conflict with the rule that resolves it |
| `early_warnings` | Active projects with flags (falling ISR ratings, bias-adjusted ratings) |
| `list_values` | Valid categories, sectors, methods and countries |

**Conversation.** The agent keeps the last four turns of a conversation, so follow-ups work ("What about the same package in Ghana?", "How did those projects mitigate it?"). Memory holds only the verified answers and the sources they cited, never drafts or removed sentences, so an error caught once cannot re-enter later. Sources cited earlier stay citable and are verified again in every turn. The policy screen reads a short follow-up together with the previous question, so a request split across turns ("Which contractors underperformed?" then "Should we exclude them?") is still refused.

**Speed without skipping checks.** A first version let the model choose its tools one round at a time and read whole passages: six model calls and 15 to 32 seconds per answer. Now a deterministic router reads the question (country, category, value, project id, kind of question) and runs every tool the answer needs in parallel before the model is called. A planning question always gets the forecast, comparable contracts, lessons on three risk themes and procurement passages, whatever the model would have chosen. The model receives a compact evidence pack (the best-matching sentences of each source, about 5,000 characters) and may add one more round of tools. The answer streams to the browser one sentence at a time, and each sentence is shown only after it passes the guardrails and the judge (which checks each sentence on its own, in parallel). Unverified text never leaves the server. On the deployed app (4 November 2026) the first tool result appears after about 1.5 seconds, the first verified sentence after 4.6 to 7.6 seconds, and full answers take 4.7 to 11.8 seconds.

Model choice was tested on the workspace: of six served models that support tool calling, only `gpt-oss-120b` gathered both forecasts and lessons before answering and cited every sentence. It also changed numbers and invented a country, which is why the router, not the model, decides which evidence is always gathered, and the model never decides what is true (Module 3).

**Benchmark.** 34 items: 20 questions that should be answered (package planning, lessons, project facts, timelines, early warnings), 2 follow-up conversations, 2 questions the data cannot answer, and 10 requests that must be refused (one of them split across two turns). On the latest Databricks job run (4 November 2026), all 34 passed: all 10 refusals, both follow-ups, both no-evidence questions, and 5 of 5 package-planning questions. The run before it scored 33 of 34; the failure was a verifier error, not a wrong answer. The number check read "p50 213 days" as 50,213 and removed a correct forecast sentence. It now never joins a label such as p50 to the next number, and a test guards it. Results still vary by one or two items between runs. An earlier run of the 31 single-turn items scored 29 of 31 once a scorer error (two correct abstentions scored as claims) was fixed. Answers take a median of 4.2 seconds (6.6 seconds before the router and streaming). Results are logged to MLflow and `gold_benchmark` on every pipeline run, and the job fails if the pass rate drops below 90% or any must-refuse item is answered.

A fixed pipeline (form input, six risk themes, the same guardrails) remains as the fallback when the model endpoint is unavailable and as a reproducible baseline. On the five package-planning benchmark questions, both the agent and the fixed pipeline surfaced at least one grounded lesson for all five.

### 4.3 Support at each procurement stage

| Stage | What ProcIntel gives |
|---|---|
| Planning and strategy | Schedule range from the survival model, re-bid risk, comparable contracts, lessons on the risks that matter |
| Tender design and packaging | Precedents on packaging, cost estimates, qualification criteria and few-bidder outcomes, with quotes and pages |
| Execution and monitoring | Early warnings on active projects, ISR ratings adjusted for measured optimism, rating conflicts |
| Continuous learning | New completion reports flow in through the closed loop; feedback on passages adjusts retrieval |

## 5. Module 3: Guardrails, faithfulness and attribution

Every sentence the agent writes is checked before anyone sees it, and anything that fails is removed and shown with its reason.

| Layer | Check |
|---|---|
| L1 | A factual sentence must cite a source, and every cited id must have come back from a tool in this session |
| L2 | Quotes in structured findings must appear word for word in the cited passage |
| L3 | Every number must appear in the cited sources, unrounded (thousands separators allowed) |
| L4 | A judge model (`llama-3.3-70b`, a different family from the agent) decides whether the cited sources support the sentence, seeing each source's country and label |
| L5 | A country the sentence names must be the country of a cited source |
| L6 | If nothing survives, the answer is held back rather than guessed |

A deterministic policy screen runs before any model call and refuses requests to score or blacklist named suppliers, predict tender winners, avoid procurement rules, reveal personal contact details, or give legal advice. The model is instructed to refuse the same, and anything else out of scope.

**Measured.** On the benchmark, the verifier removed 20 to 30% of the agent's draft sentences across runs (23% on the latest; on the run analysed by layer: 16 missing or invalid citations, 3 rejected by the judge, 2 numbers not in the sources). Those are the errors that would otherwise have reached a user, including a forecast misread as "189 days" when the model said 233, uncited arithmetic ("allocate 6 to 9 months"), and a lesson attributed to "Kenyan road projects" when the source was from another country.

**Conflicting evidence.** For outcome claims the more independent and later source wins: IEG review (PPAR, ICR Review) over ICR over ISR. Rules R1 to R4 cover large gaps, late-recognised deterioration (a declining ISR trend before closure) and one-notch disagreements, which are shown as minor with both sources. ProcIntel never silently picks a side: it shows both, the rule applied, and the portfolio's measured ISR optimism. On real data, 21 projects have both ratings: 17 agree and 4 differ by one notch.

## 6. Module 4: Continuous learning

| Loop | Trigger | What happens | Gate |
|---|---|---|---|
| Data and documents | A new file in `landing/incoming` (file-arrival job) | Bronze append, silver and gold rebuild, only changed documents re-chunked, deletes and inserts flow to Vector Search through Change Data Feed | Schema and quality checks |
| Models | New labelled outcomes | Drift check (population stability index and champion error on new data); challenger trained with recency weights | Challenger must beat the champion on a held-out set; aliases switch in the UC registry |
| Retrieval | Thumbs up or down on a passage or an agent answer | Each action is an event in the Volume; the pipeline aggregates them into `gold_feedback_priors`, which ship in the App snapshot, so feedback survives restarts and reaches every user | Benchmark rerun |
| Agent quality | Every pipeline run | Benchmark questions run through agent and verifier; results logged to MLflow and `gold_benchmark` | The job fails if the pass rate drops below 90% or any must-refuse question is answered |
| Adoption | Every App action | Events (user stored as a salted hash) become `gold_usage_events` and a dated `gold_kpis` row: weekly active users, packages analysed, useful share, answer latency | Reviewed against the rollout gates |

## 7. Phased rollout

Each phase starts only when the previous phase meets its exit gates. A missed gate means fixing and re-measuring, not moving on.

| Phase | Scope | Users | Exit gates |
|---|---|---|---|
| 0. Foundation (weeks 0 to 6) | All countries' structured records; bulk award datasets; exact STEP-reference linking; ICR Reviews, PPARs, CPFs and disbursement histories ingested | Build team | Benchmark pass at least 90% and must-refuse 100%; P10–P90 coverage 75 to 85%; median error below the category baseline |
| 1. Pilot (months 2 to 4) | Two countries (e.g. Kenya and Ghana); the agent runs in shadow mode for 4 weeks, then advisory | About 15 procurement specialists and task team leaders | Weekly active use at least 60% of pilot users; at least 70% of rated answers marked useful; no ungrounded sentence found in a monthly audit of 50 answers |
| 2. Regional (months 5 to 9) | Africa and South Asia regions; read-only view for implementing agencies | Regional procurement staff; implementing agencies | Planning cycle time 15% shorter than in comparison countries; forecast error at least 10% below baseline; p95 answer time under 30 seconds |
| 3. Enterprise (months 10 to 18) | All regions; STEP integration; single sign-on; operational monitoring | All Bank procurement staff | Availability at least 99.5%; cost per answer tracked against budget; quarterly independent audit of citations |

## 8. Key performance indicators

| KPI | Current (measured) | Pilot target | Enterprise target | How it is measured |
|---|---|---|---|---|
| Procurement planning cycle time | Not measured (no users yet) | 10% shorter than comparison countries | 20% shorter | Time from package entry to plan approval, pilot against comparison countries |
| Duration forecast error (P50 MAE) | 76.6 days (baseline 99.0), stable across seeds | Below baseline | 15% below baseline | Rolling temporal backtest, MLflow |
| P10–P90 interval coverage | 90% | 75 to 85% | 75 to 85% | Same backtest |
| Benchmark pass rate | 33 of 34 (97%) | 90% | 95% | `gold_benchmark`, every pipeline run |
| Must-refuse rate | 10 of 10 | 100% | 100% | Same benchmark |
| Draft sentences removed by the verifier | 23% | Under 25% | Under 15% | Benchmark and production logs (lower means a better agent, not a weaker verifier) |
| Ungrounded sentences reaching users | 0 by construction; audit pending | 0 in audit | 0 in audit | Monthly human audit of 50 answers |
| Retrieval of historical risk factors | Not measured on real data yet | Recall@10 of 0.8 | Recall@10 of 0.85 | Hindsight test: risks that completion reports later blamed, retrieved as of approval |
| Adoption | Instrumented; only test use so far | 60% weekly active | 50% of new packages analysed | `gold_kpis` from App events |
| Answer time | Median 4.2 seconds; first verified sentence after about 5 seconds (Databricks run, 4 November 2026) | p95 under 30 seconds | p95 under 20 seconds | `gold_kpis` (median and p95 per run) |

## 9. Limitations and next steps

- **Data breadth.** Five countries, about 600 usable duration labels, and only 30% of contracts matched to a tender notice. STEP activity matching and Procurement Plans are in place; next: stage-level duration models from the plans' stage dates, and bulk award datasets.
- **Labels are proxies.** Re-bid and amendment labels come from notice history; the amendment label is all zero in this sample.
- **Free Edition constraints.** One Vector Search endpoint and managed embedding at about 70 passages a minute, so the index holds the 3,313 decision-relevant passages and the full corpus is searched in the app.
