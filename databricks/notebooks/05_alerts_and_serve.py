# Databricks notebook source
# MAGIC %md ### 05 Early-warning table + Copilot smoke test (Vector Search retrieval, UC champion model) + snapshot for the App
# COMMAND ----------
# MAGIC %pip install -q networkx pypdf certifi databricks-vectorsearch numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==3.0.3
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import os, sys; sys.path.insert(0, os.path.abspath("../.."))        # bundle root (synced files) -> `import procintel`
dbutils.widgets.text("catalog", "workspace"); dbutils.widgets.text("schema", "procintel"); dbutils.widgets.text("vs_endpoint", "procintel_vs")
catalog, schema, vs_endpoint = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("vs_endpoint")
# COMMAND ----------
dbutils.widgets.text("as_of", "")
import pandas as pd
from procintel import Copilot, PackageSpec, render
from procintel.lakehouse import DeltaLake
from procintel.mlops import MlflowModelStore
from procintel.pipeline import alerts, load_platform
from procintel.vectorstore import VectorSearchRetriever, ensure_vector_index, index_ready
lake, store = DeltaLake(spark, catalog, schema), MlflowModelStore(f"{catalog}.{schema}")
index = ensure_vector_index(f"{catalog}.{schema}.chunks_index", vs_endpoint, f"{catalog}.{schema}.gold_chunks_vs")
use_vs = index_ready(index)                        # still building -> in-memory hybrid index; the snapshot never depends on it
print("retrieval backend:", "Mosaic AI Vector Search" if use_vs else "in-memory hybrid (vector index still building)")
plat = load_platform(lake, store, (lambda chunks: VectorSearchRetriever(index, chunks)) if use_vs else None)
display(spark.createDataFrame(alerts(lake, plat, dbutils.widgets.get("as_of") or str(pd.Timestamp.today().date()))))
# COMMAND ----------
c = plat.ds.contracts
top = c.groupby(["category", "sector", "country_code"]).size().idxmax()      # smoke-test a package type the data actually has
print(render(Copilot(plat).analyze(PackageSpec("Rehabilitation of rural roads", top[0], top[1], top[2], 5.0, c.procurement_method.mode()[0]))))
# COMMAND ----------
# MAGIC %md The Databricks App (app.py) has no Spark: it loads this snapshot of the gold/silver tables + the UC champion model from the Volume.
# COMMAND ----------
# MAGIC %md Usage and feedback from the App (one JSON file per event in the Volume) -> usage, feedback priors, adoption KPIs.
# COMMAND ----------
import os
from procintel.events import feedback_priors, load_events, usage_kpis
ev = load_events(f"/Volumes/{catalog}/{schema}/landing/events") if os.path.isdir(f"/Volumes/{catalog}/{schema}/landing/events") else []
print(f"{len(ev)} app events")
if ev:
    lake.write("gold_usage_events", pd.DataFrame(ev).astype({"chunk_ids": str}, errors="ignore"))
    pri = feedback_priors(ev)
    lake.write("gold_feedback_priors", pd.DataFrame(dict(chunk_id=list(pri), boost=list(pri.values()))) if pri
               else pd.DataFrame(dict(chunk_id=pd.Series(dtype=str), boost=pd.Series(dtype=float))))
    k = usage_kpis(ev); print(k)
    lake.write("gold_kpis", pd.DataFrame([k]), "append")
# COMMAND ----------
from procintel.serving import export_snapshot
print(export_snapshot(lake, store, f"/Volumes/{catalog}/{schema}/landing/snapshot"))
