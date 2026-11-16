# Databricks notebook source
# MAGIC %md ### 04 Mosaic AI Vector Search: delta-sync index on gold_chunks_vs (Change Data Feed keeps it current)
# COMMAND ----------
# MAGIC %pip install -q networkx pypdf certifi databricks-vectorsearch numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==3.0.3
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import os, sys; sys.path.insert(0, os.path.abspath("../.."))        # bundle root (synced files) -> `import procintel`
dbutils.widgets.text("catalog", "workspace"); dbutils.widgets.text("schema", "procintel"); dbutils.widgets.text("vs_endpoint", "procintel_vs")
catalog, schema, vs_endpoint = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("vs_endpoint")
# COMMAND ----------
from procintel.vectorstore import ensure_vector_index, sync_and_wait
# source = gold_chunks_vs (decision-relevant subset; the full corpus stays in gold_chunks for the app's in-memory index)
idx = ensure_vector_index(f"{catalog}.{schema}.chunks_index", vs_endpoint, f"{catalog}.{schema}.gold_chunks_vs")
st = sync_and_wait(idx, timeout_s=2700, raise_on_timeout=False)   # first build can outlast the job: it continues server-side
print(st)
# COMMAND ----------
if st.get("ready"):
    print(idx.similarity_search(query_text="tender prices exceeded the cost estimate", columns=["chunk_id"], num_results=3,
                                filters={"doc_type": ["ICR", "PPAR"]}, query_type="HYBRID"))
