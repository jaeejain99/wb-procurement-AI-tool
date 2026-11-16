# Databricks notebook source
# MAGIC %md ### 06 Continuous learning: new files -> bronze/silver/gold -> drift check -> challenger -> promote -> index sync
# COMMAND ----------
# MAGIC %pip install -q networkx pypdf certifi databricks-vectorsearch numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==3.0.3
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import os, sys; sys.path.insert(0, os.path.abspath("../.."))        # bundle root (synced files) -> `import procintel`
dbutils.widgets.text("catalog", "workspace"); dbutils.widgets.text("schema", "procintel"); dbutils.widgets.text("vs_endpoint", "procintel_vs")
catalog, schema, vs_endpoint = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("vs_endpoint")
# COMMAND ----------
from procintel.lakehouse import DeltaLake
from procintel.mlops import MlflowModelStore
from procintel.pipeline import learning_cycle
res = learning_cycle(DeltaLake(spark, catalog, schema), MlflowModelStore(f"{catalog}.{schema}"), f"/Volumes/{catalog}/{schema}/landing/incoming")
print(res)
# COMMAND ----------
# gold_chunks_vs changed through MERGE => Change Data Feed => the delta-sync index picks it up on sync
from procintel.vectorstore import ensure_vector_index, sync_and_wait
print(sync_and_wait(ensure_vector_index(f"{catalog}.{schema}.chunks_index", vs_endpoint, f"{catalog}.{schema}.gold_chunks_vs"),
                    timeout_s=2700, raise_on_timeout=False))
