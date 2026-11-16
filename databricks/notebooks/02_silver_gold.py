# Databricks notebook source
# MAGIC %md ### 02 Silver + Gold: entity resolution, governance as-of join, chunking, features, KG edges, ISR-vs-ICR conflicts
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
from procintel.pipeline import silver_gold
print(silver_gold(DeltaLake(spark, catalog, schema)))
# COMMAND ----------
spark.sql(f"COMMENT ON TABLE {catalog}.{schema}.gold_chunks IS 'Quotable report chunks with section path/page; full corpus for the app index; gold_chunks_vs (subset) feeds Vector Search'")
