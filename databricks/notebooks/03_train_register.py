# Databricks notebook source
# MAGIC %md ### 03 Train risk models, log to MLflow, register in Unity Catalog with champion/challenger aliases
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
from procintel.pipeline import train
user = spark.sql("SELECT current_user()").first()[0]
store = MlflowModelStore(f"{catalog}.{schema}", experiment=f"/Users/{user}/procintel")
print(train(DeltaLake(spark, catalog, schema), store))     # (version, became_champion)
