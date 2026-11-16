# Databricks notebook source
# MAGIC %md ### 01 Bronze: land raw World Bank feeds and documents (append-only Delta)
# COMMAND ----------
# MAGIC %pip install -q networkx pypdf pymupdf certifi databricks-vectorsearch numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==3.0.3
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import os, sys; sys.path.insert(0, os.path.abspath("../.."))        # bundle root (synced files) -> `import procintel`
dbutils.widgets.text("catalog", "workspace"); dbutils.widgets.text("schema", "procintel"); dbutils.widgets.text("vs_endpoint", "procintel_vs")
catalog, schema, vs_endpoint = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("vs_endpoint")
# COMMAND ----------
from procintel.live_wb import DEFAULT_COUNTRIES, MAX_DOCS, MAX_PROJECTS, fetch_all, publish
dbutils.widgets.text("countries", DEFAULT_COUNTRIES); dbutils.widgets.dropdown("fetch_live", "false", ["true", "false"])
dbutils.widgets.text("max_projects", str(MAX_PROJECTS)); dbutils.widgets.text("max_docs", str(MAX_DOCS))
landing = f"/Volumes/{catalog}/{schema}/landing/base"
if dbutils.widgets.get("fetch_live") == "true":   # needs outbound internet to the World Bank APIs
    # Projects / contractdata / procnotices / WDS documents / WGI / CPIA + Procurement Plans, fetched into a staging folder that
    # doubles as the download cache, then copied into landing/base with projects.json last (bronze never reads half a pull)
    staging = f"/Volumes/{catalog}/{schema}/landing/staging"
    fetch_all(staging, dbutils.widgets.get("countries"), int(dbutils.widgets.get("max_projects")), int(dbutils.widgets.get("max_docs")))
    print("published", publish(staging, landing), "files to", landing)
# COMMAND ----------
from procintel.lakehouse import DeltaLake
from procintel.pipeline import bronze_ingest
lake = DeltaLake(spark, catalog, schema)
print(bronze_ingest(lake, landing))
# incremental batches that landed later (file-arrival job) are part of a full rebuild too; re-ingest is idempotent (docs by
# content hash, tables deduplicated by key in silver), and it picks up documents re-parsed by a newer parser
from procintel.pipeline import new_batches
incoming = f"/Volumes/{catalog}/{schema}/landing/incoming"
if os.path.exists(os.path.join(incoming, "projects.json")):
    print(bronze_ingest(lake, incoming))
for batch in new_batches(lake, incoming):          # scheduled pulls (incoming/<date>/) not ingested yet
    print(bronze_ingest(lake, batch))
