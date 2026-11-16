# Databricks notebook source
# MAGIC %md ### 00 Fetch: pull fresh World Bank data straight into the landing Volume (no local copy)
# MAGIC Feeds + documents + Procurement Plans are fetched into `landing/staging` (kept between runs as the download cache: only
# MAGIC new documents and plans are downloaded), then published as a new batch folder `landing/incoming/<date>/` with
# MAGIC `projects.json` written last. Its arrival fires the continuous-learning job, which ingests every complete batch not ingested yet.
# MAGIC Needs outbound internet from the job's compute to search.worldbank.org and documents.worldbank.org.
# COMMAND ----------
# MAGIC %pip install -q networkx pypdf pymupdf certifi databricks-vectorsearch numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==3.0.3
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import datetime, os, sys; sys.path.insert(0, os.path.abspath("../.."))        # bundle root (synced files) -> `import procintel`
from procintel.live_wb import DEFAULT_COUNTRIES, MAX_DOCS, MAX_PROJECTS, fetch_all, publish
dbutils.widgets.text("catalog", "workspace"); dbutils.widgets.text("schema", "procintel")
dbutils.widgets.text("countries", DEFAULT_COUNTRIES)
dbutils.widgets.text("max_projects", str(MAX_PROJECTS)); dbutils.widgets.text("max_docs", str(MAX_DOCS))
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
landing = f"/Volumes/{catalog}/{schema}/landing"
# COMMAND ----------
staging = f"{landing}/staging"
fetch_all(staging, dbutils.widgets.get("countries"), int(dbutils.widgets.get("max_projects")), int(dbutils.widgets.get("max_docs")))
batch = f"{landing}/incoming/{datetime.date.today().isoformat()}"
print("published", publish(staging, batch), "files to", batch)
