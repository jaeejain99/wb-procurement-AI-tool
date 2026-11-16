"""Packaging the Copilot for Databricks Model Serving.

Serving endpoints have no Spark, so the model ships a *snapshot* of the tables it needs (exported from Unity Catalog at deploy time)
plus the champion risk model; retrieval can still go to Vector Search when PROCINTEL_VS_INDEX / PROCINTEL_VS_ENDPOINT and workspace
credentials are present in the endpoint environment, otherwise it falls back to the in-memory index built from the snapshot.
`export_snapshot` / `load_copilot` / `report_to_json` are tested locally; `log_copilot` (MLflow) is not run in the build sandbox."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from .core import Platform
from .lakehouse import Lake, LocalLake
from .mlops import LocalModelStore
from .pipeline import MODEL_NAME
from .rag import Copilot, PackageSpec

TABLES = ["bronze_docs", "silver_projects", "silver_contracts", "silver_governance", "silver_disbursements", "silver_isr",
          "gold_chunks", "gold_contract_features", "silver_open_tenders", "gold_feedback_priors", "silver_procurement_plan"]


def export_snapshot(lake: Lake, store, path: str) -> str:
    snap = LocalLake(path)
    for t in TABLES:
        if lake.exists(t):
            snap.write(t, lake.read(t))
    ms = LocalModelStore(os.path.join(path, "models"))
    v = ms.log(MODEL_NAME, store.load(MODEL_NAME), store.metrics(MODEL_NAME)); ms.set_alias(MODEL_NAME, "champion", v)
    # written last: the running App polls it and loads a newer snapshot without a restart (app.watch_snapshot)
    with open(os.path.join(path, "_version.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S"))
    return path


def load_copilot(path: str) -> Copilot:
    factory = None
    if os.getenv("PROCINTEL_VS_INDEX"):                                   # Vector Search retrieval (needs workspace credentials)
        from .vectorstore import VectorSearchRetriever, ensure_vector_index
        idx = ensure_vector_index(os.environ["PROCINTEL_VS_INDEX"], os.environ["PROCINTEL_VS_ENDPOINT"], None)
        factory = lambda chunks: VectorSearchRetriever(idx, chunks)
    plat = Platform.from_lake(LocalLake(path), LocalModelStore(os.path.join(path, "models")), factory)
    return Copilot(plat)


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return str(o)
    return str(o)


def report_to_json(rep: dict) -> str:
    keep = {k: rep[k] for k in ("facts", "explanation", "findings", "rejected", "conflicts", "abstained", "schedule", "trace")}
    return json.dumps(keep, default=_j)


def log_copilot(snapshot_dir: str, registered_name: str):
    """Log to MLflow / Unity Catalog. Input: DataFrame with PackageSpec columns. Output: JSON report per row."""
    import mlflow, mlflow.pyfunc

    class CopilotPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.cop = load_copilot(context.artifacts["snapshot"])

        def predict(self, context, model_input, params=None):
            return [report_to_json(self.cop.analyze(PackageSpec(**r))) for r in model_input.to_dict("records")]

    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run():
        return mlflow.pyfunc.log_model(artifact_path="copilot", python_model=CopilotPyfunc(), artifacts={"snapshot": snapshot_dir},
                                       registered_model_name=registered_name, pip_requirements=["scikit-learn", "pandas", "numpy", "networkx", "databricks-vectorsearch"])
