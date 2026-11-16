"""Model registry with champion/challenger aliases.

  LocalModelStore  - pickle + JSON registry (tested)
  MlflowModelStore - MLflow tracking + Unity Catalog registry (`models:/<catalog>.<schema>.<name>@champion`).
                     NOT run in the build sandbox (mlflow could not be installed); calls follow the documented MLflow 2.x API.
Both expose: log(name, models, metrics, params) -> version ; set_alias ; load(name, alias) ; metrics(name, alias)."""
from __future__ import annotations
import json, os, pickle, tempfile


class LocalModelStore:
    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True)
        self._f = os.path.join(root, "registry.json")
        self.reg = json.load(open(self._f)) if os.path.exists(self._f) else {}

    def _save(self):  # numpy scalars (np.bool_, np.float64) -> plain Python
        json.dump(self.reg, open(self._f, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    def log(self, name, models, metrics, params=None) -> int:
        r = self.reg.setdefault(name, dict(versions={}, aliases={}))
        v = len(r["versions"]) + 1
        path = os.path.join(self.root, f"{name}_v{v}.pkl")
        pickle.dump(models, open(path, "wb"))
        r["versions"][str(v)] = dict(path=path, metrics=metrics, params=params or {})
        self._save(); return v

    def set_alias(self, name, alias, version):
        self.reg[name]["aliases"][alias] = int(version); self._save()

    def has(self, name, alias="champion"): return alias in self.reg.get(name, {}).get("aliases", {})

    def _ver(self, name, alias): return str(self.reg[name]["aliases"][alias])

    def load(self, name, alias="champion"):          # resolve against root so a copied snapshot (Volume -> App) still loads
        p = os.path.join(self.root, os.path.basename(self.reg[name]["versions"][self._ver(name, alias)]["path"]))
        return pickle.load(open(p, "rb"))

    def metrics(self, name, alias="champion"): return self.reg[name]["versions"][self._ver(name, alias)]["metrics"]


class MlflowModelStore:
    def __init__(self, uc_prefix: str, experiment: str | None = None, registry_uri: str = "databricks-uc"):
        import mlflow
        mlflow.set_registry_uri(registry_uri)                    # tests use a local sqlite registry with the same alias API
        if experiment:
            mlflow.set_experiment(experiment)
        self.mlflow, self.prefix = mlflow, uc_prefix           # e.g. "main.procintel"

    def _n(self, name): return f"{self.prefix}.{name}"

    def log(self, name, models, metrics, params=None) -> int:
        mlflow = self.mlflow
        with tempfile.TemporaryDirectory() as d, mlflow.start_run() as run:
            path = os.path.join(d, "risk_models.pkl"); pickle.dump(models, open(path, "wb"))
            mlflow.log_params(params or {}); mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            ex = getattr(models, "example_", None)                 # Unity Catalog refuses models without a signature
            sig = mlflow.models.infer_signature(ex, models.predict_df(ex)) if ex is not None else None
            info = mlflow.pyfunc.log_model(artifact_path="model", python_model=_risk_pyfunc(), artifacts={"risk_models": path},
                                           registered_model_name=self._n(name), signature=sig, input_example=ex,
                                           pip_requirements=["scikit-learn==1.5.2", "pandas==2.2.3", "numpy==2.1.3", "xgboost==3.0.3"])
        return int(info.registered_model_version)

    def set_alias(self, name, alias, version):
        self.mlflow.MlflowClient().set_registered_model_alias(self._n(name), alias, str(version))

    def has(self, name, alias="champion"):
        try:
            self.mlflow.MlflowClient().get_model_version_by_alias(self._n(name), alias); return True
        except Exception:
            return False

    def load(self, name, alias="champion"):
        p = self.mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{self._n(name)}@{alias}/artifacts/risk_models.pkl")
        return pickle.load(open(p, "rb"))

    def metrics(self, name, alias="champion"):
        c = self.mlflow.MlflowClient()
        mv = c.get_model_version_by_alias(self._n(name), alias)
        return c.get_run(mv.run_id).data.metrics


def _risk_pyfunc():
    import mlflow.pyfunc

    class RiskPyfunc(mlflow.pyfunc.PythonModel):
        """Serving signature: DataFrame of contract features -> p10/p50/p90 days, P(rebid), P(amend)."""
        def load_context(self, context):
            self.m = pickle.load(open(context.artifacts["risk_models"], "rb"))

        def predict(self, context, model_input, params=None):
            return self.m.predict_df(model_input)
    return RiskPyfunc()


def register_champion(store, name, models, params=None, min_gain=0.0, metric="mae_days"):
    """Log a challenger; it becomes champion if none exists, if the champion has a different model schema version (it cannot
    be served by this code, and its metrics come from a different evaluation), or if it improves `metric` by >= min_gain."""
    v = store.log(name, models, models.metrics, params)
    store.set_alias(name, "challenger", v)
    if not store.has(name, "champion"):
        store.set_alias(name, "champion", v); return v, True
    champ = store.metrics(name, "champion")
    if int(champ.get("model_version", 1)) != int(models.metrics.get("model_version", 1)):
        print(f"{name}: schema change (v{int(champ.get('model_version', 1))} -> v{int(models.metrics.get('model_version', 1))}): "
              f"promoting on its own backtest, metrics are not comparable")
        store.set_alias(name, "champion", v); return v, True
    if (champ[metric] - models.metrics[metric]) / champ[metric] >= min_gain:
        store.set_alias(name, "champion", v); return v, True
    return v, False
