"""Module 4: closed-loop continuous learning + KPI evaluation.

Loop:  new feeds/docs -> idempotent upsert (ETL + ER) -> incremental index update (refit on drift) -> graph update
       -> model monitoring (error + PSI) -> champion/challenger retrain with recency weights -> promote only if better
       -> user feedback adjusts retrieval priors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .core import icr_ratings_from_docs
from .ingest import chunk_document, enrich_contracts, load_dataset, resolve_suppliers


def psi(a, b, bins=8):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 10 or len(b) < 10:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-4
    pb = np.histogram(b, edges)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))


class ContinuousLearner:
    def __init__(self, plat):
        self.p = plat

    # ---- 1. ingest a new batch folder (same format as the initial load)
    def ingest_batch(self, path: str) -> dict:
        p, new = self.p, load_dataset(path)
        ds = p.ds
        ds.projects = pd.concat([ds.projects, new.projects]).drop_duplicates("project_id", keep="last")
        allc = pd.concat([ds.contracts, new.contracts]).drop_duplicates("contract_id", keep="last")
        allc["supplier_id"] = allc.supplier_raw.fillna("UNKNOWN").map(resolve_suppliers(allc.supplier_raw.fillna("UNKNOWN").tolist()))
        ds.contracts = allc.reset_index(drop=True)
        ds.isr = pd.concat([ds.isr, new.isr]).drop_duplicates("isr_id", keep="last")
        ds.disbursements = pd.concat([ds.disbursements, new.disbursements]).drop_duplicates(["project_id", "date"], keep="last")
        ds.docs += [d for d in new.docs if d["doc_id"] not in {x["doc_id"] for x in ds.docs}]
        new_chunks = [c for d in new.docs for c in chunk_document(d)]
        idx = p.index.add(new_chunks)
        fresh = ds.contracts[ds.contracts.contract_id.isin(new.contracts.contract_id)]
        p.graph.add_structural(new.projects, fresh)
        p.graph.add_chunks(new_chunks, ds.contracts)
        p.icr = icr_ratings_from_docs(ds.docs)
        self.new_contracts = fresh
        return dict(new_contracts=len(new.contracts), new_docs=len(new.docs), new_chunks=len(new_chunks), index=idx)

    # ---- 2. monitor the champion on fresh outcomes
    def monitor(self, new_c: pd.DataFrame) -> dict:
        m = self.p.models
        k = new_c.dropna(subset=["tender_days"])
        mae_new = mean_absolute_error(k.tender_days, m.predict_df(k).p50_days)
        mae_ref = m.metrics["mae_days"]
        old = self.p.ds.contracts[~self.p.ds.contracts.contract_id.isin(new_c.contract_id)]
        drift = {f: round(psi(old[f], new_c[f]), 3) for f in ("log_value", "wgi_gov_effectiveness")}
        trigger = mae_new > 1.25 * mae_ref or max(drift.values()) > .25
        return dict(mae_champion_on_new=round(float(mae_new), 1), mae_reference=round(float(mae_ref), 1), psi=drift, retrain=bool(trigger))

    # ---- 3. champion / challenger
    def retrain(self, new_c: pd.DataFrame, recency_weight=3.0, min_gain=0.05, store=None, model_name="procintel_risk") -> dict:
        from .models import RiskModels
        p = self.p
        new_c = new_c.sample(frac=1, random_state=0)
        half = len(new_c) // 2
        fit_new, hold = new_c.iloc[:half], new_c.iloc[half:].dropna(subset=["tender_days"])
        old = p.ds.contracts[~p.ds.contracts.contract_id.isin(new_c.contract_id)]
        train = pd.concat([old, fit_new])
        w = np.r_[np.ones(len(old)), np.full(len(fit_new), recency_weight)]
        open_t = getattr(p.ds, "open_tenders", None)                          # keep censored rows: survival model
        chal = RiskModels().fit(train, weights=w, validate=False, open_tenders=open_t)
        mae_champ = mean_absolute_error(hold.tender_days, p.models.predict_df(hold).p50_days)
        mae_chal = mean_absolute_error(hold.tender_days, chal.predict_df(hold).p50_days)
        promote = (mae_champ - mae_chal) / mae_champ >= min_gain
        chal.metrics = dict(p.models.metrics, mae_days=float(mae_chal))          # holdout-based metrics travel with the model
        if promote:
            chal.fit(pd.concat([old, new_c]), weights=np.r_[np.ones(len(old)), np.full(len(new_c), recency_weight)], validate=False,
                     open_tenders=open_t)
            p.models = chal
            p.versions.append(dict(version=len(p.versions) + 1, metrics=dict(holdout_mae=float(mae_chal)), n_train=len(train)))
        if store is not None:                                                    # registry: champion on promotion, else keep as challenger
            v = store.log(model_name, chal, chal.metrics, dict(promoted=promote, n_train=len(train)))
            store.set_alias(model_name, "challenger", v)                         # the newest candidate is always @challenger
            if promote:
                store.set_alias(model_name, "champion", v)
        return dict(holdout_n=len(hold), mae_champion=round(float(mae_champ), 1), mae_challenger=round(float(mae_chal), 1), promoted=bool(promote),
                    version=p.versions[-1]["version"])

    # ---- 4. human feedback -> retrieval priors
    def feedback(self, chunk_ids, useful: bool, step=0.1):
        for cid in chunk_ids:
            self.p.index.boost[cid] = float(np.clip(self.p.index.boost.get(cid, 0) + (step if useful else -step), -.5, .5))


# ---------------------------------------------------------------- KPIs
def eval_retrieval(plat, labels, themes, k=10) -> dict:
    """Precision@5, Recall@k, MRR for risk-theme queries against planted ground truth (ICR chunks containing a labelled sentence)."""
    by_doc = {}
    for l in labels:
        by_doc.setdefault((l["doc_id"], l["theme"]), []).append(l["sentence"])
    P, R, MRR, n = [], [], [], 0
    for th, t in themes.items():
        rel = {c.chunk_id for c in plat.index.chunks if c.doc_type == "ICR" and any(s in c.text for s in by_doc.get((c.doc_id, th), []))}
        if not rel:
            continue
        for q in t["eval"]:
            hits = [c.chunk_id for c, *_ in plat.index.search(q, k=k, filters={"doc_type": {"ICR"}})]
            P.append(len(set(hits[:5]) & rel) / 5); R.append(len(set(hits) & rel) / min(len(rel), k))
            rr = next((1 / (i + 1) for i, h in enumerate(hits) if h in rel), 0.0); MRR.append(rr); n += 1
    return dict(queries=n, precision_at_5=round(float(np.mean(P)), 3), recall_at_10=round(float(np.mean(R)), 3), mrr=round(float(np.mean(MRR)), 3))
