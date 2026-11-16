"""Module 2a: timeline benchmarking & risk scoring.

* Duration: XGBoost accelerated failure time model (objective survival:aft) on interval-censored targets. Signed contracts give
  an exact tender-to-award time; tenders still open (or signed only after the training cut-off) give a right-censored lower
  bound, so slow packages are KEPT instead of dropped. P50 is the model's median; the P10-P90 band is the log-normal spread of
  validation residuals, widened by split-conformal calibration so the stated coverage holds.
* Censoring is always as of a date: in validation, anything signed after the cut-off is treated as still open at the cut-off,
  so the backtest never sees the future (no look-ahead).
* Re-bid and amendment risk: HistGradientBoosting classifiers with sigmoid calibration (base rate when a label has too few positives).
* Native NaN handling => missing contract values / governance scores need no imputation.
* Duration feature set chosen by NESTED temporal validation inside the training window with the one-standard-error rule:
  on small real samples sector/country/governance identify individual projects and overfit; with more countries the full set wins.
* Local explanations by feature ablation against the training reference row."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT = ["category", "procurement_method", "review_type", "sector", "country_code", "es_risk"]
NUM = ["log_value", "wgi_rule_of_law", "wgi_gov_effectiveness", "wgi_reg_quality", "cpia", "notice_month"]
FEATS = CAT + NUM
DUR_SETS = {"full": (CAT, NUM),
            "no_governance": (CAT, ["log_value", "notice_month"]),
            "core": (["category", "procurement_method", "review_type"], ["log_value"])}
Z90 = 1.2815515655446004                 # standard-normal 90th percentile
# Shallow trees with large leaves: with ~600 signed contracts, depth-3 trees with small leaves overfit and the backtest swung
# from 116 to 1,038 days MAE with the seed alone; depth 2 / 20-contract leaves gave 108-109 days on every seed (chosen for
# stability across seeds, and guarded by tests/test_poc.py::test_backtest_is_stable_across_seeds).
AFT_PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                  aft_loss_distribution_scale=1.0, learning_rate=0.05, max_depth=2, min_child_weight=20, subsample=0.9,
                  tree_method="hist", max_cat_to_onehot=8, nthread=4)
AFT_ROUNDS = 250


def _pipe(est, cat=CAT, num=NUM):
    enc = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                                  encoded_missing_value=np.nan), cat)], remainder="passthrough")
    est.set_params(categorical_features=[True] * len(cat) + [False] * len(num))
    return Pipeline([("enc", enc), ("m", est)])


def _X(df, cat=CAT, num=NUM):
    X = df.reindex(columns=cat + num).copy()
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


class BaseRate:
    """Fallback when a label has too few positives to learn from: Laplace-smoothed prior, same predict_proba interface."""
    def __init__(self, y):
        self.p = (float(np.sum(y)) + 1) / (len(y) + 2)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def survival_targets(contracts: pd.DataFrame, open_tenders: pd.DataFrame | None, as_of) -> pd.DataFrame:
    """Rows for the AFT model as known on `as_of`: lower/upper bounds in days.
    signed by as_of -> exact (lower = upper = days); noticed but not signed by as_of (incl. open tenders) -> (elapsed, +inf)."""
    as_of = pd.Timestamp(as_of)
    c = contracts[contracts.tender_days.notna() & (contracts.notice_date < as_of)].copy()
    signed = c.award_date <= as_of
    c["lower"] = np.where(signed, c.tender_days, (as_of - c.notice_date).dt.days)
    c["upper"] = np.where(signed, c.tender_days, np.inf)
    parts = [c]
    if open_tenders is not None and len(open_tenders):
        o = open_tenders[open_tenders.notice_date < as_of].copy()
        end = np.minimum(pd.to_datetime(o.censored_at).values, np.datetime64(as_of))
        o["lower"] = (pd.to_datetime(end) - o.notice_date).dt.days
        o["upper"] = np.inf
        o["_w"] = 1.0
        parts.append(o)
    t = pd.concat(parts, ignore_index=True)
    t["lower"] = t.lower.astype(float).clip(lower=1.0)
    t["upper"] = t.upper.astype(float)
    if "_w" not in t:
        t["_w"] = 1.0
    t["_w"] = t["_w"].fillna(1.0)
    return t.sort_values("notice_date", kind="stable").reset_index(drop=True)


class AFTDuration:
    """XGBoost survival:aft on interval-censored days with native categorical splits."""

    def __init__(self, cat, num, seed=0):
        self.cat, self.num, self.seed = list(cat), list(num), seed

    def _dm(self, df, t=None):
        X = df.reindex(columns=self.cat + self.num).copy()
        if "log_value" in X:
            # open tenders have no contract value yet, so a missing value would mark "still open" and the model would learn
            # "no value => decades". Fill with the category's typical value, identically in training and prediction.
            cat_med = df.get("category", pd.Series(index=df.index, dtype=object)).map(self.value_by_cat)
            X["log_value"] = pd.to_numeric(X.log_value, errors="coerce").fillna(cat_med).fillna(self.value_all)
        for c in self.cat:
            X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None), categories=self.cats[c])
        for c in self.num:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        d = xgb.DMatrix(X, enable_categorical=True, weight=None if t is None else t["_w"].values)
        if t is not None:
            d.set_float_info("label_lower_bound", t.lower.values)
            d.set_float_info("label_upper_bound", t.upper.values)
        return d

    def fit(self, t: pd.DataFrame):
        self.cats = {c: sorted({str(v) for v in t[c].dropna()}) if c in t else [] for c in self.cat}
        lv = pd.to_numeric(t.get("log_value"), errors="coerce") if "log_value" in t else pd.Series(dtype=float)
        self.value_by_cat = lv.groupby(t["category"]).median().to_dict() if len(lv) else {}
        self.value_all = float(lv.median()) if lv.notna().any() else 0.0
        self.bst = xgb.train(dict(AFT_PARAMS, seed=self.seed), self._dm(t, t), num_boost_round=AFT_ROUNDS)
        return self

    def median(self, df) -> np.ndarray:
        return self.bst.predict(self._dm(df))


ENSEMBLE = 5                     # members per duration model (see RiskModels._fit_all)
SIX_MONTHS_DAYS = 182.6
METHOD_BASELINE_MIN = 5          # a category + method pair needs this many signed training tenders, else the category median


def method_baseline(te, tr, fallback) -> pd.Series:
    """The usual time (median) for the same category AND procurement method in training; `fallback` where the pair is rare."""
    s = tr.dropna(subset=["tender_days"]).groupby(["category", "procurement_method"]).tender_days
    med, n = s.median(), s.size()
    ok = n[n >= METHOD_BASELINE_MIN].index
    key = pd.MultiIndex.from_arrays([te.category, te.procurement_method])
    vals = pd.Series(med.reindex(key).values, index=te.index).where(key.isin(ok))
    return vals.fillna(fallback)


def c_index(t, pred) -> float:
    """Share of pairs (one signed sooner than the other) that the forecast orders correctly; ties count half. 0.5 = chance."""
    t, pred = np.asarray(t, float), np.asarray(pred, float)
    sooner = t[:, None] < t[None, :]
    if not sooner.any():
        return float("nan")
    right = (pred[:, None] < pred[None, :]) + 0.5 * (pred[:, None] == pred[None, :])
    return float(right[sooner].sum() / sooner.sum())


class AFTEnsemble:
    """Average of AFT members in log-days (geometric mean of their medians)."""

    def __init__(self, members):
        self.members = members

    def median(self, df) -> np.ndarray:
        return np.exp(np.mean([np.log(np.maximum(m.median(df), 1.0)) for m in self.members], axis=0))


def impute_open_like_contracts(open_tenders, contracts, seed=0):
    """Open tenders lack what only an award reveals (contract value, review type). Left missing - or filled with one constant -
    that gap marks 'still open' and the model learns 'missing => decades'. Fill each from a random contract of the same
    category instead, so open tenders look like any other tender except for being censored."""
    if open_tenders is None or not len(open_tenders):
        return open_tenders
    o, rng = open_tenders.copy(), np.random.default_rng(seed)
    for col in ("log_value", "review_type"):
        if col not in o:
            o[col] = np.nan
        for cat, idx in o[o[col].isna()].groupby("category").groups.items():
            pool = contracts.loc[contracts.category == cat, col].dropna()
            pool = pool if len(pool) else contracts[col].dropna()
            if len(pool):
                o.loc[idx, col] = rng.choice(pool.values, size=len(idx))
    return o


class RiskModels:
    MODEL_VERSION = 4          # 1 = quantile GBMs; 2 = XGBoost survival (AFT); 4 = AFT ensemble (multiple imputation + bagging);
                               # 3 = AFT whose retrains also keep open tenders as
                               # censored rows (a v2 champion retrained by the continuous-learning loop had lost them and forecast short).
                               # Bump on any change to the pickled structure or to what the model is trained on.

    def __init__(self, seed=0):
        self.seed = seed
        self.metrics: dict = {}

    def _fit_all(self, df, weights=None, open_tenders=None, as_of=None):
        df = df.copy(); df["_w"] = 1.0 if weights is None else np.asarray(weights, dtype=float)
        as_of = pd.Timestamp(as_of) if as_of is not None else df.award_date.max()
        # multiple imputation + bagging: each member gets its own random fill-in for the open tenders and its own row subsample;
        # one draw alone made the backtest swing (81 vs 130 days by seed, 2026-11-09), the average of ENSEMBLE draws does not
        members = [self.seed * 101 + k for k in range(ENSEMBLE)]
        opens = {m: impute_open_like_contracts(open_tenders, df, m) for m in members}
        ts = {m: survival_targets(df, opens[m], as_of) for m in members}
        t = ts[members[0]]
        exact = t.lower == t.upper
        # nested temporal selection: train as known at an inner cut-off, score on tenders noticed after it and signed by as_of
        # cut on SIGNED rows: cutting on all rows left <10 signed tenders after the cut in backtests (most were still open)
        ex_dates = t.notice_date[exact].sort_values()
        inner_cut = ex_dates.iloc[int(len(ex_dates) * .8)] if len(ex_dates) else t.notice_date.iloc[int(len(t) * .8)]
        t_ins = {m: survival_targets(df[df.notice_date < inner_cut], None if opens[m] is None else opens[m][opens[m].notice_date < inner_cut],
                                     inner_cut) for m in members}
        val = t[exact & (t.notice_date >= inner_cut)]
        score, se, fits = {}, {}, {}
        if len(val) >= 10:
            for nm, fs in DUR_SETS.items():
                fits[nm] = AFTEnsemble([AFTDuration(*fs, seed=m).fit(t_ins[m]) for m in members])
                err = np.abs(fits[nm].median(val) - val.lower.values)
                score[nm], se[nm] = float(err.mean()), float(err.std(ddof=1) / np.sqrt(len(err)))
            best = min(score, key=score.get)
            # one-standard-error rule: simplest set within 1 SE of the best (sets ordered simplest-last in DUR_SETS)
            self.dur_set = next(nm for nm in reversed(DUR_SETS) if score[nm] <= score[best] + se[best])
            # interval: log-normal spread of validation residuals, then split-conformal widening to 80% coverage
            r = np.log(val.lower.values) - np.log(fits[self.dur_set].median(val))
            self.sigma = float(np.std(r, ddof=1))
            E = np.maximum(-Z90 * self.sigma - r, r - Z90 * self.sigma)
            self.qhat = float(np.quantile(E, min(1.0, np.ceil((len(E) + 1) * .8) / len(E)), method="higher"))
        else:                    # too few signed tenders to validate: simplest feature set, wide uncalibrated band
            self.dur_set, self.sigma, self.qhat = "core", 0.8, 0.0
        self.dur_scores = score
        self.dur = AFTEnsemble([AFTDuration(*DUR_SETS[self.dur_set], seed=m).fit(ts[m]) for m in members])
        self.n_censored = int((~exact).sum()); self.n_exact = int(exact.sum())
        self.clf = {}
        for tgt in ("rebid", "amend"):
            col = "rebid" if tgt == "rebid" else "amendments"
            lab = df[df[col].notna()]                         # only contracts whose label is known (reliably linked on live data)
            y_c = lab.rebid.values.astype(int) if tgt == "rebid" else (lab.amendments.values > 0).astype(int)
            if min(y_c.sum(), len(y_c) - y_c.sum()) < 3:      # real proxies can be all-zero: calibrated CV needs >=3 per class
                self.clf[tgt] = BaseRate(y_c)
                continue
            base = _pipe(HistGradientBoostingClassifier(max_iter=120, learning_rate=.06, max_depth=3, random_state=self.seed))
            m = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            self.clf[tgt] = m.fit(_X(lab), y_c)
        self.n_labelled = {t: int(df[c].notna().sum()) for t, c in (("rebid", "rebid"), ("amend", "amendments"))}
        self.ref = {c: (df[c].mode().iloc[0] if c in CAT else df[c].median()) for c in FEATS}

    def fit(self, df, weights=None, validate=True, open_tenders=None, eval_horizon_days=None):
        """open_tenders: tenders with no contract yet (right-censored at `censored_at`). eval_horizon_days: score the holdout
        only on tenders noticed at least that long before the data's end (recent notices are visible only if they were fast)."""
        order = np.argsort(df.notice_date.values, kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        weights = None if weights is None else np.asarray(weights)[order]
        if open_tenders is not None and not len(open_tenders):
            open_tenders = None
        end = max([d for d in (df.award_date.max(), None if open_tenders is None else pd.to_datetime(open_tenders.censored_at).max())
                   if d is not None and pd.notna(d)])
        if eval_horizon_days is None:
            eval_horizon_days = 365 if open_tenders is not None else 0
        if validate:
            cut = int(len(df) * .75); cut_date = df.notice_date.iloc[cut]
            tr, te = df[df.notice_date < cut_date], df[df.notice_date >= cut_date]
            o_tr = None if open_tenders is None else open_tenders[open_tenders.notice_date < cut_date]
            self._fit_all(tr, None if weights is None else weights[:len(tr)], o_tr, as_of=cut_date)   # censored AT the cut
            te = te[te.notice_date <= end - pd.Timedelta(days=eval_horizon_days)]
            self.metrics = self.evaluate(te, tr)
            self.metrics["n_train_censored"] = self.n_censored
        self._fit_all(df, weights, open_tenders, as_of=end)
        self.metrics["n_final_exact"], self.metrics["n_final_censored"] = self.n_exact, self.n_censored
        self.metrics["model_version"] = self.MODEL_VERSION
        self.example_ = _X(df.tail(5)).reset_index(drop=True)      # MLflow signature / input example (UC registry requires one)
        return self

    def predict_df(self, df) -> pd.DataFrame:
        X = _X(df)
        lmed = np.log(np.maximum(self.dur.median(df), 1.0))
        half = Z90 * self.sigma + self.qhat
        out = pd.DataFrame(dict(p10_days=np.exp(lmed - half), p50_days=np.exp(lmed), p90_days=np.exp(lmed + half)), index=df.index)
        out["p_rebid"] = self.clf["rebid"].predict_proba(X)[:, 1]
        out["p_amend"] = self.clf["amend"].predict_proba(X)[:, 1]
        return out

    def evaluate(self, te, tr) -> dict:
        p = self.predict_df(te)
        k = te.tender_days.notna()
        base = tr.groupby("category").tender_days.median()
        b = te.category.map(base).fillna(tr.tender_days.median())
        bm = method_baseline(te, tr, b)
        y = te.tender_days[k].values.astype(float)
        m = dict(n_test=int(k.sum()),
                 mae_days=float(mean_absolute_error(te.tender_days[k], p.p50_days[k])),
                 mae_baseline_category_median=float(mean_absolute_error(te.tender_days[k], b[k])),
                 # the stronger simple rule: the usual time for the same category AND method. On average error it is about as
                 # good as the model; the model's edge is ranking, probabilities and ranges (kept in the registry every run)
                 mae_baseline_category_method_median=float(mean_absolute_error(te.tender_days[k], bm[k])),
                 c_index=c_index(y, p.p50_days[k].values),
                 c_index_baseline_category_method=c_index(y, bm[k].values),
                 interval_coverage_p10_p90=float(((te.tender_days[k] >= p.p10_days[k]) & (te.tender_days[k] <= p.p90_days[k])).mean()))
        lab = (y <= SIX_MONTHS_DAYS).astype(int)               # "signed within 6 months?": AUC needs a yes/no question
        if min(int(lab.sum()), int(len(lab) - lab.sum())) >= 10:
            m["auc_signed_6m"] = float(roc_auc_score(lab, -p.p50_days[k].values))
            m["auc_signed_6m_baseline_category_method"] = float(roc_auc_score(lab, -bm[k].values))
        for tgt, col in (("rebid", "rebid"), ("amend", "amendments")):
            k = te[col].notna()                                # evaluate only where the label is known
            y = (te.rebid[k].astype(int) if tgt == "rebid" else (te.amendments[k] > 0).astype(int))
            p = self.predict_df(te[k]) if k.any() else None
            if min(int(y.sum()), int(len(y) - y.sum())) >= 10:     # an AUC from a handful of positives is noise: not reported
                m[f"auc_{tgt}"] = float(roc_auc_score(y, p[f"p_{tgt}"]))
                m[f"brier_{tgt}"] = float(brier_score_loss(y, p[f"p_{tgt}"]))
                m[f"brier_{tgt}_baseline"] = float(brier_score_loss(y, np.full(len(y), (tr.rebid.mean() if tgt == "rebid" else (tr.amendments.dropna() > 0).mean()))))
            m[f"n_test_{tgt}"], m[f"positives_test_{tgt}"] = int(len(y)), int(y.sum())
        return m

    def explain(self, row: pd.DataFrame, top=4):
        """Ablation attribution: replace each feature by its reference value; report change in P50 days and P(rebid)."""
        base = self.predict_df(row).iloc[0]
        res = []
        for f in FEATS:
            alt = row.copy(); alt[f] = self.ref[f]
            p = self.predict_df(alt).iloc[0]
            res.append(dict(feature=f, value=row[f].iloc[0], d_days=float(base.p50_days - p.p50_days), d_rebid=float(base.p_rebid - p.p_rebid)))
        return sorted(res, key=lambda r: -(abs(r["d_days"]) / max(base.p50_days, 1) + abs(r["d_rebid"])))[:top]
