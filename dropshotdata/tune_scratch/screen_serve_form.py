import numpy as np
import pandas as pd
import xgboost as xgb

sf = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\serve_form_full.parquet")
tune = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_tune.parquet")

j = tune.merge(sf, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
cov = j[["serve_form_w", "serve_form_l"]].notna().all(axis=1).mean()
print(f"tune-window (2022-23) serve_form coverage: {cov:.4f}  n={len(j):,}")

y = j["y"].values.astype(int)
w = j["serve_form_w"].values.astype(float)
l = j["serve_form_l"].values.astype(float)
j["serve_form_diff"] = np.where(y == 1, w - l, l - w)

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]
TUNED = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
         "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}


def cv_ll(numeric_cols, restrict_cols=None):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    for c in BOOLI:
        X[c] = sub[c].astype(int)
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **TUNED}
    cv = xgb.cv(p, d, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


base_same, r0, n0 = cv_ll(BASE, restrict_cols=["serve_form_diff"])
ll, r, n = cv_ll(BASE + ["serve_form_diff"])
print(f"BASE (same rows as candidate)   n={n0:6d} round={r0:4d} cv_logloss={base_same:.5f}")
print(f"BASE+serve_form_diff            n={n:6d} round={r:4d} cv_logloss={ll:.5f}  delta={ll - base_same:+.5f}")
