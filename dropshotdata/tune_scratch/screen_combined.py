"""Combine the three tune-window-only screened wins (hparams, age_diff,
fatigue_diff) and check they're not redundant with each other before paying
for the one holdout reveal. Still tune-only, still no holdout peek."""
import pandas as pd
import xgboost as xgb

rank = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\rank_screen_tune.parquet")
fatigue = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\fatigue_screen_tune.parquet")
j = rank.merge(
    fatigue[["date", "tour", "winner_uid", "loser_uid", "fatigue_diff"]],
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff", "h2h_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]

DEFAULT = {"max_depth": 5, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8,
           "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}
TUNED = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
         "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}


def cv_ll(numeric_cols, params):
    sub = j.dropna(subset=numeric_cols + CAT).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    for c in BOOLI:
        X[c] = sub[c].astype(int)
    y = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y, enable_categorical=True)
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **params}
    cv = xgb.cv(p, d, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


runs = [
    ("BASE, default hparams", BASE, DEFAULT),
    ("BASE, tuned hparams", BASE, TUNED),
    ("BASE+age, tuned hparams", BASE + ["age_diff"], TUNED),
    ("BASE+fatigue, tuned hparams", BASE + ["fatigue_diff"], TUNED),
    ("BASE+age+fatigue, tuned hparams", BASE + ["age_diff", "fatigue_diff"], TUNED),
]
base_ll = None
for name, cols, params in runs:
    ll, r, n = cv_ll(cols, params)
    tag = "" if base_ll is None else f"  delta_vs_default={ll - base_ll:+.5f}"
    if base_ll is None:
        base_ll = ll
    print(f"{name:36s} n={n:6d} round={r:4d} cv_logloss={ll:.5f}{tag}")
