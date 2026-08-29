"""Cheap in-sample screen (stats-discipline #9): does rank/age/height/hand
carry real incremental signal over the existing feature set, BEFORE paying
for a full build+xgb.cv+holdout-reveal round? Tune window (2022-23) only,
5-fold CV logloss, never touches the 2024-25 holdout."""
import numpy as np
import pandas as pd
import xgboost as xgb

j = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\rank_screen_tune.parquet")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff", "h2h_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]

CANDS = {
    "rank_log_diff": ["rank_log_diff"],
    "rank_points_log_diff": ["rank_points_log_diff"],
    "rank_both": ["rank_log_diff", "rank_points_log_diff"],
    "age_diff": ["age_diff"],
    "ht_hand": ["ht_diff", "hand_sideA_left", "hand_sideB_left"],
    "all_new": ["rank_log_diff", "rank_points_log_diff", "age_diff", "ht_diff",
                "hand_sideA_left", "hand_sideB_left"],
}

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.08,
          "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols, restrict_cols=None):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    for c in BOOLI:
        X[c] = sub[c].astype(int)
    y = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=500, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


base_ll, base_r, n = cv_ll(BASE)
print(f"BASE            n={n:6d} best_round={base_r:4d}  cv_logloss={base_ll:.5f}")
for name, extra in CANDS.items():
    # stats-discipline #5.3: score BASE and BASE+candidate on the IDENTICAL
    # row set (candidate's own dropna), not against a BASE fit on more rows -
    # ht/hand drop ~15% of rows to missing height, which alone shifts logloss.
    base_same_rows, _, n_same = cv_ll(BASE, restrict_cols=extra)
    ll, r, n2 = cv_ll(BASE + extra)
    assert n_same == n2, f"{name}: row-count mismatch {n_same} vs {n2}"
    print(f"BASE+{name:22s} n={n2:6d} best_round={r:4d}  cv_logloss={ll:.5f}  "
          f"vs BASE-same-rows={base_same_rows:.5f}  delta={ll - base_same_rows:+.5f}")
