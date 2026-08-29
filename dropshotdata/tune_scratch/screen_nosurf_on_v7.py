"""Round 14: does the nosurf hier variant (surface-blind posterior) or a
surf-specialization signal add anything ON TOP of Arm F's hier_logit_pwinner
win? Properly converged CV (uncapped enough given round 13's lesson that a
genuinely rich feature needs ~3000+ rounds, not the default 400-1000)."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
hier_raw = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")
extra = hier_raw[["date", "tour", "winner_uid", "loser_uid",
                   "d1_mean_nosurf", "d1_var_nosurf", "d2_mean_nosurf", "d2_var_nosurf"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])

j = train.merge(extra, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
d1mn, d2mn = j["d1_mean_nosurf"].values.astype(float), j["d2_mean_nosurf"].values.astype(float)
d1vn, d2vn = j["d1_var_nosurf"].values.astype(float), j["d2_var_nosurf"].values.astype(float)
j["hier_dmean_nosurf_diff"] = np.where(y == 1, d1mn - d2mn, d2mn - d1mn)
j["hier_dvar_nosurf_diff"] = np.where(y == 1, d1vn - d2vn, d2vn - d1vn)

# surf-specialization: how much does this player's surf-specific mean deviate
# from their nosurf (overall) mean, relative to the opponent's same gap
a_dmean = np.where(y == 1, j["d1_mean"].values, j["d2_mean"].values)
b_dmean = np.where(y == 1, j["d2_mean"].values, j["d1_mean"].values)
a_dmean_nosurf = np.where(y == 1, d1mn, d2mn)
b_dmean_nosurf = np.where(y == 1, d2mn, d1mn)
j["surf_specialization_diff"] = (a_dmean - a_dmean_nosurf) - (b_dmean - b_dmean_nosurf)

print("coverage:", j[["hier_dmean_nosurf_diff", "surf_specialization_diff"]].notna().mean().to_dict())

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CAT = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols, restrict_cols=None, cap=5000, esr=50):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    hit_cap = (best + 1) >= cap
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), hit_cap


CANDS = {
    "nosurf_state": ["hier_dmean_nosurf_diff", "hier_dvar_nosurf_diff"],
    "surf_specialization": ["surf_specialization_diff"],
    "both": ["hier_dmean_nosurf_diff", "hier_dvar_nosurf_diff", "surf_specialization_diff"],
}

for name, cols in CANDS.items():
    base_same, r0, n0, cap0 = cv_ll(BASE, restrict_cols=cols)
    ll, r, n, cap1 = cv_ll(BASE + cols)
    assert n0 == n
    print(f"BASE+{name:20s} n={n:7d} round={r:5d}(cap={cap1}) cv_logloss={ll:.5f}  "
          f"vs BASE-same-rows(round={r0},cap={cap0})={base_same:.5f}  delta={ll - base_same:+.5f}",
          flush=True)
