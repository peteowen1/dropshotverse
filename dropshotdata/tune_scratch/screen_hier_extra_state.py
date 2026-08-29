"""Round 13: hier's prediction parquets carry more than the raw d1/d2 state
this pipeline has ever used - p_winner_hier (the model's own MC-corrected
combined probability) and the _nosurf variant (surface-blind posterior).
Neither has been fed to XGBoost. Screened via 5-fold CV on the wide training
set (1995-2023) - the 2024-25 holdout is not touched by this screen."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
hier = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")
print(hier.columns.tolist())

train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_1995_2023.parquet")
extra = hier[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier",
              "d1_mean_nosurf", "d1_var_nosurf", "d2_mean_nosurf", "d2_var_nosurf"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])

j = train.merge(extra, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)

p_hier = j["p_winner_hier"].values.astype(float)
p_hier_sideA = np.clip(np.where(y == 1, p_hier, 1 - p_hier), 1e-6, 1 - 1e-6)
j["hier_logit_pwinner"] = np.log(p_hier_sideA / (1 - p_hier_sideA))

d1mn, d2mn = j["d1_mean_nosurf"].values.astype(float), j["d2_mean_nosurf"].values.astype(float)
d1vn, d2vn = j["d1_var_nosurf"].values.astype(float), j["d2_var_nosurf"].values.astype(float)
j["hier_dmean_nosurf_diff"] = np.where(y == 1, d1mn - d2mn, d2mn - d1mn)
j["hier_dvar_nosurf_diff"] = np.where(y == 1, d1vn - d2vn, d2vn - d1vn)
# surface-specialization signal: how much does this player's surf-specific
# mean deviate from their nosurf (overall) mean, relative to opponent
a_dmean = np.where(y == 1, j["d1_mean"].values, j["d2_mean"].values)
b_dmean = np.where(y == 1, j["d2_mean"].values, j["d1_mean"].values)
a_dmean_nosurf = np.where(y == 1, d1mn, d2mn)
b_dmean_nosurf = np.where(y == 1, d2mn, d1mn)
j["surf_specialization_diff"] = (a_dmean - a_dmean_nosurf) - (b_dmean - b_dmean_nosurf)

print("coverage:", j[["hier_logit_pwinner", "hier_dmean_nosurf_diff", "surf_specialization_diff"]]
      .notna().all(axis=1).mean())

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}

CANDS = {
    "hier_logit_pwinner": ["hier_logit_pwinner"],
    "nosurf_state": ["hier_dmean_nosurf_diff", "hier_dvar_nosurf_diff"],
    "surf_specialization": ["surf_specialization_diff"],
    "all": ["hier_logit_pwinner", "hier_dmean_nosurf_diff", "hier_dvar_nosurf_diff", "surf_specialization_diff"],
}


def cv_ll(numeric_cols, restrict_cols=None):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=2000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


for name, extra_cols in CANDS.items():
    base_same, r0, n0 = cv_ll(BASE, restrict_cols=extra_cols)
    ll, r, n = cv_ll(BASE + extra_cols)
    assert n0 == n
    print(f"BASE+{name:24s} n={n:7d} round={r:4d} cv_logloss={ll:.5f}  "
          f"vs BASE-same-rows={base_same:.5f}  delta={ll - base_same:+.5f}")
