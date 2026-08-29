import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
hier_raw = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")
extra = hier_raw[["date", "tour", "winner_uid", "loser_uid",
                   "d1_mean_nosurf", "d1_var_nosurf", "d2_mean_nosurf", "d2_var_nosurf"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])
j = train.merge(extra, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
d1mn, d2mn = j["d1_mean_nosurf"].values.astype(float), j["d2_mean_nosurf"].values.astype(float)
d1vn, d2vn = j["d1_var_nosurf"].values.astype(float), j["d2_var_nosurf"].values.astype(float)
dmean_diff = np.where(y == 1, d1mn - d2mn, d2mn - d1mn)
dvar_diff = np.where(y == 1, d1vn - d2vn, d2vn - d1vn)

is_grass = (j["surface"] == "Grass").values
j["hier_dmean_nosurf_grass"] = np.where(is_grass, dmean_diff, 0.0)
j["hier_dvar_nosurf_grass"] = np.where(is_grass, dvar_diff, 0.0)

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner",
        "serve_form_diff_targeted", "return_form_diff_targeted"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def cv_ll(numeric_cols, restrict_cols=None, cap=4000, esr=50):
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
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


restrict = ["hier_dmean_nosurf_grass", "hier_dvar_nosurf_grass"]
base_ll, r0, n0, cap0 = cv_ll(BASE, restrict_cols=restrict)
print(f"BASE (same rows)          round={r0:5d}(cap={cap0}) cv_logloss={base_ll:.5f}  n={n0:,}", flush=True)
ll1, r1, n1, cap1 = cv_ll(BASE + restrict)
print(f"BASE+nosurf-grass-masked  round={r1:5d}(cap={cap1}) cv_logloss={ll1:.5f}  delta={ll1 - base_ll:+.5f}  n={n1:,}",
      flush=True)
