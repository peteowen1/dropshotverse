import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
hier_raw = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_1995_2023.parquet")
extra = hier_raw[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])
j = train.merge(extra, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
p_hier_sideA = np.clip(np.where(y == 1, j["p_winner_hier"].values, 1 - j["p_winner_hier"].values), 1e-6, 1 - 1e-6)
j["hier_logit_pwinner"] = np.log(p_hier_sideA / (1 - p_hier_sideA))

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols, cap=4000, esr=50):
    sub = j.dropna(subset=numeric_cols + CAT).copy()
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


base_ll, r0, n0, cap0 = cv_ll(BASE)
print(f"BASE                       round={r0:5d} (hit_cap={cap0}) cv_logloss={base_ll:.5f}  n={n0}", flush=True)
ll, r, n, cap1 = cv_ll(BASE + ["hier_logit_pwinner"])
print(f"BASE+hier_logit_pwinner    round={r:5d} (hit_cap={cap1}) cv_logloss={ll:.5f}  "
      f"delta={ll - base_ll:+.5f}  n={n}", flush=True)
