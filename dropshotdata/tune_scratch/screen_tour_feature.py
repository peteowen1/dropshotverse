"""Round 14 continued: `tour` (atp/wta) has never been an explicit feature -
the pooled model only knows which tour a match belongs to indirectly through
other features' distributions. ATP lags WTA substantially in Arm F (0.0026
vs beaten). Does letting the tree explicitly split on tour help, especially
given ticket 11's own finding that ATP/WTA point-importance dynamics genuinely
differ (dP_game/dp 26% steeper at WTA's operating point)?"""
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CAT = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols, cat_cols, cap=5000, esr=50):
    sub = train.dropna(subset=numeric_cols + cat_cols).copy()
    X = sub[numeric_cols].copy()
    for c in cat_cols:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


base_ll, r0, n0, cap0 = cv_ll(BASE, CAT)
print(f"BASE (no tour)          round={r0:5d}(cap={cap0}) cv_logloss={base_ll:.5f}  n={n0}", flush=True)
ll, r, n, cap1 = cv_ll(BASE, CAT + ["tour"])
print(f"BASE+tour               round={r:5d}(cap={cap1}) cv_logloss={ll:.5f}  delta={ll - base_ll:+.5f}  n={n}",
      flush=True)
