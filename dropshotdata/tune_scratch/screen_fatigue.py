import pandas as pd
import xgboost as xgb

j = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\fatigue_screen_tune.parquet")
BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff", "h2h_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.08,
          "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols):
    sub = j.dropna(subset=numeric_cols + CAT).copy()
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
ll, r, n2 = cv_ll(BASE + ["fatigue_diff"])
print(f"BASE+fatigue_diff n={n2:6d} best_round={r:4d}  cv_logloss={ll:.5f}  delta={ll - base_ll:+.5f}")
