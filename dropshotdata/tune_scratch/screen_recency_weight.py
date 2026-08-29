"""Ticket 28 round 5 (cuts of data): does weighting 2023 rows more heavily
than 2022 in the tune window help, on the theory that the tour is closer to
its 2024-25 holdout state? Screened via 5-fold xgb.cv on the tune window
only - the weight itself only touches training, not which rows are held out
each fold, so this is a legitimate tune-only screen."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
tune = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v2_tune.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]
TUNED = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
         "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}

X = tune[NUMERIC].copy()
for c in CAT:
    X[c] = tune[c].astype("category")
for c in BOOLI:
    X[c] = tune[c].astype(int)
y = tune["y"].values.astype(int)
days_from_end = (tune["date"].max() - tune["date"]).dt.days.values.astype(float)

PARAMS_BASE = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **TUNED}


def cv_ll(weight=None):
    d = xgb.DMatrix(X, label=y, weight=weight, enable_categorical=True)
    cv = xgb.cv(PARAMS_BASE, d, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1


ll0, r0 = cv_ll(None)
print(f"unweighted (current shipped)   round={r0:4d}  cv_logloss={ll0:.5f}")

for halflife in (180, 365, 730):
    w = 0.5 ** (days_from_end / halflife)
    ll, r = cv_ll(w)
    print(f"recency-weighted halflife={halflife:4d}d  round={r:4d}  cv_logloss={ll:.5f}  delta={ll - ll0:+.5f}")
