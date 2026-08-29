"""Random hyperparameter search on the tune window (2022-23) only, via 5-fold
xgb.cv - never touches the 2024-25 holdout. Current model has only ever
selected num_boost_round; max_depth/eta/min_child_weight/gamma/reg_alpha/
reg_lambda/subsample/colsample_bytree are all still at their first-guess
defaults. stats-discipline #5.4: every tuned param gets pinned and reported
together, not left to vary silently between comparisons."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
tune = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_tune.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]

X = tune[NUMERIC].copy()
for c in CAT:
    X[c] = tune[c].astype("category")
for c in BOOLI:
    X[c] = tune[c].astype(int)
y = tune["y"].values.astype(int)
dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

DEFAULT = {"max_depth": 5, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8,
           "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}

rng = np.random.default_rng(0)
GRID = []
for _ in range(40):
    GRID.append({
        "max_depth": int(rng.choice([3, 4, 5, 6, 7])),
        "eta": float(rng.choice([0.02, 0.04, 0.06, 0.08, 0.10, 0.15])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        "min_child_weight": float(rng.choice([1, 3, 5, 10, 20, 40])),
        "gamma": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])),
    })


def run(params):
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **params}
    cv = xgb.cv(p, dtrain, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), float(cv["test-logloss-std"].iloc[best]), best + 1


base_ll, base_std, base_r = run(DEFAULT)
print(f"DEFAULT (current shipped params): cv_logloss={base_ll:.5f} +/- {base_std:.5f}  round={base_r}")
print(f"  {DEFAULT}")

results = []
for i, params in enumerate(GRID):
    ll, std, r = run(params)
    results.append((ll, std, r, params))
    print(f"[{i:2d}] cv_logloss={ll:.5f} +/- {std:.5f}  round={r:4d}  {params}")

results.sort(key=lambda t: t[0])
print("\n=== TOP 5 ===")
for ll, std, r, params in results[:5]:
    print(f"cv_logloss={ll:.5f} +/- {std:.5f}  round={r:4d}  delta_vs_default={ll - base_ll:+.5f}  {params}")
