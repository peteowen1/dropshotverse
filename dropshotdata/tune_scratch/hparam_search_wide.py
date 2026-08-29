"""Round 12 continuation: the current tuned hyperparameters (max_depth=3 etc)
were selected on a 10,290-row single-era (2022-23) CV search. Arm D's wider
CV-holdout gap (0.0226 vs Arm C's 0.0099) suggests they may no longer be
optimal for 88,481 rows spanning 19 heterogeneous years. Re-search on the
WIDE dataset's own CV (still never touches the 2024-25 holdout)."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_2005_2023.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]

X = train[NUMERIC].copy()
for c in CAT:
    X[c] = train[c].astype("category")
y = train["y"].values.astype(int)
dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

CURRENT = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
           "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}

rng = np.random.default_rng(1)
GRID = []
for _ in range(24):
    GRID.append({
        "max_depth": int(rng.choice([3, 4, 5, 6])),
        "eta": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9])),
        "colsample_bytree": float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9])),
        "min_child_weight": float(rng.choice([1, 5, 10, 20, 50, 100])),
        "gamma": float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])),
    })


def run(params):
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **params}
    cv = xgb.cv(p, dtrain, num_boost_round=2000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), float(cv["test-logloss-std"].iloc[best]), best + 1


base_ll, base_std, base_r = run(CURRENT)
print(f"CURRENT (max_depth=3 etc, tuned on 2022-23 only): cv_logloss={base_ll:.5f} +/- {base_std:.5f}  round={base_r}")

results = []
for i, params in enumerate(GRID):
    ll, std, r = run(params)
    results.append((ll, std, r, params))
    print(f"[{i:2d}] cv_logloss={ll:.5f} +/- {std:.5f}  round={r:4d}  {params}", flush=True)

results.sort(key=lambda t: t[0])
print("\n=== TOP 5 ===")
for ll, std, r, params in results[:5]:
    print(f"cv_logloss={ll:.5f} +/- {std:.5f}  round={r:4d}  delta_vs_current={ll - base_ll:+.5f}  {params}")
