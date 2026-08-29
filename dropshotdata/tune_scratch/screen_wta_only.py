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

sub = train[train["tour"] == "wta"].dropna(subset=BASE + CAT).copy()
X = sub[BASE].copy()
for c in CAT:
    X[c] = sub[c].astype("category")
y2 = sub["y"].values.astype(int)
d = xgb.DMatrix(X, label=y2, enable_categorical=True)
cv = xgb.cv(PARAMS, d, num_boost_round=5000, nfold=5, early_stopping_rounds=50,
            metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
best = int(cv["test-logloss-mean"].idxmin())
print(f"wta-only model: round={best+1:5d}(cap={(best+1)>=5000}) "
      f"cv_logloss={float(cv['test-logloss-mean'].iloc[best]):.5f}  n={len(sub):,}", flush=True)
