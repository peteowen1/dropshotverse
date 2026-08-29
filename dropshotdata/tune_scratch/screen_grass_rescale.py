import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
is_grass = (train["surface"] == "Grass").values

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner",
        "serve_form_diff_targeted", "return_form_diff_targeted"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def cv_ll(df, numeric_cols, cap=3500, esr=50):
    sub = df.dropna(subset=numeric_cols + CAT).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


base_ll, r0, n0, cap0 = cv_ll(train, BASE)
print(f"BASE                     round={r0:5d}(cap={cap0}) cv_logloss={base_ll:.5f}  n={n0:,}", flush=True)

for factor in (1.5, 2.0, 3.0):
    t2 = train.copy()
    t2["elo_surface_diff_rescaled"] = np.where(is_grass, t2["elo_surface_diff"].values * factor,
                                                t2["elo_surface_diff"].values)
    cols = [c if c != "elo_surface_diff" else "elo_surface_diff_rescaled" for c in BASE]
    ll_, r, n, cap = cv_ll(t2, cols)
    print(f"grass_rescale={factor:.1f}x  round={r:5d}(cap={cap}) cv_logloss={ll_:.5f}  delta={ll_ - base_ll:+.5f}  n={n:,}",
          flush=True)
