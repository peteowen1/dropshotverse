import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v9_train.parquet")

is_grass = (train["surface"] == "Grass").values
train["serve_form_diff_grass"] = np.where(is_grass, train["serve_form_diff"].values, 0.0)
train["return_form_diff_grass"] = np.where(is_grass, train["return_form_diff"].values, 0.0)

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def cv_ll(numeric_cols, restrict_cols=None, cap=4000, esr=50):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = train.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


restrict = ["serve_form_diff", "return_form_diff"]  # same dropna population for a fair 3-way comparison
base_ll, r0, n0, cap0 = cv_ll(BASE, restrict_cols=restrict)
print(f"BASE (same rows)                  round={r0:5d}(cap={cap0}) cv_logloss={base_ll:.5f}  n={n0:,}", flush=True)
ll1, r1, n1, cap1 = cv_ll(BASE + ["serve_form_diff", "return_form_diff"])
print(f"BASE+serve+return (unmasked)      round={r1:5d}(cap={cap1}) cv_logloss={ll1:.5f}  "
      f"delta={ll1 - base_ll:+.5f}  n={n1:,}", flush=True)
ll2, r2, n2, cap2 = cv_ll(BASE + ["serve_form_diff_grass", "return_form_diff_grass"], restrict_cols=restrict)
print(f"BASE+serve+return (grass-masked)  round={r2:5d}(cap={cap2}) cv_logloss={ll2:.5f}  "
      f"delta={ll2 - base_ll:+.5f}  n={n2:,}", flush=True)
