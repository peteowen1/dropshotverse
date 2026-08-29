"""Goal escalation: get ATP beating the market too. Round 14 showed ATP's
gap isn't a hyperparameter mismatch (separate ATP-only models with the SAME
hyperparameters did worse than pooled) - so try shifting the POOLED model's
own loss function toward ATP via sample weight, since WTA already has
0.00073 of margin to spare and ATP has none. CV-only screen first."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}

sub = train.dropna(subset=NUMERIC + CAT).copy()
X = sub[NUMERIC].copy()
for c in CAT:
    X[c] = sub[c].astype("category")
y = sub["y"].values.astype(int)
tour_arr = sub["tour"].values


def cv_by_tour(weight, cap=4000, esr=50):
    d = xgb.DMatrix(X, label=y, weight=weight, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, (best + 1) >= cap


# NOTE: xgb.cv's own reported logloss is POOLED across tours already, so it
# won't directly show the by-tour split - this screen instead trains ONE
# model per weight setting on ALL rows minus one fold (manual, not xgb.cv)
# and scores held-out ATP/WTA rows separately, to see the actual tradeoff.
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=0)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


for atp_weight in (1.0, 1.3, 1.6, 2.0, 3.0):
    w = np.where(tour_arr == "atp", atp_weight, 1.0)
    oof = np.full(len(sub), np.nan)
    for tr_idx, va_idx in kf.split(X):
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], weight=w[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        m = xgb.train(PARAMS, dtr, num_boost_round=1600)  # fixed round count, comparable across weights
        oof[va_idx] = m.predict(dva)
    atp_mask = tour_arr == "atp"
    wta_mask = tour_arr == "wta"
    print(f"atp_weight={atp_weight:.1f}  ATP_oof_ll={ll(y[atp_mask], oof[atp_mask]):.5f}  "
          f"WTA_oof_ll={ll(y[wta_mask], oof[wta_mask]):.5f}  ALL_oof_ll={ll(y, oof):.5f}", flush=True)
