"""Round 14 continued: does fully separating ATP/WTA into two independent
models (not just adding tour as a feature) help more? Full tree-structure
specialization per tour vs a shared model that only splits on tour. CV only
(no holdout touch), each tour's own 5-fold CV on its own rows."""
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


def cv_ll(df, cap=5000, esr=50):
    sub = df.dropna(subset=BASE + CAT).copy()
    X = sub[BASE].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


n_total = len(train.dropna(subset=BASE + CAT))
for tour in ("atp", "wta"):
    sub = train[train["tour"] == tour]
    ll, r, n, cap = cv_ll(sub)
    print(f"{tour}-only model: round={r:5d}(cap={cap}) cv_logloss={ll:.5f}  n={n:,} "
          f"(weight in pooled: {n/n_total:.3f})", flush=True)
