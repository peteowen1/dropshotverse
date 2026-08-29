import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
ps = pd.read_parquet(rf"{DD}\tune_scratch\playstyle_full.parquet")  # round 9's serve/return form, full archive

j = train.merge(ps, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
y = j["y"].values.astype(int)
sw, sl = j["serve_form_w"].values.astype(float), j["serve_form_l"].values.astype(float)
rw, rl = j["return_form_w"].values.astype(float), j["return_form_l"].values.astype(float)
j["serve_form_diff"] = np.where(y == 1, sw - sl, sl - sw)
j["return_form_diff"] = np.where(y == 1, rw - rl, rl - rw)

print(f"grass rows: {(j['surface']=='Grass').sum()}, coverage on grass: "
      f"{j.loc[j['surface']=='Grass', ['serve_form_diff','return_form_diff']].notna().mean().to_dict()}")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def cv_ll(numeric_cols, restrict_cols=None, cap=4000, esr=50):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


base_same, r0, n0, cap0 = cv_ll(BASE, restrict_cols=["serve_form_diff", "return_form_diff"])
ll_, r, n, cap1 = cv_ll(BASE + ["serve_form_diff", "return_form_diff"])
print(f"\nBASE (same rows)  n={n0} round={r0}(cap={cap0}) cv_logloss={base_same:.5f}")
print(f"BASE+serve+return n={n} round={r}(cap={cap1}) cv_logloss={ll_:.5f}  delta={ll_ - base_same:+.5f}")
