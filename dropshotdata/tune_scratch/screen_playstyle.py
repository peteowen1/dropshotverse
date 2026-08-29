import numpy as np
import pandas as pd
import xgboost as xgb

ps = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\playstyle_full.parquet")
tune = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_tune.parquet")

j = tune.merge(ps, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
y = j["y"].values.astype(int)

sw, sl = j["serve_form_w"].values.astype(float), j["serve_form_l"].values.astype(float)
rw, rl = j["return_form_w"].values.astype(float), j["return_form_l"].values.astype(float)
j["serve_form_diff"] = np.where(y == 1, sw - sl, sl - sw)
j["return_form_diff"] = np.where(y == 1, rw - rl, rl - rw)

hand_w = j["winner_hand"].fillna("U")
hand_l = j["loser_hand"].fillna("U")
j["hand_sideA_left"] = np.where(y == 1, (hand_w == "L").astype(int), (hand_l == "L").astype(int))
j["hand_sideB_left"] = np.where(y == 1, (hand_l == "L").astype(int), (hand_w == "L").astype(int))
j["lefty_mismatch"] = (j["hand_sideA_left"] != j["hand_sideB_left"]).astype(int)

print("coverage: serve/return_form", j[["serve_form_diff", "return_form_diff"]].notna().all(axis=1).mean(),
      " hand:", j[["hand_sideA_left", "hand_sideB_left"]].notna().all(axis=1).mean())

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]
TUNED = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
         "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}

CANDS = {
    "return_form_diff": ["return_form_diff"],
    "serve+return_form_diff": ["serve_form_diff", "return_form_diff"],
    "hand_mismatch": ["hand_sideA_left", "hand_sideB_left", "lefty_mismatch"],
    "all": ["serve_form_diff", "return_form_diff", "hand_sideA_left", "hand_sideB_left", "lefty_mismatch"],
}


def cv_ll(numeric_cols, restrict_cols=None):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    for c in BOOLI:
        X[c] = sub[c].astype(int)
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **TUNED}
    cv = xgb.cv(p, d, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


for name, extra in CANDS.items():
    base_same, r0, n0 = cv_ll(BASE, restrict_cols=extra)
    ll, r, n = cv_ll(BASE + extra)
    assert n0 == n
    print(f"BASE+{name:24s} n={n:6d} round={r:4d} cv_logloss={ll:.5f}  "
          f"vs BASE-same-rows={base_same:.5f}  delta={ll - base_same:+.5f}")
