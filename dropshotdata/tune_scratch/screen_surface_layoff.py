import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
sl = pd.read_parquet(rf"{DD}\tune_scratch\surface_layoff_full.parquet")[
    ["date", "tour", "winner_uid", "loser_uid", "weeks_since_last_surf_w", "weeks_since_last_surf_l"]]

j = train.merge(sl, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
ww, wl = j["weeks_since_last_surf_w"].values.astype(float), j["weeks_since_last_surf_l"].values.astype(float)
diff = np.where(y == 1, ww - wl, wl - ww)
j["surf_layoff_diff"] = diff

atp_grass = ((j["tour"] == "atp") & (j["surface"] == "Grass")).values
j["surf_layoff_diff_atp_grass"] = np.where(atp_grass, diff, 0.0)

print(f"coverage (global): {j['surf_layoff_diff'].notna().mean():.4f}")
print(f"ATP-Grass rows: {atp_grass.sum()}  coverage there: "
      f"{(~np.isnan(diff[atp_grass])).mean():.4f}" if atp_grass.sum() else "n/a")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner",
        "serve_form_diff_targeted", "return_form_diff_targeted"]
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


CANDS = {
    "surf_layoff_diff_global": ["surf_layoff_diff"],
    "surf_layoff_diff_atp_grass": ["surf_layoff_diff_atp_grass"],
}
for name, cols in CANDS.items():
    base_ll, r0, n0, cap0 = cv_ll(BASE, restrict_cols=cols)
    ll1, r1, n1, cap1 = cv_ll(BASE + cols)
    print(f"BASE+{name:26s} n={n1:7,d} round={r1:5d}(cap={cap1}) cv_logloss={ll1:.5f}  "
          f"vs same-rows base={base_ll:.5f}  delta={ll1 - base_ll:+.5f}", flush=True)
