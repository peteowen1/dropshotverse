"""Round 12 close-out: re-run round 7's disagreement-accuracy check against
Arm E (the new best model, 0.59817) instead of v2 (0.6017) - does a
meaningfully better model show a different disagreement pattern, or does
the "no resolved edge anywhere" finding hold up?"""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_1995_2023.parquet")
hold_full = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v2_holdout.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

import xgboost as xgb

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CATEGORICAL = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}
cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold_full[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


y_tr = train["y"].values.astype(int)
dtr = xgb.DMatrix(prep_X(train), label=y_tr, enable_categorical=True)
cv = xgb.cv(PARAMS, dtr, num_boost_round=2000, nfold=5, early_stopping_rounds=50,
            metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
best_round = int(cv["test-logloss-mean"].idxmin()) + 1
model = xgb.train(PARAMS, dtr, num_boost_round=best_round)

odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
odds_key["_in"] = True
hold_j = hold_full.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")

p_xgb = model.predict(xgb.DMatrix(prep_X(hold_j), enable_categorical=True))
y = hold_j["y"].values.astype(int)

j = hold_j[["date", "tour", "winner_uid", "loser_uid", "surface", "tourney_level"]].copy()
j["y"] = y
j["p_xgb"] = p_xgb
j = j.merge(diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"], how="left")
p_mkt_sideA = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)

xgb_pick = (p_xgb > 0.5).astype(int)
mkt_pick = (p_mkt_sideA > 0.5).astype(int)
d = xgb_pick != mkt_pick
xgb_right = (xgb_pick == y)
n_d = d.sum()
p_hat = xgb_right[d].mean()
se = np.sqrt(p_hat * (1 - p_hat) / n_d)
print(f"total n={len(j):,}  disagreements={n_d:,} ({d.mean():.2%})")
print(f"XGBoost (Arm E) right when disagreeing with market: {p_hat:.4f}  "
      f"95% CI=[{p_hat - 1.96*se:.4f}, {p_hat + 1.96*se:.4f}]")
print(f"(round 7's v2 finding, for comparison: 0.4458, CI [0.4145, 0.4771])")

for tour in ("atp", "wta"):
    m = d & (j["tour"].values == tour)
    if m.sum() < 20:
        continue
    print(f"  {tour}: n={m.sum()}  xgb_right={xgb_right[m].mean():.4f}")
