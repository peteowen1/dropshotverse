"""Sanity check Arm F's predictions before trusting the WTA-beats-market
result: degenerate near-0/1 overconfidence would inflate log-loss variance
and could produce a misleadingly good-looking mean on a finite sample even
while being poorly calibrated. Check prediction range, calibration by
decile, and re-run round 7's disagreement-accuracy check."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
hold_full = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_holdout.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
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
model = xgb.train(PARAMS, dtr, num_boost_round=3278)

odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
odds_key["_in"] = True
hold_j = hold_full.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
p = model.predict(xgb.DMatrix(prep_X(hold_j), enable_categorical=True))
y = hold_j["y"].values.astype(int)

print(f"n={len(p):,}  min={p.min():.5f}  max={p.max():.5f}  mean={p.mean():.4f}  "
      f"(true rate should be ~0.5, symmetric hash): true_y_mean={y.mean():.4f}")
print(f"near-degenerate (p<0.01 or p>0.99): {((p < 0.01) | (p > 0.99)).mean():.4%}")

# calibration by decile
bins = np.quantile(p, np.linspace(0, 1, 11))
bins[0], bins[-1] = 0, 1
idx = np.digitize(p, bins) - 1
idx = np.clip(idx, 0, 9)
print("\ncalibration by decile:")
for b in range(10):
    m = idx == b
    if m.sum() == 0:
        continue
    print(f"  decile {b}: n={m.sum():4d}  mean_pred={p[m].mean():.4f}  obs_rate={y[m].mean():.4f}  "
          f"err={p[m].mean() - y[m].mean():+.4f}")

# disagreement-accuracy re-check
j = hold_j[["date", "tour", "winner_uid", "loser_uid"]].copy()
j["y"] = y
j["p_xgb"] = p
j = j.merge(diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"], how="left")
p_mkt_sideA = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
xgb_pick = (p > 0.5).astype(int)
mkt_pick = (p_mkt_sideA > 0.5).astype(int)
d = xgb_pick != mkt_pick
xgb_right = (xgb_pick == y)
n_d = d.sum()
p_hat = xgb_right[d].mean()
se = np.sqrt(p_hat * (1 - p_hat) / n_d)
print(f"\ndisagreements: {n_d:,} ({d.mean():.2%})  "
      f"XGBoost right: {p_hat:.4f}  95%CI=[{p_hat-1.96*se:.4f},{p_hat+1.96*se:.4f}]")
print("(round 7 v2: 0.4458, Arm E: 0.4532)")
for tour in ("atp", "wta"):
    m = d & (j["tour"].values == tour)
    if m.sum() >= 20:
        print(f"  {tour}: n={m.sum()}  xgb_right={xgb_right[m].mean():.4f}")

hold_j.assign(p_xgb=p).to_parquet(rf"{DD}\tune_scratch\xgb_v7_holdout_predictions.parquet")
model.save_model(rf"{DD}\tune_scratch\xgb_v7_model.json")
print("\nsaved model + predictions")
