"""Ticket 28 round 11: bag v2 across multiple seeds (different row/column
subsampling draws), average predictions. A genuinely different lever from
everything tried so far - variance reduction via ensembling, not a new
feature, calibration fix, or bias correction. Cheap: same features, same
tuned hyperparameters, just refit N times with different `seed` and average.
Screened via tune-window OOF (mirrors recalibrate_v2.py's harness), ONE
holdout reveal."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

DD = r"C:\dev\dropshotverse\dropshotdata"
tune = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v2_tune.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")
hold_full = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v2_holdout.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CATEGORICAL = ["surface", "round", "tourney_level"]
BOOL_AS_INT = ["is_elite"]
BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
               "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
               "reg_alpha": 0.1, "reg_lambda": 1.0, "nthread": -1}
NUM_BOOST_ROUND = 225
N_BAGS = 10

cat_levels = {c: sorted(set(tune[c].dropna().unique()) | set(hold_full[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in BOOL_AS_INT:
        X[c] = df[c].astype(int)
    return X


y_tune = tune["y"].values.astype(int)
X_tune = prep_X(tune)

# tune-window OOF, bagged: for each of 5 folds, average N_BAGS models (each a
# different seed) fit on that fold's training rows, predict on its val rows.
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_single = np.full(len(tune), np.nan)
oof_bagged = np.full(len(tune), np.nan)
for tr_idx, va_idx in kf.split(X_tune):
    dtr = xgb.DMatrix(X_tune.iloc[tr_idx], label=y_tune[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_tune.iloc[va_idx], enable_categorical=True)
    preds = []
    for b in range(N_BAGS):
        m = xgb.train({**BASE_PARAMS, "seed": b}, dtr, num_boost_round=NUM_BOOST_ROUND)
        preds.append(m.predict(dva))
    preds = np.array(preds)
    oof_single[va_idx] = preds[0]          # single-seed baseline (seed=0, matches v2's own PARAMS.seed=0)
    oof_bagged[va_idx] = preds.mean(axis=0)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


print(f"tune OOF single-seed: {ll(y_tune, oof_single):.5f}")
print(f"tune OOF {N_BAGS}-bag average: {ll(y_tune, oof_bagged):.5f}  "
      f"delta={ll(y_tune, oof_bagged) - ll(y_tune, oof_single):+.5f}")

print("\n=== ONE holdout reveal ===")
X_hold = prep_X(hold_full)
dtrain_full = xgb.DMatrix(X_tune, label=y_tune, enable_categorical=True)
dhold = xgb.DMatrix(X_hold, enable_categorical=True)
preds_hold = []
for b in range(N_BAGS):
    m = xgb.train({**BASE_PARAMS, "seed": b}, dtrain_full, num_boost_round=NUM_BOOST_ROUND)
    preds_hold.append(m.predict(dhold))
preds_hold = np.array(preds_hold)
p_single = preds_hold[0]
p_bagged = preds_hold.mean(axis=0)

# restrict to the market-odds-joined subset, matching hold's own rows
key = ["date", "tour", "winner_uid", "loser_uid"]
hold_full2 = hold_full.reset_index(drop=True)
hold_full2["p_single"] = p_single
hold_full2["p_bagged"] = p_bagged
j = hold[key + ["y"]].merge(hold_full2[key + ["p_single", "p_bagged"]], on=key, how="left")
y_h = j["y"].values.astype(int)
print(f"single-seed holdout: {ll(y_h, j['p_single'].values):.5f}  (v2's own reported: 0.60171)")
print(f"{N_BAGS}-bag holdout:    {ll(y_h, j['p_bagged'].values):.5f}  "
      f"delta={ll(y_h, j['p_bagged'].values) - ll(y_h, j['p_single'].values):+.5f}")
