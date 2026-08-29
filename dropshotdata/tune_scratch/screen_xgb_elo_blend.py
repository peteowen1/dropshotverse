"""Ticket 28 round 8: logit-blend XGBoost v2 with Elo. Motivation: round 7
found XGBoost is WRONG more often than right specifically where it disagrees
with the market, and elo_overall_diff already dominates XGBoost's own
feature importance (#1 by a wide margin) - so XGBoost's incremental
disagreements with a simpler, sturdier signal (Elo) may be adding variance,
not information. A logit blend is the standard fix if so. Screened via
5-fold OOF on the tune window (2022-23) - reuses recalibrate_v2.py's OOF
predictions rather than recomputing, then applies to the untouched holdout
ONCE with the tune-selected blend weight (never fit on holdout labels)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

DD = r"C:\dev\dropshotverse\dropshotdata"
tune = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v2_tune.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CATEGORICAL = ["surface", "round", "tourney_level"]
BOOL_AS_INT = ["is_elite"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}
NUM_BOOST_ROUND = 225

cat_levels = {c: sorted(set(tune[c].dropna().unique()) | set(hold[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in BOOL_AS_INT:
        X[c] = df[c].astype(int)
    return X


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


y_tune = tune["y"].values.astype(int)
X_tune = prep_X(tune)
oof = np.full(len(tune), np.nan)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_idx, va_idx in kf.split(X_tune):
    dtr = xgb.DMatrix(X_tune.iloc[tr_idx], label=y_tune[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_tune.iloc[va_idx], enable_categorical=True)
    m = xgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND)
    oof[va_idx] = m.predict(dva)
assert not np.isnan(oof).any()

# elo_logit_pwinner IS Elo's own p_winner on the logit scale (side-A signed),
# per build_xgb_features.py's construction - reuse directly, no re-derivation.
elo_logit_tune = tune["elo_logit_pwinner"].values


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


print("=== tune-window OOF: blend weight sweep (logit space) ===")
oof_logit = logit(oof)
best_w, best_ll = None, None
for w in np.arange(0.0, 1.01, 0.1):
    blend = sigmoid(w * oof_logit + (1 - w) * elo_logit_tune)
    l = ll(y_tune, blend)
    tag = " <- XGBoost alone (w=1.0)" if w == 1.0 else (" <- Elo alone (w=0.0)" if w == 0.0 else "")
    print(f"  w_xgb={w:.1f}  oof_logloss={l:.5f}{tag}")
    if best_ll is None or l < best_ll:
        best_ll, best_w = l, w

print(f"\nbest tune-OOF blend weight: w_xgb={best_w:.1f}  (logloss={best_ll:.5f})")

print("\n=== ONE holdout reveal at the tune-selected weight ===")
elo_logit_hold = hold["elo_logit_pwinner"].values
p_xgb_hold = hold["p_xgb"].values
y_hold = hold["y"].values.astype(int)
p_blend = sigmoid(best_w * logit(p_xgb_hold) + (1 - best_w) * elo_logit_hold)

diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")
j = hold[["date", "tour", "winner_uid", "loser_uid"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]),
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")
p_mkt_sideA = np.where(y_hold == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)

for name, p in (("XGBoost v2 (raw)", p_xgb_hold), (f"Blend (w_xgb={best_w:.1f})", p_blend),
                ("Market", p_mkt_sideA)):
    l = ll(y_hold, p)
    acc = float(((p > 0.5).astype(int) == y_hold).mean())
    print(f"  {name:26s} n={len(y_hold):,}  logloss={l:.5f}  acc={acc:.4f}")
