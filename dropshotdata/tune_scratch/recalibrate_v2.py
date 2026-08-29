"""Ticket 28 round 4: v2's own calibration is real and material - ECE 0.0160
vs the market's 0.0061 (calibration_v2.py), non-monotonic bin errors (+0.047
at [0,0.1), -0.030 at [0.6,0.7), +0.052 at [0.7,0.8)). Per calibration-check
skill's hard constraints: isotonic/Platt fit ONLY inside training folds, never
on the final test labels. Manual 5-fold OOF on the TUNE window (2022-23) only,
fitted isotonic map applied to the already-computed 2024-25 holdout
predictions - the holdout's own labels are never touched by the fit.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
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
NUM_BOOST_ROUND = 225  # matches train_xgb_v2.py's selected round, held fixed here

cat_levels = {c: sorted(set(tune[c].dropna().unique()) | set(hold[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in BOOL_AS_INT:
        X[c] = df[c].astype(int)
    return X


y_tune = tune["y"].values.astype(int)
X_tune = prep_X(tune)
oof = np.full(len(tune), np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tune)):
    dtr = xgb.DMatrix(X_tune.iloc[tr_idx], label=y_tune[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_tune.iloc[va_idx], enable_categorical=True)
    m = xgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND)
    oof[va_idx] = m.predict(dva)
    print(f"fold {fold}: n_val={len(va_idx)}", flush=True)

assert not np.isnan(oof).any()
oof_ll = float(-np.mean(y_tune * np.log(np.clip(oof, 1e-9, 1 - 1e-9)) +
                         (1 - y_tune) * np.log(np.clip(1 - oof, 1e-9, 1 - 1e-9))))
print(f"\nOOF (tune, 5-fold) logloss: {oof_ll:.5f}  (vs train_xgb_v2.py's reported cv=0.59304 - "
      f"different fold seed/impl, should be close)")

iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
iso.fit(oof, y_tune)

# apply to the ALREADY-COMPUTED holdout predictions - fit never saw these rows or labels
y_hold = hold["y"].values.astype(int)
p_raw = hold["p_xgb"].values
p_cal = iso.predict(p_raw)


def score(y, p, label):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(((p > 0.5).astype(int) == y).mean())
    brier = float(np.mean((p - y) ** 2))
    print(f"{label:30s} n={len(y):6,d}  logloss={ll:.5f}  acc={acc:.4f}  brier={brier:.4f}")


print("\n=== holdout, raw vs isotonic-recalibrated (fit on tune OOF only) ===")
score(y_hold, p_raw, "XGBoost v2 (raw)")
score(y_hold, p_cal, "XGBoost v2 (isotonic-recal)")

diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")
j = hold[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]),
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")
p_mkt_sideA = np.where(y_hold == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
score(y_hold, p_mkt_sideA, "Market")

pd.DataFrame({"date": hold["date"], "tour": hold["tour"], "y": y_hold,
              "p_xgb_raw": p_raw, "p_xgb_cal": p_cal}).to_parquet(
    rf"{DD}\tune_scratch\xgb_v2_calibrated_holdout.parquet")
print("\nsaved xgb_v2_calibrated_holdout.parquet")
