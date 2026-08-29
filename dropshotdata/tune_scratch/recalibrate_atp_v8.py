"""Round 14 continued: ATP-specific isotonic recalibration. Round 4 rejected
isotonic on v2 (a much weaker, pre-hier_logit_pwinner model) - worth
retrying now that the model is far better and ATP is a specifically
identified laggard. Proper OOF fit (5-fold, tune/train rows only) per
calibration-check's hard constraint - never fit on holdout labels.
nthread bounded (this round already crashed twice from memory pressure on
unbounded 5-fold CV searches)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}
NUM_BOOST_ROUND = 3161

cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


y_tr = train["y"].values.astype(int)
X_tr = prep_X(train)
oof = np.full(len(train), np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    dtr = xgb.DMatrix(X_tr.iloc[tr_idx], label=y_tr[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_tr.iloc[va_idx], enable_categorical=True)
    m = xgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND)
    oof[va_idx] = m.predict(dva)
    print(f"fold {i} done", flush=True)

assert not np.isnan(oof).any()


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


oof_ll = ll(y_tr, oof)
print(f"\nOOF (train, 5-fold) logloss: {oof_ll:.5f}")

y_hold = hold["y"].values.astype(int)
p_raw = hold["p_xgb"].values
tour_tr = train["tour"].values
tour_hold = hold["tour"].values

for scope, tr_mask_fn, hold_mask_fn in (
    ("ATP-only isotonic", lambda: tour_tr == "atp", lambda: tour_hold == "atp"),
    ("Global isotonic (all tours pooled)", lambda: np.ones(len(train), dtype=bool),
     lambda: np.ones(len(hold), dtype=bool)),
):
    trm = tr_mask_fn()
    hom = hold_mask_fn()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(oof[trm], y_tr[trm])
    p_cal = iso.predict(p_raw[hom])
    print(f"\n{scope}: fit n={trm.sum():,}")
    print(f"  raw       n={hom.sum():,} logloss={ll(y_hold[hom], p_raw[hom]):.5f}")
    print(f"  recal     n={hom.sum():,} logloss={ll(y_hold[hom], p_cal):.5f}")
