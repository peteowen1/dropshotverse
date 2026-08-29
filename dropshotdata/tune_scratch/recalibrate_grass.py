"""Round 18: round 14's isotonic recalibration was fit on ALL ATP OOF
predictions pooled - dominated by hard/clay (the large majority of ATP
rows), which would dilute any grass-specific miscalibration. Grass has
~8,000 training rows, enough for a proper nested OOF fit restricted to
grass only (ATP-grass specifically, and separately ATP+WTA-grass pooled for
more data, since round 15 found grass dynamics are qualitatively similar
across tours even though ATP's residual gap is bigger). Never touches
holdout labels - proper nested calibration per calibration-check's hard
constraint.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v10_holdout_predictions.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner",
           "serve_form_diff_targeted", "return_form_diff_targeted"]
CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}
NUM_BOOST_ROUND = 2528

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


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


oof_ll = ll(y_tr, oof)
print(f"\nOOF (train, 5-fold) logloss: {oof_ll:.5f}")

y_hold = hold["y"].values.astype(int)
p_raw = hold["p_xgb"].values
tour_hold = hold["tour"].values
surf_hold = hold["surface"].values
tour_tr = train["tour"].values
surf_tr = train["surface"].values
round_tr = train["round"].values
round_hold = hold["round"].values

scopes = {
    "ATP-Grass only": (lambda: (tour_tr == "atp") & (surf_tr == "Grass"),
                        lambda: (tour_hold == "atp") & (surf_hold == "Grass")),
    "ATP+WTA Grass pooled": (lambda: surf_tr == "Grass", lambda: surf_hold == "Grass"),
    "ATP Grass R128 only (fit)": (lambda: (tour_tr == "atp") & (surf_tr == "Grass") & (round_tr == "R128"),
                                    lambda: (tour_hold == "atp") & (surf_hold == "Grass")),
}

for scope, (tr_mask_fn, hold_mask_fn) in scopes.items():
    trm = tr_mask_fn()
    hom = hold_mask_fn()
    if trm.sum() < 50:
        print(f"\n{scope}: fit n={trm.sum()} too small, skipping")
        continue
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(oof[trm], y_tr[trm])
    p_cal = iso.predict(p_raw[hom])
    print(f"\n{scope}: fit n={trm.sum():,}")
    print(f"  raw       n={hom.sum():,} logloss={ll(y_hold[hom], p_raw[hom]):.5f}")
    print(f"  recal     n={hom.sum():,} logloss={ll(y_hold[hom], p_cal):.5f}")

# also report the ATP-Grass OOF calibration curve itself (bin errors), to see
# WHICH direction (over/under) the miscalibration runs before trusting isotonic
atp_grass_tr = (tour_tr == "atp") & (surf_tr == "Grass")
p_g = oof[atp_grass_tr]
y_g = y_tr[atp_grass_tr]
print(f"\nATP-Grass OOF calibration (train, n={atp_grass_tr.sum()}):")
bins = np.linspace(0, 1, 6)
idx = np.clip(np.digitize(p_g, bins) - 1, 0, 4)
for b in range(5):
    m = idx == b
    if m.sum() < 10:
        continue
    print(f"  [{bins[b]:.1f},{bins[b+1]:.1f})  n={m.sum():4d}  mean_pred={p_g[m].mean():.4f}  "
          f"obs_rate={y_g[m].mean():.4f}  err={p_g[m].mean() - y_g[m].mean():+.4f}")
