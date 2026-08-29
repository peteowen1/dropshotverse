import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
NUM_BOOST_ROUND = 3161
N_BAGS = 15

cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


y_tr = train["y"].values.astype(int)
X_tr = prep_X(train)
dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)

hold_full = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_holdout.parquet")
X_hold = prep_X(hold_full)
dhold = xgb.DMatrix(X_hold, enable_categorical=True)

preds = []
for seed in range(N_BAGS):
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
              "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
              "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": seed, "nthread": 8}
    m = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
    preds.append(m.predict(dhold))
    print(f"seed {seed} done", flush=True)

preds = np.array(preds)
hold_full2 = hold_full.reset_index(drop=True)
for i in range(N_BAGS):
    hold_full2[f"p_seed{i}"] = preds[i]
hold_full2["p_bagged"] = preds.mean(axis=0)

key = ["date", "tour", "winner_uid", "loser_uid"]
j = hold[key + ["y"]].merge(hold_full2[key + ["p_seed0", "p_bagged"]], on=key, how="left")
y_h = j["y"].values.astype(int)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


print(f"\nsingle-seed (seed=0, cross-check vs Arm G's 0.58992): {ll(y_h, j['p_seed0'].values):.5f}")
print(f"5-bag average: {ll(y_h, j['p_bagged'].values):.5f}  delta={ll(y_h, j['p_bagged'].values) - ll(y_h, j['p_seed0'].values):+.5f}")

tour_j = j["tour"].values
for tour in ("atp", "wta"):
    m = tour_j == tour
    print(f"  {tour}: single={ll(y_h[m], j['p_seed0'].values[m]):.5f}  bagged={ll(y_h[m], j['p_bagged'].values[m]):.5f}")
