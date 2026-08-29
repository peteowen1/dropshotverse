import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
OUT = rf"{DD}\tune_scratch\bag_v10_seed_preds.parquet"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
hold_full = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_holdout.parquet")

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner",
           "serve_form_diff_targeted", "return_form_diff_targeted"]
CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
NUM_BOOST_ROUND = 2528
N_BAGS = 5

cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold_full[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


y_tr = train["y"].values.astype(int)
dtrain = xgb.DMatrix(prep_X(train), label=y_tr, enable_categorical=True)
dhold = xgb.DMatrix(prep_X(hold_full), enable_categorical=True)

hold_keys = hold_full[["date", "tour", "winner_uid", "loser_uid", "surface"]].reset_index(drop=True)
out = hold_keys.copy()

for seed in range(N_BAGS):
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
              "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
              "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": seed, "nthread": 8}
    m = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
    out[f"p_seed{seed}"] = m.predict(dhold)
    out.to_parquet(OUT)
    print(f"seed {seed} done, saved", flush=True)

print(f"\nall {N_BAGS} seeds saved to {OUT}")
