"""Ticket 28 round 14, Arm G: v7's feature set (Elo + hier + hier_logit_pwinner
+ engineered features) plus `tour` as an explicit categorical, trained on
1995-2023 (29yr, 118,912 rows) - hier state available for the wide window via
regen_hier_wide1995.py (frozen v25 hyperparameters, no retuning). ONE holdout
reveal, compared to Arm F's 0.59017.

Note: TRAIN/HOLDOUT below point at xgb_features_v7_{train,holdout}.parquet,
which were overwritten in place by later round-13 work (build_v7_features.py)
to hold the 1995-2023/118,912-row build - not the 2015-2023/43,862-row build
this docstring described in an earlier version of this file. Corrected here;
see tune_scratch/train_xgb_v8.log for the actual run's row count."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
TRAIN = rf"{DD}\tune_scratch\xgb_features_v7_train.parquet"
HOLDOUT = rf"{DD}\tune_scratch\xgb_features_v7_holdout.parquet"
DIAG2 = rf"{DD}\tune_scratch\diag2_enriched.parquet"
ELO_EVAL = rf"{DD}\elo_eval_predictions.parquet"

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def log(m):
    print(m, flush=True)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


train = pd.read_parquet(TRAIN)
hold_full = pd.read_parquet(HOLDOUT)
cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold_full[c].dropna().unique())) for c in CATEGORICAL}


def prep_X(df):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


y_tr = train["y"].values.astype(int)
X_tr = prep_X(train)
log(f"train rows: {len(train):,}  years {train['date'].dt.year.min()}-{train['date'].dt.year.max()}")

dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
# CV round count already known from screen_tour_feature.py's own 5-fold search
# (round=3161, cv_logloss=0.57325, properly converged, not capped) - skip
# re-running that expensive CV here (memory pressure caused two prior crashes
# this round) and just fit once at the already-established round count.
best_round = 3161
log(f"using pre-established cv round={best_round} (cv_logloss=0.57325, from screen_tour_feature.py)")

model = xgb.train(PARAMS, dtr, num_boost_round=best_round)

diag2 = pd.read_parquet(DIAG2)
odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
odds_key["_in"] = True
hold_j = hold_full.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
log(f"holdout n={len(hold_j):,}")

X_hold = prep_X(hold_j)
dhold = xgb.DMatrix(X_hold, enable_categorical=True)
p = model.predict(dhold)
y_h = hold_j["y"].values.astype(int)

elo = pd.read_parquet(ELO_EVAL)
elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])
cmp_df = hold_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
    elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left").merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
    how="left")
p_mkt = np.where(y_h == 1, cmp_df["p_mkt"].values, 1 - cmp_df["p_mkt"].values)
p_elo = np.where(y_h == 1, cmp_df["p_winner"].values, 1 - cmp_df["p_winner"].values)

tour_arr = hold_j["tour"].values
for label, mask in (("ALL", np.ones(len(hold_j), dtype=bool)), ("ATP", tour_arr == "atp"),
                     ("WTA", tour_arr == "wta")):
    log(f"\n--- {label} (n={mask.sum():,}) ---")
    log(f"  Arm G (+tour feature) logloss={ll(y_h[mask], p[mask]):.5f}  "
        f"acc={((p[mask] > 0.5).astype(int) == y_h[mask]).mean():.4f}")
    log(f"  Elo                    logloss={ll(y_h[mask], p_elo[mask]):.5f}")
    log(f"  Market                 logloss={ll(y_h[mask], p_mkt[mask]):.5f}")

log("\n(reference: v2=0.6017, ..., Arm E=0.59817, Arm F=0.59017)")

log("\n=== feature importance (gain) ===")
gain = model.get_score(importance_type="gain")
total_gain = model.get_score(importance_type="total_gain")
imp = pd.DataFrame({"feature": list(gain.keys()), "total_gain": list(total_gain.values())}).sort_values(
    "total_gain", ascending=False)
log(imp.to_string())

hold_j.assign(p_xgb=p).to_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")
model.save_model(rf"{DD}\tune_scratch\xgb_v8_model.json")
log("\nsaved model + predictions")
