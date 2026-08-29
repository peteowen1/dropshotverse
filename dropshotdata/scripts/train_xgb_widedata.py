"""Ticket 28 round 12: does training volume beat hier's extra signal?
Arm A: Elo-only features (no hier_dmean/dvar - never computed outside
2022-23/2024-25), trained on 2015-2023 (9 years, ~45.8K rows).
Arm B: SAME Elo-only feature set, but restricted to 2022-2023 only (2 years,
a subset of Arm A's own rows) - isolates the data-VOLUME axis from the
hier-REMOVAL axis (stats-discipline #5: never let two things vary at once).
Both scored ONCE on the same frozen 2024-25 market-odds-joined holdout v2
uses, for direct comparison to v2's 0.6017 (which HAS hier features, 2yrs).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
WIDE = rf"{DD}\tune_scratch\xgb_features_widedata_2015_2023.parquet"
HOLDOUT_V2 = rf"{DD}\tune_scratch\xgb_features_v2_holdout.parquet"   # same holdout pool, has all needed cols
DIAG2 = rf"{DD}\tune_scratch\diag2_enriched.parquet"
ELO_EVAL = rf"{DD}\elo_eval_predictions.parquet"

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner",
           "weeks_since_last_diff", "n_prior_diff", "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CATEGORICAL = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}


def log(m):
    print(m, flush=True)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prep_X(df, cat_levels):
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


wide = pd.read_parquet(WIDE)
hold_full = pd.read_parquet(HOLDOUT_V2)
cat_levels = {c: sorted(set(wide[c].dropna().unique()) | set(hold_full[c].dropna().unique())) for c in CATEGORICAL}

hold_odds = pd.read_parquet(DIAG2)
odds_key = hold_odds.loc[hold_odds["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
odds_key["_in"] = True
hold_j = hold_full.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
log(f"holdout n={len(hold_j):,}")

elo = pd.read_parquet(ELO_EVAL)
elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])
cmp_base = hold_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
    elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left").merge(
    hold_odds[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
    how="left")
y_hold_full = cmp_base["y"].values.astype(int)
p_elo = np.where(y_hold_full == 1, cmp_base["p_winner"].values, 1 - cmp_base["p_winner"].values)
p_mkt = np.where(y_hold_full == 1, cmp_base["p_mkt"].values, 1 - cmp_base["p_mkt"].values)
log(f"Elo (post-fix) on this holdout: {ll(y_hold_full, p_elo):.5f}")
log(f"Market on this holdout: {ll(y_hold_full, p_mkt):.5f}")


def run_arm(train_df, label):
    y_tr = train_df["y"].values.astype(int)
    X_tr = prep_X(train_df, cat_levels)
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    cv = xgb.cv(PARAMS, dtr, num_boost_round=3000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best_round = int(cv["test-logloss-mean"].idxmin()) + 1
    best_cv = float(cv["test-logloss-mean"].iloc[best_round - 1])
    log(f"[{label}] n_train={len(train_df):,}  cv round={best_round}  cv_logloss={best_cv:.5f}")

    model = xgb.train(PARAMS, dtr, num_boost_round=best_round)
    X_hold = prep_X(hold_j, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    p = model.predict(dhold)
    y_h = hold_j["y"].values.astype(int)
    log(f"[{label}] HOLDOUT logloss={ll(y_h, p):.5f}  acc={((p > 0.5).astype(int) == y_h).mean():.4f}")
    return model


log("\n=== Arm A: Elo-only features, 2015-2023 (9yr, wide data) ===")
run_arm(wide, "Arm A: 9yr wide")

log("\n=== Arm B: Elo-only features, 2022-2023 ONLY (2yr, same feature set as A) ===")
wide_2yr = wide[wide["date"].dt.year >= 2022]
run_arm(wide_2yr, "Arm B: 2yr narrow")

log("\n(for reference: v2 - Elo+hier features, 2yr - reported holdout logloss 0.6017)")
