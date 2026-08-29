"""Ticket 28: chase the MARKET (not just Elo), per Pete's /goal 2026-08-29.
Combines three findings, each screened tune-window-only (2022-23) via 5-fold
xgb.cv before ever touching the 2024-25 holdout, per stats-discipline #9:

  1. Hyperparameter retune - max_depth 5->3 plus regularization consistently
     beat the shipped defaults across the top of a 40-point random search
     (cv_logloss 0.5967 -> 0.5935, delta -0.0032, robust across multiple
     depth-3 configs, not a single lucky point - stats-discipline #2's
     "neighbourhood robustness" check).
  2. age_diff (winner_age - loser_age, side-A signed) - the one candidate
     among {rank, rank_points, age, height, hand} with a real apples-to-
     apples CV improvement (rank/rank_points were redundant with Elo,
     height/hand were noise). -0.0014 on top of tuned hparams.
  3. fatigue_diff (prior matches already played in the SAME tournament,
     causally ordered via the same ROUND_RANK table elo_baseline.py/
     hierarchical_model.py use) - genuinely new information neither Elo nor
     hier's own weeks_since_last (inter-tournament layoff) capture.
     -0.0007 on top of tuned hparams.

Combined tune-window CV: 0.5967 -> 0.5930 (-0.0041), no meaningful
redundancy between the three (see tune_scratch/screen_combined.py).

Market's own log-loss on the identical holdout rows is now reported
directly (train_xgb.py never did this - only compared to Elo/hier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
TUNE = rf"{DD}\tune_scratch\xgb_features_v2_tune.parquet"
HOLDOUT = rf"{DD}\tune_scratch\xgb_features_v2_holdout.parquet"
DIAG2 = rf"{DD}\tune_scratch\diag2_enriched.parquet"
ELO_EVAL = rf"{DD}\elo_eval_predictions.parquet"

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff"]
CATEGORICAL = ["surface", "round", "tourney_level"]
BOOL_AS_INT = ["is_elite"]

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "min_child_weight": 5.0,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 0,
    "nthread": -1,
}
NUM_BOOST_ROUND_CAP = 3000
EARLY_STOPPING_ROUNDS = 50
NFOLD = 5


def log(m: str) -> None:
    print(m, flush=True)


def prep_X(df: pd.DataFrame, cat_levels: dict[str, list] | None = None) -> pd.DataFrame:
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        if cat_levels is not None:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        else:
            X[c] = df[c].astype("category")
    for c in BOOL_AS_INT:
        X[c] = df[c].astype(int)
    return X


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "n": len(y),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def main() -> None:
    log("loading v2 tune-window features (2022-2023)...")
    tune = pd.read_parquet(TUNE)
    holdout_raw = pd.read_parquet(HOLDOUT)
    cat_levels = {c: sorted(set(tune[c].dropna().unique()) | set(holdout_raw[c].dropna().unique()))
                  for c in CATEGORICAL}
    X_tune = prep_X(tune, cat_levels)
    y_tune = tune["y"].values.astype(int)
    log(f"  tune rows: {len(tune):,}  columns: {list(X_tune.columns)}")

    dtrain = xgb.DMatrix(X_tune, label=y_tune, enable_categorical=True)

    log(f"\nrunning xgb.cv (nfold={NFOLD}, early_stopping_rounds={EARLY_STOPPING_ROUNDS}, "
        f"cap={NUM_BOOST_ROUND_CAP}, tuned params)...")
    cv_results = xgb.cv(
        params=PARAMS, dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND_CAP, nfold=NFOLD,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, metrics=("logloss",), seed=0,
        as_pandas=True, verbose_eval=False,
    )
    best_round = int(cv_results["test-logloss-mean"].idxmin()) + 1
    best_cv_ll = float(cv_results["test-logloss-mean"].iloc[best_round - 1])
    log(f"  selected num_boost_round={best_round}  (cv test-logloss={best_cv_ll:.5f})")

    log(f"\nfitting final model on ALL of tune (2022-23) with num_boost_round={best_round}...")
    final_model = xgb.train(PARAMS, dtrain, num_boost_round=best_round)

    log("\nrestricting holdout-window features (2024-2025) to the market-odds-joined subset...")
    holdout = holdout_raw
    diag2 = pd.read_parquet(DIAG2)
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in_odds"] = True
    holdout_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    holdout_j = holdout_j[holdout_j["_in_odds"] == True].drop(columns=["_in_odds"])  # noqa: E712
    holdout_j = holdout_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    log(f"  holdout rows after odds-join restriction: {len(holdout_j):,} "
        f"(prior sessions' basis: n=8,593/8,594)")

    X_hold = prep_X(holdout_j, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    p_xgb = final_model.predict(dhold)
    y_hold = holdout_j["y"].values.astype(int)

    elo = pd.read_parquet(ELO_EVAL)
    elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]]
    elo = elo.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    cmp_df = holdout_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
        elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cmp_df = cmp_df.merge(
        diag2[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc", "p_mkt"]].drop_duplicates(
            subset=["date", "tour", "winner_uid", "loser_uid"]),
        on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cmp_df["p_xgb"] = p_xgb
    log(f"  comparison coverage: elo notna={cmp_df['p_winner'].notna().sum():,}, "
        f"hier notna={cmp_df['p_winner_hier_mc'].notna().sum():,}, "
        f"mkt notna={cmp_df['p_mkt'].notna().sum():,} / {len(cmp_df):,}")

    y_arr = cmp_df["y"].values.astype(int)
    p_elo_sideA = np.where(y_arr == 1, cmp_df["p_winner"].values, 1 - cmp_df["p_winner"].values)
    p_hier_sideA = np.where(y_arr == 1, cmp_df["p_winner_hier_mc"].values, 1 - cmp_df["p_winner_hier_mc"].values)
    p_mkt_sideA = np.where(y_arr == 1, cmp_df["p_mkt"].values, 1 - cmp_df["p_mkt"].values)

    def report(label: str, mask: np.ndarray) -> None:
        n = int(mask.sum())
        log(f"\n  --- {label} (n={n:,}) ---")
        for name, arr in (("XGBoost v2", p_xgb[mask]), ("Elo (post-fix)", p_elo_sideA[mask]),
                          ("Hierarchical (v25/mc60)", p_hier_sideA[mask]), ("MARKET", p_mkt_sideA[mask])):
            s = score(y_arr[mask], arr)
            log(f"    {name:26s} logloss={s['logloss']:.4f}  acc={s['accuracy']:.4f}  brier={s['brier']:.4f}")

    tour_arr = cmp_df["tour"].values
    log("\n=== HOLDOUT RESULTS (2024-2025, market-odds-joined, n={:,}) ===".format(len(cmp_df)))
    report("ALL", np.ones(len(cmp_df), dtype=bool))
    report("ATP", tour_arr == "atp")
    report("WTA", tour_arr == "wta")

    log("\n=== FEATURE IMPORTANCE (gain) ===")
    gain = final_model.get_score(importance_type="gain")
    total_gain = final_model.get_score(importance_type="total_gain")
    weight = final_model.get_score(importance_type="weight")
    imp = pd.DataFrame({"feature": list(gain.keys()), "avg_gain": list(gain.values())}).set_index("feature")
    imp["total_gain"] = pd.Series(total_gain)
    imp["n_splits"] = pd.Series(weight)
    imp = imp.sort_values("total_gain", ascending=False)
    log(imp.to_string())

    final_model.save_model(rf"{DD}\tune_scratch\xgb_v2_model.json")
    holdout_j.assign(p_xgb=p_xgb).to_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")
    imp.to_csv(rf"{DD}\tune_scratch\xgb_v2_feature_importance.csv")
    log(f"\nsaved model, holdout predictions, and feature importance to tune_scratch\\xgb_v2_*")


if __name__ == "__main__":
    main()
