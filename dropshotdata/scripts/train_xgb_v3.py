"""Ticket 28 round 9: v3 adds serve_form_diff + return_form_diff (Pete's own
"server vs returner" archetype suggestion) to v2's feature set. Anchor check
(Isner vs Djokovic) confirmed the underlying signal is real and distinct from
aggregate win/loss ratings: Isner serve_form=0.72/return_form=0.30 vs
Djokovic serve_form=0.69/return_form=0.43 - near-identical serve level, very
different return games. Neither serve_form alone (round 3) nor return_form
alone screened as useful, but TOGETHER they gave a real tune-window CV
improvement (0.59963 -> 0.59930 on identical rows, delta -0.00033) - the
play-style interaction, not either half alone. See issues/28 round 9.

Market's own log-loss on the identical holdout rows is reported directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
TUNE = rf"{DD}\tune_scratch\xgb_features_v3_tune.parquet"
HOLDOUT = rf"{DD}\tune_scratch\xgb_features_v3_holdout.parquet"
DIAG2 = rf"{DD}\tune_scratch\diag2_enriched.parquet"
ELO_EVAL = rf"{DD}\elo_eval_predictions.parquet"

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of", "age_diff", "fatigue_diff", "serve_form_diff", "return_form_diff"]
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
        for name, arr in (("XGBoost v3", p_xgb[mask]), ("Elo (post-fix)", p_elo_sideA[mask]),
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

    final_model.save_model(rf"{DD}\tune_scratch\xgb_v3_model.json")
    holdout_j.assign(p_xgb=p_xgb).to_parquet(rf"{DD}\tune_scratch\xgb_v3_holdout_predictions.parquet")
    imp.to_csv(rf"{DD}\tune_scratch\xgb_v3_feature_importance.csv")
    log(f"\nsaved model, holdout predictions, and feature importance to tune_scratch\\xgb_v2_*")


if __name__ == "__main__":
    main()
