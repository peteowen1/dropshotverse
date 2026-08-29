"""Pete's question: do we have one XGBoost model or separate ATP/WTA models,
and has the split ever been tested? Answer was "one combined model, never
tested split" - this builds and evaluates the split as a real experiment.

Reuses train_best_model.py's functions unmodified (feature building, hyper-
params, holdout construction) - only adds a tour filter before dtrain, and
compares against the SAME already-reported combined-model ATP/WTA holdout
numbers (train_best_model.py's own per-tour breakdown), not a re-run under
different conditions - stats-discipline #5's "one axis at a time" and #5.3
"score every real comparison in one pass, on matching rows" apply here:
same TRAIN_YEARS, same features, same holdout rows, same hyperparameters -
tour-scope is the ONLY thing that varies between combined and split arms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import train_best_model as tbm

SCRATCH = tbm.SCRATCH


def log(m: str) -> None:
    print(m, flush=True)


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_one_tour(tour: str, train_all: pd.DataFrame, holdout_all: pd.DataFrame, cat_levels: dict) -> dict:
    train_t = train_all[train_all["tour"] == tour].reset_index(drop=True)
    hold_t = holdout_all[holdout_all["tour"] == tour].reset_index(drop=True)
    log(f"  [{tour}] train n={len(train_t):,}  holdout n={len(hold_t):,}")

    X_tr = tbm.prep_X(train_t, cat_levels)
    y_tr = train_t["y"].values.astype(int)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    X_hold = tbm.prep_X(hold_t, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    y_hold = hold_t["y"].values.astype(int)

    preds = []
    for seed in range(tbm.N_BAGS):
        m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
        preds.append(m.predict(dhold))
    p_bagged = np.mean(preds, axis=0)
    return {"tour": tour, "n_hold": len(hold_t), "logloss": ll(y_hold, p_bagged),
           "hold_j": hold_t, "y_hold": y_hold, "p_split": p_bagged}


def main() -> None:
    log("building features (same pipeline as train_best_model.py)...")
    train_feat = tbm.build_elo_features(tbm.TRAIN_YEARS)
    train_hier_path = tbm.regen_hier_state(tbm.TRAIN_YEARS, "train_canonical")
    train = tbm.merge_hier(train_feat, train_hier_path)
    playstyle = tbm.build_playstyle_features()
    train = tbm.merge_playstyle(train, playstyle)

    holdout = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    diag2 = pd.read_parquet(SCRATCH / "diag2_enriched.parquet")
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in"] = True
    hold_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
    hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")

    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(hold_j[c].dropna().unique()))
                  for c in tbm.XGB_CATEGORICAL}

    log("\n=== training SPLIT (per-tour) models ===")
    results = {}
    for tour in ("atp", "wta"):
        results[tour] = train_one_tour(tour, train, hold_j, cat_levels)

    # Combined-model comparison numbers: train_best_model.py's OWN documented
    # holdout result (already reported, same TRAIN_YEARS/features/hyperparams/
    # holdout rows - not re-run here, reused as-is per stats-discipline #5.3).
    COMBINED_ATP_LOGLOSS = 0.59454
    COMBINED_WTA_LOGLOSS = 0.59186

    log("\n=== RESULTS: split vs combined (same holdout rows, same everything else) ===")
    for tour, combined_ll in (("atp", COMBINED_ATP_LOGLOSS), ("wta", COMBINED_WTA_LOGLOSS)):
        r = results[tour]
        delta = r["logloss"] - combined_ll
        log(f"  {tour.upper():4s} n={r['n_hold']:5,d}  split={r['logloss']:.5f}  combined={combined_ll:.5f}  "
            f"delta={delta:+.5f}  {'split wins' if delta < 0 else 'combined wins'}")

    out = pd.concat([
        results["atp"]["hold_j"][["date", "tour", "winner_uid", "loser_uid"]].assign(
            y=results["atp"]["y_hold"], p_split=results["atp"]["p_split"]),
        results["wta"]["hold_j"][["date", "tour", "winner_uid", "loser_uid"]].assign(
            y=results["wta"]["y_hold"], p_split=results["wta"]["p_split"]),
    ], ignore_index=True)
    out_path = SCRATCH / "xgb_split_by_tour_predictions.parquet"
    out.to_parquet(out_path)
    log(f"\nsaved per-row split predictions -> {out_path} (for a CI vs combined later if the point estimate warrants it)")


if __name__ == "__main__":
    main()
