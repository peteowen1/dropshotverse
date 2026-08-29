"""Step 1 of Pete's '1 then 2 then 3 then 4': CI on variant D (match+set+
game) vs variant A (match only, ticket 28's original) before propagating D
anywhere. Needs both variants' predictions on the IDENTICAL holdout row set,
paired - ticket 28's original run never saved per-row predictions, so A is
retrained here alongside D (same shared base features/merges as
train_xgb_granularity_variants.py, so both variants score the exact same
rows - required for a valid paired comparison, stats-discipline #5.3).

Bootstrap unit: individual match rows (holdout matches are between distinct
players/events, not multiple draws from one shared simulated bracket like
the earlier tournament-sim CI - row-level resampling is the right unit here,
matching how ticket 28's own comparisons were done)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import train_best_model as tbm
from train_xgb_granularity_variants import GAME_ELO_COLS, SET_ELO_COLS, merge_granularity_elo, prep_X_variant

SCRATCH = tbm.SCRATCH
VARIANTS = {"A_match_only": tbm.XGB_NUMERIC, "D_match_set_game": tbm.XGB_NUMERIC + SET_ELO_COLS + GAME_ELO_COLS}


def log(m: str) -> None:
    print(m, flush=True)


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    log("building shared base features...")
    train_feat = tbm.build_elo_features(tbm.TRAIN_YEARS)
    train_hier_path = tbm.regen_hier_state(tbm.TRAIN_YEARS, "train_canonical")
    train = tbm.merge_hier(train_feat, train_hier_path)
    playstyle = tbm.build_playstyle_features()
    train = tbm.merge_playstyle(train, playstyle)
    train = merge_granularity_elo(train, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    train = merge_granularity_elo(train, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")

    holdout = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    holdout = merge_granularity_elo(holdout, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    holdout = merge_granularity_elo(holdout, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")

    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(holdout[c].dropna().unique()))
                  for c in tbm.XGB_CATEGORICAL}

    diag2 = pd.read_parquet(SCRATCH / "diag2_enriched.parquet")
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in"] = True
    hold_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
    hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    y_hold = hold_j["y"].values.astype(int)

    preds = {}
    for name, cols in VARIANTS.items():
        log(f"\n=== training {name} ===")
        X_tr = prep_X_variant(train, cols, cat_levels)
        y_tr = train["y"].values.astype(int)
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        X_hold = prep_X_variant(hold_j, cols, cat_levels)
        dhold = xgb.DMatrix(X_hold, enable_categorical=True)
        bag = []
        for seed in range(tbm.N_BAGS):
            m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
            bag.append(m.predict(dhold))
        preds[name] = np.mean(bag, axis=0)
        log(f"  {name}: ALL log-loss={ll(y_hold, preds[name]):.5f}")

    p_a, p_d = preds["A_match_only"], preds["D_match_set_game"]
    ll_a_row = -(y_hold * np.log(np.clip(p_a, 1e-9, 1 - 1e-9)) + (1 - y_hold) * np.log(np.clip(1 - p_a, 1e-9, 1 - 1e-9)))
    ll_d_row = -(y_hold * np.log(np.clip(p_d, 1e-9, 1 - 1e-9)) + (1 - y_hold) * np.log(np.clip(1 - p_d, 1e-9, 1 - 1e-9)))
    diff_row = ll_d_row - ll_a_row   # D - A, negative = D wins

    log(f"\n=== paired comparison, n={len(y_hold):,} ===")
    log(f"  A (match only)    logloss={ll_a_row.mean():.5f}")
    log(f"  D (match+set+game) logloss={ll_d_row.mean():.5f}")
    log(f"  mean(D-A) = {diff_row.mean():+.5f}")

    rng = np.random.default_rng(0)
    n = len(diff_row)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        boot.append(diff_row[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "D wins" if hi < 0 else ("A wins" if lo > 0 else "unresolved (CI spans zero)")
    log(f"  5000x row-level bootstrap CI on mean(D-A): [{lo:+.5f}, {hi:+.5f}]  {verdict}")

    tour_arr = hold_j["tour"].values
    for tour in ("atp", "wta"):
        mask = tour_arr == tour
        d_t = diff_row[mask]
        boot_t = []
        for _ in range(5000):
            idx = rng.integers(0, mask.sum(), mask.sum())
            boot_t.append(d_t[idx].mean())
        lo_t, hi_t = np.percentile(boot_t, [2.5, 97.5])
        v_t = "D wins" if hi_t < 0 else ("A wins" if lo_t > 0 else "unresolved (CI spans zero)")
        log(f"  {tour}: mean(D-A)={d_t.mean():+.5f}  CI [{lo_t:+.5f}, {hi_t:+.5f}]  {v_t}")


if __name__ == "__main__":
    main()
