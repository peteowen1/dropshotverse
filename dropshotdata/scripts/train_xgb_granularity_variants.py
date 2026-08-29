"""Pete: "did we put both set and match (and game?) level features in the
xgb - or replace match with set? try both." Answer was: added set alongside
match (never tried game, never tried replacing match). This tests the
remaining variants against the SAME market-joined holdout, same TRAIN_YEARS/
hyperparameters as ticket 28 and the match+set run - granularity-mix is the
ONLY axis that varies across all of these (stats-discipline #5):

  A. match only          (ticket 28, already reported: ALL=0.59322)
  B. match + set          (already reported: ALL=0.59268)
  C. set only (replaces match-level Elo entirely)
  D. match + set + game

Builds the shared elo/hier/playstyle base features ONCE, then loops over
feature-set configs C and D (B is already done, A is the pre-existing
baseline - neither re-run here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import train_best_model as tbm

SCRATCH = tbm.SCRATCH
MATCH_ELO_COLS = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner"]
SET_ELO_COLS = ["set_elo_diff", "set_elo_logit_pwinner"]
GAME_ELO_COLS = ["game_elo_diff", "game_elo_logit_pwinner"]

VARIANTS = {
    "C_set_only": [c for c in tbm.XGB_NUMERIC if c not in MATCH_ELO_COLS] + SET_ELO_COLS,
    "D_match_set_game": tbm.XGB_NUMERIC + SET_ELO_COLS + GAME_ELO_COLS,
}


def log(m: str) -> None:
    print(m, flush=True)


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def merge_granularity_elo(feat: pd.DataFrame, path: str, prefix: str) -> pd.DataFrame:
    se = pd.read_parquet(path)[
        ["date", "tour", "winner_uid", "loser_uid", f"{prefix}_elo_winner", f"{prefix}_elo_loser",
        f"p_winner_{prefix}_elo"]
    ].drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
    j = feat.merge(se, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    miss = j[[f"{prefix}_elo_winner", f"{prefix}_elo_loser", f"p_winner_{prefix}_elo"]].isna().any(axis=1)
    log(f"  {len(j):,} -> {int((~miss).sum()):,} after {prefix}-Elo merge ({miss.mean():.4%} missing, dropped)")
    j = j[~miss].reset_index(drop=True)

    y = j["y"].values.astype(int)
    ew, el = j[f"{prefix}_elo_winner"].values.astype(float), j[f"{prefix}_elo_loser"].values.astype(float)
    j[f"{prefix}_elo_diff"] = np.where(y == 1, ew - el, el - ew)
    p_se = j[f"p_winner_{prefix}_elo"].values.astype(float)
    p_sideA = np.clip(np.where(y == 1, p_se, 1 - p_se), 1e-6, 1 - 1e-6)
    j[f"{prefix}_elo_logit_pwinner"] = np.log(p_sideA / (1 - p_sideA))
    return j.drop(columns=[f"{prefix}_elo_winner", f"{prefix}_elo_loser", f"p_winner_{prefix}_elo"])


def prep_X_variant(df: pd.DataFrame, cols: list[str], cat_levels: dict) -> pd.DataFrame:
    X = df[cols].copy()
    for c in tbm.XGB_CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def main() -> None:
    log("building shared base features (elo/hier/playstyle)...")
    train_feat = tbm.build_elo_features(tbm.TRAIN_YEARS)
    train_hier_path = tbm.regen_hier_state(tbm.TRAIN_YEARS, "train_canonical")
    train = tbm.merge_hier(train_feat, train_hier_path)
    playstyle = tbm.build_playstyle_features()
    train = tbm.merge_playstyle(train, playstyle)
    log("merging set-Elo + game-Elo into train...")
    train = merge_granularity_elo(train, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    train = merge_granularity_elo(train, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")

    holdout = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    log("merging set-Elo + game-Elo into holdout...")
    holdout = merge_granularity_elo(holdout, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    holdout = merge_granularity_elo(holdout, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")

    all_needed_cols = set(tbm.XGB_NUMERIC) | set(SET_ELO_COLS) | set(GAME_ELO_COLS)
    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(holdout[c].dropna().unique()))
                  for c in tbm.XGB_CATEGORICAL}

    diag2 = pd.read_parquet(SCRATCH / "diag2_enriched.parquet")
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in"] = True
    hold_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
    hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    y_hold = hold_j["y"].values.astype(int)

    elo = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])
    cmp_df = hold_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
        elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left").merge(
        diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
            subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
        how="left")
    p_mkt = np.where(y_hold == 1, cmp_df["p_mkt"].values, 1 - cmp_df["p_mkt"].values)
    tour_arr, surf_arr = hold_j["tour"].values, hold_j["surface"].values

    for name, cols in VARIANTS.items():
        log(f"\n=== variant {name}: {cols} ===")
        X_tr = prep_X_variant(train, cols, cat_levels)
        y_tr = train["y"].values.astype(int)
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        X_hold = prep_X_variant(hold_j, cols, cat_levels)
        dhold = xgb.DMatrix(X_hold, enable_categorical=True)

        preds = []
        for seed in range(tbm.N_BAGS):
            m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
            preds.append(m.predict(dhold))
        p_bagged = np.mean(preds, axis=0)

        for label, mask in (("ALL", np.ones(len(hold_j), dtype=bool)), ("ATP", tour_arr == "atp"),
                            ("WTA", tour_arr == "wta")):
            log(f"  {label:4s} n={mask.sum():5,d}  model={ll(y_hold[mask], p_bagged[mask]):.5f}  "
                f"market={ll(y_hold[mask], p_mkt[mask]):.5f}")

        pd.DataFrame({"date": hold_j["date"], "tour": hold_j["tour"], "winner_uid": hold_j["winner_uid"],
                     "loser_uid": hold_j["loser_uid"], "y": y_hold, "p": p_bagged}).to_parquet(
            SCRATCH / f"xgb_variant_{name}_predictions.parquet")

    log("\n=== reference (already reported, not re-run here) ===")
    log("A_match_only      ALL=0.59322  ATP=0.59454  WTA=0.59186  (market ALL=0.59352 ATP=0.59372 WTA=0.59330)")
    log("B_match_plus_set  ALL=0.59268  ATP=0.59428  WTA=0.59103")


if __name__ == "__main__":
    main()
