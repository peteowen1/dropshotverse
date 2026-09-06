"""Tests Pete's "set-level Elo as an XGB feature could help beat the market
more" - adds set_elo_diff/set_elo_logit_pwinner (from set_elo_raw_for_xgb.py,
today's standout granularity result, 0.60472 standalone vs match-level
Elo's own 0.61200) to ticket 28's exact feature set, retrains, and compares
against the SAME market-joined holdout ticket 28 already validated on
(same TRAIN_YEARS, same hyperparameters, same holdout rows - set-Elo is the
ONLY new axis, stats-discipline #5).

Reuses train_best_model.py's pipeline wholesale; only adds one merge step
and two new XGB_NUMERIC columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import train_best_model as tbm

SCRATCH = tbm.SCRATCH
XGB_NUMERIC_PLUS = tbm.XGB_NUMERIC + ["set_elo_diff", "set_elo_logit_pwinner"]


def log(m: str) -> None:
    print(m, flush=True)


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def merge_set_elo(feat: pd.DataFrame, set_elo_path: str) -> pd.DataFrame:
    se = pd.read_parquet(set_elo_path)[
        ["date", "tour", "winner_uid", "loser_uid", "set_elo_winner", "set_elo_loser", "p_winner_set_elo"]
    ].drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
    j = feat.merge(se, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    miss = j[["set_elo_winner", "set_elo_loser", "p_winner_set_elo"]].isna().any(axis=1)
    log(f"  {len(j):,} -> {int((~miss).sum()):,} after set-Elo merge ({miss.mean():.4%} missing, dropped)")
    j = j[~miss].reset_index(drop=True)

    y = j["y"].values.astype(int)
    ew, el = j["set_elo_winner"].values.astype(float), j["set_elo_loser"].values.astype(float)
    j["set_elo_diff"] = np.where(y == 1, ew - el, el - ew)
    p_se = j["p_winner_set_elo"].values.astype(float)
    p_se_sideA = np.clip(np.where(y == 1, p_se, 1 - p_se), 1e-6, 1 - 1e-6)
    j["set_elo_logit_pwinner"] = np.log(p_se_sideA / (1 - p_se_sideA))
    return j.drop(columns=["set_elo_winner", "set_elo_loser", "p_winner_set_elo"])


def prep_X_plus(df: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    X = df[XGB_NUMERIC_PLUS].copy()
    for c in tbm.XGB_CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def main() -> None:
    set_elo_path = str(SCRATCH / "set_elo_raw_ratings_full.parquet")

    train_feat = tbm.build_elo_features(tbm.TRAIN_YEARS)
    train_hier_path = tbm.regen_hier_state(tbm.TRAIN_YEARS, "train_canonical")
    train = tbm.merge_hier(train_feat, train_hier_path)
    playstyle = tbm.build_playstyle_features()
    train = tbm.merge_playstyle(train, playstyle)
    log("merging set-Elo into train...")
    train = merge_set_elo(train, set_elo_path)

    holdout = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    log("merging set-Elo into holdout...")
    holdout = merge_set_elo(holdout, set_elo_path)

    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(holdout[c].dropna().unique()))
                  for c in tbm.XGB_CATEGORICAL}
    X_tr = prep_X_plus(train, cat_levels)
    y_tr = train["y"].values.astype(int)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)

    diag2 = pd.read_parquet(SCRATCH / "diag2_enriched.parquet")
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in"] = True
    hold_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
    hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    X_hold = prep_X_plus(hold_j, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    y_hold = hold_j["y"].values.astype(int)

    log(f"\ntraining {tbm.N_BAGS}-seed bagged model WITH set-Elo features, {tbm.NUM_BOOST_ROUND} rounds each...")
    preds = []
    for seed in range(tbm.N_BAGS):
        m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
        preds.append(m.predict(dhold))
        log(f"  seed {seed} done")
    p_bagged = np.mean(preds, axis=0)

    elo = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])
    cmp_df = hold_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
        elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left").merge(
        diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
            subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
        how="left")
    p_mkt = np.where(y_hold == 1, cmp_df["p_mkt"].values, 1 - cmp_df["p_mkt"].values)
    p_elo = np.where(y_hold == 1, cmp_df["p_winner"].values, 1 - cmp_df["p_winner"].values)

    tour_arr = hold_j["tour"].values
    surf_arr = hold_j["surface"].values
    log("\n=== RESULTS (WITH set-Elo features) ===")
    for label, mask in (("ALL", np.ones(len(hold_j), dtype=bool)), ("ATP", tour_arr == "atp"),
                        ("WTA", tour_arr == "wta"),
                        ("ATP-Grass", (tour_arr == "atp") & (surf_arr == "Grass")),
                        ("ATP-Hard", (tour_arr == "atp") & (surf_arr == "Hard")),
                        ("ATP-Clay", (tour_arr == "atp") & (surf_arr == "Clay"))):
        log(f"{label:10s} n={mask.sum():5,d}  model={ll(y_hold[mask], p_bagged[mask]):.5f}  "
            f"elo={ll(y_hold[mask], p_elo[mask]):.5f}  market={ll(y_hold[mask], p_mkt[mask]):.5f}")

    log("\n=== reference (ticket 28, WITHOUT set-Elo, same holdout/hyperparameters) ===")
    log("ALL        model=0.59322  elo=?  market=0.59352")
    log("ATP        model=0.59454  market=0.59372")
    log("WTA        model=0.59186  market=0.59330")

    # feature importance for the two new features specifically
    m0 = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": 0}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
    gain = m0.get_score(importance_type="gain")
    log(f"\nnew feature gain (seed 0): set_elo_diff={gain.get('set_elo_diff', 0):.2f}  "
        f"set_elo_logit_pwinner={gain.get('set_elo_logit_pwinner', 0):.2f}")

    out_path = SCRATCH / "xgb_with_set_elo_predictions.parquet"
    pd.DataFrame({"date": hold_j["date"], "tour": hold_j["tour"], "winner_uid": hold_j["winner_uid"],
                 "loser_uid": hold_j["loser_uid"], "y": y_hold, "p_with_set_elo": p_bagged}).to_parquet(out_path)
    log(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
