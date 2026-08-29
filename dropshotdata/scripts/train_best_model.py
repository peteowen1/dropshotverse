"""Canonical best-model pipeline (ticket 28, 2026-08-29 chase-the-market
effort). Reproduces the session's final model end to end from the raw
archive + the frozen, already-validated v25 hierarchical-model config - no
retuning of anything, every choice below is justified in
.scratch/dropshot-tennis-engine/issues/28-chase-the-market.md.

Result on the frozen 2024-25 market-odds-joined holdout (n=8,591), 5-seed
bagged: ALL log-loss 0.58976 vs market's 0.58903 (gap 0.00073). WTA beats
the market outright (0.58779 vs 0.58852); ATP remains the laggard (0.00214
gap). Gap closed from Elo's original 0.026, and from XGBoost v1's 0.0145.

Pipeline (each step's rationale is in issues/28-chase-the-market.md):
  1. Elo-only + engineered features (age, in-tournament fatigue, as-of h2h,
     inter-tournament layoff) for TRAIN_YEARS, from elo_raw_ratings_full
     .parquet / diag_layoff.parquet (both already full-archive) plus the raw
     yearly CSVs for age. (round 12 step 1, round 1's age/fatigue)
  2. Hierarchical-model posterior state (d1/d2 mean+var, p_winner_hier) for
     the SAME years, via hierarchical_model.py's DSV_EVAL_YEARS export
     mechanism with the frozen v25 hyperparameters - replays the full
     archive regardless of year count, so widening the window is nearly
     free (~90s). (round 12 step 2)
  3. Merge: elo_overall_diff, elo_surface_diff, elo_logit_pwinner,
     hier_dmean_diff, hier_dvar_diff, hier_dvar_sum, hier_logit_pwinner (the
     model's own combined pre-match probability - round 13's biggest single
     find, Δ-0.00941 CV, verified leak-free by source inspection and by
     letting CV converge naturally past a suspicious first look),
     weeks_since_last_diff, n_prior_diff, h2h_diff, age_diff, fatigue_diff,
     best_of, plus surface/round/tourney_level/tour as categoricals (round
     14's small tour-awareness win).
  4. Train with the hyperparameters found by a 40+24-point random search
     (round 1 + round 12's wide-dataset re-search, which confirmed no
     further retuning was needed): max_depth=3, eta=0.02, subsample=0.7,
     colsample_bytree=0.9, min_child_weight=5, gamma=1, reg_alpha=0.1,
     reg_lambda=1, num_boost_round=3161 (properly converged, not capped).
  5. Bag across N_BAGS seeds (round 14's last real lever - reduces variance,
     helps ATP specifically more than WTA).

TRAIN_YEARS is set to 1995-2023 (29yr) - round 12 found this the practical
plateau of the data-volume lever (diminishing returns: 2->9yr bought
-0.00264, 9->19yr -0.00061, 19->29yr only -0.00028; pre-1995 not attempted,
era-heterogeneity risk judged to outweigh the ~0.0001-0.0002 left on the
table).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
SCRATCH = DD / "tune_scratch"
TRAIN_YEARS = list(range(1995, 2024))
N_BAGS = 5

# Frozen v25 hierarchical-model hyperparameters (validated in prior sessions'
# tickets 17-24; NOT retuned here or anywhere in ticket 28).
V25_HIER_PARAMS = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
                    "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
                    "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
                    "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}

XGB_NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
               "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
               "h2h_diff", "best_of", "age_diff", "fatigue_diff", "hier_logit_pwinner"]
XGB_CATEGORICAL = ["surface", "round", "tourney_level", "tour"]
XGB_BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
                    "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
                    "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0,
                    "nthread": 8}
NUM_BOOST_ROUND = 3161

ROUND_RANK = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4, "ER": 5,
              "R128": 6, "R64": 7, "R32": 8, "R16": 9,
              "QF": 10, "RR": 11, "SF": 12, "F": 13, "BR": 13}


def log(m: str) -> None:
    print(m, flush=True)


# --------------------------------------------------------------- step 1
def load_raw_years(tour: str, years: list[int]) -> pd.DataFrame:
    frames = [pd.read_csv(DD / "raw/github" / tour / f"{tour}_matches_{y}.csv", low_memory=False)
              for y in years]
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    rnd = df["round"].astype(str).str.strip().replace({"f": "F"})
    df["round_rank"] = rnd.map(ROUND_RANK).fillna(99)
    return df


def compute_fatigue(raw: pd.DataFrame) -> pd.DataFrame:
    raw_sorted = raw.sort_values(["tourney_id", "round_rank"]).reset_index(drop=True)
    counts: dict[tuple, int] = {}
    fw = np.zeros(len(raw_sorted), dtype=np.int32)
    fl = np.zeros(len(raw_sorted), dtype=np.int32)
    for i, (tid, w, l) in enumerate(zip(raw_sorted["tourney_id"].values, raw_sorted["winner_uid"].values,
                                         raw_sorted["loser_uid"].values)):
        fw[i] = counts.get((tid, w), 0)
        fl[i] = counts.get((tid, l), 0)
        counts[(tid, w)] = fw[i] + 1
        counts[(tid, l)] = fl[i] + 1
    raw_sorted["prior_matches_event_w"] = fw
    raw_sorted["prior_matches_event_l"] = fl
    return raw_sorted.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])


def build_elo_features(years: list[int]) -> pd.DataFrame:
    log(f"step 1: Elo + engineered features for {years[0]}-{years[-1]}...")
    raw = pd.concat([load_raw_years("atp", years), load_raw_years("wta", years)], ignore_index=True)
    raw = compute_fatigue(raw)
    small = raw[["date", "tour", "winner_uid", "loser_uid", "winner_age", "loser_age",
                 "prior_matches_event_w", "prior_matches_event_l", "surface", "round",
                 "tourney_level", "best_of"]]

    elo_raw = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet")
    elo_raw = elo_raw[elo_raw["level_file"] == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name",
         "ro_winner", "ro_loser", "rs_winner", "rs_loser", "p_winner_recomputed"]]
    layoff = pd.read_parquet(SCRATCH / "diag_layoff.parquet")[
        ["date", "winner_uid", "loser_uid", "weeks_since_last_w", "weeks_since_last_l",
         "n_prior_w", "n_prior_l"]]

    full_elo_raw = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet")
    wins: dict[tuple, int] = {}
    h2h_w = np.zeros(len(full_elo_raw), dtype=np.int32)
    h2h_l = np.zeros(len(full_elo_raw), dtype=np.int32)
    for i, (w, l) in enumerate(zip(full_elo_raw["winner_uid"].values, full_elo_raw["loser_uid"].values)):
        h2h_w[i] = wins.get((w, l), 0)
        h2h_l[i] = wins.get((l, w), 0)
        wins[(w, l)] = wins.get((w, l), 0) + 1
    h2h = full_elo_raw[["date", "tour", "winner_uid", "loser_uid"]].copy()
    h2h["h2h_prior_wins_winner"] = h2h_w
    h2h["h2h_prior_wins_loser"] = h2h_l

    m = elo_raw.merge(small, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    m = m.merge(layoff, on=["date", "winner_uid", "loser_uid"], how="left")
    m = m.merge(h2h, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    m = m.drop_duplicates(subset=["date", "winner_uid", "loser_uid"], keep="first")

    req = ["weeks_since_last_w", "weeks_since_last_l", "n_prior_w", "n_prior_l",
           "h2h_prior_wins_winner", "h2h_prior_wins_loser", "winner_age", "loser_age"]
    m = m[~m[req].isna().any(axis=1)].reset_index(drop=True)

    import hashlib

    def side_a_is_winner(date, w_uid, l_uid):
        key = f"{date.date()}|{'|'.join(sorted([w_uid, l_uid]))}"
        return (hashlib.sha256(key.encode()).digest()[0] % 2) == 0

    a_is_winner = np.array([side_a_is_winner(d, w, l) for d, w, l in
                             zip(m["date"], m["winner_uid"], m["loser_uid"])])
    y = a_is_winner.astype(int)

    def pick(win_col, lose_col):
        wv, lv = m[win_col].values, m[lose_col].values
        return np.where(a_is_winner, wv, lv), np.where(a_is_winner, lv, wv)

    a_ro, b_ro = pick("ro_winner", "ro_loser")
    a_rs, b_rs = pick("rs_winner", "rs_loser")
    a_weeks, b_weeks = pick("weeks_since_last_w", "weeks_since_last_l")
    a_nprior, b_nprior = pick("n_prior_w", "n_prior_l")
    a_h2h, b_h2h = pick("h2h_prior_wins_winner", "h2h_prior_wins_loser")
    a_age, b_age = pick("winner_age", "loser_age")
    a_fat, b_fat = pick("prior_matches_event_w", "prior_matches_event_l")

    p_win_recomp = m["p_winner_recomputed"].values.astype(float)
    p_sideA = np.clip(np.where(a_is_winner, p_win_recomp, 1.0 - p_win_recomp), 1e-6, 1 - 1e-6)

    feat = pd.DataFrame({
        "date": m["date"], "tour": m["tour"], "winner_uid": m["winner_uid"], "loser_uid": m["loser_uid"], "y": y,
        "elo_overall_diff": a_ro - b_ro, "elo_surface_diff": a_rs - b_rs,
        "elo_logit_pwinner": np.log(p_sideA / (1.0 - p_sideA)),
        "weeks_since_last_diff": a_weeks - b_weeks, "n_prior_diff": a_nprior - b_nprior,
        "h2h_diff": a_h2h - b_h2h, "age_diff": a_age - b_age, "fatigue_diff": a_fat - b_fat,
        "surface": m["surface"].astype(str), "round": m["round"].astype(str),
        "best_of": m["best_of"].astype("Int64"), "tourney_level": m["tourney_level"].astype(str),
    })
    log(f"  {len(feat):,} rows")
    return feat


# --------------------------------------------------------------- step 2
def regen_hier_state(years: list[int], tag: str) -> Path:
    log(f"step 2: hierarchical-model posterior state for {years[0]}-{years[-1]} "
        f"(frozen v25 config, ~60-90s regardless of year count)...")
    years_str = ",".join(str(y) for y in years)
    preds_path = SCRATCH / f"v25_postfix_{tag}_preds.parquet"
    if preds_path.exists():
        log(f"  {preds_path.name} already exists, reusing")
        return preds_path
    env = {**os.environ, **V25_HIER_PARAMS, "DSV_EVAL_YEARS": years_str,
           "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(SCRATCH / f"v25_postfix_{tag}_state.pkl")}
    t0 = time.time()
    r = subprocess.run([sys.executable, str(DD / "scripts/hierarchical_model.py")],
                        cwd=str(DD / "scripts"), env=env, capture_output=True, text=True, timeout=1800)
    log(f"  exit={r.returncode}  elapsed={time.time() - t0:.0f}s")
    if r.returncode != 0:
        log(r.stdout[-2000:] + "\n" + r.stderr[-2000:])
        raise SystemExit("hierarchical_model.py failed")
    return preds_path


# --------------------------------------------------------------- step 3
def merge_hier(feat: pd.DataFrame, hier_preds_path: Path) -> pd.DataFrame:
    hier = pd.read_parquet(hier_preds_path)[
        ["date", "tour", "winner_uid", "loser_uid", "d1_mean", "d1_var", "d2_mean", "d2_var",
         "p_winner_hier"]].drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
    j = feat.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    miss = j[["d1_mean", "d1_var", "d2_mean", "d2_var", "p_winner_hier"]].isna().any(axis=1)
    log(f"  {len(j):,} -> {int((~miss).sum()):,} after hier merge ({miss.mean():.4%} missing, dropped)")
    j = j[~miss].reset_index(drop=True)

    y = j["y"].values.astype(int)
    d1m, d2m = j["d1_mean"].values.astype(float), j["d2_mean"].values.astype(float)
    d1v, d2v = j["d1_var"].values.astype(float), j["d2_var"].values.astype(float)
    j["hier_dmean_diff"] = np.where(y == 1, d1m - d2m, d2m - d1m)
    j["hier_dvar_diff"] = np.where(y == 1, d1v - d2v, d2v - d1v)
    j["hier_dvar_sum"] = d1v + d2v

    p_hier = j["p_winner_hier"].values.astype(float)
    p_hier_sideA = np.clip(np.where(y == 1, p_hier, 1 - p_hier), 1e-6, 1 - 1e-6)
    j["hier_logit_pwinner"] = np.log(p_hier_sideA / (1 - p_hier_sideA))
    return j.drop(columns=["d1_mean", "d1_var", "d2_mean", "d2_var", "p_winner_hier"])


# --------------------------------------------------------------- steps 4-5
def prep_X(df: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    X = df[XGB_NUMERIC].copy()
    for c in XGB_CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    train_feat = build_elo_features(TRAIN_YEARS)
    train_hier_path = regen_hier_state(TRAIN_YEARS, "train_canonical")
    train = merge_hier(train_feat, train_hier_path)

    # Holdout (2024-25) features already built and validated across rounds 1-14;
    # reuse the existing file directly rather than rebuilding.
    holdout = pd.read_parquet(SCRATCH / "xgb_features_v7_holdout.parquet")

    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(holdout[c].dropna().unique()))
                  for c in XGB_CATEGORICAL}
    X_tr = prep_X(train, cat_levels)
    y_tr = train["y"].values.astype(int)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)

    diag2 = pd.read_parquet(SCRATCH / "diag2_enriched.parquet")
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in"] = True
    hold_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    hold_j = hold_j[hold_j["_in"] == True].drop(columns=["_in"])  # noqa: E712
    hold_j = hold_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    X_hold = prep_X(hold_j, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    y_hold = hold_j["y"].values.astype(int)

    log(f"\nsteps 4-5: training {N_BAGS}-seed bagged model, {NUM_BOOST_ROUND} rounds each...")
    preds = []
    for seed in range(N_BAGS):
        m = xgb.train({**XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=NUM_BOOST_ROUND)
        preds.append(m.predict(dhold))
        log(f"  seed {seed} done")
    p_bagged = np.mean(preds, axis=0)

    elo = pd.read_parquet(DD / "elo_eval_predictions.parquet")
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
    log("\n=== RESULTS ===")
    for label, mask in (("ALL", np.ones(len(hold_j), dtype=bool)), ("ATP", tour_arr == "atp"),
                        ("WTA", tour_arr == "wta")):
        log(f"{label:4s} n={mask.sum():5,d}  model={ll(y_hold[mask], p_bagged[mask]):.5f}  "
            f"elo={ll(y_hold[mask], p_elo[mask]):.5f}  market={ll(y_hold[mask], p_mkt[mask]):.5f}")


if __name__ == "__main__":
    main()
