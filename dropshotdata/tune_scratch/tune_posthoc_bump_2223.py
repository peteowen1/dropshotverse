"""Part 1 continued: charter-compliant tuning of the EVAL-TIME-ONLY layoff
variance bump (diag2_posthoc_bump.py's mechanism, not the fit-time
SHORT_LAYOFF_BUMP tested in tune_short_layoff_bump.py - that one already
failed monotonically on tuning, even properly scoped to 1-12wk).

This bump does NOT touch hierarchical_model.py or refit anything - it adds a
candidate value directly to d1_var/d2_var at MC-averaging time, for matches
where either the winner or loser had a 1-12wk gap since their last match,
using the exact same mechanism diag2_posthoc_bump.py explored. Fits v25
(current default config, per RESUME-beat-elo.md) on 2022-23 ONLY (the tuning
window), joins the layoff/round/level features exactly as diag2_build.py
does, and scans the bump magnitude - properly through the corruption-neutral
(this doesn't touch fitting, so the gate is a no-op) tune(2022-23) stage of
the charter.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from market_compare import key_from_full, key_from_short, devig_power, load_odds
from mc_parallel import mc_average

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")
LAYOFF = SCRATCH / "diag_layoff.parquet"
CACHE = SCRATCH / "prepared_matches_cache.pkl"

TUNE_YEARS = (2022, 2023)
LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0
ELITE_LEVELS = frozenset({"G", "M", "F", "PM", "O"})

# v25 config, per RESUME-beat-elo.md
V25_ENV = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
          "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
          "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
          "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}


def log(m): print(m, flush=True)


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def fit_v25_tuning_window():
    preds_path = SCRATCH / "phb2223_preds.parquet"
    state_path = SCRATCH / "phb2223_state.pkl"
    env = {**__import__("os").environ, **V25_ENV,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in TUNE_YEARS),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fit failed: {r.stderr[-1000:]}")
    log(f"fit done in {time.time()-t0:.1f}s")
    return preds_path


def build_enriched(preds_path):
    hier = pd.read_parquet(preds_path)
    bad = hier["date"].dt.year >= 2024
    assert not bad.any(), f"LEAKAGE: touched {bad.sum()} rows >= 2024"
    log(f"tuning-window fit rows: {len(hier):,}")

    lay = pd.read_parquet(LAYOFF)
    hier = hier.merge(lay, on=["date", "winner_uid", "loser_uid"], how="left")

    import pickle
    # nosec: this session's own prepared-archive cache (hierarchical_model.py's
    # load_prepared()), same trust basis as diag2_build.py's identical load.
    with open(CACHE, "rb") as f:
        arc, _ = pickle.load(f)
    arc_small = arc[["date", "winner_uid", "loser_uid", "round", "tourney_level"]].drop_duplicates(
        subset=["date", "winner_uid", "loser_uid"])
    hier = hier.merge(arc_small, on=["date", "winner_uid", "loser_uid"], how="left")

    hier["is_elite"] = hier["tourney_level"].isin(ELITE_LEVELS)
    hier["winner_had_short_layoff"] = hier["weeks_since_last_w"].between(1, 12)
    hier["loser_had_short_layoff"] = hier["weeks_since_last_l"].between(1, 12)
    hier["either_short_layoff"] = hier["winner_had_short_layoff"] | hier["loser_had_short_layoff"]

    hier["k_w"] = hier["winner_name"].map(key_from_full)
    hier["k_l"] = hier["loser_name"].map(key_from_full)
    hier["pair"] = hier.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=TUNE_YEARS).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    j = hier.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
                   on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    j = j.copy()
    j["p_mkt"] = devig_power(j["PSW"].values.astype(float), j["PSL"].values.astype(float))
    j = j.dropna(subset=["p_mkt"])
    j = j.drop_duplicates(subset=["date", "winner_uid", "loser_uid"], keep="first")
    log(f"odds-joined tuning rows: {len(j):,}")
    return j


def score(df, bump, scope_mask, n_mc=60):
    d = df.copy()
    add = np.where(scope_mask, bump, 0.0)
    d["d1_var_mod"] = d["d1_var"] + add
    d["d2_var_mod"] = d["d2_var"] + add
    p = mc_average(d, n_samples=n_mc, d1_var_col="d1_var_mod", d2_var_col="d2_var_mod")
    pr = np.clip(p, 1e-9, 1 - 1e-9)
    ll_arr = -np.log(pr)
    plain = float(ll_arr.mean())
    mkt_fav = np.where(d["p_mkt"].values >= .5, d["p_mkt"].values, 1 - d["p_mkt"].values)
    weight = np.where(mkt_fav > LOPSIDED_THRESHOLD, LOPSIDED_WEIGHT, 1.0)
    weighted = float(np.sum(weight * ll_arr) / np.sum(weight))
    lop_mask = mkt_fav > LOPSIDED_THRESHOLD
    lopsided = float(ll_arr[lop_mask].mean()) if lop_mask.any() else float("nan")
    return plain, weighted, lopsided


if __name__ == "__main__":
    preds_path = fit_v25_tuning_window()
    df = build_enriched(preds_path)
    scope_any = df["either_short_layoff"].fillna(False).values
    log(f"rows with either_short_layoff=True: {scope_any.sum():,} / {len(df):,}")

    log("\n=== eval-time bump scan on 2022-23 (v25 base, either_short_layoff scope) ===")
    for bump in [0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.2]:
        plain, weighted, lop = score(df, bump, scope_any)
        log(f"  bump={bump:.3f}  plain={plain:.4f}  weighted={weighted:.4f}  lopsided={lop:.4f}")
