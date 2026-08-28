"""Issue 20 round 4, attempt 19: charter-compliant tuning pass for the
`var_scale` lever in mc_parallel.py (proposed by the parallel diag3 agent's
investigation, added to mc_parallel.py this round - see its docstring).

Mechanism (confirmed by diag3_atp_extremization.py, tour-symmetric): MC-
averaging's Jensen's-gap cost grows with p_match's local curvature, which is
largest exactly where the point estimate is already most extreme - an
outsized confidence haircut on the matches everyone else is most sure about.
`var_scale` < 1.0 shrinks d1_var/d2_var before MC sampling, a curvature-
weighted correction that should touch competitive matches only lightly.

ONE fit on 2022-23 (v25 config) - var_scale only affects the downstream MC
step, not fit-time state, so every scale value in the grid reuses the SAME
d1_mean/d1_var/d2_mean/d2_var columns from one hierarchical_model.py run.
Scores POOLED/ATP/WTA on weighted/plain/lopsided (the metric v25 was
selected on), AND the 50-70% market-confidence bucket specifically (the risk
diag3 flagged: don't erode where MC-averaging is currently net-positive).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from market_compare import key_from_full, key_from_short, devig_power, load_odds
from mc_parallel import mc_average

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")
LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0

V25 = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
      "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
      "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
      "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}


def log(m): print(m, flush=True)


def fit_and_join(years, tag):
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    state_path = SCRATCH / f"state_{tag}.pkl"
    env = {**V25, "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    full_env = {**os.environ, **env}
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fit failed: {r.stderr[-1000:]}")
    log(f"  fit {tag} done in {time.time()-t0:.0f}s")
    p = pd.read_parquet(preds_path)
    if years == (2022, 2023):
        bad = p["date"].dt.year >= 2024
        assert not bad.any(), f"LEAKAGE: {tag} touched {bad.sum()} rows >= 2024"
    p["k_w"] = p["winner_name"].map(key_from_full)
    p["k_l"] = p["loser_name"].map(key_from_full)
    p["pair"] = p.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=years).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    m = p.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
               on=["pair", "tour"], how="left", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
    m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    m = m[(m["k_w_o"] == m["k_w"]) & m[["PSW", "PSL"]].notna().all(axis=1)]
    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m = m.dropna(subset=["p_mkt"])
    m["mkt_fav"] = np.where(m.p_mkt >= .5, m.p_mkt, 1 - m.p_mkt)
    preds_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    return m


def ll(x):
    pr = np.clip(x, 1e-9, 1 - 1e-9)
    return float(-np.log(pr).mean())


def score(sub, mc_col):
    weight_v = np.where(sub["mkt_fav"].values > LOPSIDED_THRESHOLD, LOPSIDED_WEIGHT, 1.0)
    pr = np.clip(sub[mc_col].values, 1e-9, 1 - 1e-9)
    llv = -np.log(pr)
    weighted = float(np.sum(weight_v * llv) / np.sum(weight_v))
    plain = float(llv.mean())
    lop_mask = sub["mkt_fav"].values > LOPSIDED_THRESHOLD
    lopsided = float(llv[lop_mask].mean()) if lop_mask.any() else float("nan")
    mid_mask = (sub["mkt_fav"].values > 0.5) & (sub["mkt_fav"].values <= 0.7)
    midband = float(llv[mid_mask].mean()) if mid_mask.any() else float("nan")
    return {"n": len(sub), "weighted": round(weighted, 4), "plain": round(plain, 4),
           "lopsided": round(lopsided, 4), "midband_50_70": round(midband, 4)}


if __name__ == "__main__":
    log("=== fitting v25 on 2022-23 tuning window (once - var_scale only touches MC step) ===")
    tune = fit_and_join((2022, 2023), "vs_tune")

    GRID = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    results = []
    t0 = time.time()
    for scale in GRID:
        t1 = time.time()
        mc = mc_average(tune, n_samples=30, var_scale=scale)
        tune[f"mc_{scale}"] = mc
        pooled = score(tune, f"mc_{scale}")
        atp = score(tune[tune.tour == "atp"], f"mc_{scale}")
        wta = score(tune[tune.tour == "wta"], f"mc_{scale}")
        log(f"  var_scale={scale}: [{time.time()-t1:.0f}s]")
        log(f"    pooled={pooled}")
        log(f"    atp={atp}")
        log(f"    wta={wta}")
        results.append({"var_scale": scale, "pooled": pooled, "atp": atp, "wta": wta})
    log(f"\ndone scanning in {time.time()-t0:.0f}s")

    import json
    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\mc_var_scale_scan.json").write_text(
        json.dumps(results, indent=1))
    log("wrote tune_scratch/mc_var_scale_scan.json")
