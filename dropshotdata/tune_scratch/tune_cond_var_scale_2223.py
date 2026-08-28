"""Issue 20, round 5/6: charter-compliant 2022-23 tuning pass for the
CONDITIONAL var_scale lever in mc_parallel.py (var_scale_threshold /
var_scale_high_scale), per RESUME-beat-elo.md's task brief.

Motivation: a FLAT var_scale<1.0 (attempt 19, round 4) was tuned and
transferred to the holdout - REJECTED, the gap to Elo more than doubled,
largely via 50-70% midband erosion (0.6896->0.7009). A quick single-threaded
POC (tune_scratch/diag4_conditional_var_scale_poc.py, unfiltered mixed-year
slice, NOT charter-compliant) suggested a CONDITIONAL shape - only shrink
variance for rows the point estimate itself already calls lopsided
(fav_extremity = max(p_winner_hier, 1-p_winner_hier) > threshold), leaving
everything else at scale 1.0 (exact no-op) - might avoid degrading the
midband the way the flat scale did, while still closing much of the
top-bucket undershoot. This script takes that POC through the actual
charter: tune ONLY on 2022-23, score on the SAME weighted/lopsided metric
v25 was selected on, and explicitly check the midband.

ONE fit on 2022-23 (v25 config) - var_scale only affects the downstream MC
step, not fit-time state, so every (threshold, scale) grid cell reuses the
SAME d1_mean/d1_var/d2_mean/d2_var/p_winner_hier columns from one
hierarchical_model.py run. Scores POOLED/ATP/WTA on weighted/plain/lopsided
plus the 50-70% market-confidence midband, same as tune_mc_var_scale.py.
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
    log("=== fitting v25 on 2022-23 tuning window (once - conditional var_scale only touches MC step) ===")
    tune = fit_and_join((2022, 2023), "condvs_tune")
    log(f"tuning window: n={len(tune)}")

    # sanity: threshold=None must reproduce the flat-baseline (var_scale=1.0) exactly
    base_check_a = mc_average(tune, n_samples=30, seed_base=0)
    base_check_b = mc_average(tune, n_samples=30, seed_base=0, var_scale_threshold=None)
    assert np.array_equal(base_check_a, base_check_b), "threshold=None is NOT reproducing baseline exactly!"
    log("  sanity check passed: var_scale_threshold=None reproduces baseline exactly")

    thresholds = [0.80, 0.85, 0.90, 0.95]
    scales = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    t0 = time.time()
    # baseline (no-op) row for reference
    baseline_mc = base_check_a
    tune["mc_baseline"] = baseline_mc
    pooled0 = score(tune, "mc_baseline")
    atp0 = score(tune[tune.tour == "atp"], "mc_baseline")
    wta0 = score(tune[tune.tour == "wta"], "mc_baseline")
    log(f"baseline (var_scale=1.0, no-op): pooled={pooled0}")
    log(f"  atp={atp0}")
    log(f"  wta={wta0}")
    results.append({"threshold": None, "scale": 1.0, "pooled": pooled0, "atp": atp0, "wta": wta0})

    for threshold in thresholds:
        for scale in scales:
            t1 = time.time()
            mc = mc_average(tune, n_samples=30, seed_base=0,
                            var_scale_threshold=threshold, var_scale_high_scale=scale)
            col = f"mc_t{threshold}_s{scale}"
            tune[col] = mc
            pooled = score(tune, col)
            atp = score(tune[tune.tour == "atp"], col)
            wta = score(tune[tune.tour == "wta"], col)
            log(f"threshold={threshold} scale={scale}: [{time.time()-t1:.0f}s]")
            log(f"    pooled={pooled}")
            log(f"    atp={atp}")
            log(f"    wta={wta}")
            results.append({"threshold": threshold, "scale": scale,
                            "pooled": pooled, "atp": atp, "wta": wta})
            tune.drop(columns=[col], inplace=True)
    log(f"\ndone scanning {len(thresholds)}x{len(scales)} grid in {time.time()-t0:.0f}s")

    import json
    out_path = SCRATCH / "cond_var_scale_scan.json"
    out_path.write_text(json.dumps(results, indent=1))
    log(f"wrote {out_path}")

    # print sorted by weighted metric (pooled)
    log("\n=== sorted by pooled weighted metric (best first) ===")
    ranked = sorted(results, key=lambda r: r["pooled"]["weighted"])
    for r in ranked[:10]:
        log(f"  threshold={r['threshold']} scale={r['scale']}: "
           f"pooled_weighted={r['pooled']['weighted']} pooled_midband={r['pooled']['midband_50_70']} "
           f"pooled_lopsided={r['pooled']['lopsided']}")
