"""Finer scan of the conditional var_scale lever's scale axis between 0.7
(the round-1 grid's best cell, at the grid edge) and 1.0 (baseline, no-op)
to determine whether the weighted-metric optimum found in
tune_cond_var_scale_2223.py is a genuine interior optimum or another
monotonic-to-boundary artifact. Reuses the cached tuning window (no refit
needed - conditional var_scale only touches the MC step)."""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from mc_parallel import mc_average

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\cond_tune_window_2223.parquet"
LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0


def log(m): print(m, flush=True)


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
    tune = pd.read_parquet(SRC)
    log(f"tuning window n={len(tune)}")

    baseline_mc = mc_average(tune, n_samples=30, seed_base=0)
    tune["mc_baseline"] = baseline_mc
    pooled0 = score(tune, "mc_baseline")
    log(f"baseline (scale=1.0): {pooled0}")

    thresholds = [0.80, 0.85, 0.90]
    scales = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

    results = []
    t0 = time.time()
    for threshold in thresholds:
        for scale in scales:
            t1 = time.time()
            mc = mc_average(tune, n_samples=30, seed_base=0,
                            var_scale_threshold=threshold, var_scale_high_scale=scale)
            col = "mc_tmp"
            tune[col] = mc
            pooled = score(tune, col)
            log(f"threshold={threshold} scale={scale}: [{time.time()-t1:.0f}s] pooled={pooled}")
            results.append({"threshold": threshold, "scale": scale, "pooled": pooled})
            tune.drop(columns=[col], inplace=True)
    log(f"\ndone in {time.time()-t0:.0f}s")

    log("\n=== sorted by weighted (best first) ===")
    for r in sorted(results, key=lambda r: r["pooled"]["weighted"])[:10]:
        log(f"  threshold={r['threshold']} scale={r['scale']}: {r['pooled']}")
