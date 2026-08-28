"""Proof-of-concept only (like diag3_mc_var_scale_poc.py) - single-threaded,
no Pool, characterization not a tuned candidate. Tests whether a var_scale
applied ONLY to high-point-estimate rows (leaving the midband alone) avoids
the flat var_scale's holdout failure (ticket 20, Attempt 19: flat 0.7 scale
won on tuning-window weighted metric but doubled the holdout gap, partly via
midband erosion 0.6896->0.7009). Motivated by this session's round/experience
calibration cut: the undershoot is worst specifically where the underdog is a
near-unknown (high combined_var), which correlates with high point-estimate
extremity, not with being in the midband.
"""
import sys

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd

from hierarchical_model import p_match, sigmoid

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"
N_MC = 150


def mc_conditional(df, threshold, scale, n_samples=N_MC, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(len(df))
    d1m, d1v = df.d1_mean.values, df.d1_var.values
    d2m, d2v = df.d2_mean.values, df.d2_var.values
    bo = df.best_of.values
    pt = df.p_winner_hier.values
    for i in range(len(df)):
        fav_extremity = max(pt[i], 1 - pt[i])
        s = scale if fav_extremity > threshold else 1.0
        d1 = rng.normal(d1m[i], np.sqrt(max(d1v[i] * s, 1e-6)), n_samples)
        d2 = rng.normal(d2m[i], np.sqrt(max(d2v[i] * s, 1e-6)), n_samples)
        ps = [p_match(sigmoid(a), sigmoid(b), int(bo[i])) for a, b in zip(d1, d2)]
        out[i] = np.mean(ps)
    return out


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)

    top = df[df.p_elo > 0.9].reset_index(drop=True)
    mid = df[(df.p_elo > 0.5) & (df.p_elo < 0.7)].sample(n=600, random_state=0).reset_index(drop=True)

    print(f"top slice n={len(top)}, mid slice n={len(mid)}", flush=True)
    print(f"top baseline: mc600_ll={ll(top.p_winner_hier_mc600):.4f} elo_ll={ll(top.p_elo):.4f}", flush=True)
    print(f"mid baseline: mc600_ll={ll(mid.p_winner_hier_mc600):.4f} elo_ll={ll(mid.p_elo):.4f}", flush=True)

    for threshold, scale in [(0.85, 0.5), (0.90, 0.4)]:
        top_p = mc_conditional(top, threshold, scale)
        mid_p = mc_conditional(mid, threshold, scale)
        n_touched_top = int((top.p_winner_hier.apply(lambda p: max(p, 1 - p)) > threshold).sum())
        n_touched_mid = int((mid.p_winner_hier.apply(lambda p: max(p, 1 - p)) > threshold).sum())
        print(f"threshold={threshold} scale={scale}: "
              f"top_ll={ll(top_p):.4f} (n_touched={n_touched_top}/{len(top)})  "
              f"mid_ll={ll(mid_p):.4f} (n_touched={n_touched_mid}/{len(mid)})", flush=True)
    print("done.", flush=True)
