"""Diagnostic (read-only proof-of-concept, NOT a tuned/charter-compliant
candidate - purely to check whether the MC-convexity mechanism found in
diag3_atp_extremization.py Part C actually responds to a variance-scale
lever the way the Jensen's-gap theory predicts, before writing up a
recommendation. Single-threaded, foreground, no multiprocessing.Pool (memory
safety note), no fit, no edit to hierarchical_model.py or mc_parallel.py -
just imports p_match/sigmoid (pure math, no shared state) and re-samples
using the ALREADY-COMPUTED d1_mean/d1_var/d2_mean/d2_var columns already in
diag2_enriched.parquet (untouched 2024-25 holdout, already revealed for
general eval - this does not choose or report a new "gap to Elo" candidate,
it only checks whether shrinking MC variance moves predictions in the
predicted direction and by roughly how much, restricted to the most lopsided
ATP+WTA rows to keep runtime small).
"""
import sys
import time

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from hierarchical_model import p_match, sigmoid

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"
N_SAMPLES = 200


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def mc_scaled(df, scale, n_samples=N_SAMPLES, seed_base=0):
    rows = list(zip(df["d1_mean"].values, df["d1_var"].values,
                     df["d2_mean"].values, df["d2_var"].values, df["best_of"].values))
    seeds = np.random.SeedSequence(seed_base).spawn(len(rows))
    out = np.empty(len(rows))
    for i, (d1_mean, d1_var, d2_mean, d2_var, best_of) in enumerate(rows):
        rng = np.random.default_rng(seeds[i])
        d1 = rng.normal(d1_mean, np.sqrt(max(d1_var * scale, 1e-6)), n_samples)
        d2 = rng.normal(d2_mean, np.sqrt(max(d2_var * scale, 1e-6)), n_samples)
        ps = [p_match(sigmoid(a), sigmoid(b), int(best_of)) for a, b in zip(d1, d2)]
        out[i] = np.mean(ps)
    return out


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)

    for tour in ["atp", "wta"]:
        sub = df[(df.tour == tour) & (df.p_elo > 0.9)].reset_index(drop=True)
        print(f"\n=== {tour.upper()}, p_elo>0.9, n={len(sub)} ===")
        print(f"  point-estimate mean={sub['p_winner_hier'].mean():.4f}  "
              f"elo mean={sub['p_elo'].mean():.4f}  "
              f"pre-computed mc600 mean={sub['p_winner_hier_mc600'].mean():.4f}")
        t0 = time.time()
        for scale in [1.0, 0.5, 0.25, 0.1, 0.0]:
            mc = mc_scaled(sub, scale)
            print(f"  scale={scale:4.2f}  mean_p={mc.mean():.4f}  "
                  f"logloss(vs own outcome)={ll(mc):.4f}  elo_logloss={ll(sub.p_elo):.4f}")
        print(f"  (n_samples={N_SAMPLES}, wall={time.time()-t0:.1f}s)")
