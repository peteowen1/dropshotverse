"""Ticket 10 stage 2, final correction: replace the point-estimate plug-in
match probability with a Monte Carlo average over the filter's own posterior
uncertainty (d1_var, d2_var already saved by hierarchical_model.py).

Diagnosed via an 800-row sample (jensens_gap_test.py): plugging point estimates
of p1,p2 into the convex match recursion systematically OVERSTATES confidence
(Jensen's inequality - the recursion amplifies in the direction of the leading
player's advantage, so E[p_match(p1,p2)] < p_match(E[p1],E[p2]) whenever there
is genuine parameter uncertainty). Sample result: log-loss 0.868 -> 0.649.
This is not a bug fix; it is using information (posterior variance) the model
already computes but the naive scoring path was discarding.
"""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import time

import numpy as np
import pandas as pd
from hierarchical_model import p_match, sigmoid

N_SAMPLES = 60
rng = np.random.default_rng(0)

p = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\hier_eval_predictions.parquet")
t0 = time.time()
out = np.empty(len(p))
for i, row in enumerate(p.itertuples()):
    d1 = rng.normal(row.d1_mean, np.sqrt(max(row.d1_var, 1e-6)), N_SAMPLES)
    d2 = rng.normal(row.d2_mean, np.sqrt(max(row.d2_var, 1e-6)), N_SAMPLES)
    ps = [p_match(sigmoid(a), sigmoid(b), int(row.best_of)) for a, b in zip(d1, d2)]
    out[i] = np.mean(ps)
    if (i + 1) % 2000 == 0:
        print(f"  ...{i+1}/{len(p)} ({time.time()-t0:.0f}s)", flush=True)

p = p.assign(p_winner_hier_mc=out)
p.to_parquet(r"C:\dev\dropshotverse\dropshotdata\hier_eval_predictions.parquet")
print(f"done in {time.time()-t0:.0f}s. mean plug-in={p.p_winner_hier.mean():.4f} "
      f"mean posterior-avg={p.p_winner_hier_mc.mean():.4f}")
