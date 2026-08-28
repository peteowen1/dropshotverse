"""Speed fix: MC posterior-averaging was the dominant cost everywhere (a
single holdout check: ~340-470s; a tuning-window scan candidate: ~137s of
un-logged extra time on top of the printed fit_s), because p_match/p_set/
p_tiebreak are slow (540us/call, dict-based DP) and every row's MC samples
were drawn from one shared sequential RNG stream in a plain Python for-loop.

Each row's MC average is independent of every other row's, so this is
embarrassingly parallel - split rows across worker processes (24 cores
available, essentially unused until now). RNG: each row gets its own
np.random.default_rng(seed) keyed by row index (via SeedSequence spawn) so
results are reproducible regardless of how many workers run, but the actual
draws differ from the old single-shared-stream version - not bit-identical
to earlier session runs, just an equally valid independent MC estimate.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from hierarchical_model import p_match, sigmoid

N_SAMPLES_DEFAULT = 60


def _mc_chunk(args):
    (rows, n_samples, seeds, var_scale,
     var_scale_threshold, var_scale_high_scale) = args
    out = np.empty(len(rows))
    for i, (d1_mean, d1_var, d2_mean, d2_var, best_of) in enumerate(rows):
        rng = np.random.default_rng(seeds[i])
        row_scale = var_scale
        if var_scale_threshold is not None:
            # Cheap, no-MC point estimate (one p_match call, not n_samples of
            # them) - used only to decide which side of the threshold this
            # row falls on. Never touches d1_mean/d2_mean themselves.
            pt = p_match(sigmoid(d1_mean), sigmoid(d2_mean), int(best_of))
            if max(pt, 1.0 - pt) > var_scale_threshold:
                row_scale = row_scale * var_scale_high_scale
        d1 = rng.normal(d1_mean, np.sqrt(max(d1_var * row_scale, 1e-6)), n_samples)
        d2 = rng.normal(d2_mean, np.sqrt(max(d2_var * row_scale, 1e-6)), n_samples)
        ps = [p_match(sigmoid(a), sigmoid(b), int(best_of)) for a, b in zip(d1, d2)]
        out[i] = np.mean(ps)
    return out


def mc_average(df: pd.DataFrame, n_samples: int = N_SAMPLES_DEFAULT,
              n_workers: int | None = None, seed_base: int = 0,
              d1_mean_col: str = "d1_mean", d1_var_col: str = "d1_var",
              d2_mean_col: str = "d2_mean", d2_var_col: str = "d2_var",
              var_scale: float = 1.0,
              var_scale_threshold: float | None = None,
              var_scale_high_scale: float = 1.0) -> np.ndarray:
    """Parallel drop-in replacement for the serial MC-averaging loop used
    throughout this session's verification harness. Returns an array aligned
    to df's row order.

    Issue 20, round 4 (diag3_atp_extremization.py / diag3_mc_var_scale_poc.py):
    MC-averaging's Jensen's-gap cost grows with p_match's local curvature,
    which is largest exactly where the point estimate is most extreme -
    producing an outsized confidence haircut on the matches Elo/market are
    most sure about, tour-symmetrically (confirmed equally present in ATP and
    WTA). `var_scale` (default 1.0, exact no-op - identical output to the
    pre-existing behaviour) multiplies d1_var/d2_var before sampling, as a
    curvature-weighted correction that should be small on competitive
    matches (curvature near 0.5 is near-linear) and largest on already-
    extreme ones. A FLAT var_scale<1.0 was tuned and revealed on holdout in
    round 4 (attempt 19) and REJECTED - a genuine, non-boundary tuning-window
    win on the correct selection metric more than doubled the holdout gap to
    Elo, largely by eroding the 50-70% midband where MC-averaging was
    already net-positive.

    Issue 20, round 5/6 (diag4_conditional_var_scale_poc.py): a CONDITIONAL
    shape, restricted to rows the point estimate itself already calls
    lopsided, should in principle be curvature-selective in a way the flat
    scale wasn't. `var_scale_threshold` (default None = exact no-op,
    identical output to the pre-existing behaviour) is a threshold on the
    row's own no-MC point-estimate favourite probability
    (max(p_winner_hier, 1-p_winner_hier), computed cheaply here via one
    p_match(sigmoid(d1_mean), sigmoid(d2_mean), best_of) call per row - NOT
    via n_samples MC draws). Rows at or below the threshold get scale 1.0
    (untouched, exact no-op) regardless of `var_scale_high_scale`; rows
    above the threshold get `var_scale * var_scale_high_scale` applied to
    both d1_var and d2_var before sampling. `var_scale_high_scale` (default
    1.0) is the extra shrink factor for those high-confidence rows only -
    named for what it does (typically <1.0, an extra shrink) not for which
    side of the threshold it applies to (it only ever applies ABOVE the
    threshold). See tune_scratch/tune_mc_var_scale.py (flat, rejected) and
    tune_scratch/tune_cond_var_scale_2223.py (conditional, this round) for
    the 2022-23 charter scans."""
    rows = list(zip(df[d1_mean_col].values, df[d1_var_col].values,
                    df[d2_mean_col].values, df[d2_var_col].values,
                    df["best_of"].values))
    n = len(rows)
    if n == 0:
        return np.empty(0)
    seeds = np.random.SeedSequence(seed_base).spawn(n)
    n_workers = n_workers or min(12, mp.cpu_count() - 2, n)
    n_workers = max(1, n_workers)
    args_tail = (var_scale, var_scale_threshold, var_scale_high_scale)
    if n_workers == 1:
        return _mc_chunk((rows, n_samples, seeds, *args_tail))

    chunk_size = -(-n // n_workers)   # ceil div
    chunks = [(rows[i:i + chunk_size], n_samples, seeds[i:i + chunk_size], *args_tail)
             for i in range(0, n, chunk_size)]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_mc_chunk, chunks)
    return np.concatenate(results)


if __name__ == "__main__":
    import sys
    import time
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\dev\dropshotverse\dropshotdata\hier_v24_eval_predictions.parquet"
    out_col = sys.argv[2] if len(sys.argv) > 2 else "p_winner_hier_mc"
    p = pd.read_parquet(path)
    t0 = time.time()
    mc = mc_average(p)
    print(f"mc_average: {len(p):,} rows in {time.time()-t0:.1f}s "
         f"({mp.cpu_count()} cores available)")
    p = p.assign(**{out_col: mc})
    p.to_parquet(path)
    print(f"wrote {path} (column '{out_col}')")
