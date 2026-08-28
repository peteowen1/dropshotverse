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


# --------------------------------------------------- analytic delta-method correction
# Issue 20: an ALTERNATIVE to mc_average() above, not a replacement. mc_average()
# is untouched by this section (not called from it, not modified by it) and
# remains the in-use path everywhere else.
#
# Round 9's diagnosis (issue 20, round 3 third-continuation, reconfirmed every
# round since) is that the whole holdout gap-to-Elo is introduced downstream of
# an already-correctly-extremizing point estimate, by MC-averaging's Jensen's-
# gap convexity cost: p_winner_hier_mc = E[p_match(sigmoid(d1), sigmoid(d2))]
# over d1~N(d1_mean,d1_var), d2~N(d2_mean,d2_var), and p_match is convex/concave
# enough near 0/1 that the MC average sits systematically below the point
# estimate. mc_average() ESTIMATES that expectation by sampling; this section
# APPROXIMATES it analytically via a second-order (delta-method) Taylor
# expansion around the point estimate instead - same target quantity, a
# different (cheaper, exact-in-expectation-up-to-O(v^2)) way of computing it:
#
#   E[f(d1,d2)] ~= f(m1,m2) + 0.5*f_11(m1,m2)*v1 + 0.5*f_22(m1,m2)*v2
#
# where f(d1,d2) = p_match(sigmoid(d1), sigmoid(d2), best_of), (m1,m2) =
# (d1_mean, d2_mean), (v1,v2) = (d1_var, d2_var). d1 and d2 are independent in
# this model (separate players' composite-strength posteriors), so the cross
# term f_12*cov(d1,d2) is exactly 0 and is not computed. f_11/f_22 (the second
# partials of p_match's Bo3/Bo5-over-sets-over-games DP recursion) have no
# closed form, so they're estimated via a central finite difference:
#   f_11 ~= [f(m1+h,m2) - 2*f(m1,m2) + f(m1-h,m2)] / h**2
# This costs 5 p_match calls per row (1 center + 2 per side), vs
# n_samples (60-2000) calls for mc_average() - the speed win this section
# reports is a direct consequence of that, not a separate optimization.
#
# h is a NUMERICAL-APPROXIMATION-QUALITY step size, not a modeling
# hyperparameter - per the task charter, it must be picked for finite-
# difference stability (roundoff vs truncation error), not grid-searched
# against the log-loss metric. See tune_scratch/analytic_h_stability.py for
# the stability check that justifies the h=0.01 default.

H_DEFAULT = 0.01


def _analytic_chunk(args):
    rows, h = args
    out = np.empty(len(rows))
    for i, (d1_mean, d1_var, d2_mean, d2_var, best_of) in enumerate(rows):
        bo = int(best_of)
        f0 = p_match(sigmoid(d1_mean), sigmoid(d2_mean), bo)
        f1p = p_match(sigmoid(d1_mean + h), sigmoid(d2_mean), bo)
        f1m = p_match(sigmoid(d1_mean - h), sigmoid(d2_mean), bo)
        f2p = p_match(sigmoid(d1_mean), sigmoid(d2_mean + h), bo)
        f2m = p_match(sigmoid(d1_mean), sigmoid(d2_mean - h), bo)
        f11 = (f1p - 2.0 * f0 + f1m) / (h * h)
        f22 = (f2p - 2.0 * f0 + f2m) / (h * h)
        v1 = max(float(d1_var), 0.0)
        v2 = max(float(d2_var), 0.0)
        corrected = f0 + 0.5 * f11 * v1 + 0.5 * f22 * v2
        # Safety clip only - a second-order Taylor expansion is a LOCAL
        # approximation and can overshoot [0,1] under large variance /
        # high curvature (see validation report - this is documented, not
        # silently patched over). Never let it produce an invalid probability.
        out[i] = min(1.0, max(0.0, corrected))
    return out


def analytic_correction(df: pd.DataFrame, h: float = H_DEFAULT,
                        n_workers: int | None = None,
                        d1_mean_col: str = "d1_mean", d1_var_col: str = "d1_var",
                        d2_mean_col: str = "d2_mean", d2_var_col: str = "d2_var"
                        ) -> np.ndarray:
    """Delta-method (second-order Taylor) alternative to mc_average(). Same
    contract: takes a DataFrame with d1_mean/d1_var/d2_mean/d2_var/best_of
    columns (names overridable, matching mc_average()'s signature), returns
    an np.ndarray of corrected win probabilities aligned to df's row order.
    Deterministic (no RNG, no seed) - unlike mc_average(), calling this twice
    on the same input gives bit-identical output.

    Embarrassingly parallel across rows exactly like mc_average(), reusing
    the same chunk/Pool pattern. Do not run this concurrently with a
    mc_average() Pool in the same process tree - see mc_average()'s
    docstring / RESUME-beat-elo.md's memory-safety note."""
    rows = list(zip(df[d1_mean_col].values, df[d1_var_col].values,
                    df[d2_mean_col].values, df[d2_var_col].values,
                    df["best_of"].values))
    n = len(rows)
    if n == 0:
        return np.empty(0)
    n_workers = n_workers or min(12, mp.cpu_count() - 2, n)
    n_workers = max(1, n_workers)
    if n_workers == 1:
        return _analytic_chunk((rows, h))

    chunk_size = -(-n // n_workers)   # ceil div
    chunks = [(rows[i:i + chunk_size], h) for i in range(0, n, chunk_size)]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_analytic_chunk, chunks)
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
