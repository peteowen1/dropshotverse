"""Issue 20: numerical-stability check for the finite-difference step size h
used by mc_parallel.analytic_correction(). This picks h for finite-difference
stability ONLY (roundoff vs truncation error) - never against the log-loss
metric, per the task charter (that would be tuning a numerical-approximation
knob against the target, stats-discipline malpractice).

Checks, on a representative sample of (d1_mean, d2_mean, best_of) points
spanning the real range seen in hier_roundorder_tune_predictions.parquet:
  1. p_match is smooth (no discontinuity) under small perturbations of its
     serve-probability inputs - a fine grid scan of the raw function itself,
     not just the finite difference.
  2. The central-difference second derivative f_11 = [f(m+h)-2f(m)+f(m-h)]/h^2
     converges as h shrinks from 0.1 down to 0.001, and does NOT blow up from
     floating-point roundoff at the small end (which would show up as f_11
     going noisy/large rather than continuing to converge).
  3. Two candidate step sizes (0.005 and 0.01) agree with each other to a
     tight tolerance, on every sampled point - this is the actual "verify
     sensitivity" check the task brief asked for before trusting h=0.01.
"""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from hierarchical_model import p_match, sigmoid

rng = np.random.default_rng(42)


def f11_fd(d1, d2, bo, h):
    f0 = p_match(sigmoid(d1), sigmoid(d2), bo)
    fp = p_match(sigmoid(d1 + h), sigmoid(d2), bo)
    fm = p_match(sigmoid(d1 - h), sigmoid(d2), bo)
    return (fp - 2 * f0 + fm) / (h * h)


def main():
    preds = pd.read_parquet(
        r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_tune_predictions.parquet")
    # Stratified sample: cover the real d1_mean/d2_mean/best_of range, not a
    # synthetic grid, so this reflects rows the model actually produces.
    sample = preds.sample(n=60, random_state=7)
    # Add a few deliberately extreme points (near-certain matchups, the
    # highest-curvature region per issue 20's own diagnosis) not guaranteed
    # to be well-represented in a random sample.
    extremes = pd.DataFrame({
        "d1_mean": [2.5, -2.5, 0.0, 3.0, -3.0, 1.8],
        "d2_mean": [-1.5, 1.5, 0.0, -2.8, 2.8, -1.8],
        "best_of": [3, 3, 3, 5, 5, 3],
    })
    pts = pd.concat([sample[["d1_mean", "d2_mean", "best_of"]], extremes],
                    ignore_index=True)

    print(f"=== Part 1: raw p_match smoothness (fine grid scan) ===")
    d1_0, d2_0 = 0.6, -0.3
    xs = np.linspace(d1_0 - 0.05, d1_0 + 0.05, 401)
    ys = np.array([p_match(sigmoid(x), sigmoid(d2_0), 3) for x in xs])
    dydx = np.diff(ys) / np.diff(xs)
    d2ydx2 = np.diff(dydx) / np.diff(xs[:-1])
    print(f"  p_match range over d1 in [{xs[0]:.3f},{xs[-1]:.3f}]: "
         f"[{ys.min():.6f},{ys.max():.6f}]")
    print(f"  first-derivative range: [{dydx.min():.6f},{dydx.max():.6f}] "
         f"(std of successive diffs: {np.std(np.diff(dydx)):.2e})")
    print(f"  second-derivative (fine-grid) range: "
         f"[{d2ydx2.min():.4f},{d2ydx2.max():.4f}], std of successive "
         f"diffs: {np.std(np.diff(d2ydx2)):.2e}  <- should be small/smooth, "
         f"no jumps, if p_match is well-behaved for FD")

    print(f"\n=== Part 2: h-convergence of f_11 at 5 representative points ===")
    hs = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    check_pts = [(0.6, -0.3, 3), (2.5, -1.5, 3), (-2.5, 1.5, 3),
                (0.0, 0.0, 5), (3.0, -2.8, 5)]
    for (d1, d2, bo) in check_pts:
        vals = [f11_fd(d1, d2, bo, h) for h in hs]
        print(f"  (d1={d1:+.2f}, d2={d2:+.2f}, bo={bo}): " +
             "  ".join(f"h={h}:{v:+.5f}" for h, v in zip(hs, vals)))

    print(f"\n=== Part 3: h=0.005 vs h=0.01 agreement across {len(pts)} sampled rows ===")
    f11_01 = np.array([f11_fd(r.d1_mean, r.d2_mean, int(r.best_of), 0.01)
                       for r in pts.itertuples()])
    f11_005 = np.array([f11_fd(r.d1_mean, r.d2_mean, int(r.best_of), 0.005)
                        for r in pts.itertuples()])
    f11_02 = np.array([f11_fd(r.d1_mean, r.d2_mean, int(r.best_of), 0.02)
                       for r in pts.itertuples()])
    diff_01_005 = np.abs(f11_01 - f11_005)
    diff_01_02 = np.abs(f11_01 - f11_02)
    print(f"  |f11(h=0.01) - f11(h=0.005)|: mean={diff_01_005.mean():.2e}  "
         f"max={diff_01_005.max():.2e}")
    print(f"  |f11(h=0.01) - f11(h=0.02)| : mean={diff_01_02.mean():.2e}  "
         f"max={diff_01_02.max():.2e}")
    print(f"  f11 magnitude range across sample: "
         f"[{f11_01.min():.4f}, {f11_01.max():.4f}] "
         f"(mean |f11|={np.abs(f11_01).mean():.4f})")
    print(f"  worst-case relative disagreement (h=0.01 vs h=0.005), "
         f"excluding near-zero f11: "
         f"{np.max(diff_01_005[np.abs(f11_01) > 0.01] / np.abs(f11_01)[np.abs(f11_01) > 0.01]):.4f}")


if __name__ == "__main__":
    main()
