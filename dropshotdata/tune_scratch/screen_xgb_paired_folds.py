"""Paired per-fold test: is the K-ridge downstream gain real, or fold noise?

screen_xgb_downstream.py put the K change at -0.001102 downstream. Against the
fold-to-fold spread (0.0133) that looks like nothing - but the arms share folds,
rows and seed, so almost all of that spread is COMMON to both arms and cancels.
An unpaired comparison is the wrong test and would reject a real effect.

xgb.cv returns only across-fold means and stds, never per-fold values, so it
cannot do the paired comparison at all. This builds the same chronological folds
by hand (xgb.cv with shuffle=False uses contiguous blocks) and records each
fold's log-loss for each arm, so the statistic is the per-fold DIFFERENCE.

Reported: the five paired differences, their mean, and a paired t-style ratio.
With n=5 folds this cannot be conclusive on its own - five paired observations
is a small sample and the folds are chronological rather than exchangeable - so
a consistent sign across all five is the thing to look for, not a p-value.

Tune window only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact  # noqa: E402
from screen_xgb_downstream import (ARMS, CATS, ELO_COLS, FEAT, PARAMS,  # noqa: E402
                                   elo_columns)

NFOLD = 5
OUT = DD / "tune_scratch" / "screen_xgb_paired_folds.parquet"


def log(m: str = "") -> None:
    print(m, flush=True)


def design(f: pd.DataFrame):
    X = f.drop(columns=["date", "tour", "window", "winner_uid", "loser_uid", "y"],
               errors="ignore").copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, f["y"].values


def fold_scores(f: pd.DataFrame, n_round: int) -> list[float]:
    """Contiguous chronological folds, matching xgb.cv(shuffle=False)."""
    X, y = design(f)
    n = len(f)
    bounds = [(i * n // NFOLD, (i + 1) * n // NFOLD) for i in range(NFOLD)]
    out = []
    for lo, hi in bounds:
        te = np.zeros(n, dtype=bool)
        te[lo:hi] = True
        dtr = xgb.DMatrix(X[~te], label=y[~te], enable_categorical=True)
        dte = xgb.DMatrix(X[te], label=y[te], enable_categorical=True)
        bst = xgb.train(PARAMS, dtr, num_boost_round=n_round, verbose_eval=False)
        p = np.clip(bst.predict(dte), 1e-15, 1 - 1e-15)
        yy = y[te]
        out.append(float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()))
    return out


def main() -> None:
    feat = pd.read_parquet(FEAT)
    feat["tour"] = feat["tour"].astype(str)
    d = build_compact()
    log(f"coverage: {len(feat):,} rows, {NFOLD} chronological folds")

    # Boosting rounds pinned to the baseline's own CV optimum for BOTH arms, so
    # the comparison is not confounded by each arm selecting a different stopping
    # point (stats-discipline #5 rule 4 - a hyperparameter the fitter chooses
    # inside each arm is a changed axis too).
    N_ROUND = 217
    log(f"num_boost_round pinned at {N_ROUND} for every arm "
        f"(baseline's CV optimum), so stopping point is not a free axis\n")

    rows = {}
    for name in ("baseline (shipped Elo)", "round13 retirements @0.5",
                 "round14 K ridge 350/0.5/16", "both"):
        f = feat.copy()
        f[ELO_COLS] = elo_columns(d, ARMS[name], feat)[ELO_COLS].values
        s = fold_scores(f, N_ROUND)
        rows[name] = s
        log(f"  {name:<30} " + " ".join(f"{v:.5f}" for v in s)
            + f"   mean={np.mean(s):.6f}")

    base = np.array(rows["baseline (shipped Elo)"])
    log("\n--- paired per-fold differences vs baseline ---")
    recs = []
    for name in ("round13 retirements @0.5", "round14 K ridge 350/0.5/16", "both"):
        arm = np.array(rows[name])
        dlt = arm - base
        mean_d = float(dlt.mean())
        sd = float(dlt.std(ddof=1))
        se = sd / np.sqrt(NFOLD)
        t = mean_d / se if se > 0 else np.nan
        n_neg = int((dlt < 0).sum())
        log(f"  {name}")
        log(f"    per-fold delta: " + " ".join(f"{v:+.5f}" for v in dlt))
        log(f"    mean {mean_d:+.6f}  sd {sd:.6f}  se {se:.6f}  t {t:+.2f}  "
            f"folds improved {n_neg}/{NFOLD}")
        recs.append(dict(arm=name, mean_delta=mean_d, sd=sd, se=se, t=t,
                         folds_improved=n_neg))
    pd.DataFrame(recs).to_parquet(OUT, index=False)

    log("\n--- how to read this ---")
    log("  5 chronological folds is a small, non-exchangeable sample, so treat")
    log("  a consistent SIGN across all five as the evidence and the t as")
    log("  descriptive only. Anything improving 3/5 or fewer folds is noise")
    log("  regardless of its mean (stats-discipline #10).")
    log("PAIRED_COMPLETE")


if __name__ == "__main__":
    main()
