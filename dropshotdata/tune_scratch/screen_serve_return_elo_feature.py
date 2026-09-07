"""Does serve/return Elo ADD anything to the XGBoost, net of what it already has?

This is a different question from "is the rating any good", and the answers can
legitimately disagree: a signal can be the best standalone serve rating we have
and still be redundant inside a model that already carries elo, hier and
serve/return form.

The cheap screen first, per the effect-sizing rule: fit `y ~ existing` and
`y ~ existing + candidate` in-sample and unpenalised, and read the candidate's
partial correlation with the target net of the existing predictors. The
reference point is a candidate family that WAS harness-tested and came back
null - max partial correlation 0.077. A candidate at or below that has no
business consuming a walk-forward run.

Orientation matters and is easy to get backwards. The feature matrix is
orientation-randomised: `y` is 1 when side A is the match winner, so a
difference feature must be built as A minus B, which is winner minus loser when
y == 1 and loser minus winner when y == 0. Getting this wrong would produce a
candidate that correlates with the TARGET by construction and look like a huge
win, so it is asserted below against a feature already in the matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import _common                                                        # noqa: E402

NULL_PRECEDENT = 0.077          # confirmed-null candidate family, for scale
EXISTING = ["elo_logit_pwinner", "hier_logit_pwinner", "elo_overall_diff",
            "elo_surface_diff", "h2h_diff", "age_diff",
            "serve_form_diff_targeted", "return_form_diff_targeted"]


def log(m=""):
    print(m, flush=True)


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an intercept; returns residuals and adjusted R^2."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    n, p = len(y), A.shape[1] - 1
    return resid, 1 - (1 - r2) * (n - 1) / (n - p - 1)


feat = pd.read_parquet(_common.SCRATCH / "xgb_features_v10_train.parquet")
log(f"training matrix: {len(feat):,} rows")

for mode in ("point", "game"):
    path = _common.SCRATCH / f"serve_return_elo_ratings_{mode}.parquet"
    if not path.exists():
        log(f"\n{mode}: {path.name} missing, skipped")
        continue
    e = pd.read_parquet(path)
    e["date"] = pd.to_datetime(e["date"])
    m = feat.merge(e, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cov = m["serve_w"].notna().mean()
    log(f"\n=== {mode}-level serve/return Elo ===")
    log(f"  joined onto {cov:.1%} of training rows "
        f"({int(m['serve_w'].notna().sum()):,})")
    m = m[m["serve_w"].notna() & m[EXISTING].notna().all(axis=1)].copy()
    if len(m) < 5000:
        log(f"  only {len(m):,} usable rows; too few to screen")
        continue

    # A minus B, where A is the winner exactly when y == 1
    flip = np.where(m["y"].values == 1, 1.0, -1.0)
    m["serve_elo_diff"] = (m["serve_w"].values - m["serve_l"].values) * flip
    m["return_elo_diff"] = (m["return_w"].values - m["return_l"].values) * flip

    # orientation check: a correctly built difference must correlate with an
    # existing correctly built one in the SAME direction
    chk = float(np.corrcoef(m["serve_elo_diff"], m["elo_overall_diff"])[0, 1])
    log(f"  orientation check vs elo_overall_diff: corr {chk:+.3f} "
        f"{'ok' if chk > 0 else 'INVERTED - stop'}")
    assert chk > 0, "the candidate is oriented against the existing features"

    y = m["y"].values.astype(float)
    base_r, base_adj = fit(m[EXISTING].values.astype(float), y)
    log(f"  existing predictors alone: adjusted R2 {base_adj:.5f}")

    for cand in ("serve_elo_diff", "return_elo_diff"):
        partial = float(np.corrcoef(base_r, m[cand].values)[0, 1])
        _, adj = fit(m[EXISTING + [cand]].values.astype(float), y)
        log(f"    + {cand:<18} adjusted R2 {adj:.5f}  (delta {adj - base_adj:+.5f}), "
            f"partial corr {partial:+.4f}")
    _, adj_both = fit(m[EXISTING + ["serve_elo_diff", "return_elo_diff"]].values.astype(float), y)
    log(f"    + both              adjusted R2 {adj_both:.5f}  "
        f"(delta {adj_both - base_adj:+.5f})")

    best = max(abs(float(np.corrcoef(base_r, m[c].values)[0, 1]))
               for c in ("serve_elo_diff", "return_elo_diff"))
    log(f"  max partial correlation {best:.4f} against the confirmed-null "
        f"precedent of {NULL_PRECEDENT:.3f}")
    log(f"  VERDICT: {'worth a walk-forward' if best > NULL_PRECEDENT else 'below the null precedent, do not spend the harness run'}")

log("\nSCREEN_COMPLETE")
