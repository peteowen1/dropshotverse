"""Rounds 13b + 14, on the compact harness.

PART A - decompose round 13's bundled axis.
screen_ret_in_elo.py reported -0.00057 for "score retirements at reduced K",
but its `ret_k_mult` guard controlled TWO things at once: whether a retirement
moves the ratings, and whether it advances the match counters that drive K's
decay (stats-discipline #5). These arms separate them:

    A0  ret_update=0.0  ret_counts=False   today's engine (baseline)
    A1  ret_update=0.0  ret_counts=True    counter only, ratings untouched
    A2  ret_update=0.5  ret_counts=False   rating delta only, counters untouched
    A3  ret_update=0.5  ret_counts=True    both (the old bundled arm)

Predicted before running: if A1 ~ A3, the result was never about retirements -
it was about K decaying too slowly, and part B is the real finding. If A2 ~ A3,
the retirement story stands.

PART B - grid the Sackmann K schedule, which has never been tuned in this
project. `k_factor(n) = max(K_MIN, K_BASE / (n+5)**K_SHAPE)` with
(250.0, 0.4, 16.0) inherited from Sackmann. A grep for K_BASE/K_SHAPE/K_MIN
across every script in tune_scratch/ returns only today's files. Ticket 26
replaced the schedule wholesale (UTS, rejected); round 2 multiplied it by
margin-of-victory (rejected); ticket 25's layoff decay acts on the RATING, not
on K. The schedule's own shape is untouched.

Limiting case: (250.0, 0.4, 16.0) must reproduce the baseline exactly. Asserted
below, not assumed - verify_compact_identity.py already proved the harness
matches elo_baseline.py at these values.

Screened on 2022-23. The 2024-25 market holdout is NOT touched.

Every arm appends to the output parquet as it completes, so a kill costs one
arm rather than the run (the previous two attempts at this work were killed by
the OS for memory).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact, eval_mask, logloss, run_elo_fast  # noqa: E402

OUT = DD / "tune_scratch" / "screen_k_decomp.parquet"


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    d = build_compact()
    ev = eval_mask(d, years=(2022, 2023))
    log(f"coverage: {len(d['day']):,} rows, {int(d['n_players'][0]):,} players")
    log(f"  eval: {int(ev.sum()):,} tour-level completed matches, 2022-23")
    log(f"  RET rows {int((d['kind'] == 1).sum()):,}  "
        f"W/O rows {int((d['kind'] == 2).sum()):,}")

    rows: list[dict] = []

    def arm(label: str, **kw) -> float:
        t0 = time.time()
        ll = logloss(run_elo_fast(d, **kw)[ev])
        rows.append(dict(arm=label, logloss=ll, secs=round(time.time() - t0, 1), **{
            k: v for k, v in kw.items()}))
        pd.DataFrame(rows).to_parquet(OUT, index=False)   # checkpoint every arm
        log(f"  {label:<34} log-loss={ll:.6f}  ({time.time() - t0:.0f}s)")
        return ll

    log("\n=== PART A: decompose the round-13 axis (W/O held at 0) ===")
    a0 = arm("A0 baseline (today's engine)", ret_update=0.0, ret_counts=False,
             wo_update=0.0, wo_counts=False)
    a1 = arm("A1 counter only", ret_update=0.0, ret_counts=True,
             wo_update=0.0, wo_counts=False)
    a2 = arm("A2 rating delta only", ret_update=0.5, ret_counts=False,
             wo_update=0.0, wo_counts=False)
    a3 = arm("A3 both (old bundled arm)", ret_update=0.5, ret_counts=True,
             wo_update=0.0, wo_counts=False)

    log(f"\n  deltas vs A0:  counter-only {a1 - a0:+.6f}   "
        f"rating-only {a2 - a0:+.6f}   bundled {a3 - a0:+.6f}")
    if abs(a1 - a3) < abs(a2 - a3):
        log("  -> the COUNTER carries it. Round 13 was never about retirements; "
            "it is a statement about K's decay rate. Part B is the real finding.")
    else:
        log("  -> the RATING DELTA carries it. The retirement story stands.")

    log("\n=== PART B: Sackmann K schedule grid (retirements left dropped) ===")
    log("  limiting case first: (250.0, 0.4, 16.0) must equal A0 exactly")
    lim = arm("B-limit (250, 0.4, 16)", k_base=250.0, k_shape=0.4, k_min=16.0)
    assert abs(lim - a0) < 1e-12, f"limiting case FAILED: {lim} vs {a0}"
    log("  limiting case -> PASS")

    for k_shape in (0.3, 0.4, 0.5):
        for k_base in (150.0, 250.0, 350.0):
            for k_min in (8.0, 16.0, 24.0):
                if (k_base, k_shape, k_min) == (250.0, 0.4, 16.0):
                    continue
                arm(f"B kb={k_base:g} ks={k_shape:g} kmin={k_min:g}",
                    k_base=k_base, k_shape=k_shape, k_min=k_min)

    tab = pd.DataFrame(rows)
    b = tab[tab["arm"].str.startswith("B")].copy()
    b["delta"] = b["logloss"] - a0
    log("\n--- K grid, delta vs today's schedule ---")
    log(b.sort_values("logloss")[["arm", "logloss", "delta"]]
        .head(10).to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    best = b.loc[b["logloss"].idxmin()]
    log(f"\n  best: {best['arm']}  delta {best['delta']:+.6f}")
    log(f"  (compare: current XGBoost gap to market = 0.00073)")
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
