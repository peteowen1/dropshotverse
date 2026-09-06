"""Round 14b: push the K grid past the boundary the first grid stopped at.

The 27-point grid's best cell was k_base=350, k_shape=0.5, k_min=16 at
-0.000616 vs the shipped (250, 0.4, 16). That is a DOUBLE boundary result -
both k_base=350 and k_shape=0.5 are the top of their axes - so it says the
optimum is outside the grid, not that it is at this corner
(stats-discipline #5.2). Reporting it as the answer would be reporting the edge
of my own search as a property of tennis.

Second concern worth resolving in the same pass: the k_min surface is not
smooth. At k_shape=0.5, k_min=16 beat both 8 and 24 by ~0.00047 - a sharp spike
at exactly the shipped value, where a genuine optimum should be gradual. Either
there is an interaction with k_base/k_shape, or that cell is noise. This grid
samples k_min more finely (12/16/20/28) to tell which.

Mechanism, for what to expect: K = k_base / (n+5)^k_shape. Raising both k_base
and k_shape together holds early-career K roughly constant while decaying it
faster, so the winning direction reads as "converge a new player quickly, then
move an established player less". If that is real it should continue past
k_shape=0.5 and then turn over; if it does not turn over anywhere in this grid,
the result is still a boundary and still not shippable.

Tune window 2022-23 only. The 2024-25 holdout is retired (see docs/DECISIONS.md
D-10); 2026 is reserved as the final unbiased test.

Appends to its own parquet, one row per arm, and is safe to re-run or kill.
Launch DETACHED - the harness's low-memory guard kills tracked background tasks
on this machine, but not a detached process.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact, eval_mask, logloss, run_elo_fast  # noqa: E402

OUT = DD / "tune_scratch" / "screen_k_extend.parquet"
BASE_LL = 0.5999548080567144        # A0 / shipped (250, 0.4, 16), verified 3x

K_BASES = (350.0, 450.0, 550.0, 700.0)
K_SHAPES = (0.5, 0.6, 0.7)
K_MINS = (12.0, 16.0, 20.0, 28.0)


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["arm"])
    have = set(done["arm"])
    rows = done.to_dict("records")

    d = build_compact()
    ev = eval_mask(d, years=(2022, 2023))
    log(f"eval n={int(ev.sum()):,}   baseline (shipped) = {BASE_LL:.6f}")

    todo = [(f"kb={b:g} ks={s:g} kmin={m:g}", b, s, m)
            for s in K_SHAPES for b in K_BASES for m in K_MINS]
    todo = [t for t in todo if t[0] not in have]
    log(f"{len(todo)} arms to run")

    for label, kb, ks, km in todo:
        t0 = time.time()
        ll = logloss(run_elo_fast(d, k_base=kb, k_shape=ks, k_min=km)[ev])
        rows.append(dict(arm=label, logloss=ll, delta=ll - BASE_LL,
                         k_base=kb, k_shape=ks, k_min=km,
                         secs=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        log(f"  {label:<26} {ll:.6f}  {ll - BASE_LL:+.6f}  ({time.time() - t0:.0f}s)")

    tab = pd.DataFrame(rows).sort_values("logloss")
    log("\n--- top 12 ---")
    log(tab.head(12)[["arm", "logloss", "delta"]]
        .to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    best = tab.iloc[0]
    log(f"\nbest: {best['arm']}  delta {best['delta']:+.6f}")
    on_edge = []
    if best["k_base"] in (min(K_BASES), max(K_BASES)):
        on_edge.append("k_base")
    if best["k_shape"] in (min(K_SHAPES), max(K_SHAPES)):
        on_edge.append("k_shape")
    if best["k_min"] in (min(K_MINS), max(K_MINS)):
        on_edge.append("k_min")
    log(f"boundary check: {'ON EDGE for ' + ', '.join(on_edge) if on_edge else 'INTERIOR optimum'}")
    log("EXTEND_COMPLETE")


if __name__ == "__main__":
    main()
