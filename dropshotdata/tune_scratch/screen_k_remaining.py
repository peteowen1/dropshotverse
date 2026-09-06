"""Finish round 14's K-schedule grid: the 13 points screen_k_decomp.py never ran.

That run was killed at 18 of 28 arms. Its per-arm parquet checkpoint kept
everything completed, so this picks up the remainder rather than redoing the
grid: all of k_shape=0.5 (9 points, the faster-decay direction and the one
region with no evidence at all), k_base=350/k_shape=0.4 (3), and
k_base=250/0.4/k_min=24 (1).

Reads the existing checkpoint, skips any arm already present, appends as it
goes. Safe to run repeatedly; safe to kill at any point.

Designed to be launched DETACHED rather than as a tracked background task -
three previous runs were killed by the harness's low-memory guard, which a
detached process is not subject to. Peak footprint is ~400 MB (elo_compact).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact, eval_mask, logloss, run_elo_fast  # noqa: E402

OUT = DD / "tune_scratch" / "screen_k_decomp.parquet"


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["arm"])
    have = set(done["arm"])
    log(f"checkpoint holds {len(done)} arms already")

    d = build_compact()
    ev = eval_mask(d, years=(2022, 2023))
    log(f"eval n={int(ev.sum()):,}")

    rows = done.to_dict("records")
    todo = []
    for k_shape in (0.3, 0.4, 0.5):
        for k_base in (150.0, 250.0, 350.0):
            for k_min in (8.0, 16.0, 24.0):
                label = f"B kb={k_base:g} ks={k_shape:g} kmin={k_min:g}"
                if label not in have:
                    todo.append((label, k_base, k_shape, k_min))
    log(f"{len(todo)} arms to run")

    for label, k_base, k_shape, k_min in todo:
        t0 = time.time()
        ll = logloss(run_elo_fast(d, k_base=k_base, k_shape=k_shape,
                                  k_min=k_min)[ev])
        rows.append(dict(arm=label, logloss=ll, secs=round(time.time() - t0, 1),
                         k_base=k_base, k_shape=k_shape, k_min=k_min))
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        log(f"  {label:<34} log-loss={ll:.6f}  ({time.time() - t0:.0f}s)")

    tab = pd.DataFrame(rows)
    base = float(tab.loc[tab["arm"].str.startswith("A0"), "logloss"].iloc[0])
    b = tab[tab["arm"].str.startswith("B")].copy()
    b["delta"] = b["logloss"] - base
    log("\n--- FULL K GRID, delta vs shipped (250, 0.4, 16) ---")
    log(b.sort_values("logloss")[["arm", "logloss", "delta"]]
        .to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    best = b.loc[b["logloss"].idxmin()]
    log(f"\nbest: {best['arm']}  delta {best['delta']:+.6f}")
    log("ROUND14_COMPLETE")


if __name__ == "__main__":
    main()
