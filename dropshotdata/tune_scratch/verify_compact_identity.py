"""Bit-identity check for the compact harness (stats-discipline #5 rule 2).

elo_compact.run_elo_fast() claims to compute exactly what elo_baseline.py
computes, at the shipped configuration (blend=0.0, decay=0.3, Sackmann K,
retirements dropped). It is a full rewrite - int32 player codes, flat numpy
arrays, a hand-inlined layoff_decay - so "it looks right" is worth nothing. The
slow harness earned its numbers by passing this same check at max |delta| =
0.000e+00 on 89,678 rows; this one has to earn them the same way.

PASS condition, fixed before running: max |delta p_winner| = 0 exactly on
>= 99.9% of joined rows. Not "small" - the arithmetic is deterministic float64
in a fixed order, so anything above 0 means the order or the formula differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact, run_elo_fast, KIND_OK  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    d = build_compact()
    preds = run_elo_fast(d, blend=0.0, decay=0.3,
                         ret_update=0.0, wo_update=0.0)

    ok = d["kind"] == KIND_OK
    uids = d["uids"]
    mine = pd.DataFrame({
        "date": d["day"][ok].astype("datetime64[D]"),
        "winner_uid": uids[d["w"][ok]],
        "loser_uid": uids[d["l"][ok]],
        "p_mine": preds[ok],
    })
    log(f"harness scored rows: {len(mine):,}")

    pub = pd.read_parquet(DD / "elo_eval_predictions.parquet",
                          columns=["date", "winner_uid", "loser_uid", "p_winner"])
    log(f"published rows: {len(pub):,}")

    key = ["date", "winner_uid", "loser_uid"]
    dm, dp = mine.duplicated(key, keep=False), pub.duplicated(key, keep=False)
    log(f"  ambiguous keys excluded: harness {int(dm.sum()):,}, "
        f"published {int(dp.sum()):,}")
    j = mine[~dm].merge(pub[~dp], on=key, how="inner")
    log(f"  joined: {len(j):,}")

    delta = (j["p_mine"] - j["p_winner"]).abs()
    exact = float((delta == 0.0).mean())
    log(f"\n  max |delta|  = {delta.max():.3e}")
    log(f"  exactly equal = {exact:.5%}")
    passed = exact >= 0.999
    log(f"\n  COMPACT HARNESS IDENTITY -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        w = j.loc[delta.sort_values(ascending=False).index[:5]]
        log(w.to_string(index=False))
        log("\n  Do not use this harness until explained.")
        sys.exit(1)
    log("  compact harness reproduces elo_baseline.py exactly; grid results "
        "from it are attributable to the parameter under test.")


if __name__ == "__main__":
    main()
