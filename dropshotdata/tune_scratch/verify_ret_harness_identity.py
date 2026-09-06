"""Limiting-case verification for candidate 13 (stats-discipline #5 rule 2).

screen_ret_in_elo.py claims ret_k_mult=0.0 is mathematically identical to the
shipped engine, because a zero-K update leaves every rating untouched. That
claim was written down before running the screen and then NOT actually checked -
the script printed the published parquet's schema and moved on. Every number in
that screen is meaningless until this passes, so it runs before the result is
reported anywhere.

This is an IDENTITY check between two implementations, not a scoring reveal on
the 2024-25 window: no model is selected, tuned, or compared on the strength of
it. It compares my harness's p_winner against elo_baseline.py's own published
elo_eval_predictions.parquet, row for row, on the rows they share.

PASS condition, fixed in advance: max |delta p_winner| < 1e-9 on >= 99.9% of
joined rows. Anything larger means the harness differs from elo_baseline.py on
an axis I have not noticed, and candidate 13 is withdrawn until it is found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))
sys.path.insert(0, str(DD / "tune_scratch"))

from screen_ret_in_elo import load_all_levels, prepare_keeping_ret, run  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    df = prepare_keeping_ret(load_all_levels())
    preds = run(df, 0.0)
    mine = df[["date", "tour", "winner_uid", "loser_uid", "level_file"]].copy()
    mine["p_mine"] = preds
    mine = mine[mine["p_mine"].notna()]

    pub = pd.read_parquet(DD / "elo_eval_predictions.parquet",
                          columns=["date", "tour", "winner_uid", "loser_uid", "p_winner"])
    log(f"published rows: {len(pub):,}  ({pub['date'].min().date()} to "
        f"{pub['date'].max().date()})")
    log(f"harness rows (scored): {len(mine):,}")

    key = ["date", "tour", "winner_uid", "loser_uid"]
    # A player pair can meet twice on one date at different rounds; drop those
    # ambiguous keys from BOTH sides rather than letting the join fan out.
    dup_p = pub.duplicated(key, keep=False)
    dup_m = mine.duplicated(key, keep=False)
    log(f"  ambiguous keys excluded: published {int(dup_p.sum()):,}, "
        f"harness {int(dup_m.sum()):,}")
    j = mine[~dup_m].merge(pub[~dup_p], on=key, how="inner")
    log(f"  joined: {len(j):,} rows")

    d = (j["p_mine"] - j["p_winner"]).abs()
    frac_ok = float((d < 1e-9).mean())
    log(f"\n  max |delta| = {d.max():.3e}")
    log(f"  mean |delta| = {d.mean():.3e}")
    log(f"  fraction within 1e-9: {frac_ok:.5%}")
    ok = frac_ok >= 0.999
    log(f"\n  LIMITING CASE -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = j.loc[d.sort_values(ascending=False).index[:5]]
        log("\n  worst disagreements:")
        log(worst.to_string(index=False))
        log("\n  candidate 13 is WITHDRAWN until this is explained.")
    else:
        log("  harness reproduces elo_baseline.py exactly at ret_k_mult=0; the "
            "screen's deltas are attributable to the retirement update alone.")


if __name__ == "__main__":
    main()
