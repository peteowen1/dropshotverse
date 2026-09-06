"""Audit the US Open point-by-point capture WHILE IT CAN STILL BE FIXED.

The capture task has been running since 2026-08-27 and has accumulated 11,580
JSON files / 710 MB across 597 matches, and nobody has checked what is in it.
The tournament ends around 2026-09-13. After that the feed is gone and anything
not captured cannot be captured - so an audit is worth far more today than it
will be in a fortnight.

The `history` feed is the valuable one: a list of per-point records
(SetNo, GameNo, PointNumber, ServeNumber, Stage, PointID, EpochTimeStart/End,
MatchWinner, SetWinner). That is the raw material for the live in-match win
probability the map calls the justification for a point-level architecture, and
which has never been built beyond ticket 09's prototype.

What matters is not that files exist - it is whether each match's point sequence
is COMPLETE. The capture polls on a schedule, so a match could easily be caught
only partway: files present, feed valid, and the last third of the match simply
missing. That failure is invisible from a file count.

Checks, per match:
  1. is there a history feed at all
  2. does the newest snapshot reach a finished match (MatchWinner populated)
  3. are point numbers contiguous from 1, or are there gaps mid-sequence
  4. how many points, and does that look sane for the sets played

Read-only. Touches nothing the live pipeline writes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

D = Path(r"C:\dev\dropshotverse\dropshotdata\raw\usopen\2026\matches")


def log(m: str = "") -> None:
    print(m, flush=True)


def newest(files: list[Path]) -> Path | None:
    """Snapshots are named <feed>.<epoch>.<hash>.json - pick the latest epoch."""
    best, best_t = None, -1
    for f in files:
        m = re.match(r"[a-z]+\.(\d+)\.", f.name)
        if m and int(m.group(1)) > best_t:
            best, best_t = f, int(m.group(1))
    return best


def main() -> None:
    match_dirs = sorted([p for p in D.iterdir() if p.is_dir()])
    log(f"matches with a directory: {len(match_dirs):,}")

    rows = []
    for md in match_dirs:
        hist = list(md.glob("history.*.json"))
        rec = {"match": md.name, "n_hist_snapshots": len(hist)}
        if not hist:
            rec.update(has_history=False)
            rows.append(rec)
            continue
        f = newest(hist)
        try:
            pts = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                # noqa: BLE001
            rec.update(has_history=True, readable=False, err=str(e)[:60])
            rows.append(rec)
            continue
        if not isinstance(pts, list) or not pts:
            rec.update(has_history=True, readable=True, n_points=0)
            rows.append(rec)
            continue

        nums = []
        for p in pts:
            try:
                nums.append(int(str(p.get("PointNumber", "")).strip() or -1))
            except ValueError:
                nums.append(-1)
        nums = [n for n in nums if n > 0]
        sets = {str(p.get("SetNo", "")).strip() for p in pts}
        winner = str(pts[-1].get("MatchWinner", "")).strip()
        rec.update(
            has_history=True, readable=True, n_points=len(pts),
            n_numbered=len(nums),
            min_pt=min(nums) if nums else None,
            max_pt=max(nums) if nums else None,
            contiguous=(bool(nums) and sorted(nums) == list(range(1, max(nums) + 1))),
            n_sets=len({s for s in sets if s and s != "0"}),
            match_winner=winner,
            finished=winner not in ("", "0", "None"),
        )
        rows.append(rec)

    d = pd.DataFrame(rows)
    log("")
    log("=== coverage (stats-discipline #7) ===")
    log(f"  with a history feed:      {int(d['has_history'].fillna(False).sum()):,} "
        f"/ {len(d):,}")
    got = d[d["has_history"].fillna(False) & d["readable"].fillna(False)]
    log(f"  readable:                 {len(got):,}")
    if not len(got):
        return
    fin = got[got["finished"].fillna(False)]
    log(f"  reaching a finished match:{len(fin):,} "
        f"({len(fin) / len(got):.1%})")
    contig = got[got["contiguous"].fillna(False)]
    log(f"  contiguous point numbers: {len(contig):,} "
        f"({len(contig) / len(got):.1%})")
    log(f"  total points captured:    {int(got['n_points'].sum()):,}")
    log(f"  points per match: median {got['n_points'].median():.0f}, "
        f"min {got['n_points'].min():.0f}, max {got['n_points'].max():.0f}")
    log(f"  snapshots per match: median {got['n_hist_snapshots'].median():.0f}")

    log("\n=== the failures that matter ===")
    nohist = d[~d["has_history"].fillna(False)]
    log(f"  no history feed at all: {len(nohist):,}")
    if len(nohist):
        log(f"    e.g. {list(nohist['match'].head(12))}")
    unfin = got[~got["finished"].fillna(False)]
    log(f"  history present but match never finished in it: {len(unfin):,}")
    if len(unfin):
        log(f"    e.g. {list(unfin['match'].head(12))}")
    gappy = got[~got["contiguous"].fillna(False)]
    log(f"  non-contiguous point numbering: {len(gappy):,}")
    if len(gappy):
        sample = gappy.head(6)[["match", "n_points", "min_pt", "max_pt"]]
        log(sample.to_string(index=False))

    tiny = got[got["n_points"] < 30]
    log(f"  suspiciously short (<30 points): {len(tiny):,}")

    out = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch"
               r"\audit_usopen_capture.parquet")
    d.to_parquet(out, index=False)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
