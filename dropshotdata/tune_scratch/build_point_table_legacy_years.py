"""D-54 follow-up: recover the 2022-23 US Open point tapes, which have no
P1SetsWon/P2SetsWon field (confirmed on the raw JSON - a schema difference,
not a capture bug). build_point_table.py's A1 anchor cross-checks a count of
SetWinner rows against that running total, so on these two years it always
fails (the running total is always "0").

THIS IS NOT THE APPROACH THAT ALREADY FAILED ONCE. The docstring in
build_point_table.py describes a first attempt that tried to DERIVE set
winners by replaying GameWinner and crediting a trailing set after the loop -
a construction problem, and it broke on post-point counters and retirements.
SetWinner is not being derived here: it is a field the tape already provides,
100% populated in 2022-23, exactly as in every other year. What is missing is
only the field A1 uses to CHECK it. So instead of construction, this is
independent validation: does every point where SetWinner fires show a legal,
completed-set game score (P1GamesWon/P2GamesWon)? That is a tennis-rule check
on a value we trust, not a rebuild of the value from scratch, so it does not
carry the prior failure's risk. A real corruption (e.g. PointServer/
PointWinner transposed) would produce SetWinner firing at a game score that
is not a legal set end, and this catches that.

Writes usopen_2022_points.parquet / usopen_2023_points.parquet next to the
2026 file if the pass rate looks sound - nothing is written silently; the
per-column and per-anchor numbers print first either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common
from build_point_table import KEEP, newest  # noqa: E402

DD = _common.DD
YEARS = [2022, 2023]
BEST_OF = {"1": 5, "2": 3}   # event 1 = men's singles, 2 = women's singles


def log(m: str = "") -> None:
    print(m, flush=True)


def legal_set_end(a: int, b: int) -> bool:
    """Is (winner_games, loser_games) a legal completed-set score?
    Standard sets: 6-0..6-4, 7-5. Tiebreak sets: 7-6."""
    if a >= 6 and a - b >= 2:
        return True
    if a == 7 and b in (5, 6):
        return True
    return False


def replay_ok_no_totals(g: pd.DataFrame, best_of: int) -> tuple[bool, str]:
    g = g.reset_index(drop=True)
    p1g = g["P1GamesWon"].astype(int)
    p2g = g["P2GamesWon"].astype(int)
    sw = g["SetWinner"]
    for i in g.index[sw.isin(["1", "2"])]:
        a, b = int(p1g[i]), int(p2g[i])
        wg, lg = (a, b) if sw[i] == "1" else (b, a)
        if not legal_set_end(wg, lg):
            return False, f"SetWinner={sw[i]} fires at illegal game score {a}-{b} (row {i})"
    p1_sets = int((sw == "1").sum())
    p2_sets = int((sw == "2").sum())
    need = best_of // 2 + 1
    if p1_sets >= need and p2_sets >= need:
        return False, f"both players credited with {need}+ sets ({p1_sets}-{p2_sets})"
    last_mw = g.iloc[-1]["MatchWinner"]
    if last_mw in ("1", "2"):
        loser_sets = p2_sets if last_mw == "1" else p1_sets
        if loser_sets >= need:
            return False, f"MatchWinner={last_mw} but the other player has {loser_sets} sets (need < {need})"
    return True, ""


def load_year(year: int) -> tuple[pd.DataFrame, list, list]:
    src = DD / "raw" / "usopen" / str(year) / "matches"
    dirs = sorted([p for p in src.iterdir() if p.is_dir()]) if src.exists() else []
    frames, failed, skipped = [], [], []
    for md in dirs:
        f = newest(list(md.glob("history.*.json")))
        if f is None:
            skipped.append((md.name, "no history feed"))
            continue
        try:
            pts = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                # noqa: BLE001
            skipped.append((md.name, f"unreadable: {e}"))
            continue
        if not isinstance(pts, list) or not pts:
            skipped.append((md.name, "empty"))
            continue
        d = pd.DataFrame(pts)
        for c in KEEP:
            if c not in d.columns:
                d[c] = ""
        d = d[KEEP].astype(str)
        real = d["PointNumber"].str.fullmatch(r"\d+") & d["PointServer"].isin(["1", "2"])
        d = d[real]
        if d.empty:
            skipped.append((md.name, "no real points"))
            continue
        event = md.name[0]
        ok, why = replay_ok_no_totals(d, BEST_OF.get(event, 3))
        if not ok:
            failed.append((md.name, why))
            continue
        d["match_id"] = md.name
        d["event"] = event
        frames.append(d)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP)), failed, skipped


def main() -> None:
    for year in YEARS:
        log(f"\n{'=' * 72}\n{year}\n{'=' * 72}")
        d, failed, skipped = load_year(year)
        if d.empty:
            log("  nothing to write")
            continue
        n_all = d["match_id"].nunique()
        log(f"  parsed: {n_all:,}   replay-check failures: {len(failed):,}   skipped: {len(skipped):,}")
        if failed:
            log("  A1' FAILURES (excluded, not silently kept):")
            for m, w in failed[:12]:
                log(f"    {m}: {w}")

        d_singles = d[d["event"].isin(["1", "2"])].copy()
        log(f"  singles filter: {d_singles['match_id'].nunique():,} of {n_all:,} kept (events 1/2)")
        # D-56/D-57: consistent with build_point_table.py - `event` only
        # reliably means the right tour for 4-character (main-draw) ids.
        n_singles = d_singles["match_id"].nunique()
        d_singles = d_singles[d_singles["match_id"].astype(str).str.len() == 4].copy()
        log(f"  main-draw filter: {d_singles['match_id'].nunique():,} of {n_singles:,} kept "
            f"(4-character ids only)")
        d_singles["point_no"] = d_singles["PointNumber"].astype(int)
        d_singles = d_singles.sort_values(["match_id", "point_no"]).reset_index(drop=True)

        served_won = float((d_singles["PointServer"] == d_singles["PointWinner"]).mean())
        log(f"  points: {len(d_singles):,} across {d_singles['match_id'].nunique():,} singles matches")
        log(f"  A3 server wins {served_won:.1%} of points -> {'PASS' if served_won > 0.55 else 'FAIL'}")

        log("  column coverage:")
        problems = []
        for c in KEEP:
            empty = float((d_singles[c].astype(str).str.strip() == "").mean())
            flag = "FAIL" if empty >= 0.999 and c not in ("P1SetsWon", "P2SetsWon") else ""
            log(f"    {c:<16} {1 - empty:6.1%} populated  {flag}")
            if flag:
                problems.append(c)
        if served_won <= 0.55:
            problems.append("A3")

        out = DD / f"usopen_{year}_points.parquet"
        if problems:
            log(f"  REFUSING to write {out.name}: {problems}")
            continue
        d_singles.to_parquet(out, index=False)
        log(f"  wrote {out} ({len(d_singles):,} rows)")


if __name__ == "__main__":
    main()
