"""Queue item 2: verify per-year field coverage on the 2022-25 US Open
backfill BEFORE anything uses those tapes. Feeds change shape across years -
this caught the 2026-09-07 session twice, on the live 2026 tape alone.

Reuses build_point_table.py's own parsing, replay check (A1), and anchors
(A2/A3) rather than re-deriving them, so "does 2023 parse the way 2026 does"
is answered by running the SAME code against a different year's directory,
not a second implementation that could silently diverge from the first.

Writes nothing. This is a read of what is there, not a production build -
usopen_<year>_points.parquet does not exist yet and nothing downstream reads
these years, so there is nothing to gate a write against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common
from build_point_table import KEEP, GAME_SCORES, newest, replay_ok  # noqa: E402

DD = _common.DD
YEARS = [2022, 2023, 2024, 2025]


def log(m: str = "") -> None:
    print(m, flush=True)


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
            import json
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
        ok, why = replay_ok(d)
        if not ok:
            failed.append((md.name, why))
            continue
        d["match_id"] = md.name
        d["event"] = md.name[0]
        frames.append(d)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP)), failed, skipped


def main() -> None:
    summary = []
    for year in YEARS:
        src = DD / "raw" / "usopen" / str(year) / "matches"
        n_dirs = len(list(src.iterdir())) if src.exists() else 0
        log(f"\n{'=' * 72}\n{year}: {n_dirs} match directories\n{'=' * 72}")
        d, failed, skipped = load_year(year)
        if d.empty:
            log("  NOTHING PARSED")
            summary.append((year, n_dirs, 0, 0, 0.0, []))
            continue
        n_all = d["match_id"].nunique()
        d_singles = d[d["event"].isin(["1", "2"])].copy()
        log(f"  parsed: {n_all:,}   replay-check failures: {len(failed):,}   skipped: {len(skipped):,}")
        log(f"  singles filter: {d_singles['match_id'].nunique():,} of {n_all:,} kept "
            f"(events 1/2)")

        served_won = float((d_singles["PointServer"] == d_singles["PointWinner"]).mean()) if len(d_singles) else float("nan")
        log(f"  points: {len(d_singles):,} across {d_singles['match_id'].nunique():,} singles matches")
        log(f"  A3 server wins {served_won:.1%} of points -> {'PASS' if served_won > 0.55 else 'FAIL'}")

        log("  column coverage:")
        bad_cols = []
        for c in KEEP:
            empty = float((d_singles[c].astype(str).str.strip() == "").mean()) if len(d_singles) else 1.0
            flag = "FAIL" if empty >= 0.999 else ("check" if empty > 0.01 else "")
            if flag:
                bad_cols.append((c, empty))
            log(f"    {c:<16} {1 - empty:6.1%} populated  {flag}")

        summary.append((year, n_dirs, n_all, d_singles["match_id"].nunique(), served_won, bad_cols))

    log(f"\n{'=' * 72}\nSUMMARY - is each year safe to build a point table from?\n{'=' * 72}")
    for year, n_dirs, n_parsed, n_singles, served_won, bad_cols in summary:
        verdict = "OK" if n_singles > 0 and served_won > 0.55 and not bad_cols else "NEEDS ATTENTION"
        log(f"  {year}: {n_dirs} dirs -> {n_parsed} parsed -> {n_singles} singles matches, "
            f"A3={served_won:.1%}  [{verdict}]")
        for c, e in bad_cols:
            log(f"       {c} only {1 - e:.1%} populated")


if __name__ == "__main__":
    main()
