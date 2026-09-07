"""Settle arm C against arm D on the rounds that had not been played.

The whole in-match comparison rests on 231 matches of which only 7 were past
the third round, because the 2026 US Open was still being played when it was
measured. Arm D lost on those 7 and fails anchor D3, so C is what ships - but 7
matches is a signal, not a result, and the charter says so.

This is the one command that turns it into a result once the men's final is
done (2026-09-13). It rebuilds the point table from everything the capture has
collected, re-runs both arms, and prints the late-round comparison next to the
numbers on record so the difference is legible rather than something to go
hunting for.

    python rescore_after_final.py             # rebuild, re-score, compare
    python rescore_after_final.py --no-rebuild  # re-score the table as it is

Writes nothing to blog/ and touches no published artifact. It only reads the
capture and writes under tune_scratch/, so it is safe to run early - it will
simply report that the later rounds are still missing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

import _common
import usopen_live_draw as draw_mod

HERE = Path(__file__).resolve().parent
EVENTS = ("1", "2")          # 1 = men's singles, 2 = women's singles
SCRATCH = _common.DD / "tune_scratch"

# What was on record when only 7 round-4 matches existed, from
# docs/plans/validation-charter.md. Printed beside the new numbers so a shift
# is obvious without going and looking it up.
ON_RECORD = {
    "all_rounds": {"published": (0.1063, 0.59066, 0.747),
                   "A": (0.0297, 0.52936, 0.724),
                   "C": (0.0484, 0.48310, 0.782),
                   "D": (0.0278, 0.47283, 0.782)},
    "late": {"C": (0.1266, 0.22068, 0.906),
             "D": (0.1672, 0.26615, 0.906)},
    # 7 is how many round-4 matches the SCREEN scored. The table holds 8 - one
    # is skipped for having no pre-match probability - so the gate below
    # compares against the table's own count, not against this.
    "n_late_scored": 7,
    "n_late_in_table": 8,
}


def log(m: str = "") -> None:
    print(m, flush=True)


def run(script: str, *args: str) -> str:
    """Run one of the pipeline scripts, echoing it, and return its output."""
    cmd = [sys.executable, str(HERE.parent / "tune_scratch" / script)
           if script.startswith("screen_") else str(HERE / script), *args]
    log(f"\n$ {Path(cmd[1]).name} {' '.join(args)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"{script} failed with exit {p.returncode}; "
                         f"refusing to report a comparison from a broken run")
    return p.stdout


def round_coverage() -> dict[int, int]:
    """Main-draw SINGLES matches in the point table, by round.

    The table holds 562 matches and only 232 of them are the two singles draws
    this comparison is about; the other 330 are qualifying, doubles and
    juniors. They are told apart by id LENGTH - a main-draw singles id is four
    characters (event, round, two-digit match) and everything else is five.
    Reading the second character of a five-character id as its round is what
    produced "30 quarter-finals", which a 128 draw cannot have.

    The round comes from the draw module's own parser rather than from slicing,
    because match_id's layout is exactly the sort of structural field that
    rewards being read rather than assumed.
    """
    pts = _common.DD / "usopen_2026_points.parquet"
    if not pts.exists():
        return {}
    d = pd.read_parquet(pts, columns=["match_id"])
    rounds: dict[int, int] = {}
    for mid in d["match_id"].astype(str).unique():
        if len(mid) != 4 or mid[0] not in EVENTS:
            continue
        try:
            rnd = draw_mod.parse_match_id(mid)[0]
        except Exception:                                    # noqa: BLE001
            continue
        rounds[rnd] = rounds.get(rnd, 0) + 1
    return dict(sorted(rounds.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rebuild", action="store_true",
                    help="score the point table as it stands instead of rebuilding it")
    args = ap.parse_args()

    log("=" * 72)
    log("Settling arm C against arm D on the late rounds")
    log("=" * 72)

    before = round_coverage()
    log(f"point table before: {before or 'absent'}")

    if not args.no_rebuild:
        run("build_point_table.py")
    after = round_coverage()
    log(f"\npoint table after:  {after}")
    gained = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    log(f"matches added by this rebuild: "
        f"{ {k: v for k, v in gained.items() if v} or 'none'}")

    # Rounds 5, 6 and 7 are the quarter-finals, semi-finals and final - the
    # rounds the original comparison could not see at all.
    late_now = sum(v for r, v in after.items() if r >= 4)
    log(f"\nmain-draw singles from round 4 on: {late_now} "
        f"(was {ON_RECORD['n_late_in_table']} when arm C was chosen, of which "
        f"{ON_RECORD['n_late_scored']} could be scored)")
    if late_now <= ON_RECORD["n_late_in_table"]:
        log("\nNo new late-round matches. Either the final has not been played or the")
        log("capture has not collected it yet. Re-running would restate what is")
        log("already on record, so stopping here rather than producing a")
        log("comparison that looks new and is not.")
        raise SystemExit(0)

    run("screen_inmatch_seeded.py")
    out = run("screen_inmatch_recalibrated.py")

    log("\n" + "=" * 72)
    log("What was on record, decided on 7 late-round matches")
    log("=" * 72)
    log(f"{'':<34}{'ECE':>9}{'log-loss':>11}{'acc':>8}")
    for label, key in (("published pre-match", "published"), ("arm A", "A"),
                       ("arm C, seeded", "C"), ("arm D, + recalibrated", "D")):
        e, l, a = ON_RECORD["all_rounds"][key]
        log(f"  all rounds  {label:<20}{e:>9.4f}{l:>11.5f}{a:>8.3f}")
    for label, key in (("arm C, seeded", "C"), ("arm D, + recalibrated", "D")):
        e, l, a = ON_RECORD["late"][key]
        log(f"  late rounds {label:<20}{e:>9.4f}{l:>11.5f}{a:>8.3f}")

    log("\n" + "=" * 72)
    log("The question this run answers")
    log("=" * 72)
    log("Arm C beat arm D on the late rounds on 7 matches, and arm D fails its")
    log("own anchor D3 because the two tours want different temperatures. If")
    log("both still hold above, C over D is settled on evidence rather than on")
    log("a signal. If the late-round result has REVERSED, that is the answer")
    log("too - a wide interval resolves in either direction, and D-18 should be")
    log("revisited rather than defended.")
    log("\nRecord the outcome in docs/DECISIONS.md and the table in")
    log("docs/plans/validation-charter.md either way.")
    if "D3" in out:
        log("\nD3 above is the one to read first: it is what ruled arm D out.")
    log("\nRESCORE_COMPLETE")


if __name__ == "__main__":
    main()
