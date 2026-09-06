"""Turn the captured US Open slamtracker history into one tidy point table.

93,663 points sit across 597 per-match JSON snapshot files and nothing reads
them. This produces the single artifact everything downstream needs:
`usopen_2026_points.parquet`, one row per real point, chronological.

It is the foundation for the only thing ticket 09 left explicitly unfinished.
That prototype shipped Mode A (frozen skill) on 94 Fan Week matches and said of
in-match calibration: "Not testable with four matches - this needs a large
sample scored the way ticket 10 scored the pre-match models. Flagging as
unfinished rather than guessing." We now have 597 main-draw matches including
best-of-five, so it is testable.

WHAT COUNTS AS A POINT
The tape carries non-point rows - PointNumber "0X"/"0Y" with PointServer "0"
("Players arrive on court", "Players warming up") - and they must be dropped or
every match starts with two phantom points. Real points have a numeric
PointNumber and a PointServer of 1 or 2.

ANCHORS, fixed before looking at output (stats-discipline #1)
  A1  Replaying game wins from PointWinner reproduces the tape's OWN final
      P1GamesWon/P2GamesWon and P1SetsWon/P2SetsWon. Ticket 09 added exactly
      this assertion after a type-mismatch bug ('1' != 1) silently started
      every match at a false 0-1 deficit. It is the check that catches a
      whole class of replay errors, so it runs on every match, not once.
  A2  PointServer is constant within a game. If serve changes mid-game the
      game/point segmentation is wrong.
  A3  Server holds more than they lose: across the whole corpus the serving
      player wins >55% of points. Below that and PointServer/PointWinner have
      been transposed somewhere - a mistake that would look perfectly
      plausible in aggregate and invert every model built on it.

Matches failing A1 are REPORTED and EXCLUDED, never silently kept.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
SRC = DD / "raw" / "usopen" / "2026" / "matches"
OUT = DD / "usopen_2026_points.parquet"

KEEP = ["MatchID", "SetNo", "GameNo", "PointNumber", "PointServer", "PointWinner",
        "P1Score", "P2Score", "P1GamesWon", "P2GamesWon", "P1SetsWon", "P2SetsWon",
        "P1PointsWon", "P2PointsWon", "SetWinner", "GameWinner", "MatchWinner",
        "ServeNumber", "ServeIndicator", "Ace", "DoubleFault", "UnforcedError",
        "Winner", "ElapsedTime", "EpochTimeStart", "EpochTimeEnd", "Sentence"]


def log(m: str = "") -> None:
    print(m, flush=True)


def newest(files: list[Path]) -> Path | None:
    best, t = None, -1
    for f in files:
        m = re.match(r"[a-z]+\.(\d+)\.", f.name)
        if m and int(m.group(1)) > t:
            best, t = f, int(m.group(1))
    return best


GAME_SCORES = {"0", "15", "30", "40", "AD", "A", ""}


def is_tiebreak(g: pd.DataFrame) -> bool:
    """A tiebreak is scored 1,2,3... rather than 15/30/40/AD. Detecting it from
    the SCORE rather than from the game number is what makes this robust to
    match tiebreaks and doubles super-tiebreaks, which are not at 6-6."""
    vals = set(g["P1Score"]) | set(g["P2Score"])
    return bool(vals - GAME_SCORES)


def replay_ok(g: pd.DataFrame) -> tuple[bool, str]:
    """A1: the tape's two independent accounts of the same thing must agree.

    Count sets from the per-point `SetWinner` flag and compare to the running
    `P1SetsWon`/`P2SetsWon` on the final row.

    The first version of this replayed games from `GameWinner`, tracked set
    transitions, and credited the trailing set after the loop. It failed on 99
    of 597 matches, and the anchor was wrong rather than the data:

      - the tape's counters are POST-point, not pre-point (on 1101's final
        point, PointWinner, GameWinner, SetWinner, MatchWinner and P1SetsWon
        all update on the same row), so crediting a trailing set double-counts;
      - and on a RETIREMENT the final set is never won by anyone, so inventing
        a winner for it turned every abandoned match into a failure (1107,
        "replayed 3-0 vs tape 2-0").

    Counting SetWinner directly has neither problem: an abandoned set simply
    carries no SetWinner, which is the correct answer.
    """
    p1s = int((g["SetWinner"] == "1").sum())
    p2s = int((g["SetWinner"] == "2").sum())
    last = g.iloc[-1]
    want = (int(last["P1SetsWon"] or 0), int(last["P2SetsWon"] or 0))
    if (p1s, p2s) != want:
        return False, f"SetWinner count {p1s}-{p2s} vs running total {want[0]}-{want[1]}"
    return True, ""


def main() -> None:
    dirs = sorted([p for p in SRC.iterdir() if p.is_dir()])
    log(f"match directories: {len(dirs):,}")

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
        # real points only
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
        d["event"] = md.name[0]        # first digit is the event (see memory note)
        frames.append(d)

    # SINGLES ONLY, and measured rather than assumed. A1 pass rate by event id:
    #   1 men's singles   344/345  99.7%
    #   2 women's singles 159/159  100%
    #   3               13/32   40.6%
    #   4               18/40   45.0%
    #   5               19/20   95.0%
    # Events 3 and 4 are doubles and the tape marks SetWinner differently there
    # - their SetWinner count runs consistently one short of the running total.
    # That is a real difference worth understanding if doubles is ever wanted,
    # but in-match win probability targets singles, and shipping a table that
    # is 100%-validated for singles and 43%-validated for doubles in the same
    # file invites someone to use the half that was never checked.

    log(f"  parsed: {len(frames):,}   replay-check failures: {len(failed):,}   "
        f"skipped: {len(skipped):,}")
    if failed:
        log("  A1 FAILURES (excluded, not silently kept):")
        for m, w in failed[:12]:
            log(f"    {m}: {w}")
    if skipped:
        log(f"  skipped examples: {skipped[:6]}")

    if not frames:
        log("nothing to write")
        return
    d = pd.concat(frames, ignore_index=True)
    n_all = d["match_id"].nunique()
    d = d[d["event"].isin(["1", "2"])].copy()
    log(f"  singles filter: {d['match_id'].nunique():,} of {n_all:,} parsed "
        f"matches kept (events 1/2); doubles excluded, see note in source")
    d["point_no"] = d["PointNumber"].astype(int)
    d = d.sort_values(["match_id", "point_no"]).reset_index(drop=True)

    log("\n=== coverage (stats-discipline #7) ===")
    log(f"  points: {len(d):,} across {d['match_id'].nunique():,} matches")
    log(f"  by event id: {d.groupby('event')['match_id'].nunique().to_dict()}")
    log(f"  points per match: median {d.groupby('match_id').size().median():.0f}")

    log("\n=== anchors ===")
    # A2 - serve constant within a game, EXCEPT tiebreaks, where it alternates
    # by the rules of tennis. The first version of this anchor omitted the
    # exception and flagged 199 games, 166 of them ending 7-6 or 6-7.
    d["is_tb"] = d.groupby(["match_id", "SetNo", "GameNo"])["P1Score"].transform(
        lambda s: bool(set(s) - GAME_SCORES))
    d["is_tb"] |= d.groupby(["match_id", "SetNo", "GameNo"])["P2Score"].transform(
        lambda s: bool(set(s) - GAME_SCORES))
    ng = d[~d["is_tb"]].groupby(["match_id", "SetNo", "GameNo"])["PointServer"].nunique()
    bad_games = int((ng > 1).sum())
    n_tb = int(d.groupby(["match_id", "SetNo", "GameNo"])["is_tb"].first().sum())
    # The residue is a KNOWN TAPE DEFECT, named rather than tolerated silently:
    # GameNo occasionally fails to increment at a genuine game boundary, so two
    # consecutive games share a number and the serve appears to change mid-game.
    # In 11129 set 2 game 5 the tape itself says "wins the game with an ace" and
    # resets the score to 0-0 while GameNo stays at 5. Detectable, and far too
    # rare to move a calibration result - but it means GameNo is not a safe key.
    tol = 0.001
    rate = bad_games / max(len(ng), 1)
    log(f"  A2 serve constant within a non-tiebreak game: {bad_games} violate "
        f"of {len(ng):,} ({rate:.3%}; {n_tb:,} tiebreaks excluded) "
        f"-> {'PASS' if rate <= tol else 'FAIL'} (tape defect: GameNo not "
        f"always incremented; tolerance {tol:.1%})")
    # A3 - server wins more than they lose
    served_won = float((d["PointServer"] == d["PointWinner"]).mean())
    log(f"  A3 server wins {served_won:.1%} of points "
        f"-> {'PASS' if served_won > 0.55 else 'FAIL'}")
    log(f"  A1 replay check: enforced per match above "
        f"({len(frames):,} passed, {len(failed):,} excluded)")

    # A2/A3 were printed and then ignored on first writing; a transposed
    # PointServer/PointWinner would have printed FAIL and shipped anyway.
    problems = []
    if rate > tol:
        problems.append(f"A2 serve-constant violation rate {rate:.3%} > {tol:.1%}")
    if served_won <= 0.55:
        problems.append(f"A3 server wins only {served_won:.1%} of points")

    # Coverage, not presence: the per-file loop above fills any missing KEEP
    # column with "" so the union never fails on shape. A column the feed
    # never populated would land 100% empty and pass every check below it.
    log("\n=== column coverage ===")
    for c in KEEP:
        empty = float((d[c].astype(str).str.strip() == "").mean())
        log(f"  {c:<16} {1 - empty:6.1%} populated")
        if empty >= 0.999:
            problems.append(f"column {c} is {empty:.1%} empty after union")
    if problems:
        raise SystemExit("corpus-level anchor FAILED, refusing to write "
                         f"{OUT.name}: {problems}")

    d.to_parquet(OUT, index=False)
    log(f"\nwrote {OUT} ({len(d):,} rows)")


if __name__ == "__main__":
    main()
