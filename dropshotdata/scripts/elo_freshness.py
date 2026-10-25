"""Staleness guard on the three Elo replays the shipped model reads.

OPUS-redteam-2026-10-25 F1: set-level and game-level Elo were built once, on
2026-08-29, from the Sackmann archive alone, and never again. The daily
ratings job regenerated overall and surface Elo with the tennismylife bridge,
so for two months the published ratings paired a current overall Elo with a
game Elo (the model's largest input, D-43) frozen at 2026-06-02. Nothing
failed, because every reader quietly takes "each player's last rating before
date0", and a stale last rating looks exactly like a real one.

This check compares, per tour, the last tour-level match date in each of the
set and game files against the overall file's. A lag beyond MAX_LAG_DAYS
raises, so the build stops instead of publishing. Run it directly to check the
files on disk:

    python elo_freshness.py [--max-lag-days 14]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

import _common

MAX_LAG_DAYS = 14
TOURS = ("atp", "wta")

OVERALL_PATH = Path(os.environ.get("DSV_ELO_RAW_PATH", str(_common.SCRATCH / "elo_raw_ratings_full.parquet")))
SET_PATH = Path(os.environ.get("DSV_SET_ELO_PATH", str(_common.SCRATCH / "set_elo_raw_ratings_full.parquet")))
GAME_PATH = Path(os.environ.get("DSV_GAME_ELO_PATH", str(_common.SCRATCH / "game_elo_raw_ratings_full.parquet")))


class StaleEloInputError(RuntimeError):
    """A set or game Elo file lags the overall Elo file."""


def last_tour_dates(path: Path) -> dict[str, pd.Timestamp | None]:
    """Last tour-level match date per tour. None when a tour has no rows."""
    r = pd.read_parquet(path, columns=["date", "tour", "level_file"])
    r = r[r["level_file"] == "tour"]
    out: dict[str, pd.Timestamp | None] = {}
    for tour in TOURS:
        d = r.loc[r["tour"] == tour, "date"]
        out[tour] = pd.Timestamp(d.max()) if len(d) else None
    return out


def check_elo_inputs_fresh(overall: Path = OVERALL_PATH, set_: Path = SET_PATH, game: Path = GAME_PATH,
                           max_lag_days: int = MAX_LAG_DAYS) -> dict[str, dict[str, pd.Timestamp | None]]:
    """Raise StaleEloInputError if set or game Elo's last tour-level match
    date, for either tour, is more than max_lag_days behind overall Elo's.
    Returns the dates it compared so a caller can log them."""
    dates = {"overall": last_tour_dates(overall), "set": last_tour_dates(set_), "game": last_tour_dates(game)}
    problems = []
    for tour in TOURS:
        ref = dates["overall"][tour]
        if ref is None:
            problems.append(f"{tour}: overall Elo ({overall}) has no tour-level rows")
            continue
        for name, path in (("set", set_), ("game", game)):
            got = dates[name][tour]
            if got is None:
                problems.append(f"{tour}: {name} Elo ({path}) has no tour-level rows")
                continue
            lag = (ref - got).days
            if lag > max_lag_days:
                problems.append(f"{tour}: {name} Elo ends {got.date()}, overall Elo ends {ref.date()} "
                                f"({lag} days behind, limit {max_lag_days}); {path}")
    if problems:
        raise StaleEloInputError(
            "stale Elo input(s), refusing to rate with them:\n  " + "\n  ".join(problems)
            + "\nRegenerate with: python regen_set_game_elo_with_bridge.py")
    return dates


_checked: set[tuple] = set()


def check_once(overall: Path = OVERALL_PATH, set_: Path = SET_PATH, game: Path = GAME_PATH) -> None:
    """check_elo_inputs_fresh, at most once per version of the three files in
    a process (build_player_table calls this per tour and per field)."""
    key = tuple((str(p), p.stat().st_mtime_ns) for p in (overall, set_, game))
    if key in _checked:
        return
    dates = check_elo_inputs_fresh(overall, set_, game)
    print("[elo_freshness] last tour-level match: "
          + "; ".join(f"{tour} overall {dates['overall'][tour].date()} set {dates['set'][tour].date()} "
                      f"game {dates['game'][tour].date()}" for tour in TOURS), flush=True)
    _checked.add(key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overall", type=Path, default=OVERALL_PATH)
    ap.add_argument("--set", dest="set_", type=Path, default=SET_PATH)
    ap.add_argument("--game", type=Path, default=GAME_PATH)
    ap.add_argument("--max-lag-days", type=int, default=MAX_LAG_DAYS)
    a = ap.parse_args(argv)
    try:
        dates = check_elo_inputs_fresh(a.overall, a.set_, a.game, a.max_lag_days)
    except StaleEloInputError as e:
        print(f"::error::{e}" if os.environ.get("GITHUB_ACTIONS") else f"ERROR: {e}", flush=True)
        return 1
    for tour in TOURS:
        print(f"elo inputs fresh, {tour}: overall {dates['overall'][tour].date()}, set {dates['set'][tour].date()}, "
              f"game {dates['game'][tour].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
