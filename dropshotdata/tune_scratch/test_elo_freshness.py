"""elo_freshness must fail when set or game Elo lags overall Elo, and pass
when they line up (OPUS-redteam-2026-10-25 F1: game Elo sat at 2026-06-02 for
two months beside a current overall Elo and nothing failed).

Offline: tiny synthetic parquets in a temp dir. Needs pandas and pyarrow.

Also holds the pruning rules for the published upcoming-event tables (issue
#44, event_tables.py). They sit here because tests.yml already runs this file
with exactly the packages they need; simulate_wta_event.py itself needs xgboost.
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
TMP = Path(tempfile.mkdtemp())
os.environ["DROPSHOT_DATA_DIR"] = str(TMP)
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))
import elo_freshness as ef  # noqa: E402
import event_tables as et  # noqa: E402

fails = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        fails.append(label)


def write(name, last, last_wta=None, lower_after=None):
    """Tour-level rows ending at `last` (atp) and `last_wta` (wta, default
    `last`); optionally a lower-level row after that, which must not count."""
    rows = []
    for tour, end in (("atp", last), ("wta", last_wta or last)):
        if end is None:
            continue
        rows += [{"date": pd.Timestamp("2026-01-05"), "tour": tour, "level_file": "tour"},
                 {"date": pd.Timestamp(end), "tour": tour, "level_file": "tour"}]
        if lower_after:
            rows.append({"date": pd.Timestamp(lower_after), "tour": tour, "level_file": "lower"})
    p = TMP / name
    pd.DataFrame(rows).to_parquet(p)
    return p


def raises(overall, set_, game, **kw):
    try:
        ef.check_elo_inputs_fresh(overall, set_, game, **kw)
        return None
    except ef.StaleEloInputError as e:
        return str(e)


overall = write("overall.parquet", "2026-10-12")

# The F1 shape: set and game frozen at Roland Garros, overall current.
frozen_set = write("set_frozen.parquet", "2026-06-02")
frozen_game = write("game_frozen.parquet", "2026-06-02")
msg = raises(overall, frozen_set, frozen_game)
check("frozen set+game (132 days behind) raises", msg is not None)
check("message names both files", msg is not None and "set Elo ends 2026-06-02" in msg and "game Elo ends 2026-06-02" in msg)

fresh_set = write("set_fresh.parquet", "2026-10-12")
fresh_game = write("game_fresh.parquet", "2026-10-12")
check("all three aligned passes", raises(overall, fresh_set, fresh_game) is None)

check("only game frozen still raises", raises(overall, fresh_set, frozen_game) is not None)

# Lag right at the limit passes, one day over fails.
at_limit = write("game_14.parquet", "2026-09-28")
over = write("game_15.parquet", "2026-09-27")
check("14 days behind passes", raises(overall, fresh_set, at_limit) is None)
check("15 days behind raises", raises(overall, fresh_set, over) is not None)

# One tour stale is enough.
wta_stale = write("game_wta_stale.parquet", "2026-10-12", last_wta="2026-06-02")
check("wta-only lag raises", raises(overall, fresh_set, wta_stale) is not None)

# A tour missing entirely is not vacuously fine.
no_wta = write("game_no_wta.parquet", "2026-10-12", last_wta=None)
pd.DataFrame([{"date": pd.Timestamp("2026-10-12"), "tour": "atp", "level_file": "tour"}]).to_parquet(no_wta)
check("a tour with no rows raises", raises(overall, fresh_set, no_wta) is not None)

# Lower-level rows do not make a stale tour-level file look fresh.
lower_only = write("game_lower.parquet", "2026-06-02", lower_after="2026-10-12")
check("lower-level rows ignored", raises(overall, fresh_set, lower_only) is not None)

# The command line exits non-zero on a stale file.
rc = ef.main(["--overall", str(overall), "--set", str(frozen_set), "--game", str(frozen_game)])
check("CLI exits 1 on stale input", rc == 1)
rc = ef.main(["--overall", str(overall), "--set", str(fresh_set), "--game", str(fresh_game)])
check("CLI exits 0 on fresh input", rc == 0)

# ---- issue #44: upcoming-event tables hold only events that have not started.
AS_OF = pd.Timestamp("2026-10-26")
china = pd.DataFrame({"tournament_id": "2026-1020", "start_date": "2026-09-30",
                      "player_id": [f"wta:{i}" for i in range(76)], "p_title": 1 / 76})
kept, dropped = et.prune_started(china, AS_OF)
check("China Open (started 2026-09-30) is dropped", dropped == ["2026-1020"])
check("pruning every event leaves zero rows", len(kept) == 0)
check("empty result keeps the columns", list(kept.columns) == list(china.columns))
kept.to_parquet(TMP / "empty_upcoming.parquet", index=False)
back = pd.read_parquet(TMP / "empty_upcoming.parquet")
check("empty table round-trips through parquet with its schema",
      len(back) == 0 and list(back.columns) == list(china.columns) and back["start_date"].dtype == china["start_date"].dtype)

mixed = pd.DataFrame({
    "tournament_id": ["past", "today", "future", "garbage", "missing"],
    "start_date": ["2026-09-23", "2026-10-26", "2026-11-01", "not a date", None]})
kept, dropped = et.prune_started(mixed, AS_OF)
check("event starting on as-of is kept, later kept",
      list(kept["tournament_id"]) == ["today", "future"])
check("past, unparseable and missing dates are dropped", dropped == ["garbage", "missing", "past"])

# A date with an offset must not raise TypeError against the tz-naive as-of,
# and keeps its calendar date as written.
offs = pd.DataFrame({"tournament_id": ["a", "b", "c"],
                     "start_date": ["2026-10-25T23:00:00+10:00", "2026-10-26T00:30:00-08:00", "2026-11-02T00:00:00Z"]})
try:
    kept, dropped = et.prune_started(offs, AS_OF)
    check("offset-bearing start dates do not raise and keep their written date",
          dropped == ["a"] and list(kept["tournament_id"]) == ["b", "c"])
except TypeError as e:
    check(f"offset-bearing start dates do not raise ({e})", False)
check("calendar_date drops the offset, keeps the written day",
      et.calendar_date("2026-09-30T00:00:00+10:00") == pd.Timestamp("2026-09-30"))
check("calendar_date of a tz-aware as-of is tz-naive", et.calendar_date(pd.Timestamp("2026-10-26", tz="UTC")).tz is None)
check("calendar_date of garbage is NaT", pd.isna(et.calendar_date("not a date")) and pd.isna(et.calendar_date(None)))
kept, dropped = et.prune_started(mixed.iloc[:0], AS_OF)
check("empty input gives empty output", len(kept) == 0 and dropped == [])

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
