"""Fail when the published tennis data has gone stale during a live event.

THE FAILURE THIS EXISTS TO CATCH
`DropshotverseUSOpenLiveSim`'s trigger ends 2026-09-14. When it expires - or if
the task is disabled, or the machine is off, or the sim starts erroring - the
files on R2 simply stop changing. Nothing notices:

  - `refresh_usopen_live.cmd` guards the sim/upload ORDER (a failed sim must not
    republish stale data with a fresh timestamp), but it only runs when the task
    runs. A task that never fires cannot report its own failure.
  - ITG's `validate-tennis-sim.mjs` checks coverage and per-player probability
    gaps. A frozen file passes every one of those: it is structurally valid, the
    numbers are self-consistent, and only `updated_at` gives it away. Nothing
    read `updated_at`.

So the site can serve a days-old forecast as if it were live, indefinitely, with
every existing check green. That is the gap.

WHAT IT CHECKS
  1. `updated_at` is younger than --max-age-hours (default 2, comfortably more
     than the 30-minute refresh cadence).
  2. Staleness is only an ERROR while the event is still running. A finished
     tournament SHOULD stop updating - that is correct behaviour, not a fault -
     so `event_status == "complete"` downgrades a stale file to an informational
     note. This is why the flag and the check shipped together; the check is
     unusable without it, because after the final everything is stale forever.
  3. The obvious structural things, since it is reading the file anyway: rows
     present, both tours present, title probabilities summing to ~1 per tour.

EXIT CODES
  0  fresh, or legitimately complete
  1  STALE while live, or a structural problem
  2  could not read the file at all

Run it against the local publish directory (default) or any copy of the
parquet. To make it an actual alarm it needs scheduling - it is a check, not a
monitor, and on its own it changes nothing.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT = Path(r"C:\dev\dropshotverse\dropshotdata\blog\live_predictions.parquet")
# The public R2 copy the site actually reads (inthegame-blog/data-loader.js
# DATA_BASE_URL + "tennis/live_predictions.parquet"). Checking THIS copy is
# the point: the local file is always fresh right after the sim writes it, so
# a check run alongside the sim can never fail. Only the published copy can be
# stale-while-live.
PUBLIC_URL = ("https://pub-ee4bf5b599a047f9ac2b9facc1587008.r2.dev/"
              "tennis/live_predictions.parquet")


def log(m: str = "") -> None:
    print(m, flush=True)


def fetch_to_temp(url: str) -> Path:
    import tempfile
    import urllib.request
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache",
                                               "User-Agent": "dropshotverse-freshness-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    tmp = Path(tempfile.mkstemp(suffix=".parquet")[1])
    tmp.write_bytes(blob)
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--url", nargs="?", const=PUBLIC_URL, default=None,
                    help="check the published copy instead of a local file; "
                         "bare --url means the site's own R2 URL")
    ap.add_argument("--max-age-hours", type=float, default=2.0)
    args = ap.parse_args()

    if args.url:
        try:
            p = fetch_to_temp(args.url)
        except Exception as e:                               # noqa: BLE001
            log(f"FAIL: cannot fetch {args.url}: {e}")
            return 2
        log(f"url: {args.url}")
    else:
        p = Path(args.path)
    if not p.exists():
        log(f"FAIL: {p} does not exist")
        return 2
    try:
        d = pd.read_parquet(p)
    except Exception as e:                                   # noqa: BLE001
        log(f"FAIL: cannot read {p}: {e}")
        return 2

    log(f"file: {p}")
    log(f"rows: {len(d):,}")
    if not len(d):
        log("FAIL: file has no rows")
        return 1

    problems: list[str] = []

    # --- structure
    tours = sorted(d["tour"].unique()) if "tour" in d.columns else []
    log(f"tours: {tours}")
    if set(tours) != {"atp", "wta"}:
        problems.append(f"expected both tours, found {tours}")
    for tour in tours:
        s = float(d.loc[d["tour"] == tour, "p_title"].sum())
        log(f"  {tour}: {int((d['tour'] == tour).sum())} players, "
            f"title probs sum {s:.4f}")
        if not (0.99 <= s <= 1.01):
            problems.append(f"{tour} title probabilities sum to {s:.4f}, not ~1")

    # --- status
    if "event_status" in d.columns:
        statuses = dict(d.groupby("tour")["event_status"].first())
    else:
        statuses = {}
        log("NOTE: no event_status column - written by an older sim build, so "
            "staleness cannot be distinguished from a finished event")
    log(f"event_status: {statuses or 'unknown'}")
    all_complete = bool(statuses) and all(v == "complete" for v in statuses.values())
    if all_complete and "champion" in d.columns:
        champs = {t: d.loc[d["tour"] == t, "champion"].iloc[0] for t in tours}
        log(f"champions: {champs}")

    # --- freshness
    if "updated_at" not in d.columns:
        problems.append("no updated_at column - freshness cannot be checked")
    else:
        ts = pd.to_datetime(d["updated_at"].iloc[0], utc=True)
        age_h = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 3600
        log(f"updated_at: {ts.isoformat()}  ({age_h:.2f} h ago)")
        if age_h > args.max_age_hours:
            if all_complete:
                log(f"  stale by {age_h:.1f} h, but every event is complete - "
                    f"this is correct, a finished tournament stops updating")
            else:
                problems.append(
                    f"STALE: {age_h:.1f} h old (limit {args.max_age_hours}) "
                    f"while status is {statuses or 'unknown'} - the site is "
                    f"serving an out-of-date forecast as if it were live")
        else:
            log(f"  fresh (limit {args.max_age_hours} h)")

    log("")
    if problems:
        for m in problems:
            log(f"FAIL: {m}")
        return 1
    log("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
