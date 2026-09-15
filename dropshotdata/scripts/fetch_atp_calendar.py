"""The ATP calendar, from Wikipedia - the only usable source found so far.

api.atptour.com does not resolve (server-rendered Sitecore); there is no ATP
equivalent of fetch_wta_upcoming.py's api.wtatennis.com. Wikipedia's season
article (`2026 ATP Tour`) turns out to carry both a full calendar (all levels,
dates, draw-size breakdown) AND, on the same page, a live rankings table -
see docs/plans/all-events-forecast.md for how that rankings table was used
and where it failed (our own Elo is NOT a safe substitute, D-58 nearby: it
dropped Sinner and Alcaraz from a recency-filtered top 20 because of a real
tennismylife-bridge coverage gap for exactly those two players).

WHAT THIS DOES NOT DO: a full entry list per tournament. Wikipedia's own
per-tournament articles carry a seeds table (sometimes populated, sometimes a
bare template - checked directly on the 2026 China Open) and a withdrawals
table, never a full acceptance list this far out. Building an ATP field still
needs that separate piece (docs/plans/all-events-forecast.md item 2/3).

Fetch-on-demand, cached to disk, same discipline fetch_wta_upcoming.py
documents for the WTA calendar: "do not let 'the endpoint is open' stand in
for 'we are permitted to'." This is a season page fetched once a run, not a
poll.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

import _common

DD = _common.DD
CACHE = DD / "raw" / "atp_wiki"
OUT = DD / "reference" / "atp_calendar_2026.csv"
URL = "https://en.wikipedia.org/wiki/{year}_ATP_Tour"
UA = ("dropshotverse/0.1 (non-commercial research; "
      "github.com/peteowen1/dropshotverse)")

LEVEL_RE = re.compile(r"(Grand Slam|United Cup|ATP Finals|Next Gen ATP Finals|"
                      r"Laver Cup|Davis Cup|ATP 1000|ATP 500|ATP 250)")
# The page's own infobox states the total tournament count for the season
# (65 for 2026). A parse landing anywhere else means a category went
# missing the way Grand Slam did the first time this script ran - caught
# then only because someone happened to grep the output for "wimbledon".
TOTAL_RE = re.compile(r'"tournaments"\s*:\s*"(\d+)"')
DRAW_RE = re.compile(r"(\d+)S/(\d+)Q(?:/(\d+)D)?")


def log(m: str = "") -> None:
    print(m, flush=True)


def fetch_html(year: int, refetch: bool) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"atp_tour_{year}.html"
    if not refetch and path.exists():
        log(f"  (cached) {path.name}")
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": UA})
    time.sleep(1.0)
    with urllib.request.urlopen(req, timeout=20) as r:                 # noqa: S310
        html = r.read().decode("utf-8")
    path.write_text(html, encoding="utf-8")
    return html


def parse_calendar(html: str) -> pd.DataFrame:
    tmp = CACHE / "_parse_tmp.html"
    tmp.write_text(html, encoding="utf-8")
    tables = pd.read_html(tmp)
    tmp.unlink()

    seen: set[tuple[str, str]] = set()
    rows = []
    # The calendar is split across ~12 month-by-month tables, each with
    # Week/Tournament/Champions/... columns - found by inspection, not a
    # documented API, so this is located by SHAPE (has both columns) rather
    # than a fixed index, which would silently break if Wikipedia inserts or
    # removes a table above the calendar.
    for t in tables:
        if "Week" not in t.columns or "Tournament" not in t.columns:
            continue
        for _, r in t.iterrows():
            wk = str(r.get("Week", "")).strip()
            raw = str(r.get("Tournament", "")).strip()
            if not raw or raw == "nan":
                continue
            key = (wk, raw)
            if key in seen:
                continue
            seen.add(key)
            lv = LEVEL_RE.search(raw)
            if not lv:
                continue   # a data row with no recognised level is not a real event
            dm = DRAW_RE.search(raw)
            rows.append({
                "week": wk,
                "name_loc": raw[:lv.start()].strip(),
                "level": lv.group(1),
                "draw_s": int(dm.group(1)) if dm else None,
                "draw_q": int(dm.group(2)) if dm else None,
                "draw_d": int(dm.group(3)) if dm and dm.group(3) else None,
                "raw": raw,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    log(f"fetching {args.year} ATP Tour calendar from Wikipedia...")
    html = fetch_html(args.year, args.refetch)
    df = parse_calendar(html)

    log(f"  {len(df)} events with a recognised level")
    log("  by level: " + ", ".join(f"{k}={v}" for k, v in df["level"].value_counts().items()))
    if df.empty:
        raise SystemExit("parsed 0 events - the page structure has likely changed; "
                         "do not write an empty calendar over a real one")

    # Coverage, not just presence (stats-discipline): the page's own infobox
    # states the season's total tournament count. A parse missing an entire
    # LEVEL_RE category still writes a non-empty, plausible-looking file - it
    # is not "0 events", it is 4 fewer, which is exactly how Grand Slam went
    # missing the first time this script ran and nobody noticed until someone
    # happened to grep the output for "wimbledon".
    m = TOTAL_RE.search(html)
    if m and len(df) != int(m.group(1)):
        raise SystemExit(f"parsed {len(df)} events but the page's own infobox states "
                         f"{m.group(1)} - a whole category is likely missing from "
                         f"LEVEL_RE; refusing to write a silently incomplete calendar")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    log(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
