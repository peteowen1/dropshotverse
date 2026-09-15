"""A LIVE, deep ATP ranking list - the piece D-58-adjacent work (see
docs/plans/all-events-forecast.md item 2) said was still missing: Wikipedia's
own "ATP rankings as of <date>" table only goes to 20, nowhere near enough to
construct a field for even a 32-draw's direct acceptances, let alone a
96-draw Masters. Our own archive/Elo cannot substitute (D-58: it dropped
Sinner and Alcaraz from a recency-filtered top 20 outright, a real
tennismylife-bridge coverage gap for exactly the players who matter most).

tennisexplorer.com/ranking/atp-men/ was found via search and VALIDATED before
trusting it, per this repo's own discipline (a different candidate site
failed exactly this check and was rejected the same session): its top 20,
checked 2026-09-15, matches Wikipedia's own verified "as of 14 September
2026" table EXACTLY - same rank order, same points, same names, for all 20.
No other claim about this site has been checked; only that its top 20 agrees
with a source already trusted. Paginated (?page=N, 50 rows/page); this pulls
enough pages to cover the largest ATP main draw (96, so >=150 ranked players
comfortably covers direct acceptances plus a buffer for players who skip).

WHAT THIS STILL DOES NOT SOLVE: who actually entered a SPECIFIC tournament.
A high ranking is not an entry - Alcaraz ranks above several of Beijing's
confirmed WTA... (ATP, corrected) seeds and is not IN that draw (D-58/plan
item 2). This ranking list is the best-effort "who's good enough to plausibly
be in the draw" input to field construction, cross-checked against whatever
real per-tournament seeds/withdrawals Wikipedia has - it is not a substitute
for a real entry list, which no source found so far provides.

Fetch-on-demand, cached, paced one second apart - same discipline
fetch_wta_upcoming.py documents.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd

import _common

DD = _common.DD
CACHE = DD / "raw" / "tennisexplorer"
OUT = DD / "reference" / "atp_rankings.csv"
BASE = "https://www.tennisexplorer.com/ranking/atp-men/"
UA = ("dropshotverse/0.1 (non-commercial research; "
      "github.com/peteowen1/dropshotverse)")
PAUSE = 1.0


def log(m: str = "") -> None:
    print(m, flush=True)


def fetch_page(page: int, refetch: bool) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"atp_men_p{page}.html"
    if not refetch and path.exists():
        return path.read_text(encoding="utf-8")
    url = BASE if page == 1 else f"{BASE}?page={page}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    time.sleep(PAUSE)
    with urllib.request.urlopen(req, timeout=20) as r:                 # noqa: S310
        html = r.read().decode("utf-8")
    path.write_text(html, encoding="utf-8")
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="50 ranks per page")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    frames = []
    for page in range(1, args.pages + 1):
        html = fetch_page(page, args.refetch)
        tmp = CACHE / "_parse_tmp.html"
        tmp.write_text(html, encoding="utf-8")
        tables = pd.read_html(tmp)
        tmp.unlink()
        t = next((t for t in tables if "Rank" in t.columns and "Player name" in t.columns), None)
        if t is None or t.empty:
            raise SystemExit(f"page {page}: no ranking table found - page structure "
                             f"has likely changed, refusing to write a partial list")
        frames.append(t[["Rank", "Player name", "Country", "Points"]])
        log(f"  page {page}: ranks {int(t['Rank'].min())}-{int(t['Rank'].max())}")

    df = pd.concat(frames, ignore_index=True)
    df["Rank"] = df["Rank"].astype(int)
    dupes = df["Rank"].duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate ranks across pages - pagination is not "
                         f"behaving as expected, refusing to write")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    log(f"wrote {OUT} ({len(df)} ranked players, {df['Rank'].min()}-{df['Rank'].max()})")


if __name__ == "__main__":
    main()
