"""Capture US Open point-by-point tapes from the tournament's own public feeds.

Ticket 16. Two modes:
  backfill  - walk the completed-matches index and pull every match played so far
  poll      - watch live scores and re-pull in-progress matches until they finish

Everything is written verbatim. We never transform on ingest: the tape is the
evidence, and a parser bug should be fixable without a re-fetch we may not be
able to make (these feeds are tournament-scoped and vanish between events).

Read-only. Public JSON, no auth. Polite by construction: one worker, a delay
between every request, and exponential backoff on errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _common
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# The four majors run the same IBM-built platform, but not all of it is open.
# Probed on 2026-09-07, and the differences are real rather than cosmetic:
#
#   US Open    every feed answers, 2022 onward. eventDays enumerates.
#   Wimbledon  the POINT TAPE answers for 2025-2026, and so do draws, players,
#              complete and distance - but eventDays, seeds, live scores,
#              insight and match_stats are all 403. So it cannot be enumerated
#              the way the US Open is, and falls back to the draw feeds.
#   Roland Garros   nothing on this path shape.
#   Australian Open ausopen.com returns a 200 HTML page for EVERY path, which
#              is a soft 404 - it briefly looked like a hit until a check that
#              insists on JSON was applied.
#
# `enumerate_via` is therefore not a preference, it is a fact about each host.
SLAMS = {
    "usopen": {"host": "https://www.usopen.org", "loc": "en_US",
               "enumerate_via": "eventDays", "draw_codes": ("MS", "WS")},
    "wimbledon": {"host": "https://www.wimbledon.com", "loc": "en_GB",
                  "enumerate_via": "draws", "draw_codes": ("MS", "LS", "WS")},
}

SLAM = "usopen"
BASE = SLAMS[SLAM]["host"]
LOC = SLAMS[SLAM]["loc"]
YEAR = 2026
OUT = _common.RAW / SLAM / str(YEAR)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

# per-match feeds worth keeping. `history` is the point tape and the whole point
# of this exercise; the rest are cheap and contextual.
MATCH_FEEDS = {
    "history":  "/{loc}/scores/feeds/{y}/slamtracker/history/{mid}C.json",
    "complete": "/{loc}/scores/feeds/{y}/matches/complete/{mid}.json",
    "stats":    "/{loc}/scores/feeds/{y}/completed_matches/match_stats_{mid}.json",
    "distance": "/{loc}/scores/feeds/{y}/slamtracker/distance/{mid}.dist.json",
    "insight":  "/{loc}/scores/feeds/{y}/match_insight/{mid}.json",
}

DELAY = 0.7          # seconds between requests — we are a guest here
POLL_EVERY = 45      # seconds between live-score checks
ONCE_BUDGET_S = 240  # `once` runs every 5min (scheduled task) - stay well under
                     # that even if every feed stalls, so runs never overlap


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fetch(path: str, tries: int = 3) -> bytes | None:
    """GET a feed. Returns None on 404 (feed legitimately absent for this match)."""
    url = path if path.startswith("http") else BASE + path
    backoff = 2.0
    for attempt in range(tries):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=10) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 404:
                return None                      # not an error: many feeds are optional
            if e.code in (429, 503) and attempt < tries - 1:
                log(f"  {e.code} on {url.rsplit('/', 1)[-1]}, backing off {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 3
                continue
            log(f"  HTTP {e.code} {url}")
            return None
        except (URLError, TimeoutError, OSError, HTTPException) as e:
            # OSError covers ConnectionResetError mid-read (a WinError 10054
            # killed a whole sweep on 2026-09-02, the task's intermittent exit
            # 1); HTTPException covers IncompleteRead. Neither is a URLError.
            if attempt < tries - 1:
                time.sleep(backoff)
                backoff *= 3
                continue
            log(f"  network error {url}: {e}")
            return None
        finally:
            time.sleep(DELAY)
    return None


def write_if_changed(directory: Path, stem: str, blob: bytes) -> bool:
    """Content-addressed write. A match tape grows point by point, so we keep every
    distinct version — that history is what lets us replay a match as it unfolded."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(blob).hexdigest()[:12]
    existing = {p.name.split(".")[-2] for p in directory.glob(f"{stem}.*.json")}
    if digest in existing:
        return False
    epoch = int(time.time())
    (directory / f"{stem}.{epoch}.{digest}.json").write_bytes(blob)
    return True


def capture_match(mid: str, tag: str = "") -> dict[str, bool]:
    """Fetch every feed for one match. Returns which feeds were fetched in THIS
    call (not which changed on disk): the backfill needs to know that the tape
    it is about to call settled came from this pass, not from an earlier live
    sweep that left a partial copy behind."""
    fetched: dict[str, bool] = {}
    written = 0
    for name, tmpl in MATCH_FEEDS.items():
        blob = fetch(tmpl.format(loc=LOC, y=YEAR, mid=mid))
        fetched[name] = blob is not None
        if blob and write_if_changed(OUT / "matches" / str(mid), name, blob):
            written += 1
    if written:
        log(f"  match {mid}{tag}: {written} feed(s) written")
    return fetched


def capture_reference() -> None:
    """Tournament-level feeds: the draw, the player list, the feed catalogue itself."""
    refs = {
        "config_web":   f"/{LOC}/json/gen/config_web.json",
        "draws":        f"/{LOC}/scores/feeds/{YEAR}/draws/draws.json",
        "players":      f"/{LOC}/scores/feeds/{YEAR}/players/players.json",
        "competitors":  f"/{LOC}/scores/feeds/{YEAR}/players/competitors.json",
        "seeds":        f"/{LOC}/scores/feeds/{YEAR}/players/seeds_results_all.json",
        "event_days":   f"/{LOC}/scores/feeds/{YEAR}/completed_matches/eventDays.json",
        "schedule_days": f"/{LOC}/scores/feeds/{YEAR}/schedule/scheduleDays.json",
    }
    for name, path in refs.items():
        blob = fetch(path)
        if blob and write_if_changed(OUT / "reference", name, blob):
            log(f"  reference/{name}: {len(blob):,} bytes")


def match_ids_from(blob: bytes) -> set[str]:
    """Pull match ids out of any feed shape without assuming its schema."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key in ("match_id", "matchId", "id"):
                v = node.get(key)
                if isinstance(v, (str, int)) and str(v).isdigit():
                    found.add(str(v))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    try:
        walk(json.loads(blob))
    except json.JSONDecodeError:
        pass
    return found


DONE_FILE = OUT / "reference" / "backfill_done.json"


def load_done() -> set[str]:
    """Match ids whose feeds were captured AFTER they appeared in the completed
    index, i.e. whose tape is settled. Missing or unreadable means empty, which
    only costs one full pass."""
    try:
        return set(json.loads(DONE_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return set()


def save_done(done: set[str]) -> None:
    DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DONE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(done, key=int)), encoding="utf-8")
    tmp.replace(DONE_FILE)


def ids_from_draw_feeds() -> set[str]:
    """Every match id in this slam's draw feeds.

    Wimbledon serves the point tape but 403s `completed_matches/eventDays.json`,
    which is how the US Open backfill finds its matches. The draw feeds are open
    there and carry every match with its `winner`, so they enumerate just as
    well - one request per draw instead of one per tournament day.

    Only DECIDED matches are returned. An undecided one has no settled tape to
    capture, and would be re-fetched on the next run anyway.
    """
    ids: set[str] = set()
    for code in SLAMS[SLAM]["draw_codes"]:
        blob = fetch(f"/{LOC}/scores/feeds/{YEAR}/draws/{code}.json")
        if not blob:
            continue
        write_if_changed(OUT / "reference" / "draws", code, blob)
        try:
            matches = json.loads(blob).get("matches", [])
        except json.JSONDecodeError:
            log(f"  {code}.json did not parse")
            continue
        decided = [m for m in matches
                   if str(m.get("winner", "")) in ("1", "2") and m.get("match_id")]
        ids |= {str(m["match_id"]) for m in decided}
        log(f"  draw {code}: {len(matches)} matches, {len(decided)} decided")
    log(f"  {len(ids)} decided matches to capture")
    return ids


def capture_ids(ids: list[str], full: bool = False) -> None:
    """Capture every feed for each id, skipping ones already settled."""
    done = set() if full else load_done()
    todo = [i for i in ids if i not in done]
    log(f"  {len(todo)} not yet captured ({len(ids) - len(todo)} already settled)")
    for n, mid in enumerate(todo, 1):
        fetched = capture_match(mid, tag=f" [{n}/{len(todo)}]")
        if fetched.get("history"):
            done.add(mid)
        if n % 25 == 0:
            save_done(done)
    save_done(done)
    log(f"  backfill complete: {len(done)} matches settled")


def backfill(full: bool = False) -> None:
    """Capture the settled tape of every completed match not yet captured.

    Incremental by default. The first version re-fetched all ~600 matches x 6
    feeds every hour, which at the 0.7 s courtesy delay is about an hour, and
    because the scheduled task ignores new instances while one is running,
    every 5-minute live sweep in that hour was dropped. Live capture was blind
    every other hour. A match that has appeared in the completed index and been
    captured since is final, so it is remembered in DONE_FILE and skipped.
    `full=True` (the `backfill` CLI mode with --full) ignores that memory.
    """
    log("=== BACKFILL: completed matches not yet captured ===")
    capture_reference()

    if SLAMS[SLAM]["enumerate_via"] == "draws":
        ids = ids_from_draw_feeds()
        if not ids:
            log("!! no match ids from the draw feeds — nothing to capture")
            return
        capture_ids(sorted(ids), full=full)
        return

    days_blob = fetch(f"/{LOC}/scores/feeds/{YEAR}/completed_matches/eventDays.json")
    if not days_blob:
        log("!! no eventDays.json — cannot enumerate completed matches")
        return

    try:
        days = json.loads(days_blob)
    except json.JSONDecodeError:
        log("!! eventDays.json did not parse")
        return

    # eventDays.json hands us the URLs outright. Use them: the path number is
    # `tournDay` (which counts qualifying days too), NOT the displayed day —
    # day 15 lives at day_22.json. Deriving it ourselves would be wrong.
    day_urls: list[tuple[str, str]] = []
    for d in days.get("eventDays", []):
        url = d.get("url")
        if url:
            day_urls.append((str(d.get("displayDay") or d.get("tournDay") or "?"), url))

    if not day_urls:
        log("  !! no day urls in eventDays.json")
        return
    log(f"  {len(day_urls)} event days listed")

    all_ids: set[str] = set()
    for label, url in day_urls:
        blob = fetch(url)
        if not blob:
            continue
        write_if_changed(OUT / "reference" / "days", f"day_{label}", blob)
        ids = match_ids_from(blob)
        new = ids - all_ids
        all_ids |= ids
        if ids:
            log(f"  day {label}: {len(ids)} match ids ({len(new)} new)")

    done = set() if full else load_done()
    todo = sorted(all_ids - done, key=int)
    log(f"total distinct completed matches: {len(all_ids)}; "
        f"{len(done & all_ids)} already settled on disk; {len(todo)} to capture")
    captured = 0
    for i, mid in enumerate(todo, 1):
        fetched = capture_match(mid)
        # Only the point tape decides whether the match is settled, and only a
        # fetch that succeeded in THIS pass counts. A live sweep usually leaves
        # a partial history file behind before the match completes, so checking
        # the disk would mark a failed re-fetch as settled forever.
        if fetched.get("history"):
            done.add(mid)
            captured += 1
        if i % 25 == 0:
            log(f"  ...{i}/{len(todo)}")
            save_done(done)
    save_done(done)
    log(f"=== BACKFILL COMPLETE: {captured} of {len(todo)} captured, "
        f"{len(done)} settled in total ===")


def poll(hours: float) -> None:
    log(f"=== POLL: watching live scores for {hours}h, every {POLL_EVERY}s ===")
    deadline = time.time() + hours * 3600
    seen_final: set[str] = set()

    while time.time() < deadline:
        blob = fetch(f"/{LOC}/scores/feeds/{YEAR}/matches/live/scores.json")
        if blob:
            write_if_changed(OUT / "reference" / "live", "scores", blob)
            try:
                matches = json.loads(blob).get("matches", [])
            except json.JSONDecodeError:
                matches = []

            live = [m for m in matches if str(m.get("status", "")).lower() != "complete"]
            for m in matches:
                mid = str(m.get("match_id", ""))
                if not mid:
                    continue
                capture_match(mid, tag=f" [{m.get('status')}]")
                # one extra pull after a match ends, to catch the settled tape
                if str(m.get("status", "")).lower() == "complete" and mid not in seen_final:
                    seen_final.add(mid)
            if live:
                log(f"{len(live)} live: " + ", ".join(
                    f"{m.get('match_id')} {m.get('roundNameShort', '')}" for m in live[:6]))
            else:
                log("no live matches")
        time.sleep(POLL_EVERY)

    log("=== POLL WINDOW ENDED ===")


def once() -> None:
    """One sweep, then exit. This is the mode the scheduled task runs.

    Keeping each run short and stateless means the scheduler owns the cadence,
    a crash costs one interval rather than the whole capture, and reboots are a
    non-event. Content-addressed writes make repeated runs idempotent.
    """
    blob = fetch(f"/{LOC}/scores/feeds/{YEAR}/matches/live/scores.json")
    if not blob:
        log("no live feed response")
        return

    write_if_changed(OUT / "reference" / "live", "scores", blob)
    try:
        matches = json.loads(blob).get("matches", [])
    except json.JSONDecodeError:
        log("live feed did not parse")
        return

    if not matches:
        log("no matches on court")
        return

    log(f"{len(matches)} match(es) on the live feed")
    deadline = time.monotonic() + ONCE_BUDGET_S
    for i, m in enumerate(matches):
        if time.monotonic() > deadline:
            log(f"  budget exceeded, stopping after {i}/{len(matches)} matches "
                f"(the scheduler will pick up the rest next run)")
            break
        mid = str(m.get("match_id", ""))
        if mid:
            capture_match(mid, tag=f" [{m.get('status')}]")

    # Refresh the completed-match index roughly once an hour so finished matches
    # get their settled tape even if they ended between sweeps.
    if datetime.now(timezone.utc).minute < 6:
        log("hourly: refreshing completed-match index")
        backfill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backfill", "poll", "both", "once"])
    ap.add_argument("--slam", choices=sorted(SLAMS), default=SLAM,
                    help="which major. Only the two that actually serve a point "
                         "tape are listed; see the SLAMS comment for what each "
                         "host does and does not answer.")
    ap.add_argument("--year", type=int, default=YEAR,
                    help="tournament year. The slam feeds keep past editions on the "
                         "same paths - the US Open answers for 2022 onward - so this "
                         "backfills previous years with the identical capture. Every "
                         "feed is stored raw, whole, exactly as for the live year: "
                         "deciding at harvest time which fields matter is a decision "
                         "you cannot revisit without fetching it all again.")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--full", action="store_true",
                    help="backfill: re-capture every completed match, ignoring backfill_done.json")
    args = ap.parse_args()

    if args.slam != SLAM or args.year != YEAR:
        SLAM = args.slam
        BASE = SLAMS[SLAM]["host"]
        LOC = SLAMS[SLAM]["loc"]
        YEAR = args.year
        OUT = _common.RAW / SLAM / str(YEAR)
        DONE_FILE = OUT / "reference" / "backfill_done.json"

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"output: {OUT}  (year {YEAR})")
    try:
        if args.mode == "once":
            once()
        if args.mode in ("backfill", "both"):
            backfill(full=args.full)
        if args.mode in ("poll", "both"):
            poll(args.hours)
    except KeyboardInterrupt:
        log("interrupted — captured data is intact")
        sys.exit(0)
