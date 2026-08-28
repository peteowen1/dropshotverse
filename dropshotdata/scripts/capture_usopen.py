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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://www.usopen.org"
YEAR = 2026
OUT = Path(r"C:\dev\dropshotverse\dropshotdata\raw\usopen") / str(YEAR)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

# per-match feeds worth keeping. `history` is the point tape and the whole point
# of this exercise; the rest are cheap and contextual.
MATCH_FEEDS = {
    "history":  "/en_US/scores/feeds/{y}/slamtracker/history/{mid}C.json",
    "complete": "/en_US/scores/feeds/{y}/matches/complete/{mid}.json",
    "stats":    "/en_US/scores/feeds/{y}/completed_matches/match_stats_{mid}.json",
    "distance": "/en_US/scores/feeds/{y}/slamtracker/distance/{mid}.dist.json",
    "insight":  "/en_US/scores/feeds/{y}/match_insight/{mid}.json",
}

DELAY = 0.7          # seconds between requests — we are a guest here
POLL_EVERY = 45      # seconds between live-score checks


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
            with urlopen(req, timeout=30) as r:
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
        except (URLError, TimeoutError) as e:
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


def capture_match(mid: str, tag: str = "") -> int:
    written = 0
    for name, tmpl in MATCH_FEEDS.items():
        blob = fetch(tmpl.format(y=YEAR, mid=mid))
        if blob and write_if_changed(OUT / "matches" / str(mid), name, blob):
            written += 1
    if written:
        log(f"  match {mid}{tag}: {written} feed(s) written")
    return written


def capture_reference() -> None:
    """Tournament-level feeds: the draw, the player list, the feed catalogue itself."""
    refs = {
        "config_web":   "/en_US/json/gen/config_web.json",
        "draws":        f"/en_US/scores/feeds/{YEAR}/draws/draws.json",
        "players":      f"/en_US/scores/feeds/{YEAR}/players/players.json",
        "competitors":  f"/en_US/scores/feeds/{YEAR}/players/competitors.json",
        "seeds":        f"/en_US/scores/feeds/{YEAR}/players/seeds_results_all.json",
        "event_days":   f"/en_US/scores/feeds/{YEAR}/completed_matches/eventDays.json",
        "schedule_days": f"/en_US/scores/feeds/{YEAR}/schedule/scheduleDays.json",
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


def backfill() -> None:
    log("=== BACKFILL: every match played so far ===")
    capture_reference()

    days_blob = fetch(f"/en_US/scores/feeds/{YEAR}/completed_matches/eventDays.json")
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

    log(f"total distinct matches to capture: {len(all_ids)}")
    done = 0
    for i, mid in enumerate(sorted(all_ids, key=int), 1):
        capture_match(mid)
        done += 1
        if i % 25 == 0:
            log(f"  ...{i}/{len(all_ids)}")
    log(f"=== BACKFILL COMPLETE: {done} matches ===")


def poll(hours: float) -> None:
    log(f"=== POLL: watching live scores for {hours}h, every {POLL_EVERY}s ===")
    deadline = time.time() + hours * 3600
    seen_final: set[str] = set()

    while time.time() < deadline:
        blob = fetch(f"/en_US/scores/feeds/{YEAR}/matches/live/scores.json")
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
    blob = fetch(f"/en_US/scores/feeds/{YEAR}/matches/live/scores.json")
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
    for m in matches:
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
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"output: {OUT}")
    try:
        if args.mode == "once":
            once()
        if args.mode in ("backfill", "both"):
            backfill()
        if args.mode in ("poll", "both"):
            poll(args.hours)
    except KeyboardInterrupt:
        log("interrupted — captured data is intact")
        sys.exit(0)
