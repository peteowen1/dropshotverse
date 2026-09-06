"""Local R2 push, mirroring citiusdata/scripts/export_athletics_blog.R's
pattern (shells out to wrangler using whatever Cloudflare account it's
already authenticated to on this machine - CLOUDFLARE_API_TOKEN env var,
confirmed present) rather than torpdata's GitHub-Actions-triggered path.
No GitHub remote or repo secrets needed for this route.

Run build_blog_data.py first to (re)generate dropshotdata/blog/*.parquet -
this script only uploads whatever's already there.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import _common
BLOG_DIR = _common.BLOG
BUCKET_PREFIX = "inthegame-data/tennis"
# Public read side of the same bucket (check_blog_data_freshness.py uses it too).
PUBLIC_URL = "https://pub-ee4bf5b599a047f9ac2b9facc1587008.r2.dev/tennis/"


RUN_START = datetime.now(timezone.utc)
BUILT_THIS_RUN_MIN = 30   # generous: the sim takes ~1 min, the .cmd runs this straight after


def log(m: str) -> None:
    print(m, flush=True)


def r2_last_modified(name: str) -> datetime | None:
    """Last-Modified of the copy already on R2, or None if there is none.

    FABLE-redteam-2026-09-07 F1: the laptop task ran this script every 30 min
    and re-uploaded every parquet in blog/, so the 2026-08-29 300-row
    ratings.parquet clobbered the daily ratings-refresh workflow's 347-row
    table within half an hour of each run. Any file whose local copy is OLDER
    than the R2 copy is skipped. Cache-Control: no-cache so the CDN cannot hand
    back a stale header. On a non-404 failure return None so the upload proceeds
    as before (the PUT will fail on its own if the network is really down).
    """
    req = urllib.request.Request(PUBLIC_URL + name, method="HEAD",
                                 headers={"Cache-Control": "no-cache", "Pragma": "no-cache",
                                          # r2.dev returns 403 to the default Python-urllib agent
                                          "User-Agent": "dropshotdata-upload-blog-data"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            lm = r.headers.get("Last-Modified")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log(f"  WARNING: HEAD {name} failed ({e.code}); cannot compare ages, uploading anyway")
        return None
    except Exception as e:  # noqa: BLE001 - network hiccup, keep the old behaviour
        log(f"  WARNING: HEAD {name} failed ({e}); cannot compare ages, uploading anyway")
        return None
    return parsedate_to_datetime(lm).astimezone(timezone.utc) if lm else None


def main() -> None:
    wrangler = shutil.which("wrangler")   # resolves the .cmd wrapper's real path on Windows -
                                          # avoids shell=True (and the injection risk that comes with it)
                                          # while still working for an npm global-install binary
    if not wrangler:
        raise SystemExit("wrangler not found on PATH")

    files = sorted(BLOG_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {BLOG_DIR} - run build_blog_data.py first")

    uploaded = 0
    for f in files:
        key = f"{BUCKET_PREFIX}/{f.name}"
        # FABLE-redteam-2026-09-07 F1: never overwrite a newer copy on R2.
        local_mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        # A file this run just built is ALWAYS uploaded, without consulting R2.
        # The skip exists for files this run did not produce (the ratings and
        # locked-prediction tables); applying it to the live tables too made the
        # live path depend on the laptop clock agreeing with R2's, and a clock
        # that drifts behind (NTP lapse after sleep is the usual way) would have
        # made every freshly built file look stale and silently frozen the live
        # odds mid-event. RUN_START comes from the same clock as st_mtime, so
        # skew cancels. Review of 05f9232, 2026-09-07.
        if local_mtime >= RUN_START - timedelta(minutes=BUILT_THIS_RUN_MIN):
            log(f"uploading {key} ({f.stat().st_size:,} bytes, built by this run)...")
        elif (remote_lm := r2_last_modified(f.name)) is not None and local_mtime < remote_lm:
            # NB the R2 timestamp is when the object was last WRITTEN, which
            # may be a re-upload of stale content, not when that content was
            # computed. Same trap as the GitHub release date gotcha in
            # C:/dev/CLAUDE.md: this line reads as reassurance either way.
            log(f"skipping {key}: local copy {local_mtime:%Y-%m-%d %H:%M:%S}Z is older than "
                f"R2 copy {remote_lm:%Y-%m-%d %H:%M:%S}Z (not built by this run)")
            continue
        else:
            log(f"uploading {key} ({f.stat().st_size:,} bytes)...")
        # encoding/errors explicit: wrangler's own banner has a unicode emoji
        # that crashes the default cp1252 decode on Windows otherwise
        # (confirmed - upload itself succeeds, only output-capture crashed).
        r = subprocess.run([wrangler, "r2", "object", "put", key, "--file", str(f),
                           "--cache-control", "public, max-age=300", "--remote"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(r.stdout)
            log(r.stderr)
            raise SystemExit(f"upload failed for {key}")
        log(f"  OK")
        uploaded += 1

    log(f"\nuploaded {uploaded} of {len(files)} file(s) to R2 ({BUCKET_PREFIX}/)")


if __name__ == "__main__":
    main()
