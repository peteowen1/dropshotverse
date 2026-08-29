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
from pathlib import Path

BLOG_DIR = Path(r"C:\dev\dropshotverse\dropshotdata\blog")
BUCKET_PREFIX = "inthegame-data/tennis"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    wrangler = shutil.which("wrangler")   # resolves the .cmd wrapper's real path on Windows -
                                          # avoids shell=True (and the injection risk that comes with it)
                                          # while still working for an npm global-install binary
    if not wrangler:
        raise SystemExit("wrangler not found on PATH")

    files = sorted(BLOG_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {BLOG_DIR} - run build_blog_data.py first")

    for f in files:
        key = f"{BUCKET_PREFIX}/{f.name}"
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

    log(f"\nuploaded {len(files)} file(s) to R2 ({BUCKET_PREFIX}/)")


if __name__ == "__main__":
    main()
