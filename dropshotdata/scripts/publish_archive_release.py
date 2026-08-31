"""Publish the frozen Sackmann archive as a GitHub Release.

Why
---
Sackmann's `tennis_atp`/`tennis_wta` repos went 404 on 2026-08-27. The copy in
`dropshotdata/raw/github` is currently the ONLY one, on one machine, gitignored.
That is a single point of failure for the entire project's foundation, and it is
also what blocks CI (a fresh Actions runner has no archive) - see issue #4 and
ticket 13, whose "publish this as a Release" recommendation was never actioned.

What gets published
-------------------
1. `sackmann_matches.parquet` - the exact match rows `load_matches()` reads
   (the atp/wta `*_matches_*.csv` families), concatenated. This is what CI needs
   and what a consumer almost always wants.
2. `sackmann_archive_raw.tar.gz` - the full `raw/github` tree, for provenance:
   the manifest'd, sha256-verified snapshot exactly as fetched.

`raw/huggingface` is deliberately NOT published. Ticket 13 verified all 465 CSV
data files are byte-for-byte identical across the two mirrors, so it is the same
bytes twice with no extra provenance value. `raw/usopen` is also excluded - it
is live capture output, regenerable, and not part of the frozen archive.

Size context (checked 2026-08-31): pannadata's releases total ~10.8 GB with a
single 645 MB parquet; bouncerdata 1.7 GB; torpdata 1.1 GB. These assets sit
comfortably inside the established pattern.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
ARCHIVE = DD / "raw" / "github"
STAGE = DD / "tune_scratch" / "_release_stage"
REPO = "peteowen1/dropshotverse"
TAG = "sackmann-archive"
TITLE = "Frozen Sackmann tennis archive (snapshot 2026-05-25)"


def log(m: str) -> None:
    print(m, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_matches_parquet(out: Path) -> tuple[int, int]:
    """Concatenate exactly the CSV families `load_matches()` globs.

    Kept deliberately close to that function rather than 'all CSVs': publishing
    a superset would invite a consumer to load rows the model never trains on.
    """
    frames, n_files = [], 0
    for tour in ("atp", "wta"):
        pats = [f"{tour}_matches_[0-9]*.csv",
                f"{tour}_matches_qual_chall_[0-9]*.csv",
                f"{tour}_matches_qual_itf_[0-9]*.csv"]
        for pat in pats:
            for p in sorted((ARCHIVE / tour).glob(pat)):
                try:
                    df = pd.read_csv(p, low_memory=False)
                except Exception as e:                       # noqa: BLE001
                    log(f"  SKIP unreadable {p.name}: {e}")
                    continue
                if not len(df):
                    continue
                df["_tour"] = tour
                df["_source_file"] = p.name
                frames.append(df)
                n_files += 1
    if not frames:
        raise SystemExit("no match CSVs found - is ARCHIVE path right?")
    all_df = pd.concat(frames, ignore_index=True)
    # every column to a stable dtype: mixed-type object columns (the '-35.9%'
    # class of defect the tennismylife bridge already hit) would otherwise
    # fail the parquet write or silently coerce
    for c in all_df.columns:
        if all_df[c].dtype == object:
            all_df[c] = all_df[c].astype("string")
    all_df.to_parquet(out, index=False, compression="zstd")
    return n_files, len(all_df)


def build_raw_tarball(out: Path) -> int:
    n = 0
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        for p in sorted(ARCHIVE.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(ARCHIVE.parent)))
                n += 1
    return n


def main() -> None:
    gh = shutil.which("gh")
    if not gh:
        raise SystemExit("gh not found on PATH")
    STAGE.mkdir(parents=True, exist_ok=True)

    pq = STAGE / "sackmann_matches.parquet"
    log("building sackmann_matches.parquet ...")
    n_files, n_rows = build_matches_parquet(pq)
    log(f"  {n_files} CSVs -> {n_rows:,} rows, {pq.stat().st_size/1048576:.1f} MB")

    # anchor: the archive is documented at 1,774,873 singles matches across all
    # levels (ticket 13). This subset is the model-relevant families only, so it
    # must be LARGE but not larger than that total - a wild number here means
    # the glob changed underneath us.
    if not (500_000 < n_rows <= 1_774_873):
        raise SystemExit(f"row count {n_rows:,} outside sane range - refusing to publish")

    tgz = STAGE / "sackmann_archive_raw.tar.gz"
    log("building sackmann_archive_raw.tar.gz (full raw/github tree) ...")
    n_tar = build_raw_tarball(tgz)
    log(f"  {n_tar} files -> {tgz.stat().st_size/1048576:.1f} MB")

    notes = f"""Frozen snapshot of the Sackmann tennis archive, as fetched 2026-08-27 from
`Aneeshers/tennis-sackmann-archive` (commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`).

**Why this exists:** Sackmann's own `tennis_atp` / `tennis_wta` repos returned 404 from
2026-08-27. This snapshot was the only surviving copy on one machine; publishing it
removes that single point of failure and lets CI run without a 1.8 GB local tree.

## Assets

| file | contents |
|---|---|
| `sackmann_matches.parquet` | {n_rows:,} rows from {n_files} CSVs - the atp/wta `*_matches_*`, `*_qual_chall_*` and `*_qual_itf_*` families, i.e. exactly what `hierarchical_model.load_matches()` reads. Adds `_tour` and `_source_file`. zstd. |
| `sackmann_archive_raw.tar.gz` | full `raw/github` tree, {n_tar} files, for provenance. |

## Not included, deliberately

- `raw/huggingface` - ticket 13 verified all 465 CSV data files are **byte-for-byte
  identical** to the GitHub mirror. Same bytes, no extra provenance value.
- `raw/usopen` - live capture output, regenerable, not part of the frozen archive.

## Caveats

- Data ends **2026-05-25** (ATP match dates run to 2026-06-01). Anything after that
  comes from the tennismylife bridge (`scripts/bridge_tennismylife.py`), which is
  **ATP-only**.
- ATP and WTA player ids collide on 2,396 values (id `200003` is Jeremy Beale in ATP
  and Justine Henin in WTA). Namespace as `{{tour}}:{{id}}` at ingest.
- Licence: the upstream Sackmann data is CC BY-NC-SA. Non-commercial use with
  attribution.
"""
    (STAGE / "notes.md").write_text(notes, encoding="utf-8")

    log("\nchecksums:")
    for f in (pq, tgz):
        log(f"  {f.name}  {sha256(f)}")

    log(f"\ncreating release {TAG} on {REPO} ...")
    r = subprocess.run([gh, "release", "create", TAG, str(pq), str(tgz),
                        "--repo", REPO, "--title", TITLE,
                        "--notes-file", str(STAGE / "notes.md")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log(r.stdout)
        log(r.stderr)
        raise SystemExit("release creation failed")
    log(r.stdout.strip())
    log("done")


if __name__ == "__main__":
    sys.exit(main())
