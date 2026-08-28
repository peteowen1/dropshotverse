# dropshotdata — frozen base archive

Local, filesystem-only freeze of `Aneeshers/tennis-sackmann-archive`, fetched
from both its GitHub repo and its HuggingFace dataset mirror. Built for
ticket 13 (`dropshotverse/.scratch/dropshot-tennis-engine/issues/13-snapshot-base-archive.md`).

**No git repo was created here** (no `git init`), and nothing has been
published anywhere. This is a plain local copy pending Pete's go-ahead to
publish it as a GitHub Release (see the ticket's recommendation).

## Layout

```
dropshotdata/
  raw/github/           # 473 files fetched from raw.githubusercontent.com, commit 8373358...
  raw/huggingface/       # 473 files fetched from huggingface.co/datasets/.../resolve/main/
  manifest/
    manifest.json        # per-file: path, byte size, sha256, mirror agreement, row count
    manifest.csv          # same, flattened for spreadsheet viewing
    sanity_check_output.txt              # full sanity-check console output
    serve_stat_coverage_by_tour_level_decade.csv
    row_counts_by_tour_level.csv
    row_counts_by_tour_decade.csv
  scripts/
    fetch_archive.py       # downloads + checksums both mirrors
    sanity_check.R          # loads all singles match CSVs, produces the coverage tables
    add_row_counts.py       # fills in manifest.json row_count field
```

## Source

- GitHub: https://github.com/Aneeshers/tennis-sackmann-archive (commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`, pushed 2026-06-25)
- HuggingFace: https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive

See the ticket's `## Answer` section for the full verification, sanity-check,
and serve-stat coverage results.

## Not done here (needs Pete's explicit go-ahead)

Creating a `dropshotdata` GitHub repo and publishing this as a Release. When
that happens, follow `pannaverse/panna/R/piggyback.R` and
`bouncerverse/bouncerdata/scripts/upload_to_release.R` as the pattern, and
vendor `versebus.R` verbatim from `torpverse/torp/R/versebus.R`.
