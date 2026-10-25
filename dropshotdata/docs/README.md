# dropshotdata — data layout and how to get it

`dropshotdata/` is a plain directory inside the `dropshotverse` repo (not a
sub-repo). Data and models stay out of git and travel as GitHub Releases.

## Layout

```
dropshotdata/
  raw/github/          frozen Sackmann archive, 473 CSVs, snapshot 2026-05-25 (ends 2026-06-02)
  raw/odds/            tennis-data.co.uk Pinnacle closing odds, 2018-2026, ATP + WTA
  raw/usopen/          2026 US Open point-by-point capture + the two laptop task logs
  raw/wta_api/         cached api.wtatennis.com responses (fetch_wta_upcoming.py)
  manifest/            per-file sha256 / row counts for the archive freeze (ticket 13)
  reference/           committed inputs: usopen_{men,women}_entrants.json
  tune_scratch/        experiment screens AND the derived inputs the pipeline reads
  blog/                what gets published to R2 (live_predictions, pairwise, ratings, predictions)
  scripts/             the pipeline
```

## Releases

| Tag | Assets | Publisher |
|---|---|---|
| `sackmann-archive` | `sackmann_archive_raw.tar.gz` (144 MB, extracts to `raw/github/`), `sackmann_matches.parquet` (29 MB) | `scripts/publish_archive_release.py` |
| `model-inputs-2026-09` | `elo_raw_ratings_full.parquet`, `set_/game_elo_raw_ratings_full.parquet`, `diag_layoff.parquet`, `xgb_features_v10_holdout.parquet`, `xgb_bagged_model_cache_v2_setgame.zip`, `us_open_2026_predictions.json` | `scripts/publish_model_inputs_release.py` |
| `tennismylife-archive` | `tml_atp.csv`, `tml_wta.csv`, `tml_manifest.json` (the current season, overwritten daily), plus dated copies `*_YYYYMMDD.*` | `.github/workflows/archive-tennismylife.yml` |

Release tags are reused with `--clobber`, so check asset `updatedAt`, not the
release date, to know how fresh a file is.

## Fresh clone

```bash
gh release download sackmann-archive --pattern sackmann_archive_raw.tar.gz --dir dropshotdata/raw
tar -xzf dropshotdata/raw/sackmann_archive_raw.tar.gz -C dropshotdata/raw
gh release download model-inputs-2026-09 --dir dropshotdata/tune_scratch
(cd dropshotdata/tune_scratch && unzip xgb_bagged_model_cache_v2_setgame.zip)
```

That is exactly what `.github/workflows/live-projection.yml` does. Not
published: `prepared_matches_cache.pkl` (266 MB, rebuilt from the CSVs in
~20 s by `hierarchical_model.load_prepared()`).

## Source of the archive

`Aneeshers/tennis-sackmann-archive`, GitHub commit `8373358` (pushed
2026-06-25) and its HuggingFace mirror, both fetched and checksummed by
`scripts/fetch_archive.py` on 2026-08-27, the day Sackmann's own repos went
404. Licence CC BY-NC-SA. Verification detail: ticket 13.
