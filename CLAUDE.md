# CLAUDE.md — dropshotverse

Tennis: pre-match win probability, draw simulation, live in-match win
probability, ATP and WTA. Read `docs/HOME.md` first; `docs/NEXT-STEPS.md` is
the queue and `docs/DECISIONS.md` the log. Update both at the end of any
session that changes state.

## Shape of the repo

- **One git repo.** `dropshotdata/` is a plain directory, not a sub-repo.
  Commit from the root; `dev` for work, `main` via reviewed PR.
- Python 3.13 at `C:\Python313\python.exe`. No R in the pipeline.
- Every script hard-codes `C:\dev\dropshotverse\dropshotdata\...`. Running
  from anywhere else needs that path to exist (the Actions runner uses a
  junction). Making paths relative is queued.
- Data and models are not in git. Two GitHub Releases carry them:
  `sackmann-archive` (raw CSVs) and `model-inputs-2026-09` (Elo replays and
  the shipped model). `dropshotdata/docs/README.md` says how to fetch them.

## What runs where

| What | Where | Cadence |
|---|---|---|
| Live US Open projection to R2 (second producer; GitHub's cron fires sporadically) | `.github/workflows/live-projection.yml` | cron `*/30`, delivered ~hourly at best |
| Freshness alarm on the public R2 copy | `.github/workflows/check-freshness.yml` | hourly |
| Same projection from the laptop (primary through the 2026-09-13 final) | task `DropshotverseUSOpenLiveSim` runs `dropshotdata/scripts/refresh_usopen_live.cmd` | every 30 min to 2026-09-16 |
| Point-by-point capture | task `DropshotverseUSOpenCapture` runs `dropshotdata/scripts/capture_usopen.py once` | every 5 min to 2026-09-16 |

Logs for the laptop tasks: `dropshotdata/raw/usopen/*.log`. Times in the
capture log are UTC.

## Key scripts (`dropshotdata/scripts/`)

- `simulate_usopen_live.py` — re-simulate from the live bracket; refuses to
  write if an anchor fails, so the site keeps the last good projection.
- `train_best_model.py` — the shipped XGBoost, end to end. `xgb_bt_rating.py`
  caches the 5-seed bag under `tune_scratch/xgb_bagged_model_cache_v2_setgame/`.
- `hierarchical_model.py` — the point model. Not shipped standalone (issue #1).
- `fetch_wta_upcoming.py` — calendar and entry lists for upcoming WTA events.
- `build_point_table.py` — the US Open point table, with coverage gates.
- `check_blog_data_freshness.py` — `--url` checks the published copy.
- `publish_archive_release.py`, `publish_model_inputs_release.py` — the Releases.

## Rules that have bitten here

- A check that prints FAIL and continues is not a check. Anchors gate the write.
- Assert coverage per column after every union; the point table once filled
  missing columns with `""` silently.
- The 2024-25 holdout is burnt. Screen on the 2022-23 tune window; 2026 was
  the one clean forward test and has been spent (D-10).
- Research scripts go in `dropshotdata/tune_scratch/`, their findings in a
  ticket under `.scratch/dropshot-tennis-engine/issues/`, and a two-line row
  in DECISIONS. Not essays in the hubs.
