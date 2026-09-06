# dropshotverse — Next Steps

> Living queue. Sessions update this when they finish work (rule in `C:\dev\CLAUDE.md`).
> Decision log: [DECISIONS.md](DECISIONS.md) · Vault map: [HOME.md](HOME.md) ·
> Rolled-off narrative: [backlog/](backlog/)
> Last updated: **2026-09-06 evening.**
>
> **Read first if you are cold:**
> - The market question is CLOSED (D-5, D-6, D-10). Fourteen rounds, a clean
>   2026 forward test, no exploitable edge. Do not reopen it without new data —
>   [issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md).
> - The live US Open projection publishes from the laptop task; GitHub
>   Actions is the armed backup, dispatched by the ITG Worker only when R2
>   goes stale (D-13, D-14). Nothing is waiting on Pete for it.
> - `main` is real: PRs #8-#12 merged 2026-09-06, each reviewed.

## Awaiting Pete

1. ~~**Mint the GitHub token for the ITG Worker**~~ — **DONE 2026-09-06 13:41
   UTC** (the gh CLI token, Pete's call). The backup producer is armed: the
   Worker dispatches `live-projection.yml` when R2 is >40 min stale, until
   `TENNIS_LIVE_UNTIL` (2026-09-16T02:00Z). Verified by an API dispatch: green
   in 2 min 43 s. #4 closed. Laptop stays primary through the final by choice;
   per event, bump `TENNIS_LIVE_UNTIL` and `INPUTS_TAG` if the vintage changed.
2. **Do not publish a "we match the market" claim** (D-4, a recommendation):
   the 0.00073 gap is from a burnt holdout, and the 2026 forward test puts the
   honest gap at 0.0247 with 38% accuracy when dissenting.

## Open queue

1. **[#5](https://github.com/peteowen1/dropshotverse/issues/5)** — publish
   match-pairing/fixture data so the blog can build a real bracket. No blog
   consumer exists yet.
2. **Between events: a ratings-led page that is actually current** (option A,
   chosen 2026-09-06). Plan: [plans/between-events-ratings.md](plans/between-events-ratings.md).
   Found while scoping: `ratings.parquet` was built once on 2026-08-29 and the
   site labels it "Updated today" because the laptop re-uploads it every 30
   min; the index page does not read `event_status`, so after the final the
   bracket renders as live. Order: fix the date label on the blog, then a
   daily `ratings-refresh.yml` (profile the Elo regen first), then the
   finished-event hero. Option B (a Beijing forecast, field seeds ~09-23)
   follows.
3. ~~**Live sim: five measured fixes**~~ — **DONE 2026-09-06**: 270 s to 68 s
   steady state (127 s on a cold snapshot cache), every function verified
   identical on the live inputs and the seeded end-to-end output compared.
   Left on the table: the prepare-cache bypass (~20 s) and `hm.p_match` in
   the pairwise loop. [issues/40](../.scratch/dropshot-tennis-engine/issues/40-live-sim-profile-and-redundancy.md).
4. ~~**Cull `scripts/`**~~ — **DONE 2026-09-06**: 32 superseded scripts to
   `scripts/archive/` (77 to 45 files), 26 unreferenced artefacts deleted
   (55 MB). `_common.py` deferred to the paths-relative change. Ticket 40.
5. ~~**Make the scripts' paths relative**~~ — **DONE for the live path
   2026-09-06**: `scripts/_common.py` carries `DD`/`SCRATCH`/`BLOG`/`RAW`
   resolved from the file's location; 14 live-path modules use it; the
   workflow runs on `ubuntu-latest` with no junction. Still hard-coded: ~20
   research scripts in `scripts/` and most of `tune_scratch/`. Convert them
   when touched, not as a project.
6. **The surface rating is not a self-correcting rating.** `elo_baseline.py:409`
   updates it with the blended `p_w`, which at `blend=0.0` is entirely overall
   Elo, so it accumulates residuals bucketed by surface with no feedback loop
   (std 498 vs 180). `elo_surface_diff` **is** an XGBoost feature. Whether it
   should use `p_s` is untested.
7. **[#1](https://github.com/peteowen1/dropshotverse/issues/1) — hier tail
   bug, downgraded (D-12).** The shipped model does not inherit it and needs
   the feature (+0.0092 without). Only worth picking up as "would a fixed hier
   make a better feature": an upside experiment.
8. **[#2](https://github.com/peteowen1/dropshotverse/issues/2)** — the WTA
   public live-scores API. Its in-play fields have never been observed
   populating.
9. ~~**`DropshotverseUSOpenCapture` exits 1 intermittently.**~~ — **FIXED
   2026-09-06**: a `ConnectionResetError` mid-read escaped `fetch()`'s retry
   (it is an `OSError`, not a `URLError`) and killed the sweep. Now retried.

## Live deadlines (absolute dates)

- **2026-09-13** — US Open men's final. `live-projection.yml` has no expiry.
  The laptop tasks run to **2026-09-16 12:00 local** (2026-09-15 22:00 New
  York); the point capture still depends on the laptop.
- **~2026-09-23** — re-run `fetch_wta_upcoming.py --fetch 1020` for the China
  Open (Beijing, starts 2026-09-30). On 09-06 the field was 76 of 96 and
  0 seeded; a forecast now would be over a partial, unseeded field.
- **After the final** — `event_status = complete` lets the site render a
  result; the freshness alarm downgrades stale-while-complete to a note.

## Recently landed (2026-09-06)

- **Evening batch (PRs #15-#18):** live sim 276 s to 68 s with outputs
  identical (ticket 40); `scripts/` culled 77 to 45 files and 55 MB of dead
  artefacts removed; the live path resolves its own paths via `_common.py`
  and the Actions runner moved to Linux (2 min 45 s a run); the capture
  task's exit 1 traced to an uncaught connection reset and fixed.
- **Live projection can run from GitHub Actions** (D-13): `live-projection.yml`
  from the `sackmann-archive` and `model-inputs-2026-09` Releases, proven end
  to end. Its cron was removed the same day (D-14): GitHub fired it once in
  two hours and a Windows runner every 30 min would exhaust the free minutes
  in four days. `check-freshness.yml` stays hourly (ubuntu, seconds per run).
- **Five PRs merged** (#8-#12). Review findings fixed before merge: the live
  sim and the point-table build printed anchor FAILs and published anyway;
  both now refuse the write. Per-column coverage gate on the point table.
- **Point capture was blind every other hour.** The hourly backfill took ~60
  min and the task ignores new instances, so live sweeps were dropped. Now
  incremental via `reference/backfill_done.json`, ~50 s a pass.
- **Issue #1 answered** (D-12): the hier tail bug does not reach the shipped
  model; dropping the feature costs +0.0092 on 5/5 folds.
- **Issue #3 decided and closed** (D-11): keep Sackmann as base, TML as bridge.
- **The market question closed**: the 2026 forward test (2,654 untouched
  matches) put the model at 0.59638 vs the market's 0.57168, 38.0% right when
  dissenting (CI [32.3, 43.8]). Round 14's K change was the cautionary tale:
  5/5 folds in tune, +0.000029 out of sample.
- **WTA upcoming forecasting works**; **in-match WP has its first calibrated
  result** (ticket 37, Mode A, ECE 0.0471); **`usopen_2026_points.parquet`**
  (80,690 points, all 27 columns populated).

Earlier detail: [backlog/journal-2026-09-05.md](backlog/journal-2026-09-05.md).

## Known and NOT chased

- **Config D forward test** — would turn the +0.0247 upper bound into the
  shipped model's exact gap. Skipped: three measurements agree, and a model
  right 38% of the time when it dissents has no edge worth sizing.
