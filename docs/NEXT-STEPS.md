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
2. **Between events: a ratings-led page** (option A). dropshotverse side DONE
   2026-09-07: `ratings-refresh.yml` daily 06:00 UTC (3 min 50 s), hard
   anchors, `ratings_as_of` column, warn-level alarm; first real run
   published. Blog side: date label fixed on inthegame-blog dev; the
   finished-event hero is **inthegame-blog#660** (ITG session). Then option B,
   a Beijing forecast (field seeds ~2026-09-23).
   **Option B built 2026-09-07**: `simulate_wta_event.py` + `event-forecast.yml`
   publish any WTA event's pre-draw forecast to `upcoming_predictions.parquet`;
   the blog page for it goes with #660.
   Plan: [plans/between-events-ratings.md](plans/between-events-ratings.md).
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

- **2026-09-13, after the final** — rebuild `usopen_2026_points.parquet`
  (`python build_point_table.py`) and re-run `screen_inmatch_seeded.py` then
  `screen_inmatch_recalibrated.py`. That is the late-round validation arm D
  needs before it ships, and it cannot be done earlier because the matches
  have not been played.
- **2026-09-13** — US Open men's final. `live-projection.yml` has no expiry.
  The laptop tasks run to **2026-09-16 12:00 local** (2026-09-15 22:00 New
  York); the point capture still depends on the laptop.
- **~2026-09-23** — dispatch `event-forecast.yml` with group `1020` for the
  China Open (Beijing, starts 2026-09-30) once the field shows 96 entrants
  and 32 seeds. On 09-07 it was 76 of 96 and 0 seeded; the pipeline is built
  and smoke-tested, and refuses to publish a partial field without
  `allow_partial`.
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

## Red-team and methodology reviews (2026-09-07)

Another Fable session left two reviews in [reviews/](reviews/) and fixed the
two live defects itself (commit `05f9232`): the laptop uploader was clobbering
the daily ratings table every 30 minutes (F1), and the name crosswalk missed
two real players who were then published as unrated 1500 debutants, one of
them given 0.9% in a match he won (F2). Verified independently here.

Still open from those reviews, in the order they are worth doing:

1. ~~**F4 — head-to-head trained on all levels, served tour-level only.**~~
   **SIZED AND FIXED 2026-09-07.** The mismatch was real and confirmed in the
   shipped trainer (`train_best_model.py:153` counts from the unfiltered
   parquet, 70.3% of whose rows are non-tour). Its effect on what gets
   published is not: on the real US Open field, 18-19% of ordered pairs get a
   different `h2h_diff` (up to 5 meetings), and the probability moves a mean
   of **0.0017**, a p99 of 0.022 and a max of 0.041, with no player's mean win
   probability moving more than **0.002**. Serve now matches training; treat
   it as a correctness fix and do not expect it in any metric.
   Measurement: `tune_scratch/screen_h2h_train_serve_skew.py`.
2. ~~**F5 — published players carrying the default age 24.0.**~~ **FIXED
   2026-09-07, and the count was wrong.** Only **8** of the 12 rows sitting at
   exactly 24.0 were the default; the other 4 (Cazaux, Samuel, Parry,
   Fernandez) were real ages of 23.7 to 23.96 carried forward and rounded to
   one decimal. The published value cannot tell them apart, so the check is now
   the absence of an age row, not the value. `age` publishes as null when the
   archive has no row; the model still receives 24.0, because training never
   saw a missing age and serving NaN would be the same train/serve mismatch as
   F4. The age table is still frozen-archive only (max 2026-05-25); carrying
   age from the bridge is the fuller fix and is not done.
3. ~~**W2 — no repeatable out-of-time measurement left.**~~ **BUILT
   2026-09-07.** `docs/plans/validation-charter.md` (locked before running) and
   `tune_scratch/walk_forward_yearly.py`: expanding walk-forward, train
   1995..Y-1, score Y, for Y in 2019-2025, on market-joined rows, with the
   bookmaker average, Elo and a constant as baselines. A candidate ships only
   on 5 of 7 years plus a mean gain of 0.002. **First result: the shipped
   config is 0.0008 behind the bookmaker average on the mean and behind in 5
   of 7 years, while beating Elo alone by 0.026 every year.** Caveat that
   matters: 2022-2025 were used to select this config, so only 2019-2021 are
   clean for it. Run a candidate with
   `python walk_forward_yearly.py --label X --drop feature`.
4. ~~**W1 — condition on the bookmaker price.**~~ **MEASURED AND REJECTED
   2026-09-07 (D-15).** Pete: no bookmaker odds in the model at all. Prices
   stay a benchmark only. Do not reopen on size; size was not the objection.
   The measurement is kept and is worth knowing: blending
   the bookmaker price in beats the model in 6 of 6 years AND beats the market
   in 6 of 6, by 0.0049 and 0.0040, at a fitted weight of 0.52. It passes the
   charter's rule and fails the critique's own W1 rule, whose 0.015 bar was
   calibrated on a market gap of 0.0247 that the scorecard measured at 0.0008
   - the bar was unreachable because its premise was wrong. It also cannot
   serve the draw simulation: at most 0.79% of the 8,128 pairs can carry a
   price. So it is a real, small, single-match-only gain that would make
   roughly half of every published number the bookmaker's opinion. That is an
   editorial call. Detail and the table:
   [plans/validation-charter.md](plans/validation-charter.md).
   Script: `tune_scratch/screen_market_blend.py`.
5. **W4 MEASURED 2026-09-07: NO-GO on the pre-registered bar, and the reason
   is worth knowing.** Seeding the in-match recursion with the model's own
   pre-match skill (arm C) cuts per-point log-loss from 0.572 to 0.532 and
   lifts accuracy from 0.720 to 0.779 over 40,260 points, but ECE worsens from
   0.034 to 0.051 against a 0.03 bar. It is sharper and overconfident. The
   The recalibrated follow-up (arm D) was then run as its own candidate:
   **ECE 0.0346, log-loss 0.49358, accuracy 0.779** - the same calibration as
   the incumbent arm A, 0.078 better log-loss, 5.9 more accuracy points. Still
   NO-GO on the letter of the 0.03 bar, which is itself below arm A's own
   0.0342. **But arm A was never live.** What the site does with an in-progress
   match is publish the pre-match number and ignore the score
   (`simulate_usopen_live.py:88-91`), which scores ECE **0.1058**, log-loss
   0.60332. So against the real incumbent arm D is three times better
   calibrated and 0.110 better on log-loss. **Round 4 then arrived** (the point
   table was a stale snapshot; rebuilding it added 8 finished R16 matches) and
   split the answer. Conditioning on the live score is settled - both seeded
   arms beat today's published number in every round. Whether to RECALIBRATE
   is not: on round 4 the plain seeded arm C beats the recalibrated arm D on
   both calibration (0.128 vs 0.175) and log-loss (0.224 vs 0.272), on 7
   matches. **Ship the seeding, not the recalibration**, and settle C vs D
   after the final with the same rebuild-and-re-score.

   **Built on `feat/inmatch-seeding`** (2026-09-07), behind `--live-score`,
   OFF by default because the scheduled task runs this working tree. The solve
   lives once in `dropshotdata/scripts/inmatch_seeding.py`; the screen imports
   the same code and was re-run through it to confirm the measurement holds.
   Switch-on checklist, all three needed:
   1. the 2026-09-13 final, then rebuild the point table and re-score C vs D;
   2. decide whether `live_predictions.parquet` should say which model made it
      (the `source` column is hard coded to `"live"` today, so a seeded and an
      unseeded projection are indistinguishable to the site);
   3. one live run with the flag on while a match is actually in progress -
      nothing was under way on 2026-09-07, so that path has been tested on
      synthetic states only.

   Review of the branch also found and fixed a real defect in the recursion
   underneath all of this (6-6 sets, error up to 0.42, see D-16). Every arm on
   the charter was re-measured; the conclusion did not change. `tune_scratch/screen_inmatch_seeded.py`,
   `screen_inmatch_recalibrated.py`;
   [plans/validation-charter.md](plans/validation-charter.md).
6. **W3 remains** (the live simulator freezes layoff and fatigue at the draw
   date for every round) and has its own stage-reaching scorecard in
   `simulate_tournament_xgb.py`. W5 (new players enter at 1500 with no prior)
   is the last unexamined item.

## Known and NOT chased

- **Config D forward test** — would turn the +0.0247 upper bound into the
  shipped model's exact gap. Skipped: three measurements agree, and a model
  right 38% of the time when it dissents has no edge worth sizing.
