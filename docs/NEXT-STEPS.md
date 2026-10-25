# dropshotverse — Next Steps

> Living queue. Sessions update this when they finish work (rule in `C:\dev\CLAUDE.md`).
> Decision log: [DECISIONS.md](DECISIONS.md) · Vault map: [HOME.md](HOME.md) ·
> Rolled-off narrative: [backlog/](backlog/)
> Last updated: **2026-10-25.**
>
> **2022-23 backfill FIXED (D-55).** All 5 years (2022-2026) are now usable -
> the shrink-factor re-estimate (queue item 3) is unblocked.
>
> **NEW DIRECTION (2026-09-15 evening): forecast every upcoming ATP/WTA
> event, published on ITG, emulating the US Open sim page** (pre-match
> bracket forecasts only, NOT the live in-match work - that's Slam-only
> infrastructure and out of scope). **Live now**: Beijing WTA is published
> and rendering at inthegame.blog/tennis/upcoming (provisional - field still
> 76/96, 0 seeds). Two real production bugs found and fixed live during
> dispatch (a NaN-uid crash, and qualifying-tournament entrants inflating
> the main-draw count past its slot limit). A generic multi-event ITG page
> shipped (`inthegame-blog` PR #694) - a new event needs no new page, just
> another `tournament_id` in the data files this reads.
>
> **China Open (Beijing, started 2026-09-30) is still the last published forecast
> as far as this repo knows.** Last known state is 2026-09-15 (published). The
> worker did not read R2, so this is not checked since. Issue #44's prune
> (a46d4ac, dd38685) drops it on the next write, and nothing writes until then.
> Clearing it is Awaiting Pete item 7.
>
> **ATP is still blocked on a per-tournament field/entry list** - the
> calendar (61 events, full season) and a validated deep ranking source
> (150 ranks, `tennisexplorer.com`) are both solved, but nothing found
> gives who actually entered a SPECIFIC tournament (Wikipedia only
> sometimes has 8 seeds; the official site is Cloudflare-blocked; a
> third-party "entry list" site failed a plausibility check). **Next real
> step is the entry-rules reference** (plan item 3) - not started. Full
> detail, every source tried and why it failed:
> [plans/all-events-forecast.md](plans/all-events-forecast.md) - read that
> before picking this up, it is a multi-session project, not a queue item.
>
> **Read first if you are cold:**
> - The market question is CLOSED (D-5, D-6, D-10). Fourteen rounds, a clean
>   2026 forward test, no exploitable edge. Do not reopen it without new data —
>   [issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md).
> - The US Open is OVER (final 2026-09-13). The laptop capture/live-sim tasks
>   are DISABLED (2026-09-15) - the event no longer needs them. The Cloudflare
>   Worker's GitHub Actions backup self-expires 2026-09-16T02:00Z.
> - C vs D is OPEN again, not settled (D-51 reverses D-18) - see queue item 2
>   under "Queued, ranked" above it in this file. Nothing published is
>   affected either way; `--live-score` stays off.
> - PR #34 is CLOSED, not merged (D-52) - superseded by code already on `dev`.
>   Its surviving fatigue/bye piece was re-implemented fresh (D-53).
> - `main` tracks `dev` via PR through #37; local `main` needs a `git fetch`
>   if it looks stale (it is not behind on GitHub, just unfetched locally).

## Session handover — 2026-09-07

**Shipped to `main`** (five PRs, each reviewed before the PR existed):

| PR | what |
|---|---|
| #31 | in-match seeding at POINT-level resolution, behind `--live-score`, off by default |
| #32 | CI — the in-match suite plus a guard that fails the build if a producer passes `--live-score` |
| #33 | WTA calendar paging fix, and `rescore_after_final.py` |
| #35 | daily tennismylife archive, and Wimbledon capture |

**Running when this session ended.** A four-year US Open backfill, detached and
already orphaned from the session that started it, so it survives a close.
Roughly 6.5 hours total, ~1.3 GB, about 944 matches a year.

- progress: `ls dropshotdata/raw/usopen/<year>/matches | measure` (expect ~944)
- log: `%TEMP%\claude\C--dev-dropshotverse\...\scratchpad\backfill.log`
- **resumable**: `backfill_done.json` per year, saved every 25 matches. If it
  dies, re-run `capture_usopen.py backfill --year <y>` and it skips what is done.
- nothing alarms if it fails. Nobody is watching it.

**Dated, needs nothing from anyone:**

- **2026-09-13** — US Open men's final.
- **2026-09-14 08:00** — scheduled task `DropshotverseRescoreAfterFinal` runs
  `rescore_after_final.cmd`, settling arm C against arm D on real late rounds.
  Log: `dropshotdata/raw/usopen/rescore.log`.
- **2026-09-16 12:00** — the 5-minute capture and 30-minute live sim tasks stop.
- ~~**~2026-09-23** — dispatch `event-forecast.yml` group `1020` for Beijing
  once the field shows 96 entrants and 32 seeds.~~ **SUPERSEDED 2026-09-15**:
  Pete's call was to dispatch provisional NOW rather than wait - done, and
  it's live and rendering on ITG (see the top banner). The field is still
  76/96, 0 seeds at time of dispatch; re-dispatch nearer 09-30 once it
  completes to move `source` from `provisional` to `locked`. **Dispatch with
  `--ref` matching where the code lives**; `gh workflow run` defaults to the
  default branch, which cost two confusing failures. Related and worth
  knowing generally: a `schedule:` trigger only ever fires from the DEFAULT
  branch, so a cron
  workflow sitting on `dev` does nothing at all - that is why the tennismylife
  archive had to reach `main` to work, not merely to be tidy.

**Queued, ranked:**

1. **PR #34 is STALE, not mergeable — do not merge as-is (D-52).** Base is
   `dev` (not `main`), and `dev` has diverged since 2026-09-07: it now shows
   `CONFLICTING`. Worse, the D-46/D-47 `DSV_W3_LAYOFF_FIX` env-var mechanism
   (shipped 2026-09-10, superseding this branch) already solves the layoff
   half of what PR #34 does, by a different, incompatible mechanism - PR #34's
   `round_aware`/`first_round`/`IN_EVENT_WEEKS` symbols don't exist in current
   `dev` at all. What PR #34 still has that `dev` does NOT: the D-28
   fatigue/bye fix (`fatigue_diff` is still hardcoded to 0.0 in
   `xgb_bt_rating.py:429`, wrong for Beijing's 96-into-128 draw with byes).
   ~~**Next action: re-implement just the fatigue/bye piece against current
   `dev`.**~~ **DONE 2026-09-15 (D-53)**: `first_round` param on
   `build_round_pred_matrices`, default `None` (every existing caller
   untouched by construction, no `round_aware` flag needed this time), wired
   through `simulate_wta_event.py` behind the same `n_seeds >= n_byes` guard
   PR #34 had. 8 anchors pass in `tune_scratch/test_w3_fatigue_bye.py`.
   Committed and merged to `main` (PR #38). **PR #34 closed without
   merging** the same session.
2. ~~**Verify per-year field coverage** on the backfilled 2022-24 tapes.~~
   **DONE (D-54): found 2022-23 unusable as captured** (no `P1SetsWon`/
   `P2SetsWon` field). ~~Pete's call: build the fix.~~ **DONE 2026-09-15
   (D-55).** Not the approach that failed once before (SetWinner is a
   tape-provided field here, not derived from GameWinner replay) - checked
   it against `P1GamesWon`/`P2GamesWon` instead of the missing running total.
   733/946 (2022) and 754/963 (2023) directories now pass with zero anchor
   failures, 512/514 singles matches recovered. `usopen_2022_points.parquet`
   / `usopen_2023_points.parquet` written (gitignored; regenerate via
   `tune_scratch/build_point_table_legacy_years.py`).
3. **Shrink factors / hier vs serve-return Elo: DEPARTED, Pete's call
   2026-09-15 - not important right now.** Real progress made first: a
   player-identity crosswalk for the 2022-25 tapes (D-55/D-56/D-59, now
   100% clean on all 159 top-20 seeds across both tours and 4 years) and a
   first real hier-vs-actual comparison (`tune_scratch/hier_vs_actual_serve_examples.py`,
   712 matches) that flipped the direction of the question - hier reads
   UNDER-dispersed (ratio 0.831) on this sample, not over-dispersed the way
   D-40/41/42 assumed, though still inside the same wide, unstable range
   those gave (0.78-2.40). Pick up fresh when this becomes a priority again;
   don't restart the crosswalk work, it's done and committed.
4. **`event = match_id[0]` mistags qualifying-round (5-digit) matches by
   tour (D-56).** Confirmed on real players - ATP men under `event="2"` ids
   (`21109`), WTA women under `event="1"` ids (`12113`). 4-digit main-draw
   ids are fine. ~~Needs its own investigation: does anything downstream
   trust `event.isin(["1","2"])` for a 5-digit id today.~~ **DONE (D-57):
   yes, real damage found and fixed.** `SERVE_BASE`'s drift check
   (`screen_inmatch_seeded.py`) computed tour_avg over all 688 matches
   instead of the 254 main-draw ones - moved the ATP figure 0.611 -> 0.628,
   a real 0.017 shift. Fixed both `SERVE_BASE` and the drift check to
   main-draw-only. Re-ran the rescore: **C vs D is unchanged** (the seeding
   solve only uses `SERVE_BASE` as a symmetric centre), so D-51's
   conclusion stands. The C/D scoring itself was never contaminated - it
   already skips matches with no live-draw pre-match number. ~~Still open:
   audit every reader of `usopen_2026_points.parquet`.~~ **DONE.** One more
   had it - `inmatch_wp_calibration.py` (ticket 09's original arm A/B work,
   not on the active decision path, and with no protective gate at all) -
   fixed the same way (254 main-draw matches, 434 excluded).
   `screen_inmatch_recalibrated.py` was never at risk (reads
   `screen_inmatch_seeded.py`'s already-filtered output. ~~Root cause still
   open: `build_point_table.py` doesn't distinguish main-draw from
   qualifying/doubles when writing the tape.~~ **DONE (D-58): fixed at the
   source.** `build_point_table.py` and
   `tune_scratch/build_point_table_legacy_years.py` now filter to
   4-character match ids before writing; all 5 years regenerated (2022
   46,678 pts/252 matches, 2023 44,939/253, 2024 44,783/253, 2025 45,037/248,
   2026 47,431/254). Re-ran the full rescore end to end - numbers match the
   pre-fix run exactly, confirming this was a true no-op for every current
   consumer. **D-56 is fully closed**, nothing left queued on it.
5. **C vs D reopened (D-51).** The 09-15 re-score reverses D-18 on more late-
   round evidence (5,971 points vs the original 7 matches): arm D now beats
   arm C. Not urgent - `--live-score` is off and nothing published is
   affected - but do not cite D-18 as settled until this is re-run on a full
   season rather than one event's late rounds.

**Undecided, and why — the serve/return rating.** Branch
`feat/serve-return-ratings`, NOT merged, research-grade.

- The hierarchical model has always carried a per-player serve and return skill.
  `ratings.parquet` ships only `elo_overall`, `bt_rating`, `age`.
- A paired serve/return Elo was built (D-33) at both point and game level.
- **hier is ahead on ordering** on the only test with no absorption in it: the
  2026 tape, ratings frozen before the event. hier +0.478, best Elo +0.444.
- **The K sweep was invalid** (D-40). It scored each K by aggregating pre-match
  ratings across 2022-23 and correlating with performance over that same
  window, so a fast-adapting rating absorbed the outcomes it was judged on.
  Correlation rose monotonically with K and never turned over — the tell. On
  the tape the ordering reverses.
- **The shrink is free** (D-41), confirmed out of sample by a hash split:
  correlation identical to three decimals, calibration recoverable without
  spending a model parameter.
- **But the shrink factor cannot be estimated from one tournament** (D-42), and
  this undercuts every dispersion figure quoted this session. hier reads **1.38
  on one random half of the tape and 2.40 on the other**. 1.96, 0.95 and 0.78
  are point estimates with error bars wide enough to contain each other.

**Deliberately not done:**

- **Nothing published to ITG.** Publishing a serve rating whose spread is not
  yet estimable would be the plausible-looking wrong number this repo keeps
  getting bitten by.
- **`--live-score` left off.** Measured and better, but the switch-on checklist
  is unfinished, and one item needs a match actually in progress.
- **Roland Garros and the Australian Open not added** to the capturer. RG
  serves nothing on this path shape; ausopen.com returns 200 HTML for every
  path, a soft 404 that briefly looked like a hit (D-37).

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
3. ~~**Republish the model-inputs Release and commit the 2026-09-09 cutover
   (D-40 to D-43).**~~ — DONE: Release republished, PR #36 merged 23:02 AEST.
4. ~~**Flip `DSV_W3_LAYOFF_FIX` before Beijing (D-46).**~~ — **DONE 2026-09-10
   00:05 (D-47)**, with the debutant case (NaN, learned) and the second
   retrain (D-48 keep-all, D-49 rust, D-50). Republish the model-inputs
   Release and PR are in the same session.
5. **Watch the first scheduled runs on the new bag**: laptop task 00:34 AEST
   2026-09-10 (the 00:04 run failed inside the swap window, by design) and
   `ratings-refresh.yml` at 06:00 UTC, which must fetch the republished
   Release or the ratings page's BT column reverts overnight.
6. **Rust refinements, only if the comeback slice keeps rewarding it**: weight
   a match by sets or games played, and by whether it was won. D-49.
7. **Issue #44: take the China Open off the published forecast.** The code fix
   drops started events on every write, but nothing writes until the next run.
   Either dispatch `event-forecast.yml` for an event that has not started yet
   (group and `--ref` as in the workflow's inputs), or remove the China Open
   rows from the three R2 files by hand. The worker cannot write R2. Schedule
   decision: manual dispatch stays, no cron (D-60). Say so if you want a cron.

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
6. ~~**The surface rating is not a self-correcting rating.**~~ — **TESTED AND
   SHIPPED 2026-09-09 (D-42)**: `p_s` is now the default. On the 2022-23 tune
   window the retrained XGBoost is +0.0012 log-loss WORSE (paired t +1.7): the
   old column was a surface residual and that was signal. Pete chose the
   correct rating anyway. Revisit only as "both columns", never as a swap back.
7. ~~**[#1](https://github.com/peteowen1/dropshotverse/issues/1) — hier tail
   bug, downgraded (D-12).**~~ — **ANSWERED 2026-09-09 (D-41)**: the surface
   offset was absorbing half of every match's form; fixed with
   `DSV_SURF_RW_SCALE=0.0`. Standalone point model 0.745 -> 0.698; through
   XGBoost, aggregate unchanged, tail rows better. Issue #1 can close on this.
8. **[#2](https://github.com/peteowen1/dropshotverse/issues/2)** — the WTA
   public live-scores API. Its in-play fields have never been observed
   populating.
9. ~~**`DropshotverseUSOpenCapture` exits 1 intermittently.**~~ — **FIXED
   2026-09-06**: a `ConnectionResetError` mid-read escaped `fetch()`'s retry
   (it is an `OSError`, not a `URLError`) and killed the sweep. Now retried.

10. **[#45](https://github.com/peteowen1/dropshotverse/issues/45)** — same
    player, different ids across the published files. Root cause was the TML
    bridge minting a synthetic id on any name spelling difference; fixed in
    `bridge_tennismylife.py` (looser keys, ambiguous ones dropped), which the
    daily ratings refresh and the live projection both use, so the next runs
    carry the real ids. (b) is now fixed: `build_blog_data.draw_uids_by_tour`
    also unions in `blog/live_predictions.parquet` (the ratings-refresh workflow
    fetches it from R2, optional), so late entrants such as Bu get a ratings row;
    checked against the real archive (Bu `atp:207352` and Knutson `wta:213873`
    both land in the pool; test `tune_scratch/test_draw_uids_union.py`). Takes
    effect on the next ratings-refresh run. **Still open:** (a)
    `tennis/draw.json`'s producer is not in this repo (`us_open_2026_drawonly.json`
    here is the only local copy, dated 2026-08-31 and still carrying
    `wta:NEW_Gabriela_Knutson`), so its `NEW_` ids need fixing where it is built
    (the blog repo's tennis draw workflow); the live projection's own ids heal
    on its next run with the bridge fix.

## Live deadlines (absolute dates)

- **2026-09-13, after the final** — rebuild `usopen_2026_points.parquet`
  (`python build_point_table.py`) and re-run `screen_inmatch_seeded.py` then
  `screen_inmatch_recalibrated.py`. That is the late-round validation arm D
  needs before it ships, and it cannot be done earlier because the matches
  have not been played.
- **2026-09-13** — US Open men's final. `live-projection.yml` has no expiry.
  The laptop tasks run to **2026-09-16 12:00 local** (2026-09-15 22:00 New
  York); the point capture still depends on the laptop.
- ~~**2026-09-23** — dispatch `event-forecast.yml` with group `1020` for the
  China Open (Beijing, starts 2026-09-30) once the field shows 96 entrants
  and 32 seeds. On 09-07 it was 76 of 96 and 0 seeded, and the pipeline
  correctly refuses a partial field without `allow_partial`.
  **A dry run on 09-07 found it could not have run at all**: the calendar
  fetch took one page of 100, so a full-year window ended on 2 March and
  Beijing (30 September) did not exist as far as the simulator was concerned.
  Fixed — it pages now, 523 tournaments over 6 pages, and refuses a calendar
  shorter than the endpoint's own `totalElements`. Re-verified end to end on
  `dev`: Beijing resolves, 76/76 entrants at 100%, and it stops at the field
  guard as intended. **Dispatch with `--ref` matching where the fix lives**;
  `gh workflow run` defaults to the default branch, which cost one confusing
  failure here.~~ *Moot 2026-10-25: the China Open started 2026-09-30; see
  Awaiting Pete 7.*
- **2026-09-14 08:00** — scheduled task `DropshotverseRescoreAfterFinal` runs
  `rescore_after_final.cmd` once: rebuild the point table, re-run both in-match
  arms, print the late-round comparison beside the numbers on record. Log at
  `dropshotdata/raw/usopen/rescore.log`. Run early it reports that the later
  rounds are missing and stops. This is what settles C versus D on more than
  7 matches.
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

   **Shipped to `main` 2026-09-07 via PR #31**, behind `--live-score`,
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

   Review of the branch found and fixed TWO real defects underneath all of
   this: 6-6 sets decided twice (error up to 0.42, D-16) and the point tape's
   set-ending row read as a set in progress (1.58% of points, D-17). Every arm
   was re-measured twice. **C over D is now settled (D-18)** - arm D clears the
   headline bar on the corrected tape but fails its own anchor D3, the two
   tours' temperatures being 1.209 and 1.458 against a 0.15 spread bar. The
   2026-09-13 final is now a confirmation of the late-round result on more than
   7 matches, not the decider it was. `tune_scratch/screen_inmatch_seeded.py`,
   `screen_inmatch_recalibrated.py`;
   [plans/validation-charter.md](plans/validation-charter.md).
6. **W3 MEASURED 2026-09-07 (D-26): real, worth fixing, not mid-event.**
   Zeroing the stale layoff difference from round two on moves published
   probabilities by mean 0.023 (ATP) / 0.018 (WTA) across the field, p99 ~0.17.
   Restricted to the 12 players still alive it is 0.012 / 0.006 - the extreme
   layoffs are already out. The effect is largest at the START of a draw, so
   **do it before Beijing (2026-09-30)**, not during a US Open that is nearly
   over. Two things surfaced alongside it: `weeks_since_last` defaults to
   **52.0** for a player with no Elo history, so a debutant looks a year
   injured (D-27, and this is W5's real shape); and `fatigue_diff` is hard
   coded to 0.0, correct for a 128 draw with no byes and wrong for Beijing's
   96-into-128 with 32 byes (D-28). Sizing script:
   `tune_scratch/screen_w3_frozen_layoff.py`.
   **FIXED and open as [PR #34](https://github.com/peteowen1/dropshotverse/pull/34),
   deliberately NOT merged.** All three share one cause - the simulator had no
   notion of how many matches a player had already played - so they are one
   change. CI green, 8 anchors. **Merge on 2026-09-14, after the final and
   after `DropshotverseRescoreAfterFinal` has run**, so the late-round re-score
   scores the model that produced the published numbers rather than a changed
   one. Review caught two things worth knowing: the LOCKED pre-tournament
   forecast would have been silently rewritten (now opts out via
   `round_aware=False`), and the bye mapping was wrong for exactly Beijing's
   current state, where 0 seeds means the leftover byes are drawn at random
   every simulation.

## Known and NOT chased

- **Config D forward test** — would turn the +0.0247 upper bound into the
  shipped model's exact gap. Skipped: three measurements agree, and a model
  right 38% of the time when it dissents has no edge worth sizing.
- **Published-copy fallback in the WTA event merge** — with no local table file,
  `simulate_wta_event.py` merges into the copy on R2. A read that fails with
  anything but a 404 raises, so a flaky read fails the dispatch red and writes
  nothing. A 404 counts as "no table yet" (`_is_missing`), so a missing public
  copy starts the merge from empty, and the next upload holds only the new event.
  Not chased: retrying the read, or reading from another source.
- **Seed threshold for smaller draws** — `expected_seeds` asks for 8 seeds in a
  draw of 32 or fewer, 16 up to 64, and 32 above; `complete` needs no TBD slots
  and at least that many seeds. Not chased: whether those numbers match each
  tour's seeding rules for draws under 65 entrants. No check of that was made.
