# dropshotverse — Decisions

> Work queue: [NEXT-STEPS.md](NEXT-STEPS.md) · Vault map: [HOME.md](HOME.md)
> Created 2026-09-05, back-filled from `.scratch/dropshot-tennis-engine/`.
> A row is the decision, the one fact that drove it, and a link to the evidence.

## Awaiting Pete

**D-4 — publish a "we match the market" claim?** Not yet, and this is a
recommendation rather than a question: the 0.00073 gap comes from a holdout
revealed a dozen times, and the 2026 forward test puts the honest gap at
0.0247 with the model right 38% of the time when it dissents
([issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md)).

## Made

**D-1 — canonical base archive.** Answered by D-11; issue #3 closed 2026-09-06.

**D-2 — the review gate.** Resolved 2026-09-06: PRs #8-#12 merged, each after
a Sonnet reviewer pass. `main` had received nothing since 2026-08-31.

**D-3 — machine memory.** Cleared 2026-09-06 without a reboot. The kills were
Claude Code's guard on tracked tasks, not the OS. Narrative in
[backlog/journal-2026-09-05.md](backlog/journal-2026-09-05.md).

**D-5 — 2026-08-27: the destination was revised on measured evidence, from
"beat the closing line" to "calibration parity with the market".** Ticket 10
measured the gap: surface-adjusted Elo trailed Pinnacle by 0.026 log-loss and,
where it disagreed with the market, was right only 40.5% of the time. Ingram's
published point-level model trailed FiveThirtyEight's Elo by 0.006, so a
point model of comparable quality would still have lost by a wide margin.

**D-6 — 2026-08-29: Pete reopened "beat the market" anyway,** explicitly, via
`/goal`. Eleven rounds of candidates followed; the gap closed from 0.0145 to
0.00073 on the holdout and WTA beat the market there (0.58779 vs 0.58852),
while ATP still trailed by 0.00214. Superseded by D-10 and the forward test.

**D-7 — 2026-08-31: the frozen Sackmann archive was published as a GitHub
Release.** Sackmann's repos went 404 on 2026-08-27 mid-project; the snapshot
existed as a single copy on one laptop until this.

**D-8 — 2026-08-31: WTA data may go on the ITG site.** Prompted by discovering
that `api.wtatennis.com` is open and unauthenticated, which overturned ticket
03's "WTA is walled" conclusion. ATP genuinely is walled (server-rendered
Sitecore, no client API); the asymmetry runs opposite to what the project
assumed for four days.

**D-9 — 2026-09-05: keep improving the model before running the 2026 forward
test.** Pete's call, a deliberate trade: more candidate rounds on a burnt
holdout with the unbiased measurement deferred. The forward test ran the next
day anyway once rounds 12-14 were exhausted.

**D-10 — 2026-09-05: the 2024-25 holdout is retired from candidate screening.**
Revealed at least a dozen times; 2026 was reserved as the final unbiased test
and has now been spent ([issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md)).

**D-11 — 2026-09-06: keep the frozen Sackmann archive as the historical base
and TML as the live bridge.** TML serves WTA only from 1990; the archive holds
61,840 tour-level WTA matches from 1968-1989, which the all-time GOAT list
needs. The crosswalk was never the gate. Detail:
[issues/39](../.scratch/dropshot-tennis-engine/issues/39-tml-depth-decides-issue-3.md),
[issues/38](../.scratch/dropshot-tennis-engine/issues/38-crosswalk-scoping.md).

**D-12 — 2026-09-06: issue #1's hier tail bug is not a live-pipeline problem;
`hier_logit_pwinner` stays.** On identical holdout rows the standalone model
gives the winner under 1% in 0.71% of matches, the shipped bag in 0.00%; and
dropping the feature costs +0.0092 log-loss on 5/5 paired folds. Tables on
[#1](https://github.com/peteowen1/dropshotverse/issues/1),
script `tune_scratch/screen_hier_contamination.py`.

**D-13 — 2026-09-06: the live projection publishes from GitHub Actions from
two Releases.** (Windows runner plus a junction at first; moved to Linux the
same evening once the live path resolved its own paths, PR #17: 2 min 45 s a
run at 1x billing against 5 min at 2x.) Windows because every script hard-codes
`C:\dev\dropshotverse\...` and a junction resolves it; the derived inputs
became the `model-inputs-2026-09` Release; the R2 token reused is the laptop's
(Pete's call over minting an R2-only one). Workflow header comments and
[#4](https://github.com/peteowen1/dropshotverse/issues/4) carry the detail.

**D-14 — 2026-09-06: the Actions producer is dispatched by the ITG Cloudflare
Worker when R2 goes stale, not by GitHub's cron.** GitHub fired the `*/30`
schedule once in two hours, and a Windows runner (billed 2x) every 30 minutes
is ~500 minutes a day against a 2,000-minute free month. The Worker's existing
`*/30` cron (all five free slots were taken, so it piggybacks) calls
`workflow_dispatch` only when `tennis/live_predictions.parquet` is >40 min
old and only until `TENNIS_LIVE_UNTIL`. Laptop stays primary through the
final. Needs a fine-grained PAT from Pete. inthegame-blog
`worker/src/index.js`, `maybeTriggerTennisProjection`.

**D-15 — 2026-09-07: bookmaker prices are a BENCHMARK ONLY and never a model
input. Pete's call, on product grounds, and it closes W1.** The measurement was
run first and is recorded in
[plans/validation-charter.md](plans/validation-charter.md): blending the price
in beats the model in 6 of 6 years and beats the market in 6 of 6, by 0.0049
and 0.0040, at a fitted weight of 0.52. It is a real gain and it is rejected
anyway, because at w=0.52 roughly half of every number the site published
would be the bookmaker's opinion carrying our name. Pete: "I don't want to use
bookmakers odds at all in our model."

Consequences to hold onto:

- `raw/odds/` stays exactly where it is and keeps doing the one job it has:
  the benchmark in the walk-forward scorecard. Nothing to delete.
- The project goal (D-5) is calibration parity with the market. The scorecard
  says the model is **0.0008** behind the bookmaker average across seven
  years, reached on its own features. That is the honest statement of parity,
  and it is a better story than a blend would have been.
- Do not reopen W1 as an accuracy question. It was not rejected for being too
  small; it was rejected for what it would make the published number mean.
  New evidence about its size does not bear on that.

## Standing corrections

- **`slot` in the entrants files is not draw position** (2/128 agreement with
  the real draw). Building a bracket on it produces a completely wrong result
  that still looks plausible.
- **Ticket 01's "TennisMyLife's manifest has 49 ATP challenger files and zero
  WTA" is wrong**, and shaped scope from 2026-08-27 to 2026-09-06. TML serves
  WTA for every year from 1990; `bridge_tennismylife.py` had been pulling
  `2026_wta.csv` the whole time. The real limitation is no WTA *before* 1990.
- Two more (the RESUME log is stale from round 11; the archive does not end at
  Roland Garros) live in [HOME.md](HOME.md), where a cold session reads them.
- **D-16 (2026-09-07)** — `p_set_from_state` decided a 6-6 set by playing a
  thirteenth game and then applying the tiebreak on top, an error up to 0.42.
  Fixed. It was invisible to the equal-players anchor (both halves give 0.5)
  and had been documented in the code as a convention difference rather than a
  bug; the recursion now agrees with `hm.p_match` at the opening boundary to
  machine precision. Nothing published was affected - the live path does not
  reach this module on `dev`. Evidence and re-measured arms:
  `docs/plans/validation-charter.md`.
- **D-17 (2026-09-07)** — The point tape's set-ending row carries the new set
  count AND the finished set's game score at once (match 1101 point 70: sets
  1-0, games 6-4). 1,410 points, 1.58% of the tape, sat on such a row and were
  being scored as a set in progress. Fixed in the screen; every in-match number
  measured before this is void. Arm C: log-loss 0.53106 -> 0.48286.
- **D-18 (2026-09-07)** — Ship arm C (seed the in-match recursion from the
  model's pre-match number), not arm D (temperature on top). Arm D clears the
  headline bar on the corrected tape but fails its own anchor D3: the men's and
  women's temperatures are 1.209 and 1.458, a spread of 0.248 against a 0.15
  bar, so a single temperature is not what the data contains. Late rounds
  independently still prefer C. Evidence:
  `docs/plans/validation-charter.md`.
- **D-19 (2026-09-07)** — The point tape's state columns are post-point while
  `PointServer` is pre-point; the screen was pairing them. Now scored from the
  state as each point began. Two consequences: the serve was wrong on every
  game-ending point, and anchor C3 was vacuous (a post-point final state is the
  finished match, so it returned 1.0 and reported 100% for any model). C3 now
  reads 99.1%. Headline numbers barely moved; all conclusions stand.
- **D-20 (2026-09-07, CLOSED same day)** — The in-match recursion worked at
  GAME granularity while the data was point level: 84.6% of points carried a
  within-game score the model never saw, and 40-0 up on serve read the same as
  0-40 down. Added `p_game_from_state` and `p_tiebreak_from_state`, threaded
  through as arguments defaulting to 0-0 so every existing caller is
  bit-identical. Aggregate barely moved (arm C log-loss 0.48467 -> 0.48310);
  the median per-point change went from 0.0000 to 0.0109, which is the actual
  point. Evidence: `docs/plans/validation-charter.md`.
- **D-21 (2026-09-07)** — In-match seeding merged to `main` (PR #31, 8 commits,
  +1434/-76). `--live-score` is OFF in both producers, so the code is live and
  inert; the three-item switch-on checklist is in NEXT-STEPS. Five review
  rounds each returned a finding; all six defects were pre-existing and none
  had ever published a wrong number, because the live path did not reach
  `ticket09_live_modes` until this branch wired it in.
- **D-22 (2026-09-07, CLOSED same day)** — This repo had NO CI. `gh pr checks` reports
  none, so the "verify CI conclusion" step of the review gate is vacuous here
  and #31 merged on local verification alone. A workflow running
  `dropshotdata/tune_scratch/test_inmatch_seeding.py` on pull requests now
  gives the gate something real to check: `.github/workflows/tests.yml`. The
  suite was made offline and deterministic first (a checked-in feed fixture
  instead of a live fetch), so a red run means the maths changed rather than
  that a tournament ended. A second job fails the build if either producer
  passes `--live-score`, since the working tree is production and turning it on
  by accident would publish within 30 minutes.
- **D-23 (2026-09-07)** — `fetch_wta_upcoming.calendar` requested one page of
  100 and stopped, so a full-year window returned 100 of 523 tournaments,
  ending on 2 March. Every event after early March was invisible, including
  Beijing. The entry-list fetch does not go through `calendar()`, which is why
  nothing looked wrong. Found by dry-running the forecast 16 days early rather
  than on the day. It now pages and refuses a calendar shorter than the
  endpoint's reported `totalElements`.
- **D-24 (2026-09-07)** — The truncation guard added in D-23 was dead code. It
  read `totalElements`, which this endpoint does not send: the response has two
  top-level keys, `pageInfo` and `content`, and the count is
  `pageInfo.numEntries`. So the comparison never ran while the docstring
  claimed silent truncation was impossible. Now reads the right field, raises
  when the count is absent as well as when it is short, and both paths are
  exercised rather than asserted. A check that cannot fail is worse than no
  check, because it stops anyone looking.
- **D-25 (2026-09-07)** — Pre-paging runs cached a whole window under one key
  holding only its first page, and `simulate_wta_event.py` globs that directory
  with `cal_*.json`, so a fixed fetch could still be undone by a stale file.
  The glob is now `cal_*_p*.json`, `calendar()` deletes a same-window
  single-page cache when it pages, and the two on disk were removed.
- **D-26 (2026-09-07)** — W3 sized and it is real. `build_player_table`
  computes `weeks_since_last` once from DATE0 and the simulator reuses it for
  every round, so the model is fed the pre-tournament layoff gap at the final.
  From round two on both players have just played, so the true
  `weeks_since_last_diff` is ~0. Zeroing it moves published probabilities by
  mean 0.023 (ATP) and 0.018 (WTA) over the whole field, p99 ~0.17, max 0.25.
  **Worth fixing, but not mid-event**: restricted to the 12 players still alive
  the effect falls to 0.012 (ATP) and 0.006 (WTA), because the extreme layoffs
  have been eliminated. The value is largest at the START of a draw, so the fix
  belongs before Beijing (2026-09-30), not during a US Open that is nearly
  over. Evidence: `tune_scratch/screen_w3_frozen_layoff.py`.
- **D-27 (2026-09-07)** — W3 and W5 interact, which neither ticket said.
  `weeks_since_last` defaults to **52.0** for a player with no Elo history, so
  a debutant is indistinguishable from someone a year injured and carries a
  ~50-week difference against every opponent. Three ATP and one WTA entrant hit
  this. Any W3 fix should decide the debutant case explicitly rather than
  inheriting 52.0.
- **D-28 (2026-09-07)** — `fatigue_diff` is hard-coded to 0.0 in
  `xgb_bt_rating.py:400` for every simulated match. For a 128 draw with no byes
  that is exactly right, since both players have played the same number of
  matches. It is NOT right for a draw with byes: Beijing is 96 into 128 slots
  with 32 byes, so in the round after the byes a seed meets someone who has
  played a match and the true difference is -1. Untested, and it reaches the
  WTA event forecaster rather than the US Open path.
- **D-36 (2026-09-07)** — Data-source survey, after Sackmann's repos died. What
  is free and current, probed rather than assumed (~25 endpoints, 5 providers):
  **ESPN** is live and has the US Open (239 matches a draw, scores and set
  lines) but NO serve statistics and NO point-by-point - its `plays` resource
  returns `count: 0`. **The WTA API** is live and real (18,787 tournaments,
  36,138 players, `/tournaments/{id}/{year}/matches` works) but is scores only.
  **atptour.com** is 403 behind Cloudflare. **Sofascore** is 403.
  **tennismylife** is the only tour-wide source with serve stats and lags 8-14
  days - it had no US Open at all on 2026-09-07, mid-tournament.
- **D-37 (2026-09-07)** — The slam feeds are the richest free tennis data
  available and we were using one fortnight of one of them. The US Open's
  `slamtracker/history` answers for **2022 through 2026**, Wimbledon for
  2025-2026, on the same path shape. Roland Garros has nothing on that shape;
  the Australian Open returns a 200 HTML page for every path, which is a soft
  404 and which I briefly mistook for a hit. The payload is richer than the
  match archive: serve speed, serve width and depth, return depth, kick height,
  rally count, distance run and shot type, per point. Backfilling the US Open
  2022-25 takes ~89k points to roughly 450k, which is what makes WTA
  calibration measurable at all - on one event 88% of the between-player spread
  was sampling noise (D-35).
- **D-38 (2026-09-07)** — tennismylife is archived daily to its own Release
  (`.github/workflows/archive-tennismylife.yml`). It was the only tour-wide
  source of serve statistics we have, fetched live on every run, with no copy
  kept anywhere - the identical exposure that made Sackmann's 404 a crisis in
  August. Latest plus dated copies, so a bad upstream day can be stepped back
  from. An empty or schema-changed response FAILS the job; mere staleness only
  warns, because 8-14 days behind is this source's normal.
- **D-39 (2026-09-07)** — The capturer takes `--slam` and handles Wimbledon,
  which serves the point tape but 403s `eventDays.json`, `seeds`, live scores,
  `insight` and `match_stats`. It cannot be enumerated the way the US Open is,
  so it falls back to the draw feeds: MS and LS give 254 decided singles
  matches. `enumerate_via` in SLAMS records which host does what, because it is
  a fact about each site rather than a preference.
- **D-40 (2026-09-09)** — Pete asked why Zverev had the 3rd-highest BT on the
  site yet was title favourite. BT is the row mean of the published pairwise
  matrix and matched it (2.23 vs 2.17), so the column was right; the model
  underneath was not. Walking ONE row (Shelton 92% vs Khachanov, Zverev 85%)
  found two broken inputs. (1) The point model's surface offset was half a
  form rating: `inflate()` gave kappa and mu the same weekly noise, so the
  Kalman split sent 50% of every match to the surface term, and nine August
  wins moved Shelton's hard-court serve+return 0.7 point-logits (94.5% vs
  Zverev; Elo said 62%). (2) `rs` was a residual against overall Elo, not a
  surface rating (Alcaraz 1609 on hard, Musetti 1160; corr with `ro` 0.26).
  Probes: `tune_scratch/probe_zverev_shelton.py`, `probe_hier_trace.py`.
- **D-41 (2026-09-09)** — Fix (1) shipped: `DSV_SURF_RW_SCALE`, default 0.0
  (surface offsets get no process noise; a surface tendency is a trait).
  Swept 1.0/0.3/0.1/0.0 on 2022-23 under the frozen v25 params: point-model
  log-loss 0.745 -> 0.698, ECE 0.140 -> 0.120, monotone. Through XGBoost the
  aggregate is unchanged (paired t -0.3, n=9,711) but the 43 rows it moves
  >10 pts go 0.650 -> 0.600. Anchors after: Shelton 75% vs Zverev (was 94.5%),
  Alcaraz 71% vs Shelton (was 42%), Djokovic 78% vs Khachanov (was 52%).
  Answers queue item 7's "would a fixed hier make a better feature": in the
  tail, yes; in aggregate, no. `tune_scratch/sweep_surf_rw_v25.log`,
  `compare_input_fixes.py`.
- **D-42 (2026-09-09)** — Fix (2) shipped ON PETE'S CALL against the tune
  window: `rs` now updates against its own expectation
  (`DSV_SURFACE_OWN_EXPECTATION`, default 1; corr with `ro` 0.91). Retrained
  XGBoost scores +0.0012 log-loss WORSE on 2022-23 (paired t +1.7), because
  the old column was signal as a surface RESIDUAL even though it was noise as
  a rating, and the corrected one is near-collinear with overall Elo. Closes
  queue item 6. If it is ever revisited, the candidate is both columns
  (rating and residual), not a choice between them. Legacy parquet kept as
  `elo_raw_ratings_full.LEGACY_SURFACE_RESIDUAL.parquet`, legacy bag as
  `xgb_bagged_model_cache_v2_setgame.LEGACY_2026-09-09`.
- **D-43 (2026-09-09)** — Cut over mid-tournament at 22:27 AEST, between two
  runs of the 30-minute task, in one pass: parquet, bag, default, and the
  snapshot-cache key now carries the knob (`_s0`) so a stale replay cannot be
  served. Live sim rerun by hand: anchors pass; Zverev title 38.2% -> 44.6%,
  Shelton 38.2% -> 34.4%; Zverev's BT rank 9th -> 3rd. Before/after:
  `tune_scratch/compare_bt_before_after.py`. The model-inputs Release
  (`model-inputs-2026-09`) was republished at 22:50 AEST with the fixed
  parquet and the retrained bag, so the Actions backup and the daily ratings
  refresh run the same vintage as the laptop. Feature importance of the new
  bag on 2022-23 (mean |SHAP| share): game Elo 21%, surface Elo 19%,
  point-model prob 13%, age 10%, overall Elo 3%. The corrected surface column
  has taken over overall Elo's role, as D-42 predicted.
  `tune_scratch/importance_new_bag.py`.
- **D-44 (2026-09-09)** — Why Fils and Jodar sit level with Zverev in BT when
  Elo has him 50-70 points clear: `age_diff`, worth -0.2 logits to Zverev
  against every 20-year-old (Fils +0.15, Jodar +0.22, Zverev -0.06, Djokovic
  -0.34 mean over the field). Ablated: dropping age costs +0.0006 log-loss on
  2022-23 (paired t +1.3, n=9,711). **Age stays.** BT is a forecast of the next
  match, not a ranking of who has been best; if the site wants the latter it is
  a different column. `tune_scratch/probe_zverev_fils.py`, `cmp_noage.log`.
- **D-45 (2026-09-09)** — Both surface columns (corrected rating + legacy
  residual, corr 0.48) tested: -0.0003 vs production (t -1.1), a quarter of
  what D-42's swap cost, inside noise. **Not shipped.** `cmp_resid.log`.
- **D-46 (2026-09-09)** — W3 sized on the real pre-tournament ATP field with
  the fix behind `DSV_W3_LAYOFF_FIX` (zero `weeks_since_last_diff` from round
  two on): pairwise odds move 0.03 mean / 0.25 max per round; Alcaraz's title
  odds 38.4% -> 60.4% because his 20-week gap was charged seven times. D-26's
  0.023 mean understated it because it was measured mid-event. **Flip before
  Beijing, not during the final.** Debutant 52.0 default (D-27) still
  unresolved, 2 players in the field. `tune_scratch/size_w3_layoff_fix.py`.
- **D-47 (2026-09-10)** — W3 flipped ON at 00:05 AEST, Pete's call: Alcaraz
  was out and the page is unannounced. Debutants now carry
  `weeks_since_last = NaN` into the model instead of 52.0, and the training
  table keeps debut rows (1,649 in 1995-2023; debutants win 20%) so XGBoost
  learns the missing-value direction rather than taking an unlearned default.
  On its own the debut change was a no-op: the rows were re-dropped by the
  playstyle merge. That led to D-48.
- **D-48 (2026-09-10)** — **29% of tour matches were never in the training
  table.** `hierarchical_model.py`'s eval export was gated on `has_points`
  because it was written to SCORE the point model, and `train_best_model.py`
  reads the same export to build XGBoost's features. Plus 6% more dropped for
  missing serve/return form. Pete had not been told; nobody had thought of it
  as a filter. With `DSV_KEEP_ALL=1` (now default) every tour row is exported
  and missing values stay NaN: 153,387 training rows instead of 102,351, and
  on the common 2022-23 rows the retrained model is -0.0013 log-loss (paired
  t -3.1, n=9,711), the largest real gain of the two sessions. The eval set
  is now the full 10,891 tour rows, not the 9,711 with stats.
- **D-49 (2026-09-10)** — Pete's rust feature: `recent_load_diff`, the
  difference in each player's sum over prior matches of exp(-days_ago/21),
  every level of event, decayed inside a draw by exp(-2k/21) at round k so
  only the pre-event gap survives. Alcaraz minus Zverev at this US Open:
  2.7, 2.5, 2.3, 2.1 over rounds one to four. Flat in aggregate (+0.4 t vs
  keep-all) and -0.003 on the 638 rows where one player had not played in
  two weeks (t -1.7). Shipped with D-48 as the same retrain, on the mechanism
  and the slice, not the aggregate. `tune_scratch/rust_feature_example.py`.
- **D-50 (2026-09-10)** — Second cutover at 00:05 AEST: bag retrained on
  164,278 rows with keep-all, debuts, rust, the surface fix and scale 0.0,
  swapped in under the canonical name (previous bag kept as
  `.LEGACY_2026-09-09b`). The 00:04 task run started inside the swap window
  and failed on the feature mismatch, published nothing, and was replaced by
  a hand run. Live odds for the six players left moved under a point, since
  all of them have played within two weeks.
- **D-51 (2026-09-15)** — The 2026-09-14 08:00 scheduled rescore failed before
  it could settle C vs D: the completed 688-match tape's serve-point win rate
  had drifted from the mid-tournament `SERVE_BASE` constant in
  `inmatch_seeding.py` by 0.0012 (ATP) and 0.0057 (WTA, worse but unreported -
  the check raises on the first tour it hits). Refreshed `SERVE_BASE` from the
  full 106,589-point final tape (atp 0.610918, wta 0.575873) and re-ran.
  Result reverses D-18: on the robustness split (T fit on 41,460 round 1-3
  points, scored on 5,971 round 4-7 points - up from the 7 matches D-18 was
  decided on), arm D now beats arm C on late rounds, ECE 0.0940 vs 0.1015 and
  log-loss 0.54332 vs 0.55129, same 0.694 accuracy. Arm D still fails its own
  anchor D3 in the cross-fit-by-tour form (ATP T=1.095, WTA T=1.365, spread
  0.270 against a 0.15 bar) - the two tours want different temperatures, so a
  single recalibration is not honest either way. **C vs D is open again, not
  settled**; `--live-score` stays off regardless, so nothing published is
  affected. `dropshotdata/raw/usopen/rescore.log`.
- **D-52 (2026-09-15)** — PR #34 was NOT merged; the NEXT-STEPS entry saying it
  was "unblocked, ready to merge" was wrong. Low-effort review caught it: the
  PR's base is `dev`, not `main`, and `dev` has diverged since the branch was
  cut 2026-09-07 - GitHub now reports it `CONFLICTING`. The W3 layoff half of
  the PR is also superseded: D-46/D-47 shipped `DSV_W3_LAYOFF_FIX` (env-var
  gated) directly to `dev` on 2026-09-10, a different mechanism than this
  PR's `round_aware`/`first_round`/`IN_EVENT_WEEKS`, none of which exist in
  current `dev`. The one thing PR #34 has that `dev` still lacks: the D-28
  fatigue/bye fix (`fatigue_diff` hardcoded 0.0 in `xgb_bt_rating.py:429`,
  wrong once Beijing's 96-into-128 draw with byes is simulated). That piece
  needs re-implementing against current `dev`; PR #34 itself should be closed,
  not rebased - too much of it targets code that no longer exists.
- **D-53 (2026-09-15)** — Re-implemented the D-28 fatigue/bye fix fresh against
  current `dev` (not rebased from PR #34's stale branch). `build_round_pred_matrices`
  takes an optional `first_round: dict[str, int] | None`, default `None`;
  when given, `fatigue_diff` becomes `max(k-fr_i,0) - max(k-fr_j,0)` per round
  instead of the flat 0.0 `build_pairwise_features_base` sets. Default `None`
  means every existing caller (including the locked `simulate_us_open_2026.py`
  and `simulate_usopen_live.py`, both 128 draws with no byes) is untouched by
  construction - no `round_aware` flag needed this time, unlike PR #34's
  approach. `simulate_wta_event.py` now computes `first_round` from
  `seed_map`/`n_byes` (seed <= n_byes has a bye) only when `n_seeds >= n_byes`,
  same guard as PR #34 had, for the same reason: below that (today's Beijing
  field, 76/96 entrants, 0 seeds) `random_bracket_with_byes` redraws the
  leftover byes every simulation, so no fixed uid->round map is honest.
  8 anchors in `tune_scratch/test_w3_fatigue_bye.py`, all passing; a bye is a
  permanent one-match deficit for the rest of the draw, not something that
  "catches up" (differs from my first draft of the test, which assumed the
  latter). Not committed; PR #34 still open, recommended to close.

