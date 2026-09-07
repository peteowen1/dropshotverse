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
- **D-29 (2026-09-07)** — Serve and return ratings exist and have never been
  published. The hierarchical model carries a per-player serve skill (`m_a`)
  and return skill (`m_b`) with uncertainties; `ratings.parquet` ships only
  elo_overall, bt_rating and age. `scripts/serve_return_ratings.py` converts
  them with the model's own arithmetic - `sigmoid(e_server - f_returner)`
  against a tour-average opponent - so no new modelling is invented.
- **D-30 (2026-09-07)** — **Publish the ORDER, not the percentages.** Validated
  out of sample on the 2026 US Open tape from skills frozen before it started:
  the ratings order players correctly (corr +0.478 ATP, +0.591 WTA on serve
  points actually won) but are badly OVER-DISPERSED. Against the observed
  spread with binomial sampling noise removed, rated spread is **1.96x** too
  wide on ATP and **4.76x** on WTA. Hold percentage is worse because `p_game`
  amplifies: Shevchenko was rated to hold 37.4% and held 72.0%; Shelton was
  rated 95.6% and held 88.3%. So ranks or percentiles are publishable now;
  percentages and hold/break are not, until a shrink is fitted and validated on
  a genuinely future event - Beijing, 2026-09-30.
- **D-31 (2026-09-07)** — The first version of that validation tested BIAS on
  hold percentage and PASSED at +0.022 while individual players were out by 35
  points. Over-dispersion is too harsh at one end and too generous at the
  other, so the mean error cancels exactly. The anchor now measures dispersion
  with sampling noise removed from the observed side, and fails. Same shape as
  D-24: a check that cannot fail is worse than no check.
- **D-32 (2026-09-07)** — The two tours cannot share an axis. They are fit
  separately, so WTA serve skills read HIGHER than ATP ones while women win
  fewer serve points (57.0% against 60.9% on the same tape). Every published
  number must be relative to its own tour. Related: serve and return correlate
  at +0.342 (ATP) but +0.571 (WTA), so on the women's side the two numbers
  substantially restate overall quality rather than separating two skills.

