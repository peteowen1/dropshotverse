# Methodology critique: dropshotverse (2026-09-07)

Reviewer: Claude Fable 5.1, medium effort, 20-minute time box. Status: complete
(W5-W7 sized less carefully than W1-W4; see "Not reached").

## What the system does now (from the code)

1. An online Elo replays the frozen Sackmann archive plus a live TML bridge for
   the current season. Everyone starts at 1500; K decays with matches played
   (`K_BASE/(n+5)^0.4`, floor 16, `dropshotdata/scripts/elo_baseline.py:29,186`);
   a layoff term shrinks a rating toward 1500 during long gaps. Retirements,
   walkovers and defaults are dropped before the replay (`elo_baseline.py:161`).
2. A hierarchical serve/return point model (`hierarchical_model.py`, a
   Kalman-style online filter) runs alongside; its posterior means and
   variances and its own match probability are exported as features.
3. The shipped predictor is a 5-seed bagged XGBoost over 19 numeric diffs
   (overall/surface/set/game Elo, hierarchical state, weeks since last match,
   prior-match count, head-to-head, age, in-event fatigue, serve/return form)
   plus `surface`, `round`, `tourney_level`, `tour` as categoricals
   (`train_best_model.py:73-83`, `xgb_bt_rating.py:87`). Trained on 1995-2023,
   `max_depth=3`, `eta=0.02`, 2,528 rounds.
4. The tournament simulator freezes every player's features at the draw date
   (`DATE0 = 2026-08-30`, `simulate_usopen_live.py:47`), builds one 128x128
   antisymmetrised pairwise matrix per round (`simulate_tournament_xgb.py:50-85`),
   and re-samples the bracket 20,000 times, treating completed matches as
   known (`simulate_usopen_live.py:79-81`). Matches in progress are simulated
   from scratch; the live score is not used.
5. Pairwise matrices, per-stage probabilities and a field-relative
   Bradley-Terry scalar (`bt_rating`, the row mean of the logit matrix) are
   published to R2 for the ITG blog; a Cloudflare Worker dispatches an Actions
   run when the R2 copy is over 40 minutes old.
6. An in-match win-probability recursion exists and is calibrated on 78,864
   captured US Open points (`inmatch_wp_calibration.py`, ticket 37) but is not
   yet wired into the published projection.

## How accuracy is currently measured (and whether it can be trusted)

- **The headline 0.00073 log-loss gap to Pinnacle is not trustworthy** and the
  repo already says so (DECISIONS D-4, D-10): the 2024-25 holdout was revealed
  at least a dozen times across ticket 28's rounds, and the shipped
  hyperparameters were selected during those rounds (issue 28 lines 35-63,
  "random search ... only `num_boost_round` was ever selected").
- **The one clean number is an upper bound on a different model.** The 2026
  forward test (ticket 36) scored a v1 Elo-only feature set at 0.59638 vs the
  bookmaker average 0.57168 (gap +0.0247, n=2,196), and the model was right
  38.0% of the time when it dissented (CI 32.3-43.8). The shipped config D has
  never been scored on data it did not influence. The forward window is
  Jan-May 2026 only (no grass, no North American hard courts).
- **Screening is on a 2022-23 tune window with 5-fold `xgb.cv`.** Those folds
  are random, not chronological (issue 28 line 38); ticket 36 then showed a
  5/5-fold, t=-5.70 tune-window win (round 14's K) come back +0.000029 out
  of sample. The tune window is now also selection-contaminated.
- **Calibration by slice is thin.** Pre-match: log-loss by tour and ATP
  surface only (`train_best_model.py:377-386`); no reliability bins by
  favourite strength, round or best-of on any clean window. Tournament level:
  stage-reaching log-loss pooled over 16 events (n=2,048 player-stage rows,
  issue 29 line 158), and the hot-vs-cold decision rests on 500 sims per arm.
- **Baselines are honest** (market, Elo alone, seed order all reported), which
  is better than most verses here.
- Net: there is no locked metric with a go/no-go rule that has survived the
  burning of both test windows. Any future claim needs a new, rolling design
  (W2).

## Already known (filed; not re-derived)

- Live projection cut 276 s to 68 s with identical output; Actions backup and
  freshness alarm shipped 2026-09-06.
- Surface Elo updates on the blended `p_w`, so at `blend=0` it is not
  self-correcting (NEXT-STEPS item 6, `elo_baseline.py:409`).
- Hierarchical model tail bug, downgraded (issue #1, D-12).
- Layoff decay 0.3 was selected on the 2024-25 eval window itself
  (reports/elo-baseline.md "Limits").
- Pinnacle column 96% empty in 2026; the 2026 benchmark is the bookmaker
  average (ticket 36).
- The market question is "closed" in the sense of "we do not beat it" (D-5,
  D-6, D-10). W1 below does not reopen it; it uses that conclusion.

## Weaknesses, ranked by expected gain

### W1. The published probabilities ignore the bookmaker price that sits in the repo | expected gain: about 0.02 of the 0.0247 log-loss gap on pre-match, and the tournament projection inherits it | effort: 6-8 h

**Why it costs accuracy.** `raw/odds/` holds tennis-data.co.uk prices for both
tours 2018-2026 (Pinnacle to 2026-01-13, B365 and the average after). They are
used only as a scorecard. The model is 0.0247 log-loss behind the market and
right 38% of the time when it disagrees, which means that wherever the
published number differs from the market, the published number is more likely
wrong. Every downstream product (title odds, pairwise matrix, wall chart) is
built from the weaker of two signals the repo already has. The project goal
(D-5) is "calibration parity with the market"; the shortest path to parity is
to condition on the market.

**What a top shop does.** Treats the closing (or opening) price as a feature
or a prior: either `logit(p) = w * logit(p_market) + (1-w) * logit(p_model)`
with `w` fitted on out-of-time data, or the market logit as an XGBoost input
with the model learning the residual. For a tournament projection, they seed
the pre-tournament pairwise matrix from the first-round prices where they
exist and let the model fill the rest, and they anchor title odds to the
outright market.

**Experiment.** Data: the odds files joined to the 2019-2025 tour-level rows
(the join already exists in `diag2_enriched.parquet`). Fit `w` by year-ahead
walk-forward (train on years < Y, score on Y; 2019-2025 gives seven folds).
Metric: log-loss vs the bookmaker average on the market-joined rows, and
dissent accuracy. Go: the blend beats the model alone by at least 0.015 on
five of seven yearly folds and is never worse than the market by more than
0.003. No-go: gain under 0.005 (then the market and model are near-collinear
and the work is the R1-price seeding only). Expected effect: 0.015-0.022.
For the live event, first-round prices cover 127 of the 8,128 pairs; the
experiment should also report how much title-odds error the R1 seeding alone
removes (score title odds vs realised over the 16 back-tested events).

**Downstream.** `train_best_model.py` gains an optional market feature;
`build_player_table` / `build_pairwise_features_base` need a per-pair price
lookup that is NaN for unpriced pairs (XGBoost handles NaN natively). The
freshness workflow needs an odds fetch step. The site copy must say prices
are an input.

### W2. There is no repeatable out-of-time measurement left; adopt yearly walk-forward as the standing scorecard | expected gain: none directly, but it is the precondition for every other item | effort: 4 h

**Why it costs accuracy.** Both test windows are spent. The tune window is
random-fold `xgb.cv` (issue 28 line 38), which for sequential ratings lets a
fold see matches after the ones it predicts (the ratings are as-of, but the
booster is fitted on later rows of the same window). Round 14 is the recorded
casualty. Without a locked measurement, W1/W3/W4 cannot be accepted or
rejected honestly.

**What a top shop does.** Expanding-window walk-forward by season: for each
Y in 2019-2025, train on 1995..Y-1, score Y on market-joined rows. Report the
seven yearly numbers and the paired mean delta against the shipped config.
Lock the metric (log-loss vs bookmaker average, plus ECE in ten bins) and the
rule (a candidate ships if it wins on at least five of seven years and the
mean gain exceeds 0.002) before running anything.

**Experiment.** `regen_hier_state` costs about 90 s per window
(`train_best_model.py:184`), so seven windows are about 10 minutes of hier
plus seven XGBoost fits. Write it as `tune_scratch/walk_forward_yearly.py`
and make it the only accepted scorecard. Also re-score config D itself on
2026 through the TML bridge rows (ticket 36 says this widens the sample to
about 3,900 and is cheap).

**Downstream.** Replaces "one holdout reveal" language in the tickets; the
charter in `map.md` gets a new go/no-go paragraph.

### W3. The live simulator freezes layoff, fatigue and form at the draw date for every round | expected gain: roughly 0.005-0.01 stage-reaching log-loss on late rounds; up to several points of title probability for individual players | effort: 3 h

**Why it costs accuracy.** `build_player_table` computes `weeks_since_last`
once at `DATE0` (`xgb_bt_rating.py:285`), and `simulate_tournament_xgb.py:50-52`
states that "features other than `round` are identical across rounds". So a
player who came in off a 19-week injury layoff is still modelled as 19 weeks
rusty in the quarter-final, and in-event fatigue (`fatigue_diff`) is zero for
every pair in every round although the training data has it populated (the
model was trained on real values of both). Ticket 29 measured this on one
case: shrinking Alcaraz's `weeks_since_last` from 19.9 to 2.0 weeks alone
moved his number materially (issue 29 line 361). This is separate from the
hot-vs-cold rating question, which was settled (cold wins) and should stay
settled; it is about feeding the model the feature values it was trained on.

**What a top shop does.** Per-round feature state: after round r, both
players' `weeks_since_last` is 0-0.3 weeks and `fatigue` is r-1 matches (or
the real count from the feed), and `serve_form` can be updated from the
event's own point capture if the shop trusts it. Ratings stay frozen.

**Experiment.** Build the per-round matrices with round-consistent
`weeks_since_last` and `fatigue` (matrices are already built once per round,
so this is a change to the feature rows, not the loop). Score on the 16
back-tested events with the pooled stage-reaching log-loss already in
`simulate_tournament_xgb.py:191-200`, block-bootstrapped by tournament as in
ticket 29. Go: improvement at QF or later that the 2,000-draw bootstrap CI
keeps away from zero. Expected 0.005-0.01 at SF/F where layoff players are
concentrated. No-go: CI spans zero at every stage, in which case remove
`weeks_since_last` and `fatigue_diff` from the sim path to stop feeding
values the model was not trained on.

**Downstream.** `build_round_pred_matrices` takes a per-round override;
published numbers move for a handful of players, mostly returning-from-injury
seeds.

### W4. Matches in progress are simulated as unplayed although the point capture and a calibrated recursion exist | expected gain: large but concentrated; between refreshes, a player two sets up is published near his pre-match number | effort: 5 h

**Why it costs accuracy.** `simulate_from_state` uses only `results` (completed
matches, `usopen_live_draw.py:218-231`). At any moment during a session about
4-8 of the remaining matches are live. Ticket 37 shows the game-level
recursion with tour-average serve probabilities is underconfident but
calibrated within ECE 0.047, and predicts that seeding it with real pre-match
skill will remove most of that. The pre-match skill is exactly the pairwise
matrix the simulator has already built.

**What a top shop does.** Conditions each live match on its current score:
solve for the pair of serve probabilities whose fresh-match `p_match` equals
the model's pre-match probability (holding the sum at the tour average), then
call `p_match_from_state(sets, games, server)` for that pair.

**Experiment.** Data: `usopen_2026_points.parquet` (80,690 points, 492
matches) plus the model's pre-match number for those matches (needs the
name-to-uid crosswalk that ticket 37 says is missing; the live draw already
builds `name2uid`, `usopen_live_draw.py:117`). Metric: ECE and log-loss per
point, against ticket 37's arm A (0.0471 / 0.56916). Go: ECE under 0.03 and
log-loss under 0.55. Then wire it into `simulate_from_state` for matches with
`status` in progress.

**Downstream.** Live projection consumes the capture files (already on the
laptop; the Actions backup would need them in R2 or would fall back to
pre-match). The freshness cadence (30 min) becomes the binding constraint, and
the WTA live API (issue #2) becomes worth finishing.

### W5. New players and low-tier form enter the tour-level Elo at 1500 with no prior | expected gain: 0.002-0.005 log-loss on first-year and qualifier rows only; small overall, sizeable at slams where 16 qualifiers enter | effort: 4 h

**Why it costs accuracy.** `build_player_table` defaults an unrated entrant to
1500 for all four Elo variants (`xgb_bt_rating.py:279-282`) and the replay
starts everyone at `START = 1500`. The archive holds 523,355 ATP futures and
398,856 WTA ITF/challenger rows (`manifest/row_counts_by_tour_level.csv`)
that the hierarchical model uses only as a cold-start prior (comment at
`hierarchical_model.py:167`). Since `n_prior_diff` is a feature, XGBoost can
learn "unknown player is weak", but not how weak, and a qualifier who has
just won three matches against rated players on site carries none of it.

**What a top shop does.** Starts a new tour player at a level fitted to their
lower-tier record (a simple regression of first-year tour Elo on futures/
challenger Elo and ranking), and includes qualifying rounds in the replay so
a qualifier arrives with three rated results.

**Experiment.** Walk-forward (W2) restricted to rows where either player has
fewer than 20 tour matches. Compare 1500-start against a challenger-fitted
start. Go: 0.005 on that slice on five of seven years. Expected 0.002-0.005
on the slice, under 0.001 overall, so this is a correctness fix for the slam
draws rather than a project.

**Downstream.** `elo_raw_ratings_for_xgb.py` and the prior path in the hier
model; qualifier title odds change from near-zero to small.

### W6. Retirements, walkovers and the Bo3/Bo5 split are excluded from the target but priced by the product | expected gain: under 0.002 log-loss; a correctness note | effort: 2 h

`elo_baseline.py:161` drops every RET/W/O/DEF row before the replay, and the
XGBoost target comes from the same completed-match population. Roughly 3-4%
of tour matches end in retirement, and the published match probability is a
probability of advancing, which includes them. The bias is toward the
favourite (retirements are more often the underdog). Experiment: score the
model against advance-or-not including retirements on the walk-forward
folds; if the log-loss gap between the two labels exceeds 0.002 add a
retirement hazard from `weeks_since_last` and age. Otherwise note and close.

### W7. Serve and return form are masked to zero for ATP off grass by a rule chosen on the burnt holdout | expected gain: unknown; ranked low because it may be zero | effort: 1 h inside W2

`merge_playstyle` zeroes the form features for ATP Hard and Clay
(`train_best_model.py:262-264`) because round 15 found signal only on ATP
grass. That is a slice-specific rule discovered on the 2024-25 holdout, with
grass being the smallest surface (about 8% of ATP rows). The forward test
excluded grass entirely, so the rule has never been checked out of sample.
Add "mask on / mask off" as an arm of the W2 walk-forward; keep whichever
wins on five of seven years.

## What this verse does better than the others

- It measures against the market at all: every model here carries a Pinnacle
  or bookmaker-average number beside it, and the dissent-accuracy check
  ("right 38% of the time when we disagree") is the cleanest edge test in the
  ecosystem. torp, panna and bouncer report against naive or in-house
  baselines only.
- Anchors gate the write (`simulate_usopen_live.py:178-201`): an eliminated
  player with title probability, a title sum not equal to 1, or a live-count
  mismatch stops publication. Worth copying into torp's and panna's publish
  steps.
- The research record is honest about burnt holdouts and selection bias in
  writing (D-10, ticket 36's round-14 post-mortem), which is rarer than it
  should be.

## Not reached in the time box

- Whether `tourney_level` and `round` as categoricals leak future round
  structure into training (they should not, but a check on RR/BR handling was
  not done).
- Sizing W5-W7 from data rather than from proportions; they are ranked by
  mechanism only.
- The between-events `bt_rating` (row mean of a field-relative logit matrix)
  is not comparable across fields; whether the daily ratings page uses it as
  if it were was not checked.
- The hierarchical model's fixed surface-offset prior and hand-set process
  variances (`hierarchical_model.py:17-25`) were not re-examined; D-12 says
  the shipped model needs the feature, so this stays an upside experiment.
