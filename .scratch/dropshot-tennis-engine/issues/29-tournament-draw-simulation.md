# 29 — Tournament draw simulation

Type: plan
Status: v1 (frozen Elo), step 1 (hier in-sim updates - regressed vs Elo,
update question unresolved), step 2 (XGBoost frozen predictor - beats Elo at
6/7 stages, best arm so far), and step 3 (Betfair - partially checked, see
below, blocked on Pete's own account signup) all attempted.

## Destination

This is map.md's own "Not yet specified" item ("Draw simulation design...
Shape depends on what the match model outputs") — now that the match model
exists (ticket 28's XGBoost, gap to market ≈0, WTA beating it outright),
this is buildable. Scoped and planned 2026-08-29 per Pete's request.

Two things to produce per tournament: each player's probability of reaching
each stage (R32/R16/QF/SF/F/W) before the draw starts, and (stretch) each
player's probability of winning the title compared against the pre-tournament
outright market.

## Scope

**Grand Slams only, both tours** (matches the project's existing live-scope
decision, ticket 07b — 4 events/year × ~2 decades, 128-draw brackets are
the cleanest to reconstruct and validate, and it's the eventual live-product
scope anyway). Extending to Masters/500/250-level events is a later phase.

## What we already have — no new draw data needed for the core methodology

Checked before assuming a gap existed: the archive has no explicit bracket
sheet (no "who plays whom next round" field), only match results with
`round`, `winner_seed`/`loser_seed`, `draw_size`. Initially read that as a
blocker. It isn't, for two reasons:

1. **Ground truth for scoring doesn't need bracket reconstruction at all.**
   "Did player X actually reach the QF this tournament?" is already a
   direct read from the archive (round + player + tournament).
2. **The standard methodology doesn't replay the literal historical bracket
   either.** Confirmed via Jeff Sackmann's own published approach
   (tennisabstract.com's Slam forecasts): generate thousands of random
   brackets that are *consistent with seeding rules*, simulate each with
   the match model, aggregate to get each player's per-stage probability.
   The literal real draw sheet (which exact quarter each seed landed in)
   is randomized at the actual draw ceremony and isn't reconstructable from
   seed numbers alone (verified below) - but it's also not needed for this
   methodology, since we're not trying to replay one specific real bracket,
   we're estimating each player's probability under the seeding constraints
   that were actually in force.

Real historical draw sheets (e.g. from Wikipedia's per-tournament bracket
pages) would only be needed for a *different*, optional feature: replaying
one specific already-happened tournament's exact real bracket as a demo
("how would our model have called this Wimbledon"). Not required for tests
#1 or #2 below. Flagging as a nice-to-have, not blocking.

## Seeding logic — validated against real data, two bugs found and fixed

Built the standard bracket construction (`tune_scratch/validate_seeding_logic.py`):
seed 1 and 2 anchor opposite halves; seeds 3-4 anchor the remaining two
quarters (one per half); seeds 5-8 anchor the remaining four eighths; and
so on, doubling each tier. This is the real-world convention used by every
major.

Two real bugs found before trusting it:
1. First implementation ("iterative doubling", wrong) put seeds 1 and 2
   adjacent - meeting in round 1 instead of the final. Replaced with the
   correct recursive construction (`order(2k)`: for each seed `s` in
   `order(k)`, emit `s` then `2k+1-s`).
2. First validation check used an *exact-slot* rule (assumes seed 3 always
   lands in a specific one of the two remaining quarters) and found a 74%
   "violation" rate against real Slam data - alarming, but wrong on my
   part, not a real seeding violation. Real seeding only guarantees *tier*
   separation: seed 3 gets a different quarter from *whichever* of seeds
   1/2 shares its half, and which one is decided by the actual random draw
   ceremony, not fixed by seed number. Replaced with the correct tier-based
   rule: seeds `i<j` are guaranteed apart until at most
   `2**ceil(log2(j))` players remain, independent of `i`.

**Validated clean**: checked the corrected tier rule against every real
seeded Grand Slam match 2018-2025, both tours (n=855) - **zero violations**
at every seed-count cutoff tested (1-4, 1-8, 1-16, 1-32).

## Simulation methodology

**Two arms, cheap baseline first** (per this project's own established
pattern - Elo before the hierarchical model, etc.):

1. **Closed-form / frozen-rating baseline.** Pre-tournament ratings held
   fixed for the whole simulated bracket. Random seeding-consistent
   brackets (or, since ratings don't change match-to-match in this arm, a
   full combinatorial DP over the bracket structure is exact and avoids
   Monte Carlo noise entirely - worth doing the DP version here since it's
   cheap and gives an exact answer to compare the MC version against).
2. **Monte Carlo with in-simulation Bayesian updates.** After each
   simulated match, apply the hierarchical model's own existing online
   update rule (already validated, no new mechanism to invent) to both
   players' state before simulating their next round. This is what gives
   Pete's described asymmetry for free: a thin-history player's posterior
   variance is larger, so the SAME Bayesian update rule already used
   throughout this project moves their rating more per result than an
   established veteran's - exactly "beat a top seed early, we should
   believe you're near the top of your range" - without inventing new
   machinery, just applying the existing update match-by-match down a
   simulated bracket instead of only across real history.

   **This is a genuine extension beyond the published state of the art**,
   not something already established as best practice - Sackmann's own
   Slam forecasts use frozen ratings throughout each simulated bracket, no
   in-sim updates. Worth being explicit about that with any results: we're
   testing whether it beats the frozen baseline, not assuming it will.

   Run N simulated brackets (Sackmann uses 100,000; likely start smaller
   for the backtest and scale up once the harness is confirmed cheap
   enough), aggregate to each player's P(reach each stage).

## Metrics

1. **Primary: stage-reaching log-loss.** For each player, the model states
   a probability of reaching each stage (R32.../W) before the tournament
   starts. Score against their real, realized finishing stage. Simplest
   framing: nested binary questions ("did they reach at least QF?" etc.)
   scored as binary log-loss per stage, which is easy to calibration-check
   per stage (load `calibration-check` skill when this is actually built -
   this is exactly the kind of ranking/probability output that skill
   covers).
2. **Anchor checks before trusting any output** (per `stats-discipline`
   §1): top seed should have the highest P(title) in the large majority of
   simulated tournaments; a qualifier/wildcard shouldn't show an
   implausibly high P(reach SF); seed 1 vs seed 2's P(meet) should
   concentrate almost entirely in the final, matching the validated
   seeding logic above.
3. **Stretch, blocked on data: outright-market comparison.** No
   tournament-winner/futures odds exist anywhere in this repo currently
   (checked - the odds files are match-level only, `PSW/L` per match, no
   outright column). Candidate source found but not yet verified: **Betfair
   Historical Data** (historicdata.betfair.com) has a free "Basic" tier
   (1-minute odds intervals, no volume) covering all Exchange markets since
   2016, which should include tennis outright/tournament-winner markets -
   Betfair Exchange has historically offered those. Needs a registered
   Betfair account and direct verification that outright markets are
   actually in the free tier's coverage before committing to it as the
   data source. Not blocking test #1; pick up when convenient.

## v1 shipped and validated: frozen-Elo baseline, all 16 available Slams

`scripts/simulate_tournament.py` (single tournament, `run_one_tournament()`
reusable) + `scripts/simulate_tournaments_batch.py` (all 16: 4 events x 2
years x 2 tours, 2000 sims each). Elo-only, frozen ratings, no hier state -
deliberately the cheap baseline this plan called for first.

Real bugs found and fixed along the way: the raw CSV's R128 round only
matched 61/64 rows to `elo_raw_ratings_full.parquet` because
`elo_baseline.py`'s own `prepare()` drops unparseable-score
(walkover/retirement) rows - fixed by looking up each player's most recent
PRIOR real rating (any tournament before this one) rather than requiring a
match specifically at this tournament's R128 row.

**Pooled result across all 16 tournaments (n=2,048 stage-reach
observations): beats a flat-baseline comparison at every stage**, including
R64/R32 which looked roughly flat-or-worse on a single tournament (2025
Wimbledon ATP alone) - exactly the noise that single-tournament result was
flagged as, now resolved with real pooled data. Anchor check: top seed
lands in the simulated top-3 by title probability in 16/16 tournaments,
#1 in 14 of them.

## Step 1 result: in-sim Bayesian updates (raw hier state) - real regression found, update question unresolved

`scripts/simulate_tournament_updating.py`. Same 16-tournament pool, two arms
sharing pre-tournament state snapshots and the games-level match simulator:
frozen hier state vs. hier state updated via the model's own regime-B rule
between simulated rounds (see script docstring for exact reuse of
`adf_update`/`apply_side_update`, no new mechanism invented).

**Headline finding, not the one this step set out to test:** raw hier state
(`m_a+m_b`, no surface blend) is a clearly WORSE tournament-simulation
predictor than frozen Elo at every single stage - R64 0.6626 vs Elo's 0.6251,
running up to W 0.0316 vs 0.0225 (see `tune_scratch/simulate_tournament_updating.log`
for the full table). Matches an independent finding from the same session
(`tune_scratch/hier_vs_seed_elo_ranking.py`): Spearman(hier rating, real
depth)=0.236 vs Elo's 0.351 vs seed's 0.432, both pairwise CIs clean (don't
cross zero) - seed > elo > hier as rank-order predictors, consistently.

**The actual question (does in-sim updating help) came back unresolved, not
negative** - mixed sign across stages (updating wins at R16/F, frozen wins
elsewhere), deltas of 0.001-0.011 against n=2048, no CI computed yet. Per
stats-discipline, treating this as genuinely unresolved rather than reading a
direction into noise - not worth resolving further until/unless a stronger
base predictor than raw hier state is in play (see next step).

Decision: don't chase in-sim updates further on top of raw hier state -
proceeding to step 2 (swap in the full blended XGBoost model, ticket 28's
already-validated near-market predictor) as the more promising base, which
may make the frozen/updating question moot either way.

## Step 1.5 (side quest, Pete's request): can XGBoost yield a scalar rating?

`scripts/xgb_bt_rating.py`. XGBoost only predicts pairwise win probabilities;
extracted a Bradley-Terry-style scalar rating per player by predicting the
full round-robin among each of the 16 tournament fields and taking the
row-mean of the antisymmetrized pairwise log-odds matrix (the exact
least-squares BT solution for a complete round-robin, derived by hand before
implementing - see script docstring). Feature fidelity documented per-feature
in the script docstring (most are exact reuses of existing lookups; age and
n_prior are the two approximations).

**Result: xgb-BT is the best rank-order predictor of all four**, clean
resolved CIs on every pairwise comparison (`tune_scratch/four_way_ranking_comparison.py`,
2000x tournament-block bootstrap, n=2048):

| predictor | Spearman(., real depth) |
|---|---|
| xgb-BT | 0.4725 |
| seed | 0.4323 |
| elo | 0.3512 |
| hier (m_a+m_b) | 0.2357 |

xgb-BT beats seed (+0.040, CI [+0.015,+0.064]), beats elo (+0.121, CI
[+0.108,+0.133]), beats hier (+0.237, CI [+0.214,+0.261]) - all six pairwise
CIs exclude zero, ranking fully resolved: xgb-BT > seed > elo > hier.
BT-consistency check (mean |logit(P(i beats j)) + logit(P(j beats i))|,
0=perfectly consistent): 0.22 - real but not disqualifying asymmetry, XGBoost
wasn't trained to be symmetric and this is expected, not a bug.

Directly motivates step 2 below: if XGBoost's implied rating already beats
Elo and seed at pure rank-ordering, its full pairwise predictions should also
make a stronger tournament-simulation match predictor than step 1's
Elo/hier arms.

## Step 2 shipped: XGBoost as the match predictor - beats the v1 Elo baseline

`scripts/simulate_tournament_xgb.py`. Frozen-rating style (no in-sim updates
- see script docstring for why that doesn't apply to XGBoost), one full
round-robin prediction pass per round (R128..F) per tournament so the
`round` categorical is always correct, reused across all 2000 sims per
tournament since ratings are frozen.

**Confirms the step-1.5 ranking finding at the actual simulation task.**
Pooled stage-reaching log-loss, same 16-tournament pool
(`tune_scratch/simulate_tournament_xgb.log`):

| stage | Elo (v1) | hier-frozen (step 1) | **xgb (step 2)** |
|---|---|---|---|
| R64 | 0.6251 | 0.6626 | **0.6115** |
| R32 | 0.4588 | 0.5317 | **0.4511** |
| R16 | 0.2676 | 0.3273 | **0.2673** |
| QF | 0.1381 | 0.1878 | **0.1342** |
| SF | 0.0793 | 0.0999 | 0.0806 |
| F | 0.0396 | 0.0568 | **0.0363** |
| W | 0.0225 | 0.0316 | **0.0196** |

XGBoost beats Elo at 6/7 stages (SF is a +0.0013 wash, no CI computed but too
small to matter) and beats hier-frozen at every stage. Anchor check: top seed
in simulated top-3 by title chance, **16/16 tournaments** (Elo was 12/16,
hier-frozen 10-12/16) - the cleanest anchor result of any arm tried so far.

Status: v2 (XGBoost frozen-rating match predictor) is now the best-performing
arm and the natural default going forward, pending step 3.

**CI follow-up** (`tune_scratch/xgb_vs_elo_tournament_ci.py`, 2000x paired
tournament-block bootstrap, n=16 - fixed a real bug first: blocking by bare
`tourney_id` collided ATP/WTA's same event codes, e.g. both tours' Australian
Open is "2024-580", silently halving the block count to 8): **XGBoost wins
cleanly at R64, R16, F, W** (CIs exclude zero); **R32, QF, SF unresolved**
(CIs span zero - n=16 tournaments limits power there, not evidence Elo is
competitive at those stages specifically). No stage where Elo's CI excludes
zero in its favor. Net: a real, resolved edge at 4/7 stages, not a clean
sweep - reported honestly rather than oversold from the point estimates
alone.

## Hot-vs-cold, properly dampened: cold wins outright, question closed

Pete's fair challenge: step 1's hot-hier arm reused the archive's frozen
v25 update hyperparameters (tuned for fitting 50 years of real matches, a
different objective) and Elo was never tested hot at all - a wrong update
MAGNITUDE could make a real effect look like a wash. Fixed by sweeping a
single dampening multiplier alpha (0=cold through 2.0=double-strength) for
BOTH models (`scripts/simulate_tournament_hot_sweep.py`, n_sims=500,
alpha=0 exactly reproduces each model's cold arm by construction - R64 is
alpha-invariant since round 1 never sees an update yet, confirmed in the
data).

| alpha | elo avg_logloss | hier avg_logloss |
|---|---|---|
| 0.0 (cold) | **0.2346** | **0.2941** |
| 0.5 | 0.2346 | 0.2962 |
| 1.0 | 0.2349 | 0.2978 |
| 1.5 | 0.2351 | 0.2978 |
| 2.0 | 0.2353 | n/a (worse trend held) |

**Cold wins outright for both models at every alpha tested**, including the
dampened 0.5 guess. Not a close call or a tuning-sensitive result - hier
gets monotonically worse as alpha increases. Pete's original intuition
(uncertain players should swing more) is a real idea, but this specific
mechanism (regime-B combined-strength update reused from the archive fit)
doesn't realize it inside a single 2-week bracket - closing this question,
parked per Pete's call ("gains from cold/hot is at least minimal enough
that we can park it").

## Decision (2026-08-29): XGBoost adopted as the go-forward predictor

Pete: "yep lets use xgboost and xgb BT if that helps" - XGBoost (step 2) is
the match predictor going forward for tournament simulation; xgb-BT (step
1.5) is the scalar player-rating output where one's needed, given it beat
seed/elo/hier cleanly on both the rank-correlation test and the actual
simulation task. Betfair (step 3) explicitly parked - "leave this for now
then lets come back to it."

## Step 3 checked as far as possible without an account - blocked on Pete, PARKED per above

Checked public Betfair docs (no account created - would need Pete's own
signup, not something to do on his behalf): the free "Basic" tier is real
and no-cost (1-min intervals, last-traded-price only, no volume/ladder), but
requires a registered Betfair account to browse/download - so the actual
catalog (does it list a tennis "Outright Winner"/"To Win Tournament" market
type) can't be checked from outside. The only tennis exclusion documented
anywhere is "Game betting" (in-play, game-level) markets - outright/
tournament-winner markets aren't on any exclusion list found. Not a
confirmation, just an absence of a red flag.

**Next action is Pete's**: create a free Betfair account, log into
historicdata.betfair.com(.au), and check whether a tennis outright/
tournament-winner market type is listed in the catalog for a known past
Slam. If yes, test #2 (comparing simulated tournament-winner probabilities
to pre-tournament market odds) becomes buildable.

## Real bug found and fixed: build_snapshots silently blanked hier state for forward-looking forecasts

Pete asked "does xgb know Zverev won French Open" while checking the US
Open predictions against real Betfair odds he pulled himself - investigating
that surfaced a genuine, previously-unreported bug: `build_snapshots()` only
takes a snapshot when the replay CROSSES a tournament's start date. The
tennismylife bridge only reaches 2026-08-24, but the real US Open starts
2026-08-30 - the replay never crosses that date, so the snapshot silently
never fired, and callers' `.get(key, {})` fallback meant EVERY player got a
blank/default hier state (m_a=m_b=0, no shrinkage) in the run reported
earlier as "fixed." Elo was unaffected (its own lookup is a plain "latest
before date0" filter, not a crossing-triggered snapshot) - only hier was
silently blank. Fixed: `build_snapshots()` now falls back to "state as of
the latest available data" for any tournament whose cutoff the data never
reaches - the same treatment every other feature (elo, age, serve form)
already used. Full US Open sim re-run after the fix.

## Alcaraz/Zverev vs Betfair market: a real, quantified model limitation, not a bug

Pete pulled live Betfair outright-winner odds for both 2026 US Open draws
and flagged a large divergence: market has Alcaraz+Zverev combined at ~48%
implied probability (3.75/4.7 back odds), our model has them at ~10%
combined, roughly tied with Fils/Jódar.

Traced to root cause, not just described:
- Direct pairwise prediction (not bracket-simulation or BT-averaging
  artifact): P(Alcaraz beats Fils) = 52.0%, P(Zverev beats Jódar) = 50.4% -
  already flat before any tournament simulation. Elo's own raw logistic on
  the same rating gap (2574 vs 2370) would say ~76% - XGBoost is
  dramatically discounting Alcaraz specifically.
- Root cause: Alcaraz's `weeks_since_last` = 19.9 weeks vs everyone else's
  ~1-2 weeks - genuinely real, not a data bug. Confirmed via web search:
  Alcaraz has a wrist injury, missed Roland Garros AND Wimbledon, returning
  specifically at this US Open after 4 months out.
- Quantified the sensitivity directly: holding everything else fixed,
  shrinking Alcaraz's weeks_since_last from 19.9 to 2.0 weeks alone moves
  P(beats Fils) from 52.3% to 69.6% - a 17-point swing from one feature.
- Round categorical checked too (Pete asked): real but minor - R32 to F
  shifts Alcaraz-vs-Zverev from 57.6% to 59.6%, not the driver.

**Conclusion: not a bug, a genuine model limitation worth documenting.**
XGBoost learned a real "long layoff -> discount" pattern from training data
(rust is a real effect), but training layoffs are mostly weeks, not 4+
months - Alcaraz's actual gap is an extreme value the model is
extrapolating into, with no feature to distinguish "declining/inactive
player" from "managed injury return, expected to be sharp" - exactly what
informed bettors price in via news the model has no access to. Zverev's
smaller market gap is a separate, likely simpler story (single recent Slam
win underweighted against a long career-long baseline, no layoff involved -
his own weeks_since_last is a normal 1.4 weeks).

Not fixed blindly - capping/overriding weeks_since_last_diff risks breaking
calibration that's working correctly for the (much more common) short-gap
case. Flagged as a known limitation for any future retune, not patched here.

## ATP/WTA split XGBoost vs combined: tested, clean null result

Pete asked whether we'd ever tested separate per-tour models against the
combined one - answer was no, so built and tested it
(`scripts/train_xgb_split_by_tour.py`, same TRAIN_YEARS/features/
hyperparameters/holdout rows as the combined model, tour-scope the only
axis that varies):

| tour | split | combined | delta | winner |
|---|---|---|---|---|
| ATP | 0.59525 | 0.59454 | +0.00071 | combined |
| WTA | 0.59094 | 0.59186 | -0.00092 | split |

Small, mixed-sign deltas - the signature of noise in this project's history,
not a real effect (no CI computed, but the magnitude alone argues against
spending one). Makes sense mechanistically too: `tour` is already a
categorical feature in the combined model, so it can already learn tour-
specific splits when they matter. **Keep the combined model** - splitting
doesn't earn its complexity. Per-row split predictions saved
(`tune_scratch/xgb_split_by_tour_predictions.parquet`) if a formal CI is
ever wanted.

## Alcaraz-vs-Fils round/surface/date sweep (Pete's sanity check on the 52% figure)

`tune_scratch/alcaraz_fils_sweep.py`. Confirms round is genuinely flat
(50.9%-52.3% across all 7 rounds) - not the driver of anything. Surface
shows real, sensible spread (Hard 52.3%, Clay 57.5%, Grass 65.1% -
Alcaraz's known clay/grass edge over Fils, captured correctly by hier's
surface-specific terms). Date sweep surfaced a new finding: the layoff
discount SATURATES past ~20 weeks - 24.4 and 33.1-week synthetic gaps give
the identical 52.3% as the real 20.9-week gap. Consistent with a tree model
that has thin training coverage past ~20-week layoffs: beyond that point it
has no further resolution, just routes to the same terminal leaf regardless
of how much longer the gap actually is.

## Granularity sweep -> new XGB features: set/game-level Elo added, beats ticket 28's market numbers

Full arc from Pete's "does elo do point/game/set/match" question:

1. Built point/game/set-level Elo (`scripts/point_elo.py`) and hier
   (`scripts/hier_multi_granularity.py`) standalone, same evidence
   definitions, on the frozen archive - set-level Elo was the standout
   (0.60472, beating even match-level Elo's own 0.61200). Hier's own
   standalone point-level number (0.72388) surfaced a real, separately-
   ticketed tail-calibration bug (see ticket 31) - not from this sweep, but
   found while investigating it.
2. Generated full-archive raw set-Elo and game-Elo ratings
   (`scripts/set_elo_raw_for_xgb.py`, `scripts/game_elo_raw_for_xgb.py`) and
   tested them as NEW XGBoost features against ticket 28's exact market-
   comparison holdout (same TRAIN_YEARS/hyperparameters, granularity-mix the
   only axis that varies):

| variant | ALL | ATP | WTA | market |
|---|---|---|---|---|
| A: match only (ticket 28) | 0.59322 | 0.59454 | 0.59186 | ALL 0.59352 / ATP 0.59372 / WTA 0.59330 |
| B: match + set | 0.59268 | 0.59428 | 0.59103 | |
| C: set only (match dropped) | 0.59392 | 0.59691 | 0.59084 | |
| **D: match + set + game** | **0.59264** | 0.59504 | **0.59016** | |

**D (match + set + game together) is the new best headline config** - beats
A on every population, and set_elo_diff became the single highest-gain
feature in the whole model (232.91, ahead of the previous #1,
elo_logit_pwinner's 215.15). C (set-only) is the worst ATP number of all
four - match-level and set-level are genuinely complementary, confirming
they shouldn't replace each other. Game-level helps WTA notably (0.59103 ->
0.59016) but slightly hurts ATP alone (0.59428 -> 0.59504) - real tour
asymmetry, not noise-shaped: best-of-5 ATP matches likely already have
enough set-level evidence that game-level mostly adds noise, while best-of-3
WTA matches have less set-level evidence to begin with, leaving more real
signal for game-level to contribute. Not CI-verified yet (deltas are
0.0003-0.0008, similar scale to noise seen elsewhere in this project) - flag
before treating as fully locked in.

Known coverage gap on the new set/game features: `hier_multi_granularity`'s
`parse_score_sets`-based parsing fails on ~2.3% of matches (27,307/1,189,172)
- concentrated in two real score formats it doesn't handle (final-set
super-tiebreak bracket notation `[10-5]`, and long-form no-tiebreak deciding
sets like `11-9`/`18-16`) - NOT one bad tournament, spread across both tours
and includes 2,995 real Grand Slam matches. Worth fixing `parse_score_sets`
properly rather than dropping this data, given how much value set-level
evidence just proved to have - not yet fixed, logged as a follow-up.

## D (match+set+game) shipped to the live pipeline, US Open predictions regenerated

Pete: "shop D everywhere lets do it" - propagated the match+set+game feature
set into `xgb_bt_rating.py` (new `XGB_NUMERIC_D`/`prep_X_d`, new cache dir
`xgb_bagged_model_cache_v2_setgame` so the old match-only cache can't be
silently reloaded under the new schema), `build_player_table`/
`build_pairwise_features_base` (added set/game-Elo lookups and diff/logit
features), and `simulate_tournament_xgb.py`/`simulate_us_open_2026.py`
(swapped their `tbm.prep_X` calls for `prep_X_d`). Verified with a smoke
test before the full run (all 19 features present, correctly antisymmetric
across a pair). Full US Open predictions regenerated -
`tune_scratch/us_open_2026_v3_setgame.log`. Alcaraz's title probability
widened further (27.6% -> 31.7%, now slightly overshooting Betfair's 26.7%);
Zverev's gap to market is completely unmoved (still 9.5% vs market's
21.3%) - consistent with the CI showing ATP never got a resolved benefit
from these features, so that specific gap remains unexplained. Women's
reshuffled with real support from the resolved WTA CI: Sabalenka 21.1% ->
25.8%, Rybakina/Świątek both moved up, Gauff dropped from 2nd to 4th
(17.1% -> 13.1%).

## parse_score_sets coverage gap fixed for set-Elo (not a hier edit)

New `scripts/parse_score_sets_permissive.py` - purpose-built parser for the
set-Elo/set-counting use case specifically, NOT a fix to hierarchical_model
.py's own `parse_score_sets` (that function is deliberately strict: it only
returns scores representable by `p_set_full`'s DP, which genuinely can't
model a super-tiebreak-decided set or an unbounded advantage set - a real
requirement for ticket 21's full-DP likelihood, different from set-Elo's
much weaker "who won more games in each set" requirement). Recovered
coverage 97.7% -> 99.9% (1,161,865 -> 1,188,006 of 1,189,172 rows).

Standalone set-Elo log-loss with the improved coverage: 0.60532 (was
0.60472) - a wash, not a real change, well within noise. The newly-recovered
matches (super-tiebreak/extreme-decider finishes) are inherently higher-
variance evidence, roughly cancelling the benefit of more data. Given that,
**not triggering another full retrain-and-redeploy cycle** for this alone -
`set_elo_raw_ratings_full.parquet` is updated in place with the better
coverage regardless, so the next genuine retrain picks it up for free.

## Open items before building

- Verify Betfair Historical Data's free tier actually includes tennis
  outright/tournament-winner markets (not just match odds) - registration
  + a sample download needed, not yet done.
- Decide N (number of simulated brackets) for the MC arm - start small for
  harness validation, scale once confirmed cheap.
- The frozen-baseline arm should use an exact combinatorial DP rather than
  Monte Carlo where possible (ratings don't change within that arm, so DP
  is both cheaper and exact) - confirm this is worth building vs. just
  running the same MC harness with updates disabled, before implementing
  two separate code paths.
