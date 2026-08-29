# 29 — Tournament draw simulation

Type: plan
Status: planned (not yet built)

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
