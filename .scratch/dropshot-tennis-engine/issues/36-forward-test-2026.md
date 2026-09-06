# Ticket 36: the 2026 forward test

Run 2026-09-05, after Pete cleared it. The population no decision in this
project has ever touched: 2,654 tour-level completed matches in 2026, of which
2,196 join to bookmaker odds. No 2026 row influenced training, feature
construction, hyperparameters, or the K choice under test.

**Window: 2026-01-04 to 2026-05-25 only** (Jan 535, Feb 657, Mar 523, Apr 398,
May 541). NOT January to August, as this ticket and the session first said.
`elo_compact.build_compact()` reads `raw/github` CSVs, which stop at
2026-06-02; the 1,309 tour-level TML bridge rows covering June to 2026-08-24
were never in the cache and so were never in the test.

Worth recording how that hid: the odds file runs to 2026-08-24, so the join
reported 82.7% coverage and looked healthy. **A join coverage percentage
measures the overlap it was given, not the overlap you intended** - it cannot
see rows that never entered the frame at all. Same failure shape as the
Pinnacle column below: the check passed because it was measuring the wrong
thing.

Neither conclusion changes - 2,654 matches over five months is a real test and
both results were decisive - but the window excludes the entire grass swing and
the North American hard-court swing, so it is a clay-and-indoor-heavy sample.
Re-running against the bridge rows would widen it to ~3,900 and is cheap; it
needs `build_compact()` to consume `bridge_tennismylife.load_matches_with_bridge`
rather than the raw CSV glob.

This is the measurement deferred since 2026-08-29, when `RESUME-beat-elo.md`'s
round-11 stop-point flag first recorded that the 2024-25 holdout had been
revealed too many times to be an unbiased estimate of anything.

---

## Result 1: round 14's K change does NOT hold out of sample

| | tune window (2022-23) | 2026, out of sample |
|---|---|---|
| delta vs shipped K | **-0.001393** (5/5 folds, t = -5.70) | **+0.000029** |

A wash. **Round 14 is rejected.**

This is stats-discipline #10 in its exact form: a promising result, re-tested on
new data, resolved in the other direction. Worth being precise about what went
wrong, because the tune-window evidence was not weak or sloppy:

- The paired per-fold test was the RIGHT test. The unpaired version would have
  discarded the effect as 8% of fold noise; pairing correctly removed the
  0.035 between-fold swing that is common to both arms.
- 5/5 folds improving, t = -5.70, was a real and correctly-computed statistic.
- And it still did not transfer, **because the K point was SELECTED on those
  same folds.** The paired test measured how well the chosen point fitted the
  window it was chosen on. Fold consistency does not fix selection on the same
  data; it only removes fold noise, not selection bias.

The ticket-35 caveat ("the magnitude is optimistically biased; 5/5 supports the
direction") turns out to have been too generous. Not just the magnitude - the
direction did not survive either.

## Result 2: we do not beat the market, and the disagreement finding reproduces

n = 2,196, rows never used for anything:

| | log-loss |
|---|---|
| model | 0.59638 |
| market | 0.57168 |
| **gap** | **+0.02470** |

When the model picks a different favourite than the market (276 of 2,196,
12.6%), it is right **38.0%** of the time, 95% CI **[32.3%, 43.8%]** - durably
below chance. Ticket 28 round 7 found 44.6% CI [41.5%, 47.7%] on v2 over
2024-25. Independently reproduced on a different year, a different feature set
and a different bookmaker benchmark, and if anything worse.

**Three caveats, and all three cut AGAINST the model** - none of them is a
reason to think the real gap is smaller:

1. **The benchmark is weaker than Pinnacle.** See the coverage note below. The
   comparison is against the bookmaker AVERAGE, which carries more vig and is
   less sharp than Pinnacle closing, so the gap to the closing line is larger
   than +0.0247, not smaller.
2. **The model is weaker than shipped config D** - Elo features only, no
   hierarchical state, no set/game-level Elo, no age or fatigue. So +0.0247 is
   an UPPER bound on config D's gap, not an estimate of it.
3. Training ran to 2025-12-31 with `num_boost_round` pinned at 217 rather than
   re-tuned, matching the paired-fold test so the arms stay comparable.

Caveat 2 is the one that leaves real work: converting the upper bound into
config D's actual number needs hierarchical state regenerated for 2026.

---

## Coverage failure: Pinnacle is PRESENT and 96% EMPTY in 2026

Reported earlier in the session, twice, as "2026 holds 1,976 ATP and 1,921 WTA
matches with Pinnacle `PSW`/`PSL`". That was reading the column header. Actual:

| column | ATP | WTA | last date |
|---|---|---|---|
| `PSW` | 71 / 1,976 (**3.6%**) | 101 / 1,921 (**5.3%**) | **2026-01-13** |
| `AvgW` | 1,976 (100%) | 1,920 (99.9%) | 2026-08-24 |
| `B365W` | 1,968 (99.6%) | 1,916 (99.7%) | — |

tennis-data.co.uk stopped carrying Pinnacle after mid-January 2026. The first
run of this test silently scored 148 rows and produced a market comparison and a
disagreement rate (27.8%, CI [7.1%, 48.5%]) that meant nothing.

This is `C:\dev\CLAUDE.md`'s own rule - "**Assert COVERAGE, not presence**, on
every column after a join or union" - and it was missed by checking that the
column existed. `load_2026_odds()` now logs per-column coverage before using
anything, and the benchmark is stated as the bookmaker average rather than
implied to be Pinnacle.

**Consequence for the 2024-25 work:** those years' files DO carry Pinnacle
(that is where ticket 28's `market 0.5889` came from), so nothing already
published is affected. But any future statement about 2026 market performance
must say which book, because the answer changes after 2026-01-13.

---

## Where this leaves the programme

Three candidates were taken to a conclusion this session and all three were
rejected:

| round | candidate | died at |
|---|---|---|
| 12 | retirement as a pre-match fitness feature | placebo arm |
| 13 | scoring retirements in Elo | downstream, 3/5 folds |
| 14 | faster-decaying K schedule | 2026 forward test |

Round 14 is the instructive one: it passed every check available on the tune
window, including the correct paired test, and still failed the only test that
was not selected on.

Combined with round 7's disagreement finding - now reproduced on an untouched
year - the accumulated evidence is that **there is no exploitable edge in this
pipeline's available data**. That was ticket 10's conclusion on 2026-08-27,
reopened at Pete's request on 08-29, and it has now survived fourteen rounds of
attempts to overturn it plus one clean out-of-sample test.

Open, and the only remaining question that can move the number: run this same
test on shipped config D, which needs hierarchical state for 2026.
