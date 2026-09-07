# Validation charter: the standing scorecard for dropshotverse match models

Locked 2026-09-07, **before** any candidate was scored under it. Answers W2 of
`docs/reviews/FABLE-critique-2026-09-07.md`. Implementation:
`dropshotdata/tune_scratch/walk_forward_yearly.py`.

## Why this exists

Both previous test windows are spent. The 2022-23 tune window was scored with
random-fold `xgb.cv`, which lets a fold fit on matches played after the ones it
predicts; round 14 is the recorded casualty, passing 5/5 folds at t=-5.70 and
then coming back +0.000029 out of sample. The 2024-25 holdout has been revealed
at least a dozen times (D-10). The 2026 forward test was the one clean shot and
has been fired (issues/36). Until something replaces them, no claim about
W1, W3 or W4 can be accepted or rejected honestly, and F4 and F5 had to be
judged on mechanism because there was no metric to judge them by.

## Charter

| Item | Locked value |
|---|---|
| **Question** | Does a candidate configuration predict the pre-match winner better, out of time, than the shipped config? |
| **Target** | `y` = side A wins. Orientation is randomised at feature-build time, so the base rate is ~0.498 and a constant 0.5 is the naive baseline. |
| **Row grain** | One row per completed tour-level singles match. Never two rows per match. |
| **Decision time T** | Match start. Every feature is as-of by construction (Elo replay, hierarchical filter, layoff, h2h all read `date < match date`). |
| **Eligible population** | Tour-level completed singles matches, 2019-2025, that join to a bookmaker price. Rows without a price are excluded from the primary metric and their count is reported per year. |
| **Split unit** | Calendar year of the match date, expanding walk-forward: train 1995..Y-1, test Y, for Y in 2019..2025. Seven folds. |
| **Minimum training history** | 24 years (1995-2018) before the first test fold. |
| **Window** | Expanding. The archive is stable and pre-2019 data is free; a sliding window would discard history for no documented regime change. |
| **Grouping** | None needed: one row per match, and no match appears in two folds. |
| **Embargo** | None beyond the year boundary. The archive dates every match of a tournament with the tournament's start date, so a tournament spanning New Year lands wholly in the year of its start date; this is stated rather than corrected, and affects at most one event per boundary. |
| **Primary metric** | Log-loss on the market-joined rows of the test year. Lower is better. Equal weight per row within a year; the seven yearly numbers are reported and averaged **unweighted** across years, so 2020 (COVID-shortened) counts the same as 2019. |
| **Secondary diagnostics** | Expected calibration error in 10 equal-width bins; accuracy; Brier. Reported per year, never aggregated away. |
| **Baselines, scored on identical rows** | (a) the de-vigged bookmaker average, (b) Elo alone via `elo_logit_pwinner`, (c) constant 0.5. |
| **Tuning** | **None inside this scorecard.** Hyperparameters are the frozen shipped set. A candidate that needs tuning must tune on inner chronological folds strictly inside years < Y, and say so; the scorecard will not select for it. |
| **Final holdout** | There is no separate untouched holdout left, and this charter does not invent one. The protection is procedural instead: a candidate is scored **once**, all seven years at once, and the result is recorded whether or not it is liked. |
| **Success rule** | A candidate ships only if **both** hold: it beats the shipped config in **at least 5 of the 7 years**, and the unweighted mean paired delta is at least **0.002** better in log-loss. Non-negotiable alongside: ECE must not worsen by more than 0.005 in the mean. |
| **Failure conditions** | Any year where the candidate is worse by more than 0.01 log-loss fails the candidate outright regardless of the mean, as that is a regime it actively breaks. |
| **Reruns and versioning** | One score per candidate. Re-scoring the same candidate after seeing the result makes it exploratory, and the charter version must be bumped and the earlier number kept. |

## Leakage checks required before a number is believed

1. `train.date.max() < test.date.min()` asserted per fold, printed per fold.
2. No `winner_uid`/`loser_uid` pair appears in both train and test of the same fold (it can and should appear across folds; the same two players meeting again is signal, not leakage).
3. Feature columns are exactly the shipped list; anything added must be as-of and named in the candidate's description.
4. The market price is never a feature. It is only ever a baseline.

## What this replaces

"One holdout reveal" language in the tickets, and the go/no-go paragraph in
`map.md`. Anything scored before this charter existed (the 0.00073 gap, the
2024-25 holdout numbers, round 14's fold consistency) stays labelled
exploratory and is not comparable to a number produced under it.

## First result under this charter (2026-09-07): the shipped configuration

Scored once, all seven years, before any candidate was run.

| year | n | model | market | Elo | gap to market | ECE | acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 4,018 | 0.59303 | 0.59283 | 0.61388 | +0.00021 | 0.014 | 0.677 |
| 2020 | 1,931 | 0.58709 | 0.59282 | 0.62931 | **-0.00572** | 0.021 | 0.672 |
| 2021 | 4,001 | 0.58013 | 0.57624 | 0.59864 | +0.00389 | 0.017 | 0.683 |
| 2022 | 4,057 | 0.58423 | 0.58152 | 0.60767 | +0.00271 | 0.019 | 0.674 |
| 2023 | 4,145 | 0.59375 | 0.59025 | 0.61555 | +0.00350 | 0.019 | 0.663 |
| 2024 | 4,424 | 0.58995 | 0.58755 | 0.61031 | +0.00240 | 0.016 | 0.669 |
| 2025 | 4,145 | 0.59418 | 0.59550 | 0.61922 | **-0.00133** | 0.020 | 0.677 |
| **mean** | | **0.58891** | **0.58810** | 0.61494 | **+0.00081** | 0.018 | 0.674 |

Market coverage 79-89% of tour-level rows per year; the excluded rows are
matches the odds file does not carry, reported per year by the script. A
further 46 of 26,904 joined rows (0.2%) were dropped because the odds file and
the archive disagreed on who won, which means the pair-and-window join had
attached the wrong meeting; that check was added after a review noticed
`market_compare.py` had it and this script did not. Adding it moved the mean
gap from +0.00070 to +0.00081 and changed no conclusion.

**Read it this way.**

- The shipped model is **behind the bookmaker average by 0.0008 log-loss on
  average, and behind in 5 of 7 years**. It is close, and it is not ahead.
- It beats Elo alone by about 0.026 every year, consistently. That gap is the
  model's real, repeatable contribution.
- 2020 is the one clear win and is the COVID-shortened year (1,935 rows), a
  regime, not a triumph. It is reported rather than dropped.
- **2022-2025 are not clean for THIS configuration.** The feature set and
  hyperparameters were selected using the 2022-23 tune window and the 2024-25
  holdout, so those four years flatter the shipped config by an unknown
  amount. The years never used in selection are 2019, 2020 and 2021, where
  the mean gap is -0.0005, i.e. level with the market on three years of data,
  which is too few to claim anything.
- This does **not** contradict the 2026 forward test's +0.0247
  ([issues/36](../../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md))
  and does not replace it: that test scored the **v1 feature set** over
  Jan-May 2026 against a market whose own log-loss was 0.572, a much sharper
  benchmark than the 0.588 average seen here. Different model, period and
  benchmark. What can now be said is narrower and better supported: under a
  locked charter, across seven years, the shipped configuration tracks the
  bookmaker average within a thousandth of a log-loss and does not beat it.

**Why the charter still works despite the contamination.** Its purpose is the
PAIRED comparison of a candidate against the shipped config on identical folds.
Both arms carry the same selection history, so the paired delta is meaningful
even where the absolute level is not. Absolute gap-to-market claims should cite
2019-2021 only, and say n=3.

## Second result: W1, conditioning on the bookmaker price (2026-09-07)

`tune_scratch/screen_market_blend.py`. Weight fitted out of sample: the
charter's walk-forward supplies out-of-sample predictions for 2019-2025, then
for each test year w is fitted on the pooled earlier out-of-sample rows and
applied to that year. 2019 is a fit year only, so six test folds.

`logit(p_blend) = w * logit(p_market) + (1 - w) * logit(p_model)`

| year | n | fitted w | model | market | blend | gain on model | vs market |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 1,931 | 0.507 | 0.58709 | 0.59282 | 0.58465 | +0.00245 | -0.00817 |
| 2021 | 4,001 | 0.454 | 0.58013 | 0.57624 | 0.57358 | +0.00656 | -0.00267 |
| 2022 | 4,057 | 0.513 | 0.58423 | 0.58152 | 0.57856 | +0.00567 | -0.00296 |
| 2023 | 4,145 | 0.530 | 0.59375 | 0.59025 | 0.58775 | +0.00600 | -0.00249 |
| 2024 | 4,424 | 0.545 | 0.58995 | 0.58755 | 0.58433 | +0.00562 | -0.00322 |
| 2025 | 4,145 | 0.549 | 0.59418 | 0.59550 | 0.59102 | +0.00316 | -0.00449 |
| **mean** | | **0.516** | 0.58822 | 0.58731 | **0.58332** | **+0.00491** | **-0.00400** |

Mean ECE across the same folds: model 0.0185, market 0.0177, blend 0.0197.

**The blend beats the model in 6 of 6 years and beats the market in 6 of 6**,
by 0.0049 and 0.0040 respectively. Leaving out 2020, the COVID-shortened fold
that contributes the largest single market-beating margin, it still beats the
model by 0.0054 and the market by 0.0032 in all five remaining years.

**Is beating the market real, or is the de-vigged market simply miscalibrated
and the blend correcting it?** Checked, because the two look identical in a
log-loss table. Mean ECE over the six folds: **market 0.0177, model 0.0185,
blend 0.0197**. The market is the best-calibrated of the three and the blend
is the worst, so the blend is not winning by fixing a market bias - it wins on
log-loss while being less well calibrated, which means it wins on sharpness.
That is discriminating information the model holds and the market does not
price. The fitted weight of 0.52 says the same thing from the other side: two
collinear signals would drive w to 1 and the gain to nothing.

**The two pre-registered rules disagree, and that is the interesting part.**

- The **charter** rule passes it: better in 6 of 6 folds, mean gain 0.0049
  against a 0.002 bar.
- The **critique's own W1 rule** fails it: it demanded 0.015 in 5 of 7 folds
  and got 0.015 in none, and the mean gain of 0.00491 sits just under its
  0.005 "the signals are near-collinear, do not do the work" line.

The critique's threshold was calibrated on the belief that the model was
**0.0247** behind the market, so there was ~0.02 to recover. The scorecard
measured that gap at **0.0008**. There was never 0.02 on the table, so a
threshold built on it was unreachable by construction. The right reading is
not "the blend failed" but "the target it was set was wrong". This is exactly
what a locked scorecard is for: the estimate that justified the work was the
thing that turned out to be wrong.

**What it cannot do.** The draw simulator asks for p(A beats B) over 8,128
unordered pairs; at most 64 first-round matches can carry a price, **0.79%**,
and only before any later pairing is known. So this cannot carry the
tournament projection, the wall chart or the title odds. It improves the
single-match product only.

**REJECTED 2026-09-07 (D-15), on product grounds, not accuracy grounds.** The
gain is real and consistent, and it is not being taken: at w near 0.5, roughly
half of every published number would be the bookmaker's opinion carrying this
site's name. Pete's call, verbatim: "I don't want to use bookmakers odds at all
in our model." So prices keep the single job they already had, which is the
benchmark in this scorecard, and W1 is closed. It should not be reopened on
new evidence about its size, because size was never the objection.

The measurement is kept because it answers something worth knowing anyway: at
w=0.52 the model is carrying about half the information in the blend, which
means its features hold real signal the market does not price. A model that
was merely a noisy copy of the market would have driven w to 1.

## Third result: W4, seeding the in-match model with pre-match skill (2026-09-07)

`tune_scratch/screen_inmatch_seeded.py`. Ticket 37 scored two arms on the 2026
US Open point tape and left the obvious third open because the tape carries
"P1"/"P2" rather than uids. That crosswalk now exists, so arm C chooses each
pair's serve probabilities so the recursion's opening-state number equals the
shipped model's pre-match probability, holding the pair's serve sum at the
tour average so it can only differ from arm A on the skill DIFFERENCE.

219 matches, 40,260 points (287 matches skipped: no pre-match number, or a
retirement with no clean result).

| arm | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| A fixed skill (ticket 37) | **0.0342** | 0.57166 | 0.720 |
| C model-seeded | 0.0507 | **0.53188** | **0.779** |
| the critique's W4 bar | < 0.03 | < 0.55 | |

**Verdict: NO-GO on the pre-registered bar**, which required both. Log-loss
passes comfortably (0.532 against 0.55) and accuracy rises 5.9 points, but ECE
gets worse, 0.051 against a 0.03 bar and against arm A's own 0.034.

**The failure is calibration, not discrimination**, and those are different
problems. Seeding with pre-match skill makes the in-match number much sharper
and much more often right; it is also overconfident, which is what ECE
measures. That is a coherent and expected result: the pre-match model's own
confidence is being propagated into every point of the match without ever
being re-fitted for this use.

**What NOT to do with that.** The obvious move is to recalibrate arm C and
re-run, and the bar would probably then pass. That would be moving the
goalposts after seeing the number. If it is worth doing, it is worth doing as
a new candidate with its own out-of-sample discipline: fit the temperature on
earlier events and score the US Open once, exactly as the charter does for
pre-match candidates. Recorded as NO-GO, with the recalibration named as the
next experiment rather than folded into this one.

**Two anchors earned their keep.**

- C2 (arm C at 0-0 must restate the pre-match number) failed the first run at
  **0.496**, because the bisection searched only positive skill differences,
  so every match where the model made P1 the underdog was silently published
  as a coin flip. The aggregate log-loss still looked good at 0.526. Without
  that anchor this would have been reported as a win.
- It then failed again at 0.027, which turned out to be the serve-order
  effect: the pre-match model does not know who serves first and the recursion
  does. Solving against the actual opening state closed it to 0.0058, which is
  the solve's own resolution on the most extreme pairings rather than a
  systematic error, and is reported as a near-miss rather than rounded away.
- C1 (the tape's P1 is the feed's team1) passed 219 of 219. It was worth
  checking: an inverted mapping would have made every number confidently
  backwards.

## Fourth result: W4 arm D, arm C with its overconfidence taken out (2026-09-07)

`tune_scratch/screen_inmatch_recalibrated.py`. W4 failed on calibration, not
discrimination, so this is the named follow-up candidate rather than a re-run
of the one that failed. `logit(p_D) = logit(p_C) / T`.

| arm | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| A fixed skill (ticket 37) | 0.0342 | 0.57166 | 0.720 |
| C model-seeded (W4) | 0.0507 | 0.53188 | 0.779 |
| **D recalibrated** | **0.0346** | **0.49358** | **0.779** |
| the bar, unchanged | < 0.03 | < 0.55 | |

**Verdict on the letter of the bar: NO-GO.** ECE 0.0346 misses 0.03.

**In substance arm D dominates the incumbent on every axis**: the same
calibration as arm A (0.0346 against 0.0342), a log-loss 0.078 better, and 5.9
accuracy points. And the bar is worth looking at now it has been applied twice:
**0.03 is below arm A's own ECE of 0.0342**, so it demanded a candidate be
better calibrated than the thing it replaces while also being sharper. No
candidate that merely matches the incumbent's calibration could ever pass it.
That is an observation about the rule, not a request to change it after the
fact - the rule stands and the verdict stands with it.

Anchors: **D1 PASS**, both fitted temperatures exceed 1 (1.40 ATP, 1.56 WTA),
confirming arm C was overconfident rather than the diagnosis being wrong.
**D2 PASS**, accuracy moved 0.0000, as a monotone transform should.
**D3 FAIL**, the two temperatures differ by 0.155 against a 0.15 bar, a
marginal miss suggesting the tours may want their own temperature.

**The comparison above is against the wrong baseline, corrected here.** Arm A
is ticket 37's yardstick and has never been live. What the site actually does
with a match in progress is at `simulate_usopen_live.py:88-91`: only COMPLETED
matches are respected, so an in-progress match falls through to the pre-match
probability and the score is ignored entirely. Scored on the same 40,260
points, that incumbent is:

| | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| **what the site publishes today** | **0.1058** | 0.60332 | 0.737 |
| arm A, ticket 37's yardstick, never live | 0.0342 | 0.57166 | 0.720 |
| **arm D** | **0.0346** | **0.49358** | **0.779** |

Against the thing it would actually replace, arm D is **three times better
calibrated** (0.035 against 0.106), 0.110 better on log-loss and 4.2 accuracy
points better. The 0.106 is the visible cost of publishing a player two sets
up at his pre-match number.

**Two limitations, both smaller than that gap.**

1. **No earlier point tape exists**, so the temperature could not be fitted on
   a previous event as this charter requires. It is cross-fitted by tour -
   fitted on the men's draw, scored on the women's, and the reverse - so no
   match calibrates itself, but both halves are one tournament, one surface,
   one fortnight. Weaker than the charter's standard, and labelled so.
2. **The evaluation covers rounds 1-3 only** (219 matches: 128, 64, 27).
   Diagnosed properly on 2026-09-07: this is **not** a join bug, as an earlier
   version of this section claimed. The capture holds 334 main-draw match
   directories and the tournament is still being played - 8 fourth-round
   matches had finished and 1 was in progress at the time of writing, and
   `usopen_2026_points.parquet` is a snapshot built earlier the same morning.
   There is simply no completed late-round data yet. Late rounds are where a
   live number is watched, so this matters, and it resolves itself: after the
   final on 2026-09-13, rebuild the point table and re-score.

**Recommendation, corrected.** Ship it, subject to the late-round re-score
after the final. The earlier "do not ship" here was written against arm A
rather than against what the site does, which understated the gain by an order
of magnitude and named a join bug that does not exist.

## Round 4 arrives, and it splits the answer (2026-09-07, later)

Pete asked why the late-round check had to wait. Most of it does - the
quarter-finals, semi-finals and final have not been played - but the point
table was a stale snapshot, and eight fourth-round matches had finished since
it was built. Rebuilding it (88,993 points, 562 matches) and re-scoring adds a
first, small look at late rounds: 231 matches and 42,657 points overall, of
which 7 matches and 1,197 points are round 4.

**All rounds**

| | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1063 | 0.59066 | 0.747 |
| arm A | 0.0286 | 0.57013 | 0.712 |
| arm C, seeded | 0.0456 | 0.52184 | 0.784 |
| arm D, seeded + recalibrated | 0.0337 | 0.48715 | 0.784 |

**Round 4 only, 7 matches, so read it as a signal and not a result**

| | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1711 | 0.38008 | 0.825 |
| arm A | 0.1841 | 0.37622 | 0.892 |
| **arm C, seeded** | **0.1277** | **0.22445** | 0.909 |
| arm D, seeded + recalibrated | 0.1747 | 0.27189 | 0.909 |

**Two things follow, and they point in different directions.**

1. **Conditioning on the live score is settled.** In every round, both seeded
   arms beat what the site publishes today by a wide margin: overall 0.487 or
   0.522 log-loss against 0.591, accuracy 0.784 against 0.747, and calibration
   0.034 or 0.046 against 0.106. There is no reading of this table in which
   ignoring the score is the right thing to keep doing.
2. **Whether to recalibrate is NOT settled, and round 4 argues against it.**
   The temperature that helps in rounds 1-3 hurts in round 4, where arm C
   beats arm D on both calibration (0.128 against 0.175) and log-loss (0.224
   against 0.272). That is coherent: late-round matches here were lopsided
   (accuracy 0.909), and shrinking confidence is exactly the wrong move when
   the confident answer keeps being right. It rests on 7 matches, so it is a
   warning rather than a finding, and it is the first evidence that the
   early-round temperature does not transfer.

**Recommendation, updated again.** Ship the seeding. Do not ship the
recalibration on this evidence. The clean decision between C and D needs the
quarter-finals onward, which arrive on 2026-09-13, and the same rebuild-and
re-score will answer it in ten minutes. Note also that with round 4 included
arm A's ECE falls to 0.0286 and now passes the 0.03 bar the candidates were
being held to, which is a reminder that these calibration numbers move
materially on 12 extra matches.

## Arm C moved into the live simulator (2026-09-07, later still)

Built on `feat/inmatch-seeding`, behind `--live-score`, which is OFF by
default. That default is not caution for its own sake: the scheduled task runs
this checkout's working tree, so anything on by default here is published
within 30 minutes and never passes a review gate.

The solve was about to exist twice, once in the screen and once in the
simulator. That is the exact shape of the F4 head-to-head defect - training
counted one thing, serving counted another, and nothing failed - so it now
lives once, in `dropshotdata/scripts/inmatch_seeding.py`, and both callers
import it. The screen was re-run through the shared module to prove the
refactor did not change what was measured:

| arm C | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| as recorded above | 0.0456 | 0.52184 | 0.784 |
| through the shared module | 0.0460 | 0.52146 | 0.785 |

The small drift is intended and is an improvement. The recursion is memoised
on serve rates, and the key rounded them to 1e-3, which is the solve's real
resolution: anchor C2 sat at 0.00495 against a 0.005 bar, passing on a
rounding error rather than on being right. At 1e-4 it is 0.00059. Anchors C1
(100% agreement on 231 matches) and C3 (100%) are unchanged.

**Before the flag is turned on**, one thing is still open beyond the late-round
question. The published `live_predictions.parquet` has a `source` column hard
coded to `"live"`, so a seeded projection and an unseeded one are
indistinguishable in the artifact the site actually reads. The JSON carries
`_meta.live_score` and `_meta.n_live_conditioned`, but the site does not read
the JSON. Either widen that column or accept that the published copy cannot
say which model produced it - and that is a decision to make deliberately,
not to discover during an incident.

## A review found a bug in the recursion itself (2026-09-07, last)

Reviewing the branch above turned up a defect in `p_set_from_state`, which is
the set-level recursion every in-match arm on this page is built on. It is
pre-existing, it is not small, and it has never published anything wrong.

**What it did.** A set at 6-6 IS the tiebreak: no thirteenth game is played, so
the set-winner is `p_tiebreak` and nothing else. The code instead played one
more game and applied the tiebreak on top of it, computing
`pa*tb + (1-pa)*(1-tb)`. The error reached **0.42**.

**Why it survived.** With two equal players `pa` and `tb` are both 0.5, and the
wrong expression returns exactly 0.5 - which is also the right answer. The
anchor anyone reaches for first cannot see this. The visible tell was that the
same 6-6 position was worth 0.687 or 0.443 depending only on who was marked as
serving, when a tiebreak's outcome in this model does not depend on that at
all.

**It was written up in the code as not a bug.** `p_match_from_state` carried a
comment saying it agreed with `hm.p_match` at the opening boundary "to within
~0.4% rather than exactly", and that the residual was "a real, understood
convention difference, not a bug". Symmetrised over serve order the two now
agree **to machine precision**. The 0.4% was the defect, and explaining it away
is how it lasted. That check is now an assertion in
`tune_scratch/test_inmatch_seeding.py` instead of a sentence in a comment.

**Nothing published was affected.** The scheduled refresh runs
`simulate_usopen_live.py` then `upload_blog_data.py`, and on `dev` neither
reaches `ticket09_live_modes`. The three things that do are all research
scripts. This branch is what would have wired the recursion into the live path,
so the review caught it in the last window where it was still free.

**Every number on this page is re-measured below.** Fixing it makes the arms
slightly WORSE, which is the expected shape: the arms were being flattered by
the bug damping their confidence at 6-6, exactly the way a correction tuned
against a defect turns into a defect once the defect is fixed.

| all rounds, 231 matches, 42,657 points | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1063 | 0.59066 | 0.747 |
| arm A, fixed skill | 0.0306 | 0.57014 | 0.720 |
| arm C, seeded | 0.0501 | 0.52330 | 0.784 |
| arm D, seeded + recalibrated | 0.0338 | 0.48717 | 0.784 |

| late rounds, temperature fitted on rounds 1-3 | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| **arm C, seeded** | **0.1271** | **0.22265** | 0.908 |
| arm D, seeded + recalibrated | 0.1786 | 0.28145 | 0.908 |

Arm D's anchors all pass on the corrected points (D1 T = 1.395 / 1.513, D2
accuracy moved 0.0000, D3 spread 0.118), and arm C's do too (C1 100% on 231
matches, C2 0.00062, C3 100%).

**The recommendation is unchanged and slightly stronger.** Ship the seeding.
Do not ship the recalibration: on the only late-round evidence available it
still loses to plain arm C on both calibration and log-loss, now by a wider
relative margin. Settle it properly after the 2026-09-13 final.

## The tape was being read wrong, and every number above was affected (2026-09-07, final)

The 6-6 fix should have improved the arms and instead made them slightly worse.
That contradiction was the useful part. Chasing it meant walking actual rows of
the point tape across a set boundary, which is where the real defect was.

**Match 1101, points 67 to 71.** The tape records state AFTER the point, and on
the point that ENDS a set it does two things at once: it increments the set
count and it leaves the games showing the set just finished.

| point | sets | games | SetWinner |
|---|---|---|---|
| 69 | 0-0 | 5-4 | 0 |
| **70** | **1-0** | **6-4** | **1** |
| 71 | 1-0 | 0-0 | 0 |

So on point 70 the set is already counted in the set score, and `6-4` is a
leftover, not a set in progress. **1,410 points - 1.58% of the tape - are on
such a row.** Both previous readings of them were wrong, in opposite
directions: the original code scored `6-4` as a set the leader was about to
LOSE, and the terminal-state fix scored it as a set they win on top of the one
already counted. The screen now zeroes those games, which is what the tape is
actually saying.

This is the third re-measurement on this page, and it invalidates the two
above it. These are the numbers that stand.

| all rounds, 231 matches, 42,657 points | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1063 | 0.59066 | 0.747 |
| arm A, fixed skill | 0.0306 | 0.52953 | 0.721 |
| arm C, seeded | 0.0491 | 0.48286 | 0.785 |
| arm D, seeded + recalibrated | 0.0272 | 0.47351 | 0.785 |

| late rounds, temperature fitted on rounds 1-3 | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| **arm C, seeded** | **0.1285** | **0.22333** | 0.907 |
| arm D, seeded + recalibrated | 0.1703 | 0.26865 | 0.907 |

**Arm D now clears the W4 bar it previously failed** - ECE 0.0272 against 0.03,
log-loss 0.47351 against 0.55 - so on the headline metric it is a GO.

**It fails its own anchor anyway, and the anchor wins.** D3 requires the two
tours' temperatures to agree within 0.15; they are 1.209 for the men and 1.458
for the women, a spread of **0.248**. A single temperature is the whole premise
of arm D, and the two halves of the data disagree about what it is. That is the
method saying it is not one correction, which is a fact about the method rather
than a threshold to relax. It points the same way as the late-round evidence,
which independently still prefers plain arm C on both calibration and log-loss.

**Ship arm C. Do not ship arm D.** Two independent signals now say so where
before there was one, and the headline metric that favours D is exactly the
metric a temperature is fitted to move.

Arm C's own anchors all pass on the corrected tape: C1 100% agreement on 231
matches, C2 0.00062, C3 100%.

## Which side of the point are we standing on? (2026-09-07, Pete)

Pete asked whether the win probability is being modelled before the point or
after it. It was neither, consistently.

Every STATE column on the tape is recorded AFTER the point - at point 1 the
score already reads 15-0, and on the last point of match 1101 `P1SetsWon` is 3,
so the match is over. `PointServer`, by contrast, is the server OF that point,
which is a before-the-point fact. The screen paired the two.

**Two things were wrong because of it.**

1. Serve passes when a point ends a game, so on every game-ending point the
   recursion was told the wrong player serves next.
2. **Anchor C3 could not fail.** "The eventual winner is above 0.5 on the final
   point" was being asked of the final point's POST-state, which is the
   finished match, so the recursion returned exactly 1.0 and the anchor
   reported 100% for every arm. It would have reported 100% for any model at
   all. In the pre-point frame it reads **99.1%** - a real check with real
   headroom, and the two matches it misses are genuine.

The screen now scores each point from the state as that point BEGAN (the
previous point's post-state, and 0-0 for the first), paired with that point's
own server. This also makes the first point's probability the pre-match number
by construction, which is anchor C2 - now 0.00062.

| all rounds, pre-point frame, 231 matches, 42,657 points | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1063 | 0.59066 | 0.747 |
| arm A, fixed skill | 0.0306 | 0.53214 | 0.719 |
| arm C, seeded | 0.0503 | 0.48467 | 0.784 |
| arm D, seeded + recalibrated | 0.0270 | 0.47536 | 0.784 |

| late rounds | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| **arm C, seeded** | **0.1287** | **0.22269** | 0.910 |
| arm D | 0.1736 | 0.26831 | 0.910 |

The numbers barely moved and **every conclusion stands**: arm D still fails
anchor D3 (temperatures 1.209 and 1.452, spread 0.243 against a 0.15 bar), late
rounds still prefer arm C, and arm C still beats what the site publishes by a
wide margin. What changed is that the measurement is now answering a coherent
question.

**The post-point number is kept alongside the pre-point one**, so the value of
winning a given point is now visible. Across 42,657 points, arm C moves by:

| mean abs | median | p99 | max |
|---:|---:|---:|---:|
| 0.0065 | **0.0000** | 0.1176 | 0.6911 |

**That median of exactly zero is a finding, not a rounding artefact.** The
recursion works at GAME granularity: it takes sets and games and who serves
next, and never sees the point score within a game. So the published number
cannot move on 30-0, or on a break point saved - it moves only when a game or a
set completes. For a live product that is the wrong grain, and it is the next
thing worth fixing after this ships. It is not a defect in what is being
shipped here, which is strictly better than the pre-match number it replaces,
but it does bound how good this can get.

## Point level, because game level was throwing away 84.6% of what we had (2026-09-07, Pete)

Pete asked whether the data was point level or game level. The answer was
both, in the wrong combination.

**The data is point level**: 88,993 rows, one per point, carrying the
within-game score (`P1Score`/`P2Score`, written 0/15/30/40/AD) and a tiebreak
flag on 2,404 of them. **The model was game level**: its entire state was sets,
games and who serves next, with `p_game(p)` computing a game only from 0-0. So
the richer half of every row was discarded.

| | points | share |
|---|---:|---:|
| start a game, so game level is exact | 13,709 | 15.4% |
| carried a non-zero game score the model never saw | 75,284 | **84.6%** |

At 40-0 up on serve and 0-40 down, the model returned the identical number.

**What was added.** `p_game_from_state(p, a, b)` for a game from any point
score - the deuce chain, which collapses to a closed form at 3-3 and beyond -
and `p_tiebreak_from_state` for a tiebreak from any point score, recovering who
served point one from the rotation rather than being told. Both are threaded
through `p_set_from_state` and `p_match_from_state` as arguments that DEFAULT TO
0-0, so every existing caller gets exactly the number it got before. That is
asserted, not assumed: the difference is 0.00e+00.

The tiebreak tail is exact rather than capped. From any tie the next two points
are always one serve each - A serves point t iff (t-1) mod 4 is 0 or 3, and a
tie has been reached after an even number of points - so A takes both with
pa(1-pb), B with (1-pa)pb, and the geometric sum is closed form.

| all rounds, pre-point frame, point level | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| what the site publishes today | 0.1063 | 0.59066 | 0.747 |
| arm A, fixed skill | 0.0297 | 0.52936 | 0.724 |
| arm C, seeded | 0.0484 | 0.48310 | 0.782 |
| arm D, seeded + recalibrated | 0.0278 | 0.47283 | 0.782 |

| late rounds | ECE | log-loss | accuracy |
|---|---:|---:|---:|
| **arm C** | **0.1266** | **0.22068** | 0.906 |
| arm D | 0.1672 | 0.26615 | 0.906 |

**The aggregate barely moved and that is the honest headline.** Arm C goes from
0.0503 to 0.0484 on calibration and 0.48467 to 0.48310 on log-loss. Anyone
sizing this on the metric alone would call it not worth doing.

**The product behaviour changed completely, and the metric cannot see it.**

| arm C per-point change | mean abs | median | p99 | max |
|---|---:|---:|---:|---:|
| game level | 0.0065 | **0.0000** | 0.1176 | 0.6911 |
| point level | 0.0163 | **0.0109** | 0.0851 | 0.4199 |

A median of zero meant the published number was frozen for the whole of a game
and then jumped. It now moves on every point. A break point at 5-5 in a
deciding set is worth 0.363 against 0.782 at 40-0 - a spread of 0.419 that used
to be a single number. That is the difference between a live win probability
and a scoreboard that updates every six points, and no aggregate log-loss over
42,657 points was ever going to report it.

Every conclusion is unchanged: arm D still fails anchor D3 (temperatures 1.202
and 1.458, spread 0.256), late rounds still prefer arm C, and arm C still beats
what the site publishes by a wide margin.

One note on anchor C3, which reads 100% again here against 99.1% at game level.
That is not the old degeneracy returning: the final point's pre-state is still a
live position (games 5-4, not a finished match), and it reaches 100% because
knowing the point score genuinely tells you more. The check retains its
teeth - it is asked of a state the match could still turn from.

### Review caught an advantage point reading as a won game (2026-09-07)

The live feed writes an advantage game as `['A', null]`, and that null is not
"no score" - it means the other player is on 40. Reading it as zero gives 4-0,
which `p_game_from_state` resolves to a **won game at probability 1.0**, on the
poll where somebody reaches advantage. Correct is 0.9213 at p=0.65.

Confirmed against 289 of 15,385 captured polls. Every gameScore shape the feed
sends is now decided in one place, `parse_game_score`, and asserted in test 21.

**The measured numbers were never affected.** The point tape writes advantage
as `AD` in `P1Score` with `40` on the other side, which converted correctly all
along. This was purely the live path - the one that publishes - and it would
have fired on exactly the break-point case the point-level work exists to
represent.

Also tightened: the screen took a new game from the score column having been
reset to 0-0, which holds at 13,200 of 13,203 boundaries. The three exceptions
are rows missing from the tape or a live score correction. It now reads
`(SetNo, GameNo)`, so a stale score cannot leak into a new game at all.

