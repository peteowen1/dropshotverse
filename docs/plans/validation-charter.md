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

**Two limitations that travel with this number, both larger than the miss.**

1. **No earlier point tape exists**, so the temperature could not be fitted on
   a previous event as this charter requires. It is cross-fitted by tour -
   fitted on the men's draw, scored on the women's, and the reverse - so no
   match calibrates itself, but both halves are one tournament, one surface,
   one fortnight. Weaker than the charter's standard, and labelled so.
2. **The evaluation covers rounds 1-3 only** (219 matches: 128, 64, 27). The
   point tape holds later-round matches whose ids did not join to the draw
   feed, so late-round calibration is entirely untested - and late rounds are
   exactly where a live win-probability number gets watched. This gap should
   be closed before anything ships, and it is a capture-join question, not a
   modelling one.

**Recommendation.** Do not ship on this evidence. Close the round 4-plus join
gap, re-score, and refit the temperature on the first event that provides a
second tape. If it holds up there, it is a large and real improvement to the
live product: a player two sets up would finally be published as two sets up.

