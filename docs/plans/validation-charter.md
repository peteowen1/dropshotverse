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

**The blend beats the model in 6 of 6 years and beats the market in 6 of 6**,
by 0.0049 and 0.0040 respectively. The fitted weight sits at 0.52, near an even
split, which says the model carries real signal the market does not: if the two
were collinear w would run to 1 and the gain would vanish.

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

