# 25 — Elo baseline: fix the causal-ordering bug, add inactivity/layoff decay

Type: task
Status: closed

## Question

Raised 2026-08-28. `dropshotdata/scripts/elo_baseline.py` is the frozen comparison
point the entire hierarchical-model tuning effort (rounds 1-10, `RESUME-beat-elo.md`)
has been chasing all session, but Elo itself had never been touched or improved -
every "gap to Elo" number computed so far implicitly assumed the Elo baseline was
already as good as it reasonably could be. Two things suggested it wasn't:

1. `hierarchical_model.py`'s round 9 found and fixed a real causal-ordering bug in
   `prepare()` (qualifying rounds sorted after the entire main draw within a shared
   `tourney_date`). `elo_baseline.py`'s `load_matches()`/`prepare()` are near-duplicates
   of that file's - did it have the identical bug?
2. A prior-art review against `EloCalculator.java` (tennis-crystal-ball,
   github.com/mcekovic/tennis-crystal-ball, commit `fcda1e607deb289369a0ce37d0f8c756b97910a6`)
   found `elo_baseline.py` was missing several standard techniques a real production
   tennis Elo implementation uses. Inactivity/layoff decay was flagged as the clearest
   gap and the smallest, most isolated change to test first.

Explicitly out of scope for this ticket: tournament-tier/round-weighted K, margin-of-
victory signal. Both are real gaps too but are separate, larger changes - mixing them
in here would make any result unattributable to one thing (stats-discipline §5).

## Answer

### Task 1: the causal-ordering bug, confirmed present, fixed

Ported `hierarchical_model.py`'s `ROUND_RANK`/`round_rank()` fix (commit `111ecce`)
into `elo_baseline.py`'s `prepare()` almost verbatim - same dict, same normalization
of the single stray lowercase `'f'` round label, same fallback-to-NaN-with-a-warning
for any future unrecognized round value.

**Verified it changes something real, not just plausible-sounding**: a new
`count_round_inversions()` diagnostic counts (qual_row, main_row) pairs sharing a date
where the qualifying round currently sits after the main-draw row - **75,367,441
inversions before the fix, 0 after**. (This count is larger than the hierarchical-
model file's "617,380 player-date groups" figure because it's summed over every
same-date pair, not deduplicated to distinct player-date groups - same bug, same root
cause, same fix, just a different unit of measurement.) Anchor checks (Sinner/Alcaraz,
Sabalenka/Swiatek in top-5 active; no <10-match player in top 20) still pass after the
fix.

**Effect on the number that matters**: on the exact 2024-25 market-odds-joined
holdout (`eval_elo_vs_market.py`, the RESUME harness's join pattern factored into a
reusable script), Elo's log-loss moves from **0.6152 → 0.6134** from this fix alone,
before any decay tuning - a genuine, non-tuned correctness improvement.

Committed separately (`dbfeb42`), as instructed, before starting Task 2.

### Task 2: inactivity/layoff decay

`EloCalculator.adjustRating()` pulls a returning player's rating back toward
`START_RATING` via a logistic function of days since their last match, with a
level-specific decay magnitude (`INACTIVITY_ADJ_FACTOR`), saturating at a BOUNDED
fraction of the rating's distance from `START_RATING` (`INACTIVITY_ADJ_RATING_FACTOR`
= 5, i.e. at most 1/5 of the gap for the default type) - not a full reset, and with a
~30-day no-penalty grace period before the pull engages (`INACTIVITY_ADJ_NO_PENALTY_PERIOD`),
saturating by ~500 days (`INACTIVITY_ADJ_PERIOD`) of inactivity.

Ported the same shape into a new `layoff_decay()` function in `elo_baseline.py`:

```python
def layoff_decay(rating: float, days_gap: float, decay: float) -> float:
    if decay <= 0.0 or days_gap <= 0.0:
        return rating
    logistic = 1.0 / (1.0 + math.exp((LAYOFF_PERIOD - days_gap) / LAYOFF_GRADIENT))
    frac = decay * max(0.0, (logistic - _LAYOFF_DRIFT) / (1.0 - _LAYOFF_DRIFT))
    return rating - frac * (rating - START)
```

with `LAYOFF_PERIOD=500`, `LAYOFF_GRACE=30`, `LAYOFF_GRADIENT=100` (the shape constants,
borrowed directly from the reference and held fixed) and `decay` (the fraction pulled
back at saturation) as the ONE new tunable axis - the genuine one-axis arm this ticket
required, exactly like `blend` already is. `decay=0.0` is an exact structural no-op
(the caller skips the adjustment entirely rather than multiplying by zero), verified
bit-identical to the pre-existing blend-grid numbers before trusting anything nonzero.
Applied to BOTH the overall rating and the per-surface rating independently (each
tracks its own last-played date), consistent with `blend` already treating them as
parallel rating tracks. The decay is applied to the rating BEFORE it's used for that
match's prediction, and the update is computed on top of the decayed value - so the
decay "spends" itself into the persisted state at the moment of return, matching the
reference's `adjustRating()` usage pattern.

**A real population-mismatch caught mid-task.** The eval window is 848k rows, 89% of
which are lower-tier (qual/challenger/ITF). Scoring the decay grid on this file's
existing pooled "ALL" metric showed a flat/noisy effect (1/7 nonzero decay values beat
`decay=0.0`) - which would have read as a negative result. But the hierarchical-model
comparison this ticket exists to serve only ever uses `level_file=='tour'` rows (11%
of the eval window, per `RESUME-beat-elo.md`'s own join pattern) - a materially
different, less noisy population. Restricting the decay-grid selection metric to
`level_file=='tour'` (a distinction the file's own blend-grid code already flags in a
comment: "Tour comparison is only meaningful WITHIN a level") revealed the real
effect: **7/7 nonzero decay values beat `decay=0.0`**, both ATP and WTA agree the
optimum sits around decay≈0.2-0.3, and the shape is a genuine interior optimum (log-
loss falls smoothly to decay≈0.3 then rises again at 0.4-0.5), not a boundary-hugging
artifact - the same "is it interior, not monotonic-to-boundary" check round 6 of the
hierarchical-model tuning used to distinguish real effects from the session's several
false leads. `elo_baseline.py` was changed to select the decay arm on this tour-level
metric, not the noisier pooled one.

**Full grid, market-odds-joined 2024-25 holdout** (`tune_layoff_decay.py`, blend=0.0
held fixed):

| decay | n | logloss (ALL) | ATP logloss | WTA logloss |
|---|---|---|---|---|
| 0.00 | 8,598 | 0.6134 | 0.6182 | 0.6085 |
| 0.05 | 8,598 | 0.6129 | 0.6179 | 0.6078 |
| 0.10 | 8,598 | 0.6125 | 0.6177 | 0.6072 |
| 0.15 | 8,598 | 0.6122 | 0.6175 | 0.6068 |
| 0.20 | 8,598 | 0.6120 | 0.6174 | 0.6065 |
| **0.30** | 8,598 | **0.6120** | 0.6173 | 0.6065 |
| 0.40 | 8,598 | 0.6122 | 0.6173 | 0.6070 |
| 0.50 | 8,598 | 0.6128 | 0.6175 | 0.6079 |

`decay=0.3` selected (marginally ahead of 0.2, both effectively tied at the interior
optimum). Anchor checks re-verified on the `decay=0.3` arm and still pass.

**Honest overfitting-risk caveat, stated plainly per the task brief's instruction**:
this script evaluates directly on 2024-25 with no separate tuning-window split of its
own - a pre-existing property of this baseline file, not invented for this ticket -
so picking the single best point of an 8-value grid on this same window does carry
some real risk of fitting this window's noise, more than a proper tune(2022-23)/
holdout(2024-25) split would guard against. This is mitigated, not eliminated: the
deciding evidence was the neighbourhood-robustness check (7/7 points improving, both
tours independently agreeing on roughly the same optimum, an interior rather than
boundary shape) rather than trusting the single lowest grid point in isolation - but
it is not the same level of protection a genuine held-out tuning window would give.

### Net effect - THE NUMBER THAT MATTERS

Combined (Task 1 ordering fix + Task 2 decay=0.3), on the exact 2024-25 market-odds-
joined holdout: **Elo's own log-loss moves from 0.6152 (original) to 0.6120
(fixed + improved) - a genuine −0.0032 improvement**, achieved via one confirmed
correctness bug fix and one disciplined, gated, anchor-checked hyperparameter
addition. `elo_baseline.py` now writes `decay=0.3` (with `blend=0.0`) to
`elo_eval_predictions.parquet` by default.

**This directly changes the "gap to Elo" number `RESUME-beat-elo.md`'s rounds 1-10
have been reporting all session.** Hierarchical's current best (v25 + round 9's kept
ordering fix) is logloss 0.6275 (per round 10). Old gap: 0.6275 − 0.6152 = 0.0123.
New, honest gap: 0.6275 − 0.6120 ≈ **0.0155**. Fixing and improving Elo was the right
thing to do regardless of this effect - it's the project's frozen comparison point and
carried two real, fixable defects - but it makes the hierarchical model's job strictly
harder, not easier. **Anyone continuing the hierarchical-model tuning effort should
use 0.6120, not 0.6152, as Elo's number from here on**, and should expect the true gap
to beat to be closer to 0.0155 than 0.0123 until re-measured against this fixed
baseline directly (the two efforts ran in parallel this round, so no single commit has
both fixes applied at once yet).

## What's still open

- Tournament-tier/round-weighted K and margin-of-victory (games-won) signal - both
  explicitly out of scope for this ticket, both real gaps `EloCalculator.java` covers
  that this baseline still doesn't. Natural next Elo-baseline tickets if anyone wants
  to keep closing this gap.
- The hierarchical-model tuning effort (rounds 1-10) has not yet been re-run against
  the fixed+improved Elo baseline directly - all its round-to-round deltas are
  internally consistent (Elo held fixed throughout those rounds), but the ABSOLUTE
  gap number they report is now stale by the amount this ticket moved Elo.
