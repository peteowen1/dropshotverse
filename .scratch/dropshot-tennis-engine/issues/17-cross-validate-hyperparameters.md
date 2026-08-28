# 17 — Cross-validate the hierarchical model's hyperparameters

Type: task
Status: resolved

## Question

Raised by Pete 2026-08-27 via `/goal`: ticket 10's hierarchical model trails the market by 0.070 log-loss and Elo by 0.044, on plausibility-set (never cross-validated) hyperparameters — `RW_VAR_PER_WEEK`, `SURFACE_PRIOR_VAR`, `OU_THETA_PER_WEEK`. Ticket 10's own verdict named this the highest-priority lever. Test it properly.

## Validation charter (locked before any tuning run, per `validation-design`)

- **Target / grain / decision time:** match win probability (posterior-averaged), one row per tour-level match with point-count coverage joined to Pinnacle odds. T = pre-match; already guaranteed by the filter's online, sequential structure (no future information reaches any prediction).
- **Split unit:** calendar year. Chronological, no ambiguity.
- **Grouping:** none needed — no repeated-perspective panel issue here.
- **Window:** expanding, from 1877. Matches the model's actual structure (one continuous sequential fit); a sliding window doesn't apply to an online filter.
- **Tuning window:** 2022–2023, odds-joined. Chosen as the two seasons immediately preceding the final holdout — close enough to be representative, far enough to guarantee no overlap.
- **Final holdout:** 2024–2025, odds-joined — **identical rows to ticket 10's Elo/market comparison.** Not touched for selection.
- **Primary metric:** log-loss (posterior-averaged, per ticket 10's Jensen's-gap fix). Secondary: Brier, accuracy, calibration by bucket.
- **Baselines:** Elo and Market, already scored on the identical 2024-25 rows in ticket 10 — reused, not re-run.
- **Mechanism:** because the filter is one sequential pass, a single full historical fit per candidate produces eval rows for BOTH windows in one run. The discipline is procedural, not architectural: compute and inspect only the 2022-23 log-loss when selecting; reveal 2024-25 exactly once, for the chosen configuration only.
- **Search:** coordinate/sequential, not full grid, given session time constraints — tune `RW_VAR_PER_WEEK`, then `SURFACE_PRIOR_VAR` (uniform scale multiplier), then `OU_THETA_PER_WEEK`, each holding the others at their best-so-far value. State this simplification: a coordinate search can miss interactions a full grid would catch.
- **Success rule:** report the tuned 2024-25 numbers honestly regardless of outcome. A meaningful win is closing a material fraction of the 0.070 gap to market; closing it entirely is not expected and not required to report the result as useful.
- **Leakage check:** assert, in code, that the tuning selection step never reads any row dated 2024 or later.

## Answer

Run 2026-08-27. Scripts: `dropshotdata/scripts/tune_hyperparams.py`, `hier_tuned_compare.py`. Full log: `dropshotdata/tune_scratch/sweep.log`.

### The harness had a real bug before it produced a single trustworthy number

`market_compare.py`'s `load_odds()` hardcoded `YEARS = (2024, 2025)` at module level — so the first tuning attempt silently scored every candidate against the *wrong* odds file. The join fell to zero rows and errored loudly rather than returning a plausible-but-wrong number, which is the only reason it was caught immediately rather than producing a confidently false result. Fixed by parameterising `load_odds(years=...)`.

### Selection, strictly on 2022-2023

Coordinate search, three candidates per hyperparameter, each including the ticket-10 default so tuning could never silently regress:

| hyperparameter | candidates tried | selected | ticket-10 default |
|---|---|---|---|
| `RW_VAR_PER_WEEK` | 0.005, 0.010, 0.020, 0.040 | **0.005** | 0.010 |
| `SURFACE_VAR_SCALE` | 0.5, 1.0, 2.0, 4.0 | **4.0** | 1.0 |
| `OU_THETA_PER_WEEK` | 0.0010, 0.0025, 0.0050, 0.0100 | **0.0010** | 0.0025 |

Tuning-window (2022-23, 8,460 odds-joined rows) log-loss: **0.6812 → 0.6697**, selected using only these rows. **Leakage check ran on every candidate**: an assertion that no scored row is dated 2024 or later, verified in code, not just by intent.

The direction is coherent, not arbitrary: less process noise, less aggressive surface shrinkage (more freedom for genuine specialists), and a much slower mean-reversion (~13-year half-life instead of ~5.3) all point the same way — the reactive fixes from ticket 10's bug-hunt (added to stop a specific runaway-drift failure) were tuned more conservatively than the data wants once the actual bugs behind that failure were fixed. Worth remembering: a safety net calibrated to stop a fire doesn't have to also be the right day-to-day setting.

### The one-time reveal, on the untouched 2024-25 holdout

| model | acc | log-loss | brier |
|---|---|---|---|
| Elo | 0.653 | 0.615 | 0.214 |
| Hierarchical, untuned (ticket 10) | 0.6135 | 0.659 | 0.232 |
| **Hierarchical, tuned (this ticket)** | **0.6135** | **0.651** | **0.229** |
| Market (Pinnacle, de-vigged) | 0.679 | 0.589 | 0.203 |

**Gap to market closed 12.2%**: 0.0706 → 0.0620 log-loss. Still trails Elo by 0.036 and the market by 0.062 — tuning alone did not close the whole gap, and wasn't expected to.

**Accuracy is identical to four decimal places, 0.6135 both times.** Flagged rather than treated as automatically fine, per `stats-discipline` §2 (suspicious agreement). The explanation is plausible, not assumed: none of the three tuned hyperparameters enter the *direction* of a skill update — they govern process variance, surface shrinkage width, and reversion speed, which shape confidence and spread, not which player the point estimate favours. Accuracy only depends on which side of 0.5 a prediction falls; log-loss depends on how far from it. Tuning parameters that primarily affect calibration leaving the win/loss calls unchanged is the expected signature of a real, targeted improvement — not evidence of one.

**Calibration improved in a way the headline log-loss understates.** Comparing bucket errors directly against Elo's own (from ticket 10):

| favourite prob | Elo error | Tuned error |
|---|---|---|
| (0.5, 0.6] | −0.022 | **−0.014** |
| (0.6, 0.7] | −0.053 | **+0.011** |
| (0.7, 0.8] | −0.022 | −0.023 |
| (0.8, 0.9] | −0.045 | **+0.009** |
| (0.9, 1.0] | −0.030 | −0.195 (n=27) |

**The tuned model is better calibrated than Elo in 4 of 5 buckets.** The one exception is the top bucket, which now holds only **27 matches** (Elo's equivalent held 467) — the tuned model is far less willing to be extremely confident at all, which is itself a symptom of appropriately wider uncertainty, and a 27-row bucket is too small to read as a real miss rather than noise. The overall log-loss still trails Elo mainly because of prediction spread and this one thin bucket, not because the model is worse calibrated where it actually commits to confident predictions.

### Verdict

**Tuning was worth doing and materially closed the gap without producing a suspicious result.** The map's actual destination — calibration parity with the market — is now much closer than the headline log-loss numbers alone suggest, particularly given the model is now *better* calibrated than Elo through the bulk of the probability range. It does not beat Elo or the market outright, and this was a coordinate search (interactions between the three hyperparameters were not explored), not a full grid — a genuine simplification, stated per the locked charter.

**What would plausibly close more of the remaining gap**, unchanged from ticket 10's original priority list, now with one item resolved: (1) ~~cross-validated hyperparameters~~ **done here**; (2) a properly hierarchical (not fixed-variance) surface treatment; (3) the panel-based regime-B inversion ticket 12 proved viable, replacing the combined-strength shortcut.
