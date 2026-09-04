Type: bug, parked
Status: diagnosed, not fixed

## Finding

`v25_postfix_holdout_preds.parquet`'s standalone `p_winner_hier` scores
**worse than random** on the 2024-25 tour-level holdout: log-loss 0.72388
vs random's 0.69315 (-ln(0.5)). Pete caught this immediately ("isnt .69
random") - correct catch, this needed real investigation, not acceptance.

Diagnosed, not a ranking failure: mean(p_winner_hier)=0.615, accuracy=65.2%
(both well above random) - the model discriminates fine on most matches.
It's a severe TAIL-calibration bug: 6.0% of matches get the eventual
winner assigned <10% chance, 0.78% get <1%. Worst cases are real tennis
upsets the model treats as near-impossible instead of unlikely-but-
plausible: Tabilo beat Djokovic (p=0.016%), Van de Zandschulp beat Alcaraz
(p=0.07%), Muller beat Zverev (p=0.12%). Just the top 50 worst predictions
(0.5% of n=10,680) account for 4.2% of TOTAL log-loss - log-loss punishes
confident-wrong answers so harshly that a tiny overconfident tail drags the
average past the random-guessing line even though the model is right most
of the time.

Likely the same class of bug as the documented "ticket 18" kappa/surface-
variance-floor issue from earlier project history (Djokovic vs a qualifier
scoring 70.3% when it should have been ~97%+) - possibly not fully fixed,
or reintroduced, in whatever version of hierarchical_model.py produced this
specific file.

## Explicitly rejected fix, and why

Pete: "5% floor not sane floor for peak djokovic playing a rank 200 player
(99.5% maybe) - should fix the actual issue not just hack caps and cups."
Correct - a flat probability floor would cap genuinely lopsided real
matchups (which SHOULD be ~99%+) at the same ceiling as genuine coin-flips,
trading one miscalibration for another. The real fix has to be in the
variance/uncertainty mechanism that's producing overconfident tail
predictions for upset-prone matchups specifically, not a blanket clip.

## Not fixed here - parked per Pete: "log this as GH issue and move on"

No root-cause trace attempted beyond the diagnosis above. Next step
whenever picked up: find which specific matches are in the worst-50 list,
check what's unusual about their kappa/surface-variance state at prediction
time (thin history? recent surface-var-floor interaction? something else
ticket 18's fix didn't cover?), same investigative pattern that found
ticket 18's original bug.

## 2026-09-04: one hypothesis ELIMINATED, and the 0.724 baseline was wrong

`p_set` had been resolving 6-6 by playing a 13th game and then multiplying
by the tiebreak probability, so the mass reaching (7,6) was
`P(6,6) * [pa*tb + (1-pa)*(1-tb)]` instead of `P(6,6) * tb` - and the second
term credited A with B's share of the tiebreak. Measured against the closed
forms already written down in `05-point-model-literature.md` sections 1a-1c:

* `p_set` off by up to **5.7pp**, always compressing toward 0.5
* `p_set` varied by up to **5.1pp with who served first**, violating the
  Newton & Keller section 3.3 result that set-win probability is independent
  of serve order. `p_match`'s `q = 0.5*(q1+q2)` was averaging that violation
  away rather than symmetrising a genuine asymmetry.
* separately, `p_tiebreak` approximated the 6-6 tail by running to a 24-point
  cap and splitting the remainder 50/50 ("serve probabilities roughly cancel"
  - they do not; the tail favours the stronger server), worth up to 8.0e-4.

Both fixed. Verified against the literature closed forms and, for the
tiebreak tail, against a 400k-trial simulation independent of both
implementations. `stress_test_safety.py` now checks p_game/p_tiebreak/p_set
against those closed forms plus the serve-order identity; suite is 49/0.

**Why this made the reported number worse, and why that is the point.**
Re-scored on the frozen v25 posterior state, same 10,680-match holdout
(reconstruction validated to 4.4e-16 against the stored `p_winner_hier`):

| | log-loss | Brier | acc | mean p | p<10% | p<1% |
|---|---|---|---|---|---|---|
| stored v25 | 0.72388 | 0.23613 | 65.19% | 0.6149 | 6.02% | 0.777% |
| fixed math | **0.77070** | 0.24265 | **65.19%** | 0.6175 | 8.02% | 1.292% |

Accuracy is identical to two decimals - the fix reorders nothing, it only
sharpens. By bucket it IMPROVES the confident-and-right region (p>=0.65:
-0.027, -0.008) and worsens the confident-and-wrong region (p<0.35: +0.147,
+0.443, +0.966).

So the 6-6 defect was an **undiagnosed shrinkage of up to 5.7pp applied to
every match**, partially masking this ticket's overconfidence. That is a
worse version of the flat probability floor Pete explicitly rejected here -
same effect, applied blindly, including to the genuinely lopsided matchups a
floor would have wrecked, and with nobody aware of it. Tabilo over Djokovic
goes 0.016% -> 0.003%.

**Consequences for this ticket:**

1. The 0.72388 baseline was flattered by a bug. The honest v25 number on this
   holdout is 0.77070. Any comparison built on 0.724 inherits the error.
2. The tiebreak math is ELIMINATED as a cause. The overconfidence is in the
   variance/uncertainty mechanism, exactly where this ticket said it was.
   One hypothesis down, the original next step unchanged.
3. NOT YET TESTED: this re-score holds the posterior state fixed, but
   `adf_update` also calls `p_set_full`, so the fitted variances adapted to
   the compressed inputs. A full refit feeds correct set probabilities into
   the ADF updates and could learn wider variances, recovering some or all of
   the +0.047. That refit is the open question.

Nothing shipped changed: the live US Open pipeline runs a cached XGBoost
model, and hier state reaches it only via `build_xgb_features.py`'s
`d1_mean`/`d1_var` features, which move only on a feature rebuild.
