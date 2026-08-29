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
