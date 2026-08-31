# Rating model report: hierarchical point-level model

Per the `ratings-strength-models` skill's reporting template. Consolidated
2026-08-31 from tickets 08, 10, 17-24, 31 - no new analysis, existing
results only. **Not the shipped predictor** - see xgboost-shipped.md for
what actually serves the live site; this model's own standalone win
probability has a known unresolved defect (see Limits).

```text
Rating model: hierarchical point-level serve/return skill (state-space
  filter, Kalman-style, tour as a grouping factor), NOT Elo-family
Sport: Tennis, ATP + WTA + ATP Challenger, singles
Signal: point-level serve/return outcome (binomial on point counts where
  available; Markov scoreline likelihood pre-1991; explicit exclusion
  for retirements) - closed-form recursions up through match win
  probability, per ticket 05/08
Params (locked defaults in hierarchical_model.py, v25 config, ticket
  17/19/20 rounds):
  INIT_VAR = 1.2          initial belief variance, new player
  RW_VAR_PER_WEEK = 0.005 random-walk process variance
  SURFACE_VAR_SCALE = 1.0 surface-offset variance multiplier
  OU_THETA_PER_WEEK = 0.0005  Ornstein-Uhlenbeck mean-reversion rate
  LAYOFF_C = 0.06         quadratic layoff-variance inflation constant
  PHI_MAX = 4.0           ceiling on layoff inflation factor
  MAX_VAR = 4.0           ceiling on any single mu_alpha/mu_beta variance
  MAX_SHIFT = 0.2         ceiling on |shift| from any single observation
  MAX_STATE = 3.0         absolute state-magnitude ceiling
  TIER_WEIGHT = 1.0       non-elite match weight multiplier
  REGIME_B_N = 1.0        evidence weight for scoreline-only regime
As-of rule: pre-event posterior used for prediction, Monte Carlo-averaged
  over the filter's own posterior variance (point-estimate plug-in was
  tested and rejected - Jensen's inequality overstates confidence,
  log-loss 0.928 -> 0.659 fixing this alone, ticket 10)
Validation: written charter before evaluation (validation-design skill) -
  tune window 2022-23, untouched holdout 2024-25, one-time reveal
Metrics vs baselines (2024-25 holdout, market-odds-joined, n=8,594-8,598):
  Hierarchical (v25, round-order fix kept): log-loss 0.6275
  Elo (post-ticket-25 fix):                 log-loss 0.6120  <- current
    honest gap: 0.0155 (was reported as 0.0123 against Elo's earlier,
    buggy 0.6152 - must use 0.6120 going forward, ticket 25)
  Market (Pinnacle):                        log-loss 0.5888
  Hierarchical does NOT beat Elo. Nine distinct fixes/hypotheses tried
  across tickets 17-24 to close the gap; none succeeded outright, though
  cumulative tuning took the raw untuned v0 (0.659) most of the way to
  0.6275.
Calibration: worse than Elo's in most probability buckets (pre-registered
  prediction that it would improve on Elo FAILED, ticket 10). One
  specific bucket (70-90% predicted probability) shows roughly double
  Elo's own miscalibration - the leading unresolved explanation for the
  remaining gap (ticket 20). Global Platt scaling and isotonic
  recalibration were both tried and rejected as fixes - the excess
  confidence is tied to which matches are genuinely lopsided, not simply
  to the model's own stated probability, so recalibrating on its own
  output can't fix it.
Limits:
  - DOES NOT BEAT ELO. Ships nowhere as a standalone predictor; its value
    to date is as an XGBoost feature source (hier_logit_pwinner,
    hier_dmean_diff, hier_dvar_diff, hier_dvar_sum), where it measurably
    helps (ticket 28).
  - KNOWN BUG, UNFIXED: standalone win probability scores WORSE THAN
    RANDOM on the 2024-25 holdout (log-loss 0.72388 vs random's 0.69315)
    due to severe tail overconfidence on upsets (Tabilo beat Djokovic
    scored 0.016%). A flat probability floor was proposed and explicitly
    REJECTED by Pete as a bandaid ("should fix the actual issue not just
    hack caps and cups") - root cause is in the variance/uncertainty
    mechanism, not fixed here. Logged as GitHub issue #1, ticket 31.
  - Absorbing-boundary "corruption" failure mode: bugs 6-9 (ticket 18)
    plus ticket 20's regime-B rejection are five instances of the same
    class of bug - a clamped mean update still gets full variance-
    reduction credit, letting confidence run away. The architectural
    fix (heavy-tailed process noise, ticket 24) is scoped but not built;
    current mitigation is hard clamps (MAX_SHIFT, MAX_VAR, MAX_STATE).
  - Regime-B (scoreline-only, pre-1991 and thin-data matches) uses a
    combined-strength-only inversion, not the panel-based scoreline
    inversion ticket 12 proved viable but ticket 21 never built.
  - Best-of-3/best-of-5 skill transfer is asserted, not validated
    (ticket 23, open).
  - Tournament tier/round loaded but unused in the update weighting
    beyond TIER_WEIGHT's crude non-elite multiplier (ticket 22, open).
Reproduce: python dropshotdata/scripts/hierarchical_model.py
  (env-var overrides for all DSV_* params; writes eval predictions +
  pickled filter state; DSV_EVAL_YEARS defaults to 2024,2025)
```
