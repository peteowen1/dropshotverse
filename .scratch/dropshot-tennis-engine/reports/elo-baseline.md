# Rating model report: Elo baseline

Per the `ratings-strength-models` skill's reporting template. Consolidated
2026-08-31 from tickets 10, 25, 26 — no new analysis, existing results only.

```text
Rating model: Elo (sequential, UTS-inspired K bundle)
Sport: Tennis, ATP + WTA + ATP Challenger, singles
Signal: win (binary; no margin-of-victory signal)
Params:
  init = 1500
  K = base 32 x uts_kfunction(rating) x uts_context_k(level, round, best_of)
      uts_kfunction: rating-dependent amplifier, ~10x at rating=1500,
      ~5.5x at extreme ratings (see hierarchical_model.py:266 docstring)
      uts_context_k: level/round/best-of multiplier (Slam+Bo5 heaviest)
  blend = 0.0 (pure overall Elo; surface-blended Elo tested, overall wins)
  layoff decay = 0.3 (per-week rating regression toward mean during gaps;
      swept 0.0-0.5, interior optimum, robust across a 7/7-point
      neighbourhood check on both tours independently, ticket 25)
As-of rule: pre-game rating captured before either team's rating is updated;
  events sorted by (date, round_rank) with stable mergesort to fix a real
  round-order causal-leakage bug found in ticket 25 (a team's later-round
  rating could otherwise leak into an earlier-round prediction on the
  same date)
Validation: chronological holdout, 2024-25, market-odds-joined (n=8,598)
Metrics vs baselines:
  Elo:           accuracy 65.4%, log-loss 0.6120 (ATP 0.6173 / WTA 0.6065)
  Pinnacle (mkt): accuracy 67.9%, log-loss 0.5888 -- gap 0.0232
  UTS K-philosophy bundle tested as an alternative and REJECTED: worse
  by +0.0007 log-loss net (a real ATP/WTA split hides inside that net
  number - see ticket 26). Current bundle above is the kept one.
Calibration: not separately bucketed for Elo alone in this pass; used as
  the reference point hierarchical-model calibration is measured against
  (ticket 17/19/20), where it out-performs the hierarchical model in most
  buckets.
Limits:
  - No serve/return split (single overall + surface rating) - that
    granularity only exists in the hierarchical model, which is not
    what's shipped live.
  - Ratings never decay with inactivity beyond the layoff-decay term
    above; a raw current-form leaderboard fills with retired players -
    must filter to active players.
  - ATP and WTA player ids collide on 2,396 values (id 200003 is Jeremy
    Beale in ATP, Justine Henin in WTA) - namespaced as {tour}:{id} at
    ingest, not at read time (ticket 07a).
  - No margin-of-victory or tournament-tier/round weighting beyond the
    K-context multiplier already in the bundle (open items, ticket 25).
  - Decay=0.3 was selected by grid search directly on the 2024-25
    evaluation window itself, with no separate tune/holdout split of its
    own (a pre-existing property of this baseline). Mitigated by a
    neighbourhood-robustness check (both tours agree on the same
    interior optimum) but not to the standard a genuine held-out tuning
    window would give.
Reproduce: python dropshotdata/scripts/elo_baseline.py
  (constants-driven, no CLI flags; writes elo_eval_predictions.parquet)
```

Elo itself is not the shipped predictor (see xgboost-shipped.md) - it is
the project's frozen comparison baseline and a feature input to XGBoost
(`elo_overall_diff`, `elo_surface_diff`, `elo_logit_pwinner`).
