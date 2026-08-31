# Rating model report: XGBoost (shipped predictor)

Per the `ratings-strength-models` skill's reporting template. Consolidated
2026-08-31 from tickets 27, 28, 29 - no new analysis, existing results
only. **This is the live predictor** feeding inthegame-data R2 (ratings +
predictions parquet) and the ticket 29 tournament simulator / US Open
forecast.

```text
Rating model: XGBoost, feature-model on top of Elo + hierarchical +
  serve/return-form features (NOT an Elo-family or pure state-space
  rating itself - a supervised classifier over rating-derived features)
Sport: Tennis, ATP + WTA (combined; ATP/WTA-split tested and rejected as
  a clean null result, ticket 29)
Signal: win (binary; binary:logistic objective)
Params:
  max_depth = 3, eta = 0.02, subsample = 0.7, colsample_bytree = 0.9,
  min_child_weight = 5.0, gamma = 1.0, reg_alpha = 0.1, reg_lambda = 1.0
  num_boost_round = 2528
Features (config D - match + set + game granularity, current best,
  shipped 2026-08-29):
  elo_overall_diff, elo_surface_diff, elo_logit_pwinner,
  set_elo_diff, set_elo_logit_pwinner,
  game_elo_diff, game_elo_logit_pwinner,
  hier_dmean_diff, hier_dvar_diff, hier_dvar_sum, hier_logit_pwinner,
  weeks_since_last_diff, n_prior_diff, h2h_diff, best_of, age_diff,
  fatigue_diff, serve_form_diff_targeted, return_form_diff_targeted
  categorical: surface, round, tourney_level, tour
  hier_logit_pwinner (the hierarchical model's own pre-match point
  estimate) was newly added in ticket 28 and was source-verified
  leak-free before use, despite hierarchical-model itself trailing Elo
  standalone (see hierarchical-model.md) - a feature can help a
  downstream model even when its own point predictions don't win alone.
As-of rule: all Elo/hier/form feature inputs are the same pre-event, as-of
  artifacts documented in elo-baseline.md / hierarchical-model.md;
  hierarchical_model.py replays the full archive per training window
  (ticket 28's key unlock: widening the training window from 2yr to 29yr
  via a ~90s re-run of the frozen v25 config, not a new architecture)
Validation: chronological split - train 1995-2023, holdout 2024-25
  (market-odds-joined). No leakage-relevant tune window reused for final
  metric reporting (leakage-audit applied per ticket 28).
Metrics vs baselines (2024-25 holdout, market-odds-joined):
  XGBoost (config D): log-loss 0.58976
  Market (Pinnacle):  log-loss 0.58903        gap 0.00073
  Elo alone:          log-loss 0.6120         (see elo-baseline.md)
  Hierarchical alone: log-loss 0.6275         (see hierarchical-model.md)
  WTA split: XGBoost 0.58779 vs market 0.58852 - XGBOOST BEATS THE
    MARKET, first time any model in this project has done so on a real
    subset (ticket 28).
  ATP split: XGBoost trails market by 0.00214 - resistant to every
    ATP-specific lever tried (ten candidates tested and rejected: MOV-
    weighted Elo, rolling serve-form, isotonic recalibration, recency-
    weighting, WTA tournament-level K, XGBoost-Elo blending, nosurf hier
    state, full tour-model separation, home-court advantage, hand-
    mismatch). Shipped everywhere anyway for simplicity, per Pete.
Calibration: xgb-BT (Bradley-Terry-transformed XGBoost output) confirmed
  as the best rank-order predictor of all four candidate rating sources
  at the tournament-simulation task (ticket 29 step 1.5); cold-start
  (fresh-season, dampened) ratings beat hot/current ratings outright at
  every damping level tested for tournament simulation specifically.
Limits:
  - ATP gap to market (0.00214) is real and unresolved, not merely
    untried - ten fixes rejected. Documented model limitation, not a bug
    (also see the Alcaraz/Zverev vs Betfair case in ticket 29 - a
    quantified real limitation, confirmed not a bug).
  - Granularity sweep (adding set/game-level Elo) required re-parsing
    scoreline data; a coverage gap in parse_score_sets for set-Elo was
    found and fixed post-shipping (2026-08-29) but judged not worth a
    full retrain-and-redeploy cycle on its own - the improvement was a
    wash within noise (0.60398 vs 0.60472).
  - Not re-tuned against XGBoost's own hyperparameter surface after the
    set/game-Elo features were added - the base config (max_depth=3
    etc.) predates the granularity sweep.
  - Depends on hierarchical_model.py as a feature source, which still
    carries the unresolved tail-calibration bug (GitHub issue #1) in ITS
    OWN standalone predictions - not confirmed whether that defect
    contaminates hier_logit_pwinner as an XGBoost input or is fully
    absorbed/corrected by the downstream fit. Not tested directly.
Reproduce:
  python dropshotdata/scripts/train_best_model.py       # base config
  python dropshotdata/scripts/train_xgb_granularity_variants.py  # config D
  python dropshotdata/scripts/simulate_tournament_xgb.py  # ticket 29 sim
```
