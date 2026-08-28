# 27 — XGBoost over Elo + hierarchical raw state + engineered features: BEATS Elo

Type: task
Status: closed (success — new best model on the frozen holdout)

## Question

Raised 2026-08-28, per Pete's own direction after ten rounds of same-architecture
tuning on the hierarchical Bayesian filter failed to beat Elo (`RESUME-beat-elo.md`,
`issues/20-why-does-hierarchical-lose-to-elo.md`): *"elo + rating + features + some
other stuff into xgb.cv might be the best play"* — i.e. stop tuning the MC-averaging
posterior filter directly (its Jensen's-gap convexity defect is fully diagnosed and
every fix attempted at every layer has failed to transfer to holdout) and instead
train a gradient-boosted-tree model on Elo's and the hierarchical model's own RAW
state as features, letting XGBoost combine them itself, plus engineered features
(H2H, experience, layoff, categorical context) — sensible default hyperparameters,
`xgb.cv` used only to pick `num_boost_round` via early stopping, no grid search.

Explicit constraints from the brief: do not refit Elo or the hierarchical model, do
not touch `elo_baseline.py`/`hierarchical_model.py` at all (another agent was
concurrently editing `elo_baseline.py`), exclude market odds/probability as a
feature (would be circular against the project's actual goal — calibration parity
WITH the market, not beating it by copying it), symmetric side-A/side-B
construction so the label is genuinely non-trivial, tune only on 2022-23, reveal on
2024-25 exactly once.

## Answer

### A real, serious leakage bug found and fixed along the way (not in the two
forbidden files)

The first `xgb.cv` run scored an implausible 0.193 tune-window log-loss (real
tennis models don't get anywhere near that — Elo alone is ~0.60). Root-caused via
per-feature single-column `xgb.cv` isolation: `weeks_since_last_diff` ALONE gave
CV log-loss 0.323, i.e. a "days since last match" feature alone out-predicting
Elo by a huge margin — not plausible.

Traced to `tune_scratch/diag_layoff.parquet` (a read-only diagnostic artifact from
an earlier round of this session, `diag_layoff_features.py`) being **stale**: it
was last written 2026-08-28 15:47, hours before `hierarchical_model.py`'s round-9
causal-ordering fix (qualifying/round-order sorted correctly by `(date,
round_rank)` instead of date alone, commit `111ecce`, 21:55) and the regenerated
`prepared_matches_cache.pkl` (21:56) that fix produced. `diag_layoff_features.py`
itself was never re-run after the fix, so its output still reflected the OLD,
buggy same-date ordering. Traced one concrete example
(`atp:207678` on 2022-01-03): the archive correctly orders that day's R32 match
before the R16 match (`prepared_matches_cache.pkl` position confirms it), but the
stale `diag_layoff.parquet` had processed the R16 row FIRST, so by the time the R32
row's "weeks since last match" was computed, the loop had already (wrongly) set
`last_seen` to the same day from the R16 row — giving a false `weeks_since_last=0`
for a player whose real previous match was 7 weeks earlier. A round-by-round
breakdown confirmed this wasn't a one-off: winners showed `weeks_since_last==0`
in 97-98% of R32/R16/QF rows vs 7-31% for losers — a huge, systematic,
round-dependent asymmetry with no plausible causal-fatigue explanation.

**Fix**: re-ran `diag_layoff_features.py` (a pre-existing, read-only diagnostic
script — not `hierarchical_model.py` or `elo_baseline.py`, so within scope) against
the current, post-round-9-fix `prepared_matches_cache.pkl`. Confirmed fixed: the
same example row now shows `weeks_since_last_w=7.0` (correct), and the round-by-
round zero-fraction asymmetry disappeared (QF/SF/F now show near-identical winner/
loser rates, as bracket symmetry predicts). Re-running the single-feature `xgb.cv`
isolation after the fix: `weeks_since_last_diff` alone now gives log-loss 0.688
(≈ no-signal baseline, plausible for a weak feature), not 0.323. **This stale-cache
trap is a real, generalizable lesson**: any diagnostic artifact derived from
`prepared_matches_cache.pkl` needs to be checked (or just re-run) after any change
to `hierarchical_model.py`'s `prepare()`/ordering logic, the same lesson round 9's
own section of `RESUME-beat-elo.md` already flagged for the cache itself
("anyone editing `prepare()`/`load_matches()` ... verify, don't assume").

### Feature build

New scripts (none of which touch `elo_baseline.py`/`hierarchical_model.py`):

- `dropshotdata/scripts/elo_raw_ratings_for_xgb.py` — a READ-ONLY copy of
  `elo_baseline.py`'s `load_matches()`/`prepare()`/`run_elo()` logic, extended to
  emit the RAW per-side overall and surface Elo ratings pre-match (the frozen
  `elo_eval_predictions.parquet` only carries the final blended `p_winner`, which
  collapses overall vs surface). Verified faithful: recomputed blend selection
  (`b=0.0`) and decay (`0.3`) reproduce the frozen parquet's `p_winner` on the
  8,594-ish tour-level eval rows to float precision, except 8 rows involved in
  round-robin same-day rematches where a duplicate-key join legitimately swaps
  pair order (confirmed, not a bug).
- `dropshotdata/scripts/build_xgb_features.py` — joins `tune_scratch/v25_tune_preds
  .parquet` (2022-23) / `hier_v25_eval_predictions.parquet` (2024-25) — the
  hierarchical model's own already-written raw-state exports (`d1_mean`, `d1_var`,
  `d2_mean`, `d2_var`; NOT its calibrated probability) — against the Elo raw
  ratings above, `tune_scratch/diag_layoff.parquet` (regenerated, see above) for
  layoff/experience, `tune_scratch/prepared_matches_cache.pkl` for round/tier
  context, and a fresh as-of head-to-head computation (`compute_h2h()`, a single
  chronological pass over the full archive's Elo-ratings file, which is already
  correctly time-ordered — counts only STRICTLY PRIOR meetings, never the match's
  own outcome or any later match).

Symmetric construction: each match's two players assigned to side A/B via
`hash(sha256(date + sorted uid pair))` — outcome-blind, cannot leak. Every numeric
feature is `sideA − sideB`; `surface`/`round`/`best_of`/`tourney_level`/`is_elite`
kept as non-differential categorical context. Label balance verified ~49.1-49.6%
on both windows (not exactly 50% only because of the small number of dropped rows
with missing joined features, not a construction bug).

**Label-equivalence note** (worth recording since it wasn't obvious at first):
standard binary log-loss computed directly on `(y, p_pred)` is mathematically
IDENTICAL to Elo/hier's own "P(assigned to the actual winner)" convention under
this construction — both reduce to `-mean(y·log p + (1-y)·log(1-p))` — so no
extra conversion step was needed to make the XGBoost score comparable.

### Tuning and holdout (charter: tune on 2022-23 only, reveal once on 2024-25)

`xgb.cv`, 5-fold, `early_stopping_rounds=50`, `binary:logistic`/`logloss`, sensible
defaults per the brief (not searched): `max_depth=5, eta=0.08, subsample=0.8,
colsample_bytree=0.8`. Selected **`num_boost_round=36`** (cv test-logloss 0.5992).
One final model fit on all of 2022-23 (10,296 rows) at that round count, scored
ONCE on the market-odds-joined 2024-25 holdout (n=8,593 — 1 fewer than the
session's usual 8,594, a trivial dedup-edge difference, not investigated further,
same as the 8,594-vs-8,598 note in issue 25).

| | XGBoost | Elo (post-fix, 0.6120 basis) | Hierarchical (v25 snapshot) |
|---|---|---|---|
| ALL log-loss | **0.6025** | 0.6121 | 0.6267 |
| ALL accuracy | 0.6662 | 0.6530 | 0.6453 |
| ALL Brier | 0.2082 | 0.2126 | 0.2185 |
| ATP log-loss | **0.6023** | 0.6174 | 0.6213 |
| WTA log-loss | **0.6027** | 0.6067 | 0.6322 |

**XGBoost beats Elo (the session's frozen best) by −0.0096 log-loss overall, and
beats it in BOTH tours independently** (ATP −0.0151, WTA −0.0040) — the first
candidate all session to clear that bar in both splits, not just in aggregate.
Beats the hierarchical model by −0.0242 overall. (Note: the hierarchical-model
comparison number here, 0.6267, is the same-vintage v25 snapshot the brief
pointed this task at — `hier_v25_eval_predictions.parquet`, written before round
9's ordering fix was applied to `hierarchical_model.py` itself; RESUME's own
current headline number for the CURRENT `hierarchical_model.py` state is 0.6275,
essentially the same. Using the exact files the brief named, not a refit, per its
explicit instruction not to touch that file this task.)

**Sanity checks before trusting this** (per `model-validation` skill):
tune-window cv log-loss (0.5992) and holdout log-loss (0.6025) are close — no
dramatic overfitting gap. Prediction distribution on holdout ranges 0.05-0.93 (no
absurd near-0/1 overconfidence, unlike the leakage-bug run's degenerate
predictions). Calibration by decile (10 bins) tracks `mean_p` against `mean_y`
closely across the full range — no systematic over/under-confidence in any band.
36 boosted rounds at depth 5 is a modest model, not an obviously overfit one.

### Feature importance (gain-based, final model)

| feature | total_gain | avg_gain | n_splits |
|---|---|---|---|
| `elo_overall_diff` | 8194.1 | 49.36 | 166 |
| `hier_dmean_diff` | 3487.2 | 21.80 | 160 |
| `elo_surface_diff` | 729.2 | 5.61 | 130 |
| `round` | 521.6 | 4.97 | 105 |
| `hier_dvar_sum` | 460.8 | 4.43 | 104 |
| `n_prior_diff` | 453.2 | 4.62 | 98 |
| `hier_dvar_diff` | 435.1 | 4.44 | 98 |
| `weeks_since_last_diff` | 416.7 | 5.48 | 76 |
| `tourney_level` | 406.9 | 5.09 | 80 |
| `h2h_diff` | 130.2 | 4.82 | 27 |
| `surface` | 55.9 | 3.49 | 16 |
| `best_of` | 22.7 | 4.54 | 5 |
| `is_elite` | 8.9 | 2.96 | 3 |

**Elo's overall rating diff carries by far the most raw signal** (total_gain ~2.4x
the runner-up), with the hierarchical model's own raw state (`hier_dmean_diff`)
a clear second — genuinely useful for the project's future direction, since it
confirms Elo and the hierarchical model's raw posterior mean are carrying mostly
non-redundant, complementary signal (both get meaningfully used), while surface
Elo, layoff, tier and H2H are real but much smaller contributors, and `best_of`/
`is_elite` are nearly unused.

## What's still open

- The stale-`diag_layoff.parquet` class of bug — any OTHER diagnostic artifact in
  `tune_scratch/` derived from `prepared_matches_cache.pkl` that predates the
  round-9 fix (21:56 that day) may carry the same staleness. Not audited
  exhaustively this task; anyone reusing an old `tune_scratch/diag*.parquet` for
  something round/date-order-sensitive should check its mtime against the cache's.
- No hyperparameter search was done on `max_depth`/`eta`/`subsample`/
  `colsample_bytree` (deliberately, per the brief) — only `num_boost_round`. A
  future task could try a proper `xgb.cv`-based hyperparameter search (still
  tune-window only) if more headroom is wanted, though the brief's own framing
  ("chosen default parameters tuned to optimal nrounds") suggests this was
  intentionally out of scope.
- Feature importance suggests Elo overall rating is still the single strongest
  input even inside XGBoost — a natural next step for the project's actual
  direction (per the task brief) would be improving Elo itself further (tier/
  round-weighted K, margin-of-victory — both flagged as open in issue 25) rather
  than continuing to invest in the hierarchical filter architecture, since XGBoost
  is already extracting most of the hierarchical model's own signal on top of Elo.

## Artifacts

- `dropshotdata/scripts/elo_raw_ratings_for_xgb.py`,
  `dropshotdata/scripts/build_xgb_features.py`,
  `dropshotdata/scripts/train_xgb.py` — the full pipeline, re-runnable.
- `dropshotdata/tune_scratch/xgb_v1_model.json` — the fitted model.
- `dropshotdata/tune_scratch/xgb_v1_feature_importance.csv` — full importance table.
- `dropshotdata/tune_scratch/xgb_features_tune.parquet` /
  `xgb_features_holdout.parquet` /`xgb_v1_holdout_predictions.parquet` — gitignored
  (parquet), regenerate via the scripts above if needed.
