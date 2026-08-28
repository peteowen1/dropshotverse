# 27 — XGBoost over Elo + hierarchical raw state + engineered features: BEATS Elo

Type: task
Status: closed (success — new best model on the frozen holdout, VALIDATED with a
correction — see "## Validation" below. A real leakage bug was found in the raw
`hier_dmean_diff`/`hier_dvar_*` feature source, fixed, and the pipeline re-run.
Corrected headline: ALL log-loss **0.6041** (not 0.6025), still beats Elo (0.6121)
in both tours independently. The "Diagnostic slicing pass" section further below
predates this fix (RETRACTED, see its own header) and its exact numbers are
superseded — see "## Validation" for the first correction pass, then
"## Diagnostic slicing pass v2 (corrected, re-run on post-fix predictions)" near the
end of the file for the full corrected cuts sweep and an explicit what-changed-vs-
what-held comparison against the retracted version. Headline of that v2 pass:
the retracted claim "Elo beats XGBoost in the top confidence bucket" does NOT hold
— XGBoost beats Elo in every market-confidence bucket on corrected data.)

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

## Diagnostic slicing pass (same day, follow-up) — where does XGBoost's remaining gap to market concentrate?

**RETRACTED/SUPERSEDED 2026-08-28 (later same day) — this entire section was run
against the pre-leakage-fix, stale `hier_dmean_diff` predictions (and, additionally,
against a stale pre-round-9-fix Elo reference joined via `diag2_enriched.parquet`).
Its per-bucket numbers are not reliable. Its one headline claim ("Elo beats XGBoost
in the top confidence bucket") does NOT hold on the corrected data. See
"## Diagnostic slicing pass v2 (corrected, re-run on post-fix predictions)" near the
end of this file for the re-run against the final, corrected
`xgb_v1_holdout_predictions.parquet` (post commit `9678fda`), with an explicit
what-changed-vs-what-held comparison. Kept below, unedited, for the record.**

Ran the same kind of cut-by-cut sweep issue 20's round-3 diagnostics used for the
hierarchical model, on the XGBoost holdout (`tune_scratch/xgb_v1_holdout_predictions
.parquet`, n=8,593, joined to `tune_scratch/diag2_enriched.parquet` for `p_mkt`/
`p_elo`/`p_winner_hier_mc`, and to `prepared_matches_cache.pkl`'s `score` column for
set-count). Read-only throughout; no pipeline files touched. All probabilities
converted to a common "P(side A wins)" convention using the same `y`-pivot trick
`train_xgb.py` already uses, so log-loss/accuracy/calibration are directly
comparable across XGBoost/Elo/Hier/Market in every cut below.

**Headline finding**: the gap to market is not spread flat. XGBoost is
**increasingly underconfident as match confidence rises**, and this concentrates in
the high market-confidence buckets (0.8–1.0) — present at nearly equal magnitude in
both tours (not a tour-specific defect), converges independently with the
straight-set-vs-deciding-set cut, and in the very top decile actually flips XGBoost
from beating Elo to *losing* to Elo on log-loss. It's the same DIRECTION as the
hierarchical model's well-documented MC-averaging undershoot, but roughly
half-to-two-thirds the magnitude — XGBoost has partially inherited the defect, not
escaped it.

### 1) Tour — market-conditioned calibration (not just aggregate log-loss)

| tour | bucket | n | XGBoost pred | XGBoost err | Elo pred | Elo err | Hier pred | Hier err | Market pred | Market err | realized |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ATP | (0.5,0.6] | 1357 | 0.5349 | +0.0318 | 0.5474 | +0.0193 | 0.5316 | +0.0351 | 0.5536 | +0.0130 | 0.5667 |
| ATP | (0.6,0.7] | 1265 | 0.6171 | +0.0391 | 0.6352 | +0.0209 | 0.6018 | +0.0543 | 0.6471 | +0.0090 | 0.6561 |
| ATP | (0.7,0.8] | 886  | 0.7019 | +0.0306 | 0.7339 | -0.0014 | 0.6680 | +0.0645 | 0.7475 | -0.0150 | 0.7325 |
| ATP | (0.8,0.9] | 565  | 0.7773 | +0.0599 | 0.8127 | +0.0245 | 0.7309 | +0.1063 | 0.8448 | -0.0076 | 0.8372 |
| ATP | (0.9,1.0] | 284  | 0.8650 | +0.0681 | 0.8951 | +0.0380 | 0.8088 | +0.1243 | 0.9398 | -0.0067 | 0.9331 |
| WTA | (0.5,0.6] | 1286 | 0.5388 | +0.0055 | 0.5490 | -0.0047 | 0.5285 | +0.0159 | 0.5528 | -0.0085 | 0.5443 |
| WTA | (0.6,0.7] | 1197 | 0.6176 | +0.0098 | 0.6362 | -0.0088 | 0.6022 | +0.0252 | 0.6465 | -0.0191 | 0.6274 |
| WTA | (0.7,0.8] | 934  | 0.7017 | +0.0435 | 0.7221 | +0.0231 | 0.6686 | +0.0766 | 0.7471 | -0.0020 | 0.7452 |
| WTA | (0.8,0.9] | 559  | 0.7804 | +0.0747 | 0.8084 | +0.0467 | 0.7362 | +0.1189 | 0.8443 | +0.0108 | 0.8551 |
| WTA | (0.9,1.0] | 260  | 0.8684 | +0.0662 | 0.8920 | +0.0426 | 0.8251 | +0.1095 | 0.9350 | -0.0004 | 0.9346 |

XGBoost's top-bucket error is +0.0681 (ATP) / +0.0662 (WTA) — essentially symmetric
between tours. Same story as issue 20's round-3 finding for hier: whatever is
driving underconfidence at the top is architecture-level, not tour-specific.

### 2) Surface

| surface | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| Clay  | 2338 | 0.5902 | 0.6083 | 0.6159 | 0.5752 | -0.0181 | +0.0150 |
| Grass | 1059 | 0.6148 | 0.6279 | 0.6479 | 0.5952 | -0.0131 | +0.0196 |
| Hard  | 5196 | 0.6055 | 0.6159 | 0.6272 | 0.5938 | -0.0104 | +0.0117 |

No concentrated surface defect — all three surfaces beat Elo, gap ranges only
-0.0104 to -0.0181, and n is comfortably large everywhere.

### 3) Round

| round | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| F    | 198  | 0.6296 | 0.6332 | 0.6532 | 0.6034 | -0.0036 | +0.0262 |
| Q2   | 2    | — | — | — | — | *n=2, too sparse to trust, excluded* | |
| QF   | 752  | 0.6121 | 0.6214 | 0.6431 | 0.6103 | -0.0093 | +0.0018 |
| R128 | 1509 | 0.5934 | 0.6113 | 0.6156 | 0.5713 | -0.0179 | +0.0221 |
| R16  | 1519 | 0.6055 | 0.6149 | 0.6362 | 0.5904 | -0.0094 | +0.0151 |
| R32  | 2704 | 0.6037 | 0.6162 | 0.6271 | 0.5894 | -0.0125 | +0.0143 |
| R64  | 1489 | 0.5894 | 0.6004 | 0.6031 | 0.5794 | -0.0110 | +0.0100 |
| RR   | 44   | 0.6035 | 0.6030 | 0.6246 | 0.5887 | *n=44, borderline sparse* +0.0005 | +0.0148 |
| SF   | 376  | 0.6357 | 0.6665 | 0.6759 | 0.6384 | **-0.0308** | -0.0027 |

**Does XGBoost inherit hier's R128 calibration weak spot?** Partially, but it looks
shared across every model, not XGBoost-specific:

| cut | bucket | n | XGBoost err | Elo err | Hier err | Market err | realized |
|---|---|---|---|---|---|---|---|
| R128 | (0.8,0.9] | 189 | +0.1028 | +0.1004 | +0.1634 | +0.0307 | 0.8730 |
| R128 | (0.9,1.0] | 145 | +0.0781 | +0.0746 | +0.1290 | +0.0020 | 0.9448 |
| non-R128 | (0.8,0.9] | 935 | +0.0601 | +0.0224 | +0.1022 | -0.0043 | 0.8406 |
| non-R128 | (0.9,1.0] | 399 | +0.0632 | +0.0277 | +0.1129 | -0.0058 | 0.9298 |

R128's top-two buckets are worse-calibrated for XGBoost than the rest of the draw
(+0.1028/+0.0781 vs +0.0601/+0.0632), but critically **Elo's own error in R128
(0.8,0.9] is +0.1004** — nearly identical to XGBoost's +0.1028 — and even the
**market** itself shows a real +0.0307 error there (vs ~0 everywhere else), unusual
for an external reference that's normally near-perfectly calibrated. That points to
R128 (0.8,0.9] specifically containing genuinely harder-to-calibrate matches for
everyone (likely qualifiers/wildcards/big seeding gaps with real variance no model
captures), not a defect unique to XGBoost or inherited specifically from hier. R128's
plain log-loss gap to Elo (-0.0179) is actually one of the *better* rounds, consistent
with this being a shared-difficulty round rather than an XGBoost-specific weakness.

### 4) Market-confidence bucket (the main ask — full table, pooled ATP+WTA)

| bucket | n | XGBoost pred | XGBoost err | Elo pred | Elo err | Hier pred | Hier err | Market pred | Market err | realized |
|---|---|---|---|---|---|---|---|---|---|---|
| (0.5,0.6] | 2643 | 0.5368 | +0.0190 | 0.5482 | +0.0076 | 0.5301 | +0.0258 | 0.5532 | +0.0026 | 0.5558 |
| (0.6,0.7] | 2462 | 0.6173 | +0.0248 | 0.6357 | +0.0065 | 0.6020 | +0.0402 | 0.6468 | -0.0046 | 0.6422 |
| (0.7,0.8] | 1820 | 0.7018 | +0.0372 | 0.7278 | +0.0112 | 0.6683 | +0.0707 | 0.7473 | -0.0083 | 0.7390 |
| (0.8,0.9] | 1124 | 0.7788 | +0.0672 | 0.8105 | +0.0356 | 0.7336 | +0.1125 | 0.8445 | +0.0016 | 0.8461 |
| (0.9,1.0] | 544  | 0.8666 | +0.0672 | 0.8936 | +0.0402 | 0.8166 | +0.1172 | 0.9375 | -0.0037 | 0.9338 |

Same table with log-loss (not just calibration), so both views are on record:

| bucket | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| (0.5,0.6] | 2643 | 0.6954 | 0.7236 | 0.7081 | 0.6871 | -0.0282 | +0.0083 |
| (0.6,0.7] | 2462 | 0.6597 | 0.6684 | 0.6774 | 0.6475 | -0.0087 | +0.0122 |
| (0.7,0.8] | 1820 | 0.5858 | 0.5967 | 0.6253 | 0.5726 | -0.0109 | +0.0132 |
| (0.8,0.9] | 1124 | 0.4496 | 0.4508 | 0.4752 | 0.4277 | -0.0012 | +0.0219 |
| (0.9,1.0] | 544  | 0.2642 | 0.2507 | 0.3187 | 0.2346 | **+0.0135** | +0.0296 |

XGBoost's calibration error is monotonically increasing with market confidence
(+0.0190 → +0.0672), always positive (underconfident, never overconfident), and by
the last bucket its log-loss advantage over Elo has **inverted**: XGBoost 0.2642 vs
Elo 0.2507 — Elo wins the highest-confidence 544 rows despite losing everywhere else.
This n (544) is not sparse. XGBoost's error is consistently ~1.5–2x Elo's own error
in the same bucket (+0.0672 vs +0.0402 at the top), so this is not simply "every
model undershoots the market here" — XGBoost's fit is *specifically* leaving edge on
the table relative to Elo alone, in exactly the region a genuinely stronger model
should be most confident.

### 5) Player experience (`n_prior_diff` = sideA − sideB, and its magnitude)

| n_prior_diff quintile | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| most negative | 1720 | 0.6049 | 0.6231 | -0.0182 |
| | 1722 | 0.6023 | 0.6107 | -0.0084 |
| middle | 1714 | 0.5995 | 0.6096 | -0.0101 |
| | 1718 | 0.6062 | 0.6179 | -0.0117 |
| most positive | 1719 | 0.5995 | 0.6152 | -0.0157 |

| \|n_prior_diff\| quintile (experience gap magnitude) | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| smallest gap | 1731 | 0.6008 | 0.6104 | -0.0096 |
| | 1713 | 0.5984 | 0.6092 | -0.0108 |
| | 1725 | 0.5918 | 0.6015 | -0.0097 |
| largest gap | 1708 | 0.6117 | 0.6363 | -0.0246 |
| | 1716 | 0.6099 | 0.6194 | -0.0095 |

No clean monotonic lever — confirms the earlier hier finding ("noisy, no clear
lever") transfers to XGBoost too, despite XGBoost having direct access to H2H and
layoff that hier didn't use. All bins are large (n≈1700+), so this is a real null
result, not a sparsity artifact.

### 6) Best-of and match length

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| Bo3 | 7747 | 0.6107 | 0.6235 | -0.0128 |
| Bo5 | 846  | 0.5270 | 0.5400 | -0.0130 |

Bo3 vs Bo5 alone shows no asymmetry (both ≈-0.013). But splitting by sets actually
played (joined from `prepared_matches_cache.pkl`'s `score` column, same method
`diag2_bo5_sets.py` used for hier) shows the asymmetry hier found **does transfer**:

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| Bo3: straight 2 sets | 5005 | 0.5690 | 0.5765 | -0.0075 |
| Bo3: went 3 sets     | 2742 | 0.6869 | 0.7093 | **-0.0224** |
| Bo5: straight 3 sets | 360  | 0.4341 | 0.4410 | -0.0069 |
| Bo5: went 4-5 sets   | 486  | 0.5959 | 0.6133 | **-0.0174** |

XGBoost's edge over Elo is consistently *smaller* on decisive straight-set wins and
*larger* on matches that go the distance — same direction as hier's finding, and
consistent with the market-bucket result above (straight-set blowouts are
disproportionately the high-market-confidence bucket where XGBoost underperforms
relative to Elo). Independent corroboration of the same underlying weak spot from a
different angle, all n well above 50.

### 7) Tournament tier

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| is_elite=False | 4281 | 0.6174 | 0.6329 | -0.0155 |
| is_elite=True  | 4312 | 0.5877 | 0.5979 | -0.0102 |

Mostly a wash, as hier found — elite events show a slightly smaller gap, not a
meaningful asymmetry.

### Synthesis

Unlike the hierarchical model's "WTA carries the whole gap" finding, XGBoost's
remaining gap-to-market is **not tour-concentrated, not surface-concentrated, and
not tier-concentrated**. Player-experience cuts are flat/noisy (a genuine null).
Round is mostly flat except R128, and R128 looks like a shared cross-model
difficulty (even the market mis-calibrates there), not an XGBoost-specific defect.

The one real, concentrated, non-noisy weak spot is **underconfidence in the
high-market-confidence region (roughly p≥0.8)**, converging from three independent
angles: the market-confidence-bucket table directly, the tour split (present equally
in ATP and WTA), and the straight-set-vs-deciding-set split (both Bo3 and Bo5). It's
the same *direction* as the hierarchical model's well-diagnosed MC-averaging
undershoot (issue 20, round 3), but roughly half-to-two-thirds the magnitude at the
top bucket (+0.067 vs hier's +0.117) — XGBoost has inherited a smaller version of it,
not escaped it, even though the feature it consumes (`hier_dmean_diff`, the raw
posterior mean) was shown in issue 20 to extremize *correctly* on its own. The most
plausible reading: this isn't purely hier's convexity bug leaking through — a modest
36-round, depth-5 GBM fit with subsampling, selected via early stopping on
*aggregate* log-loss (which is dominated by the bulk of competitive matches, not the
tail), has little explicit incentive to push predictions all the way to the extremes,
so some shrinkage-toward-the-middle in the top decile is an expected generic GBM
behavior on top of whatever hier contributes.

**Plausible next steps, in rough order of cost:**
1. Cheapest: a post-hoc recalibration curve (isotonic or a simple logistic
   temperature/slope correction) fit ONLY on the tune-window's high-confidence tail,
   to re-extremize XGBoost's top-decile output without touching the rest of the
   range — same class of fix already explored (and rejected in its flat/global form)
   for hier in issue 20, but scoped narrowly here it may transfer better since the
   defect is smaller and cleanly bucket-localized.
2. Feed Elo's own already-blended `p_winner` (not just the raw rating diff) as an
   additional feature, so the trees have direct access to Elo's own well-calibrated
   extremization shape in the tail, rather than having to reconstruct it through
   XGBoost's own logistic link on `elo_overall_diff`.
3. A modest `num_boost_round`/`eta` re-tune biased toward more rounds at lower
   learning rate (still tune-window only, still no full grid search) might let the
   model extremize further before early-stopping's aggregate-log-loss criterion cuts
   it off — worth a quick check before more invasive fixes.

Otherwise, this is a genuinely different situation than the hierarchical model's
concentrated defect: the residual gap to market (0.0136 overall) is real but mostly
flat and well-distributed across tour/surface/tier/experience, with one identifiable,
moderate-sized, and independently-corroborated soft spot in the confident tail —
worth fixing, but not evidence of a deeper architectural problem the way hier's WTA/
MC-convexity finding was.

### Artifacts (this diagnostic pass)

Analysis scripts were run from a scratch location outside the repo (read-only,
ad hoc, not committed as pipeline artifacts): they load
`tune_scratch/xgb_v1_holdout_predictions.parquet`, `tune_scratch/diag2_enriched
.parquet`, and `prepared_matches_cache.pkl` and reuse the exact `y`-pivot convention
`train_xgb.py` already established for cross-model comparability. No new persistent
files were written to `dropshotdata/`.

## Validation (independent post-hoc audit, per `model-validation`/`stats-discipline`)

A ~0.01 log-loss jump on a first attempt, after ten rounds where nothing else this
session moved the metric by more than ~0.003, is exactly the "suspicious agreement"
pattern `stats-discipline` says to distrust by default. This section is an
adversarial audit of the 0.6025 result — trying to break it, not confirm it — done
independently of (and largely before discovering) the "Diagnostic slicing pass"
section above.

### 1. Leakage hunt

**A real, second leakage bug found — same class as the already-documented
`diag_layoff.parquet` staleness, this time in the feature build's top-2 feature.**
`build_xgb_features.py`'s `HIER_TUNE`/`HIER_HOLDOUT` pointed at
`hier_v25_eval_predictions.parquet` (mtime 16:02) and `tune_scratch/v25_tune_preds
.parquet` (mtime 15:44) — **both written hours before round 9's causal-ordering fix**
to `hierarchical_model.py` (commit `111ecce`, 21:55) and the cache it regenerated
(21:56). The doc above already noted these files predated the fix, but treated it
only as a "same-vintage comparison-fairness" footnote — it missed that these same
files are also this pipeline's FEATURE SOURCE for `d1_mean`/`d1_var`/`d2_mean`/
`d2_var` (→ `hier_dmean_diff`, the #2 feature by gain). Since `hierarchical_model
.py`'s `run()` updates each player's posterior state sequentially in archive-file
order, the pre-fix same-date multi-round misordering (e.g. R32 processed after R16
on a shared date, so a player's pre-R32 state had already absorbed their R16 result)
is a genuine look-ahead leak into these features, not just a staleness/fairness
issue.

**Quantified**: regenerated both files against the current, fixed `hierarchical_model
.py` + corrected `prepared_matches_cache.pkl` (`scripts/regen_hier_v25_postfix.py`,
a read-only subprocess re-run with the frozen v25 hyperparameters — no edit to
`hierarchical_model.py`, same pattern as `tune_scratch/reveal_v26_cond_var_scale.py`
already used all session). The shift is large and pervasive, not a few outlier
rows: **93.8% of holdout rows moved `d1_mean` by >0.01, 50.2% moved by >0.1** (median
shift ≈1/3 of `hier_dmean_diff`'s own typical magnitude). Rebuilt the XGBoost
feature set from the corrected files and re-ran `xgb.cv`/final fit unchanged:

| | XGBoost (ORIGINAL, leaked) | XGBoost (CORRECTED) | Elo (unchanged) |
|---|---|---|---|
| ALL log-loss | 0.6025 | **0.6041** | 0.6121 |
| ATP log-loss | 0.6023 | **0.6063** | 0.6174 |
| WTA log-loss | 0.6027 | **0.6019** | 0.6067 |
| ALL accuracy | 0.6662 | 0.6682 | 0.6530 |

**Verdict: the leak was real and the original number was overstated by ~0.0016
log-loss (~17% of the claimed 0.0096 edge over Elo) — but the core finding
survives.** XGBoost still beats Elo overall and in BOTH tours independently after
correction (ATP −0.0111, WTA −0.0048, vs the original −0.0151/−0.0040). Feature
importance ranking is unchanged and `hier_dmean_diff`'s gain actually rose slightly
post-fix (3487→4032) — the corrected posterior state is *more* useful to the model,
not less, so this was a real leak inflating the aggregate number, not the source of
XGBoost's edge over Elo. **`build_xgb_features.py` has been fixed** (points at the
regenerated `tune_scratch/v25_postfix_{tune,holdout}_preds.parquet`) and the
official pipeline artifacts (`xgb_features_{tune,holdout}.parquet`,
`xgb_v1_model.json`, `xgb_v1_feature_importance.csv`,
`xgb_v1_holdout_predictions.parquet`) have been regenerated with the fix — the
numbers now written to disk are the corrected ones. The stale
`hier_v25_eval_predictions.parquet`/`v25_tune_preds.parquet` are left on disk
(other diagnostics reference them as a documented "same vintage" snapshot) but are
no longer used as this pipeline's feature source.

**H2H**: hand-verified by replicating `compute_h2h()` and manually recomputing
cumulative win counts on the archive's single most-frequent pair (80 real
meetings, 1973–1988) — every row's `h2h_prior_wins_winner`/`_loser` matched a hand
tally exactly, confirming strictly-prior-only counting with no self-inclusion.

**Side assignment**: `side_a_is_winner()` hashes only `(date, sorted uid pair)` —
never touches winner/loser identity, rating, or any outcome-derived quantity — so
it is *provably*, not just empirically, outcome-blind; it cannot correlate with
rating or win probability by construction.

**Layoff fix re-verification**: `diag_layoff.parquet` (mtime 23:07) postdates the
corrected cache (21:56) — the earlier round-9-adjacent fix is still correctly
applied and was the only consumer of that file in this pipeline.

**Other intermediate files checked for the same staleness class**: `elo_raw_ratings_
full.parquet` (rebuilt fresh this task from raw CSVs via its own copy of
`prepare()`, already includes the `(date, round_rank)` fix, independent of the
cache) — clean. `prepared_matches_cache.pkl` (21:56) — post-fix. `tune_scratch/
diag2_enriched.parquet` (mtime 16:35, pre-fix) is used only for the market-odds
row-selection key and the "Hierarchical (v25/mc60)" comparison column in
`train_xgb.py`'s report — NOT as an XGBoost feature source — so it doesn't affect
the model, only the labeling of the comparison baseline (already flagged in the
original writeup as an intentional same-vintage comparison).

**Outlier predictions**: sorted the corrected holdout by `p_xgb`, hand-inspected
the 15 most extreme rows on each end. All correspond to plausible blowouts
(|elo_overall_diff| 380–950, `hier_dmean_diff` aligned in the same direction — top
players vs qualifiers/big seeding gaps), predictions range 0.052–0.960 (no
degenerate 0/1 collapse), and accuracy at the extremes (p<0.1 or p>0.9, n=307) is
93.5% — high because these ARE lopsided matches, not the ~100% "perfect
separation" signature that would indicate leakage.

**Feature-importance stability**: refit with the top 1% most extreme rows (by
`|elo_overall_diff|` or `|hier_dmean_diff|`) removed from the tune window (both
pre- and post-fix feature sets) — `elo_overall_diff` stays #1 and `hier_dmean_diff`
stays #2 by total_gain in every case, confirming the dominance ranking is a
population-level pattern, not an artifact of a handful of rows.

### 2. Calibration vs the market

**Self-referential (symmetric-augmented) check**, replicating the method already
used for the hierarchical model: `calibration_report.py` on the corrected holdout
(n=17,186 augmented rows) — **ECE 0.0095, Brier 0.2091, log-loss 0.6041**. The
10-bin reliability table tracks closely across the full range (e.g. bin
(0.9,1.0]: mean_p=0.928 vs observed=0.935; bin (0,0.1]: mean_p=0.072 vs
observed=0.065) — by this view alone, XGBoost looks well-calibrated.

**Market-conditioned check (the more reliable one — bucket by the MARKET's own
confidence, not the model's)**, replicating `diag2_fresh_cuts.py`'s method exactly,
with XGBoost added as a 4th column alongside Elo/Hier/Market (n=8,593 market-odds
rows):

| market bucket | n | XGBoost pred | XGBoost err | Elo err | Hier err | Market err | realized |
|---|---|---|---|---|---|---|---|
| (0.5,0.6] | 2625 | 0.573 | -0.045 | -0.088 | -0.079 | -0.001 | 0.528 |
| (0.6,0.7] | 2462 | 0.629 | -0.002 | -0.039 | -0.042 | -0.005 | 0.627 |
| (0.7,0.8] | 1820 | 0.714 | +0.024 | -0.008 | -0.004 | -0.008 | 0.737 |
| (0.8,0.9] | 1124 | 0.798 | **+0.048** | +0.027 | +0.099 | +0.002 | 0.846 |
| (0.9,1.0] | 544  | 0.886 | **+0.048** | +0.036 | +0.115 | -0.004 | 0.934 |

Log-loss in the same buckets (post-fix): (0.8,0.9] XGBoost **0.443** vs Elo 0.451 vs
Hier 0.471 vs Market 0.428; (0.9,1.0] XGBoost **0.248** vs Elo 0.251 vs Hier 0.317 vs
Market 0.235.

**Finding: XGBoost IS underconfident in the market's high-confidence tail — a real,
smaller, inherited version of the hierarchical model's well-documented MC-averaging
undershoot** (error +0.048 vs Hier's +0.099/+0.115, roughly 40-45% of Hier's
miscalibration in the same buckets) **but worse than Elo's own tail calibration**
(+0.048 vs Elo's +0.027/+0.036, i.e. ~1.3-1.8x Elo's error). This matches the
"Diagnostic slicing pass" section's headline finding in direction and rough
magnitude. Where it differs materially: that section (run on the pre-fix, leaked
predictions) found XGBoost's top-decile log-loss **inverts and loses to Elo**
(0.2642 vs 0.2507). **On the corrected, post-fix predictions this inversion does
NOT hold** — XGBoost 0.248 vs Elo 0.251 in (0.9,1.0], and 0.443 vs 0.451 in
(0.8,0.9] — XGBoost beats or ties Elo in every market-confidence bucket, including
the top one. The underconfidence is real and the calibration ERROR is worse than
Elo's in the tail, but it isn't (post-fix) costly enough in log-loss terms to flip
the ranking, because XGBoost's raw accuracy in that region is high enough to
compensate for being underconfident. The rest of the "Diagnostic slicing pass"
section (tour/surface/round/experience/best-of/tier cuts) was run on the pre-fix
predictions and not re-verified here; the underlying generic-GBM-shrinkage
mechanism it proposes is independent of the hier leak and plausibly still holds in
direction, but the exact numbers there should be treated as indicative, not exact,
until re-run against the corrected `xgb_v1_holdout_predictions.parquet`.

**Answering the two questions directly**:
- **Does 0.6025 survive scrutiny as-is? No** — it was overstated by a real,
  identified leak; the corrected, honest number is **0.6041**, still a genuine win
  over Elo in both tours.
- **Is XGBoost well-calibrated in the market's high-confidence tail specifically?
  Partially.** It is measurably underconfident there (worse calibration error than
  Elo alone), but that underconfidence does not cost it the tail on log-loss —
  XGBoost remains at least as good as Elo, and much better than the hierarchical
  model, in every market-confidence bucket including the top one, post-correction.
  This is a real, worthwhile target for a future narrow recalibration (per the
  slicing pass's own suggestion #1), not a reason to distrust the headline result.

### Artifacts (this validation pass)

- `dropshotdata/scripts/regen_hier_v25_postfix.py` — new, permanent, re-runnable:
  regenerates the hierarchical model's raw-state predictions against the current
  (post round-9-fix) code+cache. Writes `tune_scratch/v25_postfix_tune_preds
  .parquet` / `v25_postfix_holdout_preds.parquet` (gitignored).
- `dropshotdata/scripts/build_xgb_features.py` — fixed to point at the regenerated
  post-fix files instead of the stale `hier_v25_eval_predictions.parquet`/
  `v25_tune_preds.parquet`.
- `dropshotdata/tune_scratch/xgb_features_{tune,holdout}.parquet`, `xgb_v1_model
  .json`, `xgb_v1_feature_importance.csv`, `xgb_v1_holdout_predictions.parquet` —
  all regenerated with the fix; these are now the corrected, current numbers.
- `dropshotdata/tune_scratch/xgb_calibration_report.json` — self-referential
  calibration report (ECE/Brier/log-loss/reliability table) on the corrected model.

## Diagnostic slicing pass v2 (corrected, re-run on post-fix predictions, 2026-08-28 later same day)

Re-run of the retracted "Diagnostic slicing pass" section above, this time against
the CORRECTED, final artifacts: `tune_scratch/xgb_v1_holdout_predictions.parquet`
(mtime 23:22, written by `train_xgb.py` after the `hier_dmean_diff` leak fix — i.e.
after the regeneration underlying commit `9678fda`; n=8,593, confirmed identical to
the file the "## Validation" section scored), joined to `diag2_enriched.parquet` for
`p_mkt`/`p_winner_hier_mc` and to `prepared_matches_cache.pkl`'s `score` column for
set-count, same as the original pass. Read-only throughout.

**A second staleness bug caught before trusting any number here**: `diag2_enriched
.parquet`'s own `p_elo` column (mtime 16:35) also predates the round-9
causal-ordering fix (21:55/21:56) — using it gives Elo log-loss 0.6153 on this
holdout, not the documented 0.6121. This means the ORIGINAL retracted sweep's
per-bucket Elo reference numbers were themselves drawn from a stale Elo file, not
just the XGBoost side. Fixed here by sourcing Elo from the current
`elo_eval_predictions.parquet` (mtime 22:43, post-fix) directly, exactly as
`train_xgb.py` itself does. **Sanity check**: with this fix, this re-run's overall
numbers reproduce the doc's headline exactly — ALL XGBoost 0.6041, Elo 0.6121, ATP
XGBoost 0.6063/Elo 0.6174, WTA XGBoost 0.6019/Elo 0.6067 — confirming both the
correct predictions file and the correct Elo reference are in use below.

### 1) Market-confidence bucket (the main ask — full table, pooled ATP+WTA)

| bucket | n | XGBoost pred | XGBoost err | Elo pred | Elo err | Hier pred | Hier err | Market pred | Market err | realized |
|---|---|---|---|---|---|---|---|---|---|---|
| (0.5,0.6] | 2625 | 0.5377 | +0.0151 | 0.5439 | +0.0089 | 0.5305 | +0.0222 | 0.5536 | -0.0008 | 0.5528 |
| (0.6,0.7] | 2462 | 0.6224 | +0.0198 | 0.6281 | +0.0141 | 0.6020 | +0.0402 | 0.6468 | -0.0046 | 0.6422 |
| (0.7,0.8] | 1820 | 0.7130 | +0.0260 | 0.7192 | +0.0198 | 0.6683 | +0.0707 | 0.7473 | -0.0083 | 0.7390 |
| (0.8,0.9] | 1124 | 0.7982 | +0.0479 | 0.8035 | +0.0426 | 0.7336 | +0.1125 | 0.8445 | +0.0016 | 0.8461 |
| (0.9,1.0] | 544  | 0.8860 | +0.0478 | 0.8904 | +0.0434 | 0.8166 | +0.1172 | 0.9375 | -0.0037 | 0.9338 |

Same table with log-loss:

| bucket | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| (0.5,0.6] | 2625 | 0.6989 | 0.7146 | 0.7072 | 0.6871 | -0.0157 | +0.0118 |
| (0.6,0.7] | 2462 | 0.6685 | 0.6694 | 0.6774 | 0.6475 | -0.0009 | +0.0210 |
| (0.7,0.8] | 1820 | 0.5857 | 0.5927 | 0.6253 | 0.5726 | -0.0070 | +0.0131 |
| (0.8,0.9] | 1124 | 0.4426 | 0.4504 | 0.4752 | 0.4277 | -0.0078 | +0.0149 |
| (0.9,1.0] | 544  | 0.2484 | 0.2519 | 0.3187 | 0.2346 | **-0.0035** | +0.0138 |

**What changed vs the retracted sweep — this is the headline correction**: in the
top bucket, XGBoost log-loss is **0.2484 vs Elo's 0.2519 — XGBoost still wins**
(gap -0.0035). The retracted sweep's inversion claim (XGBoost 0.2642 losing to Elo
0.2507) does not hold anywhere in the corrected table; XGBoost beats or ties Elo in
every single market-confidence bucket, confirming the "## Validation" section's
finding (XGBoost 0.248 vs Elo 0.251 in that section's own independently-computed
version of this same check).

**What's more nuanced than either the retracted sweep or the Validation section
suggested**: XGBoost's calibration error is still positive (underconfident) and
still rises with market confidence (+0.0151 → +0.0198 → +0.0260 → +0.0479 → +0.0478,
essentially flat between the last two buckets rather than still climbing). But
**Elo's own error in the top bucket also rose once the correct, current Elo file is
used** (+0.0402 in the retracted sweep's stale-Elo table → +0.0434 here). The result:
XGBoost's top-bucket error is now only about **1.1x** Elo's own error (+0.0478 vs
+0.0434), not the **1.5-2x** the retracted sweep claimed (+0.0672 vs +0.0402). Both
models are underconfident in the tail by a similar amount on the corrected data —
this reads much less like "XGBoost is specifically leaving edge on the table
relative to Elo" and much more like "the whole tail is a hard region for anything
that isn't the market," matching what R128's top buckets independently show below.

### 2) Tour — market-conditioned calibration by tour

| tour | bucket | n | XGBoost pred | XGBoost err | Elo pred | Elo err | Hier pred | Hier err | Market pred | Market err |
|---|---|---|---|---|---|---|---|---|---|---|
| ATP | (0.5,0.6] | 1343 | 0.5360 | +0.0261 | 0.5440 | +0.0181 | 0.5321 | +0.0301 | 0.5542 | +0.0080 |
| ATP | (0.6,0.7] | 1265 | 0.6216 | +0.0345 | 0.6295 | +0.0266 | 0.6018 | +0.0543 | 0.6471 | +0.0090 |
| ATP | (0.7,0.8] | 886  | 0.7127 | +0.0198 | 0.7262 | +0.0063 | 0.6680 | +0.0645 | 0.7475 | -0.0150 |
| ATP | (0.8,0.9] | 565  | 0.7983 | +0.0389 | 0.8067 | +0.0305 | 0.7309 | +0.1063 | 0.8448 | -0.0076 |
| ATP | (0.9,1.0] | 284  | 0.8857 | +0.0474 | 0.8924 | +0.0407 | 0.8088 | +0.1243 | 0.9398 | -0.0067 |
| WTA | (0.5,0.6] | 1282 | 0.5393 | +0.0036 | 0.5437 | -0.0008 | 0.5289 | +0.0140 | 0.5530 | -0.0101 |
| WTA | (0.6,0.7] | 1197 | 0.6232 | +0.0042 | 0.6265 | +0.0009 | 0.6022 | +0.0252 | 0.6465 | -0.0191 |
| WTA | (0.7,0.8] | 934  | 0.7134 | +0.0318 | 0.7125 | +0.0327 | 0.6686 | +0.0766 | 0.7471 | -0.0020 |
| WTA | (0.8,0.9] | 559  | 0.7981 | +0.0570 | 0.8002 | +0.0549 | 0.7362 | +0.1189 | 0.8443 | +0.0108 |
| WTA | (0.9,1.0] | 260  | 0.8864 | +0.0483 | 0.8883 | +0.0463 | 0.8251 | +0.1095 | 0.9350 | -0.0004 |

**Holds unchanged**: top-bucket underconfidence is still essentially symmetric
between tours (ATP +0.0474, WTA +0.0483) — still architecture-level, not
tour-specific. **New nuance**: in WTA specifically, Elo's own error is now nearly
identical to XGBoost's at every bucket (e.g. (0.8,0.9]: XGBoost +0.0570 vs Elo
+0.0549) — the corrected Elo file makes WTA's story "both models similarly
underconfident," not "XGBoost worse than Elo."

### 3) Surface

| surface | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| Clay  | 2338 | 0.5950 | 0.6043 | 0.6159 | 0.5752 | -0.0093 | +0.0198 |
| Grass | 1059 | 0.6201 | 0.6243 | 0.6479 | 0.5952 | -0.0042 | +0.0249 |
| Hard  | 5196 | 0.6050 | 0.6131 | 0.6272 | 0.5938 | -0.0082 | +0.0112 |

**Holds, narrower**: no concentrated surface defect, same as before — but the
XGBoost−Elo gap is smaller everywhere now (-0.0042 to -0.0093, vs the retracted
sweep's -0.0104 to -0.0181), because both the XGBoost fix and the Elo-file fix
moved the two models slightly closer together. Still comfortably large n
everywhere, still all three surfaces favor XGBoost.

### 4) Round

| round | n | XGBoost ll | Elo ll | Hier ll | Market ll | gap XGB−Elo | gap XGB−Mkt |
|---|---|---|---|---|---|---|---|
| F    | 198  | 0.6171 | 0.6228 | 0.6532 | 0.6034 | -0.0057 | +0.0137 |
| Q2   | 2    | — | — | — | — | *n=2, too sparse to trust, excluded* | |
| QF   | 752  | 0.6173 | 0.6183 | 0.6431 | 0.6103 | -0.0010 | +0.0070 |
| R128 | 1509 | 0.5964 | 0.6070 | 0.6156 | 0.5713 | -0.0106 | +0.0251 |
| R16  | 1519 | 0.6052 | 0.6120 | 0.6362 | 0.5904 | -0.0068 | +0.0149 |
| R32  | 2704 | 0.6050 | 0.6134 | 0.6271 | 0.5894 | -0.0084 | +0.0156 |
| R64  | 1489 | 0.5909 | 0.5999 | 0.6031 | 0.5794 | -0.0090 | +0.0115 |
| RR   | 44   | 0.5619 | 0.5998 | 0.6246 | 0.5887 | *n=44, borderline sparse* -0.0379 | -0.0268 |
| SF   | 376  | 0.6499 | 0.6568 | 0.6759 | 0.6384 | -0.0069 | +0.0115 |

R128 top-two buckets vs the rest of the draw:

| cut | bucket | n | XGBoost err | Elo err | Hier err | Market err | realized |
|---|---|---|---|---|---|---|---|
| R128 | (0.8,0.9] | 189 | +0.0831 | +0.1066 | +0.1634 | +0.0307 | 0.8730 |
| R128 | (0.9,1.0] | 145 | +0.0514 | +0.0751 | +0.1290 | +0.0020 | 0.9448 |
| non-R128 | (0.8,0.9] | 935 | +0.0408 | +0.0297 | +0.1022 | -0.0043 | 0.8406 |
| non-R128 | (0.9,1.0] | 399 | +0.0465 | +0.0319 | +0.1129 | -0.0058 | 0.9298 |

**Changed in a way that strengthens the original conclusion**: with the correct Elo
file, Elo's own error in R128's top buckets (+0.1066/+0.0751) is now noticeably
**worse than XGBoost's** (+0.0831/+0.0514) — the reverse of the retracted sweep,
which had them nearly tied (+0.1004 vs +0.1028). R128 remains a shared hard region
(the market itself still shows elevated error there, +0.0307 vs ~0 elsewhere) but
it is now more clearly NOT an XGBoost-specific weak spot — if anything XGBoost
handles it relatively better than Elo does on the corrected data.

### 5) Player experience (`n_prior_diff`)

| n_prior_diff quintile | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| most negative | 1720 | 0.6017 | 0.6158 | -0.0141 |
| | 1722 | 0.6000 | 0.6113 | -0.0114 |
| middle | 1714 | 0.6060 | 0.6062 | -0.0002 |
| | 1718 | 0.6129 | 0.6151 | -0.0022 |
| most positive | 1719 | 0.6002 | 0.6122 | -0.0120 |

| \|n_prior_diff\| quintile (experience gap magnitude) | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| smallest gap | 1731 | 0.6069 | 0.6071 | -0.0002 |
| | 1716 | 0.6136 | 0.6209 | -0.0072 |
| | 1713 | 0.5990 | 0.6065 | -0.0076 |
| largest gap | 1725 | 0.5931 | 0.5996 | -0.0065 |
| | 1708 | 0.6081 | 0.6267 | -0.0186 |

**Holds**: still no clean monotonic lever, still a genuine null (all bins n≈1700+).
Gap magnitudes shrank a little across the board (consistent with the Elo-file fix
narrowing the general XGBoost−Elo gap), but the qualitative finding — no usable
experience-based lever — is unchanged.

### 6) Best-of and match length

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| Bo3 | 7747 | 0.6133 | 0.6208 | -0.0075 |
| Bo5 | 846  | 0.5203 | 0.5325 | -0.0121 |

By sets actually played:

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| Bo3: straight 2 sets | 5005 | 0.5700 | 0.5764 | -0.0064 |
| Bo3: went 3 sets     | 2742 | 0.6923 | 0.7019 | **-0.0095** |
| Bo5: straight 3 sets | 360  | 0.4179 | 0.4329 | **-0.0151** |
| Bo5: went 4-5 sets   | 486  | 0.5963 | 0.6062 | -0.0100 |

**Changed — this is the one piece of the retracted sweep's corroboration that does
NOT clearly survive.** The retracted sweep's story was "XGBoost's edge over Elo is
smaller on straight-set blowouts, larger on matches that go the distance" — offered
as independent corroboration of the top-bucket underconfidence finding. On the
corrected data, Bo3 still points that direction (straight-2 gap -0.0064 <
went-3 gap -0.0095), but **Bo5 now points the other way**: straight-3-set gap
(-0.0151) is *larger* than the went-4-5-sets gap (-0.0100) — the opposite of the
original claim. Both n are moderate (360/486), not sparse, so this isn't a noise
artifact from small samples, but it does mean the "goes the distance" story is now
mixed rather than a clean three-way corroboration — Bo5 alone no longer supports it.

### 7) Tournament tier

| cut | n | XGBoost ll | Elo ll | gap XGB−Elo |
|---|---|---|---|---|
| is_elite=False | 4281 | 0.6192 | 0.6293 | -0.0101 |
| is_elite=True  | 4312 | 0.5891 | 0.5951 | -0.0059 |

**Holds**: still mostly a wash, elite events still show a slightly smaller gap, same
as before (retracted sweep: -0.0155/-0.0102; here: -0.0101/-0.0059 — same shape,
smaller absolute values).

### Synthesis — what changed, what held, revised recommendation

**What changed** (the corrected-data picture is genuinely different from the
retracted sweep, not just re-labeled):
1. **The headline "Elo beats XGBoost in the top bucket" claim is false on corrected
   data.** XGBoost beats or ties Elo in every single market-confidence bucket,
   including the top one (0.2484 vs 0.2519).
2. **XGBoost's top-bucket miscalibration is only mildly worse than Elo's own**
   (~1.1x, not ~1.5-2x) — the corrected Elo file (which itself needed a
   round-9-fix refresh, a second staleness bug caught in the course of this re-run)
   shows Elo is nearly as underconfident in the tail as XGBoost is. This weakens the
   "XGBoost specifically leaves edge on the table relative to Elo" framing; it now
   looks more like a shared hard-tail problem that XGBoost has inherited only
   slightly more of, not a lot more of.
3. **R128's top buckets now favor XGBoost over Elo**, reversing the near-tie the
   retracted sweep found — one of the corroborating angles for "XGBoost has a
   defect" has flipped to actively arguing against a defect.
4. **The Bo5 set-count corroboration is now mixed**, not clean — straight-set Bo5
   matches show a *larger* XGBoost-Elo gap than distance-going Bo5 matches, opposite
   the retracted claim (Bo3 still points the original direction).

**What held**:
- No concentrated tour, surface, tier, or player-experience defect — all flat/noisy,
  same conclusion as before, just with smaller absolute gaps (because the Elo
  reference itself moved slightly).
- XGBoost's calibration error is still positive (underconfident) and still highest
  in the top two market-confidence buckets, in both tours roughly symmetrically.
- R128 remains a genuinely harder region for every model, market included — that
  conclusion is unchanged and, if anything, better supported now.

**Revised recommendation**: the case for the retracted sweep's #1 suggestion
(a narrow, tail-only post-hoc recalibration aimed at fixing an "XGBoost-specific"
defect) is **weaker** now — the defect looks shared with Elo, not XGBoost-specific,
so a recalibration narrowly targeting XGBoost's tail risks fixing something that
isn't really XGBoost's problem alone. The residual gap to MARKET at the top
(XGBoost 0.2484 vs Market 0.2346, +0.0138) is still real and worth closing, but the
better-motivated next step is still recommendation #2 from the original writeup —
**feed Elo's own already-blended `p_winner` as an additional XGBoost feature**
(not just the raw rating diff) — since Elo's own extremization shape is available
and cheap to hand to the trees directly, rather than asking XGBoost to reconstruct
it. A cheap general (not top-decile-only) recalibration check is still reasonable
to try given both models underextremize similarly, but framing it as fixing an
"XGBoost-specific inherited hier defect" is no longer well supported by this
corrected pass.

### Artifacts (this v2 pass)

Read-only, ad hoc, run from a scratch location outside the repo. Loads
`tune_scratch/xgb_v1_holdout_predictions.parquet` (corrected, post-fix),
`elo_eval_predictions.parquet` directly (NOT `diag2_enriched.parquet`'s stale
`p_elo`), `tune_scratch/diag2_enriched.parquet` for `p_mkt`/`p_winner_hier_mc`
only, and `prepared_matches_cache.pkl` for the `score` column. No new persistent
files written to `dropshotdata/`.

## Artifacts

- `dropshotdata/scripts/elo_raw_ratings_for_xgb.py`,
  `dropshotdata/scripts/build_xgb_features.py`,
  `dropshotdata/scripts/train_xgb.py`,
  `dropshotdata/scripts/regen_hier_v25_postfix.py` — the full pipeline, re-runnable.
- `dropshotdata/tune_scratch/xgb_v1_model.json` — the fitted model (corrected).
- `dropshotdata/tune_scratch/xgb_v1_feature_importance.csv` — full importance table
  (corrected).
- `dropshotdata/tune_scratch/xgb_features_tune.parquet` /
  `xgb_features_holdout.parquet` /`xgb_v1_holdout_predictions.parquet` — gitignored
  (parquet), regenerate via the scripts above if needed.
