# RESUME HERE — goal: hierarchical model must beat Elo's log-loss on the 2024-25 holdout

**Pete's instruction (still active): keep working autonomously, don't check back in until this is achieved, or you hit a genuine blocker.**

## UPDATE 2026-08-28 (XGBoost on Elo + hierarchical raw state + engineered features — BEATS ELO, per Pete's own direction — first candidate all session to clear the bar)

**Ten rounds of tuning the hierarchical filter directly never beat Elo. Stepping
outside that architecture, per Pete's instruction ("elo + rating + features + some
other stuff into xgb.cv might be the best play"), does.** Full detail, all numbers,
feature list, and the leakage bug found and fixed along the way:
`issues/27-xgboost-feature-model.md`.

1. **A real leakage bug was found and fixed first** (not in `hierarchical_model.py`
   or `elo_baseline.py` — both untouched this task). `tune_scratch/diag_layoff.parquet`
   (a read-only diagnostic from an earlier round) was STALE - built from the
   `prepared_matches_cache.pkl` that predated round 9's causal-ordering fix, so its
   `weeks_since_last_w/l` features had the exact same qual/round-ordering defect
   round 9 fixed elsewhere. A first `xgb.cv` run scored an implausible 0.193
   tune-window log-loss; per-feature isolation traced it to this one stale
   artifact (`weeks_since_last_diff` alone gave log-loss 0.323, more predictive
   than Elo itself - not plausible). Fixed by re-running the pre-existing
   `diag_layoff_features.py` diagnostic against the current (post-fix) cache - a
   read-only rerun, no edit to either forbidden file. Confirmed fixed via a traced
   example and a round-by-round symmetry check (winner/loser `weeks_since_last==0`
   rates went from a wildly asymmetric 97%/8% at R32 to near-identical at QF/SF/F,
   as bracket structure predicts). **Lesson for next time**: any `tune_scratch/
   diag*.parquet` derived from `prepared_matches_cache.pkl` needs its mtime
   checked against the cache's after any `hierarchical_model.py` ordering change -
   the same trap round 9's own section of this file already flagged for the cache
   itself, now confirmed to propagate to its downstream diagnostic consumers too.
2. **Feature set**: Elo overall + surface rating diffs (raw, not blended - a new
   read-only copy of `elo_baseline.py`'s logic, `elo_raw_ratings_for_xgb.py`,
   verified to reproduce the frozen `elo_eval_predictions.parquet` to float
   precision), the hierarchical model's raw posterior state (`d1_mean`/`d1_var`/
   `d2_mean`/`d2_var` from `hier_v25_eval_predictions.parquet`/`v25_tune_preds
   .parquet` - NOT its calibrated probability), as-of head-to-head (a fresh
   chronological pass, strictly-prior meetings only), experience/layoff, and
   round/surface/tier/best_of as categorical context. Symmetric side-A/B
   construction (outcome-blind hash of date+uid-pair) - label balance verified
   ~49-50% on both windows. Market odds explicitly excluded as a feature (would be
   circular against the project's actual goal).
3. **`xgb.cv` (5-fold, early-stopping only, no hyperparameter search) on 2022-23**
   selected `num_boost_round=36` at sensible defaults (`max_depth=5, eta=0.08,
   subsample=0.8, colsample_bytree=0.8`). One final fit, scored ONCE on the
   market-odds-joined 2024-25 holdout (n=8,593, same population as every other
   candidate this session):

   | | XGBoost | Elo (0.6120 basis) | Hierarchical (v25) |
   |---|---|---|---|
   | ALL log-loss | **0.6025** | 0.6121 | 0.6267 |
   | ATP log-loss | **0.6023** | 0.6174 | 0.6213 |
   | WTA log-loss | **0.6027** | 0.6067 | 0.6322 |

   **Beats Elo by −0.0096 overall, in BOTH tours independently** (ATP −0.0151, WTA
   −0.0040) - the first candidate all session to clear the bar in both splits, not
   just aggregate. Sanity-checked per `model-validation`: cv (0.599) and holdout
   (0.6025) log-loss are close (no overfitting gap), predictions range 0.05-0.93
   (no degenerate near-0/1 overconfidence), and calibration by decile tracks
   closely across the whole range.
4. **Feature importance (gain)**: `elo_overall_diff` dominates (total_gain 8194,
   ~2.4x the runner-up), `hier_dmean_diff` a clear second (3487) - Elo and the
   hierarchical model's raw state are both genuinely, non-redundantly useful to
   XGBoost. Surface Elo, tier, layoff, and H2H are real but much smaller
   contributors; `best_of`/`is_elite` are nearly unused.
5. **This changes the project's actual best model** - XGBoost (this section), not
   the hierarchical filter, is now the best-performing candidate on the frozen
   2024-25 holdout. Recommended next step per issue 27: since Elo's own raw rating
   is still the dominant single input even inside XGBoost, further improving Elo
   itself (tier/round-weighted K, margin-of-victory - both open in issue 25) is
   likely higher-leverage than continuing to tune the hierarchical filter
   architecture, which XGBoost is already extracting most of the useful signal
   from on top of Elo.
6. **VALIDATED WITH A CORRECTION (independent post-hoc audit, same day).** A
   ~0.01 win on a first attempt, when nothing else this session moved the metric
   by more than ~0.003, is exactly `stats-discipline`'s "suspicious agreement"
   pattern - it was audited adversarially rather than taken at face value. Found a
   second real leakage bug, same class as the already-caught `diag_layoff.parquet`
   staleness: the `d1_mean`/`d1_var`/`d2_mean`/`d2_var` feature source
   (`hier_v25_eval_predictions.parquet`/`v25_tune_preds.parquet`) was ALSO written
   hours before round 9's causal-ordering fix, and since `hierarchical_model.py`'s
   `run()` updates posterior state sequentially, that's a genuine look-ahead leak
   into `hier_dmean_diff` (the #2 feature by gain), not just a staleness footnote.
   Regenerated those files against the fixed code+cache
   (`scripts/regen_hier_v25_postfix.py`) and rebuilt the pipeline:
   **corrected ALL log-loss is 0.6041, not 0.6025** (ATP 0.6063, WTA 0.6019) - the
   leak inflated the headline by ~0.0016 (~17% of the claimed edge over Elo).
   **The core finding survives**: XGBoost still beats Elo overall and in both
   tours independently post-correction (ATP −0.0111, WTA −0.0048). H2H hand-
   verified correct (80-meeting rivalry, exact match to manual tally); side
   assignment is provably outcome-blind by construction (hashes only date+uid
   pair, never touches outcome/rating); outlier predictions and feature-importance
   stability (top-1%-outlier-removed refit) both check out clean. Market-
   conditioned calibration check (bucketing by the MARKET's own confidence, same
   method as issue 20's `diag2_fresh_cuts.py`): XGBoost IS underconfident in the
   market's high-confidence tail (error +0.048 at 80-100% market confidence) - a
   real, smaller, inherited version of hier's known MC-averaging undershoot
   (hier's own error there is +0.099/+0.115), and worse than Elo's own tail
   calibration (+0.027/+0.036) - but this does NOT cost XGBoost the tail on
   log-loss: it still beats or ties Elo in every market-confidence bucket
   including the top one, post-correction (a separate, PRE-fix diagnostic pass in
   issue 27 had found an inversion here - XGBoost losing to Elo in the top decile
   - that does not hold on the corrected predictions). **Both the official
   pipeline artifacts and `issues/27-xgboost-feature-model.md` have been updated
   with the corrected numbers** - the 0.6041 figure is what's on disk now, not
   0.6025. Full detail: `issues/27-xgboost-feature-model.md`'s "## Validation"
   section.

## UPDATE 2026-08-28 (UTS-style K-factor tested as a bundled alternative to our Sackmann-style K — REJECTED, small but real net loss — Elo's frozen number stays 0.6120, gap to hierarchical stays ≈0.0155

**This section is about `elo_baseline.py` again, a follow-up to the section directly below** (ticket 25's ordering fix + decay=0.3, which set Elo's current-best number at 0.6120). That section explicitly scoped OUT tournament-tier/round-weighted K as a separate, larger change — this is that change, tested properly. Full detail: `issues/26-uts-style-k-factor-bundle-vs-sackmann-k.md`.

1. **Built the whole thing as ONE bundled alternative K-philosophy, not a single-factor tweak**: `uts_context_k()`/`uts_kfunction()` (new in `elo_baseline.py`) port `EloCalculator.java`'s (tennis-crystal-ball, commit `fcda1e6` — fetched and read directly, not just Pete's screenshot summary) full three-part stack — `32 * level_mult * round_mult * bestof_mult * wo_mult`, times a PER-PLAYER rating-dependent `kFunction(rating) = 1 + 18/(1+2^((rating-1500)/63))`, capped at the reference's own `DELTA_RATING_CAP=200`. `run_elo()` gained `k_mode='uts'`, which REPLACES `k_factor(n)`'s matches-played decay entirely (a different K-philosophy, not a combination) — `decay` (layoff/inactivity pull) kept as a separate, orthogonal axis in both modes, per the task's own framing.
2. **Level-code mapping is a real, documented judgment call**: this archive's `tourney_level` vocabulary (`G,F,M,D,O,A,W,P,PM,I,T1-T5,C`, numeric ITF prize-money tiers, `CC,E,J`, two `+H` codes — checked via `value_counts()`, not assumed) isn't tennis-crystal-ball's own normalized scheme. Codes with a clear real-world correspondence were mapped to the reference's buckets (`PM→Masters-equivalent 0.85`, `P→ATP-500-equivalent 0.75`, `W/I/T1-T5→"other tour" 0.70`); everything else (Challenger, ITF, exhibitions, juniors, `+H`) falls to the reference's OWN `default: 0.65` fallback. **Round-switch quirk kept faithfully**: the reference has no default case in its round switch, so every qualifying round (Q1-Q5, ER) gets the SAME full weight as a Final — confirmed by reading the Java source directly, genuinely counter-intuitive, not "fixed."
3. **Four sanity checks passed before trusting any comparison number**: (a) `k_mode='sackmann'` (default) post-refactor reproduces the committed 0.6120 exactly — the refactor didn't silently change the existing path; (b) `uts_kfunction()` matches the reference's own documented checkpoints exactly (10.000/5.495/1.640/1.073/1.008 at ratings 1500/1600/1800/2000/2200); (c) rating-range sanity check on REAL archive ratings (not just the formula in isolation) — 44,585 rated players range 950-2734 around START=1500, scale roughly sane, BUT **90.7% still sit in kFunction's "provisional" zone (kFunction>3.0)** — this archive is 89% lower-tier/ITF churn, so most rated players never accumulate enough matches to reach the "stabilized" regime kFunction was designed to protect; (d) anchors (Sinner/Alcaraz, Sabalenka/Swiatek top-5, no thin player top-20) pass on every UTS arm. Committed (`45336a8`) after these four checks, before trusting any log-loss number, per the task's git discipline.
4. **Headline bundle-vs-bundle result, same 2024-25 market-odds-joined holdout (n=8,598)**: UTS K + decay=0.3 (the fair apples-to-apples arm) logloss **0.6127** vs current-best **0.6120** — **UTS is worse by +0.0007**, small but real and anchor-clean, not a wash. UTS K + decay=0.0 is worse still (0.6139) — confirms our existing layoff decay and UTS's kFunction are complementary, not redundant.
5. **A genuine, opposite-signed tour split hides inside that net number**: UTS's K *improves* ATP (0.6156 vs 0.6173, Δ−0.0017) but *degrades* WTA (0.6096 vs 0.6065, Δ+0.0031) — WTA's degradation outweighs ATP's gain in the combined number. Mirrors a pattern the hierarchical-model tuning effort (below) has hit repeatedly this session; not chased further here, flagged as a possible shared root cause.
6. **Secondary diagnostic (context vs. kFunction ablation, per the task's "if time permits" note)**: the tournament-context multiplier alone (level/round/best-of, kFunction fixed=1.0) explains almost all of the ATP improvement (0.6159, close to the full bundle's 0.6156) — kFunction alone is actually a net NEGATIVE for ATP on its own (0.6186, worse than Sackmann's 0.6173). Reading this against check (c) above: kFunction's design premise (fast-then-stable K) assumes a population that mostly reaches the stable regime; this archive mostly doesn't, so it spends most of its time amplifying noise from thin-history players rather than doing its intended job. The context piece is the more genuinely useful half of UTS's bundle, but it's not enough alone to flip the combined verdict once WTA's degradation is counted.
7. **Overfitting-risk note, lower than usual**: no grid search or tuning happened this round — the reference's constants were ported as-is, untouched. This is a single, un-tuned bundle comparison, a materially lower-risk claim than a selected-from-a-grid one (unlike ticket 25's `decay` grid, which does carry that caveat).
8. **Elo's frozen number is UNCHANGED — still 0.6120.** UTS's K-philosophy was tested and REJECTED as an improvement; `elo_eval_predictions.parquet` (the canonical comparison artifact) is untouched. **Gap to hierarchical stays ≈0.0155** (0.6275 − 0.6120), same as the section below leaves it. Anyone picking this up next: the level-code mapping for the WTA-specific tiers (W/P/PM/I/T1-T5) is a real judgment call, not code-verified against a WTA-specific reference — if the WTA degradation above is worth chasing, start there.

## UPDATE 2026-08-28 (Elo baseline work, ran independently of the round 1-10 hierarchical-model tuning above): the ELO SIDE of the comparison moved — "the number to beat" is now 0.6120, not 0.6152, so the gap widens to ~0.0155

**This section is about `dropshotdata/scripts/elo_baseline.py` itself, NOT the hierarchical model.** Elo had never been touched all session (rounds 1-10 above only ever changed `hierarchical_model.py`) — a prior-art review (`EloCalculator.java`, tennis-crystal-ball) found it was missing standard techniques, and separately, `hierarchical_model.py`'s round-9 causal-ordering fix turned out to apply identically to `elo_baseline.py` (near-duplicate `load_matches()`/`prepare()`). Every "gap to Elo" number in rounds 1-10 above (0.0114, 0.0123, etc.) was computed against the OLD, buggy, un-improved Elo — **that comparison point has now moved, independent of anything the hierarchical model does.**

1. **Task 1 — same causal-ordering bug, confirmed present here too, fixed.** Ported `hierarchical_model.py`'s `ROUND_RANK`/`round_rank()` fix (commit 111ecce) into `elo_baseline.py`'s `prepare()`. Verified the fix changes something real, not just plausible-sounding: a new `count_round_inversions()` diagnostic counts qual-after-main-draw inversions within a shared date — **75,367,441 before the fix, 0 after** (larger in absolute count than the hierarchical-model file's own check because this is summed over ALL same-date pairs, not distinct player-date groups — same underlying bug, same root cause, same fix). Anchor checks (Sinner/Alcaraz, Sabalenka/Swiatek top-5 active, no thin player in top 20) still pass. Alone, before any decay tuning, this fix moves Elo's log-loss on the exact 2024-25 market-odds-joined holdout from **0.6152 → 0.6134** — a real, non-tuned correctness improvement. Committed separately (`dbfeb42`).
2. **Task 2 — inactivity/layoff decay added, a genuine improvement.** Ported the shape of `EloCalculator.adjustRating()` (tennis-crystal-ball, commit `fcda1e6`): a returning player's overall AND surface ratings shrink toward `START` by an amount that grows with days since their last match, via a logistic curve, saturating at a bounded fraction (`decay`, the one new tunable parameter) of the distance to `START` rather than a full reset. `decay=0.0` is an exact structural no-op, verified bit-for-bit against the pre-existing blend-grid numbers before trusting anything nonzero. **Deliberately scoped to inactivity decay only** — no tournament-tier/round-weighted K, no margin-of-victory signal (both explicitly out of scope for this pass, per the task brief and this file's own one-axis discipline).
3. **A real population-mismatch caught mid-task, not assumed away.** The file's own internal "ALL" eval metric pools 89% lower-tier (qual/challenger/ITF) rows with the 11% tour-level rows the hierarchical-model comparison actually uses — and the decay grid's effect on "ALL" was flat/noisy (1/7 nonzero values beat baseline), which would have looked like a negative result. Restricting to `level_file=='tour'` (the file's own pre-existing comment already flags this split matters) revealed a real, **non-spiky, interior optimum**: 7/7 nonzero decay values beat `decay=0.0`, both ATP and WTA independently agree the optimum sits around decay≈0.2-0.3, and the effect doesn't hug a boundary (mirrors the "genuinely interior, not monotonic-to-boundary" check round 6 of the hierarchical-model work used to distinguish real effects from noise). `elo_baseline.py` was fixed to select on the tour-level metric, not the noisier pooled one, before trusting the result. **`decay=0.3` selected**, anchors re-verified and still pass at that arm.
4. **Verified on the exact market-odds-joined 2024-25 holdout** (`tune_layoff_decay.py`, n=8,598 vs the original 8,594 — trivial difference, not investigated further): logloss 0.6134 (decay=0, post ordering-fix) → **0.6120** (decay=0.3). ATP 0.6182→0.6173, WTA 0.6085→0.6065 — both tours improve. Combined with Task 1: **Elo's own log-loss on this holdout moves from 0.6152 (original, buggy) to 0.6120 (fixed + decay-improved) — a genuine −0.0032 improvement to the baseline itself.**
5. **Honest overfitting-risk caveat**: this script evaluates directly on 2024-25 with no separate tuning-window split of its own (an existing, pre-registered property of this baseline file, not invented for this task) — so `decay=0.3`'s selection as literally the single best grid point does carry some risk of fitting this exact window's noise, more than a proper tune/holdout split would allow. Mitigated, not eliminated: the grid was kept small (8 points), and the deciding evidence was the NEIGHBOURHOOD-robustness check (7/7 nonzero points improving, both tours agreeing on roughly the same optimum) rather than the single lowest point in isolation — closer in spirit to "verify it's not a boundary/noise artifact" than a from-scratch overfitting defense. Flagging this plainly rather than presenting 0.6120 as beyond-doubt.
6. **This changes the honest "gap to Elo" number every round above was computed against.** Hierarchical's current best (v25 + round 9's kept ordering fix, per round 10 above) is logloss **0.6275**. Old gap: 0.6275 − 0.6152 = **0.0123** (round 10's number). New, honest gap using the fixed+improved Elo: 0.6275 − 0.6120 = **≈0.0155**. **The bar the hierarchical model needs to clear moved further away, not closer** — fixing and improving Elo was the right thing to do regardless (it's the project's frozen comparison point and was carrying two real, fixable defects), but it makes the hierarchical-model tuning effort's job strictly harder, not easier. Anyone continuing rounds 1-10's work should use **0.6120**, not 0.6152, as Elo's number from here on.
7. Full detail: `issues/25-elo-baseline-round-order-and-layoff-decay.md`. Code: `dropshotdata/scripts/elo_baseline.py` (both fixes), `dropshotdata/scripts/eval_elo_vs_market.py` (reusable market-join harness, factored out of this RESUME file's own step 4), `dropshotdata/scripts/tune_layoff_decay.py` (the tour-level decay grid script). Commits: `dbfeb42` (Task 1), `48077c8` (Task 2).

## UPDATE 2026-08-28 (round 10): analytic delta-method correction (round 3's own recommended next step) built, validated, and FAILS its faithfulness check — not revealed on holdout — last identified direction now exhausted, no fallback scoped

**Built exactly what round 3/9 flagged as the one remaining unattempted idea, and it doesn't work — cleanly, informatively, not ambiguously.** `analytic_correction()` (new function in `dropshotdata/scripts/mc_parallel.py`, parallel to `mc_average()` — that function is untouched) replaces MC sampling of `E[p_match(sigmoid(d1),sigmoid(d2))]` with a second-order (delta-method) Taylor expansion around the point estimate: `f(m1,m2) + 0.5*f_11*v1 + 0.5*f_22*v2`, `f_11`/`f_22` from central finite differences at `h=0.01` (5 `p_match` calls/row vs 60-2000 for MC).

1. **Finite-difference step size (`h=0.01`) is numerically solid** — verified via convergence across `h∈[0.001,0.1]` and agreement between `h=0.01`/`h=0.005` to 1.5e-4 mean abs diff on a 66-point sample spanning the real archive's `d1_mean`/`d2_mean`/`best_of` range (`tune_scratch/analytic_h_stability.py`). `p_match`'s composed game→set→match DP is smooth in its inputs (fine-grid scan shows no discontinuities); this is not where the method fails.
2. **The approximation itself FAILS validation against high-precision MC (n=2000), decisively, not marginally** (`tune_scratch/analytic_validate.py`, 500 rows stratified by combined posterior variance from `hier_roundorder_tune_predictions.parquet`): overall mean abs diff 0.183 (vs a 0.03-0.04 pure-MC-sampling-noise floor at n=60), max abs diff 0.668, correlation with the reference only 0.353. **Even the LOWEST-variance quartile of REAL rows (mean combined var 0.27) already shows 2.5x the sampling-noise floor's disagreement** — this isn't a tail-only failure, it's present across the model's entire real variance range. 3.2% of the tuning-window join hit the `[0,1]` safety clip, several on the WRONG side (one row: point estimate 0.92, true MC 0.67, analytic clipped to **0.00**). A synthetic sanity check (varying variance from 0.001 to 0.8 at a fixed mean) confirmed this is the textbook 2nd-order-Taylor breakdown, not an implementation bug — and confirmed the model's real posterior variance floor (~0.05-0.06/side even for the most-experienced players) is already inside the breakdown zone. **There is essentially no practically-occurring row in this archive where a pure 2-term Taylor correction is faithful.**
3. **Scored anyway on the tuning-window metric for documentation** (not as a real candidate, per charter — it failed its own pre-registered validation gate): weighted log-loss 1.0887 vs `mc_average(n=60)`'s 0.5567 — more than double, a direct consequence of the clipping failures. **Not revealed on holdout** — no candidate that fails its own faithfulness check earns the one-shot reveal.
4. **Speed win confirmed real but moot**: 2.0x faster than production MC(n=60), 30.4x faster than high-precision MC(n=2000) — exactly as expected from 5 calls vs 60-2000 — but a fast wrong answer isn't useful on its own.
5. **`stress_test_safety.py` unaffected** (never touches `hierarchical_model.py`), confirmed — ALL PASS, same 55 checks as round 9 left it.
6. **This is a third, distinct outcome from the two the task brief anticipated** (not "close to MC" = pure speed win; not "MC estimation error was the real problem" = a genuine fix) — it's "this specific 2-term analytic shortcut isn't viable at this architecture's actual posterior-variance scale." It does NOT refute round 3's Jensen's-gap diagnosis; MC-averaging (which implicitly captures whatever higher-order terms matter, by sampling rather than truncating) remains the best available estimator of the target expectation. A higher-order or structurally different analytic approximation (4th-order Taylor, saddlepoint/Laplace, or exploiting `p_match`'s DP structure directly instead of finite-differencing it as a black box) is a genuinely different, unattempted idea — out of scope for this round, unscoped for a future one.

**Gap to Elo: still 0.0123 (round 9's config, unchanged — no candidate reached holdout this round).** Full detail, all numbers, and the synthetic sanity-check table in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 10" section. **Ten rounds now, every lever at every layer this session identified has been tried and exhausted — hyperparameters, architecture (tickets 21/24), data-prep causal ordering (round 9), and now the analytic-correction alternative to MC-averaging itself (round 3's own last-standing recommendation). No fallback direction is currently identified or scoped.** Recommend for whoever picks this up next: either scope a genuinely different analytic approximation (higher-order, or DP-structure-aware rather than black-box finite-difference — a real, non-trivial undertaking with its own validation risk), or accept 0.0123 (v25 + round-order fix) as the closing number for same-session tuning of this specific architecture.

## UPDATE 2026-08-28 (round 9): real causal-ordering bug fixed (qualifying rounds sorted after the whole main draw) — genuine tuning-window improvement, REJECTED on holdout anyway — gap widens slightly to 0.0123 — data-prep-layer bugs now also ruled out as the mechanism

**This was a real, confirmed, structural bug in `prepare()`, not a tuning knob** — found and fixed independently of rounds 1-8's hyperparameter/architecture work (which all assumed the input ordering was correct). Full mechanism, stress test, and numbers below; issue 20 updated with the same detail.

1. **Bug, confirmed systemic (not a one-off).** The archive's `tourney_date` is a TOURNAMENT-level (start) date, not per-match: 617,380 player-date groups (26% of 2,378,344 player-appearances) share a date with another match by the same player, averaging ~2.9 matches/group, up to 23. `prepare()` sorted only by `date` with a stable mergesort, so within-date order fell back to raw file-concatenation order. Root cause traced directly in `load_matches()`: for each tour it reads ALL main-draw yearly files first, THEN ALL `qual_chall` yearly files, THEN ALL `qual_itf` yearly files (`pats` loop) — so for **every single tournament, every year, both tours**, qualifying rounds land after the entire main draw in file order, even though qualifying happens first in real life. This is a property of the loading code itself, not a per-tournament quirk — confirmed via the code path, not just a spot check. Concretely: a player who had just won Q1-Q3 got evaluated in their R128 match using a state that hadn't yet absorbed those wins — exactly the population (low-experience-looking opponents in early rounds) earlier diagnostics flagged as the worst-calibrated round bucket (R128, error +0.1493). Full round vocabulary confirmed present: `Q1,Q2,Q3,Q4,Q5,ER,R128,R64,R32,R16,QF,RR,SF,F,BR` plus a single stray lowercase `f` (one WTA ITF-qualifying final on 2022-05-23, confirmed same round as `F` by cross-checking that date's other matches, not a distinct label).
2. **Fix**: new `ROUND_RANK`/`round_rank()` in `hierarchical_model.py`, sorting `prepare()`'s output by `(date, round_rank)` instead of `date` alone (mergesort kept, so genuinely-simultaneous same-date-same-round matches keep file order). `'f'` normalized to `'F'`. Any future unrecognized round label falls back to NaN (sorted last within its date) with a logged warning rather than crashing or guessing.
3. **New stress test** (reproduction 20 in `stress_test_safety.py`, now 29 checks total): a synthetic Q1/Q2/Q3/R128 same-date scenario fed in the archive's real "bad" file order (R128 first). Asserts `prepare()` reorders qualifying before R128, AND that replayed player state (n_matches, variance) reflects the 3 qualifying wins before the R128 match is reached — confirmed failing under the old date-only sort (explicit negative-control check) and passing after the fix. Also covers `round_rank()`'s unknown-label fallback and `f`/`F` normalization. **ALL PASS.**
4. **A second real bug found and fixed while verifying this one**: `load_prepared()`'s cache-freshness check only compared the cache's mtime against archive CSV mtimes — an edit to `prepare()`'s own logic (like this fix) does NOT touch any CSV, so the check would have silently kept serving the STALE pre-fix cache forever. Added a check against `hierarchical_model.py`'s own mtime too, and manually deleted the stale `tune_scratch/prepared_matches_cache.pkl` this round. **Anyone editing `prepare()`/`load_matches()` in future rounds should not assume the cache auto-invalidates on a code-only change — verify, don't assume, exactly as this task's brief flagged.**
5. **Tuned on 2022-23 only**, through the same weighted/lopsided metric v25 was selected on (`tune_v3_lopsided.py`'s exact join+scoring, market-only — Elo isn't part of the tuning-window charter metric). Corruption 0.00000%. **Genuine, non-trivial improvement**: weighted logloss 0.5759 → **0.5561** (Δ−0.0198), far larger than any of this session's noise-level "wins" (typically ±0.0001-0.0006). This looked like the first candidate all session with a real, structurally-motivated tuning-window win large enough to justify the one holdout reveal.
6. **Holdout reveal (2024-25, n=8,594 odds-joined rows, same exact join as every prior round): REJECTED anyway.** Gap to Elo widened slightly: overall **+0.0114 → +0.0123** (Hier 0.6266→0.6275, Elo unchanged 0.6152). ATP +0.0074, WTA +0.0173 (WTA still the larger share, consistent with every prior round's finding). Lopsided log-loss also got worse (0.4288→0.4099 is actually an improvement, but Elo's lopsided log-loss is 0.3853, so hier is still worse than Elo there too — full table in issue 20). **This is now the fifth confirmed case this session of a real, non-degenerate, correctly-gated tuning-window win failing to transfer to the holdout** (after `TIER_WEIGHT`, `var_scale`, `RW_VAR_PER_WEEK`-toward-zero, and the round-6 conditional `var_scale`) — but the *first* one that isn't a hyperparameter/calibration lever at all, it's a genuine data-correctness fix. That it still didn't transfer is informative: **this rules out data-prep-layer ordering bugs as a candidate explanation for the gap**, narrowing the remaining honest hypothesis space back to round 3's MC-averaging-convexity diagnosis (still the best-diagnosed root cause, still with no untried fix — see round 8's bottom line, unchanged by this round).
7. **The fix is kept regardless of the rejection** — this is a genuine correctness bug (wrong causal ordering in a temporal model), not a tuning knob, and is worth fixing even though its effect on THIS particular comparison metric was net-negative on holdout. Reverting it would mean deliberately shipping a known-wrong data-prep step. `v25`'s hyperparameters are unchanged; this round changed no hyperparameter, only the sort correctness of the input the fit consumes.

**Gap to Elo: now 0.0123 (was 0.0114 before this round's fix — the fix is real and correct, but nets out slightly negative on this specific holdout comparison).** Nine rounds, ~38 attempts now tried and rejected/non-transferring/flat, plus one genuine correctness bug found, fixed, and kept despite a net-negative transfer. **Recommended next step for whoever picks this up:** round 3's MC-averaging convexity mechanism is now the ONLY diagnosed-but-unfixed root cause left standing after nine rounds of elimination across hyperparameters, architecture (tickets 21/24), calibration, evidence-reweighting, AND now data-correctness — a genuinely new idea attacking Jensen's-gap cost directly (e.g. an analytically-corrected expectation of `p_match` over the posterior instead of Monte Carlo averaging a convex function) is the only unexplored direction with real diagnostic support behind it.

## UPDATE 2026-08-28 (round 8): ticket 24's heavy-tailed process noise BUILT, stress-tested, tuned — FLAT/no signal on tuning window, NOT revealed on holdout — gap still 0.0114 — both scoped rewrites (21 and 24) now closed with clean negatives

Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 8" section and `issues/24-robust-filtering-replace-hard-clamps.md` (now closed). This closes out round 7's one remaining recommendation — the last unattempted item from round 5's structural analysis.

1. **Built the real thing, per issue 24's own spec, approach (a).** `adf_update`'s and `adf_update_full_dp`'s Gaussian-prior Newton step now optionally treats the prior as a Student-t(nu) via a floored scale-mixture-of-normals approximation (`_heavy_tail_var`, `hierarchical_model.py`): each Newton iteration recomputes an effective prior variance `v_eff = v_d * max(1, (nu + r²)/(nu+1))` where `r² = (d-m_d)²/v_d` — a genuine IRLS scheme converging toward the Student-t-prior MAP, not a static adjustment. The floor at 1.0 is a deliberate, documented deviation from a literal Student-t (which is slightly tighter than Gaussian right at the mode): it guarantees every ordinary/unsurprising update is untouched by `nu`, isolating the whole mechanism to genuinely large surprises — exactly the MAX_SHIFT failure mode ticket 24 targets — without silently retuning any of the session's other already-tuned hyperparameters. `MAX_SHIFT`/`MAX_STATE`/`MAX_VAR` hard clamps are left fully in place as the ultimate safety net, unconditionally — this is an enhancement to the update math, not a removal of the backstop.
2. **5 new stress tests added** (reproductions 15-19 in `stress_test_safety.py`, 24 checks total now): a single large surprising update (heavy-tailed genuinely moves the mean further AND retains more posterior variance than Gaussian on the identical input — both directions verified, not assumed), a sequence of 8 alternating large surprises (bug 9's exact compounding pattern — variance never collapses toward zero), the round-5 long-layoff variance-sufficiency checks re-run under the flag (unaffected, as expected — this ticket only touches the observation update, not `inflate()`'s time-decay process noise), and `_heavy_tail_var`'s floor/monotonicity/nu-limit contract. ALL PASS. Gated behind `DSV_HEAVY_TAILED_NOISE` (default `"0"`), verified as an EXACT no-op on the default path via BOTH a synthetic reference-implementation diff (max diff 0.0) AND a real 2022-23 fit diff (implicit default vs explicit `=0`, 10,313 eval rows, max abs diff across all numeric columns = 0.0, `.equals()` True) — not just code inspection.
3. **Tuned on 2022-23 only**, through the same weighted/lopsided metric v25 was selected on: baseline (`HEAVY_TAILED_NOISE=0`) weighted=0.5759; turning it ON at the default `nu=4.0` gives weighted=0.5759 (delta **+0.0000**, an exact tie). A full `nu` scan (2.0 through 30.0, more heavy-tailed to nearly-Gaussian) stays completely flat: weighted log-loss 0.5759-0.5760 across the whole range, plain log-loss identical to 4 decimals (0.6347) at *every single value*, corruption 0.0% throughout (the gate never even engages). This is a flatter result than ticket 21's (which at least showed a real 0.0001 wobble) — the mechanism has essentially nothing to act on in this archive.
4. **Confirms round 5's diagnostic directly rather than just inferring from it.** Round 5 already found zero of the 10,678 holdout rows anywhere near the `MAX_STATE`/`MAX_SHIFT`-engaging wall (`diag2_fresh_cuts.py`) — this round shows the direct consequence: a mechanism that only changes behaviour when `MAX_SHIFT` would otherwise clamp an update has nothing to act on, so making that mechanism heavier-tailed (in either direction, more or less so) moves the tuning-window fit by exactly zero. This is now confirmed empirically for BOTH of round 5's flagged-as-not-indicated candidates (ticket 21's richer regime-B likelihood, ticket 24's heavy-tailed noise) via two independent implementations, not just the one shared premise-check.
5. **No candidate cleared the bar to spend the one holdout reveal** — nothing beat baseline by more than tuning-window noise (best case was an exact tie, not even a positive wobble). **Not revealed on holdout, correctly**, same charter call as ticket 21 and the layoff-bump lever.
6. **Ticket 24 is now closed** (see the issue file) — built properly (real update math, not a parameter swap), stress-tested for the exact failure modes it targets, tuned per charter, and found not to move the tuning-window metric. **Both of round 5's flagged "not yet attempted" architecture rewrites (ticket 21, round 7; ticket 24, this round) are now complete, each with a clean, well-evidenced negative result**, closing out the entire 21-24 frontier that motivated rounds 3-7's structural analysis.

**Gap to Elo: still 0.0114 (v25 unchanged, best config, `DSV_HEAVY_TAILED_NOISE` left at its no-op default).** Eight rounds now, ~37 attempts tried and rejected or non-transferring or flat. **Honest bottom line for whoever picks this up next**: the MC-averaging convexity mechanism identified in round 3 (third continuation) remains the best-diagnosed root cause (the mean state already extremizes correctly; the defect is downstream, in Jensen's-gap cost from averaging a convex match-win function over the posterior) and every well-motivated fix at every layer — evidence richness (ticket 21), process-noise shape (ticket 24), MC-variance-scaling (4 variants, round 3/6), calibration (Platt, isotonic), evidence reweighting (tier, WTA-lower, layoff-bump, RW_VAR_PER_WEEK toward zero) — has now been tried and rejected or found flat. This reads as a genuine architecture ceiling for same-session tuning of this specific Gaussian-filter-plus-MC-averaging design. A fix would need to attack the Jensen's-gap cost directly and structurally (e.g. a closed-form or analytically-corrected expectation of `p_match` over the posterior, instead of Monte Carlo averaging a convex function) — that is a genuinely new architecture direction, not a variant of anything tried in rounds 3-8, and hasn't been scoped or attempted.

## UPDATE 2026-08-28 (round 7): ticket 21's full-DP scoreline likelihood BUILT, stress-tested, tuned — FLAT/no signal on tuning window, NOT revealed on holdout — gap still 0.0114

Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 7" section and `issues/21-build-scoreline-inversion-regime-b.md` (now closed). This is the "concrete implementation idea" ticket 21/the RESUME file itself had flagged since round 3 as the one big not-yet-tried architecture piece — built properly this round, per Pete's explicit request to build it even though round 5's diagnostics already suggested it might not move the CURRENT gap (the defect being downstream, in MC-averaging's convexity cost, not in insufficient evidence extraction).

1. **Built the real thing, not another approximation.** `p_set`'s game-by-game DP was refactored (exact, verified bit-identical - stress test + a real fit-output diff, max delta 4.4e-16 float noise) into `p_set_full()`, which returns the complete terminal set-score distribution instead of collapsing to P(A wins). A new `adf_update_full_dp()` fits the combined-strength gap `d` against the log-likelihood of the ACTUAL observed set-by-set scoreline (e.g. exactly 6-4 3-6 7-6), via a numerical (finite-difference) Newton step generalizing `adf_update`'s closed-form Bernoulli update to a likelihood with no closed form. Symmetric-strength convention per ticket 12: `p_a_serve=sigmoid(d/2)`, `p_b_serve=sigmoid(-d/2)` (NOT equal serve rates for both sides - that degenerate reading collapses every prediction to ~50/50 regardless of `d`; verified via a monotonicity stress test).
2. **9 new stress tests added** (reproductions 6-14 in `stress_test_safety.py`): refactor bit-identity, distribution normalization, extreme scorelines (6-0 6-0) at multiple priors, the exact ticket-12 single-extreme-set grid-boundary case, far-from-zero-`d` stability, the symmetric-`d` degeneracy/monotonicity check, no-op guards, `parse_score_sets` edge cases (rejects old-rules undecided long deciders like a 70-68 fifth set), and grid-cache boundedness. ALL PASS. Gated behind `DSV_REGIME_B_FULL_DP` (default `"0"`), verified as an EXACT no-op on the default path via a real fit-output diff against pre-change code (not just code inspection) - max delta 4.4e-16.
3. **Performance was a real risk, checked and fine**: a naive per-call cost is ~50-240us depending on cache warmth; a fixed 0.01-wide grid over `d∈[-12.5,12.5]` (not an unbounded LRU) bounds memory and gives worst-case ~90s added to a full-archive fit (measured: 39s→~90-96s with FULL_DP=1 on).
4. **Tuned on 2022-23 only**, through the same weighted/lopsided metric v25 was selected on: baseline (FULL_DP=0) weighted=0.5759; FULL_DP=1 at v25's own `REGIME_B_N=0.25` gives weighted=0.5758 - a 0.0001 "win" that's pure noise, not a finding. A `REGIME_B_N` re-scan under FULL_DP=1 (0.1 through 4.0, all passing the corruption gate at 0.0% - notably N=2.0 and N=4.0 now PASS cleanly, unlike the old approximation which hard-rejected N≥2.0, confirming the richer likelihood genuinely isn't the same maximally-overconfident-fake-trial pathology) stays completely flat: weighted log-loss never moves outside 0.5758-0.5763 across a 40x range of evidence weight. The lopsided-log-loss-improves/plain-log-loss-worsens trade visible as N increases is the SAME already-rejected pattern seen in Platt scaling, isotonic calibration, and the layoff-bump lever earlier this session - recognized, not re-chased.
5. **No candidate cleared the bar to spend the one holdout reveal** - the charter is "reveal only a candidate you actually believe in," and nothing here beat baseline by more than noise. **Not revealed on holdout, correctly, per precedent** (same call made for the rejected layoff-bump lever earlier this session).
6. **A real operational lesson, not a modeling one**: the first attempt at this scan (a single `tune_ticket21_fulldp.py` script running all 8 candidates) stalled silently mid-scan - `subprocess.run()` hung with 0% CPU progress after step 1 completed, with a hidden child process (not visible via `tasklist //FI "IMAGENAME eq python.exe"` alone) still attached. Caught via a CPU-delta check (`Get-Process -Id X | Select CPU`, sampled twice 8s apart - 0 delta means genuinely stalled, not slow), killed with `taskkill //PID X //T //F`, and the remaining candidates were re-run as individual foreground invocations (`tune_step2_single.py`) for tighter visibility, one N value at a time. No orphaned files or corrupted state resulted (`run_candidate()`'s cleanup runs on every exit path). **New operational rule for next time**: when a background fit/scan appears stalled (no new log output for several minutes), don't just check `tasklist` for process existence - check CPU-time delta over a short window before concluding it's alive; a process can be present and "Responding: True" while making zero progress.
7. **Ticket 21 is now closed** (see the issue file) - the concrete implementation idea it specified has been built, stress-tested, and evaluated exactly per its own spec, and found not to move the tuning-window metric. This is a genuine, informative negative result, not an abandoned attempt: it directly confirms round 3's mechanism finding (the defect is in MC-averaging's Jensen's-gap convexity cost downstream of the mean state, not in regime-B's evidence extraction upstream) from a completely different angle - richer evidence changes the fitted state by essentially nothing, because the coarser evidence was already extracting what mattered.

**Gap to Elo: still 0.0114 (v25 unchanged, best config, `DSV_REGIME_B_FULL_DP` left at its no-op default).** Seven rounds now, ~35 attempts tried and rejected or non-transferring, with the two largest not-yet-tried levers from round 5's own recommendation (ticket 21 full-DP, this round; ticket 24 full heavy-tailed rewrite, still untried) narrowing to one. Ticket 21 is closed with a clean negative. **Recommended next step for whoever picks this up**: ticket 24 (`issues/24-robust-filtering-replace-hard-clamps.md`, Student-t/scale-mixture process noise replacing the Gaussian random-walk + hard-clamp architecture) is now the only unattempted item from round 5's structural analysis - but round 5 itself flagged that its premise (absorbing-boundary clamp pathology) is undercut by zero holdout rows anywhere near the `MAX_STATE`/`MAX_VAR` wall, so it is not obviously more promising than ticket 21 turned out to be. The MC-averaging convexity mechanism identified in round 3 (third continuation) remains the best-diagnosed root cause and has no untried fix - a genuinely new idea for addressing Jensen's-gap cost directly (not another evidence-weighting or calibration variant) is the most promising direction, if one can be found.

## UPDATE 2026-08-28 (round 6): conditional var_scale — a genuinely non-boundary tuning-window win — STILL REJECTED on holdout — gap still 0.0114 — MC-variance-scale mechanism family now fully closed out

Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 6" section. Picked up the one open thread flagged at the end of round 5: a quick POC (`tune_scratch/diag4_conditional_var_scale_poc.py`, not charter-compliant) suggested shrinking `d1_var`/`d2_var` only for rows where the model's own point estimate already calls the match lopsided (a `var_scale_threshold`/`var_scale_high_scale` pair, new params on `mc_average()` in `mc_parallel.py`) might avoid the flat scale's (attempt 19) midband-erosion failure while still closing the top-bucket undershoot.

1. **Implemented properly** — `var_scale_threshold=None` is an exact, verified no-op (bit-identical to pre-existing behaviour). `stress_test_safety.py` still ALL PASS.
2. **Tuned on 2022-23 only**, 4×5 grid (threshold 0.80-0.95, scale 0.3-0.7) through the same weighted/lopsided metric v25 was selected on. Best cell (threshold=0.85, scale=0.7): weighted 0.5671→0.5653 and lopsided 0.4238→0.4144, both genuine improvements. **Every grid cell degraded the midband** (best cell: 0.7007→0.7056) — diagnosed why: at threshold=0.85, 24.3% of the market-defined midband already has the model's *own* point estimate calling it >85% confident (the model's known weak-disagreement pathology), so shrinking variance there amplifies an already-wrong confidence.
3. **Confirmed the optimum is genuinely interior, not another monotonic-to-boundary artifact** — extended the scale grid to 0.98 and watched the metric climb smoothly back toward baseline with no further improvement past 0.7. This is the first candidate in the MC-variance-scale family to pass that specific check cleanly (unlike `TIER_WEIGHT`, flat `var_scale`, and `RW_VAR_PER_WEEK`, all of which were monotonic-to-some-boundary).
4. **Revealed once on the real holdout: REJECTED anyway.** Gap to Elo widened in every cut — overall +0.0114→+0.0139, ATP +0.0008→+0.0017, WTA +0.0223→+0.0265 — despite lopsided log-loss genuinely improving (0.4240→0.4148) exactly as the tuning window predicted; midband also degraded on holdout (0.6928→0.6976) exactly as the tuning-window diagnosis predicted. Both effects transferred correctly in *direction* — this was not spurious tuning-window noise — but the net trade was unfavourable regardless. This is the **fourth** confirmed case this session of a real tuning-window win in this mechanism family failing to transfer, and the first one that isn't a boundary artifact, which rules out "it only failed because it hugged a boundary" as an excuse for the other three.
5. **MC-variance-scale-on-`d1_var`/`d2_var` (as a mechanism family) is now closed out.** The only untried variant — gating the conditional shrink on an *external* confidence signal (market/Elo probability) instead of the model's own point estimate — would require the holdout target or a proxy for it as an eval-time input, a fundamentally different and riskier leakage category, not a natural next step. `mc_parallel.py`'s new params are left wired at their no-op defaults.

**Gap to Elo: still 0.0114 (v25 unchanged, best config).** Six rounds, ~29 attempts now tried and rejected or failed to transfer. This session's own conclusion from round 5 stands and is now reinforced by a cleaner negative result: this reads as a genuine architecture ceiling for same-session tuning, not an unexplored lever.

## UPDATE 2026-08-28 (round 5): RW_VAR_PER_WEEK lead closed (REJECTED on holdout) — gap still 0.0114 — honest scoping says ticket 21/24 are NOT indicated, this may be a real architecture ceiling

Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 5" section. This closes out round 4's one remaining flagged lead.

1. **Built the missing safety check first, per the brief's instruction not to trust a gate that never tests the risk it's meant to catch.** `stress_test_safety.py` only ever checked variance/state UPPER bounds — nothing verified a long layoff grows variance ENOUGH. Added two new checks (reproduction 5): variance-sufficiency floors at 1/3/5yr from a tight tour-veteran prior (the 5yr floor tied to `PHI_MAX`'s own documented "should saturate" design intent), and a concrete check that a 3yr-layoff returning player's MC-averaged win probability vs an average opponent stays ≤0.95. **Verified it has teeth**: passes cleanly at v25's default (`RW_VAR_PER_WEEK=0.005`), correctly FAILS at the round-4 scan's suspicious low end (`0.0001`) where 5yr-layoff variance barely moves (0.03→0.134).
2. **Re-scanned `RW_VAR_PER_WEEK` toward zero on 2022-23** through both this new gate and `corruption_rate()`. The new gate cuts in precisely between 0.0033 (fails the 5yr floor by a hair) and 0.0034 (passes) — everything below 0.0033 is now correctly rejected as unsafe, closing the exact blind spot round 4 flagged. **Within the safe region (0.0034 and up), the tuning-window metric is still monotonically improving all the way to the new gate boundary with no plateau** — the same "suspicious monotonic-to-boundary" shape as before, just cut off by a real mechanism check instead of an arbitrary grid edge. Unlike Platt/isotonic/layoff-bump, lopsided log-loss improved *alongside* the weighted metric here (not traded against it), so it looked like a real enough candidate to spend the one holdout reveal on.
3. **Holdout reveal of the best gate-passing value (`RW_VAR_PER_WEEK=0.0034`): REJECTED.** Gap to Elo got WORSE in every cut — overall +0.0114→+0.0123, ATP +0.0008→+0.0013, WTA +0.0223→+0.0237 — despite lopsided log-loss genuinely improving as tuning predicted (0.4288→0.4153). Config reverted; v25 (`RW_VAR_PER_WEEK=0.005`) remains best. This is the **third** confirmed case this session of a real, non-degenerate, correctly-gated tuning-window win failing to transfer to holdout, after `TIER_WEIGHT` and `var_scale` — all three share the same monotonic-to-boundary shape.
4. **The two new stress-test checks are kept regardless of the rejection** — a genuine, permanent improvement to the safety suite (closes a real blind spot, costs ~0.1s per run).
5. **Scoped ticket 21 vs ticket 24 vs "real ceiling" per the brief's step 5, since the RW_VAR_PER_WEEK lead is now exhausted.** Both are **not indicated** by the accumulated evidence: ticket 21's premise (regime-B needs richer per-match evidence) is undercut by the round-3 finding that the mean state already extremizes to match/exceed Elo's confidence — the actual remaining defect is downstream, in MC-averaging's Jensen's-gap convexity cost, which more evidence cannot touch. Ticket 24's premise (absorbing-boundary clamp pathology) is undercut by zero holdout rows anywhere near the `MAX_STATE`/`MAX_VAR` wall. The one theoretically-open fix for the actual diagnosed mechanism (a *conditional*, curvature-shaped MC variance-scale rather than a flat one) is now "0-for-2" on this mechanism family after `var_scale` and `RW_VAR_PER_WEEK` both failed the same way. **Honest conclusion: this reads as a genuine architecture ceiling for same-session tuning**, not an unexplored lever — recommend NOT starting either ticket on current evidence.

**Gap to Elo: still 0.0114 (v25 unchanged, best config).** Every hyperparameter axis and every well-motivated lever flagged across five rounds has now been tried, gated properly, and either rejected or failed to transfer. Absent a genuinely new idea (not a variant of MC-variance-scaling, calibration, or evidence-weighting — all exhausted), this is a reasonable point to report back rather than keep iterating on variations of what's already been tried three times.

## UPDATE 2026-08-28 (round 4, ran in parallel with the round-3-third-continuation agent below): WTA split confirmed on the real holdout, three new levers tried, all REJECTED — gap still 0.0114

Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 4" section. Headline:

1. **Confirmed on the real 8,594-row odds-joined holdout** (not just the 10,678-row exploratory set the previous round's WTA finding used): ATP gap to Elo **+0.0008** (N=60 basis) / **−0.0040** (converged mc600 basis) — parity or a slight hier edge either way. WTA gap **+0.0223** / **+0.0185**. The WTA-concentration finding holds on the number that actually matters.
2. **Three genuinely new, tour-conditioned levers tried, all REJECTED, none transferred to holdout:**
   - `DSV_WTA_LOWER_WEIGHT` (new hyperparameter in `hierarchical_model.py`, discounts WTA qual/ITF regime-B evidence only, ATP Challenger untouched) — clean, monotonic rejection: WTA's lower-tier evidence is net-positive signal, discounting it only makes WTA worse.
   - Tour-specific hyperparameter re-scan (`REGIME_B_N`, `MAX_SHIFT`, `RW_VAR_PER_WEEK`, each split ATP-only vs WTA-only vs pooled on 2022-23) — no divergence found for any of the three; the pooled optimum is genuinely close to correct for both tours, not a compromise. (`RW_VAR_PER_WEEK` toward zero shows a large, real, but suspicious monotonic-to-boundary tuning-window improvement for BOTH tours — flagged as an unexplored lead for a future round, NOT chased to holdout this round since it disables a mechanism the stress-test suite was built to protect and needs its own scrutiny; does not favour WTA specifically.)
   - Tour-specific (WTA-only) isotonic recalibration — same trade-off pathology as the two pooled recalibration attempts already rejected earlier in this ticket (Platt, isotonic): improves WTA's own log-loss trivially (0.0006 of the 0.0185 gap) while making WTA's lopsided log-loss substantially worse. Confirms the entanglement between hier's overconfidence and market-lopsidedness is a property of WTA's own distribution, not a pooling artifact.
3. **A best-of-format-controlled comparison rules out Bo3/Bo5 mix as the explanation**: ATP best-of-3-only gap is **−0.0116** (even better than ATP's blended number — its Bo5 subset is actually a net NEGATIVE, +0.0280, not a boost), so comparing WTA (100% Bo3) against ATP's own Bo3-only number is the correct apples-to-apples cut: **WTA trails ATP by 0.0301 log-loss on the identical match format.**
4. **Implemented and charter-tuned the parallel agent's `var_scale` MC-curvature lever** (see their section below for the mechanism) in `mc_parallel.py` — a genuine, non-boundary tuning-window optimum at `var_scale≈0.7` that improved the actual selection metric (weighted 0.5671→0.5654). **Revealed once on holdout: REJECTED.** Overall gap more than doubled (+0.0071→+0.0151), ATP flipped from beating Elo to losing to it. A second confirmed case (after ticket 22's `TIER_WEIGHT`) of a genuine, non-degenerate tuning-window win on the correct selection metric failing to transfer to holdout. Left wired at default `1.0` (no-op).
5. **Windows gotcha, new**: a missing `if __name__=="__main__":` guard around a script calling `mc_average()` (which uses `mp.Pool`) caused a **runaway process-spawn loop** on the first holdout-reveal attempt (Windows spawn multiprocessing re-imports the entry script in each worker) — caught via `tasklist` showing the python process count climbing (5→13 in a few seconds) and killed with `taskkill //T //F` before it became a repeat of the earlier machine crash. Always wrap the top-level body of any one-off script that calls `mc_average`/anything Pool-based in `if __name__=="__main__":` — every existing `tune_*.py` script already had this guard, which is why this hadn't been hit before.

**Recommended next step for whoever picks this up:** six independent, well-motivated attempts have now failed to close this gap across two agents this round (three tour-conditioned levers here, three MC-convexity findings from the parallel round-3-third-continuation section below) — on top of round 3's nine and round-3-continuation's four. The one unexplored thread with a real, non-boundary tuning-window signal is `RW_VAR_PER_WEEK` toward zero (attempt 17 in ticket 20's round-4 section) — but it needs its own dedicated stress-test/corruption-gate pass (it weakens the layoff/time-decay uncertainty mechanism the safety suite protects) before any holdout reveal, not a rushed one. Absent that panning out, the honest conclusion is close at hand: this may be a genuine, real-variance-in-this-era WTA/architecture mismatch that no tested lever fixes, per the task brief's own hedge.

## UPDATE 2026-08-28 (round 3, third continuation, investigation-only, ran in parallel with a WTA-hyperparameter-split agent): mechanism identified — MC-averaging convexity cost, not WTA-specific, not insufficient mean extremization

Gap to Elo still **0.0114** (no candidate run this session — investigation-only, no edits to `hierarchical_model.py`). Full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 3, third continuation session" section. Headline findings, superseding the previous section's "gap is WTA-specific" framing:

1. **The extremization undershoot on lopsided matches is present within ATP alone**, at comparable-or-greater raw calibration-error magnitude to WTA's top bucket (ATP +0.1198 vs WTA +0.1090 in the (0.9,1.0] market-confidence bucket) — it is a general architecture pattern, not a WTA artifact. ATP's near-parity AGGREGATE log-loss only looks fine because ATP has a separate, real, compensating edge over Elo in its LEAST-lopsided bucket (hier beats Elo by 0.0346 log-loss on ATP's 50-60%-confidence matches) that WTA does not share — the two tours have the SAME top-bucket defect, but only ATP has an offsetting strength elsewhere.
2. **The mean state (`d1_mean`/`d2_mean`) is NOT the problem.** Point-estimate predictions (no MC) match Elo almost exactly in the highest-confidence bucket (ATP: 0.9409 vs Elo 0.9400; WTA: 0.9606 vs Elo 0.9331 — WTA's point estimate is if anything MORE extreme than Elo's). `OU_THETA_PER_WEEK`'s convergence to ~0 already removed the one plausible mean-shrinkage mechanism; there's nothing left pulling the mean back.
3. **The entire undershoot is introduced by the MC-posterior-averaging step**, and this happens essentially identically in both tours: `mc_cost` (point estimate minus `p_winner_hier_mc600`) in the top confidence bucket is 0.1329 (ATP) vs 0.1342 (WTA) — a Jensen's-gap effect, confirmed by a proof-of-concept re-sample showing the expected monotonic response to shrinking `d1_var`/`d2_var` before MC sampling.
4. **A new, precisely-specified, NOT-yet-implemented fix**: a `var_scale` multiplier on `d1_var`/`d2_var` applied only inside `mc_parallel.py`'s MC sampling (not `hierarchical_model.py` — this lever needs no fit-time changes at all, so it doesn't conflict with any WTA-hyperparameter-split work happening in `hierarchical_model.py`). Full implementation location, rationale for why this shape may avoid the Platt/isotonic global-recalibration failure mode, and the required charter-compliant tuning steps (none done yet) are in ticket 20's new section — read that before implementing.

**Recommended next step for whoever picks this up:** (a) if the WTA-hyperparameter-split work in `hierarchical_model.py` has landed, re-run the ATP/WTA split above against the new config to see how much of the aggregate gap it closed and whether the tour-symmetric MC-convexity defect is still there (it should be, since that split doesn't touch MC sampling); (b) implement and tune the `mc_parallel.py` `var_scale` lever per ticket 20's spec — scan on 2022-23 through the weighted/lopsided metric, verify it doesn't erode the 50-70% bucket, reveal once on the holdout.

## UPDATE 2026-08-28 (round 3, second continuation session): layoff-bump lever REJECTED, gap unchanged at 0.0114 — but the gap is now known to be almost entirely WTA-specific

Config and headline numbers unchanged from the round-3 update below (v25 is still best, gap to Elo still **0.0114**, log-loss 0.6266 vs 0.6152). Two things happened this session, full detail in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 3, second continuation session" section:

1. **Part 1 — the layoff-bump lever from `diag2_posthoc_bump.py` was taken through the charter properly and REJECTED**, in both a fit-time form (a new `DSV_SHORT_LAYOFF_MAX_WEEKS` bound was added to the pre-existing but unused `DSV_SHORT_LAYOFF_BUMP` mechanism in `hierarchical_model.py`, scoping it to the diagnostic's actual 1-12wk finding instead of the already-rejected unbounded version) and a pure eval-time form (reproducing the original diagnostic exactly, but properly on the 2022-23 tuning window with market odds joined). **Both show the identical shape already rejected twice this session (Platt scaling, isotonic): a small plain-log-loss improvement bought by monotonically WORSE weighted/lopsided log-loss** — the metric v25 was actually selected on gets worse from bump=0 in both forms. No holdout reveal was needed (charter: reveal only a candidate you believe in). `DSV_SHORT_LAYOFF_MAX_WEEKS` is left wired in, default no-op, same as `DSV_TIER_WEIGHT`.
2. **Part 2 — five previously-written diagnostic scripts were run.** Four are informative negative/confirmatory results (see ticket 20 for full numbers); the fifth (`diag2_wta_churn.py`) found the single most useful thing this session: **the entire gap-to-Elo is concentrated in WTA** (WTA gap +0.0194; ATP gap **−0.0018**, i.e. hier is already at/above parity with Elo on ATP). Two independent cuts (market-confidence buckets in `diag2_fresh_cuts.py`, and match-length in `diag2_bo5_sets.py`) both show hier specifically undershoots confidence on the most lopsided/decisive wins, converging with the WTA finding rather than contradicting it. **Ticket 24's premise was also directly checked and found not to apply here**: zero of the 10,678 holdout rows are anywhere near the `MAX_STATE` wall (`wall_proxy > 4.0` count = 0), so the heavy-tailed-process-noise rewrite would not touch any holdout row even if built.

**Recommended next step for whoever picks this up:** investigate WTA specifically — first check whether hyperparameters tuned on pooled ATP+WTA 2022-23 data (`SURFACE_VAR_SCALE`, `RW_VAR_PER_WEEK`, etc.) are systematically wrong for WTA alone (a per-tour split, not yet tried), then trace why `d1_mean`/`d2_mean` separation doesn't extremize as far as Elo's on genuinely dominant true mismatches. This is a real investigative effort (WTA-specific `DSV_DEBUG_UID` traces, the way ticket 18/20 did for ATP's Djokovic/Prizmic/Sinner), not a one-line tuning knob — don't rush an implementation without first confirming the mechanism on a WTA player trace.

## UPDATE 2026-08-28 (round 3): gap closed from 0.0315 to 0.0114 — read this before the pre-round-3 sections below

Current best config is **v25**, not v20. On the same holdout join:

| | Elo | Hierarchical (v25) | Market |
|---|---|---|---|
| Overall log-loss | **0.6152** | **0.6266** | 0.5888 |
| Overall accuracy | 0.6534 | 0.6453 | 0.6792 |
| Lopsided log-loss | 0.3853 | 0.4288 | 0.3645 |

v25 config (env vars for `hierarchical_model.py`):
```
DSV_RW_VAR_PER_WEEK=0.005 DSV_SURFACE_VAR_SCALE=1.0 DSV_OU_THETA_PER_WEEK=0.00001
DSV_MAX_SHIFT=0.125 DSV_PHI_MAX=4.0 DSV_LAYOFF_C=0.06 DSV_REGIME_B_N=0.25
DSV_TIER_WEIGHT=1.0 DSV_CLIP_REINFLATE_K=0.5
```

What changed this round (full detail + all numbers in `issues/20-why-does-hierarchical-lose-to-elo.md`'s "Round 3" section — read that, not just this summary):
1. **Ticket 21 implemented** — but NOT the DP scoreline-likelihood this file originally sketched. Cheaper and it worked: regime-B's fake `adf_update(cw,vw,cl,vl,tier_n,tier_n)` (telling the filter "won 100% of trials") was replaced with the REAL games-won-per-set counts (`parse_score_games()` in `hierarchical_model.py`, 99.6% coverage) plugged straight into the *same* `adf_update` Newton solve. Zero new math. `DSV_REGIME_B_N` is now the evidence-strength dial on this real data (optimum 0.25, not 1.0).
2. `OU_THETA_PER_WEEK` re-tuned jointly with the new evidence → converged to ~0 (old mean-reversion safety net is no longer load-bearing).
3. A scoped-down piece of **ticket 24** — reinflate variance on a clamped/surprising update instead of only protecting it from shrinking (`DSV_CLIP_REINFLATE_K`, new). Jointly re-tuned with `MAX_SHIFT` → 0.125/0.5 (both moved from their old values).
4. Kappa/surface-term ablation re-tested (hypothesis: surface noise causes the weak-disagreement-with-Elo pathology) — **rejected**, removing kappa made things worse.
5. RW_VAR_PER_WEEK, SURFACE_VAR_SCALE, TIER_WEIGHT, LAYOFF_C, PHI_MAX, and newly-exposed INIT_VAR — **all re-scanned and confirmed flat/exhausted** at the v25 config. Don't re-scan these again without a reason to think the config changed enough to matter.
6. **Speed fixes** (zero behavior change, verified): `mc_parallel.py` parallelizes MC-averaging across the machine's 24 cores (was serial, ~11.6x speedup — a holdout MC pass is now ~30-35s not 340-470s). `hierarchical_model.load_prepared()` caches the parsed archive (was ~20.5s re-paid on every subprocess). `prepare()` now casts pandas-3.0's slow PyArrow-backed `str` columns to plain `object` dtype before the fit loop (was 26s of a 65s fit just for `itertuples()` overhead). **Use `mc_parallel.py` for all MC-averaging now, not a manual serial loop** — `from mc_parallel import mc_average; mc = mc_average(df, n_samples=60)`.
7. A parallel diagnostic agent was launched to cut the eval data multiple ways (market-favorite-strength buckets, surface, tour, best-of, layoff length, player experience) looking for the next lever, alongside a from-scratch weak-disagreement finding (see ticket 20) that turned out to have negligible aggregate effect when a fix was tried.

**Remaining gap: 0.0114.** Hyperparameters are exhausted — the next lever is either whatever the diagnostic agent found, or ticket 24's full heavy-tailed rewrite (not yet attempted, only the scoped-down clip-reinflate piece has been).

---

## Pre-round-3 state (for history — the numbers below are stale, v20 is no longer the best config)

## The number to beat

On the untouched 2024-25 holdout (n=8,594 odds-joined rows, `elo_eval_predictions.parquet` + Pinnacle odds via `market_compare.py`):

| | Elo | Hierarchical (current best, v20) | Market |
|---|---|---|---|
| Overall log-loss | **0.6152** | 0.6467 | 0.5888 |
| Lopsided (mkt_fav>0.80) log-loss | 0.3853 | 0.4244 | 0.3645 |

**Exit condition: get the Hierarchical overall log-loss below 0.6152 on this exact holdout, verified with the exact join pattern below.** Not on the tuning window — on this holdout, for real.

## What's already been tried and rejected — READ `issues/20-why-does-hierarchical-lose-to-elo.md` FIRST, do not re-derive this

Nine fixes tried this session, all rejected or non-transferring:
1. Forcing kappa's contribution to ~0 (ablation) — made it worse (0.6548).
2. MC posterior-averaging removal — catastrophically worse (0.78 point-estimate logloss).
3. "Well-known players need less variance-discounting" (Pete's hypothesis) — zero correlation (0.018) between combined variance and the Elo gap across the lopsided bucket.
4. Linear Platt scaling — fixed overall calibration (0.6467→0.6343) but WORSENED lopsided confidence (0.779→0.710).
5. Isotonic calibration (tried twice, second time boundary-safe) — same failure as #4, confirmed not an artefact.
6. Regime-B evidence up-weighting (`DSV_REGIME_B_N` > 1.0) — hard corruption-gate rejection at N=2.0 already.
7. Tournament-tier down-weighting (`DSV_TIER_WEIGHT`) — genuine win on 2022-23 tuning (0.5974→0.5912) that **did not transfer to holdout at all** (stayed at 0.6467-0.6468).
8. Fixing bug-8's missing variance-scaling in regime-B's own bookkeeping — made corruption WORSE (0.0157%→0.01906%) by reopening bug-9's pathology through a different path. Reverted.
9. (implicitly) every combination/interaction of the above.

**The core finding:** the model is systematically overconfident, worst in the 70-90% predicted-probability band (bias -0.087/-0.100 vs Elo's -0.022/-0.045), and this can't be fixed by recalibrating the output (tried, twice, properly) or by reweighting evidence (tried, rejected by the safety gate). The defect is upstream, in how much genuine evidence the model extracts per match — specifically regime-B (72.5% of matches, scoreline-only), which currently treats a whole match win as a fake `n=1` Bernoulli trial via `adf_update(cw, vw, cl, vl, N, N)` at `hierarchical_model.py` (search `regime B: combined-strength update`), rather than using the real, richer likelihood available.

## What to try next — in priority order

### Priority 1: Ticket 21 — build the real scoreline-inversion likelihood

File: `issues/21-build-scoreline-inversion-regime-b.md`. Ticket 12 (`issues/12-scoreline-inversion-gap.md`) already proved this is feasible: per-match scoreline inversion is hopeless (a single 6-4 6-4 puts the MLE at the grid boundary) but a panel of ~150 sets from the same player pool recovers the true serve/return split to within ±0.03. Regime-B currently has access to `score` (e.g. "6-4 6-4") and `best_of` but throws away everything except win/loss.

**Concrete implementation idea not yet tried:** instead of a full panel-based rolling inversion (complex), first try the cheaper intermediate step — parse games-won-per-set from `score`, and replace the fake `adf_update(cw, vw, cl, vl, 1, 1)` call with a likelihood built from the ACTUAL games-per-set Markov recursion (ticket 05 has the closed-form game/set/match recursions already implemented as `p_game`, `p_set`, `p_match` in `hierarchical_model.py`). Specifically:
- Parse `score` into games won per set for winner/loser (watch for tiebreak notation, retirements already excluded).
- Define `p_a_serve = p_b_serve = sigmoid(d/2)` (symmetric combined-strength assumption, unavoidable per ticket 12) where `d = cw - cl`.
- The likelihood of observing the ACTUAL games-per-set score (not just win/loss) given `d` is much richer than a single bit — use `p_set`'s internal game-by-game DP (may need a new function that returns the probability of a SPECIFIC game-score outcome, not just who wins the set) to compute `log P(observed scoreline | d)`, then Newton-step on `d` using numerical (finite-difference) gradient/Hessian of this richer log-likelihood.
- This is a bigger lift than any single-scalar hyperparameter tried so far — budget real implementation time, and it needs NEW stress tests (a scoreline-likelihood equivalent of `stress_test_safety.py`'s reproductions) before trusting it, per this session's own repeated lesson that new machinery here needs real-population tracing (`DSV_DEBUG_UID`) too, not just synthetic tests.

### Priority 2: Ticket 24 — heavy-tailed process noise (bigger, riskier, do only if 21 doesn't close the gap)

File: `issues/24-robust-filtering-replace-hard-clamps.md`. Replace the Gaussian random-walk assumption (and the `MAX_SHIFT`/`MAX_STATE` hard clamps bolted onto it) with a Student-t or scale-mixture-of-normals process noise, which naturally absorbs large surprising shifts without ever letting variance "lock in" a wrong estimate (the absorbing-boundary trap that's caused 5+ distinct bugs this session, including attempt #8 above). This needs new ADF/Laplace update math since `adf_update`'s closed-form Newton step assumes a Gaussian prior — a Student-t prior has no equivalent closed form and likely needs a scale-mixture reformulation or numerical integration.

### Priority 3: Ticket 22/23 remnants
`DSV_TIER_WEIGHT` (regime-B only) is implemented and parked at default 1.0 (no-op) — didn't transfer to holdout on its own, but MIGHT combine usefully once ticket 21's richer regime-B likelihood exists (worth re-testing tier-weighting AFTER ticket 21, not before). Ticket 23 (bo3/bo5 skill transfer) is pure research, unlikely to move the needle alone — deprioritize unless 21 and 24 are both exhausted.

## The verification harness — reuse this exactly, don't re-derive

**1. Sanity-check any code change:**
```bash
cd "C:/dev/dropshotverse/dropshotdata/scripts"
python stress_test_safety.py   # must print ALL PASS, exit 0
```

**2. Fit + eval on the holdout (only do this for a candidate you actually believe in — each fit is ~60-90s):**
```bash
cd "C:/dev/dropshotverse/dropshotdata/scripts"
export DSV_RW_VAR_PER_WEEK=0.005
export DSV_SURFACE_VAR_SCALE=1.0
export DSV_OU_THETA_PER_WEEK=0.0005
export DSV_MAX_SHIFT=0.2
export DSV_PHI_MAX=4.0
export DSV_LAYOFF_C=0.06
export DSV_EVAL_YEARS=2024,2025
export DSV_OUT_PREDS="C:\dev\dropshotverse\dropshotdata\hier_vNEXT_eval_predictions.parquet"
export DSV_OUT_STATE="C:\dev\dropshotverse\dropshotdata\hier_vNEXT_final_state.pkl"
python hierarchical_model.py > "C:/dev/dropshotverse/dropshotdata/tune_scratch/vNEXT_fit.log" 2>&1
```

**3. MC posterior-average (N=60, ~5-6 min for ~10,680 rows — DO NOT skip this, point estimates alone are catastrophically overconfident, see rejected attempt #2):**
```python
import numpy as np, pandas as pd, time
from hierarchical_model import p_match, sigmoid
N_SAMPLES=60
rng=np.random.default_rng(0)
p = pd.read_parquet(r'C:\dev\dropshotverse\dropshotdata\hier_vNEXT_eval_predictions.parquet')
out=np.empty(len(p))
for i,row in enumerate(p.itertuples()):
    d1=rng.normal(row.d1_mean, np.sqrt(max(row.d1_var,1e-6)), N_SAMPLES)
    d2=rng.normal(row.d2_mean, np.sqrt(max(row.d2_var,1e-6)), N_SAMPLES)
    ps=[p_match(sigmoid(a),sigmoid(b),int(row.best_of)) for a,b in zip(d1,d2)]
    out[i]=np.mean(ps)
p=p.assign(p_winner_hier_mc=out)
p.to_parquet(r'C:\dev\dropshotverse\dropshotdata\hier_vNEXT_eval_predictions.parquet')
```

**4. Join to Elo + market odds and score (the exact pattern used throughout this session):**
```python
import numpy as np, pandas as pd
from market_compare import key_from_full, key_from_short, devig_power, load_odds

elo = pd.read_parquet(r'C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet')
elo = elo[elo.level_file=='tour'][['date','tour','winner_uid','loser_uid','winner_name','loser_name','p_winner']].rename(columns={'p_winner':'p_elo'})
hier = pd.read_parquet(r'C:\dev\dropshotverse\dropshotdata\hier_vNEXT_eval_predictions.parquet')[['date','tour','winner_uid','loser_uid','p_winner_hier_mc']]
m = elo.merge(hier, on=['date','tour','winner_uid','loser_uid'], how='inner')
m['k_w']=m['winner_name'].map(key_from_full); m['k_l']=m['loser_name'].map(key_from_full)
m['pair']=m.apply(lambda r:'|'.join(sorted([r.k_w,r.k_l])),axis=1)
odds=load_odds(years=(2024,2025)).dropna(subset=['Winner','Loser','Date'])
odds['k_w']=odds['Winner'].map(key_from_short); odds['k_l']=odds['Loser'].map(key_from_short)
odds['date']=pd.to_datetime(odds['Date'],errors='coerce'); odds=odds.dropna(subset=['date'])
odds['pair']=odds.apply(lambda r:'|'.join(sorted([r.k_w,r.k_l])),axis=1)
j = m.merge(odds[['pair','tour','date','k_w','k_l','PSW','PSL']],on=['pair','tour'],how='left',suffixes=('','_o'))
j['gap']=(j['date_o']-j['date']).dt.days
j=j[(j['gap']>=-3)&(j['gap']<=21)]
j=j.sort_values('gap').drop_duplicates(subset=['pair','tour','date'],keep='first')
j=j[(j['k_w_o']==j['k_w']) & j[['PSW','PSL']].notna().all(axis=1)]
j['p_mkt']=devig_power(j['PSW'].values.astype(float),j['PSL'].values.astype(float))
j=j.dropna(subset=['p_mkt'])
for col,name in [('p_elo','Elo'),('p_winner_hier_mc','Hierarchical'),('p_mkt','Market')]:
    pr=np.clip(j[col].values,1e-9,1-1e-9)
    print(f'{name:15s} acc={float((pr>.5).mean()):.4f}  logloss={float(-np.log(pr).mean()):.4f}')
```

**Charter — do not violate:** tune ONLY on 2022-23 (`DSV_EVAL_YEARS=2022,2023`, tuning-window parquets go in `tune_scratch/`), reveal on 2024-25 exactly once per real candidate. Any candidate must pass `stress_test_safety.py` AND the corruption gate (`corruption_rate()` in `tune_v2.py`/`tune_v3_lopsided.py`/`tune_regime_b.py` — fraction of players with `|mu|>2.5`, must stay ≤ `0.0067% * 3 = 0.0201%`) before its log-loss is even looked at.

## Background-task gotcha hit twice this session

Backgrounding a fit/scan via `run_in_background: true` got silently killed twice mid-run (status "killed" in the task notification), and once left a STRAY python.exe process still running, racing on the same output files as a restarted job and corrupting the log. **Before trusting any scan result, run `tasklist //FI "IMAGENAME eq python.exe" //FO CSV` and confirm no unexpected processes are running.** Prefer running fits/scans in the FOREGROUND (they complete in 60-600s, well within the Bash tool's normal timeout) rather than backgrounding, unless the wait is genuinely long.

## Known trap to remember

The absorbing-boundary corruption mechanism (variance shrinks before the mean is correct, locking in a wrong value forever) is NOT one bug — it's a structural property of this Gaussian-filter+hard-clamp architecture, and it's been found in at least 6 distinct forms across this session and the prior one (bugs 6, 7, 8, 9, plus attempt #6 and #8 above). **Any new update-math change needs its own corruption-gate scan across a real range of values, not just the default**, because the default passing tells you almost nothing about whether the new mechanism is safe under stress — REGIME_B_N=1.0 passed fine both times; the pathology only showed up at N≥2.
