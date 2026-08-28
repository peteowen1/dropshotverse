# 20 — Why does the hierarchical model still lose to plain Elo?

Type: research
Status: claimed (in progress — ~38 attempts tried and rejected/non-transferring/flat across 9 rounds as of 2026-08-28; tickets 21 and 24 both built, tuned, and found flat/no-signal (rounds 7-8); round 9 fixed a genuine data-prep causal-ordering bug (qualifying rounds sorted after the whole main draw) which showed a real tuning-window win but still did not transfer to holdout, ruling out the data-prep layer as the mechanism; the diagnosed root cause, MC-averaging's Jensen's-gap convexity cost, still has no untried fix)

## Question

Raised by Pete after ticket 19 closed: the hierarchical model has strictly more
structure than Elo (serve/return decomposition, surface offsets, explicit
posterior uncertainty) — "surely at worst our model can replicate Elo... our
model has MORE levers than Elo so should be better, what's going on?"

This ticket is a diagnostic log, not a fix. It records every hypothesis tested
this session, in order, with the data cut that tested it and the verdict —
so a fresh session/skill can pick up from "ruled out" rather than re-deriving
these from scratch.

## Baseline numbers being explained (2024-25 holdout, n=8,594 odds-joined rows, v20 config)

| | Elo | Hierarchical (v20) | Market |
|---|---|---|---|
| Overall acc | 0.6534 | 0.6310 | 0.6792 |
| Overall log-loss | 0.6152 | 0.6467 | 0.5888 |
| Lopsided (mkt_fav>0.80) log-loss | 0.3853 | 0.4244 | 0.3645 |
| Lopsided mean confidence | 0.839 | 0.779 | 0.875 |

v20 config: `RW_VAR_PER_WEEK=0.005, SURFACE_VAR_SCALE=1.0, OU_THETA_PER_WEEK=0.0005, MAX_SHIFT=0.2, PHI_MAX=4.0, LAYOFF_C=0.06` (ticket 19's lopsided-weighted optimum). This is the best hyperparameter config found so far and still loses to Elo on every headline number.

## Hypothesis 1 — kappa (surface offset) dilutes the mu update, "more levers = weaker signal per lever"

**Theory:** the Kalman-gain split (`share_mu`) between `mu` and `kappa` in `apply_side_update` routes some of each match's signal away from the number that actually drives predictions, unlike Elo which puts 100% of the signal into one scalar.

**Test:** ablation — added `DSV_FORCE_SHARE_MU` env override (now permanent in `hierarchical_model.py`, default off) to force `share_mu=1.0` (kappa gets ~zero of every update) and set `OU_THETA_PER_WEEK=0` (no reversion), refit full history, eval on 2024-25.

**Result:** WORSE than the tuned v20, not better.

| | Elo | Hierarchical (share_mu=1, OU=0 ablation) | Market |
|---|---|---|---|
| Overall log-loss | 0.6152 | 0.6548 | 0.5888 |
| Lopsided log-loss | 0.3853 | 0.4510 | 0.3645 |

**Verdict: REJECTED.** Forcing the model to behave more like single-scalar Elo made it worse, not better. Kappa dilution is not the (or at least not the dominant) mechanism.

## Hypothesis 2 — regime-B (scoreline-only, 72.5% of matches) update rule is too weak

Regime B treats a whole match as a single `n=1, y=1` Bernoulli trial through `adf_update`, then splits the resulting shift 50/50 between `m_a` and `m_b`. Flagged as a plausible dilution mechanism (a full match win carries much more information than a single coin flip) but **not empirically tested this session** — the ablation above touched `share_mu` (mu vs kappa split) and `OU_THETA` only, not the regime-B `adf_update(cw, vw, cl, vl, 1.0, 1.0)` call or its 50/50 alpha/beta split. This remains untested and is a real candidate for tomorrow's follow-up.

## Hypothesis 3 — MC posterior-averaging over-corrects, throwing away earned confidence

**Theory:** the Jensen's-gap fix (averaging match-win probability over posterior samples of `d1`/`d2` instead of plugging in point estimates) removes too much confidence, since Elo has no analogous uncertainty-discounting step at all.

**Test:** compared the model's raw point-estimate prediction (`p_winner_hier`, means only, no MC) against the MC-averaged prediction (`p_winner_hier_mc`) on the same 8,594-row holdout.

**Result:**

| | Elo | Hier POINT-ESTIMATE (no MC) | Hier MC-averaged | Market |
|---|---|---|---|---|
| Overall log-loss | 0.6152 | **0.7800** | 0.6467 | 0.5888 |
| Lopsided log-loss | 0.3853 | **0.4655** | 0.4244 | 0.3645 |

Point estimates are catastrophically overconfident in aggregate (0.78 log-loss — worse than a coin-flip baseline of 0.693) even though they looked great on one cherry-picked anchor match (Djokovic d. Prizmic: point estimate 99.4% vs Elo 97.6% vs MC-averaged 87.0%).

**Verdict: REJECTED, and inverted.** MC-averaging is net-positive (removes over a full point of log-loss) and is not the source of the gap — if anything it's currently under-correcting relative to what point estimates need.

## Hypothesis 4 — variance is legitimately wider for low-data players ("maybe the adjustment doesn't need to be as big for well-known players")

**Theory (Pete's):** star players like Djokovic have tons of history and should have tiny variance; the MC-averaging haircut should shrink toward zero for them specifically. If it isn't, that's the bug.

**Test A — trace Djokovic's own state going into the AO 2024 Prizmic match** (`DSV_DEBUG_UID=atp:104925`): `v_a≈0.03` (Hard court) right before that match — genuinely tight, confirms the instinct that HIS OWN uncertainty is small.

**Test B — decompose `d1_var`:** it is NOT one player's variance — it's `v_e1 + v_f1`, the SUM of the server's serve-variance and the returner's return-variance. Traced Prizmic's own state (`atp:209976`) going into the same match: `v_b≈0.041` (Hard) — comparable in magnitude to Djokovic's, not wildly larger despite Prizmic being a teenage qualifier. Combined `d1_var+d2_var≈0.175`, enough for the Bo5 recursion's convexity to pull a 99.4% point estimate down to 87.0%.

**Test C — does the Elo-gap actually correlate with combined variance across the whole lopsided bucket, not just this one match?** Quartiled the 2024-25 lopsided bucket (n=1,669) by `d1_var+d2_var`:

| Variance quartile | mean combined var | Hier log-loss | Elo log-loss | gap vs Elo |
|---|---|---|---|---|
| Q1 (lowest, best-known players) | 0.19 | 0.413 | 0.344 | 0.069 |
| Q2 | 0.26 | 0.436 | 0.434 | 0.002 |
| Q3 | 0.34 | 0.404 | 0.376 | 0.028 |
| Q4 (highest, least-known players) | 0.64 | 0.444 | 0.387 | 0.057 |

Correlation between combined variance and log-loss gap to Elo: **0.018** — noise. The gap is flat across variance levels; the best-known-player quartile loses to Elo by *more*, not less, than the least-known-player quartile.

**Verdict: REJECTED.** Uncertainty/data-sparsity is not the mechanism. This was Pete's own hypothesis and the data doesn't support it — worth stating plainly since it was a reasonable guess that just didn't pan out.

## Hypothesis 5 — the model is systematically overconfident in a specific probability band (calibration bias, not a variance/architecture problem)

**Test:** calibration table — bucket by predicted probability assigned to whichever side the model favoured, compare to that side's actual win rate, for Elo / Hierarchical / Market side by side on the same 8,594 rows.

**Result:**

| Predicted bucket | Elo bias | Hierarchical bias | Market bias |
|---|---|---|---|
| 50-60% | -0.021 | -0.037 | +0.003 |
| 60-70% | -0.053 | -0.036 | -0.005 |
| 70-80% | -0.022 | **-0.087** | -0.008 |
| 80-90% | -0.045 | **-0.100** | +0.002 |
| 90-100% | -0.030 | -0.062 | -0.004 |

(bias = actual win rate − predicted; negative = overconfident.) Market is essentially perfectly calibrated, as expected (it's devigged odds). Elo is mildly overconfident everywhere (2-5pp). **Hierarchical is overconfident everywhere, roughly double Elo's miscalibration specifically in the 70-90% band** — exactly the band that carries the most lopsided-match volume and the highest cost-per-wrong-call.

**Verdict: CONFIRMED, currently the leading explanation.** Consistent direction, non-trivial magnitude, concentrated in a specific band — a genuine, correctable calibration defect rather than a variance or architecture problem.

## Hypothesis 5 continued — Platt scaling ran; global linear correction is NOT sufficient

Fit: 2022-23 tuning window, symmetric augmentation (`(logit(p),1)` and `(-logit(p),0)` per match), plain logistic regression, no regularisation. Result: `a≈0.0000, b=0.6554` (as expected, `a≈0` by the symmetry of the construction; `b<1` means shrink every prediction toward 0.5). Applied `sigmoid(a + b·logit(p_winner_hier_mc))` to the untouched 2024-25 holdout:

| | Elo | Hier v20 (uncalibrated) | **Hier v20 + Platt** | Market |
|---|---|---|---|---|
| Overall log-loss | 0.6152 | 0.6467 | **0.6343** | 0.5888 |
| Lopsided log-loss | 0.3853 | 0.4244 | **0.4606** | 0.3645 |
| Lopsided mean confidence | 0.839 | 0.779 | **0.710** | 0.875 |

**Overall log-loss improved** (0.6467→0.6343, closing most of the remaining gap to Elo) and overall calibration is now excellent — recalibrated bias is under 0.02 in every bucket, better than Elo's own calibration everywhere. **But the lopsided bucket got WORSE**, not better — confidence dropped further (0.779→0.710), moving away from Pete's original ask, not toward it.

**Why:** the original miscalibration is non-monotonic across the probability range (bias -0.037, -0.036, **-0.087, -0.100**, -0.062 across the five buckets — worst in the middle-high 70-90% band, comparatively mild at the very top). A single linear Platt correction (one scalar `b` on the logit scale) can't reproduce a hump-shaped bias — it necessarily shrinks hardest at the most extreme logits (since logit is unbounded, a fixed multiplicative slope produces the largest absolute-probability correction at the tails). Here that meant it undercorrected the actual problem band (70-90%) and overcorrected the >90% tail, which was already comparatively well-behaved — directly at the expense of the lopsided-match confidence metric Pete cares about most.

**Verdict: global linear Platt scaling is REJECTED as the fix**, despite improving the headline overall log-loss. It trades the wrong thing away. This is a real, useful negative result — the instinct "just recalibrate" was right in principle but the wrong functional form.

## Hypothesis 5 continued again — isotonic regression, ALSO rejected

Ran the same tune(2022-23)/holdout(2024-25) harness with `sklearn.isotonic.IsotonicRegression` (nonparametric, can fit a non-monotonic-in-probability bias curve) instead of linear Platt.

| | Elo | Hier v20 (uncal) | Hier + Platt | **Hier + isotonic** | Market |
|---|---|---|---|---|---|
| Overall acc | 0.6534 | 0.6310 | 0.6310 | **0.6165** | 0.6792 |
| Overall log-loss | 0.6152 | 0.6467 | 0.6343 | **0.6386** | 0.5888 |
| Lopsided log-loss | 0.3853 | 0.4244 | 0.4606 | **0.4673** | 0.3645 |
| Lopsided confidence | 0.839 | 0.779 | 0.710 | **0.711** | 0.875 |

**Two problems.** First, accuracy dropped (0.6310→0.6165) — a real defect: fitting isotonic regression on pooled `(logit, label)` data without explicitly enforcing sign-symmetry lets the fitted curve cross `p=0.5` at a logit slightly off zero, which flips the predicted favourite for matches near the boundary. Second, and more important even if that bug were fixed: **isotonic lands in almost exactly the same place as linear Platt on the lopsided bucket** (0.4673 vs 0.4606 log-loss, 0.711 vs 0.710 confidence) — a genuinely more flexible corrector did not do better where it mattered.

**This reframes the problem.** Both corrections were calibrated against the model's own predicted-probability axis, pooling across ALL matches. Both independently converge on shrinking lopsided-bucket confidence to ~0.71. That convergence suggests the excess confidence isn't simply "a function of what the model's own probability says" (which either calibration method could fix) — it's specifically concentrated in matches that are ALSO lopsided by the market's/reality's own yardstick, a variable neither calibration method conditions on. Recalibrating on the model's own output necessarily trades confidence away from exactly this slice to correct bias elsewhere in the distribution.

**Verdict: post-hoc probability recalibration (in either linear or nonparametric form) is REJECTED as a fix for the lopsided-confidence problem specifically**, even though both forms modestly improve overall log-loss. The real defect is upstream of the final probability — most plausibly in how much evidence per match the model extracts in the first place (hypothesis 2, regime-B, next) — not in how that output number gets mapped to a calibrated probability afterward.

## Hypothesis 2 tested: regime-B evidence-weight scan (`DSV_REGIME_B_N`)

Added a tunable `DSV_REGIME_B_N` (default `1.0`, matches existing behaviour) that scales both `y` and `n` in the regime-B `adf_update(cw, vw, cl, vl, N, N)` call at `hierarchical_model.py:602` — a fractional/tempered-likelihood way of saying "this match's outcome carries as much evidence as N independent point-level trials," instead of the current fake `n=1` framing. Scanned `N ∈ {1, 2, 3, 5, 8}` on 2022-23 through the same corruption-gated harness as tickets 17-19.

**Result: hard rejection, not a middling one.**

| REGIME_B_N | corruption | verdict |
|---|---|---|
| 1.0 (baseline) | 0.0157% | passes (limit 0.0201%) |
| 2.0 | 0.0336% | **REJECTED** |
| 3.0 | 0.0550% | REJECTED |
| 5.0 | 0.1649% | REJECTED |
| 8.0 | 0.2848% | REJECTED |

Corruption rises monotonically and every value above baseline fails the gate — this is not a case of "needs more tuning," it's a structural collision. Regime-B has no kappa/surface-offset protection (kappa is untouched by the regime-B branch entirely — see `hierarchical_model.py:595-614`), so its only defence against runaway drift is the `MAX_SHIFT` clamp. Raising the evidence weight directly increases the raw pre-clamp shift magnitude, pushing players into that clamp more often and more repeatedly in a row — the exact mechanism behind bug 9 (Sinner hitting `MAX_STATE` in his first 3-4 real matches, ticket 18). Tickets 18-19 spent most of this session eliminating that pathology; a naive `REGIME_B_N` increase reopens it immediately.

**Verdict: REJECTED as a simple scalar fix.** This doesn't disprove hypothesis 2 (regime-B may still be under-weighted in an information sense) — it proves the fix can't be "turn one knob up." Two remaining paths, both bigger than a same-day tune:

1. **Ticket 21** (full scoreline-inversion panel-based likelihood) — replaces the fake Bernoulli trial with the real thing rather than approximating its strength with a bigger fake `n`.
2. **A version of this scan that also re-tightens `MAX_SHIFT` specifically for regime-B** as `REGIME_B_N` rises, so more evidence per match doesn't translate into more clamp-hits — cheaper than #1 but needs its own stress-testing (a two-axis search, not one), and risks reintroducing a subtler version of the same absorbing-boundary trap (bug 8) if the two parameters aren't co-tuned carefully.

Recommend #1 (ticket 21) as the more principled next step given how much of this session was spent finding failure modes in exactly this kind of ad-hoc clamp interaction.

## Round 2 (Pete: "keep working until hier logloss beats Elo's")

Three more attempts, all logged with the same rigor.

**Attempt 6 — tournament-tier down-weighting (ticket 22).** Added `DSV_TIER_WEIGHT` (multiplier ≤1.0 for non-elite `tourney_level`, applied only to regime-B, never exceeding the already-validated baseline evidence). Scan on 2022-23 showed a genuine, monotonic, SAFE win — logloss and corruption both improved together as weight decreased (1.0→0.1: weighted 0.5974→0.5922, corruption 0.0157%→0.00224%), converging near `TIER_WEIGHT=0.0` (weighted 0.5912, corruption 0.00673% — almost exactly the known genuine-all-time-greats floor). **This looked like the first unambiguous win of the round-2 effort.** Verified on the untouched 2024-25 holdout as v21 (`hier_v21_eval_predictions.parquet`): **did not transfer.** Overall log-loss 0.6467→0.6468 (no change, noise-level), lopsided log-loss actually got slightly *worse* (0.4244→0.4297). The tuning-window improvement was real but specific to 2022-23 and did not generalize — a `stats-discipline` §2 case worth remembering (a monotonic, safe-looking tuning result still needs the holdout reveal before being trusted). `TIER_WEIGHT` reverted to default (1.0, no-op) in the shipped defaults; the hyperparameter itself is left in place, tested-and-parked.

**Attempt 7 — fix bug 8's missing variance-reduction scaling in regime-B.** Noticed `apply_side_update` (regime A / kappa) scales variance reduction by how much of a shift survived clamping (bug 8's fix, ticket 18) but regime-B's separate, simpler bookkeeping (`hierarchical_model.py:611-631`) never got that fix — full variance-reduction credit even when `dshift`/`lshift` were clamped away. Applied the same scaling. **Measured WORSE, not better**, even at the unchanged `REGIME_B_N=1.0` baseline: corruption rose 0.0157%→0.01906%, and `REGIME_B_N` scan rejections got more severe across the board (N=2.0 corruption 0.0336%→0.1166%, N=8.0 0.2848%→0.9891%). Mechanism: keeping variance from shrinking as fast (the "more correct" behaviour) let MORE matches produce large raw pre-clamp shifts, reopening bug 9's "several near-max clamped shifts in a row march a player to the wall" pathology instead of bug 8's. **The two absorbing-boundary mechanisms are coupled here** — fixing one exposed the other more, not less. Reverted; documented in code as a tried-and-rejected fix, not silently dropped.

**Attempt 8 — boundary-safe isotonic recalibration.** The original isotonic attempt (Hypothesis 5) had a real defect — fitting on symmetrically-augmented `(logit(p),label)` pairs let the decision boundary drift off `p=0.5`, which could have explained its accuracy drop as an artefact rather than a real result. Redid it calibrating only in favourite-probability space (`p_fav = max(p,1-p)`, `y_min=0.5` constraint, boundary provably fixed at 0.5) — the accuracy-metric anomaly is now understood as a measurement edge case (isotonic's flat lower plateau lands exactly on 0.5 for near-toss-ups, and a strict `>0.5` accuracy check scores those as misses even when the favoured side is correct). **The substantive result is UNCHANGED**: overall log-loss 0.6467→0.6386 (real improvement), lopsided log-loss 0.4244→0.4673 (real degradation), identical to four decimal places to the original attempt. This confirms Hypothesis 5's conclusion was not an artefact — calibration genuinely cannot fix the lopsided-confidence problem without trading it away, under either a linear or monotonic correction, done correctly.

## Session status: nine hypotheses tested, all rejected or non-transferring. Elo has not been beaten.

Every same-session, safety-gated lever has now been tried: kappa dilution, MC over-correction, uncertainty/data-sparsity, linear calibration, nonlinear calibration (twice, confirmed), regime-B evidence-up-weighting (twice, with and without a plausible companion fix), and tournament-tier down-weighting (real on tuning, did not transfer to holdout). **The hierarchical model's holdout log-loss remains 0.6467-0.6468 against Elo's 0.6152** — the gap has not closed. The pattern across all nine attempts is consistent: this codebase's absorbing-boundary failure mode (Gaussian filter + hard clamps) is tightly coupled across every path that touches it, and every attempt at new machinery this session — not just the ones that shipped as bugs 6-9 originally — has produced a fresh instability when tested rigorously. That is itself the header finding for whoever picks this up next: **the credible remaining paths (ticket 21, full scoreline inversion; ticket 24, heavy-tailed process noise) are both genuine multi-session architecture efforts, not further same-session tuning**, and rushing either without the same stress-test/corruption-gate/real-population-trace discipline this session insisted on throughout would risk shipping an unverified "win" that doesn't hold up — exactly the failure mode this log exists to prevent.

Five hypotheses tested end-to-end (kappa dilution, MC over-correction, uncertainty/data-sparsity, linear calibration, nonlinear calibration) and rejected; regime-B evidence-weighting tested and rejected as a simple fix. The confirmed, still-standing fact is the calibration table in Hypothesis 5: **the model is genuinely, non-monotonically overconfident (worst in the 70-90% band), and every post-hoc fix tried so far corrects the wrong slice of that curve.** Tickets 21 (scoreline inversion), 22 (tournament-tier/round weighting, currently unused entirely), and 23 (best-of-3/5 skill transfer, unvalidated since ticket 11) are scoped and waiting on the map's frontier for whoever picks this up next.

## Reusable diagnostic tooling added this session

- `DSV_FORCE_SHARE_MU` env override in `hierarchical_model.py` — forces the mu/kappa Kalman-gain split to a fixed value, for ablation only (unset in normal runs, default behaviour unchanged).
- `DSV_DEBUG_UID` (added ticket 18, reused heavily here) — traces one real player's state through the full population-embedded fit. Used this session for both Djokovic (`atp:104925`) and Prizmic (`atp:209976`) to decompose `d1_var`/`d2_var` into per-player contributions.
- The point-estimate vs MC-averaged comparison (`p_winner_hier` vs `p_winner_hier_mc`, both already written to every `hier_v*_eval_predictions.parquet`) is a free, no-refit diagnostic worth checking first on any future "is the model too confident / not confident enough" question.

## Round 3 (2026-08-28 continuation session) — ticket 21 implemented, gap closed from 0.0315 to 0.0114

Picked up from `RESUME-beat-elo.md`. Departed from that file's planned "full DP scoreline-likelihood" sketch for a cheaper, lower-risk approach — and it worked.

**Attempt 10 — ticket 21, real games-per-set evidence for regime B.** Instead of the fake `adf_update(cw,vw,cl,vl,tier_n,tier_n)` (telling the filter the winner won "the whole match" with 100% certainty — exactly the lie that made attempt #6's up-weighting unsafe), parse `score` into total games won per side (`parse_score_games()`, 99.6% parse coverage) and pass the REAL split as `y`/`n` into the existing `adf_update` Newton solve. Zero new math, same signature, same downstream clamp code. Scanned the new evidence-strength dial (`DSV_REGIME_B_N`, repurposed) from 1.0 down to 0.02 on tuning window — optimum at 0.25, **and corruption came back at a flat 0.0% at every value tested** (vs the OLD mechanism's non-zero baseline), confirming real evidence is inherently safer than the fake all-or-nothing trial. Holdout: log-loss 0.6467→0.6461. Small but real; not enough alone.

**Attempt 11 — re-tune OU_THETA_PER_WEEK jointly with ticket 21.** Rationale (stats-discipline §5.4): a hyperparameter tuned against the OLD regime-B mechanism is a changed axis once that mechanism changes. Scanned 0.005→0.0 — **monotonic improvement all the way to zero reversion**, plateauing at `~0.00001` (numerically indistinguishable from 0.0, corruption still 0.0% even with reversion essentially disabled — the old justification for OU reversion, ticket 10/17's runaway-drift safety net, is apparently no longer load-bearing once regime-B stopped injecting fake overconfident evidence). Holdout: 0.6461→0.6448.

**Attempt 12 — kappa/surface-term ablation (re-test of an old idea against new evidence).** A new finding this round (see below) showed the model's WEAK disagreements with Elo are wrong more than chance while its STRONG disagreements are usually right — hypothesized the surface-specific kappa/lambda terms inject noise that flips close calls. Tested by adding parallel `d1_mean_nosurf`/`d2_mean_nosurf` columns (mu-only, no kappa/lambda) to the eval-row output and comparing. **Rejected — removing the surface term made things WORSE** (log-loss 0.6441→0.6589, accuracy 0.6358→0.6299): kappa/lambda is net-positive signal, not the source of the disagreement pathology.

**Attempt 13 — ticket 24 (scoped down): reinflate variance on clamped updates instead of only protecting it from shrinking.** Bug 8's existing fix (`apply_side_update`'s `shift_survival` scaling) only stops variance shrinking on a clamped/surprising update; it never lets variance GROW to reflect that a big surprise was only partially absorbed — the actual "absorbing boundary" mechanism ticket 24 was scoped to fix, done as a cheap ad-hoc proxy rather than a full Student-t reformulation. New env `DSV_CLIP_REINFLATE_K` (0 = off/unchanged): `new_var *= (1 + K * clip_frac)` where `clip_frac = 1 - shift_survival*mu_survival`. Only applied to `apply_side_update` (regime A) — confirmed a no-op in regime B since real evidence rarely clamps. Scanned K on tuning window: **plain log-loss improved monotonically through K≈2-3** (0.6529→0.6483) while lopsided log-loss got monotonically WORSE — a genuine plain-vs-lopsided tradeoff, not a free lunch. Picked K=2.0 as balanced. Holdout: **0.6448→0.6383 — the biggest single jump of the session.**

**Attempt 14 — joint MAX_SHIFT × CLIP_REINFLATE_K re-tune.** These interact (reinflation only fires on a clamp; MAX_SHIFT controls how often clamps happen) — univariate scans of each had found different optima than the other's default value. Joint grid search converged on **MAX_SHIFT=0.125 (down from 0.2), K=0.5** — better on ALL THREE metrics simultaneously (plain 0.6309, weighted 0.5721, lopsided 0.4288 on tuning window, vs the univariate best of 0.6486/0.5888/0.4431). Holdout (**v25, current best**): **log-loss 0.6266**, accuracy 0.6453 (Elo: 0.6534). **Gap to Elo now 0.0114**, down from 0.0315 at the start of this round.

Follow-up re-scans of RW_VAR_PER_WEEK, SURFACE_VAR_SCALE, TIER_WEIGHT, LAYOFF_C, PHI_MAX, and (newly env-exposed) INIT_VAR around the v25 config: **all six came back completely flat** — the hyperparameter space is exhausted at this config. `TIER_WEIGHT` in particular is now a confirmed no-op (was previously the one lever with a real-but-non-transferring tuning-window effect) — the real-evidence mechanism from ticket 21 has fully absorbed what tier-weighting used to compensate for.

**New finding — weak-disagreement pathology, real but small.** Re-running the "hierarchical vs Elo favourite disagreement" cut from Round 2 on v25: when hierarchical disagrees with Elo (25% of matches) with only weak confidence (51-60%), it's wrong more often than chance (44.4% accuracy, n=1227 — improved from 41.1%/1096 pre-round-3, but the pattern persists); strong disagreements (70%+) are usually right. Tested a targeted post-hoc fix — extra shrinkage-toward-0.5 restricted to a narrow band around p=0.5 (so it wouldn't touch lopsided predictions the way Platt/isotonic did) — on the tuning window: **best case only 0.6214→0.6211, negligible** (stats-discipline §3: the effect is real but too small in aggregate to move the metric that matters). Not pursued further; the pathology's cause is still unidentified.

**Speed infrastructure fixed this round** (see `mc_parallel.py`, `hierarchical_model.py`'s `load_prepared()`, and the `str`→`object` dtype cast in `prepare()`): MC posterior-averaging was serial and dominated wall-clock (up to 470s per holdout check, ~137s of un-logged time inside every tuning-window scan candidate) — parallelized across the machine's 24 cores for an ~11.6x speedup. The prepared archive (CSV load + parse, ~20.5s, unchanged across candidates) is now cached. pandas 3.0's PyArrow-backed `str` dtype was making `itertuples()` 4x slower than necessary in the fit's hot loop (profiled: 26s of a 65s fit) — fixed by casting to plain `object` dtype once in `prepare()`. Combined: a full holdout check went from ~10-12 minutes to under 2 minutes; a tuning-window scan candidate from ~230-470s to ~50-90s. Zero behavior change (verified bit-identical `p_winner_hier` on the fit-loop fix; MC-averaging results agree with the old serial version to within expected MC noise, different RNG stream by design).

## Status as of Round 3

Gap to Elo: **0.0114** (log-loss 0.6266 vs 0.6152), down from 0.0315 at Round 2's close — five new attempts (10-14), three of them real transferring wins (10, 11, 13+14 combined), one rejected (12), one negligible (weak-disagreement shrinkage). Hyperparameter space is now exhausted. A parallel diagnostic sweep (cutting the eval data by market-favorite strength, surface, tour, best-of, layoff length, player experience) was launched to find the next lever — see its findings appended below once complete, or check `tune_scratch/diag_*` for its scratch analysis if this section hasn't been updated yet.

## Round 3, second continuation session (2026-08-28) — Attempt 15 (layoff bump, rejected) + five-script diagnostic sweep

Picked up per `RESUME-beat-elo.md`'s Part 1/Part 2 split. v25 config unchanged (`DSV_RW_VAR_PER_WEEK=0.005 DSV_SURFACE_VAR_SCALE=1.0 DSV_OU_THETA_PER_WEEK=0.00001 DSV_MAX_SHIFT=0.125 DSV_PHI_MAX=4.0 DSV_LAYOFF_C=0.06 DSV_REGIME_B_N=0.25 DSV_TIER_WEIGHT=1.0 DSV_CLIP_REINFLATE_K=0.5`).

### Attempt 15 — layoff-bump lever from `diag2_posthoc_bump.py`, REJECTED at tuning stage (both fit-time and eval-time forms)

The earlier diagnostic (`diag2_posthoc_bump.py`) found that adding a fixed variance bump to `d1_var`/`d2_var` at MC-averaging time, scoped to matches where either player had a 1-12wk layoff, improved log-loss on an exploratory 10,678-row Elo-joined set (0.6189→0.6179 at bump=0.05). This had never been through the charter or corruption gate. Two implementations were tested properly this session, both tuning ONLY on 2022-23:

**15a — fit-time, bounded.** `hierarchical_model.py` already had an unused `DSV_SHORT_LAYOFF_BUMP`/`DSV_SHORT_LAYOFF_MIN_WEEKS` mechanism (added by an earlier diagnostic pass, default 0.0/no-op) with a code comment recording that an UNBOUNDED version (any gap ≥1wk, forever, no upper cutoff) was already tried and rejected — it made the tuning metric monotonically worse (0.6194→0.6245) because `weeks≥1` fires on 48.9% of ALL player-appearances in the archive (confirmed again this session by `diag2_saturation.py`, see below), so an unbounded bump double-counts variance for ordinary tour scheduling gaps and even multi-year layoffs already handled by the separate `LAYOFF_C`/`phi` mechanism, saturating active players' variance toward `MAX_VAR` well before 100-200 matches. The diagnostic's actual finding was scoped to 1-12 weeks specifically, so a new `DSV_SHORT_LAYOFF_MAX_WEEKS` upper-bound env var was added (default `1e9`, i.e. old behaviour unchanged unless explicitly set) and the condition changed from `weeks >= MIN` to `MIN <= weeks <= MAX`. Passed `stress_test_safety.py`. Scanned `DSV_SHORT_LAYOFF_BUMP ∈ {0, 0.01, 0.02, 0.05, 0.08, 0.12}` with `DSV_SHORT_LAYOFF_MAX_WEEKS=12.0` on top of full v25 config, 2022-23 tuning window (`tune_scratch/tune_short_layoff_bump.py`):

| BUMP | weighted | plain | lopsided | corruption |
|---|---|---|---|---|
| 0.0 (baseline) | **0.5721** | 0.6309 | **0.4288** | 0.0% |
| 0.01 | 0.5770 | 0.6313 | 0.4447 | 0.0% |
| 0.02 | 0.5814 | 0.6330 | 0.4556 | 0.0% |
| 0.05 | 0.5894 | 0.6369 | 0.4737 | 0.0% |
| 0.08 | 0.5946 | 0.6397 | 0.4847 | 0.0% |
| 0.12 | 0.6000 | 0.6427 | 0.4962 | 0.0% |

Corruption stays flat at 0% (unlike the unbounded version) — the bound does fix the saturation/corruption risk — but ALL THREE metrics get monotonically WORSE from bump=0. Properly scoping the bump to 1-12wk did not rescue it; the fit-time mechanism itself is wrong, not just its scope.

**15b — eval-time-only (the diagnostic's original mechanism).** Refit v25 cleanly on 2022-23 (`tune_scratch/tune_posthoc_bump_2223.py`), joined layoff/tier/market-odds features the same way `diag2_build.py` does, and re-ran the exact post-hoc bump mechanism (add to `d1_var`/`d2_var` at `mc_average` time only, no refit) on the tuning window instead of the exploratory eval set:

| bump | plain | weighted | lopsided |
|---|---|---|---|
| 0.00 | 0.6226 | **0.5643** | **0.4222** |
| 0.01 | 0.6224 | 0.5644 | 0.4233 |
| 0.02 | 0.6222 | 0.5646 | 0.4244 |
| 0.05 | 0.6218 | 0.5653 | 0.4277 |
| 0.08 | **0.6217** | 0.5662 | 0.4310 |
| 0.12 | 0.6219 | 0.5676 | 0.4353 |
| 0.20 | 0.6228 | 0.5707 | 0.4437 |

This reproduces the original diagnostic's finding exactly: plain log-loss does bottom out slightly lower at bump≈0.08 (0.6226→0.6217, a ~0.0009 improvement — the same trivial magnitude, ~9% of the gap, flagged in the original diagnostic). But `weighted_logloss` — the actual metric v25's own hyperparameters were selected on (ticket 19's lopsided-weighted charter) — gets monotonically WORSE from bump=0, and lopsided log-loss degrades hard (0.4222→0.4437 at bump=0.2). This is the identical shape to two already-rejected fixes this session: Platt scaling and isotonic recalibration (Hypothesis 5) both improved plain log-loss while trading away lopsided confidence, and were rejected on exactly that basis. Also identical in shape to the raw (non-jointly-tuned) `CLIP_REINFLATE_K` scan in attempt 13, before `MAX_SHIFT` was co-tuned to fix the tradeoff.

**Verdict: REJECTED at the tuning stage for both implementations, no holdout reveal needed** (per charter, reveal is for a candidate you actually believe in — bump=0 already wins on the metric v25 was selected on, in both forms, monotonically). `DSV_SHORT_LAYOFF_MAX_WEEKS` is left wired into `hierarchical_model.py` (default `1e9`, no-op) for the same reason `DSV_TIER_WEIGHT` was left parked after its own non-transferring result — a future attempt may want to condition the short-layoff signal on something more specific than "any 1-12wk gap, flat bump," since the diagnostic's underlying calibration finding (see below) is real, just not fixable this way.

### Part 2 — five previously-written, never-run diagnostic scripts, all executed this session

All five are read-only (no `mc_average` calls — they reuse the pre-computed `p_winner_hier_mc600` column already baked into `hier_v25_eval_predictions.parquet`/`diag2_enriched.parquet`), run one at a time in the foreground. Full output in `tune_scratch/diag2_*.log`.

**`diag2_calib_breakdown.py` — confirms the 1-12wk overconfidence finding, at finer resolution, still real but small.** Layoff-bucket calibration on the converged mc600 column: `<=1wk` error +0.022 (underconfident), `1-4wk` −0.044, `4-12wk` −0.059 (both overconfident), `>12wk` +0.045 (n=194, noisy). Present in both elite and non-elite tourneys (−0.037/−0.077 split), not concentrated in either. This is the same finding Attempt 15 tried and failed to fix — confirmed real, still not fixable by a flat variance bump. A side cut in this script (splitting by whether the model's *favourite* specifically was the layoff-affected side) produces an extreme-looking 3%/99.5% split — traced by hand and confirmed to be a real but circular selection artefact (conditioning a "realised" rate on `fav_is_winner`, which is literally the accuracy indicator, via which side's layoff-flag gets selected), not a new finding — noted here so it isn't mistaken for one on a future pass.

**`diag2_fresh_cuts.py` — two real, convergent findings.** (B) Wall-proximity proxy (`max(|d1_mean_nosurf|, |d2_mean_nosurf|)` as a stand-in for "is any component near the `MAX_STATE=3.0` clamp"): **zero of the 10,678 holdout rows have wall_proxy > 4.0** — no eval-set match is anywhere near a wall. This directly undercuts ticket 24's premise (heavy-tailed process noise to avoid clamp-driven absorbing-boundary pathology) *as a lever for closing the current holdout gap specifically* — the mechanism it targets doesn't fire on any holdout row, whatever value it might have for corruption-robustness elsewhere in the 50-year archive. The hier-vs-Elo gap is also concentrated in the LEAST-separated quintile (Q1: gap +0.0179) and shrinks toward the most-separated quintile (Q5: gap +0.0011) — pointing away from "clamping causes overconfidence on extreme matchups" and toward "insufficient extremization on already-clear matchups," a different failure mode than ticket 24 addresses.
(C) Market-bucketed calibration (bucketing by the MARKET's own confidence, not the model's — an external, non-circular reference): Elo and Hier are both mildly overconfident together in the low-confidence (50-70%) buckets, but **Hier is substantially UNDERconfident specifically where the market is most confident** — `(0.8,0.9]` bucket: Hier predicted=0.734 vs realised=0.834 (error +0.099); `(0.9,1.0]` bucket: predicted=0.817 vs realised=0.932 (error +0.115). Elo's error in the same two buckets is only +0.027/+0.036; Market itself is ~perfectly calibrated (±0.004). This is the cleanest single restatement of the already-known lopsided-confidence problem (v25 lopsided log-loss 0.4288 vs Elo 0.3853 vs Market 0.3645) — same substance, sharper lens (external truth variable instead of the model's own probability axis, which avoids the circularity risk of self-referential calibration cuts).

**`diag2_saturation.py` — explains WHY the old unbounded `SHORT_LAYOFF_BUMP` failed (mechanism check, not a new lever).** Confirms `weeks>=1` fires on 48.9% of all player-appearances archive-wide (not a rare event), and back-of-envelope math shows even a modest per-event bump (0.02-0.04) saturates an active player's variance to `MAX_VAR=4.0` by match #100-200. This is the finding that motivated adding the `MAX_WEEKS` bound in Attempt 15 above — bounding did fix the saturation/corruption risk (corruption stayed 0% at every bump value with the bound in place) but did NOT fix the underlying tuning-window regression, so saturation was A cause of the original mechanism's failure, not the ONLY one.

**`diag2_wta_churn.py` — the most actionable new finding of this pass.** Splitting the holdout by tour: **ATP is essentially at parity with Elo** (hier_ll=0.6133 vs elo_ll=0.6150, gap=**−0.0018** — hier is marginally BETTER), while **WTA carries the entire gap** (hier_ll=0.6242 vs elo_ll=0.6048, gap=**+0.0194**). The original churn/field-depth hypothesis this script was written to test (top-16 win-share, distinct-winner counts by tour/year, 2022-25) shows no obvious structural ATP/WTA difference to explain it — that specific theory is unsupported by its own diagnostic. But the raw tour split itself is a strong, clean signal: **the entire remaining gap-to-Elo is concentrated in WTA**, not spread evenly across both tours as every prior round-3 finding implicitly assumed (the un-split holdout numbers in RESUME-beat-elo.md are a pooled ATP+WTA average). A secondary cut (established vs newer winners, `n_prior_w<50`) found hier dramatically OUTPERFORMS Elo on new/surprise ATP winners (n=99, hier_ll=0.78 vs Elo 1.12, gap=−0.34) — small n, but the direction is striking and suggests hier's richer state genuinely helps exactly where Elo has the least information, when it isn't being dragged down by the WTA-specific issue.

**`diag2_bo5_sets.py` — converges with the market-bucket finding.** Splitting by how many sets a match actually took: hier trails Elo specifically on **straight-set (most decisive) wins** — Bo5 3-set matches: gap=+0.054 (n=425); Bo3 2-set matches: gap=+0.0149 (n=6285) — while matches that go the distance are at parity or FAVOUR hier: Bo5 5-setters gap=**−0.0375** (n=224), Bo3 3-setters gap=**−0.0075** (n=3400). This is the same shape as the market-confidence-bucket finding above (straight-set blowouts are disproportionately the market's high-confidence bucket) — independent corroboration via a completely different cut of the same data (match length instead of market odds), which is what makes it worth trusting rather than a coincidence: hier undershoots confidence specifically on the most clear-cut, lopsided outcomes, and is fine (or better than Elo) on genuinely competitive ones.

### Recommended next step (not attempted this session — scoped for a future pick-up)

Three of the five diagnostics converge on the same shape from three independent angles (market-confidence buckets, set-length, and — new this pass — tour split): **the gap is not a general architecture defect, it is concentrated in WTA, in the market's/reality's most lopsided matches specifically**, where hier systematically undershoots the confidence Elo and the market both reach. This is NOT what either ticket 21 (richer per-match evidence — already shipped, and the straight-set finding argues AGAINST "not enough evidence": a blowout already carries the most decisive games-won margin available, so more granular evidence is unlikely to help) or ticket 24 (heavy-tailed process noise for clamp-related overconfidence — directly undercut by the wall-proximity finding of zero near-wall holdout rows) were built to fix. Recommend, in priority order: (1) re-examine whether `SURFACE_VAR_SCALE`/`RW_VAR_PER_WEEK`/other hyperparameters — all tuned on POOLED ATP+WTA 2022-23 data — are systematically wrong for WTA specifically (a per-tour hyperparameter split, not yet tried); (2) investigate why hier's `d1_mean`/`d2_mean` separation doesn't extremize as far as Elo's for genuinely dominant true mismatches, independent of tour, which the market-bucket and set-length cuts both point to as the proximate mechanism. Both are real investigative efforts, not one-line tuning knobs — scoping them further needs a fresh session with room to explore WTA-specific state traces (`DSV_DEBUG_UID` on WTA players) the way ticket 18/20 did for Djokovic/Prizmic/Sinner on the ATP side.

Hyperparameter space (pooled) remains exhausted; gap to Elo remains **0.0114** (0.6266 vs 0.6152, v25, unchanged this session — no candidate transferred).

## Round 3, third continuation session (2026-08-28, parallel investigation-only agent) — mechanism identified: MC-averaging convexity cost, NOT insufficient mean extremization, and it is NOT WTA-specific

Ran in parallel with another agent authorized to edit `hierarchical_model.py` for a WTA-specific hyperparameter split; this session was **investigation-only** (no edits to `hierarchical_model.py`, no `mc_average`/`Pool` calls — reused precomputed `p_winner_hier_mc600`/`p_winner_hier` from `hier_v25_eval_predictions.parquet` via `tune_scratch/diag2_enriched.parquet`, plus one small single-threaded proof-of-concept re-sample). Task: check whether the market-bucket/set-length/tour-split convergent finding (recommended next step above) holds **within ATP alone**, and trace the actual mechanism. New scripts, all read-only: `tune_scratch/diag3_atp_extremization.py`, `tune_scratch/diag3_mc_var_scale_poc.py` (full output in `tune_scratch/diag3_*.log`).

### Finding 1 — the ATP-only extremization undershoot is REAL, and of comparable raw magnitude to WTA's

Market-confidence-bucket calibration (external truth variable, same method as `diag2_fresh_cuts.py`, this time split by tour first):

| Bucket | ATP Hier error | ATP Elo error | WTA Hier error | WTA Elo error |
|---|---|---|---|---|
| (0.5,0.6] | −0.0668 | −0.0956 | −0.0908 | −0.0791 |
| (0.6,0.7] | −0.0327 | −0.0399 | −0.0526 | −0.0385 |
| (0.7,0.8] | −0.0191 | −0.0220 | +0.0103 | +0.0060 |
| (0.8,0.9] | **+0.0934** | +0.0135 | **+0.1053** | +0.0405 |
| (0.9,1.0] | **+0.1198** | +0.0342 | **+0.1090** | +0.0376 |

(error = realised − predicted; positive = underconfident/undershooting.) **ATP's top-bucket calibration error (+0.1198) is not smaller than WTA's (+0.1090) — if anything marginally larger.** The undershoot-on-lopsided-matches pattern is not a WTA artifact; it is present at essentially equal or greater raw magnitude within ATP alone.

**Why ATP's AGGREGATE log-loss gap to Elo still looks fine (≈parity) despite this:** ATP has a real, separate, compensating advantage over Elo specifically in the LEAST-lopsided bucket that WTA does not share — log-loss gap (hier−elo) by bucket, weighted by n:

| Bucket | ATP ll_gap | ATP n | WTA ll_gap | WTA n |
|---|---|---|---|---|
| (0.5,0.6] | **−0.0346** | 1343 | +0.0006 | 1282 |
| (0.6,0.7] | +0.0003 | 1265 | +0.0082 | 1197 |
| (0.7,0.8] | +0.0124 | 886 | +0.0307 | 934 |
| (0.8,0.9] | +0.0052 | 565 | +0.0345 | 559 |
| (0.9,1.0] | +0.0553 | 285 | +0.0780 | 260 |
| **n-weighted total** | **≈ −0.0038** | | **≈ +0.0186** | |

ATP's −0.0346 edge in the 50-60% bucket (hier genuinely beats Elo on ATP's closest matches — consistent with the `diag2_wta_churn.py` finding that hier crushes Elo on ATP's new/surprise winners, gap −0.34, n=99) almost exactly cancels ATP's own version of the same top-bucket undershoot defect that WTA has. **This means the "gap is concentrated in WTA" framing from the previous session is only true of the AGGREGATE number — the underlying defect is general-architecture, present in both tours; what's WTA-specific is the ABSENCE of a compensating strength elsewhere, not the presence of an extra weakness at the top.** This has a direct implication for the other agent's WTA-hyperparameter-split investigation: a WTA-only hyperparameter retune might close some of the aggregate gap by making WTA's competitive-match performance better (closer to ATP's), but should NOT be expected to fix the top-bucket undershoot itself, since that mechanism is tour-independent (see Finding 2/3).

### Finding 2 — the mean state (`d1_mean`/`d2_mean`) extremizes CORRECTLY; the undershoot is introduced entirely downstream, by MC-averaging

This directly answers the "shrinkage/prior pull" candidate mechanism from the task brief: **rejected**. For ATP rows with `p_elo>0.9` (n=351): mean point-estimate prediction (`p_winner_hier`, no MC) = **0.9409**, essentially identical to Elo's own mean (0.9400). WTA same slice (n=243): point estimate = **0.9606** vs Elo's 0.9331 — WTA's point estimate is if anything MORE extreme than Elo, not less. Individual traces confirm this (`diag3_atp_extremization.py` Part D): Nadal d. Leo Borg, Alcaraz d. Oliver Tarvet, Alcaraz d. Hady Habib — point estimate is 1.000 or 0.999 in every case, matching or exceeding Elo's own 0.99+ confidence. **`d1_mean`/`d2_mean` separation is not the defect** — `OU_THETA_PER_WEEK`'s convergence to ~0 (round 3) already removed the one plausible mean-shrinkage mechanism, and there is no other prior-pull term acting on the mean at eval time. (Note: `d1_mean`/`d2_mean` are point-level per-serve logit gaps, not directly comparable in magnitude to Elo's match-level composite logit gap — `p_match`'s Bo3/Bo5 recursion is the thing that amplifies a modest point-level edge into near-certainty, and it does so correctly at the point-estimate level. A raw `|hier_combined_gap|/|elo_logit_gap|` ratio of ~0.14 in this bucket, which looks alarming in isolation, is a scale artifact of comparing a point-level logit to a match-level logit, not evidence of undershoot — the point-ESTIMATE match probability is the correct comparator, and it already matches Elo.)

**The entire undershoot is created by the MC-posterior-averaging step** (`p_winner_hier` → `p_winner_hier_mc600`), and this is where the Jensen's-gap / MC-convexity candidate mechanism from the task brief is **confirmed**, cleanly, in both tours:

| Bucket (by p_elo) | ATP mc_cost (pt−mc600) | ATP combined_var | WTA mc_cost | WTA combined_var |
|---|---|---|---|---|
| (0.5,0.6] | 0.0277 | 0.5107 | 0.0401 | 0.5095 |
| (0.6,0.7] | 0.0656 | 0.5213 | 0.0753 | 0.4998 |
| (0.7,0.8] | 0.0974 | 0.5140 | 0.1189 | 0.5037 |
| (0.8,0.9] | 0.1197 | 0.4911 | 0.1436 | 0.5164 |
| (0.9,1.0] | **0.1329** | 0.7137 | **0.1342** | 0.6007 |

`mc_cost` (how much probability MC-averaging removes relative to the point estimate) grows monotonically with confidence in **both** tours, and is nearly identical in magnitude at every bucket — ATP 0.1329 vs WTA 0.1342 in the top bucket, despite ATP's combined posterior variance there being ~19% HIGHER (0.7137 vs 0.6007). This is the single cleanest, most tour-independent result of the session: **MC-averaging's convexity cost is the mechanism, it is architectural (present equally in both tours), and it is NOT what a WTA-specific hyperparameter split would touch** (that lever changes fit-time evidence weighting, which feeds `d1_var`/`d2_var`'s magnitude, not the downstream MC-integration step that turns a given variance into a probability haircut).

Real-match traces make the mechanism concrete (`diag3_atp_extremization.py` Part D, ATP p_elo>0.9 top rows): Nadal d. Darwin Blanch (Blanch: `n_prior_l=4`) — point estimate 1.000, MC-averaged 0.785, combined_var=3.853; Alcaraz d. Oliver Tarvet (`n_prior_l=1`) — point estimate 1.000, MC-averaged 0.737, combined_var=3.744. The cost is largest specifically when the underdog is a near-total unknown (their side's variance sits near the `INIT_VAR=1.2` prior, large), and `p_match`'s composed game→set→match recursion is most curved (most convex/concave) exactly where the point estimate is most extreme — so a genuinely correct-in-isolation Bayesian propagation of that unknown-opponent uncertainty produces an outsized haircut on precisely the matches everyone (Elo, market) is most sure about.

### Finding 3 — proof-of-concept: down-scaling MC variance moves the top-bucket prediction toward Elo/market in the expected direction, at a magnitude that would plausibly matter

`tune_scratch/diag3_mc_var_scale_poc.py`: re-sampled MC (single-threaded, no `Pool`, n_samples=200, ~5 min total) with `d1_var`/`d2_var` multiplied by a scale factor before drawing, restricted to the p_elo>0.9 slice only (this is a mechanism check, NOT a charter-compliant tuned candidate — it uses the untouched 2024-25 holdout for characterization only, no hyperparameter is being selected or reported as a "gap to Elo" result):

| scale | ATP mean_p (n=351) | WTA mean_p (n=243) |
|---|---|---|
| 1.00 (current) | 0.8086 | 0.8278 |
| 0.50 | 0.8574 | 0.8793 |
| 0.25 | 0.8927 | 0.9152 |
| 0.10 | 0.9199 | 0.9415 |
| 0.00 (= point estimate) | 0.9409 | 0.9606 |

(Elo means in this slice: ATP 0.9400, WTA 0.9331.) Monotonic in the expected direction in both tours, and a scale around 0.1–0.25 would land close to or slightly past Elo's own confidence in this specific bucket — confirming this is a real, sizeable lever, not a marginal one.

### Recommended next step — NOT yet implemented, needs its own charter-compliant tuning pass (and does not need `hierarchical_model.py` at all)

**Precise, scoped mechanism:** add a variance-scale multiplier applied ONLY at MC-sampling time, to `d1_var`/`d2_var` (and optionally `d2_var`/`d1_var`'s `_nosurf` counterparts if those are ever used for MC), NOT at fit time. This is a new lever, structurally different from every recalibration/shrinkage fix rejected so far in this ticket:

- **Where to implement:** `mc_parallel.py`'s `_mc_chunk`/`mc_average` (lines ~30-36 and ~46), NOT `hierarchical_model.py`. Add a `var_scale: float = 1.0` parameter to `mc_average()`, and inside `_mc_chunk` change the two `rng.normal(...)` calls to draw from `d1_var * var_scale` / `d2_var * var_scale` instead of the raw column values. Default 1.0 = current behaviour, exact no-op. **This means the fix does not touch `hierarchical_model.py` at all and can be implemented and tuned independently of whatever the parallel WTA-hyperparameter-split agent is doing there** — no file-contention risk, and it composes cleanly with any per-tour `d1_var`/`d2_var` changes that agent's work might produce (this lever operates strictly downstream of however `d1_var`/`d2_var` get computed).
- **Why this shape specifically, and why it's a better bet than the already-rejected global recalibrations (Platt/isotonic, Hypothesis 5):** those operated directly and uniformly on the logit/probability scale, so a flat correction necessarily traded confidence away from the wrong slice (competitive matches) to fix the top bucket — exactly the failure both were rejected for. Variance-scaling is different in kind: Jensen's-gap cost is proportional to the LOCAL CURVATURE of `p_match` times the variance, and `p_match`'s curvature is near zero around p=0.5 (nearly linear there) and largest near p→0/1. **A flat `var_scale<1` therefore has a naturally self-limiting, curvature-weighted effect** — small on competitive matches (where MC-averaging is currently net-positive per Hypothesis 3, and should stay that way), large specifically on already-extreme matches (where Finding 2/3 show it's currently too large). This is a structural argument for why this lever might avoid the Platt/isotonic failure mode, not a guarantee — it still needs the full charter check below before being trusted.
- **Required validation before this is a real candidate (none of this has been done yet):** (1) scan `var_scale` on the 2022-23 tuning window ONLY, through the SAME weighted/lopsided-weighted metric v25 was selected on (per charter) — not just plain log-loss, which is what burned the layoff-bump attempt (Attempt 15) and Platt/isotonic earlier in this ticket; (2) confirm it does NOT erode the 50-70% bucket's currently-good calibration (this is the main risk — verify the mid-bucket `mc_cost` numbers above, which are small but non-zero, don't compound badly at low `var_scale`); (3) no corruption-gate check is needed for this lever specifically, since it only affects eval-time MC sampling, never fit-time state — genuinely lower-risk than most levers tried this session; (4) reveal on the untouched 2024-25 holdout exactly once, for a candidate the tuner actually believes in after (1)-(2).
- **Secondary, smaller finding worth preserving for whoever tunes this:** the proof-of-concept only tested a single global scale on the p_elo>0.9 slice. A conditional shape (e.g. scale as a function of the point estimate's own extremity, or of which side's variance is driving `combined_var`) was not tested and might do even better, but adds complexity/tuning-surface — try the flat scalar first per the "cheapest thing that could work" pattern this session has otherwise followed (`CLIP_REINFLATE_K`, `TIER_WEIGHT`, etc. were all flat scalars first).

### Session status

ATP-only extremization undershoot: **confirmed real**, comparable magnitude to WTA's. Mechanism: **MC-averaging convexity cost (Jensen's gap), not insufficient mean extremization** — confirmed via point-estimate-vs-Elo parity (Finding 2), tour-symmetric `mc_cost` (Finding 2 table), and a working proof-of-concept showing the expected monotonic response to a variance-scale lever (Finding 3). This is a genuinely new, previously-untested lever (`DSV_MC_VAR_SCALE`-equivalent in `mc_parallel.py`, not `hierarchical_model.py`) — specified precisely above but **not implemented or tuned**, per this session's investigation-only scope. Gap to Elo unchanged at **0.0114** (no candidate was run through the charter this session).

## Round 4 (2026-08-28, ran in parallel with the third-continuation agent above) — real-holdout tour split confirmed, three new tour-conditioned/curvature levers tried, all REJECTED, gap still 0.0114

Picked up per `RESUME-beat-elo.md`'s WTA-investigation brief. This agent had exclusive access to `hierarchical_model.py` (the parallel agent above was investigation-only there by design, per its own write-up); `mc_parallel.py` was touched by both — see attempt 19 below for how that overlap resolved.

### Step 1 — confirmed the WTA split on the REAL 8,594-row odds-joined holdout, not just the 10,678-row exploratory eval set

`diag2_wta_churn.py`'s finding used `diag2_enriched.parquet` (10,678 rows, no odds-availability filter). Re-ran the exact join pattern from this file's "4. Join to Elo + market odds and score" section, split by `tour` before scoring (script: adapted inline, output saved as `tune_scratch/real_holdout_v25_joined.parquet`):

| | n | Elo | Hier (p_winner_hier_mc, N=60 — matches the documented 0.6266 exactly) | gap |
|---|---|---|---|---|
| ATP | 4,358 | 0.6204 | 0.6212 | **+0.0008** |
| WTA | 4,236 | 0.6099 | 0.6322 | **+0.0223** |
| Overall | 8,594 | 0.6152 | 0.6266 | +0.0114 |

(Using the converged `p_winner_hier_mc600` column instead: ATP gap **−0.0040**, WTA gap **+0.0185**, overall +0.0071 — either MC sample count gives the same qualitative picture.) **Confirmed: the finding holds on the real join, not just the exploratory set.** ATP is at parity or a slight hier edge; the entire aggregate gap is concentrated in WTA.

### Attempt 16 — tour-conditioned WTA lower-tier (qual/ITF) evidence discount, REJECTED (clean, monotonic)

Hypothesis: the locked project decision (`dropshotverse-tennis-effort` memory) treats WTA qual/ITF as a "match-outcome cold-start prior only," but this was never implemented — `TIER_WEIGHT` (ticket 22) discounts by `tourney_level` only, symmetrically across tours, so it discounts ATP Challenger exactly as much as WTA ITF. Checked the actual data asymmetry first: WTA's "lower" tier (qual/ITF/$25k-$100k circuit) is **~3.8x** WTA's main-tour volume (605k vs 158k rows, all years) vs ATP's **~1.2x** (232k vs 194k) — and in 2022-25, only **27.9%** of WTA lower-tier rows have point-level data vs 93-99.8% for every other tour/level combination. This is a real, measured asymmetry, not a guess.

Implemented `DSV_WTA_LOWER_WEIGHT` in `hierarchical_model.py` (new hyperparameter, default `1.0` = exact no-op, scoped to regime B only per `TIER_WEIGHT` precedent): `tier_scale *= WTA_LOWER_WEIGHT` when `row.tour=="wta" and row.level_file=="lower"`. Passed `stress_test_safety.py`. Scanned on 2022-23 tuning window (`tune_scratch/tune_wta_lower_weight.py`), scored pooled/ATP/WTA separately:

| WTA_LOWER_WEIGHT | ATP weighted | WTA weighted | WTA lopsided |
|---|---|---|---|
| 1.0 (baseline) | 0.5731 | **0.5711** | **0.4299** |
| 0.5 | 0.5731 | 0.5712 | 0.4302 |
| 0.25 | 0.5731 | 0.5719 | 0.4313 |
| 0.1 | 0.5731 | 0.5729 | 0.4331 |
| 0.05 | 0.5731 | 0.5736 | 0.4344 |
| 0.02 | 0.5731 | 0.5740 | 0.4355 |
| 0.0 | 0.5731 | 0.5741 | 0.4366 |

ATP is correctly unaffected at every value (confirms the lever is properly isolated to WTA). WTA's own metrics get monotonically **worse**, not better, as the weight decreases. **Verdict: REJECTED, cleanly** — WTA's lower-tier evidence is a net-positive signal, not the noise source the volume asymmetry suggested. Left wired at default `1.0` (no-op), same precedent as `TIER_WEIGHT`/`SHORT_LAYOFF_BUMP`.

### Attempt 17 — tour-specific hyperparameter re-scan (does the pooled optimum diverge by tour?), REJECTED — no divergence found

Re-scanned `DSV_REGIME_B_N`, `DSV_MAX_SHIFT`, `DSV_RW_VAR_PER_WEEK` on 2022-23, each held at v25 defaults otherwise, scored pooled/ATP/WTA (`tune_scratch/tune_by_tour_generic.py`, reusable for any single env var).

- **`REGIME_B_N`** (grid 0.05-1.0): flat for both tours, optimum ~0.15-0.25 for both — matches pooled v25 default (0.25) closely. No divergence.
- **`MAX_SHIFT`** (grid 0.05-0.3): both tours move in lockstep, optimum ~0.1-0.125 for both. No divergence. (Minor aside, not tour-related: 0.1 edges out the current default 0.125 by ~0.001 on weighted logloss for both tours when `CLIP_REINFLATE_K` is held at its jointly-tuned 0.5 — small, not chased, doesn't touch the WTA question.)
- **`RW_VAR_PER_WEEK`** (grid 0.0001-0.02): **both tours improve monotonically as this decreases all the way to the low end of the grid** (pooled weighted 0.5671→0.5401 going from 0.005→0.0001, ATP 0.5731→0.5308, WTA 0.5711→0.5500) — a real, substantial, but **suspicious monotonic-to-the-boundary result** (stats-discipline red flag: previous "flat/exhausted" re-scans in round 3 evidently used too narrow a band around 0.005). **Not chased to holdout this round** — off the WTA-specific brief, and turning the random-walk variance-growth rate toward zero effectively disables the layoff/time-decay uncertainty-inflation mechanism the whole session's stress-test suite was built to protect, which needs its own dedicated corruption-gate/stress-test scrutiny before trusting, not a rushed reveal. **Flagged here as a real, unexplored lead for a future round** — note directionally it does NOT favour WTA specifically: the ATP/WTA gap in absolute hier performance WIDENS as `RW_VAR_PER_WEEK` drops (WTA ends up further behind ATP's own improvement, not closer), so on its own it looks like a pooled architecture lever, not a WTA fix.

**Verdict on the brief's core question ("are pooled hyperparameters systematically wrong for WTA"): REJECTED for all three tested.** Every one of REGIME_B_N/MAX_SHIFT/RW_VAR_PER_WEEK has its ATP-optimum and WTA-optimum track together, not diverge — the pooled value is not a "compromise between two different tour optima," it's genuinely close to correct for both.

### Attempt 17b — best-of-format-controlled tour comparison: WTA still trails ATP by 0.03 logloss even holding format constant

WTA plays 100% best-of-3 (no Bo5 on tour). Checked whether ATP's aggregate near-parity is propped up by its Bo5 subset (Slams) — i.e. whether the "WTA gap" is really a "no-Bo5-to-offset-with" artifact:

| | n | Elo | Hier | gap |
|---|---|---|---|---|
| ATP best_of=3 only | 3,512 | 0.6398 | 0.6282 | **−0.0116** |
| ATP best_of=5 only | 847 | 0.5394 | 0.5674 | **+0.0280** |
| WTA (100% best_of=3) | 4,236 | 0.6099 | 0.6285 | **+0.0185** |

**This is backwards from the hypothesis**: ATP's Bo5 subset is a net NEGATIVE for hier (+0.0280, worse than ATP's aggregate), and ATP's advantage over Elo is concentrated specifically in its Bo3 matches (−0.0116, even better than ATP's blended −0.0040/+0.0008). Comparing WTA directly against ATP's own Bo3-only number (the correct apples-to-apples cut, since WTA has no other format): **WTA trails ATP by a full 0.0301 log-loss on the identical match format.** Rules out best-of format mix as an explanation. (Note this ATP Bo5 number looks worse than `diag2_bo5_sets.py`'s finding — that script split further by how many sets were actually played and found 5-setters favour hier (−0.0375) while Bo5 straight-3-setters don't (+0.054); this cut mixes both sub-patterns into one aggregate `best_of==5` number, which is why it nets positive despite the finer cut showing a mixed picture.)

### Attempt 18 — tour-specific (WTA-only) isotonic recalibration, REJECTED — same trade-off as the pooled attempts, even fit separately

The two pooled recalibration attempts (Hypothesis 5 continued, linear Platt and isotonic) were rejected earlier this session for the same reason: they improve overall log-loss but trade away lopsided-bucket confidence, because the true miscalibration is entangled with market-lopsidedness, a variable neither corrector conditions on. New angle tried here: fit isotonic **separately on WTA-only** 2022-23 data (boundary-safe, favourite-probability space, `y_min=0.5`) instead of pooled — the idea being a WTA-specific curve only has to match WTA's own (more extreme) hump, not compromise with ATP's. ATP holdout rows left completely untouched (identity transform) — a strictly more targeted intervention than either prior pooled attempt.

In-sample (2022-23, WTA-only): raw hier logloss 0.6251 → calibrated 0.6190 (real improvement), but lopsided 0.4240 → 0.4307 (worse) — same shape as pooled, immediately. Revealed once on holdout anyway to get real numbers:

| | Elo | Hier raw | Hier calibrated |
|---|---|---|---|
| WTA overall | 0.6099 | 0.6285 (gap +0.0185) | 0.6279 (gap +0.0179) |
| WTA lopsided | — | 0.4176 | 0.4375 |

Overall WTA gap improves by only 0.0006 (≈3% of the gap) while WTA lopsided log-loss gets substantially worse (0.4176→0.4375). **Verdict: REJECTED**, identical failure mode to the pooled attempts, now confirmed to hold even when the corrector never has to compromise with ATP's curve — the entanglement between "hier's own overconfidence" and "market-lopsidedness" is a property of WTA's own distribution alone, not an artifact of pooling two tours' curves together.

### Attempt 19 — the parallel agent's `var_scale` MC-curvature lever, implemented and tuned through the charter, REJECTED on holdout

The round-3-third-continuation agent (running concurrently with this one, files `tune_scratch/diag3_*.py`) found a genuinely different, tour-**symmetric** mechanism: the entire lopsided-bucket undershoot is created downstream of the mean state, by MC-averaging's Jensen's-gap cost (proportional to `p_match`'s local curvature × variance, largest exactly where predictions are already most extreme) — present equally in ATP and WTA, with WTA's aggregate deficit explained instead by ATP having a compensating edge in its own 50-60%-confidence bucket that WTA lacks. They scoped but did not implement a fix: a `var_scale` multiplier on `d1_var`/`d2_var` inside `mc_parallel.py`'s MC sampling only (explicitly NOT `hierarchical_model.py`, to avoid file contention with this agent's `hierarchical_model.py` work).

Implemented exactly as specified: `mc_average()` gained a `var_scale: float = 1.0` parameter (default = exact no-op, verified — `WTA_LOWER_WEIGHT=1.0`/`var_scale=1.0` baseline runs reproduce the documented v25 tuning-window numbers to 4 decimal places). **Windows gotcha hit and fixed**: the first run of the holdout-reveal script crashed with `multiprocessing`'s "attempt to start a new process before the current process has finished bootstrapping" (missing `if __name__=="__main__":` guard around code that calls `mc_average`, which uses `mp.Pool` — Windows' spawn-based multiprocessing re-imports the entry script in each worker, so top-level Pool-triggering code recurses). This produced a **runaway process-spawn loop** — `tasklist` showed the python process count climbing (5→11→13) across a few seconds before it was caught and killed (`taskkill /T /F` on the root PID) — exactly the memory-crash risk this file's background-task gotcha warns about, though here it was caused by a missing multiprocessing guard rather than backgrounding per se. Fixed by wrapping the reveal script's body in `if __name__=="__main__":` (the existing `tune_*.py` scripts already had this guard, which is why none of their `mc_average` calls this session hit it) and rerun cleanly.

Tuned `var_scale` on 2022-23 (one fit, `var_scale` only affects the downstream MC step so every grid value reuses the same `d1_mean`/`d1_var`/`d2_mean`/`d2_var` columns — `tune_scratch/tune_mc_var_scale.py`), scored on the SAME weighted/plain/lopsided metric v25 was selected on, plus a 50-70%-midband check per the parallel agent's own flagged risk:

| var_scale | pooled weighted | pooled plain | pooled lopsided | pooled midband(50-70) |
|---|---|---|---|---|
| 1.0 (baseline) | 0.5671 | 0.6258 | 0.4238 | 0.7007 |
| 0.9 | 0.5662 | 0.6263 | 0.4198 | 0.7028 |
| 0.8 | 0.5656 | 0.6271 | 0.4158 | 0.7054 |
| **0.7** | **0.5654** | 0.6285 | 0.4119 | 0.7086 |
| 0.6 | 0.5658 | 0.6305 | 0.4083 | 0.7127 |
| 0.5 | 0.5671 | 0.6336 | 0.4051 | 0.7180 |
| 0.25 | 0.5789 | 0.6508 | 0.4038 | 0.7416 |
| 0.1 | 0.6022 | 0.6782 | 0.4169 | 0.7736 |

A genuine, non-boundary optimum at `var_scale≈0.7` (flat 0.65-0.75), improving the actual selection metric (0.5671→0.5654) with the midband cost small relative to the lopsided gain — unlike every previously-rejected calibration lever, this one genuinely improved the tuning-window selection metric, not just plain log-loss. Believed the candidate; revealed once on the real 2024-25 holdout (`tune_scratch/real_holdout_v25_varscale07.parquet`):

| | Elo | Hier v25 (var_scale=1) | Hier v25 + var_scale=0.7 |
|---|---|---|---|
| ATP | 0.6203 | 0.6164 (gap **−0.0039**) | 0.6233 (gap **+0.0031**) |
| WTA | 0.6099 | 0.6285 (gap **+0.0185**) | 0.6374 (gap **+0.0275**) |
| Overall | 0.6152 | 0.6223 (gap **+0.0071**) | 0.6303 (gap **+0.0151**) |
| Lopsided (overall) | — | 0.4202 | 0.4123 |
| Midband 50-70 (overall) | — | 0.6896 | 0.7009 |

**Verdict: REJECTED.** The overall gap to Elo more than DOUBLED (+0.0071→+0.0151), ATP flipped from beating Elo to losing to it, and WTA's gap got substantially worse too. The tuning-window "weighted" metric improvement (0.5671→0.5654, genuine and non-boundary) **did not transfer to the holdout** — a `stats-discipline`-flagged case worth remembering explicitly: a small, real, non-degenerate tuning-window win on the SAME metric the model was originally selected on can still fail to generalize, joining `TIER_WEIGHT` (ticket 22) as the second confirmed case of exactly this failure mode in this ticket. `var_scale` is left wired into `mc_parallel.py` at default `1.0` (no-op) — the mechanism diagnosis (MC-convexity, tour-symmetric) from the parallel agent's investigation is very likely still correct as a description of WHERE the gap comes from, but a flat scalar correction is not a working fix for it; the parallel agent's own "secondary finding" (a conditional/curvature-shaped scale, not a flat one, untested) remains the open lead here, not yet tried.

### Session status (round 4)

Three genuinely new, tour-conditioned or curvature-motivated levers were tried this round (WTA-lower-tier evidence discount, tour-specific isotonic recalibration, MC-variance curvature scaling) plus a tour-specific hyperparameter re-scan across three parameters and a format-controlled (Bo3-only) tour comparison — **all rejected**, none transferred to the holdout. Gap to Elo unchanged at **0.0114** (v25, `p_winner_hier_mc` N=60 basis) / **0.0071** (converged `mc600` basis). The real-holdout tour split is now confirmed on the actual 8,594-row odds-joined set (not just the exploratory 10,678-row set): ATP is at parity or a slight hier edge, WTA carries the entire gap. Combined with the parallel agent's finding that the underlying undershoot mechanism (MC-convexity) is tour-symmetric and WTA's problem is really the ABSENCE of ATP's compensating mid-confidence-bucket edge rather than an extra WTA-specific defect — and that neither a tour-conditioned data-weighting fix, a tour-conditioned hyperparameter retune, a tour-specific calibration fix, nor a flat MC-variance-curvature fix closes any of it — **the honest read is that no lever tried across two full rounds (six independent, well-motivated attempts total between the two agents this round) fixes this.** The `RW_VAR_PER_WEEK`-toward-zero lead (attempt 17) is the one unexplored thread with a real, substantial, non-tour-specific tuning-window signal — but it needs dedicated stress-test/corruption-gate scrutiny (it disables a mechanism the safety suite was explicitly built to protect) before it can be trusted, not a rushed reveal.

## Round 5 (2026-08-28) — Attempt 21: RW_VAR_PER_WEEK toward zero, new stale-state safety gate built, candidate found, REJECTED on holdout — the RW_VAR_PER_WEEK lead is now closed

(Ran in parallel with another agent doing the surface/round/experience diagnostic cuts, appended below as its own "Attempt 20" — the two independently picked the same counter; no file overlap, both left as originally numbered rather than renumbered after the fact.)

Picked up round 4's one flagged lead. Full task: (1) build a new stress-test case for the "stale-but-confident" failure mode (variance not growing enough after a long layoff) since the existing suite only checked upper bounds; (2) re-scan `RW_VAR_PER_WEEK` toward zero on 2022-23 through both the new gate and `corruption_rate()`; (3) reveal once on holdout if a candidate survives.

### Step 1 — new stress-test case added, and it has teeth

`stress_test_safety.py`'s four existing reproductions all check UPPER bounds (variance/state capped) — none checked whether a long layoff grows variance *enough*. Added two new checks (reproduction 5):
- **(a) Variance-sufficiency floors**: start a tour-veteran-tight prior (`v=0.03`, the real Djokovic pre-match trace value from Hypothesis 4 Test A above) and require `inflate()` clears increasing floors as the gap lengthens — 1yr ≥0.3, 3yr ≥2.0, 5yr ≥3.5. The 5yr floor is tied directly to `PHI_MAX`'s own documented design intent ("a real player's uncertainty after any layoff, however long, should saturate rather than grow without bound") — a 5-year gap should land close to `MAX_VAR`, not stay near its pre-layoff value. Floors calibrated with margin under v25's actual values (0.70/3.15/4.00) so they don't block the config already in production.
- **(b) Concrete overconfident-prediction check**: a strong-but-plausible returning player (combined edge 1.5) coming back after a 3yr layoff from a tight prior, facing a fresh/average opponent — MC-averaged win probability must stay ≤0.95.

**Verified the gate actually catches the pathology**: at the default `RW_VAR_PER_WEEK=0.005`, all checks PASS (v_a = 0.6956/3.15/4.00 at 1/3/5yr, matching hand-calculation exactly). At `RW_VAR_PER_WEEK=0.0001` (the low end of round 4's suspicious scan), the three variance-floor checks correctly FAIL (v_a = 0.0433/0.0924/0.1340 — barely inflated at all even after 5 years) — confirming this is a real, working gate, not a rubber stamp.

### Step 2 — re-scanned RW_VAR_PER_WEEK toward zero through both gates

Reused `tune_scratch/tune_by_tour_generic.py` (already runs every candidate through `stress_test_safety.py` as a subprocess plus `corruption_rate()`) with the new gate live. Two passes on 2022-23, v25 base config otherwise:

| RW_VAR_PER_WEEK | gate | pooled weighted | ATP weighted | WTA weighted | pooled lopsided |
|---|---|---|---|---|---|
| 0.0001 – 0.003 | **REJECTED** (fails new 3yr/5yr variance floors) | — | — | — | — |
| 0.0033 | **REJECTED** (fails 5yr floor by a hair: v_a=3.462<3.5) | — | — | — | — |
| 0.0034 | PASS | **0.5700** | 0.5704 | 0.5696 | **0.4203** |
| 0.0035 | PASS | 0.5701 | 0.5706 | 0.5697 | 0.4209 |
| 0.0036 | PASS | 0.5703 | 0.5708 | 0.5697 | 0.4215 |
| 0.0038 | PASS | 0.5706 | 0.5712 | 0.5699 | 0.4227 |
| 0.004 | PASS | 0.5708 | 0.5716 | 0.5701 | 0.4238 |
| 0.0045 | PASS | 0.5715 | 0.5723 | 0.5706 | 0.4264 |
| 0.005 (v25 baseline) | PASS | 0.5721 | 0.5731 | 0.5711 | 0.4288 |
| 0.006 | PASS | 0.5733 | 0.5744 | 0.5722 | 0.4331 |
| 0.008 | PASS | 0.5757 | 0.5771 | 0.5743 | 0.4403 |

The new gate cuts in cleanly and precisely between 0.0033 (fails) and 0.0034 (passes). **Within the safe region, the weighted metric is still monotonically improving all the way down to the new gate boundary, with no plateau or turning point** — the exact same "suspicious monotonic-to-boundary" shape flagged in round 4, just now cut off by a principled mechanism check instead of an arbitrary grid edge. Both tours move together (no divergence), consistent with round 4's finding this isn't WTA-specific. Unlike the previously-rejected Platt/isotonic/layoff-bump levers, lopsided log-loss improves *alongside* weighted here (0.4288→0.4203), not at its expense — structurally more like attempt 13/14's `CLIP_REINFLATE_K`+`MAX_SHIFT` co-tune (which did transfer) than like the calibration failures (which didn't) — so despite the boundary-shaped curve, this looked like a real enough candidate to spend the one holdout reveal on.

### Step 3 — holdout reveal: REJECTED, gap widened

Candidate: v25 config with `DSV_RW_VAR_PER_WEEK=0.0034` (best gate-passing value). Fit + MC-average (N=60) + odds join, exact harness from RESUME-beat-elo.md, script `tune_scratch/reveal_v26_rw_var_0034.py`:

| | Elo | Hier v25 (RW=0.005, documented) | **Hier candidate (RW=0.0034)** |
|---|---|---|---|
| Overall log-loss | 0.6152 | 0.6266 | **0.6275** |
| Overall gap | — | +0.0114 | **+0.0123** |
| ATP gap | — | +0.0008 | **+0.0013** |
| WTA gap | — | +0.0223 | **+0.0237** |
| Lopsided log-loss | 0.3853 | 0.4288 | **0.4153** (real improvement) |

**Verdict: REJECTED.** The gap to Elo got worse in every cut (overall, ATP, WTA) despite lopsided log-loss genuinely improving as the tuning window predicted — the tuning-window win on `weighted_logloss` did not transfer to the actual exit metric. The regression is modest in absolute terms (+0.0009 log-loss) compared to the catastrophic `var_scale` non-transfer (attempt 19, gap more than doubled), but it moved further from the goal, not closer, so there is no case for shipping it. `RW_VAR_PER_WEEK` is left at its v25 default (0.005) — no config change ships from this round. This is the **third** confirmed case this session of a genuine, non-degenerate tuning-window win on the correct selection metric failing to transfer to holdout, after `TIER_WEIGHT` (attempt 6) and `var_scale` (attempt 19) — all three shared the same monotonic-to-boundary shape, which in hindsight looks like a reliable warning sign for this codebase specifically, not just a one-off.

**Kept from this round regardless of the rejection**: the two new `stress_test_safety.py` checks (reproduction 5) are a genuine, permanent improvement to the safety suite — they close a real blind spot (every prior check only bounded variance from above) and are cheap (~0.1s) to run on every future candidate. Left in place.

### Step 5 — scoping ticket 21 vs ticket 24 vs "real ceiling" (per RESUME's instruction not to force a weak finding)

With the `RW_VAR_PER_WEEK` lead now closed, every lever flagged across five rounds (kappa dilution, MC over-correction, uncertainty/data-sparsity, linear/nonlinear calibration ×2 pooled + ×1 WTA-only, regime-B evidence-weighting, tournament-tier discounting, WTA-lower-tier discounting, layoff-bump ×2 forms, tour-specific hyperparameter re-scan, MC-variance curvature scaling, and now RW_VAR_PER_WEEK) has been tried and rejected or failed to transfer. Scoping the two remaining architecture tickets against the accumulated evidence, not re-deriving it:

- **Ticket 21 (deeper scoreline-inversion likelihood) is not indicated.** The cheap version already shipped in round 3 (real games-per-set evidence replacing the fake all-or-nothing trial) closed most of the original gap. The mechanism actually diagnosed for the REMAINING gap (round 3, third continuation, Findings 1–3) is downstream of evidence extraction entirely: `d1_mean`/`d2_mean` already extremize to match or exceed Elo's own confidence on the most lopsided matches (Finding 2) — the undershoot is introduced entirely by the MC-posterior-averaging step's Jensen's-gap convexity cost, which a richer per-match likelihood cannot touch (it would only make the already-correct mean *more* extreme, doing nothing to the downstream haircut). `diag2_bo5_sets.py`'s finding that hier trails most on the most decisive, evidence-richest wins (straight-set blowouts) directly argues against "not enough evidence" as the remaining problem.
- **Ticket 24 (heavy-tailed process noise / replace hard clamps) is not indicated either**, for the reason already on record: `diag2_fresh_cuts.py` found zero of the 10,678 holdout rows anywhere near the `MAX_STATE`/`MAX_VAR` clamp wall (`wall_proxy > 4.0` count = 0) — the absorbing-boundary pathology ticket 24 targets simply doesn't fire on any row this holdout would score, whatever value the rewrite might have for long-run archive robustness elsewhere.
- **The one theoretically-motivated fix for the actual diagnosed mechanism** (a *conditional*, curvature-shaped MC variance-scale, not a flat scalar — round 3's own flagged "secondary, smaller finding," never implemented) remains technically open. But treat that prospect with real skepticism now: its flat-scalar cousin (`var_scale`, attempt 19) and this round's `RW_VAR_PER_WEEK` finding are two independent, structurally-similar, principled-and-gated tuning-window wins that *both* failed to transfer, on the identical mechanism family (variance magnitude feeding the MC-averaging convexity cost). A conditional version is a bigger lift (new tuning surface, its own overfitting risk) chasing a mechanism that has now resisted two clean, well-motivated attempts at the simplest version of a fix.

**Honest conclusion: this looks like a genuine architecture ceiling for same-session tuning, not an unexplored lever.** Every hyperparameter axis is exhausted (confirmed independently in rounds 3 and 4), the one diagnosed remaining mechanism (MC-averaging's Jensen's-gap cost on already-extreme predictions) is structural to the Gaussian-filter/MC-averaging architecture rather than a tunable defect, and its most natural fix has now failed twice. Recommend NOT starting ticket 21 or 24 on the strength of current evidence — both would be multi-day rewrites aimed at mechanisms this session's diagnostics directly rule out as the cause of the remaining gap. If a future session wants to keep pushing, the honest next steps are either (a) the conditional/curvature-shaped MC variance-scale, entered with explicit awareness it's now 0-for-2 on this mechanism family, or (b) accepting 0.0114 (v25) as the closing number and moving on. Gap to Elo unchanged at **0.0114**, v25 remains the best config.

### Attempt 20 — surface, round, and player-experience cuts (read-only, exploratory eval set), no concentrated lever found

Closes out the last un-run items from the original "cut the eval data multiple ways" list (market-confidence buckets, tour, best-of/set-length, and layoff length were all done in earlier rounds; surface, round, and experience were not). Single-threaded pandas, `diag2_enriched.parquet` (10,678-row exploratory set, not the strict 8,594-row holdout — same caveat as every other `diag2_*` finding), no `mc_average` call, run in parallel with the `RW_VAR_PER_WEEK` stress-test agent with zero file overlap.

**Surface** (`hier_ll − elo_ll`): Clay +0.0054 (n=3,066), Hard +0.0088 (n=6,399), Grass +0.0127 (n=1,185). Mild, monotonic-looking but small, and Grass's larger gap is on the smallest n — not concentrated the way tour was.

**Round**: no clean pattern — R128 −0.0004, R64 +0.0022, R32 +0.0044, SF +0.0022, RR +0.0205, F +0.0181, QF +0.0163, R16 +0.0243. Later/smaller-n rounds skew a bit higher but it doesn't track match importance monotonically (SF is low, QF/F/R16 are higher) — reads as noise on rounds with n<1000, not a lever.

**Player experience** (`min(n_prior_w, n_prior_l)`, bucketed): non-monotonic and noisy at the n available — (0,10]: +0.0721 (n=88, hier worse), (10,30]: −0.0443 (n=121, hier better), (30,60]: −0.0446 (n=148, hier better), (60,100]: +0.0142 (n=356), (100,∞): +0.0092 (n=9,950, matches the pooled baseline). The very-low-experience bucket (hier worse than Elo on debut-ish players) is worth flagging for a future pass with more n — it sits oddly next to the earlier `diag2_wta_churn.py` finding that hier dramatically BEATS Elo on new/surprise ATP *winners* specifically (gap −0.34, n=99) — but that used a different construction (winner-side `n_prior_w<50` only, not `min` of both sides) and n=88 here is too small to trust alone.

**Verdict:** none of the three is a concentrated lever on the scale the tour split was. This is a useful negative result — it means the remaining 0.0114 gap is not hiding behind an unexamined axis; every dimension in the original diagnostic brief has now been checked (tour, market-confidence, set-length, layoff, surface, round, experience), and the tour-symmetric MC-convexity mechanism (round 3 third continuation) remains the only identified, well-evidenced explanation, still without a working fix.

## Round 6 (2026-08-28) — Attempt 22: conditional (point-estimate-threshold) var_scale, a genuine non-boundary tuning-window optimum, REJECTED on holdout — MC-variance-scale mechanism family now fully closed out

Picked up the one open thread flagged at the end of round 5: a quick single-threaded POC (`tune_scratch/diag4_conditional_var_scale_poc.py`, unfiltered mixed-year slice, not charter-compliant) had suggested that shrinking `d1_var`/`d2_var` only for rows where the model's OWN point estimate already calls the match lopsided (rather than a flat scale on every row) might avoid the flat scale's midband-erosion failure (attempt 19) while still closing much of the top-bucket undershoot. Task: implement properly, tune on 2022-23 only through the charter metric, check the midband explicitly, and reveal once on holdout only if a genuine (non-boundary) optimum survives.

### Implementation

Extended `mc_average()` in `dropshotdata/scripts/mc_parallel.py` with two new optional parameters: `var_scale_threshold: float | None = None` and `var_scale_high_scale: float = 1.0`. When `var_scale_threshold` is set, each row's own no-MC point estimate is computed cheaply (`p_match(sigmoid(d1_mean), sigmoid(d2_mean), best_of)` — one call per row, not `n_samples` of them) and rows where `max(pt, 1-pt) > threshold` get `var_scale * var_scale_high_scale` applied to both `d1_var` and `d2_var`; all other rows get the untouched base `var_scale` (default 1.0). `var_scale_threshold=None` (the default) skips the branch entirely — verified bit-identical to the pre-existing behaviour (`np.array_equal`, same seeds) across several parameter combinations (`threshold=None`, `threshold=1.0` never-exceeded, `threshold=0.0`+`high_scale=1.0` always-touched-but-no-op). `stress_test_safety.py` still ALL PASS (this lever is eval-time only, never touches fit-time state, per the brief no corruption-gate check was needed).

### Tuning-window scan (2022-23, v25 config, `tune_scratch/tune_cond_var_scale_2223.py` + `tune_cond_var_scale_fine.py`)

2D grid, `threshold ∈ {0.80, 0.85, 0.90, 0.95}` × `scale ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`, scored on the same weighted/plain/lopsided metric v25 was selected on, plus the 50-70% midband. Baseline (no-op): weighted=0.5671, lopsided=0.4238, midband=0.7007.

Best cell: **threshold=0.85, scale=0.7** — weighted=0.5653 (genuine improvement), lopsided=0.4144 (genuine improvement), **midband=0.7056 (degraded, +0.0049)**.

Every one of the 20 grid cells degraded the midband relative to baseline, including the mildest (threshold=0.95, scale=0.7: midband=0.7025, still +0.0018 worse). Diagnosed why (`tune_scratch/diag5_midband_touch.py`): the threshold conditions on the MODEL's *own* point-estimate confidence, not an external truth signal — and at threshold=0.85, **24.3% of the market-defined 50-70% midband bucket already has the model's own point estimate calling it >85% confidence** (n=1,184/4,869). That's exactly the model's known weak-disagreement pathology (round 3: hier disagrees with Elo weakly and is wrong more than chance ~44% of the time) — shrinking variance on those rows amplifies an already-wrong confidence rather than correcting a right one. This is smaller in magnitude than the flat scale's midband hit (+0.0113) but the same mechanism in miniature: gating on the model's own probability, even conditionally, inevitably captures some of the model's own miscalibrated-confidence rows along with the genuinely-well-calibrated lopsided ones.

**Confirmed the optimum is genuinely interior, not another monotonic-to-boundary artifact** (the specific pattern that burned `TIER_WEIGHT`, flat `var_scale`, and `RW_VAR_PER_WEEK`): extended the scale grid from 0.7 up to 0.98 at the same three thresholds (`tune_cond_var_scale_fine.py`) — weighted climbs smoothly and monotonically from 0.5653 (scale=0.7) back toward the baseline 0.5671 (scale=1.0, exact no-op by construction), with no further improvement past 0.7. This is the first candidate in this ticket's MC-variance-scale family to pass the non-boundary check cleanly.

### Holdout reveal: REJECTED — gap widened in every cut despite the non-boundary shape

Revealed once (`tune_scratch/reveal_v26_cond_var_scale.py`), threshold=0.85/scale=0.7, exact RESUME-beat-elo.md join pattern, N=60 MC, split by tour. Baseline reproduces the documented v25 numbers exactly (Overall 0.6266, ATP 0.6212, WTA 0.6322 — confirms harness fidelity):

| | Elo | Hier v25 (baseline) | Hier v25 + cond_var_scale(t=0.85,s=0.7) |
|---|---|---|---|
| Overall log-loss | 0.6152 | 0.6266 (gap **+0.0114**) | 0.6291 (gap **+0.0139**) |
| ATP log-loss | 0.6204 | 0.6212 (gap **+0.0008**) | 0.6221 (gap **+0.0017**) |
| WTA log-loss | 0.6099 | 0.6322 (gap **+0.0223**) | 0.6364 (gap **+0.0265**) |
| Overall lopsided | — | 0.4240 | **0.4148** (real improvement) |
| Overall midband (50-70) | — | 0.6928 | **0.6976** (degraded) |

**Verdict: REJECTED.** The gap to Elo got worse in every cut (overall, ATP, WTA) — same direction as all three prior rejections in this mechanism family, despite this being a genuinely different, non-boundary tuning-window optimum. Both effects the tuning window predicted transferred correctly in DIRECTION (lopsided genuinely improved, midband genuinely degraded) — this was not a case of the tuning-window signal being spurious or noise — but the net trade was unfavourable on the actual exit metric regardless. `mc_parallel.py`'s new parameters are left wired in at their no-op defaults (`var_scale_threshold=None`), same precedent as every other tested-and-parked lever in this ticket.

### MC-variance-scale mechanism family: now closed out

Four independent, well-motivated, properly-gated attempts at fixing the diagnosed MC-averaging convexity/Jensen's-gap undershoot (round 3, third continuation) have now been tried and rejected: `TIER_WEIGHT` (attempt 6, different mechanism but same failure shape), flat `var_scale` (attempt 19), `RW_VAR_PER_WEEK` toward zero (round 5, a different lever touching the same downstream variance-magnitude pathway), and now conditional/point-estimate-gated `var_scale` (this round). The first, third, and this round's each had a distinct shape (monotonic-to-safety-gate-boundary, monotonic-to-grid-boundary, and now a clean non-boundary interior optimum respectively) — ruling out "it only fails because it's a boundary artifact" as a blanket excuse. **All four transfer-failed anyway**, and the mechanism behind this round's failure (gating on the model's own possibly-miscalibrated confidence necessarily also touches some already-wrong-confidence rows) is structural to the family, not an accident of this particular threshold/scale choice — a genuinely different conditioning variable (e.g. gating on an EXTERNAL confidence signal like market/Elo probability rather than the model's own point estimate) is the only variant of this family that hasn't been tried, but that requires the holdout target itself (or a proxy for it) as an eval-time input, which is a different and much riskier kind of leakage risk than anything tried so far — not a natural next step, a different category of risk. **Recommend treating the entire MC-variance-scale-on-`d1_var`/`d2_var` mechanism family as exhausted.** Gap to Elo remains **0.0114**, v25 unchanged, remains best config.

## Round 7 (2026-08-28) — Attempt 23: ticket 21's full-DP scoreline likelihood, built and tuned properly — FLAT/no signal on the tuning window, NOT revealed on holdout

Pete asked for ticket 21's full version to be built regardless of round 5's own hedge that it likely wasn't indicated by the diagnostic evidence (the mean state was already found to extremize correctly - the defect is downstream, in MC-averaging's convexity cost). Built as a real architecture piece, not a shortcut: full detail, code, and stress tests are in `issues/21-build-scoreline-inversion-regime-b.md` (now closed) and commit `a5222a5`.

### What was built

`p_set`'s game-by-game DP was refactored (exact, bit-identical) into `p_set_full()`, returning the complete terminal set-score distribution `{(games_a, games_b): probability}` instead of collapsing to a scalar P(A wins). A new `adf_update_full_dp()` fits the combined-strength gap `d` against the log-likelihood of the OBSERVED set-by-set scoreline (product over sets of `P(exact set score | d)`, looked up from `p_set_full`), via a numerical (central finite-difference) Newton step generalizing `adf_update`'s closed-form Bernoulli Laplace update to a likelihood with no closed form - same prior-combination structure, same final variance-share split into `m_a`/`m_b`.

Symmetric-strength convention, made precise (ticket 12 only specified "symmetric", which is ambiguous - resolved by deriving it from the model's own existing alpha=beta=c/2 regime-B split): `p_a_serve = sigmoid(d/2)`, `p_b_serve = sigmoid(-d/2) = 1 - p_a_serve`. **Not** `p_a_serve == p_b_serve` - that reading is degenerate (it makes `p_match(p,p,...) ≈ 0.5` for every `d`, destroying all signal from the combined-strength gap). Verified via a dedicated monotonicity stress test (`P(A wins set)` non-decreasing in `d`, reaches both extremes, exactly 0.5 at `d=0` matching regime A's own untested-player default).

Gated behind `DSV_REGIME_B_FULL_DP` (default `"0"`, exact no-op - verified two ways: code inspection of the branch order, AND a real fit-output diff against the pre-change code on the full archive, max delta 4.4e-16 = float rounding noise, not a behavior change). 9 new `stress_test_safety.py` reproductions (6-14): refactor bit-identity, distribution normalization (sums to 1.0), extreme scorelines (6-0 6-0) at four different priors, the exact ticket-12 single-extreme-set grid-boundary case (a single 6-0 set under a weak/uninformative prior - the sharpest evidence shape, checked as a Bayesian update where the prior regularizes the otherwise-hopeless per-match MLE, not as a raw MLE), far-from-zero-`d` numerical stability, the symmetric-`d` monotonicity/degeneracy check, no-op guards (empty scores, non-positive weight), `parse_score_sets` edge cases, and grid-cache boundedness. **ALL PASS.**

`parse_score_sets()` parses per-set `(winner_games, loser_games)` tuples (unlike the existing `parse_score_games()`, which lumps the whole match into a single games-won sum) and rejects anything the DP can't represent - old-rules undecided long final sets (e.g. a pre-tiebreak-era 70-68 decider) or malformed tokens - falling back to the existing games-won-count approximation for the WHOLE match in that case (never mixing likelihood models within one match).

### Performance

A real risk given this runs in the hot loop for 72.5% of ~1.9M matches regardless of `DSV_EVAL_YEARS` (the whole archive is always fit, only the eval *rows* are gated by year). Mitigated with a fixed 0.01-wide cache grid over `d ∈ [-12.5, 12.5]` (not an unbounded LRU - bounded memory and lookup cost regardless of run length, unlike a raw cache keyed on continuously-varying real `d` values). Measured: microbenchmark ~50-240us/call depending on cache warmth; full-archive fit time went from ~39s (FULL_DP=0) to ~90-96s (FULL_DP=1) - well within budget for iterative tuning.

### Tuning-window scan (2022-23, v25 config as the fixed base, weighted/lopsided metric)

| config | weighted | plain | lopsided | corruption |
|---|---|---|---|---|
| baseline (FULL_DP=0, v25 exactly) | 0.5759 | 0.6347 | 0.4325 | 0.0% |
| FULL_DP=1, N=0.10 | 0.5760 | 0.6347 | 0.4329 | 0.0% |
| **FULL_DP=1, N=0.25 (v25's own N)** | **0.5758** | 0.6347 | 0.4324 | 0.0% |
| FULL_DP=1, N=0.50 | 0.5760 | 0.6351 | 0.4321 | 0.0% |
| FULL_DP=1, N=1.00 | 0.5763 | 0.6355 | 0.4320 | 0.0% |
| FULL_DP=1, N=2.00 | 0.5763 | 0.6357 | 0.4317 | 0.0% |
| FULL_DP=1, N=4.00 | 0.5763 | 0.6358 | 0.4314 | 0.0% |

Best cell (N=0.25) beats baseline by 0.0001 - noise, not a finding, on n=8,460 rows. Across a 40x range of evidence weight, weighted log-loss never leaves the 0.5758-0.5763 band. **Notably, N=2.0 and N=4.0 now pass the corruption gate cleanly (0.0%)** - under the OLD games-won-count approximation, `REGIME_B_N > 1.0` blew the corruption gate immediately (attempt 6, this issue's "Hypothesis 2 tested" section) because scaling up a fake all-or-nothing trial only made an already-maximally-confident data point more confident. The full-DP likelihood doesn't have that pathology (it reflects the actual, non-degenerate evidence in a real scoreline), confirming the new mechanism is qualitatively different and healthier in that specific respect - it just doesn't move the *metric that matters* either direction.

The visible lopsided-improves/plain-worsens trade as N increases (0.4329→0.4314 while 0.6347→0.6358) is recognized as the exact same pattern already rejected three separate times this session (Platt scaling, isotonic calibration, the layoff-bump lever) - not treated as a new lead.

### Verdict: NOT revealed on holdout

Charter: reveal the one-shot holdout check only for a candidate you actually believe in. Nothing here beat baseline by more than tuning-window noise, so no candidate qualifies. `DSV_REGIME_B_FULL_DP` stays at its no-op default; v25 is unchanged. **Gap to Elo: still 0.0114.**

### Why this result matters (confirms round 3's mechanism from a different angle)

Round 3's third continuation found the aggregate defect lives in MC-posterior-averaging's Jensen's-gap convexity cost, downstream of the mean state - and that the mean state itself already extremizes correctly on genuinely lopsided matches. This round tested the complementary hypothesis directly: if regime-B's evidence extraction were the bottleneck (not the downstream averaging), giving the fit access to the FULL scoreline likelihood instead of a coarse games-won-count approximation should have shifted the fitted state and moved the metric. **It changed essentially nothing.** That is strong, independent confirmation that the coarser evidence was already extracting what mattered for this metric - the ceiling here is downstream of fitting, not upstream in how much information gets pulled from each match. Ticket 21 is closed on this finding; see that issue file for the formal resolution.

### Operational note: a scan stalled silently, caught and recovered

The first attempt at this scan (`tune_ticket21_fulldp.py`, one script running all 8 candidates sequentially) stalled mid-scan after step 1 completed - `subprocess.run()` blocked with 0% CPU progress and a hidden child process not visible via a plain `tasklist //FI "IMAGENAME eq python.exe"` listing. Caught by checking CPU-TIME DELTA over an 8-second window (`Get-Process -Id X | Select CPU`, sampled twice - identical CPU value across 8s of wall time means genuinely stalled, "Responding: True" notwithstanding), not just process presence. Killed with `taskkill //PID X //T //F` (cleanly took the hidden child with it), confirmed no orphaned scratch files (`run_candidate()`'s preds/state cleanup runs on every exit path, confirmed empty after the kill). The remaining 5 grid candidates were re-run as individual foreground invocations (`tune_step2_single.py`) instead of one long backgrounded script, for tighter visibility per candidate. **New rule for future scans in this session**: don't conclude "still running fine" from process presence alone if log output has stalled for several minutes - check CPU delta before waiting further.

## Round 8 (2026-08-28) — Ticket 24: heavy-tailed process noise, built and tuned properly — FLAT/no signal on the tuning window, NOT revealed on holdout — both remaining scoped rewrites (21, 24) now closed with clean negatives

Picks up round 7's one recommendation and the last unattempted item from round 5's structural analysis. Full spec in `issues/24-robust-filtering-replace-hard-clamps.md`.

### What was built

`adf_update`'s and `adf_update_full_dp`'s Newton step assumes a Gaussian prior on the combined-strength gap `d` (the state fed in from the ordinary Gaussian random walk in `inflate()`). Per issue 24's own spec, approach (a): rather than a literal Student-t random walk (which has no closed-form recursion once convolved across ~1.9M sequential per-match time steps), each individual ADF update's local Gaussian prior is treated as an approximate Student-t(nu) via its scale-mixture-of-normals representation, solved with a genuine IRLS (iteratively reweighted least squares) scheme inside the SAME Newton loop already in `adf_update` - no new outer loop, no new per-player state.

The exact formula (`_heavy_tail_var` in `hierarchical_model.py`): at each Newton iterate `d`, `v_eff = v_d * max(1, (nu + r²)/(nu+1))` where `r² = (d - m_d)² / v_d` is the standardized residual against the prior mean. This is the standard Student-t-as-scale-mixture EM weight (`E[lambda|d] = (nu+1)/(nu+r²)`, effective variance `= v_d / E[lambda|d]`), with one deliberate, documented deviation: **floored at 1.0**. A literal Student-t with scale²=v_d is actually slightly *tighter* than v_d exactly at the mode (ratio `nu/(nu+1) < 1` at r=0) - accepting that would silently retune every ordinary, unsurprising update's effective variance, entangling this fix with every other hyperparameter already tuned under the assumption that v_d IS the working variance (`RW_VAR_PER_WEEK`, `PHI_MAX`, etc.). The floor means a typical/unsurprising update (`r² <= 1`) is byte-for-byte unaffected by `nu` - the mechanism is isolated purely to genuinely large surprises, the exact case MAX_SHIFT's hard clamp exists to guard against.

`MAX_SHIFT`/`MAX_STATE`/`MAX_VAR` are left fully, unconditionally in place - this is an enhancement to the underlying update math that should reduce *reliance* on the hard clamps, not a removal of them (the task brief was explicit this should not be a removal).

### Verification (before any tuning)

- **Bit-identical no-op at the default**, verified two ways, not just code inspection: (1) a synthetic reference-implementation diff — a verbatim copy of the pre-ticket-24 `adf_update` body compared against the new code with `HEAVY_TAILED_NOISE=0` across 4 test cases, max diff 0.0 (stress test reproduction 18); (2) a real fit diff on the 2022-23 window — v25 config run once with the env var unset (implicit default) and once with `DSV_HEAVY_TAILED_NOISE=0` explicit, 10,313 eval rows each, max absolute difference across ALL numeric output columns = 0.0, `DataFrame.equals()` True.
- **5 new stress tests** (reproductions 15-19, `stress_test_safety.py`, 24 checks total now, ALL PASS):
  - Repro 15: a single large surprising update (tight prior d=3.0, evidence 2/70 - a near-total upset) - heavy-tailed moves the mean further toward the evidence (shift 3.09 vs Gaussian's 1.55) AND retains MORE posterior variance (0.050 vs Gaussian's 0.034) on the identical input, confirming both halves of the intended mechanism actually fire, not just one.
  - Repro 16: 8 alternating large surprises in a row (bug 9's exact compounding pattern) - final state stays finite and MAX_STATE-bounded, and minimum variance observed across all 8 rounds never drops below 0.04 (no re-opened absorbing-boundary trap).
  - Repro 17: round 5's three long-layoff variance-sufficiency floors + the 3yr-layoff-overconfidence check, re-run explicitly under `HEAVY_TAILED_NOISE=1` - unaffected (as expected, since this ticket only touches the observation-update prior term, not `inflate()`'s time-decay process-noise growth), confirming no regression rather than assuming one.
  - Repro 18: the bit-identical no-op check described above, as a permanent regression test.
  - Repro 19: `_heavy_tail_var`'s own contract - exactly `v_d` at zero residual, strictly increasing beyond it, and less inflation at larger `nu` (converging toward the Gaussian limit).

### Tuning (2022-23 only, same weighted/lopsided metric v25 was selected on)

| Config | Weighted | Plain | Lopsided | Corruption |
|---|---|---|---|---|
| v25 baseline (`HEAVY_TAILED_NOISE=0`) | 0.5759 | 0.6347 | 0.4325 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=2.0 | 0.5760 | 0.6347 | 0.4329 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=3.0 | 0.5760 | 0.6347 | 0.4328 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=4.0 (default) | 0.5759 | 0.6347 | 0.4327 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=6.0 | 0.5759 | 0.6347 | 0.4327 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=10.0 | 0.5759 | 0.6347 | 0.4326 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=20.0 | 0.5759 | 0.6347 | 0.4325 | 0.0% |
| `HEAVY_TAILED_NOISE=1`, nu=30.0 | 0.5759 | 0.6347 | 0.4325 | 0.0% |

**Completely flat.** Turning the mechanism on at its own default (nu=4.0) is an exact tie with baseline (delta +0.0000) - not even a small wobble in either direction, the flattest result of any lever tried this session (flatter than ticket 21's, which at least showed a genuine, if tiny, 0.0001 movement). Plain log-loss is identical to 4 decimals (0.6347) at *every single nu value* tested, across a 15x range from very heavy tails (nu=2) to nearly-Gaussian (nu=30). Corruption stays at exactly 0.0% throughout - the gate never engages because the mechanism itself never meaningfully engages.

### Why this happened - confirms round 5's diagnostic directly, not just by inference

Round 5's diagnostic (`diag2_fresh_cuts.py`) already found zero of the 10,678 holdout rows have a "wall-proximity" measure anywhere near the `MAX_STATE` clamp - the premise round 5 flagged as undercutting ticket 24's motivation for the CURRENT gap. This round shows the direct mechanical consequence of that finding: `_heavy_tail_var` only produces a DIFFERENT effective variance than the plain Gaussian when `r² > 1`, i.e. exactly the situation where the raw Newton step would otherwise be pulling hard against MAX_SHIFT. If that situation essentially never arises on real archive data (consistent with round 3-continuation's separate finding, ticket 21's section above, that "real evidence almost never hits MAX_SHIFT... shift-survival scaling is a no-op at this evidence strength"), then a mechanism gated on exactly that situation has nothing to act on, regardless of how aggressively `nu` is tuned. **This is now confirmed empirically, not just inferred from a proxy measure, for the second of round 5's two flagged "not indicated" candidates** (ticket 21's richer regime-B likelihood, closed round 7; ticket 24's heavy-tailed noise, closed here) - both independently show flat tuning-window response, via two structurally different implementations, converging on the same conclusion: the MAX_SHIFT-adjacent/absorbing-boundary corner of this architecture, however real and bug-prone historically (bugs 6-9), simply isn't where THIS SPECIFIC holdout's residual gap-to-Elo lives.

### Verdict: NOT revealed on holdout

Charter: reveal the one-shot holdout check only for a candidate you actually believe in. The best case across the entire nu scan was an EXACT tie with baseline, not even a positive wobble - no candidate qualifies. `DSV_HEAVY_TAILED_NOISE` stays at its no-op default; v25 is unchanged. **Gap to Elo: still 0.0114.**

### Where this leaves the session

Both of round 5's flagged "not yet attempted" architecture rewrites - ticket 21 (round 7) and ticket 24 (this round) - are now closed, each with a clean, well-evidenced negative result reached via proper implementation, stress-testing, and charter-compliant tuning, not an abandoned or rushed attempt. Round 3's third-continuation finding (the MC-posterior-averaging Jensen's-gap convexity cost, downstream of an already-correctly-extremizing mean state) remains the best-diagnosed root cause of the residual gap, and now has no untried fix at any layer that's been tried so far: not evidence richness (21), not process-noise shape (24), not MC-variance-scaling (4 variants across rounds 3/6), not calibration (Platt, isotonic), not evidence reweighting (tier/WTA-lower/layoff-bump/RW_VAR_PER_WEEK). A genuinely new idea would need to attack the Jensen's-gap cost directly and structurally - e.g. an analytically-corrected or closed-form expectation of `p_match` over the posterior instead of Monte Carlo averaging a convex function of it - rather than another variant of evidence-weighting, calibration, or process-noise shape. That is unscoped and unattempted as of this round.

## Round 9 (2026-08-28) — a real causal-ordering bug found and fixed at the DATA-PREP layer (not a tuning knob): genuine tuning-window win, REJECTED on holdout, gap widens slightly to 0.0123

This round did not come from round 8's recommendation (which had exhausted the architecture-rewrite frontier and pointed only at Jensen's-gap cost as the remaining unattempted direction). It came from re-examining an assumption every previous round had taken for granted: that `prepare()`'s within-date match ordering was correct.

### The bug, confirmed systemic

`tourney_date` is a TOURNAMENT-level (start) date field, not a per-match date. Verified via the cached prepared archive: 617,380 player-date groups (26% of 2,378,344 player-appearances) share a date with another match by the same player, averaging ~2.9 matches/group, up to 23 on one date. `prepare()`'s only sort was `df.sort_values("date", kind="mergesort")` - stable, but that means within-date order was whatever raw file-concatenation order `load_matches()` produced, an unexamined assumption.

Traced the root cause directly in `load_matches()`'s code, not just a spot check: for each tour, the `pats` loop reads `atp_matches_[year].csv` for ALL years first, THEN `atp_matches_qual_chall_[year].csv` for ALL years, THEN `atp_matches_qual_itf_[year].csv` for ALL years. This means qualifying-round matches are concatenated after the ENTIRE main draw, for every tournament, every year, both tours - a structural property of the loader, confirmed systemic by inspecting the code path itself (stronger evidence than sampling individual tournaments, though a few same-date groups were also spot-checked and matched this pattern). Real-world impact: a player who had just won Q1, Q2, Q3 that same calendar day (in real life, always BEFORE the main draw starts) was evaluated in their R128 match using a state that hadn't yet absorbed those wins - treated as closer to a near-unknown than they should have been. This lines up with an earlier round's calibration-by-round finding that R128 was the single worst-calibrated round bucket (error +0.1493).

Round vocabulary confirmed present in the archive: `Q1,Q2,Q3,Q4,Q5,ER,R128,R64,R32,R16,QF,RR,SF,F,BR` plus one stray lowercase `f` - checked directly (a single WTA level-15 ITF-qualifying final on 2022-05-23; cross-referenced that date's other matches, which include a proper-case `F` row for the ATP main event same day, confirming `f`/`F` are the same round, just an inconsistent-casing data artifact, not a distinct label).

### The fix

New `ROUND_RANK` dict + `round_rank()` function in `hierarchical_model.py`:
```
Q1=0, Q2=1, Q3=2, Q4=3, Q5=4, ER=5, R128=6, R64=7, R32=8, R16=9, QF=10, RR=11, SF=12, F=13, BR=13
```
`RR` (round-robin, season-ending Finals events) ranked as a group stage before the SF/F knockout bracket it feeds. `BR` (bronze/3rd-place playoff) ranked with `F` - both happen after `SF`, effectively concurrent. `'f'` normalized to `'F'` before mapping. `prepare()`'s sort changed from `sort_values("date", kind="mergesort")` to `sort_values(["date", "_round_rank"], kind="mergesort", na_position="last")` - mergesort stability kept, so matches with identical date AND round rank (genuinely simultaneous, e.g. two different R32 matches same day) keep their original file-order relative sequence, same as before. Any round label not in `ROUND_RANK` gets `NaN` (sorted last within its date) with a logged `warnings.warn`, rather than crashing or guessing - checked against the real archive's full vocabulary above, which has zero unmapped/NaN rounds, so this path is a pure safety net for future data, not something that fires today.

**A second real bug found while verifying this one**: `load_prepared()`'s cache-freshness check compared the cache's mtime only against archive CSV mtimes (`ARCHIVE.rglob("*.csv")`) - an edit to `prepare()`'s own sort logic, like this fix, touches zero CSVs, so that check alone would have kept silently serving the STALE pre-fix cache indefinitely. Added a second check against `Path(__file__).stat().st_mtime` (this script's own mtime) so a code-only change to `prepare()`/`load_matches()` now also invalidates the cache. Manually deleted the stale `tune_scratch/prepared_matches_cache.pkl` this round as well, since the code edit alone wouldn't have forced a rebuild before this fix landed. **Lesson for future rounds**: don't assume a cache-freshness check covers every invalidation trigger just because it exists - it covered data-refresh but not code-refresh, and would have silently produced wrong results otherwise.

### New stress test

Reproduction 20 in `stress_test_safety.py` (29 checks total now, ALL PASS): builds a synthetic 4-match same-date scenario (Player A wins Q1, Q2, Q3, then R128, all `tourney_date=20230601`) fed to `hm.prepare()` in the archive's real "bad" file order (the R128 row listed first, mimicking main-draw-file-before-qual-files). Checks:
- `prepare()`'s output orders these `['Q1','Q2','Q3','R128']`, not the input's `['R128','Q1','Q2','Q3']`.
- Replaying the FIXED order through the same low-level primitives `run()` uses (`PlayerState`, `inflate`, `adf_update`, `apply_side_update`) shows Player A's state going INTO the R128 match already reflects the 3 qualifying wins (`n_matches==3`, `v_a` shrunk below `INIT_VAR`) - the actual causal-ordering assertion.
- A negative control: the OLD date-only sort (reproduced explicitly, not just asserted) puts R128 first, confirming the bug was real and this specific test would have failed pre-fix.
- `round_rank()`'s unknown-label fallback (NaN + logged warning, no crash) and `'f'`/`'F'` normalization, tested directly.

### Tuning (2022-23 only, `tune_v3_lopsided.py`'s exact weighted/lopsided metric - market-only, no Elo needed at the tuning-window stage)

| Config | Weighted | Plain | Lopsided | Corruption | n |
|---|---|---|---|---|---|
| v25 baseline (pre-round-9-fix) | 0.5759 | 0.6347 | 0.4325 | 0.0% | - |
| v25 + round-order fix | **0.5561** | 0.6174 | 0.4068 | 0.00000% | 8,459 |

A genuine, non-trivial improvement (Δweighted = −0.0198) - an order of magnitude larger than any of this session's noise-level "wins" (typically ±0.0001-0.0006 on this same metric). This is the first candidate all session with a tuning-window improvement large enough, and structurally-motivated enough (a real bug fix, not a hyperparameter nudge), to justify spending the one holdout reveal.

### Holdout reveal (2024-25, n=8,594 odds-joined rows, the exact join used every prior round): REJECTED

| | Elo | Hier (pre-round-9, v25) | Hier (post-round-9 fix) | Market |
|---|---|---|---|---|
| Overall log-loss | 0.6152 | 0.6266 | **0.6275** | 0.5888 |
| Overall accuracy | 0.6534 | 0.6453 | 0.6465 | 0.6792 |
| Lopsided (mkt_fav>0.80) log-loss | 0.3853 | 0.4288 | 0.4099 | 0.3645 |

Gap to Elo: **+0.0114 → +0.0123** (slightly worse). ATP gap +0.0074 (n=4,358), WTA gap +0.0173 (n=4,236) - WTA still carries the larger share of the gap, consistent with every prior round's finding on this axis. Lopsided log-loss did genuinely improve in absolute terms (0.4288→0.4099, the tuning-window signal DID transfer directionally on that one cut) but Elo's own lopsided log-loss (0.3853) is still meaningfully better, so hier remains behind Elo there too.

**This is the fifth confirmed case this session of a real, non-degenerate, correctly-gated tuning-window win failing to transfer to the holdout** - after `TIER_WEIGHT` (round 3-continuation), flat `var_scale` (round 4), `RW_VAR_PER_WEEK`-toward-zero (round 5), and the round-6 conditional `var_scale`. It is the first of the five that isn't a hyperparameter or calibration lever at all - it's a genuine, structural, verified-correct fix to how the model's input data is ordered. That it STILL didn't transfer is itself informative, not just another repetition: **it rules out data-prep-layer causal-ordering bugs as a candidate explanation for the residual gap**, closing off a category of hypothesis that hadn't been checked in rounds 1-8 (all of which implicitly assumed the input ordering was already correct).

### The fix is kept regardless of the rejection

This is a genuine correctness bug - the model was, until this round, evaluating predictions using future-relative-to-real-life information gaps in the wrong direction (systematically UNDER-crediting recently-won qualifying matches). That it happens to net out slightly negative on THIS SPECIFIC holdout comparison against Elo does not make the old behavior more correct; reverting it would mean deliberately shipping a known-wrong data-prep step to chase a specific metric. Committed at `111ecce` (fix + stress test) regardless of the holdout numbers, per this session's own standing policy of not gating genuine correctness fixes on whether they happen to help the current headline metric.

### Where this leaves the session

Round 3's MC-averaging-convexity diagnosis is now the ONLY diagnosed-but-unfixed root cause left standing after nine rounds of systematic elimination across every layer touched so far: hyperparameters (rounds 1-2, re-scanned rounds 3-6), architecture (tickets 21 and 24, rounds 7-8), calibration (Platt, isotonic), evidence-reweighting (tier, WTA-lower, layoff-bump, RW_VAR_PER_WEEK), MC-variance-scaling (4 variants), and now data-prep causal ordering (this round). **Recommended next step for whoever picks this up:** the only remaining direction with real diagnostic support is attacking the Jensen's-gap cost directly and structurally - e.g. an analytically-corrected or closed-form expectation of `p_match` over the posterior instead of Monte Carlo averaging a convex function of it. This has been flagged as the answer since round 3 and remains completely unscoped and unattempted. Absent a genuinely new idea there, nine rounds of otherwise-exhaustive, well-executed search argue this may be close to the ceiling of what same-session tuning of this specific architecture can achieve against this specific Elo baseline on this specific holdout.
