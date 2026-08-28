# RESUME HERE — goal: hierarchical model must beat Elo's log-loss on the 2024-25 holdout

**Pete's instruction (still active): keep working autonomously, don't check back in until this is achieved, or you hit a genuine blocker.**

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
