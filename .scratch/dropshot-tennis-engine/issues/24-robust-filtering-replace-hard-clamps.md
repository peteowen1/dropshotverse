# 24 — Replace hard clamps with heavy-tailed process noise (robust filtering)

Type: task
Status: closed (2026-08-28, round 8) — built properly, stress-tested, tuned, FLAT/no signal, not revealed on holdout

## Question

`MAX_SHIFT` and `MAX_STATE` are hard clamps bolted onto a Gaussian random-walk
filter, added reactively across tickets 10 and 18 to stop specific observed
runaway-drift failures (bugs 3, 9). Ticket 20 found a fifth instance of the
same underlying failure mode (the `DSV_REGIME_B_N` scan), reopening the
"absorbing boundary" trap through a path with none of the existing protection.
Four fixes so far have each patched one specific way a Gaussian filter's
variance can shrink to near-zero before the mean is actually correct, locking
in a wrong estimate that no future evidence can move. This is a structural
property of Gaussian-noise Kalman-style filters combined with hard clamps, not
a single bug — Elo has no analogous failure mode because it has no internal
confidence state that can be miscalibrated in the first place.

Does replacing the Gaussian random-walk process noise with a heavier-tailed
distribution (Student-t is the standard choice) remove the need for
`MAX_SHIFT`/`MAX_STATE` as hard clamps entirely — i.e. does the model become
naturally self-limiting (large, surprising shifts get absorbed probabilistically
rather than needing a hard ceiling), while ALSO fixing the exact regime-B
evidence-weighting problem ticket 20 hit (since regime-B's only protection
today is the hard `MAX_SHIFT` clamp, which is precisely the thing this would
remove)?

## Notes

- This is the "textbook" fix for absorbing-boundary pathologies in Bayesian
  filtering — heavy tails let the posterior update stay large when the
  observation is surprising, without the variance having falsely converged.
- Needs new update math (the ADF/Laplace step in `adf_update` assumes a
  Gaussian prior; a Student-t prior doesn't have the same closed-form Newton
  step and may need particle filtering, variational approximation, or a
  scale-mixture-of-normals reformulation — this is real numerical work, not a
  parameter swap).
- Must go through the full corruption-gated tune(2022-23)/holdout(2024-25)
  charter, AND be re-verified against every stress test in
  `stress_test_safety.py` plus the specific real-population anchors (Djokovic,
  Sinner, Prizmic) traced this session — a new failure mode in unfamiliar
  territory (heavy-tailed filtering) is exactly the kind of thing that needs
  the real-population trace (`DSV_DEBUG_UID`), not just synthetic tests, per
  ticket 18's own finding that synthetic tests alone missed the dominant bug.
- Biggest, riskiest item on the current frontier (21-24) — recommend doing
  this only after ticket 21 (scoreline inversion) or ticket 22 (tier
  weighting) have been tried, since either might close enough of the gap to
  make this unnecessary, and both are lower-risk.

## Resolution (round 8, 2026-08-28)

**Built exactly as scoped, approach (a): a scale-mixture-of-normals local
Gaussian approximation with an adjusted/inflated variance per update**, not a
parameter swap and not full numerical integration. `_heavy_tail_var` in
`hierarchical_model.py` replaces `adf_update`'s and `adf_update_full_dp`'s
Gaussian-prior Newton term with an IRLS-reweighted effective variance
`v_eff = v_d * max(1, (nu + r²)/(nu+1))`, floored at 1.0 by design so
ordinary/unsurprising updates are byte-for-byte unaffected and only
genuinely large surprises (the exact MAX_SHIFT failure mode) get the
heavy-tailed treatment. `MAX_SHIFT`/`MAX_STATE` hard clamps were **kept in
full**, not removed — this answers the ticket's own question ("does this
remove the NEED for clamps") empirically rather than assuming an answer:
see below.

Gated behind `DSV_HEAVY_TAILED_NOISE` (default 0, verified bit-identical
no-op both synthetically and on a real 2022-23 fit — max diff 0.0 both
ways). 5 new stress tests added (`stress_test_safety.py` reproductions
15-19): single large surprise (heavy-tailed absorbs it — moves further AND
retains more posterior variance than Gaussian on the same input), a
sequence of 8 alternating large surprises (bug 9's exact pattern — variance
never collapses), the round-5 long-layoff sufficiency checks re-verified
unaffected under the flag, and the helper's own floor/monotonicity/nu-limit
contract. ALL PASS.

**Tuned on 2022-23 through the same weighted/lopsided metric v25 was
selected on: completely flat.** Turning the mechanism on at its default
nu=4.0 is an EXACT tie with baseline (delta +0.0000). A full nu scan
(2.0-30.0) never moves plain log-loss past 4 decimals (0.6347 throughout)
and weighted log-loss stays in a 0.5759-0.5760 band. Corruption 0.0%
throughout — the gate never engages because the mechanism itself barely
engages.

**Root cause, confirmed empirically not just inferred**: round 5's own
diagnostic (`diag2_fresh_cuts.py`, see issue 20) already found zero of the
10,678 holdout rows sit anywhere near the `MAX_STATE`/`MAX_SHIFT`-engaging
wall. `_heavy_tail_var` only differs from the plain Gaussian variance when
`r²>1` — exactly the situation that wall-proximity measure was checking for.
If that situation essentially never arises on real archive data (consistent
also with round-3-continuation's separate finding that real evidence almost
never hits MAX_SHIFT), a mechanism gated on it has nothing to act on,
regardless of how aggressively `nu` is tuned — which is exactly the flat
scan observed. **So: no, replacing the Gaussian process noise with a
heavy-tailed alternative does NOT remove the need for the hard clamps on
this archive** — not because the heavy-tailed math doesn't work (the stress
tests confirm it does exactly what it's designed to do on a genuine
surprise), but because the clamps are already so rarely binding on real data
that there's essentially nothing for either mechanism (clamp or heavy tail)
to visibly disagree about at the aggregate metric level.

**Not revealed on holdout** — no candidate beat baseline by more than
tuning-window noise (best case was an exact tie), so per charter this
reveal was correctly withheld. `DSV_HEAVY_TAILED_NOISE` stays at its no-op
default; v25 (unchanged) remains the best config.

This closes the last unattempted item from round 5's structural analysis.
Both scoped rewrites on the 21-24 frontier (ticket 21, closed round 7;
ticket 24, closed here) are now resolved with clean, well-evidenced
negative results. See `issues/20-why-does-hierarchical-lose-to-elo.md`'s
"Round 8" section for full detail and numbers.
