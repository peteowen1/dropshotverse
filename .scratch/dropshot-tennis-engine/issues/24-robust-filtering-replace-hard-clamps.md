# 24 — Replace hard clamps with heavy-tailed process noise (robust filtering)

Type: task
Status: open

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
