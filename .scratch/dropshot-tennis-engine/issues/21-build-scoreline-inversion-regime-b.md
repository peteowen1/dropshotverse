# 21 — Build the panel-based scoreline inversion for regime-B matches

Type: task
Status: open

## Question

Ticket 12 established that per-match scoreline inversion is hopeless (a single 6-4 6-4 puts the MLE at the grid boundary) but a **panel** of ~150 sets from the same pool of players recovers the true serve/return split to within ±0.03 — feasibility proven, never built. Regime B (72.5% of the archive, scoreline-only) currently gets only a combined-strength update (ticket 20 hypothesis 2 also flags this path as under-tested for a different, more basic reason: even the combined-strength update's `n=1` evidence weighting is unvalidated).

Build the actual panel-based inversion: given a rolling panel of recent matches for a player-pool (or the update step re-derived to use the full Markov scoreline likelihood, not a Bradley-Terry `sigmoid(cw-cl)` proxy), does regime-B skill estimation improve — measured the same way as every other change this session (tune 2022-23, corruption-gated, reveal once on 2024-25)? This subsumes and should be attempted only *after* ticket 20's regime-B evidence-weighting question is answered, since a `REGIME_B_N` fix might already close much of the gap far more cheaply.

## Notes

- Depends on ticket 20's regime-B finding (evidence weight vs. real inversion may be two different sizes of fix for the same symptom — cheapest first).
- Reuses ticket 12's panel-of-150-sets result directly; re-derive rather than re-research.
- Must go through the same corruption gate and tune/holdout charter as tickets 17-20 — do not skip stress-testing given how much of this session was bug-hunting in exactly this kind of new machinery.
