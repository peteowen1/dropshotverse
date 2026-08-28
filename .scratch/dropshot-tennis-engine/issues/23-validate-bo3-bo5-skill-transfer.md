# 23 — Is best-of-3 / best-of-5 skill transfer real?

Type: research
Status: open

## Question

Ticket 11 flagged this explicitly as **NOT ESTABLISHED**: the model currently assumes a player's serve/return skill (`mu`, `kappa`) is the same parameter whether they're playing best-of-3 (ATP non-Slam, all WTA) or best-of-5 (ATP Slams) — i.e. it pools skill across formats and only lets `best_of` affect the match-win recursion, not the underlying per-point skill itself. If some players are systematically better/worse at the longer format (fitness, mental durability, serve reliability over 5 sets) independent of their per-point skill, every Slam prediction is quietly biased in a way that wouldn't show up in the diagnostics run so far (ticket 20's calibration and variance cuts were format-blind).

Is there a real, identifiable best-of-3-vs-5 skill differential once player skill and opponent are controlled for? If yes, does it change the architecture (a format-specific offset analogous to `kappa`'s surface offset) or is it small enough to ignore (apply `stats-discipline` §3's quadratic-error-reduction sizing before proposing a fix).

## Notes

- Research-type: this needs literature/data investigation before any model change, not a hyperparameter tune.
- If real and architecturally significant, the natural fix mirrors `kappa` (surface offset) — a `phi_i,format` shrunk hierarchical offset — which is a much bigger change than tickets 20-22's scalar tweaks and should get its own charter before touching `hierarchical_model.py`.
