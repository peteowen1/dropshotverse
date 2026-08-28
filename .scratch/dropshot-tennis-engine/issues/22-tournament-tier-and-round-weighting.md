# 22 — Weight match updates by tournament tier and round

Type: task
Status: open

## Question

`tourney_level` and `round` are loaded into every row (`hierarchical_model.py:382`) but never used anywhere in the update logic — a Slam final currently updates skill by exactly the same amount as a Challenger R1 match. Design and test a weighting scheme (analogous to `REGIME_B_N` from ticket 20 — scaling the effective evidence `n` of a match's outcome by tier/round) and measure whether it closes any of the gap to Elo, via the same corruption-gated tune(2022-23)/holdout(2024-25) charter as every other hyperparameter this session.

## Notes

- Cheapest of the three architecture gaps identified in ticket 20's follow-up discussion (`REGIME_B_N` gives a working template: one new tunable scalar, same gate, same harness).
- Open design question to settle before tuning: weight by tier alone, round alone, or a combined "importance" score — and whether it should scale evidence (`n`) the way `REGIME_B_N` does, or scale `RW_VAR_PER_WEEK`'s effective injection differently. Recommend starting with tier-only, since round-importance interacts with best-of and is more likely to need real research (see literature already surfaced in ticket 05 on point-importance effects, Paserman 2010/2023).
