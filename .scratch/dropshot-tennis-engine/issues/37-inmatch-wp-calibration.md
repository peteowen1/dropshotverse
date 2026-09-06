# Ticket 37: in-match win probability, calibrated at scale

Closes the one item ticket 09 left open. That prototype shipped Mode A on 94
Fan Week matches and wrote, of its own question 3: *"Which is better calibrated
in-match? Not testable with four matches - this needs a large sample scored the
way ticket 10 scored the pre-match models. Flagging as unfinished rather than
guessing."*

Run 2026-09-06 on 78,864 points across 492 completed main-draw singles matches
(`build_point_table.py`, ticket 36's capture audit).

## Result: Mode A wins on every metric

| arm | ECE | log-loss | Brier |
|---|---|---|---|
| **A - fixed skill** | **0.0471** | **0.56916** | **0.18788** |
| B - online skill | 0.0737 | 0.59044 | 0.19021 |

Ticket 09 recommended shipping A "on this admittedly small sample" and said B's
case would strengthen if a larger-sample calibration check showed it tracked
outcomes better. It does not. It tracks them worse, on all three measures.

Both arms are strictly leak-free. A gives both players the tour-average serve
probability (measured: 0.6090 men, 0.5715 women) and so uses no player
information at all. B estimates each player's serve-win probability from the
points they have served SO FAR IN THIS MATCH, shrunk to the tour average with a
30-point prior, updated only after each point is predicted.

Running B this way also sidesteps what blocked the comparison before: it needs
no pre-match rating and no player-id crosswalk, neither of which the point tape
carries (it has "P1"/"P2" and names in prose, not uids).

## The two arms fail in OPPOSITE directions, which is what makes it readable

**A is systematically underconfident.** Predicted 0.747 in the 0.7-0.8 bucket,
realised 0.836; predicted 0.847 in 0.8-0.9, realised 0.929.

**B is systematically overconfident.** Predicted 0.852 in 0.8-0.9, realised
0.744; predicted 0.962 in 0.9-1.0, realised 0.883. And it puts far more mass in
the extremes - 12,408 points in the bottom decile against A's 2,590, 14,235 in
the top against A's 2,407.

That is ticket 09's question 2 - *"Does B ever behave badly - chasing noise,
overreacting to one good service game?"* - answered. On four matches it saw "no
chasing-noise or wild overreaction observed". On 492 matches, B reads a player's
early serve dominance as durable skill and extrapolates it.

## The anchor that would have misled

C2 asked whether the eventual winner is favoured on the final point. **B scores
96.7% against A's 90.0%** - which reads as B winning and is not. By match point
the evidence is overwhelming; being *more* certain there is overconfidence
paying off in the one place it cannot cost anything. A metric evaluated only
where the answer is already obvious cannot distinguish a better model from a
bolder one.

Kept as an anchor because it does catch a genuinely broken model (one that does
not know who is winning at match point), but it must not be read as a quality
ranking.

## What this does NOT establish, and the real next step

**A's underconfidence is not a defect of the recursion.** It is the direct
consequence of giving both players equal skill: a fresh match is a coin flip
(C1 = 0.5000 exactly), and every subsequent probability is anchored to that. The
favourite in a mismatched pairing is genuinely stronger than A can express.

So the missing piece is neither the recursion nor online updating - it is
**seeding the recursion with real pre-match skill**. That needs each match's two
players resolved to rated uids, which the tape does not provide and which is the
same crosswalk problem as issue #3.

Predicted, for whoever runs it: A's underconfidence should largely disappear
once p1/p2 come from pre-match ratings rather than a shared tour average, and
the ECE should fall well below 0.0471. If it does not, the recursion is the
problem after all and that would be the finding.

## Reproduce

    python dropshotdata/scripts/build_point_table.py
    python dropshotdata/scripts/inmatch_wp_calibration.py

Evaluation is at game granularity (sets, games, who serves next), matching what
`ticket09_live_modes.p_match_from_state` accepts - already verified there
against the fresh-match function at the (0,0,0,0) boundary.
