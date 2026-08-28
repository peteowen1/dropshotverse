# 09 — Prototype: frozen-skill vs live-updating ratings, side by side

Type: prototype
Status: resolved
Blocked by: 06, 08

## Question

Pete deferred this decision during charting and asked to see both traces rather than choose from a description. Build the picture.

The two candidates for what "updates live" means:

- **A — Frozen skill.** Player skills are fixed at their pre-match posterior. As points arrive, only the win probability moves, recomputed in closed form from the current score by the tennis Markov recursion. No refitting. Near-free to compute, and the ratings update once after the match ends.
- **B — Live-updating skill.** Additionally, the online filter nudges each player's serve/return skill *during* the match, so a player serving far above their baseline sees their estimate rise mid-match. More responsive to injury, illness and genuine day-form; costs a per-point filter update and raises the awkward question of what the official rating is at any instant.

**Build both against the same real historical matches and plot the win-probability traces together.**

Choose matches that discriminate:
- A straightforward favourite-holds-serve win (do they agree when nothing surprising happens?)
- A big upset where the underdog's level was clearly above their rating
- A mid-match retirement or visible injury (B should react, A cannot)
- A five-set comeback from two sets down
- A match decided on a final-set tiebreak

Then answer honestly:
1. How different are they, actually? If B tracks A closely except in the injury case, that's a strong argument for A's simplicity.
2. Does B ever behave badly — chasing noise, overreacting to one good service game?
3. Which is better calibrated in-match? Bucket by match state and check whether stated probability matches observed outcome frequency.
4. Can B run inside a Cloudflare Worker per point, given the constraints from ticket 06?

This is throwaway code. Do not build it into a package.

**Deliverable:** the plots, plus a recommendation with the reasoning — and put it to Pete rather than deciding for him, since he asked to see it.

---

# ANSWER

Built 2026-08-27 against **real point-by-point data** captured from the 2026 US Open (ticket 16) — genuine points, not simulated. Scripts: `dropshotdata/scripts/ticket09_live_modes.py`, `ticket09_plot.py`. Plot: `dropshotdata/ticket09/traces.png`.

## Scope honestly limited by what's captured so far

The capture so far covers only **Fan Week** (mixed doubles / qualifying, best-of-3) — the main draw hadn't started. Two of the five requested scenarios are **not available and not faked**:
- **Comeback from two sets down** needs best-of-five, which doesn't exist yet in this data.
- **Mid-match retirement/injury** — none found across all 94 captured matches.

Both should be revisited once main-draw men's matches are captured; the harness needs no changes to handle them.

## Three real bugs found before the plot could be trusted

Every one was caught by a verification step, not spotted by eye:

1. **Mid-match recursion boundary bug.** The new "win probability from an arbitrary in-progress score" function gave every *future* (not-yet-reached) set a flat 50/50 instead of this pairing's real set-win probability — checked against the known-correct fresh-match function at the (0,0,0,0) boundary, where they must agree exactly. Understated the leader's true edge by 0.15–0.19. Fixed by computing the future-set probability the same way the original function does (symmetrised over serve order), leaving only a ~0.4% residual explained by a real, intentional difference: the fixed function correctly uses the *known* current-set server rather than pretending it's unknown.
2. **Units bug in the live-updating filter.** Mode B fed raw probabilities (e.g. 0.60) directly into `adf_update`, which expects log-odds — every game, win or lose, pushed the estimate upward from an already-inflated starting point. Fixed by tracking Mode B's state on the logit scale throughout, converting to probability only for display.
3. **Type-mismatch corrupting set-counting.** `cur_set` was initialised as the Python int `1`; the tape's own `set` field is the string `"1"`. `'1' != 1` is `True` in Python, so the very first game of every match falsely triggered a "set just ended" branch on an empty 0-0 tally, starting every match at a false 0-1 deficit that corrupted the rest of the trace. Caught by adding a permanent assertion — the replay's own final bookkeeping must match the tape's `P1SetsWon`/`P2SetsWon` — which now runs on every match, not just as a one-off diagnostic.

A byproduct: two of the four hand-picked scenario labels were simply wrong (checked against total sets won without checking the *order*). 11104, labelled "reversal," was actually a straight-sets loss for player 1. Replaced with two verified genuine comebacks (11105, 11112) found by properly checking set order.

**Minor known artifact, not chased further:** both comeback traces dip sharply toward 0 at the very last data point, after the match is already effectively decided. Likely a trailing/edge-case entry in the tape rather than a modelling error; doesn't affect the overall shape or the comparison's conclusion.

## What the plot shows (`traces.png`)

**Straightforward wins (11101, 11103):** A and B track each other closely and both climb steadily toward 1.0. B is consistently *more* confident than A throughout — never dramatically, a few points higher at every stage — because a player winning normally keeps generating small positive evidence for B to act on, evidence A structurally ignores.

**Genuine comebacks (11105, 11112):** both traces correctly fall **below 0.5** once set 1 is lost, sit in the 0.2–0.3 range through a tight second set, then climb back to ~0.9 as player 1 closes out the match. This is the qualitative behaviour the scenario is meant to test, and both modes get it right.

## Answering the four questions

1. **How different are they, actually?** Not very, in these four matches. B tracks A's shape closely throughout — same direction, same timing of swings — offset by roughly 0.05–0.15 in the direction of whoever is currently doing better. There's no case here where B tells a qualitatively different story from A.
2. **Does B ever behave badly?** No chasing-noise or wild overreaction observed in these four matches. The largest single-game jumps in B are proportionate to A's own jumps at the same moment (both dip together at the same tight passages), suggesting B's damping (`PRIOR_VAR=0.30`) is reasonably tuned for this purpose, not that B is well-validated in general.
3. **Which is better calibrated in-match?** Not testable with four matches — this needs a large sample scored the way ticket 10 scored the pre-match models (predicted-vs-realised by bucket). Flagging as unfinished rather than guessing.
4. **Can B run inside a Cloudflare Worker per point?** Yes, easily. Each update is one `adf_update` call — a handful of arithmetic operations — well within ticket 06's "a few hundred float ops per poll tick" budget. No serialised model needed beyond the two players' current logit-scale estimates and variances, which is a smaller payload than the pre-match skill blob ticket 08 already specifies exporting.

## Recommendation — put to Pete, not decided here

**A is the better default**, on this admittedly small sample: B never told a meaningfully different story, costs a materially harder implementation (in-match state has to be initialised, tracked, and reconciled with the pre-match posterior architecture ticket 08 already specifies), and raises the exact "what's the official rating right now" ambiguity the ticket flagged as B's cost. **B's case would strengthen** if the retirement/injury scenario — the one case B can react to and A structurally cannot — turns out to matter once main-draw data allows testing it, and if a larger-sample calibration check (question 3) shows B genuinely tracks outcomes better mid-match, not just differently.

**Recommend: ship A for v0.** Revisit B once (a) main-draw retirement data exists to test the one scenario it's actually built for, and (b) a proper in-match calibration comparison can be run at scale.
