"""Condition a match probability on the score so far, seeded by the model.

This is the "arm C" measured in docs/plans/validation-charter.md, extracted so
the shipped simulator and the screen that measured it run the SAME code. They
were briefly going to be two copies, which is how the F4 head-to-head skew got
in: training counted one thing, serving counted another, and nothing failed.

What it does. The pre-match model gives p(player 1 beats player 2) for a
pairing. The point recursion in ticket09_live_modes gives p(player 1 wins) from
any (sets, games, server) state, but needs each player's serve-point
probability, which the match model does not produce. So: hold the pair's serve
probabilities summing to twice the tour average, and solve for the DIFFERENCE
that reproduces the model's pre-match number at the opening state. The result
carries all of the model's pre-match skill and none of the score's noise, and
then the recursion updates it with the score that has actually been played.

Holding the sum fixed is what keeps this honest. It means a seeded pairing can
differ from the tour-average pairing only in who is better, never in both
players being better servers, so the arm cannot win by inflating serve hold
rates until the favourite runs away with it.

Measured on the 2026 US Open point tape, 231 matches and 42,657 points: ECE
0.0456 against the published 0.1063, log-loss 0.5218 against 0.5907, accuracy
0.784 against 0.747. It beats the published number in every round, including
the only late-round evidence available (round 4, 7 matches).
"""
from __future__ import annotations

from ticket09_live_modes import p_match_from_state

# Serve-point win rate by tour, from the MAIN-DRAW-ONLY points of the completed
# 2026 US Open tape (dropshotdata/usopen_2026_points.parquet, event 1 = men, 2 =
# women; 254 main-draw matches, 47,431 points). D-58: match ids with 5
# characters are qualifying/doubles/juniors, not this tour's main draw - some
# are even the WRONG tour's players under the right-looking event digit
# (confirmed on real names). Including them moved the ATP figure from 0.628 to
# a blended 0.611, understating exactly the population `first_round`/live
# seeding actually plays against. Main draw is also the only population that
# matters here: this constant only ever gets used against main-draw pairings.
# A constant rather than a recomputation because the live path runs on a
# runner that has no point tape, and because the solve only uses this as the
# CENTRE it splits around - the pairing's skill difference is what carries the
# information, and that comes from the model.
SERVE_BASE = {"atp": 0.627564, "wta": 0.578741}

# How far the solved opening state may sit from the model's pre-match number
# before the solve is treated as failed rather than approximate. This is the
# charter's anchor C2 bar, enforced per match instead of reported in aggregate.
TOLERANCE = 0.005

_wp_cache: dict[tuple, float] = {}


def wp(p1: float, p2: float, s1: int, s2: int, g1: int, g2: int,
       p1_serves: bool, best_of: int, pt1: int = 0, pt2: int = 0) -> float:
    """p_match_from_state, memoised on serve rates rounded to 1e-4.

    The rounding is the solve's real resolution, not a detail: at 1e-3 the
    bisection cannot distinguish serve rates any finer, which showed up as a
    maximum anchor-C2 error of 0.00495 against a 0.005 bar. 1e-4 puts it two
    orders of magnitude clear at no measurable cost, because the states being
    memoised are shared across matches rather than within one solve.
    """
    key = (round(p1, 4), round(p2, 4), s1, s2, g1, g2, p1_serves, best_of, pt1, pt2)
    v = _wp_cache.get(key)
    if v is None:
        v = p_match_from_state(*key)
        _wp_cache[key] = v
    return v


def _opening(base: float, d: float, best_of: int,
             p1_serves_first: bool | None) -> float:
    """The opening-state probability for a serve split, over a known or
    unknown first server. None means unknown - during a tiebreak the feed's
    serve field names a point server, so who served game one cannot be
    recovered - and is answered by symmetrising, which is exactly the
    assumption the pre-match model itself makes."""
    if p1_serves_first is None:
        return 0.5 * (wp(base + d, base - d, 0, 0, 0, 0, True, best_of)
                      + wp(base + d, base - d, 0, 0, 0, 0, False, best_of))
    return wp(base + d, base - d, 0, 0, 0, 0, p1_serves_first, best_of)


def solve_serves(p_target: float, base: float, best_of: int,
                 p1_serves_first: bool | None) -> tuple[float, float]:
    """(p1, p2) with p1 + p2 == 2*base whose opening-state probability ~= p_target.

    Monotone increasing in the difference d, and d must be allowed to go
    NEGATIVE: at d=0 the recursion states exactly 0.5, so searching only d>=0
    clamps every match where the model makes player 1 the underdog to a coin
    flip. The first version did that and anchor C2 caught it at a maximum error
    of 0.496 - half the matches, silently, while the aggregate log-loss still
    looked fine.

    Solved against the recursion AT THE OPENING STATE with the actual first
    server, not against hm.p_match. The two differ by the serve-order effect:
    the pre-match model does not know who serves first, the recursion does, and
    anchor C2 measured that gap at 0.027 when this solved against p_match.
    Solving against the state the arm is evaluated from makes this exactly
    "the model's pre-match number, then updated by the score".
    """
    p_target = min(max(p_target, 1e-4), 1 - 1e-4)
    bound = min(base, 1 - base) * 0.999
    lo, hi = -bound, bound
    for _ in range(40):
        d = (lo + hi) / 2
        if _opening(base, d, best_of, p1_serves_first) < p_target:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    # The bisection returns the closest achievable answer whether or not the
    # target was reachable, so saturating at the bound is indistinguishable
    # from solving unless the residual is checked. Measured across the whole
    # clamped range it peaks at 0.00049 and d never exceeds 0.37 of the bound,
    # so this does not fire today - it fires if SERVE_BASE or the recursion
    # changes underneath it, which is the point.
    residual = abs(_opening(base, d, best_of, p1_serves_first) - p_target)
    if residual > TOLERANCE:
        raise ValueError(
            f"solve_serves could not reach {p_target:.6f} from base {base:.6f} "
            f"(best of {best_of}): closest was off by {residual:.6f}, at "
            f"{abs(d) / bound:.1%} of the serve-split bound")
    return base + d, base - d


def seeded_probability(p_pre: float, state: dict, tour: str, best_of: int) -> float:
    """p(player 1 wins) given the model's pre-match number and the live score.

    `state` is one entry of `usopen_live_draw.load_live_draw()["live"]`, whose
    player 1 is the feed's team1 - the same orientation `p_pre` must be in.
    """
    base = SERVE_BASE[tour]
    p1, p2 = solve_serves(p_pre, base, best_of, state["p1_served_first"])
    return wp(p1, p2, state["sets1"], state["sets2"],
              state["games1"], state["games2"], state["p1_serves"], best_of,
              state.get("points1", 0), state.get("points2", 0))
