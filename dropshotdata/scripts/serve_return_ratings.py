"""Serve and return ratings, on a scale a reader can act on.

The hierarchical point model has carried a per-player serve skill and return
skill since it was built, with an uncertainty on each, and neither has ever
been published. `ratings.parquet` ships elo_overall, bt_rating and age.

The model's own arithmetic is the conversion. For a point served by A against
B, on surface s:

    e_A = A.m_a + A.kappa(s)            effective serve
    f_B = B.m_b + B.lam(s)              effective return
    P(A wins the point) = sigmoid(e_A - f_B)

So the honest interpretable number is that same expression against a
tour-average opponent: what share of points would this player win on serve
against the average returner in this field, and what share would they win when
returning the average server. That is a percentage, it is what the model
actually believes, and no new modelling is invented to produce it.

Hold and break come from the model's own `p_game`, which is what the match
recursion already uses.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

It does not put the tours on one axis. They are fit separately, so the raw
parameters are not comparable: WTA serve skills run higher than ATP ones while
women in fact win FEWER serve points (57.0% against 60.9% on the 2026 US Open
tape). Every number here is relative to that player's OWN tour's average, which
is the only comparison the model supports.

It does not claim the ATP and WTA splits are equally informative. Serve and
return correlate at +0.202 on the ATP side and +0.655 on the WTA side, so for
the men the two numbers separate genuinely different skills and for the women
they substantially restate overall quality. `serve_return_split_quality`
reports that correlation so the caller can decide rather than assume.

Validated out of sample against the 2026 US Open point tape, from skills
frozen before the tournament started: correlation with the share of serve
points actually won was +0.320 (ATP, 74 players) and +0.541 (WTA, 50).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hierarchical_model import p_game, sigmoid

REFERENCE_SURFACE = "Hard"


def effective(state, surface: str = REFERENCE_SURFACE) -> tuple[float, float]:
    """(effective serve, effective return) for one player on one surface.

    Exactly the two quantities hierarchical_model forms before taking a
    sigmoid of their difference; kept in one place so a published rating and a
    simulated point cannot drift apart about what a serve skill is.
    """
    k, _ = state.get_kappa(surface)
    lam, _ = state.get_lam(surface)
    return state.m_a + k, state.m_b + lam


def rate_field(snapshot: dict, field_uids: list[str], surface: str = REFERENCE_SURFACE,
               min_matches: int = 20) -> pd.DataFrame:
    """Serve and return ratings for a field, each against that field's average.

    `min_matches` governs who defines the average, not who gets a rating: a
    player with two matches of history should not move the baseline everyone
    else is measured against, but they still get a number, marked provisional.
    """
    rows = []
    for uid in field_uids:
        st = snapshot.get(uid)
        if st is None:
            continue
        e, f = effective(st, surface)
        rows.append(dict(uid=uid, e=e, f=f, n_matches=int(st.n_matches),
                         serve_sd=float(st.v_a) ** 0.5, return_sd=float(st.v_b) ** 0.5))
    if not rows:
        return pd.DataFrame(columns=["uid", "serve_rating", "return_rating", "hold_pct",
                                     "break_pct", "n_matches", "provisional"])
    t = pd.DataFrame(rows)

    established = t[t["n_matches"] >= min_matches]
    ref = established if len(established) >= 10 else t
    e_avg, f_avg = float(ref["e"].mean()), float(ref["f"].mean())

    # share of points won serving against the average returner, and returning
    # against the average server
    # hierarchical_model.sigmoid is scalar-only (it clamps at |x| > 35), so it
    # is applied element-wise rather than replaced with a vectorised copy: a
    # published rating and a simulated point should come out of the same
    # function, clamp included.
    t["serve_rating"] = [sigmoid(float(x)) for x in (t["e"] - f_avg)]
    t["return_rating"] = [1.0 - sigmoid(float(e_avg - x)) for x in t["f"]]
    # the model's own game recursion, so a published hold matches a simulated one
    t["hold_pct"] = [p_game(p) for p in t["serve_rating"]]
    t["break_pct"] = [1.0 - p_game(1.0 - p) for p in t["return_rating"]]
    t["provisional"] = t["n_matches"] < min_matches
    return t[["uid", "serve_rating", "return_rating", "hold_pct", "break_pct",
              "serve_sd", "return_sd", "n_matches", "provisional"]]


def serve_return_split_quality(rated: pd.DataFrame) -> float:
    """How far the two ratings are saying different things.

    Near zero means serve and return are separating genuinely distinct skills.
    High means they are largely restating overall quality, and presenting them
    as two independent numbers would overstate what the model knows. Measured
    at +0.202 for ATP and +0.655 for WTA on 2026-09-07.
    """
    est = rated[~rated["provisional"]]
    if len(est) < 10:
        return float("nan")
    return float(np.corrcoef(est["serve_rating"], est["return_rating"])[0, 1])
