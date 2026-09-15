"""In-match win probability, scored for calibration at scale.

This finishes the one thing ticket 09 flagged as unfinished. That prototype
answered three of its four questions on 94 Fan Week matches and said of the
fourth: "Which is better calibrated in-match? Not testable with four matches -
this needs a large sample scored the way ticket 10 scored the pre-match models
(predicted-vs-realised by bucket). Flagging as unfinished rather than guessing."

There are now 79,917 points across 503 main-draw singles matches
(build_point_table.py), so it is testable.

TWO ARMS, BOTH STRICTLY LEAK-FREE
  A  fixed skill. Both players serve at the tour average. No player
     information at all, so this measures the RECURSION: given only the score,
     is the stated probability calibrated?
  B  online skill. Each player's serve-win probability is estimated from the
     points they have served SO FAR IN THIS MATCH, shrunk toward the tour
     average by a prior weight. Strictly prior information at every point - it
     never sees the point being predicted, let alone the result.

B is ticket 09's Mode B in spirit, and running it this way sidesteps what
blocked a proper comparison before: it needs no pre-match rating and no
player-id crosswalk, both of which the point tape lacks (it carries "P1"/"P2"
and names in prose, not uids).

WHAT ARM A CANNOT TELL YOU, stated up front
A gives both players equal skill, so it is structurally underconfident for
mismatched pairings and its tail buckets will look badly calibrated for a
reason that is nothing to do with the recursion. That is a property of the arm,
not a finding about the model - which is exactly why B exists alongside it.

The recursion itself is ticket09_live_modes.p_match_from_state, already
verified there against the fresh-match function at the (0,0,0,0) boundary.
Evaluation is at GAME granularity (sets, games, who serves next); the tape's
within-game point score is not fed in, matching what that function accepts.

ANCHORS, fixed before looking at output
  C1  At 0-0 with equal skills, arm A must state 0.500 +/- 0.005. If a fresh
      match is not a coin flip under symmetric assumptions, the recursion or
      the state extraction is wrong.
  C2  On the FINAL point of a match, the eventual winner's pre-point
      probability must exceed 0.5 in the large majority of matches (>90%).
      A model that does not know who is about to win a match at match point
      is broken in a way no aggregate metric would show.
  C3  Aggregate calibration: mean predicted probability within 0.02 of the
      realised win rate over all points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common
DD = _common.DD
sys.path.insert(0, str(DD / "scripts"))

from ticket09_live_modes import p_match_from_state  # noqa: E402

PTS = DD / "usopen_2026_points.parquet"
OUT = DD / "tune_scratch" / "inmatch_wp_calibration.parquet"
PRIOR_N = 30.0        # shrinkage weight for arm B, in "prior points served"


def log(m: str = "") -> None:
    print(m, flush=True)


_cache: dict[tuple, float] = {}


def wp(p1: float, p2: float, s1: int, s2: int, g1: int, g2: int,
       p1_serves: bool, best_of: int) -> float:
    key = (round(p1, 3), round(p2, 3), s1, s2, g1, g2, p1_serves, best_of)
    v = _cache.get(key)
    if v is None:
        v = p_match_from_state(key[0], key[1], s1, s2, g1, g2, p1_serves, best_of)
        _cache[key] = v
    return v


def main() -> None:
    d = pd.read_parquet(PTS)
    # D-56/D-57: 5-character match ids are qualifying/doubles/juniors, and
    # `event = match_id[0]` is not reliable for them - confirmed men's names
    # under a "2" (women's) 5-char id and vice versa. Unlike
    # screen_inmatch_seeded.py, this script has no other gate (it scores every
    # match with a MatchWinner, not just ones with a live-draw pre-match
    # number), so without this filter a mistagged match gets the WRONG
    # best_of and pollutes tour_avg for both arms.
    n_all = d["match_id"].nunique()
    d = d[d["match_id"].astype(str).str.len() == 4].copy()
    log(f"points: {len(d):,} across {d['match_id'].nunique():,} main-draw matches "
        f"({n_all - d['match_id'].nunique():,} qualifying/doubles/junior matches excluded)")

    # best_of by event: 1 = men's singles (5), 2 = women's singles (3)
    d["best_of"] = np.where(d["event"] == "1", 5, 3)

    # tour-average serve win probability, measured not assumed
    tour_avg = d.groupby("event").apply(
        lambda g: float((g["PointServer"] == g["PointWinner"]).mean()),
        include_groups=False)
    log(f"tour-average serve win probability: "
        + ", ".join(f"event {k}={v:.4f}" for k, v in tour_avg.items()))

    rows = []
    for mid, g in d.groupby("match_id", sort=False):
        g = g.sort_values("point_no")
        bo = int(g["best_of"].iloc[0])
        base = float(tour_avg[g["event"].iloc[0]])
        # label: who won the match, from the tape's own final MatchWinner
        mw = g["MatchWinner"].iloc[-1]
        if mw not in ("1", "2"):
            continue                       # retirement/unfinished - no label
        y1 = int(mw == "1")

        # online per-player serve tallies (arm B), updated AFTER each point
        srv_won = {1: 0.0, 2: 0.0}
        srv_pl = {1: 0.0, 2: 0.0}

        for r in g.itertuples():
            s1, s2 = int(r.P1SetsWon or 0), int(r.P2SetsWon or 0)
            g1, g2 = int(r.P1GamesWon or 0), int(r.P2GamesWon or 0)
            p1_serves = (r.PointServer == "1")
            # arm A - equal skill
            a = wp(base, base, s1, s2, g1, g2, p1_serves, bo)
            # arm B - online, shrunk
            pb1 = (srv_won[1] + PRIOR_N * base) / (srv_pl[1] + PRIOR_N)
            pb2 = (srv_won[2] + PRIOR_N * base) / (srv_pl[2] + PRIOR_N)
            b = wp(pb1, pb2, s1, s2, g1, g2, p1_serves, bo)
            rows.append((mid, r.point_no, s1, s2, g1, g2, a, b, y1))
            # update AFTER predicting - never sees this point
            sv = 1 if r.PointServer == "1" else 2
            srv_pl[sv] += 1.0
            srv_won[sv] += 1.0 if r.PointWinner == r.PointServer else 0.0

    res = pd.DataFrame(rows, columns=["match_id", "point_no", "s1", "s2",
                                      "g1", "g2", "wp_A", "wp_B", "y"])
    log(f"scored: {len(res):,} points over {res['match_id'].nunique():,} "
        f"matches with a decided winner")

    log("\n=== anchors ===")
    fresh = wp(float(tour_avg["1"]), float(tour_avg["1"]), 0, 0, 0, 0, True, 5)
    log(f"  C1 fresh match, equal skill: {fresh:.4f} "
        f"-> {'PASS' if abs(fresh - 0.5) < 0.005 else 'FAIL'}")

    last = res.sort_values("point_no").groupby("match_id").tail(1)
    right = float(((last["wp_A"] > 0.5) == (last["y"] == 1)).mean())
    rightB = float(((last["wp_B"] > 0.5) == (last["y"] == 1)).mean())
    log(f"  C2 winner favoured on the final point: A {right:.1%}, B {rightB:.1%} "
        f"-> {'PASS' if min(right, rightB) > 0.90 else 'FAIL'}")

    for arm in ("wp_A", "wp_B"):
        m, o = res[arm].mean(), res["y"].mean()
        log(f"  C3 aggregate {arm}: mean predicted {m:.4f} vs realised {o:.4f} "
            f"(diff {m - o:+.4f}) -> {'PASS' if abs(m - o) < 0.02 else 'FAIL'}")

    log("\n=== calibration by decile (predicted vs realised) ===")
    for arm in ("wp_A", "wp_B"):
        res["bin"] = pd.cut(res[arm], np.arange(0, 1.01, 0.1), include_lowest=True)
        t = res.groupby("bin", observed=True).agg(
            n=("y", "size"), pred=(arm, "mean"), obs=("y", "mean"))
        t["gap"] = t["pred"] - t["obs"]
        ece = float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())
        log(f"\n  {arm}   (ECE {ece:.4f})")
        log(t.to_string(float_format=lambda x: f"{x:.4f}"))

    log("\n=== log-loss / Brier ===")
    for arm in ("wp_A", "wp_B"):
        p = np.clip(res[arm].values, 1e-9, 1 - 1e-9)
        y = res["y"].values
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        br = float(((p - y) ** 2).mean())
        log(f"  {arm}: log-loss {ll:.5f}   Brier {br:.5f}")

    res.drop(columns=["bin"], errors="ignore").to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
