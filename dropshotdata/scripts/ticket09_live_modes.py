"""Ticket 09: frozen-skill (A) vs live-updating-skill (B) win-probability traces,
replayed against REAL point-by-point data captured from the 2026 US Open
(ticket 16) - genuine points, not simulated.

Scope note, stated honestly: the capture so far covers only Fan Week (mixed
doubles / qualifying, best-of-3). The main draw has not started. So a genuine
"comeback from two sets down" (needs best-of-5) and a mid-match retirement are
NOT available in this data yet - both scenarios are marked unavailable below
rather than faked. This is throwaway code per the ticket; nothing here is
built into the package.

Granularity: game-by-game, not point-by-point. Building a full within-game
partial-score recursion was judged not worth the risk in the time available -
every scenario the ticket asks to discriminate (upset, comeback, tiebreak) is
already visible at game granularity, since the recursion functions from
hierarchical_model.py extend cleanly to "win probability from an arbitrary
(sets, games) state" without needing point-level state.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from hierarchical_model import p_game, p_tiebreak, sigmoid, adf_update

MATCHES_DIR = Path(r"C:\dev\dropshotverse\dropshotdata\raw\usopen\2026\matches")
OUT_DIR = Path(r"C:\dev\dropshotverse\dropshotdata\ticket09")
OUT_DIR.mkdir(exist_ok=True)

PRIOR_P = 0.60          # neutral tour-average prior (ticket 11), used for BOTH
                        # modes at match start - neither model knows these
                        # specific (mostly unranked) players in advance
PRIOR_VAR = 0.30        # how much ONE game's evidence can move mode B's estimate


def load_tape(match_id: str):
    files = sorted((MATCHES_DIR / match_id).glob("history.*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def games_from_tape(pts):
    """Collapse a point tape into a per-game record: who served, who won it,
    and whether it went to deuce (informative for how 'surprising' it was)."""
    games = []
    cur = None
    for p in pts:
        key = (p.get("SetNo"), p.get("GameNo"))
        if key != cur:
            cur = key
            games.append({"set": p.get("SetNo"), "game": p.get("GameNo"),
                         "server": p.get("PointServer"), "points": []})
        if games:
            games[-1]["points"].append(p)
    out = []
    for g in games:
        pts_ = [p for p in g["points"] if p.get("GameWinner") not in (None, "0", 0)]
        if not pts_:
            continue
        winner = pts_[-1]["GameWinner"]
        server = g["points"][0].get("PointServer")
        out.append({"set": g["set"], "server": server, "winner": winner,
                   "n_points": len(g["points"])})
    return out


def set_is_decided(a: int, b: int) -> bool:
    """Is a standard set already won at this game score?

    6-4 and 6-0 (six games, two clear), 7-5, and 7-6 off the tiebreak. 6-5 is
    not: the set continues. Shared with usopen_live_draw so the two cannot
    drift apart about what a finished set looks like.

    Standard sets only. A format whose deciding set is settled by a
    super-tiebreak at some other game score would need its own rule; the 2026
    US Open uses a 6-6 tiebreak in every set including the decider, which is
    the only event this code runs on.
    """
    hi, lo = max(a, b), min(a, b)
    return (hi == 7 and lo <= 6) or (hi == 6 and hi - lo >= 2)


# A game's point score is written 0/15/30/40/AD; a TIEBREAK's is written as
# plain numbers, and both arrive as strings from the tape and from the feed.
# One table, shared, so the screen that measures and the simulator that serves
# cannot disagree about what "40" means.
_POINT_VALUE = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4, "ADV": 4}


def point_count(value, in_tiebreak: bool = False) -> int:
    """Points won, as a count, from whatever the source calls it."""
    s = str(value).strip().upper()
    if in_tiebreak:
        return int(s) if s.isdigit() else 0
    if s in _POINT_VALUE:
        return _POINT_VALUE[s]
    return int(s) if s.isdigit() else 0


def p_game_from_state(p: float, a: int, b: int) -> float:
    """P(the SERVER wins this game) from an arbitrary point score.

    `a` and `b` are points won by the server and the returner, as counts:
    0/15/30/40/AD is 0/1/2/3/4. The tape and the feed both give the tennis
    strings; the caller converts.

    p_game(p) is the same quantity from 0-0 and stays the entry point for a
    game not yet started. This exists because the recursion above it only ever
    knew games and sets, so 84.6% of the points on the 2026 tape were scored
    from a state that had thrown away a real, non-zero game score: 40-0 up on
    serve and 0-40 down returned the identical number.

    From 3-3 on this is the classic deuce chain, which has a closed form:
    the server wins from deuce with p^2 / (p^2 + (1-p)^2), because any split
    pair returns to deuce. Only the offset from deuce matters after that, so
    every score at or beyond 3-3 collapses to one of three states.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0
    if a >= 3 and b >= 3:
        pd = p * p / (p * p + (1 - p) * (1 - p))     # from deuce
        d = a - b
        if d == 0:
            return pd
        return p + (1 - p) * pd if d == 1 else p * pd
    return p * p_game_from_state(p, a + 1, b) + (1 - p) * p_game_from_state(p, a, b + 1)


def _tb_tail(pa: float, pb: float) -> float:
    """P(A eventually wins) from any TIE in a tiebreak, A's serve prob `pa`.

    From a tie the next two points are always one serve each. That is not an
    approximation: A serves point t iff (t-1) mod 4 is 0 or 3, and at a tie an
    even number of points has been played, so the next two always split. Order
    does not matter to the product, so A takes both with pa*(1-pb), B takes
    both with (1-pa)*pb, and anything else returns to a tie.
    """
    up, dn = pa * (1 - pb), (1 - pa) * pb
    tot = up + dn
    return 0.5 if tot <= 0 else up / tot


def p_tiebreak_from_state(pa: float, pb: float, a: int, b: int,
                          a_serves_next: bool, target: int = 7) -> float:
    """P(A wins the tiebreak) from an arbitrary point score.

    p_tiebreak(pa, pb) is the same quantity from 0-0 with A serving first, and
    is left alone as the entry point for a tiebreak not yet started.

    Who served point ONE is recovered from the rotation rather than asked for:
    A serves point t iff (t-1) mod 4 is 0 or 3, so with `t0 = a + b` points
    played, whether A serves next pins down whether A served first.
    """
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    t0 = a + b
    on_pattern = (t0 % 4) in (0, 3)          # would A serve point t0+1 if A served point 1
    a_first = (a_serves_next == on_pattern)
    tail = _tb_tail(pa, pb)
    if a == b and a >= target - 1:
        return tail

    states = {(a, b): 1.0}
    win_a = 0.0
    for t in range(t0 + 1, t0 + 40):
        if not states:
            break
        on = (t - 1) % 4 in (0, 3)
        a_serves = on if a_first else (not on)
        pt = pa if a_serves else (1 - pb)
        nxt: dict[tuple[int, int], float] = {}
        for (x, y), pr in states.items():
            for (nx, ny), w in (((x + 1, y), pt), ((x, y + 1), 1 - pt)):
                prob = pr * w
                if prob == 0.0:
                    continue
                if nx >= target and nx - ny >= 2:
                    win_a += prob
                elif ny >= target and ny - nx >= 2:
                    pass
                elif nx == ny and nx >= target - 1:
                    win_a += prob * tail          # exact geometric tail, no cap
                else:
                    nxt[(nx, ny)] = nxt.get((nx, ny), 0.0) + prob
        states = nxt
    return win_a + sum(states.values()) * tail


def p_set_from_state(p1: float, p2: float, games1: int, games2: int,
                     p1_serves_next: bool, points1: int = 0, points2: int = 0) -> float:
    """P(player1 wins the SET) from an arbitrary in-progress game score.

    `points1`/`points2` are the point score WITHIN the game now being played
    (or within the tiebreak, at 6-6). They default to 0-0, which is the game
    not yet started and reproduces this function's previous behaviour exactly -
    asserted in tune_scratch/test_inmatch_seeding.py, because every caller that
    does not pass them must keep getting the number it got before.

    Only the CURRENT game uses them. Every later game in the recursion starts
    at 0-0, which is what p_game and p_tiebreak already assume."""
    states = {(games1, games2): 1.0}
    final = 0.0
    for _ in range(20):
        nxt = {}
        for (a, b), pr in states.items():
            if set_is_decided(a, b):
                # This fires only for a score handed IN as the set in progress:
                # the transitions below score a finished set directly and never
                # put one back in `states`. It used to `continue`, which drops
                # the mass instead of awarding it, so the function returned 0.0
                # for every already-won set - 6-4, 7-5, 7-6, all of them.
                #
                # Reached whenever the feed's `totalSetsWon` lags its
                # `scores.sets` by one poll, and the shape is the worst kind:
                # at 5-4 the server sat at 0.946, and WINNING that set 6-4 took
                # them to 0.649. usopen_live_draw now refuses that state
                # outright; this makes the function correct for it regardless
                # of who calls it.
                final += pr if a > b else 0.0
                continue
            if a == 6 and b == 6:
                # A set at 6-6 IS the tiebreak: no thirteenth game is played,
                # so the set-winner is p_tiebreak and nothing else.
                #
                # This used to fall through to the game step below, which
                # played one more game and THEN applied the tiebreak on top,
                # giving pa*tb + (1-pa)*(1-tb) instead of tb. The error reached
                # 0.42, and it showed up as the same 6-6 position being worth
                # 0.687 or 0.443 depending only on who was marked as serving -
                # when a tiebreak's outcome here does not depend on that at all.
                #
                # It survived because the anchor anyone would reach for hides
                # it: with two equal players pa = 0.5 and tb = 0.5, so the
                # wrong expression returns exactly 0.5, which is also the right
                # answer. Asymmetric pairings are where it bites, which is
                # every pairing that matters.
                final += pr * (p_tiebreak_from_state(p1, p2, points1, points2,
                                                     p1_serves_next)
                               if (a, b) == (games1, games2)
                               else p_tiebreak(p1, p2))
                continue
            idx = (a + b) - (games1 + games2)
            a_serves = (idx % 2 == 0) == p1_serves_next
            if idx == 0 and (points1 or points2):
                # the game actually in progress, from the score it is at
                pa = (p_game_from_state(p1, points1, points2) if a_serves
                      else 1 - p_game_from_state(p2, points2, points1))
            else:
                pa = p_game(p1) if a_serves else 1 - p_game(p2)
            for winner, w in ((0, pa), (1, 1 - pa)):
                na, nb = (a + 1, b) if winner == 0 else (a, b + 1)
                prob = pr * w
                if prob == 0:
                    continue
                if (na >= 6 or nb >= 6) and abs(na - nb) >= 2 and max(na, nb) <= 7:
                    final += prob if na > nb else 0.0
                elif na == 7 or nb == 7:
                    tb = p_tiebreak(p1, p2)
                    final += prob * (tb if na == 7 else (1 - tb))
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    final += sum(states.values()) * 0.5
    return final


def p_match_from_state(p1: float, p2: float, sets1: int, sets2: int,
                       games1: int, games2: int, p1_serves_next: bool,
                       best_of: int, points1: int = 0, points2: int = 0) -> float:
    """P(player1 wins the match) from an arbitrary (sets, games) state.

    Bug caught by the (0,0,0,0)-boundary self-test: an earlier version gave
    every FUTURE (not-yet-reached) set a flat 50/50, instead of this pairing's
    real set-win probability - understating the leader's true edge by ~0.15-0.19
    in the boundary case. Fixed by computing q_current (the set in progress,
    from the games actually played) separately from q_fresh (any future set,
    symmetrised over serve order exactly like the original p_match)."""
    need = 2 if best_of == 3 else 3
    if sets1 == need:
        return 1.0
    if sets2 == need:
        return 0.0
    # Verified against p_match at the (0,0,0,0) boundary. Symmetrised over
    # serve order it now matches to machine precision.
    #
    # It used to sit ~0.4% off, and that residual was written up here as "a
    # real, understood convention difference, not a bug". It was a bug: the
    # 6-6 handling in p_set_from_state played a thirteenth game and then
    # applied the tiebreak on top of it. Explaining a discrepancy away is how
    # it survived, so the boundary agreement is now asserted in
    # tune_scratch/test_inmatch_seeding.py rather than described in a comment.
    #
    # The remaining asymmetry is genuine and is the reason both orders are
    # averaged above: p_match never knows who serves first, while a real
    # in-progress match does.
    q_current = p_set_from_state(p1, p2, games1, games2, p1_serves_next,
                                 points1, points2)
    q_fresh = 0.5 * (p_set_from_state(p1, p2, 0, 0, True)
                     + p_set_from_state(p1, p2, 0, 0, False))
    states = {(sets1, sets2): 1.0}
    final = 0.0
    for _ in range(need * 2):
        nxt = {}
        for (a, b), pr in states.items():
            qq = q_current if (a, b) == (sets1, sets2) else q_fresh
            for winner, w in ((0, qq), (1, 1 - qq)):
                na, nb = (a + 1, b) if winner == 0 else (a, b + 1)
                prob = pr * w
                if na == need:
                    final += prob
                elif nb == need:
                    pass
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    return final


def replay(match_id: str, label: str):
    pts = load_tape(match_id)
    if not pts:
        return None
    games = games_from_tape(pts)
    if not games:
        return None
    best_of = 3   # Fan Week matches captured so far are all best-of-3

    import math
    logit = lambda p: math.log(p / (1 - p))

    p1a, p2a = PRIOR_P, PRIOR_P        # mode A: frozen forever, probability scale
    # mode B state tracked on the LOGIT scale, matching adf_update's actual
    # interface. An earlier version fed raw probabilities (0.60) straight into
    # adf_update, which treats its inputs as log-odds - a units bug that made
    # every game, win or lose, push the estimate upward from an already-
    # inflated starting point. Caught before any plot was produced.
    d1b, d2b, v1b, v2b = logit(PRIOR_P), logit(PRIOR_P), PRIOR_VAR, PRIOR_VAR
    p1b, p2b = sigmoid(d1b), sigmoid(d2b)

    sets1 = sets2 = games1 = games2 = 0
    cur_set = games[0]["set"] if games else "1"   # match the raw tape's own type
    # (was the int 1: '1' != 1 is True in Python, so the very first game
    # falsely triggered a "set just ended" branch on an empty 0-0 tally,
    # starting every match at a false 0-1 deficit. Caught by cross-checking
    # the replay's own bookkeeping against the raw tape's P1SetsWon/P2SetsWon.)
    trace = []
    trace.append({"game_idx": 0, "set": 1, "score": "0-0 0-0",
                  "p_a": p_match_from_state(p1a, p2a, 0, 0, 0, 0, True, best_of),
                  "p_b": p_match_from_state(p1b, p2b, 0, 0, 0, 0, True, best_of)})

    for i, g in enumerate(games):
        if g["set"] != cur_set:
            # set just changed - roll games into sets won
            if games1 > games2:
                sets1 += 1
            else:
                sets2 += 1
            games1 = games2 = 0
            cur_set = g["set"]

        server_is_1 = str(g["server"]) == "1"
        p1_won = str(g["winner"]) == "1"

        # mode B: small within-match ADF nudge from this game's outcome,
        # correctly on the logit scale throughout
        if server_is_1:
            d1b, v1b, _, _ = adf_update(d1b, v1b, 0.0, 1.0, 1.0 if p1_won else 0.0, 1.0)
        else:
            d2b, v2b, _, _ = adf_update(d2b, v2b, 0.0, 1.0, 0.0 if p1_won else 1.0, 1.0)
        p1b, p2b = sigmoid(d1b), sigmoid(d2b)

        if p1_won:
            games1 += 1
        else:
            games2 += 1

        p1_serves_next = str(g["server"]) != "1"    # next game, server alternates
        pa = p_match_from_state(p1a, p2a, sets1, sets2, games1, games2, p1_serves_next, best_of)
        pb = p_match_from_state(p1b, p2b, sets1, sets2, games1, games2, p1_serves_next, best_of)
        trace.append({"game_idx": i + 1, "set": g["set"],
                      "score": f"sets {sets1}-{sets2} games {games1}-{games2}",
                      "p_a": pa, "p_b": pb, "n_points": g["n_points"]})

    # Verify the replay's own bookkeeping against the tape's ground truth
    # before trusting anything else - this is exactly the check that caught
    # the cur_set type-mismatch bug above, kept as a permanent assertion
    # rather than a one-off diagnostic.
    final_sets1 = sets1 + (1 if games1 > games2 else 0)
    final_sets2 = sets2 + (1 if games2 > games1 else 0)
    true_s1, true_s2 = int(pts[-1].get("P1SetsWon", -1)), int(pts[-1].get("P2SetsWon", -1))
    if (final_sets1, final_sets2) != (true_s1, true_s2):
        raise AssertionError(
            f"{match_id}: replay bookkeeping ({final_sets1}-{final_sets2}) "
            f"disagrees with tape ground truth ({true_s1}-{true_s2})")

    return {"match_id": match_id, "label": label, "trace": trace,
            "final_p1_a": p1a, "final_p2_a": p2a, "final_p1b": p1b, "final_p2b": p2b}


if __name__ == "__main__":
    # Labels verified against each tape's own P1SetsWon/P2SetsWon and per-set
    # game tallies before use - an earlier pass mislabeled 11104 as a
    # "reversal" when player 1 actually lost that match in straight sets, and
    # separately mislabeled 11105 as "dropped set 2" when it lost set 1.
    scenarios = {
        "11101": "straightforward win, player 1 (2-0)",
        "11103": "straightforward win, player 1 (2-0)",
        "11105": "genuine comeback: lost set 1, player 1 wins 2-1",
        "11112": "genuine comeback: lost set 1, player 1 wins 2-1",
    }
    results = []
    for mid, label in scenarios.items():
        r = replay(mid, label)
        if r:
            results.append(r)
            print(f"{mid} ({label}): {len(r['trace'])} game-updates, "
                  f"final A=({r['final_p1_a']:.3f},{r['final_p2_a']:.3f}) "
                  f"final B=({r['final_p1b']:.3f},{r['final_p2b']:.3f})")

    with open(OUT_DIR / "traces.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_DIR / 'traces.json'}")

    print("\n=== scenarios NOT available in current capture (Fan Week / qualifying only) ===")
    print("  - comeback from 2 sets down: needs best-of-5, main draw hasn't started")
    print("  - mid-match retirement/injury: none found in the 94 captured matches")
    print("  Both should be revisited once main-draw men's matches are captured.")
