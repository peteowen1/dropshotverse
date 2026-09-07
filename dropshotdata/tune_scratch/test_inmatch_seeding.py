"""Exercise the live-score seeding on the real feed plus synthetic states."""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import usopen_live_draw as d          # noqa: E402
from inmatch_seeding import (SERVE_BASE, TOLERANCE, seeded_probability,  # noqa: E402
                             solve_serves, wp)
from ticket09_live_modes import (p_game_from_state, p_match_from_state,  # noqa: E402
                                 p_set_from_state, p_tiebreak,
                                 p_tiebreak_from_state, point_count,
                                 set_is_decided)
from hierarchical_model import p_game  # noqa: E402


def hdr(t):
    print("\n=== " + t + " ===")


hdr("1. live_state on the real feed")
data = d.fetch_draw("MS")
ms = data["matches"]
live = [m for m in ms if d.live_state(m) is not None]
print(f"matches in the feed: {len(ms)}; under way right now: {len(live)}")
for m in live:
    print(" ", m["match_id"], d.live_state(m))
done = [m for m in ms if m.get("winner") in ("1", "2")]
assert all(d.live_state(m) is None for m in done), "a decided match was called live"
print(f"all {len(done)} decided matches correctly return None")

hdr("2. live_state on a synthetic in-progress match")
import copy
m = copy.deepcopy(done[-1])          # R4, Paul vs Alcaraz, 3-0 to Alcaraz
m["winner"] = None
m["status"] = "In Progress"
m["team1"]["totalSetsWon"], m["team2"]["totalSetsWon"] = 0, 1
m["team1"]["won"] = m["team2"]["won"] = False
m["team1"]["serve"], m["team2"]["serve"] = True, False
# one completed set (4-6) plus a current set at 2-3
m["scores"]["sets"] = [
    [{"score": 4}, {"score": 6}],
    [{"score": 2}, {"score": 3}],
]
st = d.live_state(m)
print("state:", st)
assert st["sets1"] == 0 and st["sets2"] == 1, st
assert st["games1"] == 2 and st["games2"] == 3, st
assert st["p1_serves"] is True
# 4+6+2+3 = 15 games played, odd, so the man serving now did NOT serve first
assert st["p1_served_first"] is False, st
print("sets, games, server and served-first parity all read correctly")

hdr("3. served-first parity, both ways")
for played, serves_now, expect in [(1, True, False), (2, False, False),
                                   (3, False, True), (4, True, True)]:
    mm = copy.deepcopy(m)
    mm["team1"]["serve"], mm["team2"]["serve"] = serves_now, not serves_now
    mm["team1"]["totalSetsWon"] = mm["team2"]["totalSetsWon"] = 0
    mm["scores"]["sets"] = [[{"score": played}, {"score": 0}]]
    got = d.live_state(mm)
    assert got["p1_served_first"] is expect, (played, serves_now, got)
    print(f"  {played} games played, p1 serves now={serves_now} -> served first={expect} ok")

hdr("3b. a match with no score yet is deliberately NOT live")
mm = copy.deepcopy(m)
mm["team1"]["totalSetsWon"] = mm["team2"]["totalSetsWon"] = 0
mm["scores"]["sets"] = [[{"score": 0}, {"score": 0}]]
assert d.live_state(mm) is None
print("0-0 returns None: nothing to condition on, and seeding it would restate")
print("the pre-match number anyway, so the caller can treat it as no change")

hdr("4. the solve reproduces the pre-match number at the opening state")
worst = 0.0
for tour, bo in (("atp", 5), ("wta", 3)):
    base = SERVE_BASE[tour]
    for p_pre in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        for first in (True, False):
            p1, p2 = solve_serves(p_pre, base, bo, first)
            back = wp(p1, p2, 0, 0, 0, 0, first, bo)
            worst = max(worst, abs(back - p_pre))
            assert abs(p1 + p2 - 2 * base) < 1e-9, "serve sum not held"
print(f"max |solved opening state - pre-match| = {worst:.5f} "
      f"{'PASS' if worst < 0.005 else 'FAIL'} (anchor C2)")
assert worst < 0.005

hdr("5. an underdog leading must gain, a favourite trailing must lose")
for tour, bo in (("atp", 5), ("wta", 3)):
    need = 3 if bo == 5 else 2
    p_pre = 0.25
    ahead = seeded_probability(p_pre, dict(sets1=need - 1, sets2=0, games1=0, games2=0,
                                           p1_serves=True, p1_served_first=True), tour, bo)
    behind = seeded_probability(p_pre, dict(sets1=0, sets2=need - 1, games1=0, games2=0,
                                            p1_serves=True, p1_served_first=True), tour, bo)
    print(f"  [{tour}] pre-match {p_pre:.0%}: one set from winning -> {ahead:.1%}, "
          f"one set from losing -> {behind:.1%}")
    assert ahead > p_pre, "leading did not help"
    assert behind < p_pre, "trailing did not hurt"
    assert ahead > 0.5, "a set from victory should be favourite"

hdr("6. a decided state is absorbing")
p = seeded_probability(0.05, dict(sets1=3, sets2=0, games1=0, games2=0,
                                  p1_serves=True, p1_served_first=True), "atp", 5)
print(f"  3 sets to 0 in a best-of-5 -> {p:.6f}")
assert p == 1.0

hdr("7. a set at 6-6 IS the tiebreak, and nothing else")
worst = 0.0
for p1, p2 in ((0.61, 0.61), (0.70, 0.55), (0.55, 0.70), (0.80, 0.45), (0.90, 0.35)):
    for serves in (True, False):
        got, want = p_set_from_state(p1, p2, 6, 6, serves), p_tiebreak(p1, p2)
        worst = max(worst, abs(got - want))
print(f"  worst |p_set(6,6) - p_tiebreak| over 5 pairings x both servers = {worst:.6f}")
assert worst < 1e-9, "6-6 is not being decided by the tiebreak alone"
# the regression that hid the old bug: equal players give 0.5 either way, so
# the anchor anyone reaches for first cannot see it. Asymmetric pairings can.
assert abs(p_set_from_state(0.80, 0.45, 6, 6, True)
           - p_set_from_state(0.80, 0.45, 6, 6, False)) < 1e-9, \
    "6-6 still depends on who is marked as serving"
print("  and it no longer depends on who is marked as serving")

hdr("8. a malformed feed payload reports nothing, not a guess")
base = copy.deepcopy(done[-1])
base["winner"] = None
base["team1"]["serve"], base["team2"]["serve"] = True, False
cases = {
    "sets won but scores.sets missing": dict(s1=1, s2=0, sets=[]),
    "history truncated to the current set": dict(s1=1, s2=1, sets=[[{"score": 3}, {"score": 2}]]),
    "one side missing from a set entry": dict(s1=0, s2=0, sets=[[{"score": 4}]]),
    "a score field absent": dict(s1=0, s2=0, sets=[[{"score": 4}, {}]]),
    "more entries than sets plus one": dict(s1=0, s2=0, sets=[[{"score": 6}, {"score": 4}],
                                                              [{"score": 6}, {"score": 4}]]),
}
for label, c in cases.items():
    mm = copy.deepcopy(base)
    mm["team1"]["totalSetsWon"], mm["team2"]["totalSetsWon"] = c["s1"], c["s2"]
    mm["scores"]["sets"] = c["sets"]
    assert d.live_state(mm) is None, f"{label}: returned a state instead of None"
    print(f"  {label}: None")

hdr("9. serve must name exactly one player")
for s1, s2 in ((False, False), (True, True)):
    mm = copy.deepcopy(base)
    mm["team1"]["totalSetsWon"], mm["team2"]["totalSetsWon"] = 0, 0
    mm["scores"]["sets"] = [[{"score": 3}, {"score": 2}]]
    mm["team1"]["serve"], mm["team2"]["serve"] = s1, s2
    assert d.live_state(mm) is None, f"serve={s1}/{s2} was accepted"
    print(f"  serve {s1}/{s2}: None")

hdr("10. inside a tiebreak, who served first is not claimed")
mm = copy.deepcopy(base)
mm["team1"]["totalSetsWon"], mm["team2"]["totalSetsWon"] = 0, 0
mm["scores"]["sets"] = [[{"score": 6}, {"score": 6}]]
st = d.live_state(mm)
print("  state:", st)
assert st["in_tiebreak"] is True
assert st["p1_served_first"] is None, "a point server was mistaken for a game server"
# and the solve must still work, by symmetrising over both orders
for tour, bo in (("atp", 5), ("wta", 3)):
    v = seeded_probability(0.65, st, tour, bo)
    print(f"  [{tour}] pre-match 65% at 6-6 in set one -> {v:.1%}")
    assert 0.0 < v < 1.0
print("  solved by symmetrising, which is what the pre-match model assumes too")

hdr("11. an unreachable solve raises rather than saturating")
# A reachable target must not raise.
try:
    solve_serves(0.5, 0.5, 5, True)
    print(f"  a reachable target solves (tolerance {TOLERANCE})")
except ValueError as e:
    raise AssertionError(f"a reachable target should not raise: {e}")
# And an unreachable one must. The bound is min(base, 1-base)*0.999, so a base
# pinned near 1 leaves almost no room to move the difference and the solver
# cannot get anywhere near a target on the far side. Without this case the
# residual check could be deleted outright and this test would still pass -
# which is what it did before.
raised = False
try:
    solve_serves(0.01, 0.999, 5, True)
except ValueError as e:
    raised = True
    print(f"  an unreachable target raises: {str(e)[:72]}...")
assert raised, "solve_serves saturated at its bound instead of raising"

hdr("12. an already decided set score is scored, not dropped")
for a, b, want in ((6, 4, True), (6, 0, True), (7, 5, True), (7, 6, True),
                   (4, 6, True), (6, 5, False), (5, 5, False), (6, 6, False)):
    assert set_is_decided(a, b) is want, (a, b)
for a, b in ((6, 4), (7, 6), (6, 0), (7, 5), (4, 6), (6, 7)):
    got, want = p_set_from_state(0.65, 0.55, a, b, True), (1.0 if a > b else 0.0)
    assert abs(got - want) < 1e-12, f"{a}-{b} scored {got}, should be {want}"
print("  every decided score returns 1.0 or 0.0; it used to return 0.0 for all of them")
# The shape that made this dangerous: winning a set made you look WORSE.
close = p_match_from_state(0.65, 0.55, 1, 1, 5, 4, True, 5)
won = p_match_from_state(0.65, 0.55, 1, 1, 6, 4, True, 5)
print(f"  1 set all: 5-4 -> {close:.4f}, then winning that set 6-4 -> {won:.4f}")
assert won > close, "winning a set still lowers the match probability"

hdr("13. probability mass is conserved from every undecided start")
for g1, g2 in ((0, 0), (5, 5), (6, 6), (6, 5), (3, 1), (0, 5)):
    t = (p_set_from_state(0.65, 0.55, g1, g2, True)
         + p_set_from_state(0.55, 0.65, g2, g1, False))
    assert abs(t - 1.0) < 1e-9, f"{g1}-{g2} sums to {t}"
print("  p + mirror == 1 from 0-0, 5-5, 6-6, 6-5, 3-1 and 0-5")

hdr("14. a feed that lags itself is refused, not scored")
mm = copy.deepcopy(base)
mm["team1"]["totalSetsWon"], mm["team2"]["totalSetsWon"] = 0, 0
mm["scores"]["sets"] = [[{"score": 6}, {"score": 4}]]
assert d.live_state(mm) is None, "a finished set was accepted as in progress"
print("  sets won 0-0 while the set in progress reads 6-4: None")

hdr("15. the recursion agrees with the pre-match model at the opening state")
import hierarchical_model as hm
worst = 0.0
for tour, bo in (("atp", 5), ("wta", 3)):
    b = SERVE_BASE[tour]
    for delta in (0.0, 0.02, 0.05, 0.10, 0.20):
        pm = hm.p_match(b + delta, b - delta, bo)
        sym = 0.5 * (p_match_from_state(b + delta, b - delta, 0, 0, 0, 0, True, bo)
                     + p_match_from_state(b + delta, b - delta, 0, 0, 0, 0, False, bo))
        worst = max(worst, abs(sym - pm))
print(f"  worst gap, symmetrised over serve order: {worst:.2e}")
# This was ~0.4% and the code described it as a convention difference rather
# than a defect. It was the 6-6 bug. Asserted so it cannot be explained away.
assert worst < 1e-9, "the recursion and p_match have drifted apart again"

hdr("16. the point tape's set-ending row means 'set won', not 'set in progress'")
import pandas as pd
import _common
tape = _common.DD / "usopen_2026_points.parquet"
if tape.exists():
    t = pd.read_parquet(tape, columns=["match_id", "point_no", "P1SetsWon", "P2SetsWon",
                                       "P1GamesWon", "P2GamesWon", "SetWinner"])
    t = t[t["match_id"] == t["match_id"].iloc[0]].sort_values("point_no").reset_index(drop=True)
    a = t["P1GamesWon"].fillna(0).astype(int)
    b = t["P2GamesWon"].fillna(0).astype(int)
    ends = [i for i in range(len(t)) if set_is_decided(a[i], b[i])]
    assert ends, "no set-ending row found in the first match"
    i = ends[0]
    row, nxt = t.loc[i], t.loc[i + 1]
    print(f"  point {row['point_no']}: sets {int(row['P1SetsWon'])}-{int(row['P2SetsWon'])}, "
          f"games {a[i]}-{b[i]}, SetWinner={int(row['SetWinner'])}")
    print(f"  point {nxt['point_no']}: sets {int(nxt['P1SetsWon'])}-{int(nxt['P2SetsWon'])}, "
          f"games {int(nxt['P1GamesWon'])}-{int(nxt['P2GamesWon'])}")
    # the set is ALREADY in the set count on the row that shows 6-4, so those
    # games are a leftover. Scoring them as a set in progress counts it twice.
    assert int(row["SetWinner"]) in (1, 2), "the set-ending row should name a set winner"
    assert (int(row["P1SetsWon"]) + int(row["P2SetsWon"])
            == int(nxt["P1SetsWon"]) + int(nxt["P2SetsWon"])), \
        "the set count changes AFTER the leftover row, so it is not a leftover"
    assert int(nxt["P1GamesWon"]) == 0 and int(nxt["P2GamesWon"]) == 0
    print("  so the screen zeroes those games; counting them would score the set twice")
else:
    print(f"  skipped: {tape} not present")

hdr("17. the point-level primitives reduce to the game-level ones at 0-0")
w = max(abs(p_game_from_state(x, 0, 0) - p_game(x))
        for x in (0.35, 0.5, 0.55, 0.61, 0.75, 0.9))
print(f"  p_game_from_state(p,0,0) vs p_game(p): worst {w:.2e}")
assert w < 1e-12
w = max(abs(p_tiebreak_from_state(pa, pb, 0, 0, True) - p_tiebreak(pa, pb))
        for pa in (0.5, 0.61, 0.8) for pb in (0.45, 0.57, 0.7))
print(f"  p_tiebreak_from_state(...,0,0,True) vs p_tiebreak: worst {w:.2e}")
assert w < 1e-9
# Every existing caller passes no point score, so all of them must be untouched.
w = 0.0
for g1, g2 in ((0, 0), (3, 2), (5, 5), (6, 6), (6, 5)):
    w = max(w, abs(p_set_from_state(0.65, 0.55, g1, g2, True)
                   - p_set_from_state(0.65, 0.55, g1, g2, True, 0, 0)))
    w = max(w, abs(p_match_from_state(0.65, 0.55, 1, 1, g1, g2, True, 5)
                   - p_match_from_state(0.65, 0.55, 1, 1, g1, g2, True, 5, 0, 0)))
print(f"  set and match with points defaulted: worst {w:.2e}")
assert w == 0.0, "adding the point score changed a caller that does not pass one"

hdr("18. the point score moves the number, in the right direction")
order = [p_game_from_state(0.65, a, b) for a, b in ((3, 0), (2, 0), (0, 0), (0, 2), (0, 3))]
print("  P(hold) at 40-0, 30-0, 0-0, 0-30, 0-40: "
      + ", ".join(f"{v:.4f}" for v in order))
assert order == sorted(order, reverse=True), "hold probability is not monotone in the score"
assert abs(p_game_from_state(0.5, 3, 3) - 0.5) < 1e-12, "deuce at p=0.5 must be 0.5"
assert abs(p_game_from_state(0.5, 0, 0) - 0.5) < 1e-12
# advantage in must beat deuce must beat advantage out
adv_in, deuce, adv_out = (p_game_from_state(0.65, 4, 3), p_game_from_state(0.65, 3, 3),
                          p_game_from_state(0.65, 3, 4))
print(f"  AD in {adv_in:.4f} > deuce {deuce:.4f} > AD out {adv_out:.4f}")
assert adv_in > deuce > adv_out

hdr("19. what a break point is worth, which the model could not see before")
lo = p_match_from_state(0.65, 0.55, 2, 2, 5, 5, True, 5, 0, 3)
hi = p_match_from_state(0.65, 0.55, 2, 2, 5, 5, True, 5, 3, 0)
mid_ = p_match_from_state(0.65, 0.55, 2, 2, 5, 5, True, 5, 0, 0)
print(f"  5-5 in a deciding set, serving: 0-40 {lo:.4f}, 0-0 {mid_:.4f}, 40-0 {hi:.4f}")
print(f"  spread {hi - lo:.4f}, all of which used to be one number")
assert hi > mid_ > lo

hdr("20. score strings convert the same way everywhere")
for v, want in (("0", 0), ("15", 1), ("30", 2), ("40", 3), ("AD", 4), ("A", 4)):
    assert point_count(v) == want, (v, point_count(v))
for v, want in (("0", 0), ("5", 5), ("11", 11)):
    assert point_count(v, in_tiebreak=True) == want, (v, point_count(v, True))
assert point_count(None) == 0 and point_count("") == 0
print("  0/15/30/40/AD in a game, plain numbers in a tiebreak, 0 for anything absent")

print("\nALL SEEDING TESTS PASSED")
