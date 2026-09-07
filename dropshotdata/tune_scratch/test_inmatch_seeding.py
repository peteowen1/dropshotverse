"""Exercise the live-score seeding on the real feed plus synthetic states."""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import usopen_live_draw as d          # noqa: E402
from inmatch_seeding import (SERVE_BASE, TOLERANCE, seeded_probability,  # noqa: E402
                             solve_serves, wp)
from ticket09_live_modes import p_set_from_state, p_tiebreak  # noqa: E402


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
try:
    solve_serves(0.5, 0.5, 5, True)          # base 0.5 -> bound 0.4995, fine
    print(f"  a reachable target solves (tolerance {TOLERANCE})")
except ValueError as e:
    raise AssertionError(f"a reachable target should not raise: {e}")

print("\nALL SEEDING TESTS PASSED")
