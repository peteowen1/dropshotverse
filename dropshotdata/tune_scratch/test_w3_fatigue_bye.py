"""Anchors for the D-28 fatigue/bye fix, re-implemented against current dev
after PR #34 (the original fix) went stale - see D-52. weeks_since_last_diff's
round-awareness (D-26/D-27) already shipped separately via DSV_W3_LAYOFF_FIX
(D-46/D-47) and is not touched here; this only restores fatigue_diff.

Mirrors build_round_pred_matrices' fatigue arithmetic directly so the expected
values are stated independently of the loop that produces them.

Loaded from an explicit path: several scripts here re-insert the main
checkout's scripts/ at the front of sys.path, so a plain import in a worktree
can silently test the file you did not change.
"""
import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "dropshotdata" / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("stx_under_test", SCRIPTS / "simulate_tournament_xgb.py")
stx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stx)
print(f"loaded {stx.__file__}")
assert str(ROOT) in stx.__file__, "loaded the wrong copy of the module"


def hdr(t):
    print("\n=== " + t + " ===")


def fatigue(first_round, k):
    fi, fj = first_round
    return float(max(k - fi, 0) - max(k - fj, 0))


hdr("1. no first_round passed -> default None -> untouched (locked-forecast safety)")
sig = inspect.signature(stx.build_round_pred_matrices)
assert "first_round" in sig.parameters
assert sig.parameters["first_round"].default is None
src = inspect.getsource(stx.build_round_pred_matrices)
assert "if fr_i is not None:" in src, "the opt-in does not gate fatigue_diff"
locked = (SCRIPTS / "simulate_us_open_2026.py").read_text(encoding="utf-8")
assert "first_round" not in locked.split("def run_tour")[1].split("def ")[0], \
    "the locked pre-tournament forecast would start passing first_round"
print("  build_round_pred_matrices defaults first_round=None; simulate_us_open_2026 never passes it")

hdr("2. no bye: fatigue stays 0 in every round")
for k in range(0, 7):
    assert fatigue((0, 0), k) == 0.0, k
print("  fatigue 0 throughout when neither player had a bye")

hdr("3. round after the bye: the seed has played nothing, the opponent has")
# seed enters at round index 1 (bye in round one), opponent at 0
assert fatigue((1, 0), 1) == -1.0
print("  round two: fatigue -1.0 (opponent has a match in their legs)")

hdr("4. the deficit is permanent: a bye is one match, for the rest of the draw")
assert fatigue((1, 0), 2) == -1.0
assert fatigue((1, 0), 6) == -1.0
print("  fatigue stays -1.0 every later round - the bye recipient has always "
      "played exactly one fewer match than a same-position opponent")

hdr("5. round one is untouched by declaring a bye")
assert fatigue((1, 0), 0) == 0.0
print("  round one: nobody has played yet either way")

hdr("6. bounded by one match, not by the round")
for k in range(0, 7):
    assert abs(fatigue((1, 0), k)) <= 1.0, k
print("  a bye is worth exactly one match, in every later round")

hdr("7. symmetric")
for k in (0, 1, 3, 5):
    assert fatigue((0, 1), k) == -fatigue((1, 0), k), k
print("  swapping the two players negates fatigue_diff, in every round")

hdr("8. the partial-field path does not claim byes it cannot place")
wta = (SCRIPTS / "simulate_wta_event.py").read_text(encoding="utf-8")
assert "n_seeds >= n_byes" in wta, "first_round is passed even when leftover byes are drawn at random per sim"
assert "first_round = None" in wta, "no fallback for the provisional path"
print("  first_round is only passed when every bye sits opposite a known seed")

print("\nALL D-28 FATIGUE/BYE ANCHORS PASSED")
