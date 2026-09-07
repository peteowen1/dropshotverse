"""Anchors for making layoff and fatigue round-aware (D-26, D-27, D-28).

Written before looking at the new numbers, and deliberately without the model:
what needs pinning here is which value each feature takes in each round, which
is arithmetic. The size of the resulting probability move is a separate
question, answered by tune_scratch/screen_w3_frozen_layoff.py.

The module is loaded from an explicit path because several scripts in this repo
re-insert the main checkout's `scripts/` at the front of sys.path, so a plain
import in a worktree can silently test the file you did not change.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "dropshotdata" / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "stx_under_test", SCRIPTS / "simulate_tournament_xgb.py")
stx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stx)
print(f"loaded {stx.__file__}")
assert str(ROOT) in stx.__file__, "loaded the wrong copy of the module"

W = stx.IN_EVENT_WEEKS


def hdr(t):
    print("\n=== " + t + " ===")


def features(wsl, first_round, k):
    """What the two round-aware columns become, for one ordered pair, at round k.

    Mirrors build_round_pred_matrices' body. Kept here so the expected values
    are stated independently of the loop that produces them.
    """
    (wi, wj), (fi, fj) = wsl, first_round
    pi, pj = max(k - fi, 0), max(k - fj, 0)
    layoff = (W if pi > 0 else wi) - (W if pj > 0 else wj)
    return layoff, float(pi - pj)


hdr("1. round one is unchanged, which is where the old code was right")
lay, fat = features((4.0, 0.5), (0, 0), 0)
assert lay == 4.0 - 0.5, lay
assert fat == 0.0, fat
print(f"  layoff difference = the pre-tournament gap ({lay:+.2f} weeks), fatigue 0")

hdr("2. from round two on, both players have just played")
for k in (1, 2, 3, 4, 5, 6):
    lay, fat = features((4.0, 0.5), (0, 0), k)
    assert lay == 0.0, (k, lay)
    assert fat == 0.0, (k, fat)
print(f"  layoff difference 0 in rounds two to seven ({W:.3f} weeks each side)")
print("  fatigue difference 0 throughout, because neither player had a bye")

hdr("3. a four-week layoff no longer follows a player into the final")
old_style = 4.0 - 0.5
new_final = features((4.0, 0.5), (0, 0), 6)[0]
print(f"  final: was {old_style:+.2f} weeks, now {new_final:+.2f}")
assert old_style != new_final

hdr("4. byes: the seed has played nothing in the round after the bye")
# seed enters at round index 1, opponent at 0
lay, fat = features((3.0, 1.0), (1, 0), 1)
assert fat == -1.0, fat
assert lay == 3.0 - W, lay
print(f"  round two: fatigue {fat:+.1f} (opponent has a match in their legs), "
      f"layoff {lay:+.2f} (seed still carries theirs, opponent does not)")
lay, fat = features((3.0, 1.0), (1, 0), 2)
assert fat == -1.0, fat
assert lay == 0.0, lay
print(f"  round three: fatigue still {fat:+.1f}, layoff now {lay:+.1f} - both have played")
# and the bye must not touch round one, where nobody has played either way
lay0, fat0 = features((3.0, 1.0), (1, 0), 0)
assert fat0 == 0.0 and lay0 == 3.0 - 1.0
print("  round one untouched by declaring a bye")

hdr("5. the fatigue difference is bounded by one match, not by the round")
for k in range(0, 7):
    _, fat = features((2.0, 2.0), (1, 0), k)
    assert abs(fat) <= 1.0, (k, fat)
print("  a bye is worth exactly one match, in every later round")

hdr("6. it is symmetric")
for k in (0, 1, 3):
    a = features((4.0, 0.5), (0, 1), k)
    b = features((0.5, 4.0), (1, 0), k)
    assert np.allclose(a, [-x for x in b]), (k, a, b)
print("  swapping the two players negates both columns, in every round")

print("\nALL W3 ANCHORS PASSED")
