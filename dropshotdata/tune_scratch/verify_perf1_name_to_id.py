"""Ticket 40 fix 1: the vectorised build_name_to_id must equal the iterrows()
original, dict-for-dict, for both tours. OLD is imported from the live working
tree (dev), NEW from the perf worktree. Same CSVs on disk for both."""
import importlib.util
import sys
import time
from pathlib import Path

OLD = Path(r"C:\dev\dropshotverse\dropshotdata\scripts\bridge_tennismylife.py")
NEW = Path(sys.argv[1]) / "dropshotdata" / "scripts" / "bridge_tennismylife.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old, new = load("bridge_old", OLD), load("bridge_new", NEW)
for tour in ("atp", "wta"):
    t0 = time.time(); a = old.build_name_to_id(tour); ta = time.time() - t0
    t0 = time.time(); b = new.build_name_to_id(tour); tb = time.time() - t0
    same = a == b
    print(f"{tour}: old {len(a):,} keys in {ta:.1f}s | new {len(b):,} keys in {tb:.1f}s | "
          f"identical={same}")
    if not same:
        ka, kb = set(a), set(b)
        print("  only-old:", sorted(ka - kb)[:5], " only-new:", sorted(kb - ka)[:5])
        diff = [k for k in ka & kb if a[k] != b[k]]
        print("  value mismatches:", len(diff), diff[:5])
        sys.exit(1)
print("PERF1_IDENTICAL")
