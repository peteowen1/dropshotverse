"""The TML bridge must give a player the archive's id when the two sources spell
the name differently (issue #45), and must not when the match is ambiguous.

Offline: builds a tiny archive in a temp dir and points DROPSHOT_DATA_DIR at it.
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

tmp = Path(tempfile.mkdtemp())
os.environ["DROPSHOT_DATA_DIR"] = str(tmp)
import bridge_tennismylife as b  # noqa: E402


def archive(tour, year, rows):
    d = tmp / "raw" / "github" / tour
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["winner_id", "winner_name", "loser_id", "loser_name"]).to_csv(
        d / f"{tour}_matches_{year}.csv", index=False)


# 2025 rows are recent enough to be trusted; the 2001 homonym is not.
archive("wta", 2025, [
    (213873, "Gabriela Andrea Knutson", 211817, "Elena Gabriela Ruse"),
    (216133, "Xin Yu Wang", 900001, "Ann Lee"),
    (900002, "Ann Marie Lee", 900003, "Pat Kim"),      # "Ann Lee" now reaches two ids
])
archive("wta", 2001, [(500001, "Pat Kim", 500002, "Old Timer")])

loose = b.build_loose_name_to_id("wta")


def resolve(name):
    return next((loose[k] for k in b.loose_keys(name) if k in loose), None)


fails = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        fails.append(label)


check("middle name dropped (Gabriela Knutson)", resolve("Gabriela Knutson"), 213873)
check("hyphen vs space (Elena-Gabriela Ruse)", resolve("Elena-Gabriela Ruse"), 211817)
check("words split differently (Xinyu Wang)", resolve("Xinyu Wang"), 216133)
check("accents ignored", resolve("Gabriéla Knutson"), 213873)
check("ambiguous key is dropped (Ann Lee -> 900001 or 900002)", resolve("Ann Lee"), None)
check("2001 homonym ignored, 2025 id wins (Pat Kim)", resolve("Pat Kim"), 900003)
check("unknown player stays unmatched", resolve("Nobody Atall"), None)
check("no tokens", b.loose_keys("..."), [])

if fails:
    raise SystemExit(f"FAILED: {fails}")
print("\nall bridge id checks passed")
