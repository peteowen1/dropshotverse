"""The TML bridge must read the current season, not a hard-coded 2026 (the
year was once fixed in the URL, which would have frozen the ratings at the end
of 2026 while every check stayed green).

Offline: a fake reader stands in for the network.
"""
import datetime as dt
import os
import sys
import tempfile
import urllib.error
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))
os.environ["DROPSHOT_DATA_DIR"] = str(Path(tempfile.mkdtemp()))
import bridge_tennismylife as b  # noqa: E402

fails = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        fails.append(label)


def fake_reader(published):
    """Serves a one-row frame for each URL in `published`, 404 for the rest."""
    asked = []

    def read(url, low_memory=False):
        asked.append(url)
        if url not in published:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return pd.DataFrame({"url": [url]})
    return read, asked


A26, A27 = b.tml_url("atp", 2026), b.tml_url("atp", 2027)
W27 = b.tml_url("wta", 2027)
check("ATP URL pattern", A27, "https://stats.tennismylife.org/data/2027.csv")
check("WTA URL pattern", W27, "https://stats.tennismylife.org/data/2027_wta.csv")
check("2026 reads only 2026", b.tml_seasons(dt.date(2026, 10, 25)), [2026])
check("2027 reads 2026 and 2027", b.tml_seasons(dt.date(2027, 3, 1)), [2026, 2027])

read, asked = fake_reader({A26, A27})
got = b.read_tml("atp", today=dt.date(2027, 3, 1), reader=read)
check("2027 stacks both seasons", list(got["url"]), [A26, A27])

read, _ = fake_reader({A26})
got = b.read_tml("atp", today=dt.date(2027, 1, 2), reader=read)
check("January: unpublished new season skipped", list(got["url"]), [A26])

read, _ = fake_reader({A26})
try:
    b.read_tml("atp", today=dt.date(2027, 3, 1), reader=read)
    check("March: missing current season raises", "no error", "HTTPError")
except urllib.error.HTTPError:
    check("March: missing current season raises", "HTTPError", "HTTPError")

read, _ = fake_reader({A27})
try:
    b.read_tml("atp", today=dt.date(2027, 1, 2), reader=read)
    check("missing past season raises even in January", "no error", "HTTPError")
except urllib.error.HTTPError:
    check("missing past season raises even in January", "HTTPError", "HTTPError")

if fails:
    raise SystemExit(f"FAILED: {fails}")
print("\nall bridge season checks passed")
