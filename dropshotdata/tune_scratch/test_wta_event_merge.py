"""simulate_wta_event's merge must fail closed.

Only "the file does not exist yet" may fall back to writing one event alone.
Any other read failure (403, timeout, malformed file) has to raise, because
the fallback overwrites the published tables and wipes every other event.
Offline: the published-copy reads are stubbed.
"""
import sys
import tempfile
import urllib.error
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import simulate_wta_event as s  # noqa: E402

URL = "https://example.invalid/x"
missing = Path(tempfile.gettempdir()) / "dropshot_no_such_file_xyz.parquet"
assert not missing.exists()


def http_error(code):
    def f(*a, **k):
        raise urllib.error.HTTPError(URL, code, "stub", {}, None)
    return f


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


orig_read, orig_fetch = pd.read_parquet, s._fetch_text

# 404 on the published copy = first run: fall back.
s.pd.read_parquet = http_error(404)
assert s.read_existing_parquet(missing, URL) is None
s._fetch_text = http_error(404)
assert s.read_existing_draw(missing.with_suffix(".json"), URL) is None
print("404 -> None (first run): ok")

# 403 (what R2 sends urllib's default agent), 500, timeout: must raise.
for code in (403, 500):
    s.pd.read_parquet = http_error(code)
    assert raises(lambda: s.read_existing_parquet(missing, URL), urllib.error.HTTPError), code
    s._fetch_text = http_error(code)
    assert raises(lambda: s.read_existing_draw(missing.with_suffix(".json"), URL), urllib.error.HTTPError), code


def timeout(*a, **k):
    raise TimeoutError("stub")


s.pd.read_parquet = timeout
assert raises(lambda: s.read_existing_parquet(missing, URL), TimeoutError)
s._fetch_text = timeout
assert raises(lambda: s.read_existing_draw(missing.with_suffix(".json"), URL), TimeoutError)
print("403 / 500 / timeout -> raise: ok")

s.pd.read_parquet, s._fetch_text = orig_read, orig_fetch

# A local copy wins and is read as-is; a malformed local draw raises.
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "t.parquet"
    pd.DataFrame({"tournament_id": ["a", "b"]}).to_parquet(p, index=False)
    assert len(s.read_existing_parquet(p, URL)) == 2
    j = Path(d) / "draw.json"
    j.write_text("{not json", encoding="utf-8")
    assert raises(lambda: s.read_existing_draw(j, URL), ValueError)
print("local copy read; malformed local draw -> raise: ok")

print("\nALL WTA EVENT MERGE TESTS PASSED")
