"""draw_uids_by_tour must add late entrants from the live projection (issue #45).

The locked pre-tournament JSON lacks anyone who entered after it was frozen
(Yunchaokete Bu), so he got no ratings row. Offline: temp files, no archive.
Needs the build_blog_data imports (xgboost etc.), so it is not in tests.yml.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))
import build_blog_data as b  # noqa: E402

tmp = Path(tempfile.mkdtemp())
locked = tmp / "locked.json"
locked.write_text(json.dumps({"atp": {"probs": {"atp:1": {}, "atp:2": {}}},
                              "wta": {"probs": {"wta:1": {}}}}), encoding="utf-8")
live = tmp / "live.parquet"
pd.DataFrame({"tour": ["atp", "atp", "wta"], "player_id": ["atp:2", "atp:207352", "wta:213873"]}).to_parquet(live)
b.PREDICTIONS_JSON = locked

got = b.draw_uids_by_tour(live_path=live)
assert got == {"atp": {"atp:1", "atp:2", "atp:207352"}, "wta": {"wta:1", "wta:213873"}}, got
# No live file: exactly the locked field, as before.
assert b.draw_uids_by_tour(live_path=tmp / "absent.parquet") == {"atp": {"atp:1", "atp:2"}, "wta": {"wta:1"}}
print("OK")
