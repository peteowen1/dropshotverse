"""Ticket 40 fixes 2-5: every changed function must produce exactly what the
old one did, on the live path's real inputs.

OLD = C:\\dev\\dropshotverse (dev, what the laptop task runs). NEW = the perf
worktree passed as argv[1]. Shared, unpatched modules (hierarchical_model,
bridge after fix 1, simulate_tournament, ...) resolve from OLD via sys.path;
only the four patched modules are loaded from NEW by file path.

Checks:
  3. usopen_live_draw.build_name_to_uid      dict equality
  5. train_best_model.build_playstyle_features  frame equality, exact
  2. simulate_tournament_updating.build_snapshots  every PlayerState field,
     old vs new-miss vs new-hit (cache written then read)
  4. xgb_bt_rating.build_pairwise_features_base  frame equality, exact,
     plus the pair lists, both tours
Prints timings for old and new on each.
"""
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path(r"C:\dev\dropshotverse\dropshotdata\scripts")
NEW = Path(sys.argv[1]) / "dropshotdata" / "scripts"
sys.path.insert(0, str(OLD))

import bridge_tennismylife as bridge          # noqa: E402  (fix 1 already merged into OLD? no: OLD is dev; identical output either way)
import hierarchical_model as hm               # noqa: E402
import simulate_tournament_updating as stu_old  # noqa: E402
import usopen_live_draw as draw_old           # noqa: E402
import train_best_model as tbm_old            # noqa: E402
import xgb_bt_rating as xbr_old               # noqa: E402
import simulate_usopen_live as live           # noqa: E402  (DATE0, SURFACE, TOUR constants)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


stu_new = load("stu_new", NEW / "simulate_tournament_updating.py")
draw_new = load("draw_new", NEW / "usopen_live_draw.py")
tbm_new = load("tbm_new", NEW / "train_best_model.py")
xbr_new = load("xbr_new", NEW / "xgb_bt_rating.py")


def log(m=""):
    print(m, flush=True)


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0


failures = []

# ---------------------------------------------------------------- inputs
log("=== inputs: bridged, prepared archive ===")
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
log(f"  {len(df):,} rows {df['date'].min().date()}..{df['date'].max().date()}")

# ---------------------------------------------------------------- 3. name -> uid
log("\n=== 3. build_name_to_uid ===")
a, ta = timed(draw_old.build_name_to_uid, df)
b, tb = timed(draw_new.build_name_to_uid, df)
log(f"  old {len(a):,} keys {ta:.1f}s | new {len(b):,} keys {tb:.1f}s | identical={a == b}")
if a != b:
    ka, kb = set(a), set(b)
    diff = [k for k in ka & kb if a[k] != b[k]]
    log(f"  only-old {sorted(ka-kb)[:5]} only-new {sorted(kb-ka)[:5]} value-mismatch {len(diff)} {diff[:5]}")
    failures.append("name_to_uid")
n2u = a

# ---------------------------------------------------------------- 5. playstyle
log("\n=== 5. build_playstyle_features ===")
pa, ta = timed(tbm_old.build_playstyle_features)
pb, tb = timed(tbm_new.build_playstyle_features)
log(f"  old {pa.shape} {ta:.1f}s | new {pb.shape} {tb:.1f}s")
try:
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    log("  identical=True (exact)")
except AssertionError as e:
    # report the size of the difference before failing: an ewm kernel
    # difference at 1e-16 is a different finding from a wrong shift
    num = [c for c in pa.columns if pa[c].dtype.kind == "f"]
    mx = max(float(np.nanmax(np.abs(pa[c].values - pb[c].values))) for c in num)
    nan_mismatch = int(sum((pa[c].isna() != pb[c].isna()).sum() for c in num))
    log(f"  NOT exact: max |delta| {mx:.3e}, NaN-pattern mismatches {nan_mismatch}")
    log("  " + str(e).splitlines()[0])
    failures.append("playstyle")
playstyle_full = pa

# ---------------------------------------------------------------- draws, metas
log("\n=== inputs: live draws ===")
draws, metas = {}, {}
for tour in ("atp", "wta"):
    d = draw_old.load_live_draw(tour, n2u)
    problems = draw_old.verify(d)
    assert not problems, problems
    draws[tour] = d
    metas[f"{tour}:2026-usopen"] = {"date0": live.DATE0, "field_uids": sorted(set(d["bracket"]))}
    log(f"  [{tour}] bracket {len(d['bracket'])}, results {len(d['results'])}")

# ---------------------------------------------------------------- 2. snapshots
log("\n=== 2. build_snapshots (old / new miss / new hit) ===")
cache_dir = Path(sys.argv[1]) / "dropshotdata" / "tune_scratch" / "snapshot_cache_verify"
shutil.rmtree(cache_dir, ignore_errors=True)
s_old, t_old = timed(stu_old.build_snapshots, metas, df=df)
s_miss, t_miss = timed(stu_new.build_snapshots, metas, df=df, cache_dir=cache_dir)
s_hit, t_hit = timed(stu_new.build_snapshots, metas, df=df, cache_dir=cache_dir)
log(f"  old {t_old:.0f}s | new miss {t_miss:.0f}s | new hit {t_hit:.1f}s")
FIELDS = ("m_a", "v_a", "m_b", "v_b", "kappa", "kappa_var", "lam", "lam_var", "last_seen", "n_matches")


def same_state(x, y):
    return all(getattr(x, f) == getattr(y, f) for f in FIELDS)


for label, other in (("new-miss", s_miss), ("new-hit", s_hit)):
    ok = set(s_old) == set(other) and all(
        set(s_old[k]) == set(other[k]) and all(same_state(s_old[k][u], other[k][u]) for u in s_old[k])
        for k in s_old)
    n_states = sum(len(v) for v in s_old.values())
    log(f"  old vs {label}: {n_states} player states, identical={ok}")
    if not ok:
        failures.append(f"snapshots_{label}")
shutil.rmtree(cache_dir, ignore_errors=True)
all_snaps = s_old

# ---------------------------------------------------------------- 4. pairwise
log("\n=== 4. build_pairwise_features_base ===")
models, cat_levels, _ps = xbr_old.train_bagged_model()
age_table = live.load_age_table()
for tour in ("atp", "wta"):
    d = draws[tour]
    field_uids = sorted(set(d["bracket"]))
    best_of = 5 if tour == "atp" else 3
    snap = all_snaps[f"{tour}:2026-usopen"]
    h2h = xbr_old.build_h2h_dict(set(field_uids))
    ptab = xbr_old.build_player_table(field_uids, tour, live.DATE0, live.SURFACE, snap, playstyle_full, age_table)
    (fa, ia, ja), ta = timed(xbr_old.build_pairwise_features_base, field_uids, ptab, tour, live.SURFACE, best_of, "G", h2h, live.DATE0)
    (fb, ib, jb), tb = timed(xbr_new.build_pairwise_features_base, field_uids, ptab, tour, live.SURFACE, best_of, "G", h2h, live.DATE0)
    pairs_ok = (ia == ib) and (ja == jb)
    try:
        pd.testing.assert_frame_equal(fa, fb, check_exact=True)
        frame_ok = True
    except AssertionError as e:
        frame_ok = False
        log("  " + str(e).splitlines()[0])
    log(f"  [{tour}] old {fa.shape} {ta:.1f}s | new {fb.shape} {tb:.1f}s | frame identical={frame_ok} pairs identical={pairs_ok}")
    if not (frame_ok and pairs_ok):
        failures.append(f"pairwise_{tour}")

log("\n=== result ===")
if failures:
    log(f"FAILED: {failures}")
    sys.exit(1)
log("PERF_LIVE_PATH_IDENTICAL")
