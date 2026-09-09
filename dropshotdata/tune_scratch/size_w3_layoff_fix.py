"""Size the W3 layoff fix (DSV_W3_LAYOFF_FIX) on the real US Open ATP field,
pre-tournament (no results applied), WITHOUT writing anything under blog/.
Builds the per-round matrices with the flag off and on, reports how much the
pairwise odds move per round, and runs a frozen-bracket sim for title odds."""
import os, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd, importlib
import bridge_tennismylife as bridge, hierarchical_model as hm, simulate_tournament_updating as stu, usopen_live_draw as draw_mod
import simulate_tournament_xgb as stx
from simulate_tournament import STAGE_ORDER
from simulate_usopen_live import DATE0, SURFACE, DD
from xgb_bt_rating import build_h2h_dict, build_player_table, load_age_table, train_bagged_model
models, cat_levels, playstyle_full = train_bagged_model()
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm)); n2u = draw_mod.build_name_to_uid(df); age_table = load_age_table()
d = draw_mod.load_live_draw("atp", n2u); field = sorted(set(d["bracket"])); names = d["names"]; bracket = d["bracket"]
snap = stu.build_snapshots({"atp:2026-usopen": {"date0": DATE0, "field_uids": field}}, df=df, cache_dir=DD/"tune_scratch"/"snapshot_cache")["atp:2026-usopen"]
h2h = build_h2h_dict(set(field)); ptab = build_player_table(field, "atp", DATE0, SURFACE, snap, playstyle_full, age_table)
idx = {u: k for k, u in enumerate(field)}
print("weeks_since_last on the field: median %.1f, p90 %.1f, max %.1f, n at 52.0 default = %d" % (
    ptab.weeks_since_last.median(), ptab.weeks_since_last.quantile(.9), ptab.weeks_since_last.max(), int((ptab.weeks_since_last==52.0).sum())))
res = {}
for flag in ("0", "1"):
    os.environ["DSV_W3_LAYOFF_FIX"] = flag
    res[flag] = stx.build_round_pred_matrices(field, ptab, "atp", SURFACE, 5, "G", h2h, DATE0, cat_levels, models, STAGE_ORDER[:-1])
print("\nmean and max |delta p| per round, flag on vs off (R128 must be 0.000 by construction):")
for rnd in STAGE_ORDER[:-1]:
    dlt = np.abs(res["1"][rnd] - res["0"][rnd]); print(f"  {rnd:>4}: mean {np.nanmean(dlt):.4f}  p99 {np.nanpercentile(dlt,99):.3f}  max {np.nanmax(dlt):.3f}")
rng = np.random.default_rng(7); N = 20000
out = {}
for flag in ("0","1"):
    cnt = {u: 0 for u in field}
    for _ in range(N):
        r = stx.simulate_frozen_xgb(bracket, idx, res[flag], rng)
        for u, st in r.items():
            if st == "W": cnt[u] += 1
    out[flag] = pd.Series({u: c/N for u, c in cnt.items()})
t = pd.DataFrame({"title_off": out["0"], "title_on": out["1"]}); t["name"] = t.index.map(names); t["weeks"] = ptab.weeks_since_last; t["delta"] = t.title_on - t.title_off
print("\nPre-tournament title odds, flag off vs on (top 12 by |delta|); weeks = layoff before the draw:")
print(t.reindex(t.delta.abs().sort_values(ascending=False).index).head(12)[["name","weeks","title_off","title_on","delta"]].round(3).to_string(index=False))
print("\nmean |delta| title over field: %.4f ; sum |delta| = %.3f" % (t.delta.abs().mean(), t.delta.abs().sum()))
