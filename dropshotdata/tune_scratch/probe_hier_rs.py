import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd
import bridge_tennismylife as bridge, hierarchical_model as hm, simulate_tournament_updating as stu, usopen_live_draw as draw_mod
from simulate_usopen_live import DATE0, SURFACE, DD
from xgb_bt_rating import SCRATCH
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm)); n2u = draw_mod.build_name_to_uid(df)
d = draw_mod.load_live_draw("atp", n2u); field = sorted(set(d["bracket"])); names = d["names"]
snap = stu.build_snapshots({"atp:2026-usopen": {"date0": DATE0, "field_uids": field}}, df=df, cache_dir=DD/"tune_scratch"/"snapshot_cache")["atp:2026-usopen"]
uid = {k: [u for u in field if k in names[u]][0] for k in ["Zverev","Shelton","Khachanov","Safiullin","Alcaraz","Sinner"] if any(k in names[u] for u in field)}
print("=== point-model state: p(win point on serve) vs each other on Hard ===")
for a in uid:
  for b in uid:
    if a==b: continue
    p1,p2 = stu.predict_p1_p2(snap[uid[a]], snap[uid[b]], SURFACE)
    print(f"{a:>10} serving vs {b:<10} p_serve={p1:.3f}  opp p_serve={p2:.3f}  p_match(bo5)={hm.p_match(p1,p2,5):.4f}")
for a in uid:
    s = snap[uid[a]]; k,vk = s.get_kappa(SURFACE); l,vl = s.get_lam(SURFACE)
    print(a, "a=%.3f b=%.3f kappa=%.3f lam=%.3f n=%s" % (s.a if hasattr(s,'a') else float('nan'), s.b if hasattr(s,'b') else float('nan'), k, l, getattr(s,'n',None)))
print("\n=== surface Elo (rs) on Hard: top of field vs overall ===")
elo = pd.read_parquet(SCRATCH/"elo_raw_ratings_full.parquet")
e = elo[(elo.tour=="atp")&(elo.level_file=="tour")&(elo.date<DATE0)]
print(e.surface.value_counts().head(8))
eh = e[e.surface=="Hard"]
w = eh[["date","winner_uid","rs_winner","ro_winner"]].rename(columns={"winner_uid":"uid","rs_winner":"rs","ro_winner":"ro"}); l = eh[["date","loser_uid","rs_loser","ro_loser"]].rename(columns={"loser_uid":"uid","rs_loser":"rs","ro_loser":"ro"})
lr = pd.concat([w,l]).sort_values("date").drop_duplicates("uid",keep="last").set_index("uid")
t = lr.loc[[u for u in field if u in lr.index]].assign(name=lambda x: x.index.map(names)).sort_values("ro",ascending=False).head(12)
print(t[["name","date","ro","rs"]].to_string())
print("corr(ro,rs) over field:", round(lr.loc[[u for u in field if u in lr.index]][["ro","rs"]].corr().iloc[0,1],3))
