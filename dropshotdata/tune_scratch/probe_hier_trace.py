"""Instrumented replay of the point model for two players: log every update
to their Hard-court state from 2025-01-01 on, plus the final state. Same loop
as simulate_tournament_updating.build_snapshots (regime A/B), so the numbers
are the ones the live sim uses."""
import sys, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd
import bridge_tennismylife as bridge, hierarchical_model as hm, simulate_tournament_updating as stu
from simulate_tournament_updating import apply_regime_b_update
from simulate_usopen_live import DATE0
TRACE = {"atp:210097": "Shelton", "atp:100644": "Zverev"}
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
df = df[df.date < DATE0].sort_values(["date"], kind="stable")
print("rows", len(df), "has_points share 2026:", round(df[df.date>="2026-01-01"].has_points.mean(),3),
      "2025:", round(df[(df.date>="2025-01-01")&(df.date<"2026-01-01")].has_points.mean(),3))
players = {}
get = lambda u: players.setdefault(u, hm.PlayerState())
rows = []
t0=time.time()
for row in df.itertuples(index=False):
    w, l, s, date = row.winner_uid, row.loser_uid, row.surface, row.date
    sv, sl = get(w), get(l); hm.inflate(sv, date); hm.inflate(sl, date)
    k_w, vk_w = sv.get_kappa(s); k_l, vk_l = sl.get_kappa(s); lm_w, vlm_w = sv.get_lam(s); lm_l, vlm_l = sl.get_lam(s)
    if row.has_points:
        e_w, f_l, v_e, v_f = sv.m_a + k_w, sl.m_b + lm_l, sv.v_a + vk_w, sl.v_b + vlm_l
        new_ew, new_ve, new_fl, new_vf = hm.adf_update(e_w, v_e, f_l, v_f, float(row.w_won), float(row.w_svpt))
        sv.m_a, sv.v_a, k_w, vk_w = hm.apply_side_update(sv.m_a, sv.v_a, k_w, vk_w, e_w, new_ew, new_ve)
        sl.m_b, sl.v_b, lm_l, vlm_l = hm.apply_side_update(sl.m_b, sl.v_b, lm_l, vlm_l, f_l, new_fl, new_vf)
        e_l, f_w, v_e2, v_f2 = sl.m_a + k_l, sv.m_b + lm_w, sl.v_a + vk_l, sv.v_b + vlm_w
        new_el, new_ve2, new_fw, new_vf2 = hm.adf_update(e_l, v_e2, f_w, v_f2, float(row.l_won), float(row.l_svpt))
        sl.m_a, sl.v_a, k_l, vk_l = hm.apply_side_update(sl.m_a, sl.v_a, k_l, vk_l, e_l, new_el, new_ve2)
        sv.m_b, sv.v_b, lm_w, vlm_w = hm.apply_side_update(sv.m_b, sv.v_b, lm_w, vlm_w, f_w, new_fw, new_vf2)
        sv.kappa[s], sv.kappa_var[s], sl.kappa[s], sl.kappa_var[s] = k_w, vk_w, k_l, vk_l
        sv.lam[s], sv.lam_var[s], sl.lam[s], sl.lam_var[s] = lm_w, vlm_w, lm_l, vlm_l
    else:
        tier_scale = hm.REGIME_B_N if row.tourney_level in hm.ELITE_LEVELS else hm.REGIME_B_N * hm.TIER_WEIGHT
        if row.tour == "wta" and row.level_file == "lower": tier_scale *= hm.WTA_LOWER_WEIGHT
        gw, gl = (row.total_games_w, row.total_games_l) if not np.isnan(row.total_games_w) else (1.0, 0.0)
        apply_regime_b_update(sv, sl, gw, gl, tier_scale)
    sv.n_matches += 1; sl.n_matches += 1
    for u in (w, l):
        if u in TRACE and date >= pd.Timestamp("2025-01-01"):
            st = players[u]; kh, vkh = st.get_kappa("Hard"); lh, vlh = st.get_lam("Hard")
            rows.append(dict(who=TRACE[u], date=date.date(), surf=s, res="W" if u==w else "L",
                             opp=(row.loser_name if u==w else row.winner_name), pts=bool(row.has_points),
                             m_a=st.m_a, v_a=st.v_a, m_b=st.m_b, v_b=st.v_b, kH=kh, vkH=vkh, lH=lh, vlH=vlh))
print("replay", round(time.time()-t0), "s")
out = pd.DataFrame(rows); out.to_csv(Path(__file__).parent / "probe_hier_trace.csv", index=False)
pd.set_option("display.width", 250)
for who in TRACE.values():
    o = out[out.who==who]
    print(f"\n=== {who}: {len(o)} matches since 2025; points share {o.pts.mean():.2f}; Hard share {(o.surf=='Hard').mean():.2f} ===")
    print(o[o.date>=pd.Timestamp('2026-01-01').date()].round(3).to_string(index=False))
    st = players[[u for u,n in TRACE.items() if n==who][0]]
    print("FINAL", who, "m_a=%.3f v_a=%.4f m_b=%.3f v_b=%.4f" % (st.m_a, st.v_a, st.m_b, st.v_b),
          {s: (round(st.kappa[s],3), round(st.kappa_var[s],4)) for s in st.kappa}, {s: (round(st.lam[s],3), round(st.lam_var[s],4)) for s in st.lam})
