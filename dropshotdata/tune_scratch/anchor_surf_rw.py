"""Anchor rows for one SURF_RW_SCALE: replay to the US Open cutoff (with bridge
rows) and print the point model's own match odds for the pairs that exposed
the flaw. Usage: DSV_SURF_RW_SCALE=0.3 python anchor_surf_rw.py"""
import os, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd
import bridge_tennismylife as bridge, hierarchical_model as hm
from simulate_tournament_updating import apply_regime_b_update, predict_p1_p2
from simulate_usopen_live import DATE0
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm)); df = df[df.date < DATE0]
players = {}; get = lambda u: players.setdefault(u, hm.PlayerState())
for row in df.itertuples(index=False):
    w,l,s,date = row.winner_uid,row.loser_uid,row.surface,row.date
    sv,sl = get(w),get(l); hm.inflate(sv,date); hm.inflate(sl,date)
    k_w,vk_w = sv.get_kappa(s); k_l,vk_l = sl.get_kappa(s); lm_w,vlm_w = sv.get_lam(s); lm_l,vlm_l = sl.get_lam(s)
    if row.has_points:
        e_w,f_l,v_e,v_f = sv.m_a+k_w, sl.m_b+lm_l, sv.v_a+vk_w, sl.v_b+vlm_l
        a,b,c,d = hm.adf_update(e_w,v_e,f_l,v_f,float(row.w_won),float(row.w_svpt))
        sv.m_a,sv.v_a,k_w,vk_w = hm.apply_side_update(sv.m_a,sv.v_a,k_w,vk_w,e_w,a,b)
        sl.m_b,sl.v_b,lm_l,vlm_l = hm.apply_side_update(sl.m_b,sl.v_b,lm_l,vlm_l,f_l,c,d)
        e_l,f_w,v_e2,v_f2 = sl.m_a+k_l, sv.m_b+lm_w, sl.v_a+vk_l, sv.v_b+vlm_w
        a,b,c,d = hm.adf_update(e_l,v_e2,f_w,v_f2,float(row.l_won),float(row.l_svpt))
        sl.m_a,sl.v_a,k_l,vk_l = hm.apply_side_update(sl.m_a,sl.v_a,k_l,vk_l,e_l,a,b)
        sv.m_b,sv.v_b,lm_w,vlm_w = hm.apply_side_update(sv.m_b,sv.v_b,lm_w,vlm_w,f_w,c,d)
        sv.kappa[s],sv.kappa_var[s],sl.kappa[s],sl.kappa_var[s] = k_w,vk_w,k_l,vk_l
        sv.lam[s],sv.lam_var[s],sl.lam[s],sl.lam_var[s] = lm_w,vlm_w,lm_l,vlm_l
    else:
        ts = hm.REGIME_B_N if row.tourney_level in hm.ELITE_LEVELS else hm.REGIME_B_N*hm.TIER_WEIGHT
        if row.tour=="wta" and row.level_file=="lower": ts *= hm.WTA_LOWER_WEIGHT
        gw,gl = (row.total_games_w,row.total_games_l) if not np.isnan(row.total_games_w) else (1.0,0.0)
        apply_regime_b_update(sv,sl,gw,gl,ts)
    sv.n_matches += 1; sl.n_matches += 1
U = {"Zverev":"atp:100644","Shelton":"atp:210097","Khachanov":"atp:105807","Alcaraz":"atp:207989","Sinner":"atp:206173","Djokovic":"atp:104925","Safiullin":None}
# resolve Khachanov/Safiullin ids from names if the guesses are wrong
names = pd.concat([df[["winner_uid","winner_name"]].set_axis(["uid","name"],axis=1), df[["loser_uid","loser_name"]].set_axis(["uid","name"],axis=1)]).drop_duplicates("uid").set_index("name").uid
for k in list(U):
    if U[k] is None or U[k] not in players:
        m = [n for n in names.index if k in str(n) and str(n).startswith(("Karen","Roman","Alexander","Ben","Carlos","Jannik","Novak"))]
        U[k] = names[m[0]] if m else None
print(f"SURF_RW_SCALE={hm.SURF_RW_SCALE}")
for k,u in U.items():
    st = players[u]; print(f"{k:>10} m_a={st.m_a:+.3f} v_a={st.v_a:.4f} m_b={st.m_b:+.3f} v_b={st.v_b:.4f} kappa={ {s:round(v,3) for s,v in st.kappa.items()} } lam={ {s:round(v,3) for s,v in st.lam.items()} }")
print("\nP(row beats col), bo5 Hard:")
ks = list(U); M = pd.DataFrame(index=ks, columns=ks, dtype=float)
for a in ks:
    for b in ks:
        if a!=b:
            p1,p2 = predict_p1_p2(players[U[a]],players[U[b]],"Hard"); M.loc[a,b] = hm.p_match(p1,p2,5)
print(M.round(3).to_string())
