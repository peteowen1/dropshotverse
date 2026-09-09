"""Why does the model rate Fils and Jodar level with or above Zverev when Elo
has Zverev clear? Per-feature XGBoost contributions on the US Open field,
R32 / Hard / F contexts, for Zverev vs Fils, Jodar, Sinner-if-present."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd, xgboost as xgb
import bridge_tennismylife as bridge, hierarchical_model as hm, simulate_tournament_updating as stu, usopen_live_draw as draw_mod
from simulate_usopen_live import DATE0, SURFACE, DD
from xgb_bt_rating import build_h2h_dict, build_player_table, load_age_table, build_pairwise_features_base, prep_X_d, train_bagged_model
models, cat_levels, playstyle_full = train_bagged_model()
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm)); n2u = draw_mod.build_name_to_uid(df); age_table = load_age_table()
d = draw_mod.load_live_draw("atp", n2u); field = sorted(set(d["bracket"])); names = d["names"]
snap = stu.build_snapshots({"atp:2026-usopen": {"date0": DATE0, "field_uids": field}}, df=df, cache_dir=DD/"tune_scratch"/"snapshot_cache")["atp:2026-usopen"]
h2h = build_h2h_dict(set(field)); ptab = build_player_table(field, "atp", DATE0, SURFACE, snap, playstyle_full, age_table)
uid = {k: [u for u in field if k in names[u]][0] for k in ["Zverev","Fils","Jodar","Alcaraz","Shelton"]}
cols = [c for c in ptab.columns if c != "state"]
print("=== inputs ==="); print(ptab.loc[list(uid.values()), cols].rename(index={v:k for k,v in uid.items()}).T.round(3).to_string())
base, pi, pj = build_pairwise_features_base(field, ptab, "atp", SURFACE, 5, "G", h2h, DATE0); base["round"] = "R32"
X = prep_X_d(base, cat_levels); dm = xgb.DMatrix(X, enable_categorical=True)
pred = np.mean([m.predict(dm) for m in models], axis=0); contrib = np.mean([m.predict(dm, pred_contribs=True) for m in models], axis=0)
key = {(a,b):k for k,(a,b) in enumerate(zip(pi,pj))}
pairs = [("Zverev","Fils"),("Zverev","Jodar"),("Zverev","Alcaraz"),("Fils","Alcaraz"),("Zverev","Shelton")]
C = {}
for a,b in pairs:
    k = key[(uid[a],uid[b])]; C[f"{a}>{b} p={pred[k]:.3f}"] = pd.Series(contrib[k], index=list(X.columns)+["bias"])
C = pd.DataFrame(C); C = C.loc[C.abs().max(axis=1).sort_values(ascending=False).index]; C.loc["SUM"] = C.sum()
print("\n=== logit contributions (R32, Hard) ==="); print(C.round(3).head(12).to_string())
# field-average BT and the age share of it
lg = np.log(pred/(1-pred)); B = pd.DataFrame({"i":pi,"lg":lg,"age_c":contrib[:, list(X.columns).index("age_diff")], "hier_c":contrib[:, list(X.columns).index("hier_logit_pwinner")], "surf_c":contrib[:, list(X.columns).index("elo_surface_diff")], "game_c":contrib[:, list(X.columns).index("game_elo_diff")]})
bt = B.groupby("i").mean(); bt["name"] = bt.index.map(names); bt["ro"] = ptab.ro; bt["age"] = ptab.age
print("\n=== field-average BT and what builds it (mean contribution over 127 opponents) ===")
print(bt.sort_values("lg", ascending=False).head(10)[["name","ro","age","lg","age_c","hier_c","surf_c","game_c"]].round(3).to_string(index=False))
