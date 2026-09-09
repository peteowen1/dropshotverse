"""Walk two rows end to end: why does the model give Shelton 94% vs Khachanov
but Zverev only 85%, while Zverev beats Shelton head-to-head? Prints the
feature rows and per-feature XGBoost contributions (pred_contribs)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np, pandas as pd, xgboost as xgb
import bridge_tennismylife as bridge, hierarchical_model as hm
import simulate_tournament_updating as stu, usopen_live_draw as draw_mod
from simulate_usopen_live import DATE0, SURFACE, DD
from xgb_bt_rating import (build_h2h_dict, build_player_table, load_age_table,
                           build_pairwise_features_base, prep_X_d, train_bagged_model)

models, cat_levels, playstyle_full = train_bagged_model()
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
n2u = draw_mod.build_name_to_uid(df)
age_table = load_age_table()
d = draw_mod.load_live_draw("atp", n2u)
field = sorted(set(d["bracket"])); names = d["names"]
snap = stu.build_snapshots({"atp:2026-usopen": {"date0": DATE0, "field_uids": field}}, df=df,
                           cache_dir=DD / "tune_scratch" / "snapshot_cache")["atp:2026-usopen"]
h2h = build_h2h_dict(set(field))
ptab = build_player_table(field, "atp", DATE0, SURFACE, snap, playstyle_full, age_table)
uid = {k: [u for u in field if k in names[u]][0] for k in ["Zverev", "Shelton", "Khachanov", "Tiafoe", "Safiullin", "Alcaraz"]}

print("\n=== player table rows (inputs) ===")
cols = [c for c in ptab.columns if c != "state"]
print(ptab.loc[list(uid.values()), cols].rename(index={v: k for k, v in uid.items()}).T.to_string())

base, pi, pj = build_pairwise_features_base(field, ptab, "atp", SURFACE, 5, "G", h2h, DATE0)
base["round"] = "F"
X = prep_X_d(base, cat_levels)
dm = xgb.DMatrix(X, enable_categorical=True)
pred = np.mean([m.predict(dm) for m in models], axis=0)
contrib = np.mean([m.predict(dm, pred_contribs=True) for m in models], axis=0)
key = {(a, b): k for k, (a, b) in enumerate(zip(pi, pj))}
pairs = [("Zverev", "Khachanov"), ("Shelton", "Khachanov"), ("Zverev", "Shelton"), ("Zverev", "Safiullin"), ("Shelton", "Safiullin"), ("Zverev", "Tiafoe"), ("Shelton", "Tiafoe")]
feat = list(X.columns) + ["bias"]
rows_f, rows_c = {}, {}
for a, b in pairs:
    k = key[(uid[a], uid[b])]
    lab = f"{a}>{b} p={pred[k]:.3f}"
    rows_f[lab] = base.iloc[k][[c for c in X.columns]]
    rows_c[lab] = pd.Series(contrib[k], index=feat)
print("\n=== feature values ===")
print(pd.DataFrame(rows_f).to_string())
print("\n=== logit contributions (bag mean); sum == logit(p) ===")
C = pd.DataFrame(rows_c); C.loc["SUM"] = C.sum()
print(C.round(3).to_string())
