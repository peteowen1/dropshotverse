"""Compare the live sim's field-average strength (the BT column on the site)
and title odds before and after the 2026-09-09 input fixes."""
import numpy as np, pandas as pd
from pathlib import Path
T = Path(__file__).resolve().parent; B = T / "inputfix_before"; A = T.parent / "blog"
def bt(pw):
    m = pw[(pw.tour=="atp") & (pw["round"]=="R32")].copy(); m["lg"] = np.log(m.p_a_beats_b/(1-m.p_a_beats_b))
    return m.groupby("player_id_a").lg.mean()
pb, pa = pd.read_parquet(B/"pairwise.parquet"), pd.read_parquet(A/"pairwise.parquet")
lb, la = pd.read_parquet(B/"live_predictions.parquet"), pd.read_parquet(A/"live_predictions.parquet")
names = la.set_index("player_id").player_name
d = pd.DataFrame({"bt_before": bt(pb), "bt_after": bt(pa)}); d["name"] = d.index.map(names)
d = d.join(lb.set_index("player_id").p_title.rename("title_before")).join(la.set_index("player_id").p_title.rename("title_after"))
alive = d[d.title_after > 0].sort_values("title_after", ascending=False)
print("BT = model's mean log-odds vs the 128 field (R32, Hard); title = P(champion). Higher is stronger.")
print(alive[["name","bt_before","bt_after","title_before","title_after"]].round(3).to_string(index=False))
print("\nrank of Zverev by BT before/after:", int((d.bt_before > d.loc[d.name=="Alexander Zverev","bt_before"].iloc[0]).sum())+1, int((d.bt_after > d.loc[d.name=="Alexander Zverev","bt_after"].iloc[0]).sum())+1)
for rd in ["SF","F"]:
    ids = alive.index.tolist()
    for lab, pw in (("before", pb), ("after", pa)):
        f = pw[(pw["round"]==rd)&pw.player_id_a.isin(ids)&pw.player_id_b.isin(ids)].pivot(index="player_id_a",columns="player_id_b",values="p_a_beats_b")
        f.index = f.index.map(lambda u: names[u].split()[-1]); f.columns = f.columns.map(lambda u: names[u].split()[-1])
        print(f"\n{rd} {lab}: P(row beats col)"); print(f.round(2).to_string())
