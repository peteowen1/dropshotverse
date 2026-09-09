"""Score the SURF_RW_SCALE sweep on the 2022-23 tune window. Log-loss, lower
is better; ECE-ish reliability by decile; split by regime (point counts vs
scoreline) and tour."""
import sys, numpy as np, pandas as pd
PREFIX = sys.argv[1] if len(sys.argv)>1 else "surf_rw"
from pathlib import Path
T = Path(__file__).parent
def ll(p): p=np.clip(p,1e-9,1-1e-9); return float(-np.log(p).mean())
rows=[]
for s in ["1.0","0.3","0.1","0.0"]:
    f = T/f"{PREFIX}_{s}_preds.parquet"
    if not f.exists(): continue
    d = pd.read_parquet(f); p = d.p_winner_hier.values
    r = dict(scale=s, n=len(d), logloss=ll(p), atp=ll(d[d.tour=="atp"].p_winner_hier.values), wta=ll(d[d.tour=="wta"].p_winner_hier.values),
             fav_share=float((p>0.5).mean()), mean_p=float(p.mean()), share_gt90=float((p>0.9).mean()))
    for surf in ["Hard","Clay","Grass"]:
        r["ll_"+surf] = ll(d[d.surface==surf].p_winner_hier.values)
    rows.append(r)
pd.set_option("display.width",200); print(pd.DataFrame(rows).round(4).to_string(index=False))
