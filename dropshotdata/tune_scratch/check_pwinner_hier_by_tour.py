import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hier_raw = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")

for tour in ("atp", "wta"):
    sub = hier_raw[hier_raw["tour"] == tour]
    p = np.clip(sub["p_winner_hier"].values, 1e-9, 1 - 1e-9)
    ll = -np.log(p).mean()
    acc = (p > 0.5).mean()
    print(f"{tour}: n={len(sub):,}  standalone_logloss={ll:.4f}  acc={acc:.4f}  "
          f"mean_p={p.mean():.4f}  saturating(0/1)={((p<1e-6)|(p>1-1e-6)).mean():.4%}")

# also check p1, p2 (point-level serve-win probabilities) by tour, sanity vs
# ticket 11's known finding: ATP 64.0% vs WTA 58.3% service points won
print("\np1 (winner's own serve-point win prob) by tour, sanity vs ticket 11's 64.0%/58.3%:")
for tour in ("atp", "wta"):
    sub = hier_raw[hier_raw["tour"] == tour]
    print(f"  {tour}: mean p1={sub['p1'].mean():.4f}  mean p2={sub['p2'].mean():.4f}")
