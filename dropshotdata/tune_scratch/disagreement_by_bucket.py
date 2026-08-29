"""Follow-up to disagreement_check.py: is there ANY subgroup where XGBoost's
disagreement accuracy clears 50%, hidden inside an aggregate that's net
below it? Checked by market-confidence bucket and tourney_level - the two
axes not yet checked (tour and surface were both uniformly bad)."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

j = hold[["date", "tour", "winner_uid", "loser_uid", "y", "p_xgb", "surface", "tourney_level"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]),
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")

y = j["y"].values.astype(int)
p_xgb = j["p_xgb"].values
p_mkt_sideA = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
xgb_pick = (p_xgb > 0.5).astype(int)
mkt_pick = (p_mkt_sideA > 0.5).astype(int)
d = xgb_pick != mkt_pick
xgb_right = (xgb_pick == y)

mkt_conf = np.maximum(p_mkt_sideA, 1 - p_mkt_sideA)
j["mkt_conf"] = mkt_conf
bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
j["bucket"] = pd.cut(mkt_conf, bins)
print("=== disagreement accuracy by market-confidence bucket ===")
for b, grp in j[d].groupby("bucket", observed=True):
    idx = grp.index
    n = len(idx)
    if n < 15:
        print(f"{str(b):15s} n={n:4d}  (too few to read)")
        continue
    acc = xgb_right[idx].mean()
    se = np.sqrt(acc * (1 - acc) / n)
    print(f"{str(b):15s} n={n:4d}  xgb_right={acc:.4f}  95%CI=[{acc-1.96*se:.4f},{acc+1.96*se:.4f}]")

print("\n=== disagreement accuracy by tourney_level ===")
for lvl, grp in j[d].groupby("tourney_level"):
    idx = grp.index
    n = len(idx)
    if n < 20:
        continue
    acc = xgb_right[idx].mean()
    se = np.sqrt(acc * (1 - acc) / n)
    print(f"{lvl:5s} n={n:4d}  xgb_right={acc:.4f}  95%CI=[{acc-1.96*se:.4f},{acc+1.96*se:.4f}]")
