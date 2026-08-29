import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v10_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

j = hold[(hold["tour"] == "atp") & (hold["surface"] == "Grass")][
    ["date", "tour", "winner_uid", "loser_uid", "y", "p_xgb", "round", "tourney_level", "best_of"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
    how="left")
y = j["y"].values.astype(int)
p_xgb = j["p_xgb"].values
p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


print(f"ATP-Grass total: n={len(j)}  model={ll(y, p_xgb):.5f}  market={ll(y, p_mkt):.5f}  "
      f"gap={ll(y, p_xgb) - ll(y, p_mkt):+.5f}")

print("\nby tourney_level (G=Slam i.e. Wimbledon, A=regular tour i.e. Halle/Queens/etc):")
for lvl, grp in j.groupby("tourney_level"):
    idx = grp.index
    if len(idx) < 15:
        print(f"  {lvl}: n={len(idx)} too few")
        continue
    print(f"  {lvl:5s} n={len(idx):4d}  model={ll(y[idx], p_xgb[idx]):.5f}  market={ll(y[idx], p_mkt[idx]):.5f}  "
          f"gap={ll(y[idx], p_xgb[idx]) - ll(y[idx], p_mkt[idx]):+.5f}")

print("\nby best_of:")
for bo, grp in j.groupby("best_of"):
    idx = grp.index
    if len(idx) < 15:
        continue
    print(f"  bo={bo}  n={len(idx):4d}  model={ll(y[idx], p_xgb[idx]):.5f}  market={ll(y[idx], p_mkt[idx]):.5f}  "
          f"gap={ll(y[idx], p_xgb[idx]) - ll(y[idx], p_mkt[idx]):+.5f}")

print("\nby round:")
for r, grp in j.groupby("round"):
    idx = grp.index
    if len(idx) < 15:
        continue
    print(f"  {r:5s} n={len(idx):4d}  model={ll(y[idx], p_xgb[idx]):.5f}  market={ll(y[idx], p_mkt[idx]):.5f}  "
          f"gap={ll(y[idx], p_xgb[idx]) - ll(y[idx], p_mkt[idx]):+.5f}")

mkt_conf = np.maximum(p_mkt, 1 - p_mkt)
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
j["bucket"] = pd.cut(mkt_conf, bins)
print("\nby market-confidence bucket:")
for b, grp in j.groupby("bucket", observed=True):
    idx = grp.index
    if len(idx) < 15:
        continue
    print(f"  {str(b):12s} n={len(idx):4d}  model={ll(y[idx], p_xgb[idx]):.5f}  market={ll(y[idx], p_mkt[idx]):.5f}  "
          f"gap={ll(y[idx], p_xgb[idx]) - ll(y[idx], p_mkt[idx]):+.5f}")
