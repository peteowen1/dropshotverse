import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
seeds = pd.read_parquet(rf"{DD}\tune_scratch\bag_v10_seed_preds.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v10_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

key = ["date", "tour", "winner_uid", "loser_uid"]
seed_cols = [c for c in seeds.columns if c.startswith("p_seed")]
j = hold[key + ["y", "surface"]].merge(seeds[key + seed_cols], on=key, how="left")
j = j.merge(diag2[key + ["p_mkt"]].drop_duplicates(subset=key), on=key, how="left")
y = j["y"].values.astype(int)
p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
p_bag = j[seed_cols].values.mean(axis=1)
p_single = j["p_seed0"].values


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


tour_arr = j["tour"].values
surf_arr = j["surface"].values
masks = {
    "ALL": np.ones(len(j), dtype=bool),
    "ATP": tour_arr == "atp",
    "WTA": tour_arr == "wta",
    "ATP-Grass": (tour_arr == "atp") & (surf_arr == "Grass"),
    "ATP-Hard": (tour_arr == "atp") & (surf_arr == "Hard"),
    "ATP-Clay": (tour_arr == "atp") & (surf_arr == "Clay"),
}
print("single-seed vs 5-bag:")
for label, m in masks.items():
    print(f"  {label:10s} n={m.sum():5,d}  single={ll(y[m], p_single[m]):.5f}  bagged={ll(y[m], p_bag[m]):.5f}  "
          f"market={ll(y[m], p_mkt[m]):.5f}  bagged_gap={ll(y[m], p_bag[m]) - ll(y[m], p_mkt[m]):+.5f}")
