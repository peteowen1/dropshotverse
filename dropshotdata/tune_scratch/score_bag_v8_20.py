import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
seeds = pd.read_parquet(rf"{DD}\tune_scratch\bag_v8_seed_preds.parquet")
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

key = ["date", "tour", "winner_uid", "loser_uid"]
seed_cols = [c for c in seeds.columns if c.startswith("p_seed")]
print(f"seeds available: {len(seed_cols)}")

j = hold[key + ["y"]].merge(seeds[key + seed_cols], on=key, how="left")
j = j.merge(diag2[key + ["p_mkt"]].drop_duplicates(subset=key), on=key, how="left")
y = j["y"].values.astype(int)
p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


for n in (1, 5, 10, 15, 20):
    cols = seed_cols[:n]
    p_bag = j[cols].values.mean(axis=1)
    tour_arr = j["tour"].values
    print(f"\nn_bags={n}")
    for label, mask in (("ALL", np.ones(len(j), dtype=bool)), ("ATP", tour_arr == "atp"),
                        ("WTA", tour_arr == "wta")):
        print(f"  {label:4s} model={ll(y[mask], p_bag[mask]):.5f}  market={ll(y[mask], p_mkt[mask]):.5f}  "
              f"gap={ll(y[mask], p_bag[mask]) - ll(y[mask], p_mkt[mask]):+.5f}")
