import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
v8 = pd.read_parquet(rf"{DD}\tune_scratch\bag_v8_seed_preds.parquet")  # has p_seed0..19, use seed0 as v8 single-model ref
v8_single = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")
v9 = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v9_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

key = ["date", "tour", "winner_uid", "loser_uid"]
v9_keys = v9[key + ["surface"]].drop_duplicates()
v8_same = v8_single.merge(v9_keys, on=key, how="inner", suffixes=("", "_v9surf"))
print(f"v8 full n={len(v8_single):,}  v9 n={len(v9):,}  v8-restricted-to-v9-rows n={len(v8_same):,}")


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(df, label, extra_mask=None):
    j = df.merge(diag2[key + ["p_mkt"]].drop_duplicates(subset=key), on=key, how="left")
    y = j["y"].values.astype(int)
    p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
    mask = np.ones(len(j), dtype=bool) if extra_mask is None else extra_mask(j)
    print(f"  {label:12s} n={mask.sum():5,d}  model={ll(y[mask], j['p_xgb'].values[mask]):.5f}  "
          f"market={ll(y[mask], p_mkt[mask]):.5f}  gap={ll(y[mask], j['p_xgb'].values[mask]) - ll(y[mask], p_mkt[mask]):+.5f}")


print("\n=== v8 (Arm G, single model) restricted to v9's rows ===")
score(v8_same, "ALL")
score(v8_same, "ATP", lambda j: j["tour"].values == "atp")
score(v8_same, "WTA", lambda j: j["tour"].values == "wta")
score(v8_same, "ATP-Grass", lambda j: (j["tour"].values == "atp") & (j["surface"].values == "Grass"))

print("\n=== v9 (Arm H, +serve/return form), its own native rows ===")
score(v9, "ALL")
score(v9, "ATP", lambda j: j["tour"].values == "atp")
score(v9, "WTA", lambda j: j["tour"].values == "wta")
score(v9, "ATP-Grass", lambda j: (j["tour"].values == "atp") & (j["surface"].values == "Grass"))
