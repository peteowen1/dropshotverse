import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v8_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

j = hold[["date", "tour", "winner_uid", "loser_uid", "y", "p_xgb", "surface",
          "elo_surface_diff", "hier_dvar_diff"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
    how="left")
y = j["y"].values.astype(int)
p_xgb = j["p_xgb"].values
p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


print("by tour x surface:")
for tour in ("atp", "wta"):
    for s in ("Hard", "Clay", "Grass"):
        m = (j["tour"] == tour) & (j["surface"] == s)
        idx = j.index[m]
        if len(idx) < 10:
            continue
        print(f"  {tour} {s:6s} n={len(idx):4d}  model={ll(y[idx], p_xgb[idx]):.5f}  "
              f"market={ll(y[idx], p_mkt[idx]):.5f}  gap={ll(y[idx], p_xgb[idx]) - ll(y[idx], p_mkt[idx]):+.5f}  "
              f"mean|elo_surf_diff|={j.loc[idx, 'elo_surface_diff'].abs().mean():.1f}")

# training-window grass sample size, for context - how much grass data exists
# to learn surface-specific ratings from, per tour
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")
print("\ntraining rows by tour x surface (1995-2023):")
print(train.groupby(["tour", "surface"]).size())
