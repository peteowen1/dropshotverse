import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v10_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

j = hold[["date", "tour", "winner_uid", "loser_uid", "y", "p_xgb", "surface"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]), on=["date", "tour", "winner_uid", "loser_uid"],
    how="left")
y = j["y"].values.astype(int)
p_model = np.clip(j["p_xgb"].values, 1e-9, 1 - 1e-9)
p_mkt = np.clip(np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values), 1e-9, 1 - 1e-9)

# per-row log-loss for each, paired difference (model - market): positive
# means model is worse on that row. CI on the MEAN paired difference.
ll_model = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
ll_mkt = -(y * np.log(p_mkt) + (1 - y) * np.log(1 - p_mkt))
diff = ll_model - ll_mkt


def report(mask, label):
    d = diff[mask]
    n = len(d)
    mean_diff = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    ci_lo, ci_hi = mean_diff - 1.96 * se, mean_diff + 1.96 * se
    print(f"{label:12s} n={n:5,d}  mean_gap={mean_diff:+.5f}  se={se:.5f}  "
          f"95% CI=[{ci_lo:+.5f}, {ci_hi:+.5f}]  "
          f"({'includes zero - not statistically distinguishable' if ci_lo <= 0 <= ci_hi else 'excludes zero - real'})")


tour = j["tour"].values
surf = j["surface"].values
report(np.ones(len(j), dtype=bool), "ALL")
report(tour == "atp", "ATP")
report(tour == "wta", "WTA")
report((tour == "atp") & (surf == "Grass"), "ATP-Grass")
report((tour == "atp") & (surf != "Grass"), "ATP-non-Grass")
