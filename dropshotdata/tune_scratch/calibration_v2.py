import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
hold = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")
diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")

j = hold[["date", "tour", "winner_uid", "loser_uid", "y", "p_xgb"]].merge(
    diag2[["date", "tour", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"]),
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")

y = j["y"].values.astype(int)
p_xgb = j["p_xgb"].values
p_mkt_sideA = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)


def calib_report(p, label):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, 9)
    print(f"\n=== {label} calibration (10 equal-width bins) ===")
    print(f"{'bin':12s}{'n':>7s}{'mean_pred':>11s}{'obs_rate':>10s}{'error':>8s}")
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        mp = p[m].mean()
        obs = y[m].mean()
        err = mp - obs
        ece += (m.sum() / len(p)) * abs(err)
        print(f"[{bins[b]:.1f},{bins[b+1]:.1f})  {m.sum():6d}  {mp:10.4f}  {obs:9.4f}  {err:+7.4f}")
    print(f"ECE={ece:.4f}")


calib_report(p_xgb, "XGBoost v2")
calib_report(p_mkt_sideA, "Market")

# tour-level bias: is XGBoost systematically over/under-confident by tour?
for tour in ("atp", "wta"):
    m = j["tour"].values == tour
    print(f"\n{tour}: mean_pred={p_xgb[m].mean():.4f}  obs_rate={y[m].mean():.4f}  "
          f"bias={p_xgb[m].mean() - y[m].mean():+.4f}  n={m.sum()}")
