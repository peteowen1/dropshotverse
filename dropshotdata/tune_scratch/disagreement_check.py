"""Ticket 28 round 7 (bias check, per Pete's explicit ask): when XGBoost v2
picks a different favorite than the market, is it right more often than
chance? If disagreement accuracy is ~50% or below, XGBoost has no real edge
in those spots and is just adding noise relative to the market; if it's
durably above 50%, there might be a genuine, if small, informational edge
somewhere identifiable (e.g. one tour, one surface, one confidence range)."""
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
disagree = xgb_pick != mkt_pick
print(f"total n={len(j):,}  disagreements={disagree.sum():,} ({disagree.mean():.2%})")

xgb_right = (xgb_pick == y)
mkt_right = (mkt_pick == y)
print(f"\noverall accuracy: XGBoost={xgb_right.mean():.4f}  Market={mkt_right.mean():.4f}")

d = disagree
print(f"\nwhen they disagree (n={d.sum():,}):")
print(f"  XGBoost right: {xgb_right[d].mean():.4f}")
print(f"  Market right:  {mkt_right[d].mean():.4f}")
# 95% CI on XGBoost's disagreement accuracy (normal approx)
n_d = d.sum()
p_hat = xgb_right[d].mean()
se = np.sqrt(p_hat * (1 - p_hat) / n_d)
print(f"  XGBoost disagreement accuracy 95% CI: [{p_hat - 1.96*se:.4f}, {p_hat + 1.96*se:.4f}]  "
      f"(0.50 {'INSIDE' if p_hat - 1.96*se <= 0.50 <= p_hat + 1.96*se else 'outside'} the interval)")

print("\nby tour:")
for tour in ("atp", "wta"):
    m = d & (j["tour"].values == tour)
    if m.sum() < 20:
        continue
    print(f"  {tour}: n={m.sum():4d}  xgb_right={xgb_right[m].mean():.4f}  mkt_right={mkt_right[m].mean():.4f}")

print("\nby surface:")
for surf in j["surface"].unique():
    m = d & (j["surface"].values == surf)
    if m.sum() < 20:
        continue
    print(f"  {surf:8s}: n={m.sum():4d}  xgb_right={xgb_right[m].mean():.4f}  mkt_right={mkt_right[m].mean():.4f}")

# directional bias: does XGBoost systematically favor side A/B, home tour bias etc? already
# checked overall calibration is near-zero bias by construction (symmetric hash). Check
# whether XGBoost's disagreements skew toward favoring the market's LONGSHOT (a classic
# "smart money" pattern) or the market's FAVORITE.
mkt_fav_is_xgb_dog = d & (p_mkt_sideA > 0.5) & (p_xgb < 0.5)   # xgb backs the market's underdog
mkt_dog_is_xgb_fav = d & (p_mkt_sideA < 0.5) & (p_xgb > 0.5)   # xgb backs... wait this is same as above by construction
print(f"\ndisagreement direction: XGBoost backs market's underdog in "
      f"{(d & (p_mkt_sideA > 0.5)).sum():,} of {d.sum():,} disagreements "
      f"({(d & (p_mkt_sideA > 0.5)).sum() / d.sum():.2%})")
m = d & (p_mkt_sideA > 0.5)
print(f"  when XGBoost backs the market underdog: right {xgb_right[m].mean():.4f} of {m.sum()} times")
m2 = d & (p_mkt_sideA <= 0.5)
print(f"  when XGBoost backs the market favorite (mkt already <0.5 on this side... symmetric other side): "
      f"right {xgb_right[m2].mean():.4f} of {m2.sum()} times")
