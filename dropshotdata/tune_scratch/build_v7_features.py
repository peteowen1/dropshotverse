"""Round 13: add hier_logit_pwinner (hier's own pre-match point-estimate
p_winner, converted to a side-A logit) to Arm E's feature set. Verified
before building: (1) leak-free by source inspection - hierarchical_model.py
computes p_winner_hier from player state BEFORE this row's own posterior
update; (2) properly converges at round 3085 (not capped) with delta
-0.00941 CV, the largest confirmed effect this session; (3) merge coverage
100%, no missing-data confound.

Built for BOTH the wide training set (1995-2023) and the fixed 2024-25
holdout, from the same v25-config prediction parquets already on disk
(v25_postfix_wide1995_preds.parquet for training,
v25_postfix_holdout_preds.parquet for holdout - the original v2/issue-27
holdout hier source, still valid since it's the same v25 config).
"""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"


def add_hier_logit(feat_path, hier_preds_path, out_path):
    feat = pd.read_parquet(feat_path)
    hier = pd.read_parquet(hier_preds_path)[
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_hier"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])
    j = feat.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    n0 = len(j)
    miss = j["p_winner_hier"].isna()
    print(f"  {feat_path}: {n0} -> {int((~miss).sum())} after dropping {int(miss.sum())} missing "
          f"({miss.mean():.4%})")
    j = j[~miss].reset_index(drop=True)
    y = j["y"].values.astype(int)
    p_sideA = np.clip(np.where(y == 1, j["p_winner_hier"].values, 1 - j["p_winner_hier"].values), 1e-6, 1 - 1e-6)
    j["hier_logit_pwinner"] = np.log(p_sideA / (1 - p_sideA))
    j = j.drop(columns=["p_winner_hier"])
    j.to_parquet(out_path)
    print(f"  wrote {out_path} ({len(j):,} rows)")
    return j


print("training window (1995-2023):")
add_hier_logit(rf"{DD}\tune_scratch\xgb_features_widedata_hier_1995_2023.parquet",
                rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet",
                rf"{DD}\tune_scratch\xgb_features_v7_train.parquet")

print("\nholdout window (2024-2025):")
add_hier_logit(rf"{DD}\tune_scratch\xgb_features_v2_holdout.parquet",
                rf"{DD}\tune_scratch\v25_postfix_holdout_preds.parquet",
                rf"{DD}\tune_scratch\xgb_features_v7_holdout.parquet")
