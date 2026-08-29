"""Ticket 28 round 12, Arm C: wide-window (2015-2023) Elo+engineered features
PLUS hier state, now that v25_postfix_wide_preds.parquet exists. Merges hier's
d1_mean/d1_var/d2_mean/d2_var onto the already-built wide feature set, using
the SAME side-assignment convention (y already fixed from build_widedata_
features.py's own hash) - so d1/d2 (winner/loser-keyed) must be re-mapped
to sideA/sideB using the SAME a_is_winner logic, recovered from y itself
(y==1 means side A is the winner)."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
wide = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_2015_2023.parquet")
hier = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide_preds.parquet")[
    ["date", "tour", "winner_uid", "loser_uid", "d1_mean", "d1_var", "d2_mean", "d2_var"]]
hier = hier.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])

j = wide.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
n0 = len(j)
miss = j[["d1_mean", "d1_var", "d2_mean", "d2_var"]].isna().any(axis=1)
print(f"{n0:,} -> {int((~miss).sum()):,} after dropping {int(miss.sum()):,} missing-hier rows "
      f"({miss.mean():.4%})")
j = j[~miss].reset_index(drop=True)

y = j["y"].values.astype(int)
d1m, d2m = j["d1_mean"].values.astype(float), j["d2_mean"].values.astype(float)
d1v, d2v = j["d1_var"].values.astype(float), j["d2_var"].values.astype(float)
# d1/d2 are winner/loser-keyed (hierarchical_model.py's own convention, same
# as build_xgb_features.py's HIER_TUNE/HIER_HOLDOUT source) - side A gets d1
# when side A is the winner (y==1), matching every other winner/loser-keyed
# column's treatment throughout this session.
j["hier_dmean_diff"] = np.where(y == 1, d1m - d2m, d2m - d1m)
j["hier_dvar_diff"] = np.where(y == 1, d1v - d2v, d2v - d1v)
j["hier_dvar_sum"] = d1v + d2v

j.to_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_2015_2023.parquet")
print(f"saved {len(j):,} rows")
