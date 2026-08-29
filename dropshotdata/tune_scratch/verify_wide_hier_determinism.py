"""Sanity check before trusting Arm C: the wide (2015-2023) hier run and the
original v2 (2022-23-only) hier run both replay the SAME full history up to
2022 with the SAME frozen v25 hyperparameters - they should produce
IDENTICAL d1_mean/d1_var/d2_mean/d2_var for the 2022-23 rows they share.
If they don't match, something about the wide run differs from what v2
actually used, and Arm C's result can't be trusted as "same model, more
data" - it would be "different model."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata\tune_scratch"
wide = pd.read_parquet(rf"{DD}\v25_postfix_wide_preds.parquet")
orig = pd.read_parquet(rf"{DD}\v25_postfix_tune_preds.parquet")   # v2's original 2022-23 source

key = ["date", "tour", "winner_uid", "loser_uid"]
wide_2223 = wide[wide["date"].dt.year.isin([2022, 2023])][key + ["d1_mean", "d1_var", "d2_mean", "d2_var"]]
orig_small = orig[key + ["d1_mean", "d1_var", "d2_mean", "d2_var"]]

j = wide_2223.merge(orig_small, on=key, suffixes=("_wide", "_orig"), how="inner")
print(f"matched rows: {len(j):,} (wide 2022-23: {len(wide_2223):,}, orig: {len(orig_small):,})")

for col in ("d1_mean", "d1_var", "d2_mean", "d2_var"):
    diff = (j[f"{col}_wide"] - j[f"{col}_orig"]).abs()
    print(f"  {col}: max abs diff = {diff.max():.2e}  mean abs diff = {diff.mean():.2e}")
