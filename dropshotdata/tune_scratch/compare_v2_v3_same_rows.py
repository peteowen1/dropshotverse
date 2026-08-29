"""stats-discipline #5.3: v3's holdout population (n=8,294) differs from v2's
(n=8,591) because serve/return_form has ~6% missingness. Score v2's own
predictions restricted to the IDENTICAL rows v3 was scored on, before
trusting any v2-vs-v3 delta."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
v2 = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v2_holdout_predictions.parquet")
v3 = pd.read_parquet(rf"{DD}\tune_scratch\xgb_v3_holdout_predictions.parquet")

key = ["date", "tour", "winner_uid", "loser_uid"]
v3_keys = v3[key].drop_duplicates()
v2_same = v2.merge(v3_keys, on=key, how="inner")
print(f"v2 full n={len(v2):,}  v3 n={len(v3):,}  v2-restricted-to-v3-rows n={len(v2_same):,}")

diag2 = pd.read_parquet(rf"{DD}\tune_scratch\diag2_enriched.parquet")


def score_all(df, label):
    j = df.merge(diag2[key + ["p_mkt"]].drop_duplicates(subset=key), on=key, how="left")
    y = j["y"].values.astype(int)
    p_mkt = np.where(y == 1, j["p_mkt"].values, 1 - j["p_mkt"].values)
    for name, p in ((label, j["p_xgb"].values), ("Market (same rows)", p_mkt)):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        acc = ((p > 0.5).astype(int) == y).mean()
        print(f"  {name:26s} n={len(y):,}  logloss={ll:.5f}  acc={acc:.4f}")


print("\n=== v2 restricted to v3's exact rows (n=8,294) ===")
score_all(v2_same, "XGBoost v2 (same rows as v3)")
print("\n=== v3, its own native rows (n=8,294) ===")
score_all(v3, "XGBoost v3")
