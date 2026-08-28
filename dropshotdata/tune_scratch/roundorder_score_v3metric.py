"""Score the round-order-fix v25 fit (already produced by hierarchical_model.py
with DSV_EVAL_YEARS=2022,2023) against the EXACT weighted/lopsided tuning-window
metric from tune_v3_lopsided.py's run_candidate() - the metric v25 was actually
selected on. Reuses that file's join/scoring logic verbatim (not re-derived)."""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import pickle
import numpy as np
import pandas as pd
from hierarchical_model import p_match, sigmoid
from market_compare import key_from_full, key_from_short, devig_power, load_odds

TUNE_YEARS = (2022, 2023)
LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0

preds_path = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_tune_predictions.parquet"
state_path = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_tune_state.pkl"

# corruption gate
with open(state_path, "rb") as f:
    st = pickle.load(f)   # nosec: our own state file, written moments ago by our own fit
vals = [v[0] for v in st["players"].values()] + [v[2] for v in st["players"].values()]
corrupt = sum(1 for x in vals if abs(x) > 2.5) / len(vals)
print(f"corruption_rate = {corrupt*100:.5f}% (gate: <= 0.0201%)")

p = pd.read_parquet(preds_path)
print(f"predictions rows: {len(p)}")
bad = p["date"].dt.year >= 2024
assert not bad.any(), f"LEAKAGE: touched {bad.sum()} rows >= 2024"

p["k_w"] = p["winner_name"].map(key_from_full)
p["k_l"] = p["loser_name"].map(key_from_full)
p["pair"] = p.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

odds = load_odds(years=TUNE_YEARS).dropna(subset=["Winner", "Loser", "Date"])
odds["k_w"] = odds["Winner"].map(key_from_short)
odds["k_l"] = odds["Loser"].map(key_from_short)
odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
odds = odds.dropna(subset=["date"])
odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

m = p.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
           on=["pair", "tour"], how="left", suffixes=("", "_o"))
m["gap"] = (m["date_o"] - m["date"]).dt.days
m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
m = m[(m["k_w_o"] == m["k_w"]) & m[["PSW", "PSL"]].notna().all(axis=1)]
print(f"joined rows (with odds): {len(m)}")

m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
m = m.dropna(subset=["p_mkt"])
m["mkt_fav"] = np.where(m.p_mkt >= .5, m.p_mkt, 1 - m.p_mkt)
weight = np.where(m["mkt_fav"].values > LOPSIDED_THRESHOLD, LOPSIDED_WEIGHT, 1.0)

rng = np.random.default_rng(0)
mc = np.empty(len(m))
for i, row in enumerate(m.itertuples()):
    d1 = rng.normal(row.d1_mean, np.sqrt(max(row.d1_var, 1e-6)), 60)
    d2 = rng.normal(row.d2_mean, np.sqrt(max(row.d2_var, 1e-6)), 60)
    ps = [p_match(sigmoid(a), sigmoid(b), int(row.best_of)) for a, b in zip(d1, d2)]
    mc[i] = np.mean(ps)

pr = np.clip(mc, 1e-9, 1 - 1e-9)
ll = -np.log(pr)
weighted_logloss = float(np.sum(weight * ll) / np.sum(weight))
plain_logloss = float(ll.mean())
lop_mask = m["mkt_fav"].values > LOPSIDED_THRESHOLD
lop_logloss = float(ll[lop_mask].mean()) if lop_mask.any() else float("nan")

print(f"\nn={len(m)}")
print(f"weighted_logloss = {weighted_logloss:.4f}  (v25 pre-fix baseline: 0.5759 per RESUME round 7)")
print(f"plain_logloss    = {plain_logloss:.4f}")
print(f"lopsided_logloss = {lop_logloss:.4f}")
