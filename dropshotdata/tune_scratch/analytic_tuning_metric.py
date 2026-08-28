"""Issue 20: score analytic_correction() against the exact tuning-window
weighted/lopsided metric (tune_v3_lopsided.py's join+scoring, reused
verbatim via roundorder_score_v3metric.py's pattern) on the 2022-23
tuning window, alongside mc_average(n=60) (the harness's normal path) for
direct comparison on the SAME odds-joined rows.

NOTE: analytic_correction() FAILED the faithfulness-to-high-precision-MC
validation (analytic_validate.py: mean abs diff 0.183 vs a high-precision
mc(n=2000) reference, ~5x the mc(n=60)-vs-mc(n=2000) pure sampling-noise
floor of 0.034, with individual disagreements up to 0.67 and several rows
clipping to 0.0/1.0). Per charter this means analytic_correction is NOT a
validated candidate this session should believe in - this script is run
anyway, for documentation/completeness only (to quantify the practical
impact of the validation failure), NOT as a step toward a holdout reveal.
No holdout reveal follows this script."""
import sys
import time
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from mc_parallel import mc_average, analytic_correction
from market_compare import key_from_full, key_from_short, devig_power, load_odds

TUNE_YEARS = (2022, 2023)
LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0

PREDS = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_tune_predictions.parquet"


def score(pr, weight, lop_mask):
    pr = np.clip(pr, 1e-9, 1 - 1e-9)
    ll = -np.log(pr)
    weighted_logloss = float(np.sum(weight * ll) / np.sum(weight))
    plain_logloss = float(ll.mean())
    lop_logloss = float(ll[lop_mask].mean()) if lop_mask.any() else float("nan")
    return weighted_logloss, plain_logloss, lop_logloss


if __name__ == "__main__":
    p = pd.read_parquet(PREDS)
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
    lop_mask = m["mkt_fav"].values > LOPSIDED_THRESHOLD
    print(f"n={len(m)}, lopsided n={lop_mask.sum()}")

    t0 = time.time()
    mc60 = mc_average(m, n_samples=60, seed_base=0)
    t_mc = time.time() - t0
    t0 = time.time()
    an = analytic_correction(m, h=0.01)
    t_an = time.time() - t0

    w_mc, p_mc, l_mc = score(mc60, weight, lop_mask)
    w_an, p_an, l_an = score(an, weight, lop_mask)

    print(f"\nmc_average(n=60):        weighted={w_mc:.4f}  plain={p_mc:.4f}  "
         f"lopsided={l_mc:.4f}   ({t_mc:.1f}s, {t_mc/len(m)*1000:.2f}ms/row)")
    print(f"analytic_correction:     weighted={w_an:.4f}  plain={p_an:.4f}  "
         f"lopsided={l_an:.4f}   ({t_an:.1f}s, {t_an/len(m)*1000:.2f}ms/row)")
    print(f"speedup (analytic vs mc60 on this join): {t_mc/t_an:.1f}x")
    print(f"\ndelta (analytic - mc60): weighted={w_an-w_mc:+.4f}  "
         f"plain={p_an-p_mc:+.4f}  lopsided={l_an-l_mc:+.4f}")

    n_clipped = int(((an <= 1e-9) | (an >= 1 - 1e-9)).sum())
    print(f"\nrows where analytic hit the [0,1] safety clip: {n_clipped} / {len(m)} "
         f"({n_clipped/len(m)*100:.1f}%)")
