"""Diagnostic (read-only), fresh cut: WTA tour churn / field depth (first
agent's #5, flagged unverified). Theory: WTA has historically had faster
ranking churn / less top-heavy stability than ATP, so a slow-reverting
random-walk skill model (OU_THETA_PER_WEEK now ~0, RW_VAR_PER_WEEK=0.005)
may fit ATP's more stable hierarchy better than WTA's, showing up as a
larger hier-vs-Elo gap on the WTA side specifically.

Uses prepared_matches_cache.pkl (full archive) to build a churn measure -
rolling-window distinct-player-count / concentration of match-wins among the
window's top players - and compares hier_v25 vs Elo on the 2024-25 holdout,
split by tour.
"""
import pickle

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"
CACHE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl"


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)
    df["p_hier"] = df["p_winner_hier_mc600"]

    print("=== basic tour split, current v25 config (converged mc600) ===")
    for tour, g in df.groupby("tour"):
        print(f"  tour={tour}  n={len(g):5d}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  "
              f"gap={ll(g.p_hier)-ll(g.p_elo):+.4f}  acc_hier={float((g.p_hier>.5).mean()):.4f}  acc_elo={float((g.p_elo>.5).mean()):.4f}")

    # nosec: this session's own prepared-archive cache, same trust basis as
    # every other diag*/diag2* script this session that loads it.
    with open(CACHE, "rb") as f:
        arc, _ = pickle.load(f)
    arc = arc[arc["level_file"] == "tour"].copy()
    arc["year"] = arc["date"].dt.year

    print("\n=== field-depth / churn measure: for each tour+year, top-16-players' share of TOTAL match-wins ===")
    print("    (lower share = more churn/depth, wins spread across more players)")
    rows = []
    for (tour, year), g in arc.groupby(["tour", "year"]):
        wc = g["winner_uid"].value_counts()
        total_wins = wc.sum()
        top16_share = wc.head(16).sum() / total_wins if total_wins else np.nan
        n_distinct_winners = g["winner_uid"].nunique()
        rows.append({"tour": tour, "year": year, "top16_share": top16_share,
                     "n_distinct_winners": n_distinct_winners, "n_matches": len(g)})
    churn = pd.DataFrame(rows)
    recent = churn[churn.year.between(2022, 2025)]
    print(recent.pivot_table(index="year", columns="tour", values="top16_share").round(4).to_string())
    print("\ndistinct winners per year (higher = deeper/more churny field):")
    print(recent.pivot_table(index="year", columns="tour", values="n_distinct_winners").round(0).to_string())

    print("\n=== hier vs Elo gap by tour, split further by whether the actual match WINNER is a")
    print("    'new-ish' player in the archive (n_prior < 50) - the population most affected by churn ===")
    for tour, g in df.groupby("tour"):
        newish = g["n_prior_w"] < 50
        for label, mask in [("established winner (n_prior_w>=50)", ~newish), ("newer winner (n_prior_w<50)", newish)]:
            gg = g[mask]
            if len(gg) < 50:
                continue
            print(f"  tour={tour:4s} {label:38s} n={len(gg):5d}  hier_ll={ll(gg.p_hier):.4f}  "
                  f"elo_ll={ll(gg.p_elo):.4f}  gap={ll(gg.p_hier)-ll(gg.p_elo):+.4f}")
