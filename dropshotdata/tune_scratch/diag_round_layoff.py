"""Diagnostic (read-only): test the hypothesis raised by diag_slices.py's
layoff-bucket calibration table - that hier's calibration flips sign with
layoff length (underconfident at <=1wk gap i.e. mid-tournament, overconfident
at 1-12wk gap i.e. start of a new tournament) because "weeks since last match"
is confounded with tournament round: round 1 matches almost always follow a
gap of >=1wk (previous tournament ended), while round 3+ matches follow <=1wk
(previous round of the SAME tournament, and by construction only players who
just won reach them - a survivorship/momentum signal the static mu/kappa
state does not carry).

Depends on tune_scratch/diag_joined.parquet (from diag_base_repro.py) and the
raw prepared_matches_cache.pkl (for the 'round' column, not present in the
hier eval parquet).
"""
import pickle

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)

JOINED = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_joined.parquet"
CACHE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl"

EARLY = {"R128", "R64", "R32", "Q1", "Q2", "Q3", "Q4", "Q5"}
LATE = {"R16", "QF", "SF", "F", "RR", "BR"}


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    j = pd.read_parquet(JOINED).rename(columns={"p_winner_hier_mc": "p_hier"})
    # nosec: same session's own cache, per task brief - safe to load.
    with open(CACHE, "rb") as f:
        arc, _ = pickle.load(f)
    arc_small = arc[["date", "winner_uid", "loser_uid", "round"]].drop_duplicates(
        subset=["date", "winner_uid", "loser_uid"])
    j = j.merge(arc_small, on=["date", "winner_uid", "loser_uid"], how="left")
    print(f"round merge coverage: {j['round'].notna().sum():,}/{len(j):,}")

    lay = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_layoff.parquet")
    j = j.merge(lay, on=["date", "winner_uid", "loser_uid"], how="left")
    j["max_layoff_weeks"] = j[["weeks_since_last_w", "weeks_since_last_l"]].max(axis=1)
    j["layoff_bucket"] = pd.cut(j["max_layoff_weeks"], [-0.01, 1, 4, 12, 1e6],
                                 labels=["<=1wk", "1-4wk", "4-12wk", ">12wk"])

    j["round_group"] = np.where(j["round"].isin(EARLY), "early(R32-Q)",
                        np.where(j["round"].isin(LATE), "late(R16+)", "other"))

    print("\n=== cross-tab: layoff bucket vs round group (confirms the confound) ===")
    print(pd.crosstab(j["layoff_bucket"], j["round_group"]))

    print("\n=== hier calibration/log-loss by round_group (holding layoff confound in view) ===")
    for rg, g in j.groupby("round_group"):
        if len(g) < 100:
            continue
        g = g.copy()
        g["p_fav"] = np.where(g.p_hier >= .5, g.p_hier, 1 - g.p_hier)
        g["fav_won"] = (g.p_hier >= .5).astype(int)
        print(f"\n  round_group={rg}  n={len(g)}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  "
              f"diff={ll(g.p_hier)-ll(g.p_elo):+.4f}")
        b = pd.cut(g["p_fav"], [.5, .6, .7, .8, .9, 1.0])
        out = g.groupby(b, observed=True).agg(n=("p_fav", "size"), predicted=("p_fav", "mean"),
                                                realised=("fav_won", "mean"))
        out["error"] = out["realised"] - out["predicted"]
        print(out.round(4).to_string())

    print("\n=== same, but now hold ROUND GROUP FIXED and vary layoff bucket within it ===")
    print("    (if the effect is really about round/survivorship, layoff should stop mattering once round is fixed;")
    print("     if layoff still flips sign within a fixed round group, layoff itself carries independent signal)")
    for rg in ["early(R32-Q)", "late(R16+)"]:
        sub = j[j.round_group == rg]
        for lb, g in sub.groupby("layoff_bucket", observed=True):
            if len(g) < 80:
                continue
            g = g.copy()
            g["p_fav"] = np.where(g.p_hier >= .5, g.p_hier, 1 - g.p_hier)
            g["fav_won"] = (g.p_hier >= .5).astype(int)
            err = (g["fav_won"].mean() - g["p_fav"].mean())
            print(f"  round_group={rg:14s} layoff={str(lb):8s} n={len(g):5d}  "
                  f"predicted={g['p_fav'].mean():.4f}  realised={g['fav_won'].mean():.4f}  error={err:+.4f}  "
                  f"hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}")
