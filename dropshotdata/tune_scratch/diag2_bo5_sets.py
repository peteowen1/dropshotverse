"""Diagnostic (read-only), fresh cut: within best_of=5 (Grand Slam) matches,
does the hier-vs-Elo gap concentrate in matches that go the distance (3-2/
2-3) vs straight-sets wins? First agent's #2 flagged Bo5 as a real gap
(n=847) but didn't break it down by how many sets were played. Needs the
'score' column (not in the hier eval parquet) - joined from
prepared_matches_cache.pkl by (date, winner_uid, loser_uid).
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


def n_sets(score):
    toks = [t for t in str(score).split() if "-" in t]
    return len(toks)


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)
    df["p_hier"] = df["p_winner_hier_mc600"]

    # nosec: this session's own prepared-archive cache, same trust basis as
    # every other diag*/diag2* script this session that loads it.
    with open(CACHE, "rb") as f:
        arc, _ = pickle.load(f)
    arc_small = arc[["date", "winner_uid", "loser_uid", "score"]].drop_duplicates(
        subset=["date", "winner_uid", "loser_uid"])
    df = df.merge(arc_small, on=["date", "winner_uid", "loser_uid"], how="left")
    df["n_sets_played"] = df["score"].map(n_sets)

    bo5 = df[df.best_of == 5]
    print(f"Bo5 total n={len(bo5)}")
    print(f"  overall: hier_ll={ll(bo5.p_hier):.4f}  elo_ll={ll(bo5.p_elo):.4f}  gap={ll(bo5.p_hier)-ll(bo5.p_elo):+.4f}")
    print("\nbreakdown by number of sets actually played:")
    for ns, g in bo5.groupby("n_sets_played"):
        if len(g) < 20:
            continue
        acc_h = float((g.p_hier > .5).mean())
        acc_e = float((g.p_elo > .5).mean())
        print(f"  {ns}-set matches  n={len(g):4d}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  "
              f"gap={ll(g.p_hier)-ll(g.p_elo):+.4f}  acc_hier={acc_h:.4f}  acc_elo={acc_e:.4f}")

    print("\n'went the distance' (4-5 sets) vs straight-ish (3 sets) grouping:")
    bo5 = bo5.copy()
    bo5["distance"] = np.where(bo5["n_sets_played"] >= 4, "went 4-5 sets", "straight 3 sets")
    for label, g in bo5.groupby("distance"):
        print(f"  {label:16s} n={len(g):4d}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  "
              f"gap={ll(g.p_hier)-ll(g.p_elo):+.4f}")

    print("\nfor comparison, Bo3 matches split the same way (2 sets vs 3 sets):")
    bo3 = df[df.best_of == 3].copy()
    bo3["distance"] = np.where(bo3["n_sets_played"] >= 3, "went 3 sets", "straight 2 sets")
    for label, g in bo3.groupby("distance"):
        print(f"  {label:16s} n={len(g):5d}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  "
              f"gap={ll(g.p_hier)-ll(g.p_elo):+.4f}")
