"""Diagnostic (read-only): re-run the layoff-length calibration table using
the CONVERGED mc600 probability column (not the biased mc60 the first
diagnostic agent used), and break it down further by tourney_level (elite vs
non-elite) and which side had the gap, to find exactly how narrow/wide the
original calibration finding really is. No mc_average recomputation needed -
uses the p_winner_hier_mc600 column already baked into hier_v25_eval_predictions.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)
    df["p_hier"] = df["p_winner_hier_mc600"]
    df["p_fav"] = np.where(df.p_hier >= .5, df.p_hier, 1 - df.p_hier)
    df["fav_won"] = (df.p_hier >= .5).astype(int)
    df["layoff_bucket"] = pd.cut(df["max_layoff_weeks"], [-0.01, 1, 4, 12, 1e6],
                                  labels=["<=1wk", "1-4wk", "4-12wk", ">12wk"])

    print("=== (converged mc600) calibration by layoff bucket, ALL eval rows ===")
    out = df.groupby("layoff_bucket", observed=True).agg(
        n=("p_fav", "size"), predicted=("p_fav", "mean"), realised=("fav_won", "mean"),
        hier_ll=("p_hier", ll), elo_ll=("p_elo", ll))
    out["error"] = out["realised"] - out["predicted"]
    print(out.round(4).to_string())

    print("\n=== same, split by is_elite (calibration finding was diagnosed on ALL tour matches -")
    print("    does it concentrate in elite (Slam/Masters/Finals/PM/Olympics) or non-elite tour matches?) ===")
    for elite, g in df.groupby("is_elite"):
        print(f"\n  is_elite={elite}  n={len(g)}")
        out = g.groupby("layoff_bucket", observed=True).agg(
            n=("p_fav", "size"), predicted=("p_fav", "mean"), realised=("fav_won", "mean"))
        out["error"] = out["realised"] - out["predicted"]
        print(out.round(4).to_string())

    print("\n=== split by WHICH side had the short layoff (favorite's side vs underdog's side) ===")
    df["fav_is_winner"] = df["fav_won"] == 1
    # did the match FAVORITE (per hier) have the short layoff, or the underdog?
    df["fav_had_layoff"] = np.where(df["fav_is_winner"], df["winner_had_short_layoff"], df["loser_had_short_layoff"])
    df["dog_had_layoff"] = np.where(df["fav_is_winner"], df["loser_had_short_layoff"], df["winner_had_short_layoff"])
    for label, mask in [("favorite had 1-12wk gap", df["fav_had_layoff"] & ~df["dog_had_layoff"]),
                         ("underdog had 1-12wk gap", df["dog_had_layoff"] & ~df["fav_had_layoff"]),
                         ("both had 1-12wk gap", df["fav_had_layoff"] & df["dog_had_layoff"]),
                         ("neither", ~df["fav_had_layoff"] & ~df["dog_had_layoff"])]:
        g = df[mask]
        if len(g) < 30:
            continue
        err = g["fav_won"].mean() - g["p_fav"].mean()
        print(f"  {label:28s} n={len(g):5d}  predicted={g['p_fav'].mean():.4f}  "
              f"realised={g['fav_won'].mean():.4f}  error={err:+.4f}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}")

    print("\n=== best_of split within the 1-12wk-gap bucket (does Bo5 concentrate the effect?) ===")
    short = df[df["max_layoff_weeks"].between(1, 12)]
    for bo, g in short.groupby("best_of"):
        if len(g) < 30:
            continue
        err = g["fav_won"].mean() - g["p_fav"].mean()
        print(f"  best_of={bo}  n={len(g):5d}  predicted={g['p_fav'].mean():.4f}  realised={g['fav_won'].mean():.4f}  "
              f"error={err:+.4f}  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}")

    print("\n=== established (n_prior on affected side) vs new-ish players, within 1-12wk-gap bucket ===")
    short = short.copy()
    short["affected_n_prior"] = np.where(short["winner_had_short_layoff"], short["n_prior_w"],
                                  np.where(short["loser_had_short_layoff"], short["n_prior_l"], np.nan))
    short["exp_bucket"] = pd.cut(short["affected_n_prior"], [-1, 20, 50, 150, 1e6],
                                  labels=["<20", "20-50", "50-150", "150+"])
    out = short.groupby("exp_bucket", observed=True).agg(
        n=("p_fav", "size"), predicted=("p_fav", "mean"), realised=("fav_won", "mean"),
        hier_ll=("p_hier", ll), elo_ll=("p_elo", ll))
    out["error"] = out["realised"] - out["predicted"]
    print(out.round(4).to_string())
