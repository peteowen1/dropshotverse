"""Diagnostic (read-only), fresh cuts #1 and #2:
  (B) wall-proximity proxy: no refit/replay of run() available without
      touching hierarchical_model.py, so use |d1_mean_nosurf| / |d2_mean_nosurf|
      (mu-only, no kappa/lambda - directly comparable to raw m_a/m_b magnitude
      since it's their difference) as a proxy for "is a component state pinned
      near the MAX_STATE=3.0 wall". A true single-sided proxy: since
      d1_mean_nosurf = sv.m_a - sl.m_b, a value near +/-6 requires at least one
      side near its own +/-3 wall; even a value near +/-4-5 is only reachable
      if at least one side is well past +/-1.5-2 already skewed toward a wall.
  (C) market-implied-probability-bucket calibration, Elo vs Hier vs Market,
      using the existing odds join (p_mkt) already merged into diag2_enriched.
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

    print("=== (B) wall-proximity proxy: max(|d1_mean_nosurf|, |d2_mean_nosurf|) ===")
    df["wall_proxy"] = df[["d1_mean_nosurf", "d2_mean_nosurf"]].abs().max(axis=1)
    print(df["wall_proxy"].describe())
    q = pd.qcut(df["wall_proxy"], 5, labels=[f"Q{i+1}" for i in range(5)])
    out = df.groupby(q, observed=True).agg(
        n=("wall_proxy", "size"), mean_wall_proxy=("wall_proxy", "mean"),
        hier_ll=("p_hier", ll), elo_ll=("p_elo", ll))
    out["gap"] = out["hier_ll"] - out["elo_ll"]
    print(out.round(4).to_string())
    print(f"\ncorrelation(wall_proxy, per-row hier logloss contribution - elo's): computed via bucket gap above")
    print(f"rows with wall_proxy > 4.0 (at least one side plausibly near a MAX_STATE wall): {(df['wall_proxy']>4.0).sum():,}")
    if (df['wall_proxy']>4.0).sum() > 20:
        g = df[df['wall_proxy']>4.0]
        print(f"  hier_ll={ll(g.p_hier):.4f}  elo_ll={ll(g.p_elo):.4f}  n={len(g)}")

    print("\n=== (C) market-implied-probability-bucket calibration (Elo / Hier / Market) ===")
    has_mkt = df["p_mkt"].notna()
    m = df[has_mkt].copy()
    print(f"n with market odds: {len(m):,}")
    m["mkt_fav_p"] = np.maximum(m["p_mkt"], 1 - m["p_mkt"])
    # which side does the market favor - is it the actual winner?
    m["mkt_favors_winner"] = m["p_mkt"] >= 0.5
    bucket = pd.cut(m["mkt_fav_p"], [.5, .6, .7, .8, .9, 1.0])
    for name, col in [("Elo", "p_elo"), ("Hier", "p_hier"), ("Market", "p_mkt")]:
        m[f"{name}_fav"] = np.where(m[col] >= .5, m[col], 1 - m[col])
        m[f"{name}_correct"] = (m[col] >= .5).astype(int)
    print("\n  --- bucketed by MARKET's own confidence (apples-to-apples slice for all three models) ---")
    for name in ["Elo", "Hier", "Market"]:
        out = m.groupby(bucket, observed=True).agg(
            n=(f"{name}_fav", "size"), predicted=(f"{name}_fav", "mean"),
            realised=(f"{name}_correct", "mean"))
        out["error"] = out["realised"] - out["predicted"]
        print(f"\n  {name}:")
        print(out.round(4).to_string())
    print("\n  --- log-loss by market-bucket (own probabilities, not fav-only) ---")
    for name, col in [("Elo", "p_elo"), ("Hier", "p_hier"), ("Market", "p_mkt")]:
        out = m.groupby(bucket, observed=True)[col].apply(ll)
        print(f"  {name}: {dict(out.round(4))}")
