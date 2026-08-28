"""Diagnostic (read-only), ticket 20 continuation: does the extremization
undershoot found in the pooled/WTA-heavy data also show up WITHIN ATP alone,
on ATP's own most lopsided matches? ATP is at rough parity in AGGREGATE
(diag2_wta_churn.py: gap=-0.0018) but that could still hide an extremization
undershoot on ATP's own blowouts if something else in ATP is overperforming
to compensate (e.g. hier doing unusually well on ATP's competitive matches,
per diag2_wta_churn's own "newer winner" finding: hier beats Elo by 0.34
log-loss on ATP matches with a new/surprise winner, n=99).

Uses diag2_enriched.parquet (already has p_elo, p_mkt, d1_mean/d2_mean,
d1_var/d2_var, p_winner_hier (point estimate), p_winner_hier_mc600). No
refit, no mc_average call, no modification to hierarchical_model.py.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)
    df["p_hier"] = df["p_winner_hier_mc600"]
    df["p_hier_pt"] = df["p_winner_hier"]   # point estimate, no MC

    # hier's own "combined strength" logit gap for the eventual WINNER, on the
    # SAME single-scalar footing as Elo's implied gap: d1_mean is the winner's
    # serve-vs-loser's-return point-level logit edge, -d2_mean is the winner's
    # return-vs-loser's-serve edge (d2_mean = loser's edge serving against the
    # winner). Average of the two = winner's overall point-level dominance,
    # comparable in spirit to a single Elo rating difference.
    df["hier_combined_gap"] = (df["d1_mean"] - df["d2_mean"]) / 2.0
    df["elo_logit_gap"] = logit(df["p_elo"])
    df["combined_var"] = df["d1_var"] + df["d2_var"]

    print("=" * 100)
    print("PART A: ATP-only market-confidence-bucket calibration (external truth var, not model's own axis)")
    print("=" * 100)
    for tour_filter in ["atp", "wta", None]:
        sub = df if tour_filter is None else df[df.tour == tour_filter]
        has_mkt = sub["p_mkt"].notna()
        m = sub[has_mkt].copy()
        if len(m) < 30:
            continue
        m["mkt_fav_p"] = np.maximum(m["p_mkt"], 1 - m["p_mkt"])
        bucket = pd.cut(m["mkt_fav_p"], [.5, .6, .7, .8, .9, 1.0])
        for name, col in [("Elo", "p_elo"), ("Hier", "p_hier")]:
            m[f"{name}_fav"] = np.where(m[col] >= .5, m[col], 1 - m[col])
            m[f"{name}_correct"] = (m[col] >= .5).astype(int)
        label = tour_filter.upper() if tour_filter else "POOLED"
        print(f"\n--- {label} (n with market odds = {len(m):,}) ---")
        for name in ["Elo", "Hier"]:
            out = m.groupby(bucket, observed=True).agg(
                n=(f"{name}_fav", "size"), predicted=(f"{name}_fav", "mean"),
                realised=(f"{name}_correct", "mean"))
            out["error"] = out["realised"] - out["predicted"]
            print(f"  {name}:")
            print("  " + out.round(4).to_string().replace("\n", "\n  "))
        # log-loss by bucket, hier vs elo, gap column
        out_e = m.groupby(bucket, observed=True)["p_elo"].apply(ll)
        out_h = m.groupby(bucket, observed=True)["p_hier"].apply(ll)
        gap = (out_h - out_e).round(4)
        print(f"  log-loss gap (hier-elo) by bucket: {dict(gap)}")

    print()
    print("=" * 100)
    print("PART B: hier's own combined-strength logit gap vs Elo's implied logit gap, ATP vs WTA,")
    print("        restricted to genuinely lopsided matches by ELO's own confidence (p_elo>0.9)")
    print("=" * 100)
    for tour_filter in ["atp", "wta"]:
        sub = df[(df.tour == tour_filter) & (df.p_elo > 0.9)]
        ratio = (sub["hier_combined_gap"].abs() / sub["elo_logit_gap"].abs())
        print(f"\n--- {tour_filter.upper()}, p_elo>0.9, n={len(sub):,} ---")
        print(f"  mean |elo_logit_gap|          = {sub['elo_logit_gap'].abs().mean():.3f}")
        print(f"  mean |hier_combined_gap|      = {sub['hier_combined_gap'].abs().mean():.3f}")
        print(f"  mean ratio |hier|/|elo| (per-row) = {ratio.mean():.3f}  median={ratio.median():.3f}")
        print(f"  mean combined_var (d1_var+d2_var) = {sub['combined_var'].mean():.4f}")
        print(f"  mean p_hier (mc600) = {sub['p_hier'].mean():.4f}   mean p_hier_pt (point est) = {sub['p_hier_pt'].mean():.4f}   mean p_elo = {sub['p_elo'].mean():.4f}")
        print(f"  logloss: hier(mc600)={ll(sub.p_hier):.4f}  hier(point-est)={ll(sub.p_hier_pt):.4f}  elo={ll(sub.p_elo):.4f}")

    print()
    print("=" * 100)
    print("PART B2: same ratio/gap comparison across ALL rows (not just p_elo>0.9), binned by")
    print("         elo_logit_gap magnitude, ATP only - is the compression constant (linear/global")
    print("         shrinkage) or does it specifically widen at the extreme end (localized undershoot)?")
    print("=" * 100)
    for tour_filter in ["atp", "wta"]:
        sub = df[df.tour == tour_filter].copy()
        sub["abs_elo_gap"] = sub["elo_logit_gap"].abs()
        q = pd.qcut(sub["abs_elo_gap"], 6, duplicates="drop")
        out = sub.groupby(q, observed=True).agg(
            n=("abs_elo_gap", "size"),
            mean_abs_elo_gap=("abs_elo_gap", "mean"),
            mean_abs_hier_gap=("hier_combined_gap", lambda s: s.abs().mean()),
            mean_combined_var=("combined_var", "mean"),
            hier_ll=("p_hier", ll), elo_ll=("p_elo", ll),
        )
        out["hier/elo_gap_ratio"] = out["mean_abs_hier_gap"] / out["mean_abs_elo_gap"]
        out["ll_gap"] = out["hier_ll"] - out["elo_ll"]
        print(f"\n--- {tour_filter.upper()} ---")
        print(out.round(4).to_string())

    print()
    print("=" * 100)
    print("PART C: MC-convexity cost specifically on ATP's most lopsided matches -")
    print("        does point-estimate probability track Elo/market much more closely than the")
    print("        MC-averaged probability in this slice? (tests the Jensen's-gap hypothesis)")
    print("=" * 100)
    for tour_filter in ["atp", "wta"]:
        sub = df[df.tour == tour_filter].copy()
        sub["mkt_or_elo_conf"] = sub["p_elo"]  # elo as the external-ish confidence axis (available for all rows)
        bucket = pd.cut(sub["mkt_or_elo_conf"], [.5, .6, .7, .8, .9, 1.0])
        g = sub.groupby(bucket, observed=True).agg(
            n=("p_elo", "size"),
            mean_p_elo=("p_elo", "mean"),
            mean_p_hier_pt=("p_hier_pt", "mean"),
            mean_p_hier_mc600=("p_hier", "mean"),
            mean_combined_var=("combined_var", "mean"),
        )
        g["mc_cost"] = g["mean_p_hier_pt"] - g["mean_p_hier_mc600"]
        print(f"\n--- {tour_filter.upper()} (bucketed by p_elo) ---")
        print(g.round(4).to_string())

    print()
    print("=" * 100)
    print("PART D: trace individual ATP blowouts (p_elo>0.9), top 5 by elo confidence, show")
    print("        hier's combined gap / var / mc-cost / regime mix isn't visible per-row but")
    print("        n_prior (experience) and layoff are")
    print("=" * 100)
    sub = df[(df.tour == "atp") & (df.p_elo > 0.9)].copy()
    sub = sub.sort_values("p_elo", ascending=False).head(8)
    cols = ["date", "winner_name", "loser_name", "p_elo", "p_hier_pt", "p_hier",
            "hier_combined_gap", "elo_logit_gap", "combined_var", "n_prior_w", "n_prior_l"]
    print(sub[cols].round(3).to_string(index=False))
