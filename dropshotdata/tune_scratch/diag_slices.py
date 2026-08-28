"""Diagnostic (read-only): cut the matched Elo/Hier/Market eval set (n=8,594,
2024-25 holdout, hier_v25) many independent ways and report log-loss +
calibration for each. Depends on tune_scratch/diag_joined.parquet (from
diag_base_repro.py) and tune_scratch/diag_layoff.parquet (from
diag_layoff_features.py). Writes only printed output.
"""
import numpy as np
import pandas as pd

JOINED = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_joined.parquet"
LAYOFF = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_layoff.parquet"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def ll_series(p):
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    return -np.log(p)


def summarize(df, group_col, min_n=40, cols=("p_elo", "p_hier", "p_mkt")):
    rows = []
    for key, g in df.groupby(group_col, observed=True):
        if len(g) < min_n:
            continue
        row = {group_col: key, "n": len(g)}
        for c in cols:
            row[f"{c}_ll"] = ll_series(g[c]).mean()
            row[f"{c}_acc"] = (g[c] > 0.5).mean()
        row["hier_minus_elo_ll"] = row["p_hier_ll"] - row["p_elo_ll"]
        row["hier_minus_mkt_ll"] = row["p_hier_ll"] - row["p_mkt_ll"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group_col)
    return out


def calibration(df, prob_col, label, bins=(.5, .6, .7, .8, .9, 1.0)):
    """Symmetric favourite-strength calibration: for each match take the
    favourite-side probability (>=0.5) of `prob_col` and check whether the
    favourite actually won at the rate predicted."""
    d = df.copy()
    d["p_fav"] = np.where(d[prob_col] >= 0.5, d[prob_col], 1 - d[prob_col])
    d["fav_is_actual_winner"] = (d[prob_col] >= 0.5).astype(int)  # p_* is P(actual winner)
    d["b"] = pd.cut(d["p_fav"], bins)
    out = d.groupby("b", observed=True).agg(
        n=("p_fav", "size"), predicted=("p_fav", "mean"),
        realised=("fav_is_actual_winner", "mean"))
    out["error"] = out["realised"] - out["predicted"]
    print(f"\n-- calibration: {label} --")
    print(out.round(4).to_string())


if __name__ == "__main__":
    j = pd.read_parquet(JOINED)
    j = j.rename(columns={"p_winner_hier_mc": "p_hier"})
    print(f"loaded diag_joined: {len(j):,} rows")

    lay = pd.read_parquet(LAYOFF)
    j = j.merge(lay, on=["date", "winner_uid", "loser_uid"], how="left")
    print(f"layoff merge coverage: {j['weeks_since_last_w'].notna().sum():,}/{len(j):,} winner-side, "
          f"{j['weeks_since_last_l'].notna().sum():,}/{len(j):,} loser-side")

    # ============================================================ CUT 1: market-implied favourite strength
    j["mkt_fav_p"] = np.where(j.p_mkt >= .5, j.p_mkt, 1 - j.p_mkt)
    j["mkt_bucket"] = pd.cut(j["mkt_fav_p"], [.5, .6, .7, .8, .9, 1.0])
    print("\n=== CUT 1: market-implied favourite strength ===")
    print(summarize(j, "mkt_bucket").round(4).to_string(index=False))

    # ============================================================ CUT 2: surface
    print("\n=== CUT 2: surface ===")
    print(summarize(j, "surface").round(4).to_string(index=False))

    # ============================================================ CUT 3: tour
    print("\n=== CUT 3: tour ===")
    print(summarize(j, "tour").round(4).to_string(index=False))

    # ============================================================ CUT 4: best_of
    print("\n=== CUT 4: best_of ===")
    print(summarize(j, "best_of").round(4).to_string(index=False))

    # ============================================================ CUT 5: hier posterior variance
    j["hier_var_total"] = j["d1_var"] + j["d2_var"]
    j["var_bucket"] = pd.qcut(j["hier_var_total"], 5, duplicates="drop")
    print("\n=== CUT 5: hier posterior variance (d1_var+d2_var), quintiles ===")
    print(summarize(j, "var_bucket").round(4).to_string(index=False))
    print("  (does higher variance correlate with worse hier log-loss RELATIVE to elo?")
    print("   if yes and hier isn't flattening its probs enough there, variance is under-used in p_match.)")

    # ============================================================ CUT 6: layoff (winner side, loser side, max of both)
    j["max_layoff_weeks"] = j[["weeks_since_last_w", "weeks_since_last_l"]].max(axis=1)
    j["min_layoff_weeks"] = j[["weeks_since_last_w", "weeks_since_last_l"]].min(axis=1)
    j["layoff_bucket"] = pd.cut(j["max_layoff_weeks"],
                                 [-0.01, 1, 4, 12, 26, 52, 1e6],
                                 labels=["<=1wk", "1-4wk", "4-12wk", "12-26wk", "26-52wk", ">52wk"])
    print("\n=== CUT 6: max(layoff weeks) between the two players (either just had a break) ===")
    print(summarize(j, "layoff_bucket").round(4).to_string(index=False))

    j["either_gt26"] = j["max_layoff_weeks"] > 26
    print("\n=== CUT 6b: either player off >26 weeks (LAYOFF_WEEKS threshold in inflate()) vs not ===")
    print(summarize(j, "either_gt26").round(4).to_string(index=False))

    # ============================================================ CUT 7: experience (min of the two players' prior match counts)
    j["min_experience"] = j[["n_prior_w", "n_prior_l"]].min(axis=1)
    j["exp_bucket"] = pd.cut(j["min_experience"], [-1, 10, 25, 50, 100, 250, 1e6],
                              labels=["<=10", "11-25", "26-50", "51-100", "101-250", ">250"])
    print("\n=== CUT 7: experience of the LESS-experienced player in the match (prior tour matches) ===")
    print(summarize(j, "exp_bucket").round(4).to_string(index=False))

    # ============================================================ CUT 8: directional disagreement, extended
    j["hier_fav_uid"] = np.where(j.p_hier >= .5, j.winner_uid, j.loser_uid)
    j["elo_fav_uid"] = np.where(j.p_elo >= .5, j.winner_uid, j.loser_uid)
    j["disagree"] = j["hier_fav_uid"] != j["elo_fav_uid"]
    j["hier_conf"] = np.where(j.p_hier >= .5, j.p_hier, 1 - j.p_hier)
    j["hier_correct"] = (j.p_hier >= .5).astype(int)  # p_hier is P(actual winner), so >=.5 means hier's implied pick won
    dis = j[j["disagree"]].copy()
    dis["conf_bucket"] = pd.cut(dis["hier_conf"], [.5, .6, .7, .8, .9, 1.0])
    print(f"\n=== CUT 8: disagreement matches (n={len(dis)}, {100*len(dis)/len(j):.1f}% of all) ===")
    print("  hier accuracy by hier's own confidence bucket (extends the known 51-60pct-weak-disagreement finding):")
    acc_by_conf = dis.groupby("conf_bucket", observed=True).agg(
        n=("hier_correct", "size"), hier_acc=("hier_correct", "mean"))
    print(acc_by_conf.round(4).to_string())

    weak = dis[(dis.hier_conf >= .51) & (dis.hier_conf <= .60)].copy()
    print(f"\n  weak-disagreement bucket (51-60pct conf), n={len(weak)}, overall acc={weak.hier_correct.mean():.4f}")
    print("  -- breakdown of the weak-disagreement-and-wrong bucket by surface / tour / layoff / experience --")
    for cut_col in ["surface", "tour", "layoff_bucket", "exp_bucket", "best_of"]:
        t = weak.groupby(cut_col, observed=True).agg(n=("hier_correct", "size"), acc=("hier_correct", "mean"))
        t = t[t.n >= 20]
        if len(t):
            print(f"\n  by {cut_col}:")
            print(t.round(4).to_string())

    # ============================================================ CALIBRATION: elo vs hier vs market, overall
    calibration(j, "p_elo", "Elo (overall)")
    calibration(j, "p_hier", "Hier MC (overall)")
    calibration(j, "p_mkt", "Market (overall)")

    # calibration by surface for hier specifically (only surfaces with enough n)
    for surf in j["surface"].unique():
        sub = j[j.surface == surf]
        if len(sub) >= 300:
            calibration(sub, "p_hier", f"Hier MC, surface={surf} (n={len(sub)})")

    # calibration by layoff bucket for hier
    for lb in j["layoff_bucket"].cat.categories:
        sub = j[j.layoff_bucket == lb]
        if len(sub) >= 300:
            calibration(sub, "p_hier", f"Hier MC, layoff={lb} (n={len(sub)})")

    j.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_joined_enriched.parquet")
    print("\nwrote tune_scratch/diag_joined_enriched.parquet")
