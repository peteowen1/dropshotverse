"""Ticket 10 stage 2: score the hierarchical model against Elo and the market,
on the IDENTICAL joined-odds rows used for Elo (stats-discipline 5.3) - never
compare numbers computed in separate executions on different row sets.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from market_compare import key_from_full, key_from_short, devig_power, load_odds

ELO_PARQUET = r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet"
HIER_PARQUET = r"C:\dev\dropshotverse\dropshotdata\hier_eval_predictions.parquet"
YEARS = (2024, 2025)


def log(m): print(m, flush=True)


if __name__ == "__main__":
    elo = pd.read_parquet(ELO_PARQUET)
    elo = elo[elo.level_file == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name", "p_winner"]
    ].rename(columns={"p_winner": "p_elo"})

    hier = pd.read_parquet(HIER_PARQUET)[
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_hier", "p_winner_hier_mc"]
    ]

    # join on identity, not a fresh odds-join, so both models are scored on
    # EXACTLY the same rows (stats-discipline 5.3)
    both = elo.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    log(f"Elo eval rows (tour level): {len(elo):,}")
    log(f"Hier eval rows            : {len(hier):,}")
    log(f"Rows with BOTH predictions: {len(both):,} "
        f"({100*len(both)/len(elo):.1f}% of Elo's eval set)")

    both["k_w"] = both["winner_name"].map(key_from_full)
    both["k_l"] = both["loser_name"].map(key_from_full)
    both["pair"] = both.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds()
    odds = odds.dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    m = both.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
                   on=["pair", "tour"], how="left", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
    m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    m["odds_agrees"] = m["k_w_o"] == m["k_w"]
    m = m[m["odds_agrees"] & m[["PSW", "PSL"]].notna().all(axis=1)]

    log(f"\n=== JOIN COVERAGE (identical for both models by construction) ===")
    log(f"  joined to Pinnacle odds: {len(m):,} ({100*len(m)/len(both):.1f}% of common eval rows)")
    for tour in ("atp", "wta"):
        log(f"    {tour}: {int((m.tour==tour).sum()):,}")

    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m = m.dropna(subset=["p_mkt"])

    rows = []
    for label, col in (("Elo", "p_elo"), ("Hierarchical (plug-in)", "p_winner_hier"),
                       ("Hierarchical (posterior-avg)", "p_winner_hier_mc"),
                       ("Market", "p_mkt")):
        for tour in ("all", "atp", "wta"):
            sub = m if tour == "all" else m[m.tour == tour]
            p = np.clip(sub[col].values, 1e-9, 1 - 1e-9)
            rows.append({"model": label, "tour": tour, "n": len(sub),
                        "acc": round(float((p > .5).mean()), 4),
                        "logloss": round(float(-np.log(p).mean()), 4),
                        "brier": round(float(((1 - p) ** 2).mean()), 4)})
    res = pd.DataFrame(rows)
    log("\n=== HEAD TO HEAD: Elo vs Hierarchical vs Market, IDENTICAL rows ===")
    log(res.to_string(index=False))

    piv = res[res.tour == "all"].set_index("model")["logloss"]
    for label in ("Hierarchical (plug-in)", "Hierarchical (posterior-avg)"):
        log(f"\n{label} vs Elo   : {piv[label]-piv['Elo']:+.4f} log-loss "
            f"({'BETTER' if piv[label]<piv['Elo'] else 'WORSE'})")
        log(f"{label} vs Market: {piv[label]-piv['Market']:+.4f} log-loss "
            f"({'closed' if abs(piv[label]-piv['Market'])<abs(piv['Elo']-piv['Market']) else 'did not close'} the gap)")

    log("\n=== CALIBRATION BY CUT: Hierarchical (posterior-avg) ===")
    sym = m.copy()
    sym["p_fav"] = np.where(sym.p_winner_hier_mc >= .5, sym.p_winner_hier_mc, 1 - sym.p_winner_hier_mc)
    sym["fav_won"] = (sym.p_winner_hier_mc >= .5).astype(int)
    sym["b"] = pd.cut(sym.p_fav, [.5, .6, .7, .8, .9, 1.0])
    out = sym.groupby("b", observed=True).agg(n=("p_fav", "size"), predicted=("p_fav", "mean"),
                                              realised=("fav_won", "mean")).round(3)
    out["error"] = (out["realised"] - out["predicted"]).round(3)
    log(out.to_string())

    log("\n=== CALIBRATION BY CUT: Elo (for direct comparison) ===")
    sym2 = m.copy()
    sym2["p_fav"] = np.where(sym2.p_elo >= .5, sym2.p_elo, 1 - sym2.p_elo)
    sym2["fav_won"] = (sym2.p_elo >= .5).astype(int)
    sym2["b"] = pd.cut(sym2.p_fav, [.5, .6, .7, .8, .9, 1.0])
    out2 = sym2.groupby("b", observed=True).agg(n=("p_fav", "size"), predicted=("p_fav", "mean"),
                                                realised=("fav_won", "mean")).round(3)
    out2["error"] = (out2["realised"] - out2["predicted"]).round(3)
    log(out2.to_string())
