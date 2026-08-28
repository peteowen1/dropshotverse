"""Ticket 17: the ONE-TIME reveal. Score the tuned hierarchical model against
Elo and the market on the identical 2024-25 odds-joined rows ticket 10 used -
this window was never read during hyperparameter selection.
"""
import numpy as np
import pandas as pd

from market_compare import key_from_full, key_from_short, devig_power, load_odds

ELO_PARQUET = r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet"
HIER_TUNED = r"C:\dev\dropshotverse\dropshotdata\hier_tuned_eval_predictions.parquet"
HIER_ORIG = r"C:\dev\dropshotverse\dropshotdata\hier_eval_predictions.parquet"


def log(m): print(m, flush=True)


if __name__ == "__main__":
    elo = pd.read_parquet(ELO_PARQUET)
    elo = elo[elo.level_file == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name", "p_winner"]
    ].rename(columns={"p_winner": "p_elo"})

    tuned = pd.read_parquet(HIER_TUNED)[
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc"]
    ].rename(columns={"p_winner_hier_mc": "p_tuned"})
    orig = pd.read_parquet(HIER_ORIG)[
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc"]
    ].rename(columns={"p_winner_hier_mc": "p_orig"})

    both = elo.merge(tuned, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    both = both.merge(orig, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    log(f"rows with all three predictions: {len(both):,}")

    both["k_w"] = both["winner_name"].map(key_from_full)
    both["k_l"] = both["loser_name"].map(key_from_full)
    both["pair"] = both.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=(2024, 2025))
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
    m = m[(m["k_w_o"] == m["k_w"]) & m[["PSW", "PSL"]].notna().all(axis=1)]
    log(f"joined to Pinnacle odds: {len(m):,}")

    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m = m.dropna(subset=["p_mkt"])

    rows = []
    for label, col in (("Elo", "p_elo"), ("Hierarchical (untuned, ticket 10)", "p_orig"),
                       ("Hierarchical (TUNED, ticket 17)", "p_tuned"), ("Market", "p_mkt")):
        for tour in ("all", "atp", "wta"):
            sub = m if tour == "all" else m[m.tour == tour]
            p = np.clip(sub[col].values, 1e-9, 1 - 1e-9)
            rows.append({"model": label, "tour": tour, "n": len(sub),
                        "acc": round(float((p > .5).mean()), 4),
                        "logloss": round(float(-np.log(p).mean()), 4),
                        "brier": round(float(((1 - p) ** 2).mean()), 4)})
    res = pd.DataFrame(rows)
    log("\n=== FINAL REVEAL: identical rows, never touched during tuning ===")
    log(res.to_string(index=False))

    piv = res[res.tour == "all"].set_index("model")["logloss"]
    log(f"\nTuned vs untuned : {piv['Hierarchical (TUNED, ticket 17)']-piv['Hierarchical (untuned, ticket 10)']:+.4f} log-loss")
    log(f"Tuned vs Elo     : {piv['Hierarchical (TUNED, ticket 17)']-piv['Elo']:+.4f} log-loss "
        f"({'BEATS Elo' if piv['Hierarchical (TUNED, ticket 17)']<piv['Elo'] else 'still worse than Elo'})")
    log(f"Tuned vs Market  : {piv['Hierarchical (TUNED, ticket 17)']-piv['Market']:+.4f} log-loss")

    gap_before = piv['Hierarchical (untuned, ticket 10)'] - piv['Market']
    gap_after = piv['Hierarchical (TUNED, ticket 17)'] - piv['Market']
    log(f"\nGap to market: {gap_before:.4f} (untuned) -> {gap_after:.4f} (tuned), "
        f"closed {100*(1-gap_after/gap_before):.1f}% of it")

    log("\n=== CALIBRATION BY CUT: Tuned ===")
    sym = m.copy()
    sym["p_fav"] = np.where(sym.p_tuned >= .5, sym.p_tuned, 1 - sym.p_tuned)
    sym["fav_won"] = (sym.p_tuned >= .5).astype(int)
    sym["b"] = pd.cut(sym.p_fav, [.5, .6, .7, .8, .9, 1.0])
    out = sym.groupby("b", observed=True).agg(n=("p_fav", "size"), predicted=("p_fav", "mean"),
                                              realised=("fav_won", "mean")).round(3)
    out["error"] = (out["realised"] - out["predicted"]).round(3)
    log(out.to_string())
