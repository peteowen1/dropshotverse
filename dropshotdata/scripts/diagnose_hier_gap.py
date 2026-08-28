"""Why does the tuned hierarchical model trail Elo on log-loss despite
identical accuracy? Full diagnostic: dataset description, biggest misses vs
both baselines, variance comparison, per-bucket log-loss, calibration/bias,
and cuts by tour/surface/round/margin.

stats-discipline #7: coverage logged before any number is trusted.
"""
import numpy as np
import pandas as pd

from market_compare import key_from_full, key_from_short, devig_power, load_odds

ELO_PARQUET = r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet"
HIER_TUNED = r"C:\dev\dropshotverse\dropshotdata\hier_tuned_eval_predictions.parquet"


def log(m): print(m, flush=True)


def ll(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.log(p)


if __name__ == "__main__":
    elo = pd.read_parquet(ELO_PARQUET)
    elo_full = elo.copy()   # keep unfiltered for round/surface columns if present
    elo = elo[elo.level_file == "tour"][
        ["date", "tour", "surface", "winner_uid", "loser_uid",
         "winner_name", "loser_name", "p_winner", "round", "best_of"]
    ].rename(columns={"p_winner": "p_elo"})

    hier = pd.read_parquet(HIER_TUNED)[
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc", "p1", "p2"]
    ].rename(columns={"p_winner_hier_mc": "p_hier"})

    m = elo.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    log(f"rows after Elo/Hier join: {len(m):,}")

    m["k_w"] = m["winner_name"].map(key_from_full)
    m["k_l"] = m["loser_name"].map(key_from_full)
    m["pair"] = m.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    j = m.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
               on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    j["p_mkt"] = devig_power(j["PSW"].values.astype(float), j["PSL"].values.astype(float))
    j = j.dropna(subset=["p_mkt"])

    log("\n=== DATASET (stats-discipline #7) ===")
    log(f"  n = {len(j):,} tour-level ATP+WTA matches, 2024-2025, joined to Pinnacle closing odds")
    log(f"  date range: {j.date.min().date()} -> {j.date.max().date()}")
    log(f"  tour split: {j.tour.value_counts().to_dict()}")
    log(f"  surface split: {j.surface.value_counts().to_dict()}")
    log(f"  round split: {j['round'].value_counts().to_dict()}")
    log(f"  best_of split: {j.best_of.value_counts().to_dict()}")

    for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
        j[f"ll_{name}"] = ll(j[col].values)

    log("\n=== HEADLINE (sanity check against ticket 17's numbers) ===")
    for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
        log(f"  {name}: acc={(j[col] > 0.5).mean():.4f}  logloss={j[f'll_{name}'].mean():.4f}")

    log("\n=== VARIANCE OF PREDICTIONS (spread of the favourite's stated probability) ===")
    for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
        p_fav = np.where(j[col] >= .5, j[col], 1 - j[col])
        log(f"  {name}: mean p_fav={p_fav.mean():.4f}  std={p_fav.std():.4f}  "
            f"min={p_fav.min():.3f}  max={p_fav.max():.3f}  "
            f"pct in (0.9,1.0]={100*(p_fav>0.9).mean():.1f}%  pct in [0.5,0.6)={100*(p_fav<0.6).mean():.1f}%")

    log("\n=== LOG-LOSS BY FAVOURITE-PROBABILITY BUCKET, ALL THREE MODELS ===")
    for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
        sub = j.copy()
        sub["p_fav"] = np.where(sub[col] >= .5, sub[col], 1 - sub[col])
        sub["fav_won"] = (sub[col] >= .5).astype(int)
        sub["ll"] = ll(sub[col].values)
        sub["b"] = pd.cut(sub.p_fav, [.5, .6, .7, .8, .9, 1.0])
        out = sub.groupby("b", observed=True).agg(
            n=("p_fav", "size"), predicted=("p_fav", "mean"),
            realised=("fav_won", "mean"), mean_logloss=("ll", "mean")).round(4)
        out["bias"] = (out["realised"] - out["predicted"]).round(4)
        log(f"\n  --- {name} ---")
        log("  " + out.to_string().replace("\n", "\n  "))

    log("\n=== BIGGEST MISSES: where Hierarchical's log-loss is far worse than BOTH Elo and Market ===")
    j["hier_minus_elo"] = j["ll_Hierarchical"] - j["ll_Elo"]
    j["hier_minus_mkt"] = j["ll_Hierarchical"] - j["ll_Market"]
    j["worst_score"] = j["hier_minus_elo"] + j["hier_minus_mkt"]
    worst = j.nlargest(20, "worst_score")
    cols = ["date", "tour", "surface", "round", "winner_name", "loser_name",
           "p_elo", "p_hier", "p_mkt", "ll_Elo", "ll_Hierarchical", "ll_Market"]
    log(worst[cols].round(3).to_string(index=False))

    log("\n=== ARE THE BIGGEST MISSES CONCENTRATED ANYWHERE? ===")
    log("worst 200 by hier_minus_mkt, breakdown:")
    w200 = j.nlargest(200, "hier_minus_mkt")
    log(f"  tour:    {w200.tour.value_counts(normalize=True).round(3).to_dict()}  "
        f"(full set: {j.tour.value_counts(normalize=True).round(3).to_dict()})")
    log(f"  surface: {w200.surface.value_counts(normalize=True).round(3).to_dict()}")
    log(f"  round:   {w200['round'].value_counts(normalize=True).round(3).to_dict()}")
    log(f"  best_of: {w200.best_of.value_counts(normalize=True).round(3).to_dict()}")
    log(f"  hier said favourite was: mean p_fav={np.where(w200.p_hier>=.5,w200.p_hier,1-w200.p_hier).mean():.3f} "
        f"vs market's mean p_fav on same matches={np.where(w200.p_mkt>=.5,w200.p_mkt,1-w200.p_mkt).mean():.3f}")
    log(f"  in these 200, hier favourite actually won: {(w200['p_hier']>=.5).mean()*100:.1f}% of the time "
        f"(so hier picked the WRONG side how often here: {100*(1-(w200['p_hier']>=.5).mean()):.1f}%)")
    # does hier disagree on WHO the favourite is, vs market, in the worst misses?
    disagree = ((w200.p_hier >= .5) != (w200.p_mkt >= .5)).mean()
    log(f"  hier and market pick a DIFFERENT favourite in {100*disagree:.1f}% of these 200 "
        f"(vs {100*((j.p_hier>=.5)!=(j.p_mkt>=.5)).mean():.1f}% overall)")

    log("\n=== CUTS: log-loss, mean predicted-vs-realised bias, by tour ===")
    for tour in ("atp", "wta"):
        sub = j[j.tour == tour]
        log(f"  {tour}: n={len(sub)}")
        for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
            p_fav = np.where(sub[col] >= .5, sub[col], 1 - sub[col])
            fav_won = (sub[col] >= .5)
            log(f"    {name}: logloss={ll(sub[col]).mean():.4f}  "
                f"mean_p_fav={p_fav.mean():.3f}  realised_fav_win_rate={fav_won.mean():.3f}  "
                f"bias={fav_won.mean()-p_fav.mean():+.4f}")

    log("\n=== CUTS: by round (early vs late) ===")
    j["round_group"] = j["round"].apply(lambda r: "early (R128/R64/R32/Q)" if str(r) in
                                        ("R128", "R64", "R32", "Q1", "Q2", "Q3") else "late (R16+)")
    for grp in j.round_group.unique():
        sub = j[j.round_group == grp]
        log(f"  {grp}: n={len(sub)}")
        for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
            log(f"    {name}: logloss={ll(sub[col]).mean():.4f}")

    log("\n=== CUTS: by market-implied match closeness ===")
    j["mkt_fav"] = np.where(j.p_mkt >= .5, j.p_mkt, 1 - j.p_mkt)
    j["closeness"] = pd.cut(j.mkt_fav, [.5, .65, .8, 1.0], labels=["close", "moderate", "lopsided"])
    for grp in ["close", "moderate", "lopsided"]:
        sub = j[j.closeness == grp]
        log(f"  {grp} (market fav {sub.mkt_fav.min():.2f}-{sub.mkt_fav.max():.2f}, n={len(sub)}):")
        for col, name in (("p_elo", "Elo"), ("p_hier", "Hierarchical"), ("p_mkt", "Market")):
            p_fav = np.where(sub[col] >= .5, sub[col], 1 - sub[col])
            log(f"    {name}: logloss={ll(sub[col]).mean():.4f}  mean_p_fav={p_fav.mean():.3f}")

    log("\n=== DOES HIER DISAGREE WITH THE MARKET'S FAVOURITE MORE THAN ELO DOES? ===")
    hier_disagree = ((j.p_hier >= .5) != (j.p_mkt >= .5))
    elo_disagree = ((j.p_elo >= .5) != (j.p_mkt >= .5))
    log(f"  Hier disagrees with market favourite: {100*hier_disagree.mean():.1f}% of matches")
    log(f"  Elo  disagrees with market favourite: {100*elo_disagree.mean():.1f}% of matches")
    log(f"  Hier right when it disagrees: {100*(j.loc[hier_disagree,'p_hier']>=.5).mean():.1f}%")
    log(f"  Elo  right when it disagrees: {100*(j.loc[elo_disagree,'p_elo']>=.5).mean():.1f}%")

    j.to_parquet(r"C:\dev\dropshotverse\dropshotdata\diagnose_joined.parquet")
    log("\nwrote diagnose_joined.parquet for further ad hoc slicing")
