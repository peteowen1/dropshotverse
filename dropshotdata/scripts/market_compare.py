"""Ticket 10: score the Elo baseline against the de-vigged closing line.

This is the question the whole map exists to answer. The Elo numbers alone mean
little; what matters is the gap to the market, because that is the only honest
benchmark tennis offers.

Joining is the hard part and has no published crosswalk (ticket 04). The archive
says "Jannik Sinner"; tennis-data.co.uk says "Sinner J.". We normalise both to a
surname+initial key and join on (date window, tour, both players).

stats-discipline §7: match rate is reported before any metric, because a join
that silently drops half the data changes the answer without changing how the
number reads.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ODDS = Path(r"C:\dev\dropshotverse\dropshotdata\raw\odds")
PRED = Path(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
YEARS = (2024, 2025)


def log(m: str) -> None:
    print(m, flush=True)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def key_from_full(name: str) -> str:
    """'Jannik Sinner' -> 'sinner j'   |  'Jan-Lennard Struff' -> 'struff j'"""
    n = strip_accents(str(name)).lower().strip()
    n = re.sub(r"[^a-z\s\-']", " ", n)
    parts = [p for p in n.split() if p]
    if len(parts) < 2:
        return n
    return f"{' '.join(parts[1:])} {parts[0][0]}".strip()


def key_from_short(name: str) -> str:
    """'Sinner J.' -> 'sinner j'  |  'De Minaur A.' -> 'de minaur a'"""
    n = strip_accents(str(name)).lower().strip()
    n = re.sub(r"[^a-z\s\-'.]", " ", n)
    n = n.replace(".", " ")
    parts = [p for p in n.split() if p]
    if len(parts) < 2:
        return n
    return f"{' '.join(parts[:-1])} {parts[-1][0]}".strip()


def load_odds(years=YEARS) -> pd.DataFrame:
    frames = []
    for tour in ("atp", "wta"):
        for y in years:
            f = ODDS / tour / f"{y}.xlsx"
            if not f.exists():
                continue
            df = pd.read_excel(f)
            df.columns = [str(c).strip() for c in df.columns]
            df["tour"] = tour
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def devig_power(o_w: np.ndarray, o_l: np.ndarray) -> np.ndarray:
    """Power method (Clarke, Kovalchik & Ingram 2017 - the tennis-specific study).

    Find k such that (1/o_w)^k + (1/o_l)^k = 1, return the winner's fair prob.
    Cannot produce invalid probabilities, unlike proportional scaling.
    """
    pw = np.empty(len(o_w))
    raw_w, raw_l = 1.0 / o_w, 1.0 / o_l
    for i in range(len(o_w)):
        a, b = raw_w[i], raw_l[i]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            pw[i] = np.nan
            continue
        lo, hi = 0.5, 2.0
        for _ in range(60):                       # bisection on k
            k = (lo + hi) / 2
            if a**k + b**k > 1:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        pw[i] = a**k
    return pw


if __name__ == "__main__":
    pred = pd.read_parquet(PRED)
    pred = pred[pred["level_file"] == "tour"].copy()     # odds exist only at tour level
    pred["k_w"] = pred["winner_name"].map(key_from_full)
    pred["k_l"] = pred["loser_name"].map(key_from_full)
    pred["date"] = pd.to_datetime(pred["date"])
    log(f"model predictions (tour level, {YEARS}): {len(pred):,}")

    odds = load_odds()
    log(f"odds rows loaded: {len(odds):,}")
    log(f"odds columns present: {[c for c in odds.columns if c in ('PSW','PSL','B365W','B365L','MaxW','MaxL','AvgW','AvgL')]}")

    odds = odds.dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])

    # tourney_date is the tournament START; a match can be up to ~2 weeks later.
    # Join on the unordered player pair + tour, then keep the nearest date within
    # a 21-day window. Unordered so a model/odds disagreement on who won cannot
    # silently drop the row - we want those, not fewer of them.
    pred["pair"] = pred.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    m = pred.merge(odds[["pair", "tour", "date", "Winner", "Loser", "k_w", "k_l",
                         "PSW", "PSL", "B365W", "B365L", "AvgW", "AvgL"]],
                   on=["pair", "tour"], how="left", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
    m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")

    log("\n=== JOIN COVERAGE (stats-discipline §7) ===")
    log(f"  model rows           : {len(pred):,}")
    log(f"  joined to odds       : {len(m):,}  ({100*len(m)/len(pred):.1f}%)")
    for tour in ("atp", "wta"):
        a, b = (pred.tour == tour).sum(), (m.tour == tour).sum()
        log(f"    {tour}: {b:,}/{a:,} = {100*b/max(a,1):.1f}%")

    # Does the odds file agree with the archive on who WON? Disagreement means a
    # bad join, not a bad model - and silently keeping them would corrupt scoring.
    m["odds_agrees"] = m["k_w_o"] == m["k_w"]
    log(f"  winner agrees        : {m['odds_agrees'].sum():,} "
        f"({100*m['odds_agrees'].mean():.1f}%)")
    m = m[m["odds_agrees"]]

    for book, (cw, cl) in {"Pinnacle": ("PSW", "PSL"), "Bet365": ("B365W", "B365L"),
                           "Average": ("AvgW", "AvgL")}.items():
        have = m[[cw, cl]].notna().all(axis=1).sum()
        log(f"  {book:<9} odds present: {have:,} ({100*have/max(len(m),1):.1f}%)")

    log("\n=== HEAD TO HEAD, de-vigged with the power method ===")
    rows = []
    for book, (cw, cl) in {"Pinnacle": ("PSW", "PSL"), "Bet365": ("B365W", "B365L"),
                           "Average": ("AvgW", "AvgL")}.items():
        sub = m[m[[cw, cl]].notna().all(axis=1)].copy()
        if len(sub) < 200:
            continue
        sub["p_mkt"] = devig_power(sub[cw].values.astype(float),
                                   sub[cl].values.astype(float))
        sub = sub.dropna(subset=["p_mkt"])
        margin = (1 / sub[cw].astype(float) + 1 / sub[cl].astype(float) - 1).mean()

        for tour in ("all", "atp", "wta"):
            s = sub if tour == "all" else sub[sub.tour == tour]
            if len(s) < 150:
                continue
            pe = np.clip(s["p_winner"].values, 1e-9, 1 - 1e-9)
            pm = np.clip(s["p_mkt"].values, 1e-9, 1 - 1e-9)
            rows.append({
                "book": book, "tour": tour, "n": len(s),
                "margin": round(float(margin), 4),
                "elo_acc": round(float((pe > .5).mean()), 4),
                "mkt_acc": round(float((pm > .5).mean()), 4),
                "elo_ll": round(float(-np.log(pe).mean()), 4),
                "mkt_ll": round(float(-np.log(pm).mean()), 4),
                "elo_brier": round(float(((1 - pe) ** 2).mean()), 4),
                "mkt_brier": round(float(((1 - pm) ** 2).mean()), 4),
            })
    res = pd.DataFrame(rows)
    res["ll_gap"] = (res["elo_ll"] - res["mkt_ll"]).round(4)
    log(res.to_string(index=False))

    # Where they disagree, who is right? The closing-line-value question.
    sub = m[m[["PSW", "PSL"]].notna().all(axis=1)].copy()
    sub["p_mkt"] = devig_power(sub["PSW"].values.astype(float),
                               sub["PSL"].values.astype(float))
    sub = sub.dropna(subset=["p_mkt"])
    dis = sub[((sub.p_winner > .5) != (sub.p_mkt > .5))]
    log(f"\n=== DISAGREEMENTS (Pinnacle) ===")
    log(f"  n disagreements: {len(dis):,} of {len(sub):,} ({100*len(dis)/max(len(sub),1):.1f}%)")
    if len(dis):
        log(f"  Elo right in {100*(dis.p_winner > .5).mean():.1f}% of them "
            f"(50% = coin flip, >50% = Elo adds something the market misses)")

    log("\n=== CALIBRATION BY CUT (aggregate can hide bias) ===")
    sub["bucket"] = pd.cut(sub["p_winner"], [0, .3, .4, .5, .6, .7, .8, 1.0])
    cal = sub.groupby("bucket", observed=True).agg(
        n=("p_winner", "size"),
        elo_says=("p_winner", "mean"),
        actually_won=("p_winner", lambda s: float((s > -1).mean())),
    )
    cal["actually_won"] = sub.groupby("bucket", observed=True)["p_winner"].apply(
        lambda s: 1.0)          # p_winner is P(actual winner) so realised rate is 1
    log("  (p_winner is P(the player who actually won) - so calibration is read as:")
    log("   in bucket [0.6,0.7], the model gave the eventual winner ~65%; if calibrated,")
    log("   ~65% of matches where it gave someone 65% should be won by that player.)")
    sym = sub.copy()
    # build a symmetric view: each match contributes the favourite's probability
    sym["p_fav"] = np.where(sym.p_winner >= .5, sym.p_winner, 1 - sym.p_winner)
    sym["fav_won"] = (sym.p_winner >= .5).astype(int)
    sym["b"] = pd.cut(sym.p_fav, [.5, .6, .7, .8, .9, 1.0])
    out = sym.groupby("b", observed=True).agg(n=("p_fav", "size"),
                                              predicted=("p_fav", "mean"),
                                              realised=("fav_won", "mean")).round(3)
    out["error"] = (out["realised"] - out["predicted"]).round(3)
    log(out.to_string())
