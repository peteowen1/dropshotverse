"""Task 2 follow-up: elo_baseline.py's internal "ALL" metric for the decay
grid is dominated by lower-tier (qual/challenger/ITF) matches (758k of 848k
eval rows) - a different population from the one the hierarchical-model
tuning effort actually compares against (RESUME-beat-elo.md's 8,594-row
tour-level, market-odds-joined holdout). elo_baseline.py's own comment
already flags this: "Tour comparison is only meaningful WITHIN a level."

This script re-scores the SAME decay grid (blend held at 0.0, the blend
grid's winner) on the exact tour-level, market-joined population instead, so
the decay decision is made on the population that matters, not the blended
one. One axis only (decay); blend fixed throughout (stats-discipline §5).
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")   # archived 2026-09-06: elo_baseline / train_best_model stayed in scripts/
from elo_baseline import load_matches, prepare, run_elo, score, EVAL_YEARS
from market_compare import key_from_full, key_from_short, devig_power, load_odds

BLEND = 0.0  # winner of elo_baseline.py's blend grid, held fixed
DECAY_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def market_join(elo_preds: pd.DataFrame) -> pd.DataFrame:
    elo = elo_preds[elo_preds.level_file == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name", "p_winner"]
    ].rename(columns={"p_winner": "p_elo"})
    elo["k_w"] = elo["winner_name"].map(key_from_full)
    elo["k_l"] = elo["loser_name"].map(key_from_full)
    elo["pair"] = elo.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=EVAL_YEARS).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    j = elo.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
                  on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    j["p_mkt"] = devig_power(j["PSW"].values.astype(float), j["PSL"].values.astype(float))
    j = j.dropna(subset=["p_mkt"])
    return j


if __name__ == "__main__":
    print("loading + preparing archive...", flush=True)
    raw = load_matches()
    df, dropped = prepare(raw)
    ev = df["date"].dt.year.isin(EVAL_YEARS)

    rows = []
    for decay in DECAY_GRID:
        fitted, _ = run_elo(df, BLEND, decay)
        preds = fitted[ev].copy()
        j = market_join(preds)
        pr = np.clip(j["p_elo"].values, 1e-9, 1 - 1e-9)
        row = {"decay": decay, "n": len(j), "acc": float((pr > 0.5).mean()),
               "logloss": float(-np.log(pr).mean()), "brier": float(((1 - pr) ** 2).mean())}
        for tour in ("atp", "wta"):
            sub = j[j["tour"] == tour]
            pr_t = np.clip(sub["p_elo"].values, 1e-9, 1 - 1e-9)
            row[f"{tour}_n"] = len(sub)
            row[f"{tour}_logloss"] = float(-np.log(pr_t).mean())
        rows.append(row)
        print(f"  decay={decay}: n={row['n']} logloss={row['logloss']:.4f} "
              f"(atp={row['atp_logloss']:.4f} wta={row['wta_logloss']:.4f})", flush=True)

    out = pd.DataFrame(rows)
    print("\n=== TOUR-LEVEL, MARKET-JOINED DECAY GRID (the number the hierarchical model is actually chasing) ===")
    print(out.to_string(index=False))
    out.to_csv(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\layoff_decay_market_grid.csv", index=False)
