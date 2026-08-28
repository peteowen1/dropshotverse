"""Diagnostic (read-only), round-3-continuation: build the enriched eval frame
used by every diag2_* script in this batch. Loads hier_v25_eval_predictions
(10,680 regime-A tour-level eval rows), merges layoff (diag_layoff.parquet,
full-archive weeks-since-last per match) and tourney_level/round
(prepared_matches_cache.pkl), and joins Elo (full 10,680) + market odds
(subset, via the same pattern as diag_base_repro.py). Writes
tune_scratch/diag2_enriched.parquet. No modification to any modeling script.
"""
import pickle
import sys

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from market_compare import key_from_full, key_from_short, devig_power, load_odds

HIER_PATH = r"C:\dev\dropshotverse\dropshotdata\hier_v25_eval_predictions.parquet"
LAYOFF = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_layoff.parquet"
CACHE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl"
ELO = r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet"
OUT = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"

ELITE_LEVELS = frozenset({"G", "M", "F", "PM", "O"})


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def acc(p):
    return float((np.asarray(p, dtype=float) > 0.5).mean())


def build():
    hier = pd.read_parquet(HIER_PATH)
    print(f"hier eval rows: {len(hier):,}")

    lay = pd.read_parquet(LAYOFF)
    hier = hier.merge(lay, on=["date", "winner_uid", "loser_uid"], how="left")
    print(f"layoff merge coverage: {hier['weeks_since_last_w'].notna().sum() + hier['weeks_since_last_l'].notna().sum():,} "
          f"non-null side-values / {2*len(hier):,} possible")

    # nosec: this session's own prepared-archive cache (hierarchical_model.py's
    # load_prepared()), not an externally-sourced pickle - same trust basis as
    # diag_round_layoff.py's identical load earlier this session.
    with open(CACHE, "rb") as f:
        arc, _ = pickle.load(f)
    arc_small = arc[["date", "winner_uid", "loser_uid", "round", "tourney_level"]].drop_duplicates(
        subset=["date", "winner_uid", "loser_uid"])
    hier = hier.merge(arc_small, on=["date", "winner_uid", "loser_uid"], how="left")
    print(f"round/level merge coverage: {hier['tourney_level'].notna().sum():,}/{len(hier):,}")

    hier["is_elite"] = hier["tourney_level"].isin(ELITE_LEVELS)
    hier["max_layoff_weeks"] = hier[["weeks_since_last_w", "weeks_since_last_l"]].max(axis=1)
    hier["min_layoff_weeks"] = hier[["weeks_since_last_w", "weeks_since_last_l"]].min(axis=1)
    # which side (if any) had the short layoff - needed to know which variance
    # component a scoped fix should touch
    hier["winner_had_short_layoff"] = hier["weeks_since_last_w"].between(1, 12)
    hier["loser_had_short_layoff"] = hier["weeks_since_last_l"].between(1, 12)
    hier["either_short_layoff"] = hier["winner_had_short_layoff"] | hier["loser_had_short_layoff"]

    elo = pd.read_parquet(ELO)
    elo = elo[elo.level_file == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]].rename(
        columns={"p_winner": "p_elo"})
    hier = hier.merge(elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    print(f"elo merge coverage: {hier['p_elo'].notna().sum():,}/{len(hier):,}")

    # market odds join, subset only (same pattern as diag_base_repro.py)
    hier["k_w"] = hier["winner_name"].map(key_from_full)
    hier["k_l"] = hier["loser_name"].map(key_from_full)
    hier["pair"] = hier.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    j = hier.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
                    on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j_odds = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j_odds = j_odds.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j_odds = j_odds[(j_odds["k_w_o"] == j_odds["k_w"]) & j_odds[["PSW", "PSL"]].notna().all(axis=1)]
    j_odds = j_odds.copy()
    j_odds["p_mkt"] = devig_power(j_odds["PSW"].values.astype(float), j_odds["PSL"].values.astype(float))
    j_odds = j_odds.dropna(subset=["p_mkt"])
    # keep only the first odds-match per original row (dedup on original index)
    j_odds = j_odds.drop_duplicates(subset=["date", "winner_uid", "loser_uid"], keep="first")

    hier = hier.merge(j_odds[["date", "winner_uid", "loser_uid", "p_mkt"]],
                       on=["date", "winner_uid", "loser_uid"], how="left")
    print(f"market odds coverage: {hier['p_mkt'].notna().sum():,}/{len(hier):,}")

    before = len(hier)
    hier = hier.drop_duplicates(subset=["date", "winner_uid", "loser_uid"], keep="first")
    if len(hier) != before:
        print(f"deduped {before - len(hier)} rows introduced by many-to-one merges "
              f"(round-robin rematches / elo dupes) -> {len(hier):,}")

    hier.to_parquet(OUT)
    print(f"wrote {OUT}  shape={hier.shape}")
    return hier


if __name__ == "__main__":
    df = build()
    print("\n=== quick sanity: baseline log-loss (n=60 col already in file) ===")
    sub = df.dropna(subset=["p_elo"])
    print(f"  n={len(sub):,}  hier(mc60)={ll(sub.p_winner_hier_mc):.4f}  "
          f"hier(mc600)={ll(sub.p_winner_hier_mc600):.4f}  elo={ll(sub.p_elo):.4f}")
