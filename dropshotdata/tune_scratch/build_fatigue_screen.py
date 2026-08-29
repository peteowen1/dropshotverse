"""Candidate feature: in-tournament fatigue - how many matches has each
player already played in THIS event before the current match? Genuinely new
information: weeks_since_last_diff (already in the model) captures inter-
tournament layoff, not intra-tournament congestion (a player who won a
5-hour 5-setter two days ago vs one who had a walkover). Read-only wrt the
existing pipeline; screened on the tune window (2022-23) only via 5-fold
xgb.cv, per stats-discipline #9 - does not touch the 2024-25 holdout.

Round-order causal bug (tickets 9/25's own repeated finding) applies here
too: within one tourney_id, qualifying and round-robin rounds must be
ordered correctly before a chronological match count means anything -
reuses elo_baseline.py's own ROUND_RANK table verbatim.
"""
import numpy as np
import pandas as pd

ROUND_RANK = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4, "ER": 5,
    "R128": 6, "R64": 7, "R32": 8, "R16": 9,
    "QF": 10, "RR": 11, "SF": 12, "F": 13, "BR": 13,
}


def load_raw(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"C:\dev\dropshotverse\dropshotdata\raw\github\{tour}\{tour}_matches_{y}.csv",
                         low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    rnd = df["round"].astype(str).str.strip().replace({"f": "F"})
    df["round_rank"] = rnd.map(ROUND_RANK)
    unk = df["round_rank"].isna()
    if unk.any():
        print(f"  [{tour}] {int(unk.sum())} rows with unmapped round -> sorted last")
        df["round_rank"] = df["round_rank"].fillna(99)
    return df


def compute_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """Prior-matches-this-event count, causally ordered within tourney_id."""
    df = df.sort_values(["tourney_id", "round_rank"]).reset_index(drop=True)
    counts: dict[tuple, int] = {}
    n = len(df)
    fw = np.zeros(n, dtype=np.int32)
    fl = np.zeros(n, dtype=np.int32)
    tids = df["tourney_id"].values
    wuids = df["winner_uid"].values
    luids = df["loser_uid"].values
    for i in range(n):
        tid, w, l = tids[i], wuids[i], luids[i]
        fw[i] = counts.get((tid, w), 0)
        fl[i] = counts.get((tid, l), 0)
        counts[(tid, w)] = fw[i] + 1
        counts[(tid, l)] = fl[i] + 1
    df["prior_matches_event_w"] = fw
    df["prior_matches_event_l"] = fl
    return df


raw = pd.concat([load_raw("atp", [2022, 2023]), load_raw("wta", [2022, 2023])], ignore_index=True)
raw = compute_fatigue(raw)
raw = raw.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
print("raw rows with fatigue computed:", len(raw))
print(raw["prior_matches_event_w"].value_counts().sort_index().head(10))

tune = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_tune.parquet")
j = tune.merge(
    raw[["date", "tour", "winner_uid", "loser_uid", "prior_matches_event_w", "prior_matches_event_l"]],
    on=["date", "tour", "winner_uid", "loser_uid"], how="left")
print("merged na rate:", j["prior_matches_event_w"].isna().mean())

y = j["y"].values.astype(int)
w = j["prior_matches_event_w"].values.astype(float)
l = j["prior_matches_event_l"].values.astype(float)
j["fatigue_diff"] = np.where(y == 1, w - l, l - w)
print(j["fatigue_diff"].describe())
j.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\fatigue_screen_tune.parquet")
print("saved")
