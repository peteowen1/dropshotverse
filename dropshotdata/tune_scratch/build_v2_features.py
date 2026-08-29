"""Ticket 28: add age_diff and fatigue_diff (both screened tune-only via
5-fold xgb.cv, stats-discipline #9) to the existing xgb_features_{tune,
holdout} parquets, for both windows. Read-only wrt build_xgb_features.py -
writes new xgb_features_v2_{tune,holdout}.parquet rather than overwriting.
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
    df["round_rank"] = df["round_rank"].fillna(99)
    return df


def compute_fatigue(df: pd.DataFrame) -> pd.DataFrame:
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


def build_window(feat_path, years):
    raw = pd.concat([load_raw("atp", years), load_raw("wta", years)], ignore_index=True)
    raw = compute_fatigue(raw)
    raw = raw.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
    keep = ["date", "tour", "winner_uid", "loser_uid", "winner_age", "loser_age",
            "prior_matches_event_w", "prior_matches_event_l"]
    raw_small = raw[keep]

    feat = pd.read_parquet(feat_path)
    n0 = len(feat)
    j = feat.merge(raw_small, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")

    y = j["y"].values.astype(int)
    aw = j["winner_age"].values.astype(float)
    al = j["loser_age"].values.astype(float)
    j["age_diff"] = np.where(y == 1, aw - al, al - aw)
    fw = j["prior_matches_event_w"].values.astype(float)
    fl = j["prior_matches_event_l"].values.astype(float)
    j["fatigue_diff"] = np.where(y == 1, fw - fl, fl - fw)

    miss = j["age_diff"].isna() | j["fatigue_diff"].isna()
    print(f"  {feat_path}: {n0} -> {len(j)} rows after merge, {int(miss.sum())} missing "
          f"age/fatigue ({miss.mean():.4%}), dropped")
    j = j[~miss].drop(columns=["winner_age", "loser_age", "prior_matches_event_w", "prior_matches_event_l"])
    return j.reset_index(drop=True)


tune = build_window(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_tune.parquet", [2022, 2023])
holdout = build_window(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_holdout.parquet", [2024, 2025])

tune.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_tune.parquet")
holdout.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_holdout.parquet")
print(f"tune: {len(tune):,} rows, holdout: {len(holdout):,} rows")
