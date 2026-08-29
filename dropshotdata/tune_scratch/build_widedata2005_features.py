"""Ticket 28 round 12: a genuinely different axis - DATA VOLUME vs feature
richness. Everything tried in rounds 1-11 used the same 2-year (2022-23)
training window (hier_dmean/dvar's posterior state was only ever computed
for that window + the 2024-25 holdout - extending the hierarchical model
itself to more years is a much bigger undertaking, out of scope here).

But elo_raw_ratings_full.parquet, diag_layoff.parquet, and the raw archive
CSVs cover the FULL history back to 1967 - so an Elo-only feature set (no
hier state) CAN be trained on many more years. Question: does 9 years of
Elo+engineered-feature data (2015-2023, ~50K rows) beat 2 years of
Elo+hier+engineered data (2022-23, ~10K rows, this session's shipped v2)?
If yes, it argues data volume matters more than hier's extra signal for this
model class. If no, it's a genuine, informative confirmation that hier's
contribution is worth more than 4x the training rows.

Read-only wrt existing pipeline files - reuses elo_raw_ratings_full.parquet
and diag_layoff.parquet (both already full-archive) plus a fresh causal
fatigue computation and h2h computation over the wider window.
"""
import hashlib
import pickle

import numpy as np
import pandas as pd

ROUND_RANK = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4, "ER": 5,
    "R128": 6, "R64": 7, "R32": 8, "R16": 9,
    "QF": 10, "RR": 11, "SF": 12, "F": 13, "BR": 13,
}
DD = r"C:\dev\dropshotverse\dropshotdata"
TRAIN_YEARS = list(range(2005, 2024))   # 2005-2023, 19 years - round 12 continuation


def side_a_is_winner(date, w_uid, l_uid):
    key = f"{date.date()}|{'|'.join(sorted([w_uid, l_uid]))}"
    h = hashlib.sha256(key.encode()).digest()
    return (h[0] % 2) == 0


def load_raw_years(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"{DD}\raw\github\{tour}\{tour}_matches_{y}.csv", low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    rnd = df["round"].astype(str).str.strip().replace({"f": "F"})
    df["round_rank"] = rnd.map(ROUND_RANK).fillna(99)
    return df


print("loading raw archive for fatigue + age + hand, 2015-2023...")
raw = pd.concat([load_raw_years("atp", TRAIN_YEARS), load_raw_years("wta", TRAIN_YEARS)], ignore_index=True)

# fatigue: prior-matches-this-event, same causal method as round 1/build_v2_features.py
raw_sorted = raw.sort_values(["tourney_id", "round_rank"]).reset_index(drop=True)
counts = {}
n = len(raw_sorted)
fw = np.zeros(n, dtype=np.int32)
fl = np.zeros(n, dtype=np.int32)
for i, (tid, w, l) in enumerate(zip(raw_sorted["tourney_id"].values, raw_sorted["winner_uid"].values,
                                     raw_sorted["loser_uid"].values)):
    fw[i] = counts.get((tid, w), 0)
    fl[i] = counts.get((tid, l), 0)
    counts[(tid, w)] = fw[i] + 1
    counts[(tid, l)] = fl[i] + 1
raw_sorted["prior_matches_event_w"] = fw
raw_sorted["prior_matches_event_l"] = fl
raw_sorted = raw_sorted.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])

small = raw_sorted[["date", "tour", "winner_uid", "loser_uid", "winner_age", "loser_age",
                     "prior_matches_event_w", "prior_matches_event_l", "surface", "round",
                     "tourney_level", "best_of"]]

print("loading elo_raw_ratings_full.parquet + diag_layoff.parquet (both already full-archive)...")
elo_raw = pd.read_parquet(rf"{DD}\tune_scratch\elo_raw_ratings_full.parquet")
elo_raw = elo_raw[elo_raw["level_file"] == "tour"][
    ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name",
     "ro_winner", "ro_loser", "rs_winner", "rs_loser", "p_winner_recomputed"]]
layoff = pd.read_parquet(rf"{DD}\tune_scratch\diag_layoff.parquet")[
    ["date", "winner_uid", "loser_uid", "weeks_since_last_w", "weeks_since_last_l", "n_prior_w", "n_prior_l"]]

print("computing as-of h2h from the FULL archive (same method as build_xgb_features.py)...")
full_elo_raw_all = pd.read_parquet(rf"{DD}\tune_scratch\elo_raw_ratings_full.parquet")
wins = {}
h2h_w = np.zeros(len(full_elo_raw_all), dtype=np.int32)
h2h_l = np.zeros(len(full_elo_raw_all), dtype=np.int32)
for i, (w, l) in enumerate(zip(full_elo_raw_all["winner_uid"].values, full_elo_raw_all["loser_uid"].values)):
    h2h_w[i] = wins.get((w, l), 0)
    h2h_l[i] = wins.get((l, w), 0)
    wins[(w, l)] = wins.get((w, l), 0) + 1
h2h = full_elo_raw_all[["date", "tour", "winner_uid", "loser_uid"]].copy()
h2h["h2h_prior_wins_winner"] = h2h_w
h2h["h2h_prior_wins_loser"] = h2h_l

print("merging...")
m = elo_raw.merge(small, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
m = m.merge(layoff, on=["date", "winner_uid", "loser_uid"], how="left")
m = m.merge(h2h, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
m = m.drop_duplicates(subset=["date", "winner_uid", "loser_uid"], keep="first")

req = ["weeks_since_last_w", "weeks_since_last_l", "n_prior_w", "n_prior_l",
       "h2h_prior_wins_winner", "h2h_prior_wins_loser", "winner_age", "loser_age"]
miss = m[req].isna().any(axis=1)
print(f"{len(m):,} rows -> {int((~miss).sum()):,} after dropping {int(miss.sum()):,} missing-feature rows "
      f"({miss.mean():.4%})")
m = m[~miss].reset_index(drop=True)

a_is_winner = np.array([side_a_is_winner(d, w, l) for d, w, l in
                         zip(m["date"], m["winner_uid"], m["loser_uid"])])
y = a_is_winner.astype(int)


def pick(win_col, lose_col):
    wv, lv = m[win_col].values, m[lose_col].values
    return np.where(a_is_winner, wv, lv), np.where(a_is_winner, lv, wv)


a_ro, b_ro = pick("ro_winner", "ro_loser")
a_rs, b_rs = pick("rs_winner", "rs_loser")
a_weeks, b_weeks = pick("weeks_since_last_w", "weeks_since_last_l")
a_nprior, b_nprior = pick("n_prior_w", "n_prior_l")
a_h2h, b_h2h = pick("h2h_prior_wins_winner", "h2h_prior_wins_loser")
a_age, b_age = pick("winner_age", "loser_age")
a_fat, b_fat = pick("prior_matches_event_w", "prior_matches_event_l")

p_win_recomp = m["p_winner_recomputed"].values.astype(float)
p_sideA = np.where(a_is_winner, p_win_recomp, 1.0 - p_win_recomp)
p_sideA = np.clip(p_sideA, 1e-6, 1 - 1e-6)
elo_logit_pwinner = np.log(p_sideA / (1.0 - p_sideA))

feat = pd.DataFrame({
    "date": m["date"], "tour": m["tour"], "winner_uid": m["winner_uid"], "loser_uid": m["loser_uid"],
    "y": y,
    "elo_overall_diff": a_ro - b_ro, "elo_surface_diff": a_rs - b_rs,
    "elo_logit_pwinner": elo_logit_pwinner,
    "weeks_since_last_diff": a_weeks - b_weeks, "n_prior_diff": a_nprior - b_nprior,
    "h2h_diff": a_h2h - b_h2h, "age_diff": a_age - b_age, "fatigue_diff": a_fat - b_fat,
    "surface": m["surface"].astype(str), "round": m["round"].astype(str),
    "best_of": m["best_of"].astype("Int64"), "tourney_level": m["tourney_level"].astype(str),
})
feat.to_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_2005_2023.parquet")
print(f"\nsaved {len(feat):,} rows, years {feat['date'].dt.year.min()}-{feat['date'].dt.year.max()}")
print(feat["date"].dt.year.value_counts().sort_index().to_string())
