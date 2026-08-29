"""Ticket 28 round 9 (Pete's own suggestions): handedness mismatch and
serve-vs-return archetype, tested separately from round 3's rejected serve-
dominance feature (which only used SERVE stats and duplicated Elo).

1. hand_sideA_left / hand_sideB_left / lefty_vs_righty_mismatch - tested
   alone this time (round 3's height+hand bundle conflated a real signal
   with ~15-36% WTA height-missingness noise).
2. return_form: rolling EWM of RETURN points won % - the complement of
   round 3's serve_form, derived from the SAME raw columns but a genuinely
   different quantity (a player's return performance in a match = 1 minus
   their OPPONENT's serve performance in that match). Isner-type servers and
   Djokovic-type returners should separate cleanly on this axis in ways
   aggregate Elo (win/loss only) doesn't distinguish - Pete's "server vs
   returner" archetype point. Tested both as its own diff AND alongside
   serve_form_diff so XGBoost's own splits can find the cross-interaction
   (my serve strength vs their return weakness) rather than me hand-coding it.

Read-only wrt existing pipeline files. Tune-window (2022-23) screen only,
via 5-fold xgb.cv on the tuned hyperparameters - holdout untouched.
"""
import pickle

import numpy as np
import pandas as pd

with open(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl", "rb") as f:
    arc, _ = pickle.load(f)
arc = arc[arc["level_file"] == "tour"].reset_index(drop=True).copy()
arc["orig_idx"] = arc.index

# ---- return_form: return pts won % = 1 - opponent's serve pts won % in THIS match
arc["serve_pct_w"] = np.where(arc["has_points"] & (arc["w_svpt"] > 0),
                               (arc["w_1stWon"] + arc["w_2ndWon"]) / arc["w_svpt"], np.nan)
arc["serve_pct_l"] = np.where(arc["has_points"] & (arc["l_svpt"] > 0),
                               (arc["l_1stWon"] + arc["l_2ndWon"]) / arc["l_svpt"], np.nan)
arc["return_pct_w"] = 1.0 - arc["serve_pct_l"]   # winner's return performance = 1 - loser's serve pct
arc["return_pct_l"] = 1.0 - arc["serve_pct_w"]

long_w = arc[["orig_idx", "winner_uid", "serve_pct_w", "return_pct_w"]].rename(
    columns={"winner_uid": "uid", "serve_pct_w": "serve_pct", "return_pct_w": "return_pct"})
long_l = arc[["orig_idx", "loser_uid", "serve_pct_l", "return_pct_l"]].rename(
    columns={"loser_uid": "uid", "serve_pct_l": "serve_pct", "return_pct_l": "return_pct"})
long_all = pd.concat([long_w, long_l], ignore_index=True).sort_values(["uid", "orig_idx"], kind="stable")

for col in ("serve_pct", "return_pct"):
    long_all[f"{col}_form"] = long_all.groupby("uid")[col].transform(
        lambda s: s.shift(1).ewm(span=15, min_periods=3).mean())

fw = long_all.rename(columns={"uid": "winner_uid", "serve_pct_form": "serve_form_w",
                               "return_pct_form": "return_form_w"})[
    ["orig_idx", "winner_uid", "serve_form_w", "return_form_w"]]
fl = long_all.rename(columns={"uid": "loser_uid", "serve_pct_form": "serve_form_l",
                               "return_pct_form": "return_form_l"})[
    ["orig_idx", "loser_uid", "serve_form_l", "return_form_l"]]
arc = arc.merge(fw, on=["orig_idx", "winner_uid"], how="left").merge(fl, on=["orig_idx", "loser_uid"], how="left")

# anchor: Isner should be a strong server, weak returner; a classic grinder
# (Djokovic) should be roughly the opposite pattern - checked directly.
for uid, label in (("atp:104545", "Isner"), ("atp:104925", "Djokovic")):
    rows_w = arc[arc["winner_uid"] == uid][["serve_form_w", "return_form_w"]].rename(
        columns={"serve_form_w": "serve", "return_form_w": "return"})
    rows_l = arc[arc["loser_uid"] == uid][["serve_form_l", "return_form_l"]].rename(
        columns={"serve_form_l": "serve", "return_form_l": "return"})
    both = pd.concat([rows_w, rows_l]).dropna()
    if len(both):
        print(f"anchor {label}: mean serve_form={both['serve'].mean():.4f}  "
              f"mean return_form={both['return'].mean():.4f}  n={len(both)}")

def load_hand(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"C:\dev\dropshotverse\dropshotdata\raw\github\{tour}\{tour}_matches_{y}.csv",
                         low_memory=False, usecols=["tourney_date", "winner_id", "loser_id",
                                                     "winner_hand", "loser_hand"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["tour"] = tour
    d["date"] = pd.to_datetime(d["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        d[f"{side}_uid"] = d["tour"] + ":" + d[f"{side}_id"].astype("Int64").astype(str)
    return d[["date", "tour", "winner_uid", "loser_uid", "winner_hand", "loser_hand"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])


hand = pd.concat([load_hand("atp", [2022, 2023, 2024, 2025]), load_hand("wta", [2022, 2023, 2024, 2025])],
                  ignore_index=True)
arc = arc.merge(hand, on=["date", "tour", "winner_uid", "loser_uid"], how="left")

playstyle = arc[["date", "tour", "winner_uid", "loser_uid", "winner_hand", "loser_hand",
                 "serve_form_w", "serve_form_l", "return_form_w", "return_form_l"]]
playstyle.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\playstyle_full.parquet")
print(f"saved {len(playstyle):,} rows, hand coverage={hand.shape[0]:,} source rows")
