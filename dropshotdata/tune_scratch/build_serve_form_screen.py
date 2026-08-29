"""Ticket 28 round 3: rolling PRE-match serve-quality form. Ticket 05's own
literature review found point-level serve/return skill is the primary signal
published models use - this pipeline has only ever used win/loss outcomes
(Elo, hierarchical state) plus context. Genuinely untapped information.

serve_win_pct = (1stWon + 2ndWon) / svpt for the match just played - an
as-of EXPONENTIALLY-WEIGHTED trailing average per player, using ONLY strictly
PRIOR matches (shift-by-1 within each player's own chronological sequence),
so it is legitimate pre-match information, not a leak of this match's own
outcome. Screened on the tune window (2022-23) only via 5-fold xgb.cv,
stats-discipline #9 - holdout untouched.

Uses prepared_matches_cache.pkl - this session's own already-verified cache
(hierarchical_model.py's load_prepared(), the same trust basis
build_xgb_features.py already documented for this exact file - own-generated,
not externally sourced, so unpickling it carries no more risk than reading
back our own parquet output).
"""
import pickle

import numpy as np
import pandas as pd

with open(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl", "rb") as f:
    arc, _ = pickle.load(f)

arc = arc[arc["level_file"] == "tour"].reset_index(drop=True).copy()
arc["orig_idx"] = arc.index
arc["serve_pct_w"] = np.where(arc["has_points"] & (arc["w_svpt"] > 0),
                               (arc["w_1stWon"] + arc["w_2ndWon"]) / arc["w_svpt"], np.nan)
arc["serve_pct_l"] = np.where(arc["has_points"] & (arc["l_svpt"] > 0),
                               (arc["l_1stWon"] + arc["l_2ndWon"]) / arc["l_svpt"], np.nan)
print(f"has_points coverage (tour level): {arc['has_points'].mean():.4f}")

long_w = arc[["orig_idx", "winner_uid", "serve_pct_w"]].rename(
    columns={"winner_uid": "uid", "serve_pct_w": "serve_pct"})
long_l = arc[["orig_idx", "loser_uid", "serve_pct_l"]].rename(
    columns={"loser_uid": "uid", "serve_pct_l": "serve_pct"})
long_all = pd.concat([long_w, long_l], ignore_index=True)
# arc's own row order is already causally chronological (hierarchical_model.py's
# prepare() sort, round-9-fixed) - within-player order follows from that.
long_all = long_all.sort_values(["uid", "orig_idx"], kind="stable")

long_all["serve_form"] = (
    long_all.groupby("uid")["serve_pct"]
    .transform(lambda s: s.shift(1).ewm(span=15, min_periods=3).mean())
)

form_w = long_all.rename(columns={"uid": "winner_uid", "serve_form": "serve_form_w"})[
    ["orig_idx", "winner_uid", "serve_form_w"]]
form_l = long_all.rename(columns={"uid": "loser_uid", "serve_form": "serve_form_l"})[
    ["orig_idx", "loser_uid", "serve_form_l"]]

arc = arc.merge(form_w, on=["orig_idx", "winner_uid"], how="left")
arc = arc.merge(form_l, on=["orig_idx", "loser_uid"], how="left")

cov = arc[["serve_form_w", "serve_form_l"]].notna().all(axis=1).mean()
print(f"serve_form coverage (both sides, min_periods=3): {cov:.4f}")

# anchor: a known server-dominant player (Isner) should show high serve_form
isner_rows = arc[(arc["winner_uid"] == "atp:104545") | (arc["loser_uid"] == "atp:104545")]
if len(isner_rows):
    vals = pd.concat([
        isner_rows.loc[isner_rows["winner_uid"] == "atp:104545", "serve_form_w"],
        isner_rows.loc[isner_rows["loser_uid"] == "atp:104545", "serve_form_l"],
    ]).dropna()
    print(f"anchor: Isner (atp:104545) mean serve_form = {vals.mean():.4f} "
          f"(n={len(vals)}, expect well above tour average ~0.60-0.63)")

out = arc[["date", "tour", "winner_uid", "loser_uid", "serve_form_w", "serve_form_l"]]
out.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\serve_form_full.parquet")
print(f"saved {len(out):,} rows")
