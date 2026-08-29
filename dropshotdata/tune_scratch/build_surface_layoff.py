"""Round 17: R128 within ATP-Grass has gap +0.051 (n=95) - more than double
any other segment this session. R128 at a Slam is essentially Wimbledon's
opening round, the first grass match of the season for most of the field.
weeks_since_last_diff (already in the model) is OVERALL layoff, not
surface-specific - it doesn't tell the model "this player's SURFACE rating
specifically is stale." Building weeks_since_last_SURFACE_match as a new,
explicit staleness signal - lets the tree condition confidence on it
directly, rather than (round 16's rejected approach) trying to fix the
underlying Elo mechanism itself."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"

# same causal round-ordering table used throughout this session
# (elo_baseline.py, hierarchical_model.py, build_widedata*.py) - `date` is a
# TOURNAMENT start date, identical across every round of one event, so a
# date-only sort leaves same-date rows in arbitrary file order. Missing here
# would corrupt "weeks since last surface match" with lookahead state.
ROUND_RANK = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4, "ER": 5,
              "R128": 6, "R64": 7, "R32": 8, "R16": 9,
              "QF": 10, "RR": 11, "SF": 12, "F": 13, "BR": 13}


def load_raw_years(tour, years):
    frames = [pd.read_csv(rf"{DD}\raw\github\{tour}\{tour}_matches_{y}.csv", low_memory=False)
              for y in years]
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    df["surface"] = df["surface"].astype(str).str.strip().str.title()
    rnd = df["round"].astype(str).str.strip().replace({"f": "F"})
    df["round_rank"] = rnd.map(ROUND_RANK).fillna(99)
    return df[["date", "tour", "winner_uid", "loser_uid", "surface", "round_rank"]]


years = list(range(1995, 2026))  # includes 2024-25 holdout years too
raw = pd.concat([load_raw_years("atp", years), load_raw_years("wta", years)], ignore_index=True)
raw = raw.sort_values(["date", "round_rank"], kind="stable").reset_index(drop=True)

last_seen_surf: dict[tuple[str, str], pd.Timestamp] = {}
weeks_w = np.full(len(raw), np.nan)
weeks_l = np.full(len(raw), np.nan)
for i, (w, l, s, d) in enumerate(zip(raw["winner_uid"].values, raw["loser_uid"].values,
                                      raw["surface"].values, raw["date"].values)):
    d = pd.Timestamp(d)
    lw = last_seen_surf.get((w, s))
    ll = last_seen_surf.get((l, s))
    if lw is not None:
        weeks_w[i] = (d - lw).days / 7.0
    if ll is not None:
        weeks_l[i] = (d - ll).days / 7.0
    last_seen_surf[(w, s)] = d
    last_seen_surf[(l, s)] = d

raw["weeks_since_last_surf_w"] = weeks_w
raw["weeks_since_last_surf_l"] = weeks_l
raw = raw.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])

# anchor: ATP-Grass R128 rows should show a large, mostly-first-grass-of-year
# weeks_since_last_surf (a genuine, expected pattern to verify before trusting)
check = raw[(raw["tour"] == "atp") & (raw["surface"] == "Grass") & (raw["date"] >= "2024-01-01")]
print(f"ATP-Grass 2024-25 rows: {len(check)}")
print(f"  median weeks_since_last_surf (winner side): {check['weeks_since_last_surf_w'].median():.1f}")
print(f"  weeks_since_last_surf > 40 (i.e. season-opener-like): "
      f"{(check['weeks_since_last_surf_w'] > 40).mean():.2%}")

raw.to_parquet(rf"{DD}\tune_scratch\surface_layoff_full.parquet")
print(f"saved {len(raw):,} rows")
