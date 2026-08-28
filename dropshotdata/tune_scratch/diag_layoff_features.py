"""Diagnostic (read-only): derive per-match layoff (weeks since each player's
previous match in the FULL archive, any tour/level/regime) and experience
(n prior matches) features, replicating exactly what PlayerState.last_seen /
n_matches / inflate() track in hierarchical_model.py's run() loop - but without
touching that file. Single forward pass over the cached prepared archive
(already sorted by date), O(n) with plain dicts.

Writes tune_scratch/diag_layoff.parquet: one row per (date, winner_uid,
loser_uid) match in the full archive, with weeks_since_last_w/l and
n_prior_w/l (BEFORE this match, matching what the model would have known at
prediction time).
"""
import pickle

import numpy as np
import pandas as pd

CACHE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl"
OUT = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_layoff.parquet"

if __name__ == "__main__":
    # nosec: this cache is written by hierarchical_model.py's own load_prepared()
    # earlier in this same session (per the task brief, "safe to load") - not an
    # externally-sourced pickle.
    with open(CACHE, "rb") as f:
        df, dropped = pickle.load(f)
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    print(f"archive rows: {len(df):,}  date range {df['date'].min()} .. {df['date'].max()}")

    last_seen: dict[str, pd.Timestamp] = {}
    n_matches: dict[str, int] = {}

    weeks_w = np.full(len(df), np.nan)
    weeks_l = np.full(len(df), np.nan)
    nprior_w = np.zeros(len(df), dtype=np.int32)
    nprior_l = np.zeros(len(df), dtype=np.int32)

    dates = df["date"].values
    wuids = df["winner_uid"].values
    luids = df["loser_uid"].values

    for i in range(len(df)):
        w, l, d = wuids[i], luids[i], dates[i]
        nprior_w[i] = n_matches.get(w, 0)
        nprior_l[i] = n_matches.get(l, 0)
        lw = last_seen.get(w)
        ll_ = last_seen.get(l)
        if lw is not None:
            weeks_w[i] = (d - lw) / np.timedelta64(1, "D") / 7.0
        if ll_ is not None:
            weeks_l[i] = (d - ll_) / np.timedelta64(1, "D") / 7.0
        last_seen[w] = d
        last_seen[l] = d
        n_matches[w] = n_matches.get(w, 0) + 1
        n_matches[l] = n_matches.get(l, 0) + 1
        if (i + 1) % 300000 == 0:
            print(f"  ...{i+1:,}/{len(df):,}")

    out = pd.DataFrame({
        "date": df["date"].values,
        "winner_uid": wuids,
        "loser_uid": luids,
        "weeks_since_last_w": weeks_w,
        "weeks_since_last_l": weeks_l,
        "n_prior_w": nprior_w,
        "n_prior_l": nprior_l,
    })
    out.to_parquet(OUT)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print(out[["weeks_since_last_w", "weeks_since_last_l", "n_prior_w", "n_prior_l"]].describe())
