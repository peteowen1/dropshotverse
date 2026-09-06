"""Regenerates tune_scratch/elo_raw_ratings_full.parquet with the
tennismylife bridge appended (see bridge_tennismylife.py) - fixes the
stale-ratings gap Pete flagged (archive stops 2026-05-25 since Sackmann's
repos went 404). Reuses elo_raw_ratings_for_xgb.py's own functions
unmodified (that script already documents its own verification that its
output matches elo_baseline.py's frozen p_winner exactly); this script only
adds the bridge rows before its prepare() call.

blend=0.0 is used directly (not re-derived via the blend grid) since it's
this project's own already-locked frozen config, documented in every model-
building script this session (see V25_HIER_PARAMS' sibling constant in
train_best_model.py and elsewhere) - re-deriving it here would risk a
different answer under different data and silently drift from what every
other script assumes.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

import elo_raw_ratings_for_xgb as erx
from bridge_tennismylife import build_bridge, log

import _common
SCRATCH = _common.SCRATCH
OUT_PATH = SCRATCH / "elo_raw_ratings_full.parquet"
BLEND = 0.0
DECAY = 0.3


def main() -> None:
    if OUT_PATH.exists():
        backup = SCRATCH / "elo_raw_ratings_full.PRE_BRIDGE_BACKUP.parquet"
        if not backup.exists():
            shutil.copy(OUT_PATH, backup)
            log(f"backed up pre-bridge parquet -> {backup}")

    log("loading archive...")
    raw = erx.load_matches()
    bridges = [build_bridge(tour) for tour in ("atp", "wta")]
    combined_raw = pd.concat([raw, *bridges], ignore_index=True)
    log(f"combined: {len(raw):,} archive rows + {sum(len(b) for b in bridges):,} bridge rows "
        f"= {len(combined_raw):,}")

    df = erx.prepare(combined_raw)
    log(f"prepared: {len(df):,} rows kept, date range {df['date'].min().date()} -> {df['date'].max().date()}")

    log(f"fitting Elo (blend={BLEND}, decay={DECAY})...")
    fitted = erx.run_elo_raw(df, BLEND, DECAY)

    out_cols = ["date", "tour", "level_file", "surface", "round", "best_of",
               "winner_uid", "loser_uid", "winner_name", "loser_name",
               "ro_winner", "ro_loser", "rs_winner", "rs_loser",
               "n_all_winner", "n_all_loser", "n_surf_winner", "n_surf_loser",
               "p_winner_recomputed"]
    fitted[out_cols].to_parquet(OUT_PATH)
    log(f"wrote {OUT_PATH} ({len(fitted):,} rows)")


if __name__ == "__main__":
    main()
