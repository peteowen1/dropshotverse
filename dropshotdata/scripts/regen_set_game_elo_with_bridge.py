"""Regenerates tune_scratch/set_elo_raw_ratings_full.parquet and
game_elo_raw_ratings_full.parquet with the tennismylife bridge appended, the
set/game counterpart of regen_elo_raw_with_bridge.py.

Why (OPUS-redteam-2026-10-25 F1): both files were built once, on 2026-08-29,
by set_elo_raw_for_xgb.py and game_elo_raw_for_xgb.py from the Sackmann
archive alone (which ends at Roland Garros 2026), then published to the
model-inputs-2026-09 Release. The daily ratings job and the event forecast
only ever ran regen_elo_raw_with_bridge.py, so overall and surface Elo were
current while game Elo, the model's largest input (D-43), stayed at
Roland Garros 2026 (last tour-level row 2026-05-25, last row of any level
2026-06-02). elo_freshness.py now stops any build where that happens again.

Same input as every other bridged reader: hierarchical_model.prepare() over
hierarchical_model.load_matches() plus the bridge rows (bridge_tennismylife.
load_matches_with_bridge), which is what the ratings build itself replays.
Bridge rows are all dated after 2026-05-25, so every earlier row keeps its
position and its ratings: checked 2026-10-25, the 1,187,504 rows dated before
2026-05-25 are identical to the frozen file's. Only Roland Garros 2026 onward
changes. Output columns and dtypes are unchanged, so the
readers (xgb_bt_rating.py) need no change.

    python regen_set_game_elo_with_bridge.py                # both, into tune_scratch/
    python regen_set_game_elo_with_bridge.py --out-dir D    # write elsewhere (a scratch check)
    python regen_set_game_elo_with_bridge.py --archive-only # the original 08-29 build, no bridge

The prepared-matches pickle cache is never used or written here: the bridged
frame changes daily, and a read-only run must not touch tune_scratch/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

import _common
import bridge_tennismylife as bridge
import game_elo_raw_for_xgb as gelo
import hierarchical_model as hm
import set_elo_raw_for_xgb as selo

log = _common.log


def load_df(archive_only: bool) -> pd.DataFrame:
    raw = hm.load_matches() if archive_only else bridge.load_matches_with_bridge(hm)
    df, _ = hm.prepare(raw)
    return df


def write_atomic(out: pd.DataFrame, path: Path) -> None:
    """Write beside the target, then rename, so a crash mid-write leaves the
    previous file rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp)
    os.replace(tmp, path)


def check_output(out: pd.DataFrame, df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Coverage, not presence: one row per input match, no NaN ratings, and the
    last tour-level date per tour equal to the input's."""
    problems = []
    if len(out) != len(df):
        problems.append(f"{len(out):,} rows out for {len(df):,} in")
    for c in cols:
        n_na = int(out[c].isna().sum())
        if n_na:
            problems.append(f"{c}: {n_na:,} NaN")
    for tour in ("atp", "wta"):
        want = df.loc[(df["tour"] == tour) & (df["level_file"] == "tour"), "date"].max()
        got = out.loc[(out["tour"] == tour) & (out["level_file"] == "tour"), "date"].max()
        if pd.isna(got) or got != want:
            problems.append(f"{tour}: last tour-level date {got} != input's {want}")
    if problems:
        raise SystemExit(f"{name} Elo output failed its checks, not written:\n  " + "\n  ".join(problems))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild set and game Elo with the tennismylife bridge.")
    ap.add_argument("--out-dir", type=Path, default=_common.SCRATCH)
    ap.add_argument("--only", choices=("set", "game"))
    ap.add_argument("--archive-only", action="store_true",
                    help="no bridge rows: reproduces the frozen 2026-08-29 build")
    a = ap.parse_args(argv)

    t0 = time.time()
    log(f"loading {'archive only' if a.archive_only else 'archive + tennismylife bridge'}...")
    df = load_df(a.archive_only)
    log(f"prepared: {len(df):,} rows, {df['date'].min().date()} -> {df['date'].max().date()} "
        f"({time.time() - t0:.0f} s)")

    jobs = [("set", selo.replay_set_elo, selo.OUT_NAME, ["set_elo_winner", "set_elo_loser", "p_winner_set_elo"]),
            ("game", gelo.replay_game_elo, gelo.OUT_NAME, ["game_elo_winner", "game_elo_loser", "p_winner_game_elo"])]
    for name, replay, out_name, cols in jobs:
        if a.only and a.only != name:
            continue
        t1 = time.time()
        log(f"replaying {name} Elo...")
        out = replay(df)
        check_output(out, df, cols, name)
        path = a.out_dir / out_name
        write_atomic(out, path)
        last = {t: out.loc[(out["tour"] == t) & (out["level_file"] == "tour"), "date"].max().date()
                for t in ("atp", "wta")}
        log(f"wrote {path} ({len(out):,} rows, last tour-level match atp {last['atp']} wta {last['wta']}, "
            f"{time.time() - t1:.0f} s)")
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
