"""Phase 5: placebo events. Run phase 4's identical machinery on dates where
NOTHING happened.

Phase 4 rejected the candidate on all three anchors, but produced a shape odd
enough to distrust before reporting: the baseline window sits at 0 by
construction and EVERY other window is ~-0.065, except the 30 days immediately
after the retirement, which is +0.034. "Player does better right after retiring
and worse everywhere else" is not a story about injury; it is the signature of
either a selection effect or a bug in my own window construction.

This is stats-discipline #5's limiting-case check in diagnostic form: re-run the
identical code on random (player, date) pairs drawn from the same players, where
no retirement occurred. Whatever structure survives belongs to the METHOD, not
to retirements.

Predicted before running:
  E1  If the phase-4 shape is real, placebo events show a FLAT profile - every
      window near 0 relative to baseline.
  E2  If the phase-4 shape is an artifact of conditioning on >= 3 baseline
      observations (a player is only "eligible" when actively playing, and
      activity itself predicts form), the placebo profile REPRODUCES the same
      negative wings, and phase 4 measured nothing about retirement at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))
sys.path.insert(0, str(DD / "tune_scratch"))

from screen_fitness_signal import classify, load_raw  # noqa: E402
from screen_fitness_phase2 import build_counts  # noqa: E402
from screen_fitness_phase3 import to_player_rows  # noqa: E402

WINDOWS = [(-120, -60), (-60, -30), (-30, 0), (0, 30), (30, 90), (90, 180)]
BASELINE = (-120, -60)
MIN_BASE_OBS = 3
RNG = np.random.default_rng(4242)


def log(m: str = "") -> None:
    print(m, flush=True)


def profile(pr_by_uid: dict, events: pd.DataFrame, label: str) -> pd.DataFrame:
    recs = []
    for uid, edate in zip(events["uid"].values, events["date"].values):
        g = pr_by_uid.get(uid)
        if g is None:
            continue
        off = (g["date"].values - edate) / np.timedelta64(1, "D")
        bm = (off >= BASELINE[0]) & (off < BASELINE[1])
        if bm.sum() < MIN_BASE_OBS:
            continue
        base = float(g["resid"].values[bm].mean())
        row = {"uid": uid}
        for lo, hi in WINDOWS:
            m = (off >= lo) & (off < hi)
            row[f"[{lo},{hi})"] = (float(g["resid"].values[m].mean()) - base
                                   if m.sum() else np.nan)
        recs.append(row)
    df = pd.DataFrame(recs)
    means = df[[f"[{lo},{hi})" for lo, hi in WINDOWS]].mean()
    log(f"\n  {label}: {len(df):,} events over {df['uid'].nunique():,} players")
    log("    " + "  ".join(f"{k}={v:+.4f}" for k, v in means.items()))
    return means


def main() -> None:
    raw = build_counts(classify(load_raw()))
    completed = raw[~(raw["is_ret"] | raw["is_wo"])].copy()
    elo = pd.read_parquet(DD / "tune_scratch" / "elo_raw_ratings_full.parquet",
                          columns=["date", "tour", "winner_uid", "loser_uid",
                                   "p_winner_recomputed"])
    j = completed.merge(
        elo.drop_duplicates(["date", "tour", "winner_uid", "loser_uid"]),
        on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    pr = to_player_rows(j)[["uid", "tour", "date", "resid"]]
    pr["date"] = pd.to_datetime(pr["date"])
    pr_by_uid = {u: g for u, g in pr.groupby("uid", sort=False)}

    real = raw.loc[raw["breakdown_uid"].notna(), ["breakdown_uid", "date"]]
    real = real.rename(columns={"breakdown_uid": "uid"}).drop_duplicates()
    log("=== REAL retirement/walkover events ===")
    m_real = profile(pr_by_uid, real, "real")

    # Placebo: same players, same number of events each, but the date is drawn
    # uniformly from that player's own COMPLETED-match dates (so the placebo
    # sits inside an active stretch of career, exactly like a real event does).
    log("\n=== PLACEBO events (same players, random dates from their own career) ===")
    rows = []
    per_player = real.groupby("uid").size()
    for uid, k in per_player.items():
        g = pr_by_uid.get(uid)
        if g is None or len(g) == 0:
            continue
        picks = RNG.choice(g["date"].values, size=min(k, len(g)), replace=False)
        rows.extend({"uid": uid, "date": p} for p in picks)
    placebo = pd.DataFrame(rows)
    m_plac = profile(pr_by_uid, placebo, "placebo")

    log("\n--- verdict ---")
    diff = m_real - m_plac
    log("  real MINUS placebo (this is the retirement effect, net of method):")
    log("    " + "  ".join(f"{k}={v:+.4f}" for k, v in diff.items()))
    post = diff["[0,30)"]
    pre = diff["[-60,-30)"]
    log(f"\n  net post-event [0,30) = {post:+.4f}")
    log(f"  net placebo   [-60,-30) = {pre:+.4f}")
    log(f"  E1/E2 -> {'method artifact dominates' if abs(m_plac).max() > 0.02 else 'placebo is flat; phase-4 shape was real'}")


if __name__ == "__main__":
    main()
