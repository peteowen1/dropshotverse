"""Phase 4: event study. Compare the weeks AFTER a retirement to the same
player's weeks shortly BEFORE it.

Phase 3's within-player effect (-0.0366, CI excluding zero, both tours) passed
every anchor and is still not trustworthy as an injury effect, because its
dose-response was FLAT from 14 days out to a year and weakest in the first
fortnight. Acute injury does not do that. The cause: phase 3's control cell was
">=365d since a breakdown OR never broke down", and a player's never-broke-down
observations are all early-career. Elo lags a player's own improvement upward
and their decline downward, so the contrast was partly early-career vs
late-career - a confound that survives within-player differencing because it is
within-CAREER (stats-discipline #5, third time in this investigation).

An event study fixes it by making the control LOCAL IN TIME. Career trajectory
is near-constant across a four-month span, so a window 60-120 days before the
retirement is the same player at the same career stage in the same form - the
only thing that differs is the retirement itself.

  baseline window  : [-120, -60) days before the breakdown
  event windows    : [-60,-30), [-30,0), [0,+30), [+30,+90), [+90,+180)
  statistic        : mean Elo residual in each window MINUS the same player's
                     baseline-window mean, averaged over events, bootstrap CI
                     clustered on PLAYER (not event - one player contributes
                     many correlated events)

Pre-registered anchors (#1):
  D1  [0,+30) is negative relative to baseline. This is the effect itself.
  D2  [-120,-60) is 0 by construction; [-60,-30) - the placebo - is NOT
      materially negative. If the "effect" is already there two months BEFORE
      the retirement, it is career trajectory, not injury, and the candidate
      dies here.
  D3  [+90,+180) is closer to zero than [0,+30) - the player recovers.
      A permanent step down would mean something other than acute injury.
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
N_BOOT = 2000
RNG = np.random.default_rng(20260905)


def log(m: str = "") -> None:
    print(m, flush=True)


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

    # every breakdown event: (player, date)
    ev = raw.loc[raw["breakdown_uid"].notna(), ["breakdown_uid", "date"]].copy()
    ev = ev.rename(columns={"breakdown_uid": "uid"}).drop_duplicates()
    log(f"coverage: {len(ev):,} breakdown events over {ev['uid'].nunique():,} players; "
        f"{len(pr):,} player-match observations to draw windows from")

    # For each event, pull that player's observations and bucket by day offset.
    pr_by_uid = {u: g for u, g in pr.groupby("uid", sort=False)}
    recs = []
    for uid, edate in zip(ev["uid"].values, ev["date"].values):
        g = pr_by_uid.get(uid)
        if g is None:
            continue
        off = (g["date"].values - edate) / np.timedelta64(1, "D")
        base_mask = (off >= BASELINE[0]) & (off < BASELINE[1])
        if base_mask.sum() < MIN_BASE_OBS:
            continue
        base = float(g["resid"].values[base_mask].mean())
        row = {"uid": uid, "tour": g["tour"].iloc[0], "n_base": int(base_mask.sum())}
        for lo, hi in WINDOWS:
            m = (off >= lo) & (off < hi)
            row[f"w{lo}_{hi}"] = (float(g["resid"].values[m].mean()) - base
                                  if m.sum() else np.nan)
            row[f"n{lo}_{hi}"] = int(m.sum())
        recs.append(row)

    ev_df = pd.DataFrame(recs)
    log(f"  events retained (>= {MIN_BASE_OBS} baseline observations): "
        f"{len(ev_df):,} over {ev_df['uid'].nunique():,} players")

    log("\n--- event study: Elo residual relative to the player's own "
        "[-120,-60d) baseline ---")
    log("  (negative = player underperforms their own pre-retirement form)")
    uids = ev_df["uid"].values
    uniq = np.unique(uids)
    out = []
    for lo, hi in WINDOWS:
        col, ncol = f"w{lo}_{hi}", f"n{lo}_{hi}"
        sub = ev_df[ev_df[col].notna()]
        if sub.empty:
            continue
        point = float(sub[col].mean())
        # bootstrap clustered on player
        su = sub["uid"].values
        groups = {u: sub[col].values[su == u] for u in np.unique(su)}
        keys = list(groups)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = RNG.integers(0, len(keys), len(keys))
            boot[b] = np.concatenate([groups[keys[i]] for i in pick]).mean()
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        out.append(dict(window=f"[{lo},{hi})", events=len(sub),
                        obs=int(sub[ncol].sum()), effect=point,
                        ci_lo=lo_ci, ci_hi=hi_ci))
    tab = pd.DataFrame(out)
    log(tab.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    g = {r["window"]: r for _, r in tab.iterrows()}
    after = g["[0,30)"]["effect"]
    placebo = g["[-60,-30)"]["effect"]
    late = g["[90,180)"]["effect"]
    log(f"\n  D1 (post-event [0,30) negative)        -> "
        f"{'PASS' if after < 0 else 'FAIL'}  ({after:+.4f}, "
        f"CI [{g['[0,30)']['ci_lo']:+.4f}, {g['[0,30)']['ci_hi']:+.4f}])")
    log(f"  D2 (placebo [-60,-30) not negative)    -> "
        f"{'PASS' if placebo > after / 2 else 'FAIL — effect predates the event'}"
        f"  ({placebo:+.4f})")
    log(f"  D3 (recovers by [90,180))              -> "
        f"{'PASS' if late > after else 'FAIL'}  ({late:+.4f} vs {after:+.4f})")

    log("\n--- by tour ---")
    for tour in ("atp", "wta"):
        s = ev_df[ev_df["tour"] == tour]
        log(f"  {tour.upper()}: [0,30) = {s['w0_30'].mean():+.4f} "
            f"(n_events={s['w0_30'].notna().sum():,})   "
            f"placebo [-60,-30) = {s['w-60_-30'].mean():+.4f}")

    ev_df.to_parquet(DD / "tune_scratch" / "fitness_event_study.parquet", index=False)
    log(f"\nwrote {DD / 'tune_scratch' / 'fitness_event_study.parquet'}")


if __name__ == "__main__":
    main()
