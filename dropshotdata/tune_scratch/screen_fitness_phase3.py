"""Phase 3: the within-player test. Does a player underperform THEIR OWN
baseline in the weeks after they retire from a match?

Phase 2's between-group anchors failed and deserved to. "Never broke down" is
mostly "has not played many matches", and Elo underrates unconverged players -
so the control group carried a -1.8% bias of its own and every between-group
contrast was measuring career stage as much as fitness (stats-discipline #5:
the arms differed on more than one axis).

This design compares each player to HIMSELF. Every player-level confound -
skill, age, career stage, injury-proneness, how much they play, which tour -
is differenced out, because both cells of the comparison are the same person.
What is left is time-varying: the thing we actually want.

  unit of observation : one player-match (each match contributes 2)
  residual            : won (1/0) minus Elo's own pre-match p for that player
  treatment cell      : residual where 0 <= days since own last RET/WO < 30
  control cell        : residual where days since own last RET/WO >= 365,
                        or the player had not yet broken down at all
  statistic           : per-player (treated_mean - control_mean), averaged over
                        players with >= MIN_OBS observations in BOTH cells,
                        with a paired bootstrap CI over players

Pre-registered anchors (#1):
  C1  The paired effect is NEGATIVE (player wins less than their own norm in
      the 30 days after retiring). A positive effect kills the candidate.
  C2  The 95% bootstrap CI excludes zero. A CI spanning zero is UNRESOLVED,
      not "promising" (stats-discipline #10).
  C3  Direction holds in BOTH tours computed separately. A one-tour-only
      effect is a subgroup finding, not a mechanism.
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

MIN_OBS = 10          # per player, per cell
N_BOOT = 2000
RNG = np.random.default_rng(20260905)


def log(m: str = "") -> None:
    print(m, flush=True)


def to_player_rows(j: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two player-match observations."""
    w = pd.DataFrame({
        "uid": j["winner_uid"].values,
        "tour": j["tour"].values,
        "date": j["date"].values,
        "won": 1.0,
        "p_elo": j["p_winner_recomputed"].values,
        "days_since_bd": j["days_since_bd_w"].values,
    })
    l = pd.DataFrame({
        "uid": j["loser_uid"].values,
        "tour": j["tour"].values,
        "date": j["date"].values,
        "won": 0.0,
        "p_elo": 1.0 - j["p_winner_recomputed"].values,
        "days_since_bd": j["days_since_bd_l"].values,
    })
    pr = pd.concat([w, l], ignore_index=True)
    pr["resid"] = pr["won"] - pr["p_elo"]
    return pr


def paired_effect(pr: pd.DataFrame, label: str) -> tuple[float, float, float, int]:
    treated = (pr["days_since_bd"] >= 0) & (pr["days_since_bd"] < 30)
    control = pr["days_since_bd"].isna() | (pr["days_since_bd"] >= 365)

    t = pr[treated].groupby("uid")["resid"].agg(["mean", "size"])
    c = pr[control].groupby("uid")["resid"].agg(["mean", "size"])
    both = t.join(c, lsuffix="_t", rsuffix="_c", how="inner")
    both = both[(both["size_t"] >= MIN_OBS) & (both["size_c"] >= MIN_OBS)]
    if both.empty:
        log(f"  {label}: no players clear MIN_OBS={MIN_OBS} in both cells")
        return (np.nan, np.nan, np.nan, 0)

    diff = (both["mean_t"] - both["mean_c"]).values
    n_players = len(diff)
    point = float(diff.mean())
    boot = np.array([diff[RNG.integers(0, n_players, n_players)].mean()
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    log(f"  {label}: players={n_players:,}  "
        f"obs treated={int(both['size_t'].sum()):,} control={int(both['size_c'].sum()):,}")
    log(f"    within-player effect = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return point, float(lo), float(hi), n_players


def main() -> None:
    raw = build_counts(classify(load_raw()))
    completed = raw[~(raw["is_ret"] | raw["is_wo"])].copy()
    elo = pd.read_parquet(DD / "tune_scratch" / "elo_raw_ratings_full.parquet",
                          columns=["date", "tour", "winner_uid", "loser_uid",
                                   "p_winner_recomputed"])
    j = completed.merge(
        elo.drop_duplicates(["date", "tour", "winner_uid", "loser_uid"]),
        on=["date", "tour", "winner_uid", "loser_uid"], how="inner")

    pr = to_player_rows(j)
    log(f"coverage: {len(j):,} matches -> {len(pr):,} player-match observations "
        f"({pr['date'].min()} to {pr['date'].max()})")
    log(f"  observations with a prior breakdown on record: "
        f"{pr['days_since_bd'].notna().sum():,} ({pr['days_since_bd'].notna().mean():.1%})")
    log(f"  mean residual overall (Elo calibration sanity): "
        f"{pr['resid'].mean():+.5f}  <- should sit near 0")

    log("\n--- anchors C1/C2: within-player effect of a retirement in the last 30 days ---")
    point, lo, hi, npl = paired_effect(pr, "ALL")
    log(f"\n  C1 (effect < 0)      -> {'PASS' if point < 0 else 'FAIL'}")
    log(f"  C2 (CI excludes 0)   -> {'PASS' if hi < 0 else 'FAIL — UNRESOLVED'}")

    log("\n--- anchor C3: both tours separately ---")
    res = {}
    for tour in ("atp", "wta"):
        res[tour] = paired_effect(pr[pr["tour"] == tour], tour.upper())
    both_neg = all(res[t][0] < 0 for t in res)
    log(f"\n  C3 (same direction both tours) -> {'PASS' if both_neg else 'FAIL'}")

    log("\n--- dose-response, within player (not an anchor, a shape check) ---")
    for lo_d, hi_d in ((0, 14), (14, 30), (30, 90), (90, 365)):
        sub = pr.copy()
        sub["days_since_bd"] = np.where(
            (pr["days_since_bd"] >= lo_d) & (pr["days_since_bd"] < hi_d),
            0.0,                                  # mark as treated
            np.where(pr["days_since_bd"].isna() | (pr["days_since_bd"] >= 365),
                     np.nan, -1.0))               # control stays NaN, else excluded
        sub = sub[sub["days_since_bd"] != -1.0]
        paired_effect(sub, f"{lo_d}-{hi_d}d")


if __name__ == "__main__":
    main()
