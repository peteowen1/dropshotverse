"""Phase 2: is the persistent deficit a FITNESS trait, or an Elo BOOKKEEPING bug?

Phase 1 found a clean dose-response (deficit 2.69% at 0-30d decaying to 0.55%
at 365d+). The decay is what a fitness story predicts. The FLOOR is not: a
player who broke down 3 years ago should, on a pure fitness story, be
indistinguishable from one who never broke down at all. They are not - they are
still overrated by 0.55%.

Competing explanation, and the reason this matters more than the feature:
elo_baseline.prepare() DROPS every RET/W/O row before the engine sees it, so a
player who retires mid-match never takes the rating debit for that loss. Their
opponent never takes the credit. If that is what the floor is, the fix is not a
new feature at all - it is scoring those matches (possibly at reduced K), and
it would be a correctness fix to the ratings themselves, which feed everything
downstream including the published US Open forecast.

The two stories make OPPOSITE predictions, which is what makes this decidable:

  FITNESS trait   -> deficit tracks TIME since the last breakdown, and is flat
                     in the career COUNT once you condition on time.
  BOOKKEEPING     -> deficit tracks the COUNT of undebited breakdowns, and
                     survives conditioning on time (a debit never applied does
                     not heal).

Pre-registered anchors (#1, written before running):
  B1  Players who have NEVER broken down have deficit ~0 (|deficit| < 0.005).
      If the never-group is also biased, neither story is the explanation and
      something else is wrong with the Elo calibration.
  B2  Among players whose last breakdown was >365d ago (fitness recovered by
      any reasonable clinical standard), deficit rises with career breakdown
      count. Bookkeeping predicts yes; pure fitness predicts flat.
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


def log(m: str = "") -> None:
    print(m, flush=True)


def build_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """As-of career breakdown COUNT and days-since, per side, strictly prior."""
    cnt: dict[str, int] = {}
    last: dict[str, pd.Timestamp] = {}
    n = len(raw)
    cw = np.zeros(n, dtype=np.int32)
    cl = np.zeros(n, dtype=np.int32)
    dw = np.full(n, np.nan)
    dl = np.full(n, np.nan)

    dates = raw["date"].values
    w_uids = raw["winner_uid"].values
    l_uids = raw["loser_uid"].values
    bd_uids = raw["breakdown_uid"].values

    for i in range(n):
        d = dates[i]
        cw[i] = cnt.get(w_uids[i], 0)
        cl[i] = cnt.get(l_uids[i], 0)
        pw, pl = last.get(w_uids[i]), last.get(l_uids[i])
        if pw is not None:
            dw[i] = (d - pw) / np.timedelta64(1, "D")
        if pl is not None:
            dl[i] = (d - pl) / np.timedelta64(1, "D")
        bd = bd_uids[i]
        if bd is not None:
            cnt[bd] = cnt.get(bd, 0) + 1
            last[bd] = d

    raw = raw.copy()
    raw["bd_count_w"], raw["bd_count_l"] = cw, cl
    raw["days_since_bd_w"], raw["days_since_bd_l"] = dw, dl
    return raw


def deficit(j: pd.DataFrame, only_w: pd.Series, only_l: pd.Series) -> tuple:
    actual = int(only_w.sum())
    n = int(only_w.sum() + only_l.sum())
    if n == 0:
        return (0, np.nan, np.nan, np.nan)
    expected = float(j.loc[only_w, "p_winner_recomputed"].sum()
                     + (1 - j.loc[only_l, "p_winner_recomputed"]).sum())
    act, exp = actual / n, expected / n
    se = np.sqrt(act * (1 - act) / n)
    return (n, act, exp, exp - act), se


def main() -> None:
    raw = build_counts(classify(load_raw()))
    completed = raw[~(raw["is_ret"] | raw["is_wo"])].copy()

    elo = pd.read_parquet(DD / "tune_scratch" / "elo_raw_ratings_full.parquet",
                          columns=["date", "tour", "winner_uid", "loser_uid",
                                   "p_winner_recomputed"])
    j = completed.merge(
        elo.drop_duplicates(["date", "tour", "winner_uid", "loser_uid"]),
        on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    log(f"coverage: {len(j):,} completed tour matches joined to Elo "
        f"({j['date'].min().date()} to {j['date'].max().date()})")

    # ---------------- anchor B1: the never-broke-down control group
    log("\n--- anchor B1: players who have NEVER broken down ---")
    nw = j["bd_count_w"] == 0
    nl = j["bd_count_l"] == 0
    only_w, only_l = nw & ~nl, nl & ~nw
    (n, act, exp, d), se = deficit(j, only_w, only_l)
    log(f"  n={n:,}  actual={act:.4f}  Elo-expected={exp:.4f}  "
        f"deficit={d:+.4f} (se {se:.4f})")
    log(f"  B1 (|deficit| < 0.005) -> {'PASS' if abs(d) < 0.005 else 'FAIL'}")
    b1_note = ("never-group is UNBIASED, so the bias belongs to the "
               "breakdown population" if abs(d) < 0.005 else
               "never-group is ALSO biased - neither story explains it alone")
    log(f"  reading: {b1_note}")

    # ---------------- anchor B2: count effect with time held out
    log("\n--- anchor B2: career breakdown COUNT, restricted to last "
        "breakdown >365d ago (fitness long since recovered) ---")
    log("  fitness story predicts FLAT across count; bookkeeping predicts RISING")
    rows = []
    for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 100)):
        w_hit = (j["bd_count_w"] >= lo) & (j["bd_count_w"] < hi) & (j["days_since_bd_w"] > 365)
        l_hit = (j["bd_count_l"] >= lo) & (j["bd_count_l"] < hi) & (j["days_since_bd_l"] > 365)
        (n, act, exp, d), se = deficit(j, w_hit & ~l_hit, l_hit & ~w_hit)
        if n:
            rows.append(dict(count=f"{lo}-{hi - 1}", n=n, actual=act,
                             expected=exp, deficit=d, se=se))
    tab = pd.DataFrame(rows)
    log(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(tab) >= 2:
        lo_d, hi_d = tab.iloc[0]["deficit"], tab.iloc[-1]["deficit"]
        rising = hi_d > lo_d + tab.iloc[-1]["se"]
        log(f"\n  B2: deficit {lo_d:+.4f} (1 breakdown) -> {hi_d:+.4f} "
            f"({tab.iloc[-1]['count']} breakdowns) -> "
            f"{'RISING (bookkeeping)' if rising else 'FLAT (fitness only)'}")

    # ---------------- how big is the missing debit, in rating points?
    log("\n--- the undebited mass ---")
    n_ret = int(raw["is_ret"].sum() + raw["is_wo"].sum())
    log(f"  RET/W-O tour matches never scored by the Elo engine: {n_ret:,}")
    per_player = (pd.Series(raw.loc[raw["breakdown_uid"].notna(), "breakdown_uid"])
                  .value_counts())
    log(f"  distinct players affected: {len(per_player):,}")
    log(f"  worst-affected (career RET/W-O count): "
        f"{per_player.head(5).to_dict()}")
    log(f"  median 1, p90 {per_player.quantile(0.90):.0f}, max {per_player.max()}")


if __name__ == "__main__":
    main()
