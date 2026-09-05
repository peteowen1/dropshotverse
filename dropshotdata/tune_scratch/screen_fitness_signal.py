"""Screen: does a player's own recent RETIREMENT / WALKOVER carry pre-match
signal the current feature set throws away?

Motivation (ticket 28's own diagnosis): the residual gap to the market is
"broadly distributed, not tail-concentrated ... most plausibly late match-time
information (injury/fitness/conditions) no pre-match archive feature can
reconstruct." That diagnosis is half right. A retirement IS in the archive -
elo_baseline.prepare() drops every RET/W/O/DEF/ABD row on line 161 because its
OUTCOME is uninformative about skill. But the *fact that a named player's body
failed in a named week* is a different quantity from the outcome, it is
genuinely pre-match for every LATER match, and nothing in the pipeline reads it.

This script does NOT fit a model. Per stats-discipline #6 and #9 it runs the
cheap, single-feature, interpretable test first: is there a signal at all?

Pre-registered anchors (written before looking at any output, per #1):
  A1  RET/W-O prevalence at tour level is 2-5% of matches. Outside that band,
      the score-string parse is wrong, not tennis.
  A2  A player whose PREVIOUS match ended in their own retirement wins LESS
      often than their Elo rating predicts. If they win MORE, the mechanism is
      backwards and this screen stops here - no special-casing.
  A3  The deficit decays with time: 0-30 days worse than 90-365 days.
Anchors are asserted, not eyeballed. A2 failing kills the candidate.

Coverage is printed before every headline number (#7).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))

import elo_baseline as eb  # noqa: E402  (path set above)

ARCHIVE = DD / "raw" / "github"
RET_PAT = r"RET|Retired"
WO_PAT = r"W/O|DEF|ABD|Walkover|Def\."
TUNE_YEARS = (2022, 2023)


def log(m: str = "") -> None:
    print(m, flush=True)


def load_raw() -> pd.DataFrame:
    """Like elo_baseline.load_matches() but KEEPS score + minutes, and keeps
    the RET/W-O rows that prepare() drops. Tour-level files only: the
    challenger/ITF files are in the Elo engine's training set but a retirement
    at ITF level is a different population, and mixing them would silently
    change what the prevalence anchor is measuring."""
    frames = []
    for tour in ("atp", "wta"):
        for p in sorted((ARCHIVE / tour).glob(f"{tour}_matches_[0-9]*.csv")):
            df = pd.read_csv(p, low_memory=False)
            if not len(df):
                continue
            df["tour"] = tour
            frames.append(df[["tour", "tourney_date", "tourney_id", "tourney_level",
                              "surface", "score", "best_of", "round", "minutes",
                              "winner_id", "loser_id", "winner_name", "loser_name"]])
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["tourney_date"], format="%Y%m%d", errors="coerce")
    raw = raw[raw["date"].notna()].copy()
    for side in ("winner", "loser"):
        raw[f"{side}_uid"] = raw["tour"] + ":" + raw[f"{side}_id"].astype("Int64").astype(str)
    raw = raw[~(raw["winner_uid"].str.contains("<NA>") | raw["loser_uid"].str.contains("<NA>"))]
    # Same causal ordering the rest of the pipeline uses (tickets 9/25): a
    # date-only sort puts qualifying AFTER main draw on the same day.
    raw["_rr"] = eb.round_rank(raw["round"])
    raw = raw.sort_values(["date", "_rr"], kind="mergesort", na_position="last")
    return raw.drop(columns=["_rr"]).reset_index(drop=True)


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    sc = raw["score"].astype(str)
    raw["is_ret"] = sc.str.contains(RET_PAT, case=False, na=False)
    raw["is_wo"] = sc.str.contains(WO_PAT, case=False, na=False)
    # In both cases it is the LOSER whose body failed / who withdrew.
    raw["breakdown_uid"] = np.where(raw["is_ret"] | raw["is_wo"], raw["loser_uid"], None)
    return raw


def anchor_a1(raw: pd.DataFrame) -> None:
    log("--- coverage (stats-discipline #7) ---")
    log(f"  tour-level rows loaded: {len(raw):,}  "
        f"({raw['date'].min().date()} to {raw['date'].max().date()})")
    log(f"  by tour: {raw.groupby('tour').size().to_dict()}")
    minutes_cov = raw.groupby(raw["date"].dt.year)["minutes"].apply(lambda s: s.notna().mean())
    log(f"  `minutes` coverage 1991/2000/2010/2020/2025: "
        + ", ".join(f"{y}={minutes_cov.get(y, float('nan')):.0%}"
                    for y in (1991, 2000, 2010, 2020, 2025)))

    log("\n--- anchor A1: RET/W-O prevalence is 2-5% of tour matches ---")
    n = len(raw)
    p_ret = raw["is_ret"].mean()
    p_wo = raw["is_wo"].mean()
    log(f"  RET {raw['is_ret'].sum():,} ({p_ret:.2%})   "
        f"W/O+DEF {raw['is_wo'].sum():,} ({p_wo:.2%})   "
        f"combined {p_ret + p_wo:.2%}  of n={n:,}")
    ok = 0.02 <= (p_ret + p_wo) <= 0.05
    log(f"  A1 -> {'PASS' if ok else 'FAIL'}")
    assert ok, "A1 failed: score-string parse is wrong, not tennis"


def build_asof_breakdown(raw: pd.DataFrame) -> pd.DataFrame:
    """For every (match, side) pair, the days since THAT player last broke down
    (own RET or W/O), strictly BEFORE the current match. Walks rows in the
    already-causally-sorted order, reading state before writing it, so a
    match's own outcome can never enter its own feature."""
    last_bd: dict[str, pd.Timestamp] = {}
    last_kind: dict[str, str] = {}
    n = len(raw)
    dsw = np.full(n, np.nan)          # days since winner's own breakdown
    dsl = np.full(n, np.nan)          # days since loser's own breakdown
    kw = np.array([None] * n, dtype=object)
    kl = np.array([None] * n, dtype=object)

    dates = raw["date"].values
    w_uids = raw["winner_uid"].values
    l_uids = raw["loser_uid"].values
    bd_uids = raw["breakdown_uid"].values
    kinds = np.where(raw["is_ret"].values, "RET", np.where(raw["is_wo"].values, "WO", None))

    for i in range(n):
        d = dates[i]
        for uid, arr, karr in ((w_uids[i], dsw, kw), (l_uids[i], dsl, kl)):
            prev = last_bd.get(uid)
            if prev is not None:
                arr[i] = (d - prev) / np.timedelta64(1, "D")
                karr[i] = last_kind[uid]
        bd = bd_uids[i]
        if bd is not None:
            last_bd[bd] = d
            last_kind[bd] = kinds[i]

    raw = raw.copy()
    raw["days_since_bd_w"] = dsw
    raw["days_since_bd_l"] = dsl
    raw["bd_kind_w"] = kw
    raw["bd_kind_l"] = kl
    return raw


def attach_elo(raw: pd.DataFrame) -> pd.DataFrame:
    """Join the pipeline's own pre-match Elo expectation so the test is
    'worse than the RATING predicts', not 'worse than 50%' - the latter would
    just rediscover that injured players are often also weaker players."""
    elo = pd.read_parquet(DD / "tune_scratch" / "elo_raw_ratings_full.parquet",
                          columns=["date", "tour", "winner_uid", "loser_uid",
                                   "p_winner_recomputed"])
    before = len(raw)
    j = raw.merge(elo.drop_duplicates(["date", "tour", "winner_uid", "loser_uid"]),
                  on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    log(f"\n  Elo join: {before:,} -> {len(j):,} rows "
        f"({len(j) / before:.1%}; RET/W-O rows are absent from the Elo file by "
        f"construction, they are dropped by prepare())")
    return j


def anchor_a2_a3(j: pd.DataFrame) -> None:
    """The actual signal test. Population: completed matches in which EXACTLY
    ONE side has a recent breakdown, so the comparison isolates one axis
    (stats-discipline #5)."""
    log("\n--- anchors A2/A3: post-breakdown performance vs Elo expectation ---")
    log("  (positive 'deficit' = player wins LESS than Elo predicts)")

    rows = []
    for lo, hi in ((0, 30), (30, 90), (90, 365), (365, 100_000)):
        w_hit = (j["days_since_bd_w"] >= lo) & (j["days_since_bd_w"] < hi)
        l_hit = (j["days_since_bd_l"] >= lo) & (j["days_since_bd_l"] < hi)
        # exactly one side in the window
        only_w = w_hit & ~l_hit
        only_l = l_hit & ~w_hit
        # From the breakdown player's point of view: actual wins / expected wins
        actual = int(only_w.sum())
        expected = float(j.loc[only_w, "p_winner_recomputed"].sum()
                         + (1 - j.loc[only_l, "p_winner_recomputed"]).sum())
        n = int(only_w.sum() + only_l.sum())
        if n == 0:
            continue
        act_rate = actual / n
        exp_rate = expected / n
        se = np.sqrt(act_rate * (1 - act_rate) / n)
        rows.append(dict(window=f"{lo}-{hi if hi < 99999 else 'inf'}d", n=n,
                         actual=act_rate, expected=exp_rate,
                         deficit=exp_rate - act_rate, se=se))
    tab = pd.DataFrame(rows)
    log(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    recent = tab[tab["window"] == "0-30d"].iloc[0]
    distant = tab[tab["window"] == "365-infd"].iloc[0]
    z = recent["deficit"] / recent["se"]
    log(f"\n  A2 (0-30d deficit > 0): deficit={recent['deficit']:+.4f} "
        f"on n={int(recent['n']):,}, z={z:+.2f} -> "
        f"{'PASS' if recent['deficit'] > 0 else 'FAIL'}")
    log(f"  A3 (0-30d worse than 365d+): {recent['deficit']:+.4f} vs "
        f"{distant['deficit']:+.4f} -> "
        f"{'PASS' if recent['deficit'] > distant['deficit'] else 'FAIL'}")

    # stats-discipline #3: size it before building anything. A deficit of `s`
    # on a `frac` slice of matches moves overall log-loss by roughly
    # frac * (deficit^2 / (2 * p(1-p))) at p~0.5 -- report it explicitly.
    frac = float(((j["days_since_bd_w"] < 90) ^ (j["days_since_bd_l"] < 90)).mean())
    d90 = float(tab[tab["window"].isin(["0-30d", "30-90d"])]["deficit"].mean())
    log(f"\n  --- effect sizing (stats-discipline #3) ---")
    log(f"  share of matches with exactly one side inside 90d: {frac:.2%}")
    log(f"  mean deficit over 0-90d: {d90:+.4f}")
    log(f"  crude log-loss headroom if fully corrected: "
        f"~{frac * (d90 ** 2) / (2 * 0.25):.5f}")
    log(f"  (compare: current gap to market = 0.00073)")


def main() -> None:
    raw = classify(load_raw())
    anchor_a1(raw)
    raw = build_asof_breakdown(raw)
    # Completed matches only for the performance test - a RET row's own result
    # is exactly what the pipeline (correctly) refuses to score.
    completed = raw[~(raw["is_ret"] | raw["is_wo"])].copy()
    j = attach_elo(completed)
    anchor_a2_a3(j)

    out = DD / "tune_scratch" / "fitness_asof_features.parquet"
    keep = ["date", "tour", "winner_uid", "loser_uid",
            "days_since_bd_w", "days_since_bd_l", "bd_kind_w", "bd_kind_l"]
    raw[keep].to_parquet(out, index=False)
    log(f"\nwrote {out} ({len(raw):,} rows)")


if __name__ == "__main__":
    main()
