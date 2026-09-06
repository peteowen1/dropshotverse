"""Ticket 28 round 2: margin-of-victory (games-won) as an Elo UPDATE-time
signal - explicitly NOT a prediction-time feature (this match's own margin
predicting this match's own outcome would be leakage; margin only exists
once the match is over). This is the missing piece tickets 22/25 both
flagged as still open.

Read-only wrt elo_baseline.py - imports its load_matches()/prepare() (the
already-fixed, causally-ordered loader) rather than duplicating or editing
it, per this session's own established convention (elo_raw_ratings_for_xgb.py
did the same for XGBoost's feature source).

MOV mechanism (538-style dampened multiplier, adapted to tennis games):
  margin_share = (games_won_by_winner - games_won_by_loser) / total_games,
  clipped to [0, 1] - straight-set win ~0.5-0.7, tight 3-setter ~0.05-0.15.
  mult = 1 + MOV_SCALE * margin_share / (1 + |pre-match rating diff| / DAMP_C)
  - the DAMP_C term shrinks the multiplier when the favorite was already
  expected to win big (a blowout by the pre-match favorite is less
  informative than the same blowout by the pre-match underdog), matching
  538's own NFL/NBA justification for dampening MOV by pre-game spread.
  Hard-clipped to [0.5, 2.0] - this project's own repeated lesson (bugs 6-9,
  ticket 24) is that an unbounded per-match multiplier is how a rating
  corrupts; never ship one without a bound.

`decay` is held fixed at 0.3 throughout (ticket 25's locked value) - this
is a NEW, separate axis (stats-discipline #5), not a re-tune of decay.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import elo_baseline as eb  # noqa: E402

_SET_SCORE_RE = __import__("re").compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")


def parse_score_games(score: str) -> tuple[float, float]:
    """Verbatim port of hierarchical_model.py's parse_score_games (read-only
    copy, not an edit of that file)."""
    total_w = total_l = 0
    tokens = str(score).split()
    if not tokens:
        return math.nan, math.nan
    for tok in tokens:
        m = _SET_SCORE_RE.match(tok)
        if not m:
            return math.nan, math.nan
        total_w += int(m.group(1))
        total_l += int(m.group(2))
    if total_w + total_l <= 0:
        return math.nan, math.nan
    return float(total_w), float(total_l)


def run_elo_mov(df: pd.DataFrame, decay: float, mov_scale: float, damp_c: float) -> pd.DataFrame:
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values
    gw, gl = zip(*df["score"].astype(str).map(parse_score_games))
    gw, gl = np.array(gw), np.array(gl)
    total_g = gw + gl
    margin_share = np.where(total_g > 0, np.clip((gw - gl) / np.where(total_g > 0, total_g, 1), 0, 1), np.nan)

    preds = np.empty(len(df))
    mults = np.empty(len(df))
    for idx, (w, l, s, day) in enumerate(zip(df["winner_uid"].values, df["loser_uid"].values,
                                              df["surface"].values, days)):
        ro_w, ro_l = overall.get(w, eb.START), overall.get(l, eb.START)
        rs_w = surf.get((w, s), eb.START)
        rs_l = surf.get((l, s), eb.START)

        if decay > 0.0:
            ro_w = eb.layoff_decay(ro_w, day - last_all.get(w, day), decay)
            ro_l = eb.layoff_decay(ro_l, day - last_all.get(l, day), decay)
            rs_w = eb.layoff_decay(rs_w, day - last_surf.get((w, s), day), decay)
            rs_l = eb.layoff_decay(rs_l, day - last_surf.get((l, s), day), decay)

        diff = ro_w - ro_l   # blend=0 (pure overall Elo) throughout, ticket 25's basis
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        preds[idx] = p_w

        ms = margin_share[idx]
        if np.isnan(ms):
            mult = 1.0   # RET/W-O/irregular score: no MOV info, unweighted update
        else:
            mult = 1.0 + mov_scale * ms / (1.0 + abs(diff) / damp_c)
            mult = float(np.clip(mult, 0.5, 2.0))
        mults[idx] = mult

        kw, kl = eb.k_factor(n_all.get(w, 0)), eb.k_factor(n_all.get(l, 0))
        overall[w] = ro_w + kw * mult * (1 - p_w)
        overall[l] = ro_l - kl * mult * (1 - p_w)

        ksw, ksl = eb.k_factor(n_surf.get((w, s), 0)), eb.k_factor(n_surf.get((l, s), 0))
        surf[(w, s)] = rs_w + ksw * mult * (1 - p_w)
        surf[(l, s)] = rs_l - ksl * mult * (1 - p_w)

        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1
        last_all[w] = day
        last_all[l] = day
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
        last_surf[(w, s)] = day
        last_surf[(l, s)] = day

    out = df.copy()
    out["p_winner"] = preds
    out["mov_mult"] = mults
    return out


def main() -> None:
    raw = eb.load_matches()
    df, dropped = eb.prepare(raw)
    print("dropped:", dropped)

    # anchors (stats-discipline #1) - checked on the mov=0 no-op arm first to
    # confirm mult=1.0 everywhere reproduces the frozen decay=0.3 arm exactly.
    baseline_arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="sackmann")

    noop_arm = run_elo_mov(df, decay=0.3, mov_scale=0.0, damp_c=1e9)
    max_dev = float(np.max(np.abs(noop_arm["p_winner"].values - baseline_arm["p_winner"].values)))
    print(f"no-op check (mov_scale=0): max |p_winner diff| vs frozen decay=0.3 arm = {max_dev:.2e} "
          f"({'OK structural no-op' if max_dev < 1e-9 else 'FAIL - mechanism differs, stop'})")
    assert max_dev < 1e-9, "MOV mechanism is not a true no-op at mov_scale=0 - do not trust anything below"

    # tour-level, market-odds-joined holdout scoring - same pattern as tickets 25/26.
    # Odds loaded ONCE (was re-read from disk + re-devigged on every grid
    # point - the original timed out after 5min doing this 16x over).
    sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
    from market_compare import key_from_full, key_from_short, load_odds  # noqa: E402

    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds_small = odds[["pair", "tour", "date", "k_w", "PSW", "PSL"]]

    def score_on_holdout(arm: pd.DataFrame, label: str) -> None:
        sub = arm[(arm["level_file"] == "tour") & (arm["date"] >= "2024-01-01")].copy()
        sub["k_w"] = sub["winner_name"].map(key_from_full)
        sub["k_l"] = sub["loser_name"].map(key_from_full)
        sub["pair"] = sub.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
        j = sub.merge(odds_small, on=["pair", "tour"], how="left", suffixes=("", "_o"))
        j["gap"] = (j["date_o"] - j["date"]).dt.days
        j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
        j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
        j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
        pr = np.clip(j["p_winner"].values, 1e-9, 1 - 1e-9)
        ll = float(-np.log(pr).mean())
        acc = float((pr > 0.5).mean())
        print(f"  {label:40s} n={len(j):5,d}  logloss={ll:.5f}  acc={acc:.4f}", flush=True)

    print("\n=== no-op arm (should match Elo's frozen 0.6120 basis) ===")
    score_on_holdout(noop_arm, "mov_scale=0.0 (no-op)")

    print("\n=== MOV grid (decay=0.3 fixed) ===")
    best = None
    for mov_scale in (0.3, 0.6, 1.0, 1.5, 2.0):
        for damp_c in (100.0, 200.0, 400.0):
            arm = run_elo_mov(df, decay=0.3, mov_scale=mov_scale, damp_c=damp_c)
            sub = arm[(arm["level_file"] == "tour") & (arm["date"] >= "2024-01-01")]
            label = f"mov_scale={mov_scale} damp_c={damp_c}"
            score_on_holdout(arm, label)


if __name__ == "__main__":
    main()
