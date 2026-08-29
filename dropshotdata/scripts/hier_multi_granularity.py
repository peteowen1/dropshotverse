"""Companion to point_elo.py's Elo granularity sweep - same 4 granularities,
same evidence definitions (overall point-win fraction / games-won fraction /
sets-won fraction / match win-loss), but updated via hier's own Bayesian
machinery (adf_update + inflate) instead of classic Elo's K-factor.

Deliberately uses a SINGLE combined per-player strength scalar (repurposing
PlayerState.m_a/v_a, ignoring m_b/kappa/lambda entirely) rather than hier's
real serve/return-split architecture - matching point_elo.py's single-rating
design exactly, so granularity is the ONLY axis that differs between this
table and the Elo one, not also model complexity (stats-discipline #5). This
is NOT hier's real regime-A/B mechanism (that's the existing, already-
reported 0.72388 point-level number) - it's the same simplified single-
strength architecture at every granularity, for a clean comparison.

Evidence converted to (y, n) binomial counts (not fractions, since
adf_update takes counts) using the exact same quantities as point_elo.py:
  - point: (points won by winner overall, total points)
  - game:  (games won by winner, total games)
  - set:   (sets won by winner, total sets)
  - match: (1, 1) - a single win/loss trial, hier's own regime-B fallback
    for a totally unparseable score, applied here to EVERY match uniformly.

Read-only wrt hierarchical_model.py - imports adf_update/inflate/PlayerState/
MAX_STATE/MAX_VAR/MAX_SHIFT/parse_score_sets rather than reimplementing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hierarchical_model as hm

DD_TUNE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch"
EVAL_YEARS = (2024, 2025)


def log(m: str) -> None:
    print(m, flush=True)


def evidence_for_mode(mode: str, row) -> tuple[float, float] | None:
    """Returns (y, n) = (winner's evidence count, total trials), or None if
    unavailable for this row - same "no silent guessing" discipline as the
    Elo sweep."""
    if mode == "point":
        if not row.has_points or row.w_svpt <= 0 or row.l_svpt <= 0:
            return None
        total_pts = row.w_svpt + row.l_svpt
        pts_won_by_winner = row.w_won + (row.l_svpt - row.l_won)
        return pts_won_by_winner, total_pts
    if mode == "game":
        if pd.isna(row.total_games_w) or pd.isna(row.total_games_l):
            return None
        tot = row.total_games_w + row.total_games_l
        if tot <= 0:
            return None
        return row.total_games_w, tot
    if mode == "set":
        sets = hm.parse_score_sets(row.score)
        if not sets:
            return None
        sets_w = sum(1 for gw, gl in sets if gw > gl)
        tot = len(sets)
        if tot <= 0:
            return None
        return sets_w, tot
    if mode == "match":
        return 1.0, 1.0
    raise ValueError(mode)


def run_mode(mode: str, df: pd.DataFrame) -> pd.DataFrame:
    players: dict[str, hm.PlayerState] = {}

    def get(uid: str) -> hm.PlayerState:
        s = players.get(uid)
        if s is None:
            s = hm.PlayerState()
            players[uid] = s
        return s

    eval_rows = []
    n = len(df)
    used = 0
    for i, row in enumerate(df.itertuples(index=False)):
        ev = evidence_for_mode(mode, row)
        w, l = row.winner_uid, row.loser_uid
        date = row.date
        is_eval = date.year in EVAL_YEARS and row.level_file == "tour"

        sv, sl = get(w), get(l)
        hm.inflate(sv, date)
        hm.inflate(sl, date)

        if ev is None:
            continue
        used += 1
        y, n_trials = ev

        if is_eval:
            d = sv.m_a - sl.m_a
            p_w = hm.sigmoid(d)
            eval_rows.append({"date": date, "tour": row.tour, "winner_uid": w, "loser_uid": l, "p_winner": p_w})

        new_cw, new_vw, new_cl, new_vl = hm.adf_update(sv.m_a, sv.v_a, sl.m_a, sl.v_a, y, n_trials)
        dshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cw - sv.m_a))
        lshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cl - sl.m_a))
        sv.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, sv.m_a + dshift))
        sl.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, sl.m_a + lshift))
        redw = max(0.0, 1 - new_vw / sv.v_a) if sv.v_a > 0 else 0.0
        redl = max(0.0, 1 - new_vl / sl.v_a) if sl.v_a > 0 else 0.0
        sv.v_a = min(hm.MAX_VAR, sv.v_a * (1 - redw))
        sl.v_a = min(hm.MAX_VAR, sl.v_a * (1 - redl))
        sv.n_matches += 1
        sl.n_matches += 1

        if (i + 1) % 300_000 == 0:
            log(f"    [{mode}] ...{i+1:,}/{n:,}  (used {used:,})")

    log(f"  [{mode}] usable rows: {used:,}/{n:,} ({100*used/n:.1f}%)")
    return pd.DataFrame(eval_rows)


def score(eval_df: pd.DataFrame) -> None:
    p = np.clip(eval_df["p_winner"].values, 1e-9, 1 - 1e-9)
    ll = float(-np.log(p).mean())
    log(f"    ALL: n={len(eval_df):,}  log-loss={ll:.5f}")
    for tour in ("atp", "wta"):
        sub = eval_df[eval_df["tour"] == tour]
        if len(sub) == 0:
            continue
        pt = np.clip(sub["p_winner"].values, 1e-9, 1 - 1e-9)
        log(f"    {tour}: n={len(sub):,}  log-loss={-np.log(pt).mean():.5f}")


def main() -> None:
    log("loading prepared archive...")
    df, dropped = hm.load_prepared(use_cache=True)
    log(f"  {len(df):,} rows total")

    for mode in ("point", "game", "set", "match"):
        log(f"\n=== fitting {mode}-level hier (single combined-strength scalar) ===")
        eval_df = run_mode(mode, df)
        score(eval_df)
        eval_df.to_parquet(f"{DD_TUNE}/hier_{mode}_level_eval_predictions.parquet")


if __name__ == "__main__":
    main()
