"""Pete's ask: does Elo operate at point/game/set/match level, and can we
build all 4? Answer was match-level only, always - this builds the other 3.

One unified update mechanism (classic Elo: K-factor decaying by matches
played, K*(actual-expected)), four evidence sources swapped in per mode:
  - point: fraction of SERVE POINTS won this match (needs has_points)
  - game:  fraction of GAMES won this match (needs a parseable score)
  - set:   fraction of SETS won this match (needs a parseable score)
  - match: the match's own outcome (0/1) - this IS elo_baseline.py's
    existing frozen model; included here only for a clean side-by-side
    comparison table, not retrained (reuses its own 0.6120 locked number).

Deliberately NOT evidence-weighted by the underlying trial count (unlike
hier's adf_update, a binomial likelihood over n) - each match contributes
ONE atomic K-sized update regardless of how many points/games/sets it took,
same philosophy as classic match-level Elo. Keeping this sharp isolates
"does the formulation matter" (classic Elo vs hier's Bayesian machinery)
from "does the granularity matter" - point, game and set Elo differ from
each other ONLY in which fraction feeds the same update rule.

Read-only wrt hierarchical_model.py and elo_baseline.py - imports
load_prepared()/k_factor()/p_match()/parse_score_games() rather than
reimplementing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import elo_baseline as eb
import hierarchical_model as hm

DD_TUNE = r"C:\dev\dropshotverse\dropshotdata\tune_scratch"
EVAL_YEARS = (2024, 2025)
START = 1500.0
SCALE = 400.0   # same logistic scale as match-level Elo


def log(m: str) -> None:
    print(m, flush=True)


def logistic(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / SCALE))


def evidence_for_mode(mode: str, row) -> float | None:
    """Returns the WINNER's fraction of [points/games/sets] won in this
    match (always complementary - loser's fraction is exactly 1 minus this,
    same as a classic win/loss flag), or None if this match can't supply
    that evidence (caller skips it - same "no silent guessing" discipline
    as hier's own has_points gate).

    Point-level deliberately uses OVERALL point-win-rate (winner's own serve
    points won PLUS points won returning, over all points played) rather
    than the two serve-specific percentages separately - those aren't
    complementary (both players can independently hold serve at very
    different rates), which would need two separate ratings per player (a
    serve/return split, hier's own approach) to update cleanly. A single
    combined rating - the whole point of testing classic Elo's simpler
    mechanism against hier's - needs a single complementary fraction, so
    overall point-win-rate is the correct match to that mechanism, not an
    approximation of it."""
    if mode == "point":
        if not row.has_points or row.w_svpt <= 0 or row.l_svpt <= 0:
            return None
        total_pts = row.w_svpt + row.l_svpt
        pts_won_by_winner = row.w_won + (row.l_svpt - row.l_won)   # own serve won + return points won
        return pts_won_by_winner / total_pts
    if mode == "game":
        if pd.isna(row.total_games_w) or pd.isna(row.total_games_l):
            return None
        tot = row.total_games_w + row.total_games_l
        if tot <= 0:
            return None
        return row.total_games_w / tot
    if mode == "set":
        sets = hm.parse_score_sets(row.score)
        if not sets:
            return None
        # convention (matches hier's own adf_update_full_dp usage elsewhere
        # in hierarchical_model.py): each tuple is (winner_games, loser_games)
        sets_w = sum(1 for gw, gl in sets if gw > gl)
        tot = len(sets)
        if tot <= 0:
            return None
        return sets_w / tot
    raise ValueError(mode)


def run_mode(mode: str, df: pd.DataFrame) -> pd.DataFrame:
    rating: dict[str, float] = {}
    n_matches: dict[str, int] = {}
    eval_rows = []
    n = len(df)
    used = 0
    for i, row in enumerate(df.itertuples(index=False)):
        frac_w = evidence_for_mode(mode, row)
        w, l = row.winner_uid, row.loser_uid
        is_eval = row.date.year in EVAL_YEARS and row.level_file == "tour"

        if frac_w is None:
            continue   # no prediction possible for this row under this granularity - excluded, not guessed
        used += 1
        rw, rl = rating.get(w, START), rating.get(l, START)
        exp_w = logistic(rw - rl)

        if is_eval:
            eval_rows.append({"date": row.date, "tour": row.tour, "winner_uid": w, "loser_uid": l, "p_winner": exp_w})

        # single atomic update per match, "actual" is the complementary
        # fraction (soft), not a 0/1 win flag - the one deliberate
        # generalization from classic Elo, applied identically at every
        # granularity. Exact same symmetric-surprise pattern as
        # elo_baseline.py's own match-level update (overall[w] +=
        # kw*(1-p_w), overall[l] -= kl*(1-p_w)), just with frac_w-exp_w
        # standing in for (1-p_w) - reduces to that exactly when frac_w=1.
        surprise = frac_w - exp_w
        k_w, k_l = eb.k_factor(n_matches.get(w, 0)), eb.k_factor(n_matches.get(l, 0))
        rating[w] = rw + k_w * surprise
        rating[l] = rl - k_l * surprise
        n_matches[w] = n_matches.get(w, 0) + 1
        n_matches[l] = n_matches.get(l, 0) + 1

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

    log("\n=== reference: match-level Elo (existing frozen model, not retrained) ===")
    log("  ALL/tour-level locked config (blend=0.0, decay=0.3, k_mode=sackmann): 0.61200")

    for mode in ("point", "game", "set"):
        log(f"\n=== fitting {mode}-level Elo ===")
        eval_df = run_mode(mode, df)
        score(eval_df)
        eval_df.to_parquet(f"{DD_TUNE}/elo_{mode}_level_eval_predictions.parquet")


if __name__ == "__main__":
    main()
