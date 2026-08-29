"""Pete: "using set and match level ELO in XGB model is exciting could help
us beat market more." Match-level Elo is already 3 features in the model
(elo_overall_diff, elo_surface_diff, elo_logit_pwinner) - the genuinely NEW
piece is set-level Elo, today's standout result (0.60472 standalone,
beating even match-level Elo's own 0.61200).

Mirrors elo_raw_ratings_for_xgb.py's structure exactly (same pattern:
overall Elo rating using the frozen K-factor mechanism, this time updated on
per-match SETS-WON fraction instead of match win/loss) - produces one row
per archive match with pre-match set-Elo ratings for both players, ready to
merge into train_best_model.py's feature pipeline the same way the existing
Elo/hier features already are.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import hierarchical_model as hm
from parse_score_sets_permissive import parse_score_sets_permissive

ARCHIVE = Path(r"C:\dev\dropshotverse\dropshotdata\raw\github")
K_BASE, K_SHAPE, K_MIN = 250.0, 0.4, 16.0
START = 1500.0
SCALE = 400.0


def log(m: str) -> None:
    print(m, flush=True)


def k_factor(n: int) -> float:
    return max(K_MIN, K_BASE / ((n + 5) ** K_SHAPE))


def logistic(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / SCALE))


def main() -> None:
    log("loading prepared archive (reusing hierarchical_model's load_prepared - same"
        " has_points/total_games_w/l/score columns and round-order fix)...")
    df, dropped = hm.load_prepared(use_cache=True)
    log(f"  {len(df):,} rows")

    rating: dict[str, float] = {}
    n_matches: dict[str, int] = {}
    n = len(df)
    out_ro_w = np.empty(n)
    out_ro_l = np.empty(n)
    out_p = np.empty(n)
    used = np.zeros(n, dtype=bool)

    for i, row in enumerate(df.itertuples(index=False)):
        w, l = row.winner_uid, row.loser_uid
        rw, rl = rating.get(w, START), rating.get(l, START)
        out_ro_w[i], out_ro_l[i] = rw, rl
        p_w = logistic(rw - rl)
        out_p[i] = p_w

        sets = parse_score_sets_permissive(row.score)
        if not sets:
            continue
        sets_w = sum(1 for gw, gl in sets if gw > gl)
        tot = len(sets)
        if tot <= 0:
            continue
        used[i] = True
        frac_w = sets_w / tot
        surprise = frac_w - p_w
        kw, kl = k_factor(n_matches.get(w, 0)), k_factor(n_matches.get(l, 0))
        rating[w] = rw + kw * surprise
        rating[l] = rl - kl * surprise
        n_matches[w] = n_matches.get(w, 0) + 1
        n_matches[l] = n_matches.get(l, 0) + 1

        if (i + 1) % 300_000 == 0:
            log(f"  ...{i+1:,}/{n:,}")

    log(f"usable (parseable score) rows: {used.sum():,}/{n:,} ({100*used.mean():.1f}%)")

    out = pd.DataFrame({
        "date": df["date"].values, "tour": df["tour"].values, "level_file": df["level_file"].values,
        "winner_uid": df["winner_uid"].values, "loser_uid": df["loser_uid"].values,
        "set_elo_winner": out_ro_w, "set_elo_loser": out_ro_l, "p_winner_set_elo": out_p,
    })
    out_path = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\set_elo_raw_ratings_full.parquet"
    out.to_parquet(out_path)
    log(f"wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
