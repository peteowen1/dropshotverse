"""Same as set_elo_raw_for_xgb.py but at GAME granularity (games-won
fraction as evidence) - completes the match/set/game feature-toggle test
Pete asked for. Mirrors set_elo_raw_for_xgb.py's structure exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd

import hierarchical_model as hm

START = 1500.0
SCALE = 400.0


def log(m: str) -> None:
    print(m, flush=True)


def k_factor(n: int) -> float:
    return max(16.0, 250.0 / ((n + 5) ** 0.4))


def logistic(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / SCALE))


def main() -> None:
    log("loading prepared archive...")
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

        if pd.isna(row.total_games_w) or pd.isna(row.total_games_l):
            continue
        tot = row.total_games_w + row.total_games_l
        if tot <= 0:
            continue
        used[i] = True
        frac_w = row.total_games_w / tot
        surprise = frac_w - p_w
        kw, kl = k_factor(n_matches.get(w, 0)), k_factor(n_matches.get(l, 0))
        rating[w] = rw + kw * surprise
        rating[l] = rl - kl * surprise
        n_matches[w] = n_matches.get(w, 0) + 1
        n_matches[l] = n_matches.get(l, 0) + 1

        if (i + 1) % 300_000 == 0:
            log(f"  ...{i+1:,}/{n:,}")

    log(f"usable rows: {used.sum():,}/{n:,} ({100*used.mean():.1f}%)")
    out = pd.DataFrame({
        "date": df["date"].values, "tour": df["tour"].values, "level_file": df["level_file"].values,
        "winner_uid": df["winner_uid"].values, "loser_uid": df["loser_uid"].values,
        "game_elo_winner": out_ro_w, "game_elo_loser": out_ro_l, "p_winner_game_elo": out_p,
    })
    out_path = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\game_elo_raw_ratings_full.parquet"
    out.to_parquet(out_path)
    log(f"wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
