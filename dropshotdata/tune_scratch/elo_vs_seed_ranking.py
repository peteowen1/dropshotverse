"""Side question from Pete while ticket 29 step 1 was running in the
background: the "top seed" anchor check in simulate_tournament.py compares
against the official SEED (committee-assigned), not Elo. Does Elo actually
rank players better than seed number does, against real tournament outcomes?

Cheap screen (stats-discipline #9): Spearman rank correlation of each
predictor against real finishing-stage depth, pooled across the same 16
Slams ticket 29 already validated on. Not a full simulation - reuses
load_tournament_field's real_stage ground truth and the same Elo-lookup
logic run_one_tournament() already uses, just correlated directly instead of
run through a bracket simulator.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from simulate_tournament import DD, STAGE_ORDER, load_tournament_field  # noqa: E402

TOURNAMENTS = [
    (tour, f"{year}-{code}", year)
    for tour in ("atp", "wta")
    for year in (2024, 2025)
    for code in ("580", "520", "540", "560")
]


def main() -> None:
    rows = []
    for tour, tourney_id, year in TOURNAMENTS:
        tf = load_tournament_field(tour, tourney_id, year)
        field_uids = tf["field_uids"]
        seed_map = tf["seed_map"]
        real_stage = tf["real_stage"]
        date0 = tf["date0"]

        elo = pd.read_parquet(DD / "tune_scratch/elo_raw_ratings_full.parquet")
        elo = elo[(elo["tour"] == tour) & (elo["level_file"] == "tour") & (elo["date"] < date0)]
        long_w = elo[["date", "winner_uid", "ro_winner"]].rename(columns={"winner_uid": "uid", "ro_winner": "ro"})
        long_l = elo[["date", "loser_uid", "ro_loser"]].rename(columns={"loser_uid": "uid", "ro_loser": "ro"})
        long_all = pd.concat([long_w, long_l], ignore_index=True)
        long_all = long_all[long_all["uid"].isin(field_uids)].sort_values("date")
        latest = long_all.drop_duplicates(subset="uid", keep="last").set_index("uid")

        for uid in field_uids:
            seed = seed_map.get(uid, np.nan)
            elo_rating = float(latest.loc[uid, "ro"]) if uid in latest.index else 1500.0
            depth = STAGE_ORDER.index(real_stage[uid])
            rows.append({"tour": tour, "tourney_id": tourney_id, "uid": uid,
                        "seed": seed, "elo": elo_rating, "depth": depth})

    df = pd.DataFrame(rows)
    print(f"n={len(df)} rows, {df['tourney_id'].nunique()} tournaments, "
          f"seeded={df['seed'].notna().sum()}, unseeded={df['seed'].isna().sum()}")

    # unseeded players get the worst possible rank (below every real seed
    # number, which tops out at 32 for these draws)
    df["seed_rank"] = df["seed"].fillna(999)

    def spearman(x: np.ndarray, y: np.ndarray) -> float:
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])

    # lower seed_rank = better = should correlate with HIGHER depth, so
    # correlate -seed_rank with depth to keep both signs "higher = better"
    rho_seed = spearman(-df["seed_rank"].to_numpy(), df["depth"].to_numpy())
    rho_elo = spearman(df["elo"].to_numpy(), df["depth"].to_numpy())
    print(f"\nSpearman(seed rank inverted, real depth) = {rho_seed:.4f}")
    print(f"Spearman(elo rating,          real depth) = {rho_elo:.4f}")
    print(f"delta (elo - seed) = {rho_elo - rho_seed:+.4f}")

    # block-bootstrap CI on the delta, resampling whole TOURNAMENTS (not rows)
    # to respect within-tournament correlation - same discipline as ticket 29's
    # pooled-across-tournaments log-loss, not a single-tournament noise trap.
    rng = np.random.default_rng(0)
    tids = df["tourney_id"].unique()
    deltas = []
    for _ in range(2000):
        sample_tids = rng.choice(tids, size=len(tids), replace=True)
        parts = [df[df["tourney_id"] == t] for t in sample_tids]
        boot = pd.concat(parts, ignore_index=True)
        r_s = spearman(-boot["seed_rank"].to_numpy(), boot["depth"].to_numpy())
        r_e = spearman(boot["elo"].to_numpy(), boot["depth"].to_numpy())
        deltas.append(r_e - r_s)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"\n2000x tournament-block bootstrap CI on delta: [{lo:+.4f}, {hi:+.4f}]")
    verdict = "Elo beats seeding" if lo > 0 else ("seeding beats Elo" if hi < 0 else "unresolved (CI spans zero)")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
