"""Extends elo_vs_seed_ranking.py: how does the hierarchical model's own
combined-strength rating (m_a+m_b, pre-tournament snapshot state - the same
snapshots ticket 29 step 1's in-sim-update simulator already builds) rank
against real tournament depth, next to seed and Elo?

Reuses simulate_tournament_updating.build_snapshots() rather than
re-deriving the replay loop - same read-only-wrt-hierarchical_model.py
discipline as that script.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from simulate_tournament import DD, STAGE_ORDER, load_tournament_field  # noqa: E402
import simulate_tournament_updating as stu  # noqa: E402

TOURNAMENTS = stu.TOURNAMENTS


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    tournaments_meta = {}
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        tournaments_meta[key] = load_tournament_field(tour, tourney_id, year)

    snapshots = stu.build_snapshots(tournaments_meta)

    rows = []
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        tf = tournaments_meta[key]
        field_uids = tf["field_uids"]
        seed_map = tf["seed_map"]
        real_stage = tf["real_stage"]
        date0 = tf["date0"]
        snap = snapshots[key]

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
            state = snap.get(uid)
            hier_rating = (state.m_a + state.m_b) if state is not None else 0.0
            depth = STAGE_ORDER.index(real_stage[uid])
            rows.append({"tour": tour, "tourney_id": tourney_id, "uid": uid, "seed": seed,
                        "elo": elo_rating, "hier": hier_rating, "depth": depth})

    df = pd.DataFrame(rows)
    df["seed_rank"] = df["seed"].fillna(999)
    print(f"n={len(df)} rows, {df['tourney_id'].nunique()} tournaments")

    preds = {"seed (inverted)": -df["seed_rank"].to_numpy(),
            "elo": df["elo"].to_numpy(),
            "hier (m_a+m_b)": df["hier"].to_numpy()}
    rhos = {name: spearman(vals, df["depth"].to_numpy()) for name, vals in preds.items()}
    for name, rho in rhos.items():
        print(f"  Spearman({name:16s}, depth) = {rho:.4f}")

    rng = np.random.default_rng(0)
    tids = df["tourney_id"].unique()
    boot_rhos = {name: [] for name in preds}
    for _ in range(2000):
        sample_tids = rng.choice(tids, size=len(tids), replace=True)
        parts = [df[df["tourney_id"] == t] for t in sample_tids]
        boot = pd.concat(parts, ignore_index=True)
        for name, col in (("seed (inverted)", -boot["seed_rank"].to_numpy()),
                          ("elo", boot["elo"].to_numpy()),
                          ("hier (m_a+m_b)", boot["hier"].to_numpy())):
            boot_rhos[name].append(spearman(col, boot["depth"].to_numpy()))

    print("\npairwise deltas (row - col), with 95% tournament-block bootstrap CI:")
    names = list(preds.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = rhos[a] - rhos[b]
            boot_d = np.array(boot_rhos[a]) - np.array(boot_rhos[b])
            lo, hi = np.percentile(boot_d, [2.5, 97.5])
            verdict = f"{a} wins" if lo > 0 else (f"{b} wins" if hi < 0 else "unresolved")
            print(f"  {a} - {b}: {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
