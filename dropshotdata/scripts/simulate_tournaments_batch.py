"""Ticket 29: run the frozen-Elo draw simulation across every available
Grand Slam instance (2024-2025, both tours - 16 tournaments) and pool the
results, rather than drawing any conclusion from a single tournament's
n=128. Reuses simulate_tournament.run_one_tournament() unchanged."""
from __future__ import annotations

import numpy as np

from simulate_tournament import STAGE_ORDER, run_one_tournament

TOURNAMENTS = [
    (tour, f"{year}-{code}", year)
    for tour in ("atp", "wta")
    for year in (2024, 2025)
    for code in ("580", "520", "540", "560")  # Australian Open, Roland Garros, Wimbledon, US Open
]


def main() -> None:
    log = lambda m: print(m, flush=True)
    log(f"running {len(TOURNAMENTS)} tournaments, 2000 sims each...")

    all_y: dict[str, list[int]] = {s: [] for s in STAGE_ORDER[1:]}
    all_p: dict[str, list[float]] = {s: [] for s in STAGE_ORDER[1:]}
    per_tournament_top_seed_rank: list[tuple[str, str, int, int]] = []

    for tour, tourney_id, year in TOURNAMENTS:
        result = run_one_tournament(tour, tourney_id, year, n_sims=2000, seed=hash(tourney_id + tour) % 100000,
                                     log=log)
        field_uids = result["field_uids"]
        real_stage = result["real_stage"]
        probs = result["probs"]

        p_title = {uid: probs[uid]["W"] for uid in field_uids}
        ranked = sorted(p_title.items(), key=lambda kv: -kv[1])
        top_seed_uid = result["top_seed_uid"]
        if top_seed_uid:
            rank = [u for u, _ in ranked].index(top_seed_uid) + 1
            per_tournament_top_seed_rank.append((tour, tourney_id, rank, len(field_uids)))

        for s_idx, stage in enumerate(STAGE_ORDER[1:], start=1):
            y = [1 if STAGE_ORDER.index(real_stage[u]) >= s_idx else 0 for u in field_uids]
            p = [probs[u][stage] for u in field_uids]
            all_y[stage].extend(y)
            all_p[stage].extend(p)

    log("\n=== anchor: top seed's simulated title-probability rank, per tournament ===")
    for tour, tid, rank, n in per_tournament_top_seed_rank:
        flag = "OK" if rank <= 3 else "check"
        log(f"  {tour} {tid}: top seed ranked #{rank} of {n} by simulated P(title)  [{flag}]")
    n_top3 = sum(1 for *_, rank, _ in per_tournament_top_seed_rank for rank in [rank] if rank <= 3)
    log(f"  top seed in simulated top-3 by title chance: {n_top3}/{len(per_tournament_top_seed_rank)} tournaments")

    log(f"\n=== POOLED stage-reaching log-loss across all {len(TOURNAMENTS)} tournaments ===")
    for stage in STAGE_ORDER[1:]:
        y = np.array(all_y[stage])
        p = np.clip(np.array(all_p[stage]), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        base_p = np.clip(np.full_like(p, y.mean()), 1e-9, 1 - 1e-9)
        base_ll = float(-np.mean(y * np.log(base_p) + (1 - y) * np.log(1 - base_p)))
        log(f"  >= {stage:4s}  n={len(y):4d}  base_rate={y.mean():.3f}  "
            f"model_logloss={ll:.4f}  flat_logloss={base_ll:.4f}  "
            f"{'model wins' if ll < base_ll else 'FLAT WINS'}  (delta={ll - base_ll:+.4f})")


if __name__ == "__main__":
    main()
