"""Validate the standard recursive bracket-seeding construction against real
Grand Slam results. Rather than eyeballing one draw against Wikipedia, check
systematically: for every real match between two seeded players at a Slam,
does the round they actually met in respect the MINIMUM round implied by
standard seeding separation (seed 1 & 2 can only meet in F, seeds 1-4 only
from SF, 1-8 only from QF, 1-16 only from R16, etc.)?
"""
import math

import numpy as np
import pandas as pd

ROUND_ORDER = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}


def standard_seed_positions(n: int) -> list[int]:
    """The well-known recursive tournament-bracket seeding construction:
    order(1)=[1]; order(2k) built from order(k) by, for each seed s in
    order(k), emitting s then (2k+1-s). This is the real convention (e.g.
    for a 4-draw: [1,4,2,3] -> R1 pairs (1v4),(2v3), so the two top seeds
    face the two weakest seeds first and can only meet each other in the
    final; for 8: [1,8,4,5,2,7,3,6] -> seed 3 lands in seed 2's half and
    seed 4 in seed 1's half, so if the top 4 all hold, the SFs are 1v4 and
    2v3 - exactly how real majors seed their draws)."""
    if n == 1:
        return [1]
    prev = standard_seed_positions(n // 2)
    result = []
    for s in prev:
        result.append(s)
        result.append(n + 1 - s)
    return result


def min_round_index_exact_slot(seed_i: int, seed_j: int, slots: list[int]) -> int:
    """Earliest round two seeds could meet under ONE SPECIFIC exact-slot
    bracket realization (the recursive construction above). This is
    stricter than what real seeding actually guarantees: real majors only
    fix which TIER each seed's half/quarter/eighth belongs to; which
    specific seed within a tier lands in which specific slot is decided by
    an actual random draw ceremony, not a fixed formula. Kept for reference/
    sanity only - min_round_index_tier below is the real guarantee to check
    real data against."""
    pos_i = slots.index(seed_i)
    pos_j = slots.index(seed_j)
    block = 2
    while pos_i // block != pos_j // block:
        block *= 2
    return int(math.log2(block)) - 1


def min_round_index_tier(seed_i: int, seed_j: int, n: int) -> int:
    """The REAL seeding guarantee: seeds i<j can meet no earlier than the
    round where only tier_size(j) = 2**ceil(log2(j)) players remain (seed
    j's own tier - i, being equal-or-better tier, adds no extra
    restriction). Independent of which specific slot each seed randomly
    drew within its tier. n=128 -> 7 rounds, round_idx 0=R128..6=F."""
    total_rounds = int(math.log2(n))
    tier = 1 if seed_j <= 1 else 2 ** math.ceil(math.log2(seed_j))
    return total_rounds - int(math.log2(tier))


def main() -> None:
    n = 128
    slots = standard_seed_positions(n)
    print(f"seed 1 at slot {slots.index(1)}, seed 2 at slot {slots.index(2)} "
          f"(expect 0 and {n - 1}, maximally apart)")
    print(f"seed 3 at slot {slots.index(3)}, seed 4 at slot {slots.index(4)}")

    r12 = min_round_index_tier(1, 2, n)
    print(f"[tier rule] min round for seed 1 vs seed 2: {r12} (expect 6=F) -> {'OK' if r12 == 6 else 'FAIL'}")
    r13 = min_round_index_tier(1, 3, n)
    r14 = min_round_index_tier(1, 4, n)
    print(f"[tier rule] min round for seed 1 vs seed 3: {r13}, seed 1 vs seed 4: {r14} "
          f"(expect both 5=SF - either could land in seed 1's half) -> "
          f"{'OK' if r13 == 5 and r14 == 5 else 'check'}")
    r18 = min_round_index_tier(1, 8, n)
    print(f"[tier rule] min round for seed 1 vs seed 8: {r18} (expect 4=QF) -> {'OK' if r18 == 4 else 'check'}")

    # empirical check against real archive Grand Slam results, both tours
    DD = r"C:\dev\dropshotverse\dropshotdata"
    frames = []
    for tour in ("atp", "wta"):
        for y in range(2018, 2026):
            try:
                d = pd.read_csv(rf"{DD}\raw\github\{tour}\{tour}_matches_{y}.csv", low_memory=False)
            except FileNotFoundError:
                continue
            d["tour"] = tour
            frames.append(d)
    arc = pd.concat(frames, ignore_index=True)
    slams = arc[(arc["tourney_level"] == "G") & (arc["draw_size"] == 128)].copy()
    slams["winner_seed_num"] = pd.to_numeric(slams["winner_seed"], errors="coerce")
    slams["loser_seed_num"] = pd.to_numeric(slams["loser_seed"], errors="coerce")
    slams = slams.dropna(subset=["winner_seed_num", "loser_seed_num"])
    slams["round_idx"] = slams["round"].map(ROUND_ORDER)
    slams = slams.dropna(subset=["round_idx"])

    round_names = list(ROUND_ORDER.keys())

    def run_check(max_seed: int, label: str) -> None:
        checked = 0
        violations = 0
        detail = []
        for row in slams.itertuples():
            si, sj = int(row.winner_seed_num), int(row.loser_seed_num)
            if si > max_seed or sj > max_seed or si < 1 or sj < 1:
                continue
            lo, hi = min(si, sj), max(si, sj)
            min_r = min_round_index_tier(lo, hi, n)
            actual_r = int(row.round_idx)
            checked += 1
            if actual_r < min_r:
                violations += 1
                detail.append((row.tourney_name, str(row.tourney_date), si, sj, row.round, round_names[min_r]))
        print(f"\n[{label}] checked={checked:,}  violations={violations}  "
              f"rate={violations / checked:.4%}" if checked else f"\n[{label}] no seeded pairs found")
        for d in detail[:8]:
            print(f"  VIOLATION: tourney={d[0]} date={d[1]} seed{d[2]} vs seed{d[3]} "
                  f"met in {d[4]}, standard seeding requires >= {d[5]}")

    print(f"\nGrand Slam matches 2018-2025, n before filter={len(slams):,}")
    run_check(32, "seeds 1-32 (full seeded field)")
    run_check(16, "seeds 1-16 only (deterministic placement zone)")
    run_check(8, "seeds 1-8 only")
    run_check(4, "seeds 1-4 only")


if __name__ == "__main__":
    main()
