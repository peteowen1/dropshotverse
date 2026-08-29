"""Ticket 29 v1: frozen-rating (Elo-only) tournament draw simulation, proof
of concept on one real Grand Slam. Deliberately simple first cut - Elo
alone, no hier state, matches Sackmann's own stated approach ("surface-
weighted Elo ratings") and this project's cheap-baseline-first discipline.
Read-only wrt elo_baseline.py/hierarchical_model.py - only consumes their
already-computed prediction parquets.

Pipeline:
  1. Pull the real field (every entrant's pre-tournament overall + surface
     Elo rating, read directly from their own R128 row - this IS their
     rating entering the tournament, unaffected by anything that happens
     in later real rounds).
  2. Generate N random seeding-consistent brackets (tier-based rule from
     ticket 29's validated seeding logic - reused, not reinvented).
  3. Simulate each bracket round by round with FROZEN ratings (no in-sim
     updates - that's the next increment, not this one).
  4. Aggregate each real player's simulated P(reach each stage).
  5. Score against their REAL, realized finishing stage - log-loss + anchor
     checks.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
TOUR = "atp"
TOURNEY_ID = "2025-540"   # Wimbledon 2025
SURFACE = "Grass"
N_SIMS = 2000
SEED = 0

STAGE_ORDER = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]  # W = won the final


def standard_seed_positions(n: int) -> list[int]:
    if n == 1:
        return [1]
    prev = standard_seed_positions(n // 2)
    result = []
    for s in prev:
        result.append(s)
        result.append(n + 1 - s)
    return result


def build_random_bracket(field: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """One random seeding-consistent bracket: a list of uids in slot order.
    Seeded players go to the tier-consistent anchor slots (which specific
    seed lands in which slot within a tier is randomized - the real-world
    convention, validated against 855 real Slam matches with the same
    tier rule in tune_scratch/validate_seeding_logic.py). Unseeded players
    fill the remaining slots at random."""
    n = len(field)
    slot_seed_numbers = standard_seed_positions(n)  # slot i's REQUIRED seed-number-tier, per the fixed template

    seeded = field[field["seed"].notna()].sort_values("seed")
    seed_to_uid = dict(zip(seeded["seed"].astype(int), seeded["uid"]))
    unseeded_uids = list(field.loc[field["seed"].isna(), "uid"])
    rng.shuffle(unseeded_uids)

    # group template slot positions by which TIER (power-of-2 bracket) they
    # belong to, then randomly permute actual seed-uids WITHIN each tier
    # before assigning to that tier's anchor slots - this is what makes it
    # "random within tier, fixed across tiers" per the validated rule.
    tiers: dict[int, list[int]] = {}  # tier size -> list of slot indices needing a seed from that tier
    for slot_idx, seed_num in enumerate(slot_seed_numbers):
        tier_size = 1 if seed_num <= 1 else 2 ** math.ceil(math.log2(seed_num))
        tiers.setdefault(tier_size, []).append(slot_idx)

    bracket = [None] * n
    assigned_seed_numbers = sorted(seed_to_uid.keys())
    for tier_size in sorted(tiers.keys()):
        slot_positions = tiers[tier_size]
        prev_tier_size = tier_size // 2 if tier_size > 1 else 0
        tier_seed_numbers = [s for s in assigned_seed_numbers if prev_tier_size < s <= tier_size]
        tier_uids = [seed_to_uid[s] for s in tier_seed_numbers if s in seed_to_uid]
        rng.shuffle(tier_uids)
        avail_slots = [p for p in slot_positions]
        rng.shuffle(avail_slots)
        for uid, slot in zip(tier_uids, avail_slots):
            bracket[slot] = uid
        # any slots in this tier not filled by a real seed (fewer seeds than
        # template slots, e.g. a withdrawn seed) get unseeded players later

    for i in range(n):
        if bracket[i] is None:
            bracket[i] = unseeded_uids.pop()
    assert not unseeded_uids, f"{len(unseeded_uids)} unseeded players left unplaced"
    return bracket


def simulate_frozen(bracket: list[str], ratings: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    """Walk one bracket round by round with FROZEN ratings. Returns
    {uid: stage_reached} for every entrant (their last stage survived to,
    using STAGE_ORDER; the eventual champion gets 'W')."""
    stage_reached = {uid: "R128" for uid in bracket}
    current = list(bracket)
    round_idx = 0
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            diff = ratings[a] - ratings[b]
            p_a = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            winner = a if rng.random() < p_a else b
            nxt.append(winner)
            # both survived to at least the NEXT stage; record it for the winner
        round_idx += 1
        next_stage = STAGE_ORDER[round_idx] if round_idx < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


def main() -> None:
    log = lambda m: print(m, flush=True)

    log(f"loading real field for {TOUR} {TOURNEY_ID} (surface={SURFACE})...")
    raw = pd.read_csv(DD / "raw/github" / TOUR / f"{TOUR}_matches_2025.csv", low_memory=False)
    tmatch = raw[raw["tourney_id"] == TOURNEY_ID].copy()
    tmatch["date"] = pd.to_datetime(tmatch["tourney_date"], format="%Y%m%d")
    for side in ("winner", "loser"):
        tmatch[f"{side}_uid"] = TOUR + ":" + tmatch[f"{side}_id"].astype("Int64").astype(str)
    r128 = tmatch[tmatch["round"] == "R128"]
    log(f"  R128 matches: {len(r128)}")

    date0 = tmatch["date"].iloc[0]
    # real 128-name field, from the raw CSV's own R128 round (not restricted
    # to rows that also parsed cleanly in elo_baseline.py's prepare() - a
    # walkover/retirement row is dropped there but the player still played,
    # or received a bye, and needs a rating).
    seed_map: dict[str, float] = {}
    field_uids: list[str] = []
    for row in r128.itertuples():
        for uid, seed_val in ((row.winner_uid, row.winner_seed), (row.loser_uid, row.loser_seed)):
            field_uids.append(uid)
            if pd.notna(seed_val):
                seed_map[uid] = seed_val
    field_uids = sorted(set(field_uids))
    log(f"  real field from raw R128 round: {len(field_uids)} (expect 128)")

    # rating lookup: each field member's MOST RECENT real rating strictly
    # before this tournament - any tournament, not just this one's R128 row
    # - robust to walkover/retirement rows elo_baseline.py's own prepare()
    # drops (the player still has a rating from whatever they played before).
    elo = pd.read_parquet(DD / "tune_scratch/elo_raw_ratings_full.parquet")
    elo = elo[(elo["tour"] == TOUR) & (elo["level_file"] == "tour") & (elo["date"] < date0)]
    long_w = elo[["date", "winner_uid", "ro_winner", "rs_winner"]].rename(
        columns={"winner_uid": "uid", "ro_winner": "ro", "rs_winner": "rs"})
    long_l = elo[["date", "loser_uid", "ro_loser", "rs_loser"]].rename(
        columns={"loser_uid": "uid", "ro_loser": "ro", "rs_loser": "rs"})
    long_all = pd.concat([long_w, long_l], ignore_index=True)
    long_all = long_all[long_all["uid"].isin(field_uids)].sort_values("date")
    latest = long_all.drop_duplicates(subset="uid", keep="last").set_index("uid")

    ratings: dict[str, float] = {}
    surf_ratings: dict[str, float] = {}
    for uid in field_uids:
        if uid in latest.index:
            ratings[uid] = float(latest.loc[uid, "ro"])
            surf_ratings[uid] = float(latest.loc[uid, "rs"])
        else:
            ratings[uid] = 1500.0  # never played tour-level before - Elo's own START prior
            surf_ratings[uid] = 1500.0

    missing_history = sum(1 for u in field_uids if ratings[u] == 1500.0)
    log(f"  ratings resolved for all {len(ratings)} field members "
        f"({missing_history} with no prior tour-level history, defaulted to START=1500)")

    field = pd.DataFrame({"uid": field_uids, "seed": [seed_map.get(u, np.nan) for u in field_uids]})
    log(f"  field size: {len(field)}  (seeded: {field['seed'].notna().sum()})")

    # anchor: top seed should have a real, sane rating, not missing/degenerate
    top_seed_uid = field.loc[field["seed"] == field["seed"].min(), "uid"].iloc[0]
    log(f"  top seed uid={top_seed_uid}  rating={ratings[top_seed_uid]:.1f}  "
        f"(sanity: should be well above 2000 for a Slam #1 seed)")

    # real, realized finishing stage per player (ground truth). "Reached"
    # means PLAYED that round (win or lose) - the deepest round a player
    # appears in at all, not "won their way into the next one" (which
    # silently drops anyone who reached a round and then lost it).
    real_depth: dict[str, int] = {uid: 0 for uid in field_uids}  # 0 = R128 (everyone reaches this)
    champion_uid: str | None = None
    for row in tmatch.itertuples():
        r = row.round
        if r not in STAGE_ORDER:
            continue
        idx = STAGE_ORDER.index(r)
        for uid in (row.winner_uid, row.loser_uid):
            if uid in real_depth:
                real_depth[uid] = max(real_depth[uid], idx)
        if r == "F":
            champion_uid = row.winner_uid
    real_stage = {uid: STAGE_ORDER[d] for uid, d in real_depth.items()}
    if champion_uid in real_stage:
        real_stage[champion_uid] = "W"

    log(f"\nsimulating {N_SIMS} frozen-rating brackets...")
    rng = np.random.default_rng(SEED)
    stage_counts = {uid: {s: 0 for s in STAGE_ORDER} for uid in field_uids}
    for sim in range(N_SIMS):
        bracket = build_random_bracket(field, rng)
        result = simulate_frozen(bracket, ratings, rng)
        for uid, stage in result.items():
            reached_idx = STAGE_ORDER.index(stage)
            for s_idx in range(reached_idx + 1):
                stage_counts[uid][STAGE_ORDER[s_idx]] += 1   # "reached at least this stage"

    probs = {uid: {s: stage_counts[uid][s] / N_SIMS for s in STAGE_ORDER} for uid in field_uids}

    # anchor: top seed's P(title) should be among the highest in the field
    p_title = {uid: probs[uid]["W"] for uid in field_uids}
    ranked = sorted(p_title.items(), key=lambda kv: -kv[1])
    top5 = ranked[:5]
    log(f"\ntop 5 by simulated P(title): {[(u, f'{p:.3f}') for u, p in top5]}")
    log(f"top seed's own P(title) rank: {[u for u, _ in ranked].index(top_seed_uid) + 1} of {len(ranked)}")

    # scoring: stage-reaching log-loss (nested "reached at least stage S" binary questions)
    log("\n=== stage-reaching log-loss (binary: did they reach >= this stage) ===")
    log(f"{'':10s}{'model':>10s}{'flat-base-rate':>16s}")
    for s_idx, stage in enumerate(STAGE_ORDER[1:], start=1):  # skip R128 - everyone "reaches" it trivially
        y = np.array([1 if STAGE_ORDER.index(real_stage[u]) >= s_idx else 0 for u in field_uids])
        p = np.clip(np.array([probs[u][stage] for u in field_uids]), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        # trivial baseline: everyone gets the same probability (the true
        # base rate itself, the best a "no skill info" model could do) -
        # the model should beat this convincingly if ratings carry real signal
        base_p = np.clip(np.full_like(p, y.mean()), 1e-9, 1 - 1e-9)
        base_ll = float(-np.mean(y * np.log(base_p) + (1 - y) * np.log(1 - base_p)))
        log(f"  >= {stage:4s}  n={len(y)}  base_rate={y.mean():.3f}  "
            f"model_logloss={ll:.4f}  flat_logloss={base_ll:.4f}  "
            f"{'model wins' if ll < base_ll else 'FLAT WINS - check model'}")


if __name__ == "__main__":
    main()
