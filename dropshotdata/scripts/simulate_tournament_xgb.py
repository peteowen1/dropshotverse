"""Ticket 29 step 2 of Pete's "1 then 2 then 3": swap the tournament
simulator's match predictor from Elo (v1) or raw hier state (step 1, which
regressed vs Elo) to the real blended XGBoost model (ticket 28's already-
validated near-market predictor) - now that the harness itself (bracket
construction, stage-reaching log-loss, ground truth) is proven by v1.

Frozen-rating style only (no in-sim updates - XGBoost isn't a sequential
Bayesian model, there's no natural "update between rounds" for it the way
there was for the hierarchical model in step 1). Reuses xgb_bt_rating.py's
model training and per-player feature snapshot machinery wholesale - this
script only adds the round-by-round bracket simulation on top.

Key difference from xgb_bt_rating.py's round-robin: that script fixed round
at a neutral "R32" for a single scalar BT rating. A bracket sim needs the
REAL round for each match (the categorical the model was trained on), so
this precomputes one full round-robin prediction matrix PER ROUND (7 rounds
x ~16k pairs per tournament) - frozen ratings mean the underlying features
don't change bracket-to-bracket, only who plays whom does, so this is exact
and reused across all n_sims simulated brackets.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from simulate_tournament import DD, STAGE_ORDER, build_random_bracket, load_tournament_field
import simulate_tournament_updating as stu
from xgb_bt_rating import TOURNAMENTS, build_h2h_dict, build_pairwise_features_base, build_player_table, \
    load_age_table, log, prep_X_d, train_bagged_model

SCRATCH = DD / "tune_scratch"


def antisymmetrise(M: np.ndarray) -> np.ndarray:
    """Force p_ij + p_ji == 1 by averaging the two orientations on the logit
    scale. Diagonal is left as-is (NaN in a round-robin matrix, never read).

    dropshotverse#7: the model predicts each ordered pair independently, so
    the two directions disagreed - by 0.035 on average, up to 0.54, and with
    a systematic -0.024 lean against whoever sat in slot A. This is the
    invariant the problem has and the model does not."""
    L = np.log(M / (1 - M))
    return 1.0 / (1.0 + np.exp(-(L - L.T) / 2.0))


def build_round_pred_matrices(field_uids: list[str], ptab: pd.DataFrame, tour: str, surface: str, best_of: int,
                              tourney_level: str, h2h: dict, date0, cat_levels: dict,
                              models: list, rounds: list[str]) -> dict[str, np.ndarray]:
    """One full round-robin prediction pass per round value. Features other
    than `round` are identical across rounds (frozen pre-tournament state),
    so the shared base feature matrix is built once and only the `round`
    categorical changes per pass.

    Predictions are ANTISYMMETRISED (dropshotverse#7). The model is fed a
    vector built from an ordered pair, so swapping the two players gives a
    different vector rather than a mirrored one and nothing ties the two
    predictions together: p(A beats B) + p(B beats A) did not sum to 1, by
    0.035 on average and up to 0.54, and for 836 pairs BOTH orientations
    named the same player favourite. The lean is systematic, not orientation
    noise - mean(p_ij + p_ji - 1) = -0.024, i.e. occupying slot A cost about
    2.4 points regardless of who you were.

    Enforcing it costs nothing: the matrix already holds both directions, so
    this is a transform on predictions already made, not a model change or a
    retrain. xgb_bt_rating.py and build_blog_data.py already apply the same
    (L - L.T)/2 before taking BT row means; it was never applied at source.
    """
    n = len(field_uids)
    idx = {u: k for k, u in enumerate(field_uids)}
    base, pairs_i_uid, pairs_j_uid = build_pairwise_features_base(field_uids, ptab, tour, surface, best_of,
                                                                   tourney_level, h2h, date0)
    pairs_i = np.array([idx[u] for u in pairs_i_uid])
    pairs_j = np.array([idx[u] for u in pairs_j_uid])

    matrices: dict[str, np.ndarray] = {}
    # W3 (D-26/D-27): weeks_since_last was computed once from date0, so every
    # round was fed the PRE-tournament layoff gap. From the second round on
    # both players have just played, so the true difference is ~0. Behind a
    # flag until it is measured and Pete decides when to flip it (it moves
    # published odds, most at the start of a draw). The debutant case (52.0
    # default, D-27) is left as-is in round one and zeroed with everyone else
    # from round two.
    w3_fix = os.environ.get("DSV_W3_LAYOFF_FIX", "0") == "1"
    for k_rnd, rnd in enumerate(rounds):
        Xdf = base.copy()
        Xdf["round"] = rnd
        if w3_fix and k_rnd > 0:
            Xdf["weeks_since_last_diff"] = 0.0
        X = prep_X_d(Xdf, cat_levels)
        dmat = xgb.DMatrix(X, enable_categorical=True)
        p = np.clip(np.mean([m.predict(dmat) for m in models], axis=0), 1e-9, 1 - 1e-9)
        M = np.full((n, n), np.nan)
        M[pairs_i, pairs_j] = p
        matrices[rnd] = antisymmetrise(M)
    return matrices


def simulate_frozen_xgb(bracket: list[str], idx: dict[str, int], round_matrices: dict[str, np.ndarray],
                        rng: np.random.Generator) -> dict[str, str]:
    stage_reached = {uid: "R128" for uid in bracket}
    current = list(bracket)
    round_idx = 0
    while len(current) > 1:
        rnd = STAGE_ORDER[round_idx]
        M = round_matrices[rnd]
        nxt = []
        for k in range(0, len(current), 2):
            a, b = current[k], current[k + 1]
            p_a = M[idx[a], idx[b]]
            winner = a if rng.random() < p_a else b
            nxt.append(winner)
        round_idx += 1
        next_stage = STAGE_ORDER[round_idx] if round_idx < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


def main() -> None:
    n_sims = 2000
    models, cat_levels, playstyle_full = train_bagged_model()

    log("\nloading tournament fields + hier snapshots...")
    tournaments_meta = {}
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        tournaments_meta[key] = load_tournament_field(tour, tourney_id, year)
    snapshots = stu.build_snapshots(tournaments_meta)

    all_field_uids = set()
    for meta in tournaments_meta.values():
        all_field_uids.update(meta["field_uids"])
    log(f"loading age table ({len(all_field_uids)} pool players)...")
    age_table = load_age_table()
    log("building h2h dict...")
    h2h = build_h2h_dict(all_field_uids)

    rounds_needed = STAGE_ORDER[:-1]   # R128..F - the rounds a match is actually played in

    all_y: dict[str, list[int]] = {s: [] for s in STAGE_ORDER[1:]}
    all_p: dict[str, list[float]] = {s: [] for s in STAGE_ORDER[1:]}
    top_seed_ranks: list[tuple] = []
    rows_out = []   # per-tournament/per-stage y,p - so a later CI script can block-bootstrap by tournament

    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        meta = tournaments_meta[key]
        field_uids = meta["field_uids"]
        field = meta["field"]
        surface = meta["surface"]
        date0 = meta["date0"]
        real_stage = meta["real_stage"]
        top_seed_uid = meta["top_seed_uid"]
        snap = snapshots.get(key, {})
        best_of = 5 if tour == "atp" else 3
        idx = {u: k for k, u in enumerate(field_uids)}

        log(f"  [{key}] building player table + {len(rounds_needed)} round-robin passes...")
        ptab = build_player_table(field_uids, tour, date0, surface, snap, playstyle_full, age_table)
        round_matrices = build_round_pred_matrices(field_uids, ptab, tour, surface, best_of, "G", h2h, date0,
                                                    cat_levels, models, rounds_needed)

        seed = hash(key) % 100000
        rng = np.random.default_rng(seed)
        stage_counts = {uid: {s: 0 for s in STAGE_ORDER} for uid in field_uids}
        for _ in range(n_sims):
            bracket = build_random_bracket(field, rng)
            result = simulate_frozen_xgb(bracket, idx, round_matrices, rng)
            for uid, stage in result.items():
                reached_idx = STAGE_ORDER.index(stage)
                for s_idx in range(reached_idx + 1):
                    stage_counts[uid][STAGE_ORDER[s_idx]] += 1
        probs = {uid: {s: stage_counts[uid][s] / n_sims for s in STAGE_ORDER} for uid in field_uids}

        if top_seed_uid:
            p_title = {uid: probs[uid]["W"] for uid in field_uids}
            ranked = sorted(p_title.items(), key=lambda kv: -kv[1])
            rank = [u for u, _ in ranked].index(top_seed_uid) + 1
            top_seed_ranks.append((tour, tourney_id, rank, len(field_uids)))

        for s_idx, stage in enumerate(STAGE_ORDER[1:], start=1):
            y = [1 if STAGE_ORDER.index(real_stage[u]) >= s_idx else 0 for u in field_uids]
            p = [probs[u][stage] for u in field_uids]
            all_y[stage].extend(y)
            all_p[stage].extend(p)
            for u, yi, pi in zip(field_uids, y, p):
                rows_out.append({"arm": "xgb", "tour": tour, "tourney_id": tourney_id, "uid": u,
                                "stage": stage, "y": yi, "p": pi})

    log("\n=== anchor: top seed's simulated title-probability rank ===")
    n_top3 = 0
    for t, tid, rank, nf in top_seed_ranks:
        flag = "OK" if rank <= 3 else "check"
        if rank <= 3:
            n_top3 += 1
        log(f"  {t} {tid}: rank #{rank} of {nf}  [{flag}]")
    log(f"  top seed in top-3: {n_top3}/{len(top_seed_ranks)}")

    log(f"\n=== POOLED stage-reaching log-loss: XGBoost-frozen match predictor ===")
    for stage in STAGE_ORDER[1:]:
        y = np.array(all_y[stage])
        p = np.clip(np.array(all_p[stage]), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        base_p = np.clip(np.full_like(p, y.mean()), 1e-9, 1 - 1e-9)
        base_ll = float(-np.mean(y * np.log(base_p) + (1 - y) * np.log(1 - base_p)))
        log(f"  >= {stage:4s}  n={len(y):4d}  base_rate={y.mean():.3f}  "
            f"model_logloss={ll:.4f}  flat_logloss={base_ll:.4f}  "
            f"{'model wins' if ll < base_ll else 'FLAT WINS'}  (delta={ll - base_ll:+.4f})")

    out_path = SCRATCH / "simulate_tournament_yp_xgb.parquet"
    pd.DataFrame(rows_out).to_parquet(out_path)
    log(f"\nsaved {len(rows_out):,} per-tournament/per-stage rows -> {out_path}")


if __name__ == "__main__":
    main()
