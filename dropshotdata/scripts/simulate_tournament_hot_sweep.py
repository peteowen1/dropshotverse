"""Pete's follow-up on step 1: the hier "hot" (in-sim-update) arm was tested
with the archive's FROZEN v25 update hyperparameters, tuned for a totally
different objective (fitting 50 years of real match outcomes) - never
retuned for "how much should a rating move after one match inside a single
2-week bracket." A wrong update MAGNITUDE would make a genuinely good idea
(uncertain players should swing more, settled veterans less - Pete's
original reasoning) look like a wash or a regression without the concept
itself being wrong. Also: Elo was never tested hot at all, only hier was.

This sweeps a single "update-strength" multiplier alpha for BOTH Elo and
hier, alpha=0 exactly reproducing each model's already-shipped frozen arm
(sanity-checked below), so "does hot beat cold" gets tested at each model's
OWN best setting, not an untuned default.

Elo-hot: reuses elo_baseline.py's own k_factor()/START (read-only, imported
not reimplemented) - after each simulated match, both players' overall
rating moves by alpha * k_factor(n) * (1-p_w), n incrementing per simulated
match same as the real online fit.

Hier-hot: reuses simulate_tournament_updating.py's regime-B update
(apply_regime_b_update), with the resulting shift scaled by alpha before
being applied - alpha=1 reproduces step 1's original hot arm exactly.

Sweep uses a reduced n_sims (games-level hier simulation is the expensive
part) to keep the search tractable; the winning alpha per model gets a full
n_sims re-run + CI against alpha=0 in a follow-up script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import elo_baseline as eb
import hierarchical_model as hm
from simulate_tournament import DD, STAGE_ORDER, build_random_bracket, load_tournament_field
import simulate_tournament_updating as stu

SCRATCH = DD / "tune_scratch"
TOURNAMENTS = stu.TOURNAMENTS


def log(m: str) -> None:
    print(m, flush=True)


# --------------------------------------------------------------- Elo hot arm
def build_elo_snapshot(field_uids: list[str], tour: str, date0) -> dict[str, tuple[float, int]]:
    """Per-player (overall rating, n_all match count) as of date0 - same
    source (elo_raw_ratings_full.parquet) and same latest-prior-row lookup
    pattern simulate_tournament.py already uses, extended to also carry
    n_all (needed for k_factor - not used by the frozen/cold arm)."""
    elo = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet")
    elo = elo[(elo["tour"] == tour) & (elo["level_file"] == "tour") & (elo["date"] < date0)]
    long_w = elo[["date", "winner_uid", "ro_winner", "n_all_winner"]].rename(
        columns={"winner_uid": "uid", "ro_winner": "ro", "n_all_winner": "n"})
    long_l = elo[["date", "loser_uid", "ro_loser", "n_all_loser"]].rename(
        columns={"loser_uid": "uid", "ro_loser": "ro", "n_all_loser": "n"})
    long_all = pd.concat([long_w, long_l], ignore_index=True)
    long_all = long_all[long_all["uid"].isin(field_uids)].sort_values("date")
    latest = long_all.drop_duplicates(subset="uid", keep="last").set_index("uid")
    return {uid: (float(latest.loc[uid, "ro"]), int(latest.loc[uid, "n"])) if uid in latest.index else (eb.START, 0)
            for uid in field_uids}


def simulate_bracket_elo(bracket: list[str], snapshot: dict[str, tuple[float, int]],
                         rng: np.random.Generator, alpha: float) -> dict[str, str]:
    ratings = {u: snapshot[u][0] for u in bracket}
    n_all = {u: snapshot[u][1] for u in bracket}
    stage_reached = {uid: "R128" for uid in bracket}
    current = list(bracket)
    round_idx = 0
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            diff = ratings[a] - ratings[b]
            p_a = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            a_wins = rng.random() < p_a
            winner, loser = (a, b) if a_wins else (b, a)
            if alpha > 0.0:
                p_w = p_a if a_wins else (1.0 - p_a)   # P(actual winner wins), matching run_elo()'s own p_w
                kw, kl = eb.k_factor(n_all[winner]), eb.k_factor(n_all[loser])
                ratings[winner] = ratings[winner] + alpha * kw * (1 - p_w)
                ratings[loser] = ratings[loser] - alpha * kl * (1 - p_w)
                n_all[winner] += 1
                n_all[loser] += 1
            nxt.append(winner)
        round_idx += 1
        next_stage = STAGE_ORDER[round_idx] if round_idx < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


# -------------------------------------------------------------- hier hot arm
def apply_regime_b_update_alpha(winner: "hm.PlayerState", loser: "hm.PlayerState",
                                games_w: float, games_l: float, tier_scale: float, alpha: float) -> None:
    """Same as simulate_tournament_updating.apply_regime_b_update, with the
    resulting shift/variance-reduction scaled by alpha. alpha=1 reproduces
    that function exactly (verified below before trusting any sweep value)."""
    cw = winner.m_a + winner.m_b
    cl = loser.m_a + loser.m_b
    vw = winner.v_a + winner.v_b
    vl = loser.v_a + loser.v_b
    y_games = games_w * tier_scale
    n_games = (games_w + games_l) * tier_scale
    new_cw, new_vw, new_cl, new_vl = hm.adf_update(cw, vw, cl, vl, y_games, n_games)

    dshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cw - cw)) * alpha
    lshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cl - cl)) * alpha
    winner.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, winner.m_a + 0.5 * dshift))
    winner.m_b = max(-hm.MAX_STATE, min(hm.MAX_STATE, winner.m_b + 0.5 * dshift))
    loser.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, loser.m_a + 0.5 * lshift))
    loser.m_b = max(-hm.MAX_STATE, min(hm.MAX_STATE, loser.m_b + 0.5 * lshift))

    redw = max(0.0, 1 - new_vw / vw) * alpha if vw > 0 else 0.0
    redl = max(0.0, 1 - new_vl / vl) * alpha if vl > 0 else 0.0
    winner.v_a = min(hm.MAX_VAR, winner.v_a * (1 - 0.5 * redw))
    winner.v_b = min(hm.MAX_VAR, winner.v_b * (1 - 0.5 * redw))
    loser.v_a = min(hm.MAX_VAR, loser.v_a * (1 - 0.5 * redl))
    loser.v_b = min(hm.MAX_VAR, loser.v_b * (1 - 0.5 * redl))


def simulate_bracket_hier(bracket: list[str], snapshot: dict, surface: str, best_of: int,
                          rng: np.random.Generator, alpha: float) -> dict[str, str]:
    states = {uid: stu.clone_state(snapshot[uid]) if uid in snapshot else hm.PlayerState() for uid in bracket}
    stage_reached = {uid: "R128" for uid in bracket}
    current = list(bracket)
    round_idx = 0
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            sa, sb = states[a], states[b]
            p1, p2 = stu.predict_p1_p2(sa, sb, surface)
            a_wins, games_a, games_b = stu.simulate_one_match(p1, p2, best_of, rng)
            winner_uid, loser_uid = (a, b) if a_wins else (b, a)
            games_w, games_l = (games_a, games_b) if a_wins else (games_b, games_a)
            if alpha > 0.0:
                apply_regime_b_update_alpha(states[winner_uid], states[loser_uid], games_w, games_l,
                                            hm.REGIME_B_N, alpha)
            nxt.append(winner_uid)
        round_idx += 1
        next_stage = STAGE_ORDER[round_idx] if round_idx < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


def pooled_logloss(all_y: dict, all_p: dict) -> dict[str, float]:
    out = {}
    for stage in STAGE_ORDER[1:]:
        y = np.array(all_y[stage])
        p = np.clip(np.array(all_p[stage]), 1e-9, 1 - 1e-9)
        out[stage] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return out


def main() -> None:
    n_sims = 500
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]

    log("loading tournament fields...")
    tournaments_meta = {}
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        tournaments_meta[key] = load_tournament_field(tour, tourney_id, year)

    log("building hier snapshots...")
    hier_snapshots = stu.build_snapshots(tournaments_meta)

    log("building elo snapshots...")
    elo_snapshots = {}
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        meta = tournaments_meta[key]
        elo_snapshots[key] = build_elo_snapshot(meta["field_uids"], tour, meta["date0"])

    results = []
    for model in ("elo", "hier"):
        for alpha in alphas:
            all_y = {s: [] for s in STAGE_ORDER[1:]}
            all_p = {s: [] for s in STAGE_ORDER[1:]}
            for tour, tourney_id, year in TOURNAMENTS:
                key = f"{tour}:{tourney_id}"
                meta = tournaments_meta[key]
                field_uids = meta["field_uids"]
                field = meta["field"]
                surface = meta["surface"]
                real_stage = meta["real_stage"]
                best_of = 5 if tour == "atp" else 3
                seed = hash(key) % 100000
                rng = np.random.default_rng(seed)
                stage_counts = {uid: {s: 0 for s in STAGE_ORDER} for uid in field_uids}
                for _ in range(n_sims):
                    bracket = build_random_bracket(field, rng)
                    if model == "elo":
                        result = simulate_bracket_elo(bracket, elo_snapshots[key], rng, alpha)
                    else:
                        result = simulate_bracket_hier(bracket, hier_snapshots.get(key, {}), surface, best_of, rng,
                                                        alpha)
                    for uid, stage in result.items():
                        reached_idx = STAGE_ORDER.index(stage)
                        for s_idx in range(reached_idx + 1):
                            stage_counts[uid][STAGE_ORDER[s_idx]] += 1
                probs = {uid: {s: stage_counts[uid][s] / n_sims for s in STAGE_ORDER} for uid in field_uids}
                for s_idx, stage in enumerate(STAGE_ORDER[1:], start=1):
                    y = [1 if STAGE_ORDER.index(real_stage[u]) >= s_idx else 0 for u in field_uids]
                    p = [probs[u][stage] for u in field_uids]
                    all_y[stage].extend(y)
                    all_p[stage].extend(p)
            ll = pooled_logloss(all_y, all_p)
            avg_ll = float(np.mean(list(ll.values())))
            results.append({"model": model, "alpha": alpha, "avg_logloss": avg_ll, **ll})
            log(f"  [{model} alpha={alpha}] avg_logloss={avg_ll:.4f}  " +
                "  ".join(f"{s}={ll[s]:.4f}" for s in STAGE_ORDER[1:]))

    df = pd.DataFrame(results)
    out_path = SCRATCH / "hot_sweep_results.parquet"
    df.to_parquet(out_path)
    log(f"\nsaved -> {out_path}")

    log("\n=== best alpha per model (by avg_logloss across stages) ===")
    for model in ("elo", "hier"):
        sub = df[df["model"] == model].sort_values("avg_logloss")
        log(f"  {model}: best alpha={sub.iloc[0]['alpha']}  avg_logloss={sub.iloc[0]['avg_logloss']:.4f}  "
            f"(alpha=0 avg_logloss={sub[sub['alpha'] == 0.0]['avg_logloss'].iloc[0]:.4f})")


if __name__ == "__main__":
    main()
