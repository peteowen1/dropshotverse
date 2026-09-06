"""Ticket 29 step 1 of "1 then 2 then 3": the actually-novel piece Pete asked
for - does applying the hierarchical model's OWN existing online update rule
between simulated rounds (not just across real history) improve stage-
reaching log-loss over a frozen-rating baseline? This is explicitly framed as
a test, not an assumption (see issue 29's "genuine extension beyond the
published state of the art" note) - Sackmann's own Slam forecasts use frozen
ratings throughout.

Two arms, SAME pre-tournament state snapshot, SAME brackets, SAME match-level
win/loss draws (the games-level simulator is shared) - the ONLY thing that
differs between them is whether adf_update() is applied to a player's state
after each simulated round. Isolates the update mechanism as the single axis
under test (stats-discipline rule 5), rather than conflating it with a switch
away from Elo.

Match model: reuses hierarchical_model.py's own regime-B ("we don't have
point-level serve data, only a scoreline") update path, since that's exactly
what a SIMULATED match is - the games-level walk below produces a real
scoreline (games won by each side), and regime B is the model's own designed
way of turning a scoreline into a state update. No new mechanism invented,
per issue 29's explicit ask.

Read-only wrt hierarchical_model.py - only imports its functions/constants.
Frozen v25 hyperparameters (same as every other model-building script this
project ships) are set via os.environ BEFORE the import, since the module
reads them at import time.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

os.environ.setdefault("DSV_RW_VAR_PER_WEEK", "0.005")
os.environ.setdefault("DSV_SURFACE_VAR_SCALE", "1.0")
os.environ.setdefault("DSV_OU_THETA_PER_WEEK", "0.00001")
os.environ.setdefault("DSV_MAX_SHIFT", "0.125")
os.environ.setdefault("DSV_PHI_MAX", "4.0")
os.environ.setdefault("DSV_LAYOFF_C", "0.06")
os.environ.setdefault("DSV_REGIME_B_N", "0.25")
os.environ.setdefault("DSV_TIER_WEIGHT", "1.0")
os.environ.setdefault("DSV_CLIP_REINFLATE_K", "0.5")

import numpy as np
import pandas as pd

import hierarchical_model as hm
from simulate_tournament import STAGE_ORDER, build_random_bracket, load_tournament_field

TOURNAMENTS = [
    (tour, f"{year}-{code}", year)
    for tour in ("atp", "wta")
    for year in (2024, 2025)
    for code in ("580", "520", "540", "560")  # Australian Open, Roland Garros, Wimbledon, US Open
]


def clone_state(s: "hm.PlayerState") -> "hm.PlayerState":
    """Cheap manual clone - copy.deepcopy on ~2,000 PlayerState objects per
    tournament x 2000 sims would dominate runtime for no benefit; every field
    is either a float or a small dict, so a shallow dict.copy() per field is
    sufficient and exact."""
    c = hm.PlayerState()
    c.m_a, c.v_a, c.m_b, c.v_b = s.m_a, s.v_a, s.m_b, s.v_b
    c.kappa = s.kappa.copy()
    c.kappa_var = s.kappa_var.copy()
    c.lam = s.lam.copy()
    c.lam_var = s.lam_var.copy()
    c.last_seen = s.last_seen
    c.n_matches = s.n_matches
    return c


def simulate_one_set(p1: float, p2: float, a_serves_first: bool, rng: np.random.Generator) -> tuple[int, int]:
    """One set, game-by-game, matching hierarchical_model.p_set_full's own
    DP transition exactly (p_game(p1) when A serves, 1-p_game(p2) when B
    serves), just sampled instead of enumerated - so the terminal (games_a,
    games_b) distribution this produces IS p_set_full's distribution."""
    a = b = 0
    game_idx = 0
    while True:
        if (a >= 6 or b >= 6) and abs(a - b) >= 2:
            return a, b
        if a == 6 and b == 6:
            tb_p = hm.p_tiebreak(p1, p2)
            return (7, 6) if rng.random() < tb_p else (6, 7)
        a_serves_game = (game_idx % 2 == 0) == a_serves_first
        pg = hm.p_game(p1) if a_serves_game else 1 - hm.p_game(p2)
        if rng.random() < pg:
            a += 1
        else:
            b += 1
        game_idx += 1


def simulate_one_match(p1: float, p2: float, best_of: int, rng: np.random.Generator) -> tuple[bool, int, int]:
    """One self-consistent match realization: winner + total games for each
    side, both derived from the SAME game-by-game walk (no separate analytic
    win draw to keep consistent with a games count)."""
    need = 2 if best_of == 3 else 3
    sets_a = sets_b = 0
    games_a_total = games_b_total = 0
    while sets_a < need and sets_b < need:
        a_first = rng.random() < 0.5
        ga, gb = simulate_one_set(p1, p2, a_first, rng)
        games_a_total += ga
        games_b_total += gb
        if ga > gb:
            sets_a += 1
        else:
            sets_b += 1
    return sets_a == need, games_a_total, games_b_total


def predict_p1_p2(sa: "hm.PlayerState", sb: "hm.PlayerState", surface: str) -> tuple[float, float]:
    """Same formula hierarchical_model.run()'s eval branch uses: each
    player's own-serve win probability from their (mu + surface kappa/lam)
    effective state."""
    k_a, _ = sa.get_kappa(surface)
    k_b, _ = sb.get_kappa(surface)
    lm_a, _ = sa.get_lam(surface)
    lm_b, _ = sb.get_lam(surface)
    e_a = sa.m_a + k_a
    f_b = sb.m_b + lm_b
    e_b = sb.m_a + k_b
    f_a = sa.m_b + lm_a
    return hm.sigmoid(e_a - f_b), hm.sigmoid(e_b - f_a)


def apply_regime_b_update(winner: "hm.PlayerState", loser: "hm.PlayerState",
                          games_w: float, games_l: float, tier_scale: float) -> None:
    """Exact reproduction of hierarchical_model.run()'s regime-B branch
    (combined mu_a+mu_b strength, split 50/50 back to alpha/beta). games_w/
    games_l are RAW games won by each side; tier_scale is applied here, same
    as run()'s own y_games/n_games construction."""
    cw = winner.m_a + winner.m_b
    cl = loser.m_a + loser.m_b
    vw = winner.v_a + winner.v_b
    vl = loser.v_a + loser.v_b
    y_games = games_w * tier_scale
    n_games = (games_w + games_l) * tier_scale
    new_cw, new_vw, new_cl, new_vl = hm.adf_update(cw, vw, cl, vl, y_games, n_games)

    dshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cw - cw))
    lshift = max(-hm.MAX_SHIFT, min(hm.MAX_SHIFT, new_cl - cl))
    winner.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, winner.m_a + 0.5 * dshift))
    winner.m_b = max(-hm.MAX_STATE, min(hm.MAX_STATE, winner.m_b + 0.5 * dshift))
    loser.m_a = max(-hm.MAX_STATE, min(hm.MAX_STATE, loser.m_a + 0.5 * lshift))
    loser.m_b = max(-hm.MAX_STATE, min(hm.MAX_STATE, loser.m_b + 0.5 * lshift))

    redw = max(0.0, 1 - new_vw / vw) if vw > 0 else 0.0
    redl = max(0.0, 1 - new_vl / vl) if vl > 0 else 0.0
    winner.v_a = min(hm.MAX_VAR, winner.v_a * (1 - 0.5 * redw))
    winner.v_b = min(hm.MAX_VAR, winner.v_b * (1 - 0.5 * redw))
    loser.v_a = min(hm.MAX_VAR, loser.v_a * (1 - 0.5 * redl))
    loser.v_b = min(hm.MAX_VAR, loser.v_b * (1 - 0.5 * redl))


def simulate_bracket(bracket: list[str], snapshot: dict, surface: str, best_of: int,
                     rng: np.random.Generator, apply_updates: bool) -> dict[str, str]:
    # A missing entry means the player has zero prior tour-level matches in
    # the archive (a tour debut) - hm.run()'s own get() gives such a player a
    # fresh PlayerState() the same way.
    states = {uid: clone_state(snapshot[uid]) if uid in snapshot else hm.PlayerState() for uid in bracket}
    stage_reached = {uid: "R128" for uid in bracket}
    current = list(bracket)
    round_idx = 0
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            sa, sb = states[a], states[b]
            p1, p2 = predict_p1_p2(sa, sb, surface)
            a_wins, games_a, games_b = simulate_one_match(p1, p2, best_of, rng)
            winner_uid, loser_uid = (a, b) if a_wins else (b, a)
            games_w, games_l = (games_a, games_b) if a_wins else (games_b, games_a)
            if apply_updates:
                apply_regime_b_update(states[winner_uid], states[loser_uid], games_w, games_l, hm.REGIME_B_N)
            nxt.append(winner_uid)
        round_idx += 1
        next_stage = STAGE_ORDER[round_idx] if round_idx < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


def snapshot_fingerprint(df: "pd.DataFrame", date0) -> str:
    """Hash of every archive+bridge row that can influence a snapshot at
    `date0`: the rows before it, in order, on the columns the replay reads.
    ~1-2 s on 1.2M rows. A corrected TML row or a new archive vintage changes
    it; a same-day rerun does not."""
    import hashlib
    cols = ["date", "winner_uid", "loser_uid", "surface", "has_points", "w_svpt", "w_won",
            "l_svpt", "l_won", "total_games_w", "total_games_l", "tourney_level", "tour", "level_file"]
    sub = df.loc[df["date"] < date0, [c for c in cols if c in df.columns]]
    h = pd.util.hash_pandas_object(sub, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16] + f"_{len(sub)}"


def build_snapshots(tournaments_meta: dict, df: "pd.DataFrame | None" = None,
                    cache_dir: "Path | None" = None) -> dict[str, dict[str, "hm.PlayerState"]]:
    """Single pass over the archive, snapshotting each tournament's field's
    pre-tournament state the moment the replay reaches that tournament's
    start date - far cheaper than N separate truncated run() calls, and
    gives byte-identical state to what a truncated run() would produce since
    it's the exact same sequential update code.

    `df` lets a caller pass an already-prepared dataframe (e.g. the archive
    + tennismylife bridge from bridge_tennismylife.py, for a forward-looking
    forecast that needs current ratings) instead of the plain cached
    archive - backtests against real historical ground truth should NOT
    pass a bridged df (that would leak synthetic/future-sourced data into a
    validation), only a genuine forward-looking use case should.

    `cache_dir` (ticket 40): the replay is a pure function of (field, date0,
    rows before date0), and the live sim called it with the same inputs every
    30 minutes at ~60-100 s a time. When set, each tournament's snapshot is
    pickled under cache_dir keyed by tournament key, date0 and a fingerprint
    of the rows before date0; a hit skips the replay entirely. Opt-in, so
    backtests that pass many cutoffs are unaffected."""
    log = lambda m: print(m, flush=True)
    if df is None:
        log("loading prepared archive...")
        df, _dropped = hm.load_prepared(use_cache=True)

    cached: dict = {}
    cache_paths: dict = {}
    if cache_dir is not None:
        import pickle
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for key, meta in tournaments_meta.items():
            fp = snapshot_fingerprint(df, meta["date0"])
            safe = key.replace(":", "_").replace("/", "_")
            path = cache_dir / f"{safe}_{pd.Timestamp(meta['date0']).date()}_{fp}.pkl"
            cache_paths[key] = path
            if path.exists():
                with open(path, "rb") as fh:
                    cached[key] = pickle.load(fh)   # nosec: our own cache, written by this function
        if cached:
            log(f"snapshot cache: {len(cached)}/{len(tournaments_meta)} tournament(s) loaded from {cache_dir}")
        tournaments_meta = {k: v for k, v in tournaments_meta.items() if k not in cached}
        if not tournaments_meta:
            return cached

    pending: dict = {}
    for key, meta in tournaments_meta.items():
        pending.setdefault(meta["date0"], []).append(key)
    field_by_key = {key: set(meta["field_uids"]) for key, meta in tournaments_meta.items()}

    players: dict[str, "hm.PlayerState"] = {}

    def get(uid: str) -> "hm.PlayerState":
        s = players.get(uid)
        if s is None:
            s = hm.PlayerState()
            players[uid] = s
        return s

    snapshots: dict[str, dict[str, "hm.PlayerState"]] = {}
    n = len(df)
    t0 = time.time()
    log(f"replaying {n:,} matches, snapshotting {len(pending)} cutoff dates...")
    for i, row in enumerate(df.itertuples(index=False)):
        if not pending:
            break
        if row.date >= min(pending.keys()):
            due = [d for d in pending if row.date >= d]
            for d in sorted(due):
                for key in pending.pop(d):
                    snapshots[key] = {uid: clone_state(players[uid]) for uid in field_by_key[key] if uid in players}
            if not pending:
                break

        w, l = row.winner_uid, row.loser_uid
        s = row.surface
        date = row.date
        sv, sl = get(w), get(l)
        hm.inflate(sv, date)
        hm.inflate(sl, date)

        k_w, vk_w = sv.get_kappa(s)
        k_l, vk_l = sl.get_kappa(s)
        lm_w, vlm_w = sv.get_lam(s)
        lm_l, vlm_l = sl.get_lam(s)

        if row.has_points:
            e_w = sv.m_a + k_w
            f_l = sl.m_b + lm_l
            v_e = sv.v_a + vk_w
            v_f = sl.v_b + vlm_l
            n1, y1 = float(row.w_svpt), float(row.w_won)
            new_ew, new_ve, new_fl, new_vf = hm.adf_update(e_w, v_e, f_l, v_f, y1, n1)
            sv.m_a, sv.v_a, k_w, vk_w = hm.apply_side_update(sv.m_a, sv.v_a, k_w, vk_w, e_w, new_ew, new_ve)
            sl.m_b, sl.v_b, lm_l, vlm_l = hm.apply_side_update(sl.m_b, sl.v_b, lm_l, vlm_l, f_l, new_fl, new_vf)

            e_l = sl.m_a + k_l
            f_w = sv.m_b + lm_w
            v_e2 = sl.v_a + vk_l
            v_f2 = sv.v_b + vlm_w
            n2, y2 = float(row.l_svpt), float(row.l_won)
            new_el, new_ve2, new_fw, new_vf2 = hm.adf_update(e_l, v_e2, f_w, v_f2, y2, n2)
            sl.m_a, sl.v_a, k_l, vk_l = hm.apply_side_update(sl.m_a, sl.v_a, k_l, vk_l, e_l, new_el, new_ve2)
            sv.m_b, sv.v_b, lm_w, vlm_w = hm.apply_side_update(sv.m_b, sv.v_b, lm_w, vlm_w, f_w, new_fw, new_vf2)

            sv.kappa[s], sv.kappa_var[s] = k_w, vk_w
            sl.kappa[s], sl.kappa_var[s] = k_l, vk_l
            sv.lam[s], sv.lam_var[s] = lm_w, vlm_w
            sl.lam[s], sl.lam_var[s] = lm_l, vlm_l
        else:
            # Exact reproduction of hierarchical_model.run()'s tier_scale logic
            # (REGIME_B_FULL_DP left off, matching this project's default).
            tier_scale = hm.REGIME_B_N if row.tourney_level in hm.ELITE_LEVELS else hm.REGIME_B_N * hm.TIER_WEIGHT
            if row.tour == "wta" and row.level_file == "lower":
                tier_scale *= hm.WTA_LOWER_WEIGHT
            if not np.isnan(row.total_games_w):
                games_w, games_l = row.total_games_w, row.total_games_l
            else:
                games_w, games_l = 1.0, 0.0   # unparseable score: old fallback (y=n=tier_scale)
            apply_regime_b_update(sv, sl, games_w, games_l, tier_scale)

        sv.n_matches += 1
        sl.n_matches += 1
        if (i + 1) % 300_000 == 0:
            log(f"  ...{i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")

    if pending:
        # The archive/bridge never reached this tournament's true start date
        # (e.g. a forward-looking forecast whose data source lags the real
        # calendar) - the loop above only fires ON CROSSING a cutoff, so it
        # silently never snapshots these. Bug found 2026-08-29: this used to
        # fall through to an empty {} via callers' .get(key, {}), silently
        # blanking EVERY player's hier state (m_a=m_b=0, no shrinkage) rather
        # than using their real latest-known state - a large, undiscovered
        # regression in a "fresh data" run that only had FRESH ELO, not
        # fresh hier. Correct behaviour for this case is "state as of the
        # latest available data", the same fallback every other feature
        # (elo, age, serve form) already uses.
        log(f"  {sum(len(v) for v in pending.values())} tournament(s) never reached their own cutoff date in this "
            f"data (latest available: {df['date'].max().date()}) - snapshotting at end-of-data instead")
        for d, keys in pending.items():
            for key in keys:
                snapshots[key] = {uid: clone_state(players[uid]) for uid in field_by_key[key] if uid in players}

    log(f"replay done: {time.time()-t0:.0f}s, {len(snapshots)}/{len(tournaments_meta)} tournaments snapshotted")
    if cache_dir is not None:
        import pickle
        for key, snap in snapshots.items():
            with open(cache_paths[key], "wb") as fh:
                pickle.dump(snap, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"snapshot cache: wrote {len(snapshots)} to {cache_dir}")
    snapshots.update(cached)
    return snapshots


def main() -> None:
    log = lambda m: print(m, flush=True)
    n_sims = int(os.environ.get("DSV_N_SIMS", "2000"))

    log("loading tournament fields...")
    tournaments_meta = {}
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        tournaments_meta[key] = load_tournament_field(tour, tourney_id, year)

    snapshots = build_snapshots(tournaments_meta)

    all_y: dict[str, dict[str, list[int]]] = {arm: {s: [] for s in STAGE_ORDER[1:]} for arm in ("frozen", "updating")}
    all_p: dict[str, dict[str, list[float]]] = {arm: {s: [] for s in STAGE_ORDER[1:]} for arm in ("frozen", "updating")}
    top_seed_ranks: dict[str, list[tuple]] = {"frozen": [], "updating": []}

    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        meta = tournaments_meta[key]
        field_uids = meta["field_uids"]
        field = meta["field"]
        surface = meta["surface"]
        real_stage = meta["real_stage"]
        top_seed_uid = meta["top_seed_uid"]
        snapshot = snapshots.get(key, {})
        best_of = 5 if tour == "atp" else 3
        n_missing = sum(1 for u in field_uids if u not in snapshot)
        log(f"  [{key}] field={len(field_uids)} missing_state={n_missing} surface={surface}")

        seed = hash(key) % 100000
        for arm, apply_updates in (("frozen", False), ("updating", True)):
            rng = np.random.default_rng(seed + (0 if arm == "frozen" else 1))
            stage_counts = {uid: {s: 0 for s in STAGE_ORDER} for uid in field_uids}
            for _ in range(n_sims):
                bracket = build_random_bracket(field, rng)
                result = simulate_bracket(bracket, snapshot, surface, best_of, rng, apply_updates)
                for uid, stage in result.items():
                    reached_idx = STAGE_ORDER.index(stage)
                    for s_idx in range(reached_idx + 1):
                        stage_counts[uid][STAGE_ORDER[s_idx]] += 1
            probs = {uid: {s: stage_counts[uid][s] / n_sims for s in STAGE_ORDER} for uid in field_uids}

            if top_seed_uid:
                p_title = {uid: probs[uid]["W"] for uid in field_uids}
                ranked = sorted(p_title.items(), key=lambda kv: -kv[1])
                rank = [u for u, _ in ranked].index(top_seed_uid) + 1
                top_seed_ranks[arm].append((tour, tourney_id, rank, len(field_uids)))

            for s_idx, stage in enumerate(STAGE_ORDER[1:], start=1):
                y = [1 if STAGE_ORDER.index(real_stage[u]) >= s_idx else 0 for u in field_uids]
                p = [probs[u][stage] for u in field_uids]
                all_y[arm][stage].extend(y)
                all_p[arm][stage].extend(p)

    for arm in ("frozen", "updating"):
        log(f"\n=== anchor ({arm}): top seed's simulated title-probability rank ===")
        n_top3 = 0
        for t, tid, rank, nf in top_seed_ranks[arm]:
            flag = "OK" if rank <= 3 else "check"
            if rank <= 3:
                n_top3 += 1
            log(f"  {t} {tid}: rank #{rank} of {nf}  [{flag}]")
        log(f"  top seed in top-3: {n_top3}/{len(top_seed_ranks[arm])}")

    log("\n=== POOLED stage-reaching log-loss: frozen-hier vs updating-hier ===")
    for stage in STAGE_ORDER[1:]:
        row = {}
        for arm in ("frozen", "updating"):
            y = np.array(all_y[arm][stage])
            p = np.clip(np.array(all_p[arm][stage]), 1e-9, 1 - 1e-9)
            ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            row[arm] = (ll, len(y), float(y.mean()))
        ll_f, n_f, base_f = row["frozen"]
        ll_u, n_u, base_u = row["updating"]
        winner = "updating wins" if ll_u < ll_f else ("frozen wins" if ll_f < ll_u else "tie")
        log(f"  >= {stage:4s}  n={n_f:4d}  base_rate={base_f:.3f}  "
            f"frozen_logloss={ll_f:.4f}  updating_logloss={ll_u:.4f}  "
            f"delta(upd-frz)={ll_u - ll_f:+.4f}  {winner}")


if __name__ == "__main__":
    main()
