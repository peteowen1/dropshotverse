"""Live-updating US Open tournament projection: re-simulate from the CURRENT state.

Difference from `simulate_us_open_2026.py`
------------------------------------------
That script produces the *locked* pre-tournament forecast. It calls
`build_random_bracket()` - a fresh seeding-consistent but INVENTED draw per
simulation - and starts all 128 players at R128. It therefore cannot absorb a
result at all.

This script uses the real bracket from `usopen_live_draw.py` and treats every
completed match as deterministic, sampling only the matches still to be played.
Run it again after more results land and the numbers move.

What is NOT re-done here
------------------------
Player ratings are still frozen as of the tournament's start date (`DATE0`).
Only the *bracket state* updates. That is deliberate and matches ticket 29's
measured finding: cold (frozen, pre-tournament) ratings beat hot/current
ratings outright at every damping level tested for tournament simulation. So a
player who wins does NOT get a rating bump mid-event - they simply advance,
which is where nearly all the probability movement legitimately comes from.

Usage
-----
    python simulate_usopen_live.py              # both tours, writes JSON
    python simulate_usopen_live.py --sims 20000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import bridge_tennismylife as bridge
import hierarchical_model as hm
import simulate_tournament_updating as stu
import usopen_live_draw as draw_mod
from simulate_tournament import DD, STAGE_ORDER
from simulate_tournament_xgb import build_round_pred_matrices
from xgb_bt_rating import (build_h2h_dict, build_player_table, load_age_table,
                           train_bagged_model)

# Same values simulate_us_open_2026.py locks - kept here rather than imported so
# this script does not depend on that module's import-time behaviour.
DATE0 = pd.Timestamp("2026-08-30")   # real 2026 US Open main-draw start
SURFACE = "Hard"

OUT_DIR = Path(__file__).resolve().parent.parent / "blog"
N_SIMS_DEFAULT = 20000

# Fixed integer seeds, one per tour - the same convention every other simulator
# in this repo uses. This previously seeded off hash(str), which Python salts
# per process (PYTHONHASHSEED), so no published refresh could be reproduced and
# part of every "the number moved" was seed churn rather than the bracket.
TOUR_SEED = {"atp": 0, "wta": 1}


def log(m: str) -> None:
    print(m, flush=True)


def simulate_from_state(bracket: list[str], results: dict[str, str], event: str,
                        idx: dict[str, int], round_matrices: dict[str, np.ndarray],
                        rng: np.random.Generator) -> dict[str, str]:
    """One bracket playthrough that RESPECTS completed matches.

    Ordering invariant this relies on: survivors stay in bracket order, so in
    round `r` the k-th match (1-based) is always fed by list positions
    2k-2 and 2k-1. That is the same invariant `simulate_frozen_xgb` uses; here
    it additionally lets us address the feed's match id for the same pairing.
    """
    stage_reached = {uid: STAGE_ORDER[0] for uid in bracket}
    current = list(bracket)
    round_no = 1

    while len(current) > 1:
        stage = STAGE_ORDER[round_no - 1]
        M = round_matrices[stage]
        nxt: list[str] = []
        for k in range(0, len(current), 2):
            a, b = current[k], current[k + 1]
            mid = draw_mod.match_id(round_no, k // 2 + 1, event)
            known = results.get(mid)
            if known == a or known == b:
                winner = known           # real result - not a coin toss
            else:
                p_a = M[idx[a], idx[b]]
                winner = a if rng.random() < p_a else b
            nxt.append(winner)
        round_no += 1
        next_stage = STAGE_ORDER[round_no - 1] if round_no - 1 < len(STAGE_ORDER) else "W"
        for uid in nxt:
            stage_reached[uid] = next_stage
        current = nxt
    return stage_reached


def run_tour(tour: str, draw: dict, snap: dict, models: list, cat_levels: dict,
             playstyle_full: pd.DataFrame, age_table: pd.DataFrame, n_sims: int) -> dict:
    best_of = 5 if tour == "atp" else 3
    bracket = draw["bracket"]
    field_uids = sorted(set(bracket))
    names, seed_map = draw["names"], draw["seed_map"]

    n_done = len(draw["results"])
    log(f"  [{tour}] {len(field_uids)} players, {n_done} matches already decided, "
        f"{127 - n_done} left to simulate")

    h2h = build_h2h_dict(set(field_uids))
    ptab = build_player_table(field_uids, tour, DATE0, SURFACE, snap, playstyle_full, age_table)
    round_matrices = build_round_pred_matrices(field_uids, ptab, tour, SURFACE, best_of, "G",
                                               h2h, DATE0, cat_levels, models, STAGE_ORDER[:-1])

    idx = {u: k for k, u in enumerate(field_uids)}
    rng = np.random.default_rng(TOUR_SEED[tour])
    counts = {uid: {s: 0 for s in STAGE_ORDER} for uid in field_uids}
    for _ in range(n_sims):
        reached = simulate_from_state(bracket, draw["results"], draw["event"], idx,
                                      round_matrices, rng)
        for uid, stage in reached.items():
            counts[uid][stage] += 1

    # cumulative: P(reached at least stage s)
    probs: dict[str, dict[str, float]] = {}
    for uid in field_uids:
        cum, run = {}, 0
        for s in reversed(STAGE_ORDER):
            run += counts[uid][s]
            cum[s] = run / n_sims
        probs[uid] = cum
    # Elimination comes from the feed (who actually lost), never from a stage
    # probability. A player knocked out in R2 still has P(reach R64) == 1.0.
    eliminated = {u for u in field_uids if u in draw["losers"]}
    # round_matrices/idx are carried out so the pairwise probabilities can be
    # PUBLISHED. They were built here, used to run the simulation, and thrown
    # away -- which is the whole of dropshotverse#6: the blog replays the draw
    # from bt_rating, a single number that is the row-mean of this matrix at one
    # fixed context (REFERENCE_SURFACE/REFERENCE_ROUND), so it cannot reproduce
    # the model's own match odds. Publishing the matrix closes that gap.
    return {"field_uids": field_uids, "names": names, "seed_map": seed_map,
            "round_matrices": round_matrices, "idx": idx,
            "probs": probs, "n_done": n_done, "eliminated": eliminated}


def report(label: str, res: dict, top_n: int = 12) -> None:
    probs, names, seeds = res["probs"], res["names"], res["seed_map"]
    out = res["eliminated"]
    order = sorted(res["field_uids"], key=lambda u: -probs[u]["W"])
    log(f"\n=== {label} - title probability, LIVE "
        f"({res['n_done']} matches decided, {len(out)} players out) ===")
    log(f"{'#':>3} {'player':<26} {'seed':>4} {'title':>7} {'final':>7} {'semi':>7} {'status':>9}")
    for i, uid in enumerate(order[:top_n], 1):
        sd = seeds.get(uid)
        sd = "-" if sd is None else str(int(sd))
        alive = "OUT" if uid in out else ""
        log(f"{i:>3} {names.get(uid, uid):<26} {sd:>4} {probs[uid]['W']:>6.1%} "
            f"{probs[uid]['F']:>7.1%} {probs[uid]['SF']:>7.1%} {alive:>9}")

    # Anchor 1: a player the feed says has LOST must hold zero title probability.
    # This can genuinely fail - if a completed result were not applied (a
    # match_id the simulator addresses differently from the feed, say), the
    # loser would keep advancing in simulation and show a non-zero title
    # chance. The previous form of this check tested W > 0 and R64 == 0, which
    # is unsatisfiable for cumulative probabilities and so always printed PASS.
    zombies = [u for u in res["field_uids"] if u in out and probs[u]["W"] > 0]
    log(f"  anchor: no eliminated player holds title probability -> "
        f"{'PASS' if not zombies else 'FAIL ' + str([names.get(u, u) for u in zombies[:3]])}")
    failures = []
    if zombies:
        failures.append("eliminated player holds title probability")

    # Anchor 2: exactly one player wins, so title probability must sum to 1.
    total = sum(probs[u]["W"] for u in res["field_uids"])
    log(f"  anchor: title probabilities sum to 1.0 -> {total:.4f} "
        f"{'PASS' if abs(total - 1.0) < 1e-6 else 'FAIL'}")
    if abs(total - 1.0) >= 1e-6:
        failures.append(f"title probabilities sum to {total:.4f}")

    # Anchor 3: survivors and the elimination set must partition the field.
    alive_n = len(res["field_uids"]) - len(out)
    expected_alive = len(res["field_uids"]) - res["n_done"]
    log(f"  anchor: {alive_n} alive vs {expected_alive} expected from "
        f"{res['n_done']} decided -> "
        f"{'PASS' if alive_n == expected_alive else 'FAIL'}")
    if alive_n != expected_alive:
        failures.append(f"{alive_n} alive vs {expected_alive} expected")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_SIMS_DEFAULT)
    ap.add_argument("--ignore-results", action="store_true",
                    help="Use the REAL bracket but pretend nothing has been played. "
                         "Diagnostic only: comparing locked -> this isolates the effect of "
                         "knowing the draw, and this -> live isolates the effect of results. "
                         "The locked-vs-live difference alone conflates the two.")
    ap.add_argument("--out", default="us_open_2026_live_predictions.json")
    args = ap.parse_args()

    log("loading model...")
    models, cat_levels, playstyle_full = train_bagged_model()
    log("loading bridged archive for name crosswalk...")
    df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
    n2u = draw_mod.build_name_to_uid(df)
    age_table = load_age_table()

    draws, metas = {}, {}
    for tour in ("atp", "wta"):
        d = draw_mod.load_live_draw(tour, n2u)
        problems = draw_mod.verify(d)
        if problems:
            raise SystemExit(f"{tour}: draw failed verification, refusing to simulate "
                             f"on a bracket we cannot trust: {problems[:3]}")
        if args.ignore_results:
            # real bracket, clean slate - diagnostic arm. `losers` must be
            # cleared alongside `results` or the report would call players out
            # in an arm that is pretending nothing has been played.
            d["results"], d["losers"] = {}, set()
        draws[tour] = d
        metas[f"{tour}:2026-usopen"] = {"date0": DATE0, "field_uids": sorted(set(d["bracket"]))}

    # Snapshot cache (ticket 40): the replay is a pure function of the field,
    # DATE0 and the rows before it, so a same-day rerun loads it from disk.
    all_snaps = stu.build_snapshots(metas, df=df, cache_dir=DD / "tune_scratch" / "snapshot_cache")

    out = {}
    for tour in ("atp", "wta"):
        snap = all_snaps.get(f"{tour}:2026-usopen", {})
        out[tour] = run_tour(tour, draws[tour], snap, models, cat_levels,
                             playstyle_full, age_table, args.sims)
    # The anchors used to be printed and ignored, so a failing one reached R2
    # with exit 0 and the .cmd wrapper's errorlevel guard never fired. A FAIL
    # now refuses the write, same as a draw that fails verification above:
    # the site keeps the last good projection rather than publishing a wrong one.
    failures = {"atp": report("MEN'S SINGLES", out["atp"]),
                "wta": report("WOMEN'S SINGLES", out["wta"])}
    failures = {t: f for t, f in failures.items() if f}
    if failures:
        raise SystemExit(f"anchor check FAILED, refusing to write or publish: {failures}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {t: {"names": r["names"],
                   "seed_map": {k: int(v) for k, v in r["seed_map"].items()},
                   "probs": r["probs"], "n_done": r["n_done"],
                   "eliminated": sorted(r["eliminated"])}   # same fact as the parquet column
               for t, r in out.items()}
    payload["_meta"] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                        "n_sims": args.sims,
                        "source": "draw-only" if args.ignore_results else "live",
                        "ratings_as_of": str(DATE0.date())}
    path = OUT_DIR / args.out
    path.write_text(json.dumps(payload), encoding="utf-8")
    log(f"\nwrote {path}")

    # Parquet in the existing blog schema, so upload_blog_data.py (which globs
    # *.parquet) publishes it with no change and the site reads one format.
    # Skipped for the --ignore-results diagnostic arm: that is an analysis
    # artifact and must never reach the site as if it were the real forecast.
    if args.ignore_results:
        log("--ignore-results: diagnostic arm, deliberately NOT writing site parquet")
        return

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for tour, r in out.items():
        q = draw_mod.quarters(draws[tour]["bracket"])
        # event_status lets the site render a FINISHED event as a result rather
        # than as a live projection. Without it a completed tournament keeps
        # rendering as a forecast forever once the scheduled task's trigger
        # expires (2026-09-14), and nothing alarms - a frozen parquet is still a
        # structurally valid parquet, so ITG's validator keeps passing on stale
        # data. Computed per tour because the two finals are on different days.
        n_total = len(r["field_uids"]) - 1          # 128-draw -> 127 matches
        status = "complete" if r["n_done"] >= n_total else "live"
        champion = None
        if status == "complete":
            still_in = [u for u in r["field_uids"] if u not in r["eliminated"]]
            champion = r["names"].get(still_in[0]) if len(still_in) == 1 else None
        log(f"  [{tour}] event_status={status} ({r['n_done']}/{n_total} decided)"
            + (f", champion {champion}" if champion else ""))
        for uid in r["field_uids"]:
            p = r["probs"][uid]
            rows.append({
                "tournament_id": "2026-usopen", "tournament_name": "US Open", "year": 2026,
                "tour": tour, "player_id": uid, "player_name": r["names"].get(uid, uid),
                "seed": int(r["seed_map"][uid]) if uid in r["seed_map"] else None,
                "quarter": q.get(uid),
                "p_r64": round(p["R64"], 4), "p_r32": round(p["R32"], 4),
                "p_r16": round(p["R16"], 4), "p_qf": round(p["QF"], 4),
                "p_sf": round(p["SF"], 4), "p_f": round(p["F"], 4),
                "p_title": round(p["W"], 4),
                # out == the feed reports them losing a match. NOT derivable
                # from a stage probability: these are cumulative, so an R2
                # loser still has p_r64 == 1.0.
                "eliminated": bool(uid in r["eliminated"]),
                "matches_decided": r["n_done"],
                "matches_total": n_total,
                "event_status": status,
                "champion": champion,
                "source": "live", "ratings_as_of": str(DATE0.date()), "updated_at": now,
            })
    pq = OUT_DIR / "live_predictions.parquet"
    pd.DataFrame(rows).sort_values(["tour", "p_title"], ascending=[True, False]).to_parquet(pq, index=False)
    log(f"wrote {pq} ({len(rows)} rows)")

    # --- pairwise match probabilities (dropshotverse#6) --------------------
    #
    # The model's OWN answer to "does A beat B", per round, which is what the
    # simulation above actually runs on. Until now only bt_rating shipped, and
    # that is np.nanmean of this matrix's antisymmetric part at one fixed
    # context -- an average over the whole field. An average cannot say WHO you
    # are good against, so no single-number replay of it can reproduce these
    # odds. On 2026-09-04 the blog's simulator sat 13 points from the published
    # p_title on Alexander Zverev because of exactly that.
    #
    # Every ORDERED pair is stored, not just i<j. build_round_pred_matrices'
    # output is not guaranteed antisymmetric -- xgb_bt_rating.py explicitly
    # antisymmetrises with (L - L.T)/2 before taking its row means, which it
    # would not need to do if the raw matrix already were -- so halving the file
    # would silently discard the asymmetry rather than compress it. The measured
    # asymmetry is reported below so its size is known rather than assumed.
    prows = []
    asym_max = 0.0
    for tour, r in out.items():
        uids, idx_t = r["field_uids"], r["idx"]
        for stage, M in r["round_matrices"].items():
            arr = np.asarray(M, dtype=float)
            # How far from p_ij + p_ji == 1 this round's matrix sits. A number
            # near zero means a half-matrix would have been safe; a large one
            # means storing both directions was necessary.
            off = np.abs(arr + arr.T - 1.0)
            np.fill_diagonal(off, 0.0)
            asym_max = max(asym_max, float(np.nanmax(off)))
            for a in uids:
                ia = idx_t[a]
                for b in uids:
                    if a == b:
                        continue
                    prows.append({
                        "tournament_id": "2026-usopen", "tour": tour, "round": stage,
                        "player_id_a": a, "player_id_b": b,
                        "p_a_beats_b": round(float(arr[ia, idx_t[b]]), 5),
                        "ratings_as_of": str(DATE0.date()), "updated_at": now,
                    })
    ppq = OUT_DIR / "pairwise.parquet"
    pd.DataFrame(prows).to_parquet(ppq, index=False)
    log(f"wrote {ppq} ({len(prows):,} rows, {ppq.stat().st_size / 1e6:.1f} MB)")
    log(f"  max |p_ij + p_ji - 1| across every round = {asym_max:.4f} "
        f"({'antisymmetric to rounding' if asym_max < 0.01 else 'genuinely asymmetric - storing both directions was required'})")


if __name__ == "__main__":
    main()
