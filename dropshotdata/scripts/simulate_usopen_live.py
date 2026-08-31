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
from simulate_tournament import STAGE_ORDER
from simulate_tournament_xgb import build_round_pred_matrices
from xgb_bt_rating import (build_h2h_dict, build_player_table, load_age_table,
                           train_bagged_model)

# Same values simulate_us_open_2026.py locks - kept here rather than imported so
# this script does not depend on that module's import-time behaviour.
DATE0 = pd.Timestamp("2026-08-30")   # real 2026 US Open main-draw start
SURFACE = "Hard"

OUT_DIR = Path(__file__).resolve().parent.parent / "blog"
N_SIMS_DEFAULT = 20000


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
    rng = np.random.default_rng(hash(tour + "usopen-live") % 100000)
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
    return {"field_uids": field_uids, "names": names, "seed_map": seed_map,
            "probs": probs, "n_done": n_done}


def report(label: str, res: dict, top_n: int = 12) -> None:
    probs, names, seeds = res["probs"], res["names"], res["seed_map"]
    order = sorted(res["field_uids"], key=lambda u: -probs[u]["W"])
    log(f"\n=== {label} - title probability, LIVE "
        f"({res['n_done']} matches decided) ===")
    log(f"{'#':>3} {'player':<26} {'seed':>4} {'title':>7} {'final':>7} {'semi':>7} {'status':>9}")
    for i, uid in enumerate(order[:top_n], 1):
        sd = seeds.get(uid)
        sd = "-" if sd is None else str(int(sd))
        alive = "OUT" if probs[uid]["W"] == 0 and probs[uid]["R64"] == 0 else ""
        log(f"{i:>3} {names.get(uid, uid):<26} {sd:>4} {probs[uid]['W']:>6.1%} "
            f"{probs[uid]['F']:>7.1%} {probs[uid]['SF']:>7.1%} {alive:>9}")

    # anchor: anyone already eliminated must have exactly zero title probability
    eliminated = [u for u in res["field_uids"] if probs[u]["W"] > 0 and probs[u]["R64"] == 0]
    log(f"  anchor: no eliminated player holds title probability -> "
        f"{'PASS' if not eliminated else 'FAIL ' + str(eliminated[:3])}")


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
            d["results"] = {}          # real bracket, clean slate - diagnostic arm
        draws[tour] = d
        metas[f"{tour}:2026-usopen"] = {"date0": DATE0, "field_uids": sorted(set(d["bracket"]))}

    all_snaps = stu.build_snapshots(metas, df=df)

    out = {}
    for tour in ("atp", "wta"):
        snap = all_snaps.get(f"{tour}:2026-usopen", {})
        out[tour] = run_tour(tour, draws[tour], snap, models, cat_levels,
                             playstyle_full, age_table, args.sims)
    report("MEN'S SINGLES", out["atp"])
    report("WOMEN'S SINGLES", out["wta"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {t: {"names": r["names"],
                   "seed_map": {k: int(v) for k, v in r["seed_map"].items()},
                   "probs": r["probs"], "n_done": r["n_done"]}
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
                # still alive == has not yet lost; R64==0 means eliminated in R1
                "eliminated": bool(p["R64"] == 0.0),
                "matches_decided": r["n_done"],
                "source": "live", "ratings_as_of": str(DATE0.date()), "updated_at": now,
            })
    pq = OUT_DIR / "live_predictions.parquet"
    pd.DataFrame(rows).sort_values(["tour", "p_title"], ascending=[True, False]).to_parquet(pq, index=False)
    log(f"wrote {pq} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
