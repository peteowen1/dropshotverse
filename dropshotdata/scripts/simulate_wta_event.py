"""Forecast an upcoming WTA event from its entry list, before the draw exists.

Option B of docs/plans/between-events-ratings.md. The US Open forecast was
hand-built around usopen.org's feeds; this is the generic version for any WTA
event whose field `fetch_wta_upcoming.py` can resolve. First target: Beijing
2026 (tournament group 1020, WTA 1000, 96 draw, starts 2026-09-30).

WHAT IT DOES
  1. Reads the event row from the cached WTA calendar (start date, surface,
     draw size, level) and the entrants file the fetcher wrote.
  2. Builds the field: archive uids, numeric seeds. An entrant the archive
     cannot rate gets a synthetic uid and a fresh rating state, exactly as the
     US Open forecast treated Merida and Wang.
  3. Rates the field as of the event start with the shipped XGBoost pairwise
     model (hier snapshot at start date, current Elo, playstyle), one matrix
     per round, antisymmetrised.
  4. Simulates seeding-consistent random brackets. A 96 draw is 128 slots with
     32 BYES sitting opposite seeds 1 to 32; a bye always loses. Slots the
     entry list has not filled yet (qualifiers land the week before) are TBD
     placeholders that play like the weakest entrant, so a provisional run is
     conservative rather than flattering.
  5. Refuses to publish unless the field is complete and seeded; a partial or
     unseeded run is written only with --allow-partial and tagged
     source="provisional".
  6. Writes rows keyed by tournament_id into blog/upcoming_predictions.parquet,
     replacing that tournament's previous rows and keeping every other event.
     The locked US Open tables are never touched.

ANCHORS, decided before looking: title probabilities sum to 1; every seed
holds p_r64 = 1 (the bye); the top seed is top 5 by title probability; no NaN.
A FAIL refuses the write.

LEVEL CODE: the model's tourney_level categorical uses the archive's codes.
Beijing 2025 is "PM" in wta_matches_2025.csv (checked). Mapping used here:
WTA 1000 with a 96 draw -> PM; other WTA 1000 -> P; WTA 500 -> P; WTA 250 -> I;
Grand Slam -> G. Anything else refuses rather than guesses.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import _common
import bridge_tennismylife as bridge
import hierarchical_model as hm
import simulate_tournament_updating as stu
from simulate_tournament import STAGE_ORDER, standard_seed_positions
from simulate_tournament_xgb import build_round_pred_matrices
from xgb_bt_rating import build_h2h_dict, build_player_table, load_age_table, train_bagged_model

REF = _common.DD / "reference"
CACHE = _common.RAW / "wta_api"
OUT_DEFAULT = _common.BLOG / "upcoming_predictions.parquet"
PUBLIC_URL = "https://pub-ee4bf5b599a047f9ac2b9facc1587008.r2.dev/tennis/upcoming_predictions.parquet"
BYE = "BYE"
TBD = "TBD"


def log(m: str = "") -> None:
    print(m, flush=True)


# ------------------------------------------------------------------ inputs
def calendar_row(group_id: int, year: int) -> dict:
    hits = []
    # `cal_*_p*.json` and not `cal_*.json`: the second also matches the
    # pre-paging cache files, which hold only the first 100 tournaments of
    # their window. Reading one of those here is how a fixed fetch still
    # produces "that event does not exist".
    for p in sorted(CACHE.glob("cal_*_p*.json")):
        body = json.loads(p.read_text(encoding="utf-8"))
        items = body if isinstance(body, list) else next((v for v in body.values() if isinstance(v, list)), [])
        for t in items:
            g = t.get("tournamentGroup") or {}
            if int(g.get("id", -1)) == group_id and int(t.get("year", 0)) == year:
                hits.append(t)
    if not hits:
        raise SystemExit(f"no cached calendar row for group {group_id} {year}; run "
                         f"fetch_wta_upcoming.py with a window covering the event first")
    return hits[-1]


def level_code(level: str, draw: int) -> str:
    lv = (level or "").upper()
    if "SLAM" in lv or lv == "G":
        return "G"
    if "1000" in lv:
        return "PM" if draw >= 96 else "P"
    if "500" in lv:
        return "P"
    if "250" in lv:
        return "I"
    raise SystemExit(f"unmapped WTA level {level!r}; add it to level_code() with an archive check")


def expected_seeds(draw: int) -> int:
    """Tour convention: 8 seeds to a 32 draw, 16 to a 64, 32 to a 96 or 128."""
    return 8 if draw <= 32 else 16 if draw <= 64 else 32


def load_entrants(group_id: int, year: int, refetch: bool) -> list[dict]:
    path = REF / f"wta_{group_id}_{year}_entrants.json"
    if refetch or not path.exists():
        log(f"fetching entry list for {group_id} {year} via fetch_wta_upcoming.py ...")
        r = subprocess.run([sys.executable, str(_common.SCRIPTS / "fetch_wta_upcoming.py"),
                            "--fetch", str(group_id), "--year", str(year),
                            "--from", f"{year}-01-01", "--to", f"{year}-12-31"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in r.stdout.splitlines()[-12:]:
            log("  | " + line)
        if r.returncode != 0 or not path.exists():
            raise SystemExit("entry list fetch failed or was refused (coverage); see lines above")
    return json.loads(path.read_text(encoding="utf-8"))


def build_field(entrants: list[dict], tour: str) -> dict:
    uids, names, seed_map = [], {}, {}
    synthetic = []
    for e in entrants:
        uid = e.get("uid") or f"{tour}:NEW_{e['name'].replace(' ', '_')}"
        if not e.get("uid"):
            synthetic.append(e["name"])
        uids.append(uid)
        names[uid] = e["name"]
        s = str(e.get("seed", "")).strip()
        if s.isdigit():
            seed_map[uid] = int(s)
    uids = sorted(set(uids))
    return {"field_uids": uids, "names": names, "seed_map": seed_map, "synthetic": synthetic}


# ------------------------------------------------------------------ bracket with byes
def random_bracket_with_byes(field_uids: list[str], seed_map: dict[str, int], n_slots: int,
                             n_byes: int, rng: np.random.Generator) -> list[str]:
    """Seeds to their tier anchors (randomised within tier, as
    simulate_tournament.build_random_bracket does), then a BYE in the slot
    paired with each of the top n_byes seeds, unseeded entrants at random in
    the rest, and TBD placeholders for slots the entry list has not filled."""
    slot_seed = standard_seed_positions(n_slots)
    bracket: list[str | None] = [None] * n_slots
    seeded = sorted(seed_map.items(), key=lambda kv: kv[1])
    seed_to_uid = {s: u for u, s in seeded}
    tiers: dict[int, list[int]] = {}
    for slot, s in enumerate(slot_seed):
        tier = 1 if s <= 1 else 2 ** math.ceil(math.log2(s))
        tiers.setdefault(tier, []).append(slot)
    seed_slot: dict[int, int] = {}
    for tier in sorted(tiers):
        prev = tier // 2 if tier > 1 else 0
        nums = [s for s in seed_to_uid if prev < s <= tier]
        slots = list(tiers[tier])
        rng.shuffle(slots)
        for s, slot in zip(sorted(nums), slots):
            bracket[slot] = seed_to_uid[s]
            seed_slot[s] = slot
    # byes opposite the top seeds, in seed order; if fewer seeds than byes the
    # rest of the byes go opposite the remaining seeds' tier slots at random
    bye_left = n_byes
    for s in sorted(seed_slot):
        if bye_left == 0:
            break
        partner = seed_slot[s] ^ 1
        if bracket[partner] is None:
            bracket[partner] = BYE
            bye_left -= 1
    # Everything else at random: leftover byes (only when the field has fewer
    # seeds than byes, i.e. a provisional run), then unseeded entrants, then
    # TBD for slots the entry list has not filled. Random rather than
    # first-free so leftover byes do not cluster at the top of the draw and
    # meet each other.
    unseeded = [u for u in field_uids if u not in seed_map]
    rng.shuffle(unseeded)
    free = [i for i in range(n_slots) if bracket[i] is None]
    rng.shuffle(free)
    if len(unseeded) > len(free) - bye_left:
        raise SystemExit(f"entry list ({len(field_uids)} entrants) does not fit the {n_slots - n_byes} main-draw "
                         f"slots: alternates or unwithdrawn entrants still listed; re-run nearer the event")
    for i in free:
        if bye_left > 0:
            bracket[i] = BYE
            bye_left -= 1
        elif unseeded:
            bracket[i] = unseeded.pop()
        else:
            bracket[i] = TBD
    return bracket  # type: ignore[return-value]


def simulate_bracket(bracket: list[str], idx: dict[str, int], matrices: dict[str, np.ndarray],
                     weakest: int, rng: np.random.Generator) -> dict[str, str]:
    """simulate_frozen_xgb with BYE (always loses) and TBD (plays like the
    weakest rated entrant)."""
    reached = {u: STAGE_ORDER[0] for u in bracket if u not in (BYE, TBD)}
    current = list(bracket)
    r = 0
    while len(current) > 1:
        M = matrices[STAGE_ORDER[r]]
        nxt = []
        for k in range(0, len(current), 2):
            a, b = current[k], current[k + 1]
            if a == BYE and b == BYE:
                nxt.append(BYE)
                continue
            if b == BYE:
                nxt.append(a)
                continue
            if a == BYE:
                nxt.append(b)
                continue
            ia = weakest if a == TBD else idx[a]
            ib = weakest if b == TBD else idx[b]
            p_a = 0.5 if (a == TBD and b == TBD) else M[ia, ib]
            nxt.append(a if rng.random() < p_a else b)
        r += 1
        stage = STAGE_ORDER[r] if r < len(STAGE_ORDER) else "W"
        for u in nxt:
            if u not in (BYE, TBD):
                reached[u] = stage
        current = nxt
    return reached


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True, help="WTA tournamentGroup id, e.g. 1020 = Beijing")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--no-refetch", action="store_true",
                    help="use the entrants file on disk instead of pulling it fresh (default: always refetch, "
                         "so a stale local list can never feed a run)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even if the field is incomplete or unseeded (tagged provisional)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--no-merge", action="store_true", help="do not merge other events from the published table")
    args = ap.parse_args()

    ev = calendar_row(args.group, args.year)
    tour = "wta"
    name = (ev.get("tournamentGroup") or {}).get("name", str(args.group)).title()
    title = ev.get("title") or name
    date0 = pd.Timestamp(ev["startDate"])
    surface = str(ev.get("surface") or "Hard").title()
    draw = int(ev.get("singlesDrawSize") or 0)
    level = level_code((ev.get("tournamentGroup") or {}).get("level", ""), draw)
    n_slots = 1 << (draw - 1).bit_length() if draw else 128
    n_byes = n_slots - draw
    tournament_id = f"{args.year}-{args.group}"
    log(f"=== {title}: starts {date0.date()}, {surface}, draw {draw} -> {n_slots} slots, {n_byes} byes, level {level} ===")

    entrants = load_entrants(args.group, args.year, refetch=not args.no_refetch)
    meta = build_field(entrants, tour)
    field_uids, names, seed_map = meta["field_uids"], meta["names"], meta["seed_map"]
    n_seeds = len(seed_map)
    n_tbd = max(0, draw - len(field_uids))
    log(f"field: {len(field_uids)} entrants of {draw}, {n_seeds} seeded, {n_tbd} slots still to be filled, "
        f"{len(meta['synthetic'])} unrated (synthetic uid): {meta['synthetic'][:6]}")
    complete = n_tbd == 0 and n_seeds >= expected_seeds(draw)
    if not complete and not args.allow_partial:
        raise SystemExit("field is not complete and seeded yet; re-run nearer the event, or pass "
                         "--allow-partial for a provisional forecast")
    source = "locked" if complete else "provisional"

    log("model + inputs ...")
    models, cat_levels, playstyle_full = train_bagged_model()
    age_table = load_age_table()
    df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
    ratings_as_of = df["date"].max()
    # Archive spelling for anyone we can rate ("Aryna Sabalenka", not the API's
    # "ARYNA SABALENKA"), most recent appearance wins; API name for the rest.
    w = df[["date", "winner_uid", "winner_name"]].rename(columns={"winner_uid": "uid", "winner_name": "name"})
    l = df[["date", "loser_uid", "loser_name"]].rename(columns={"loser_uid": "uid", "loser_name": "name"})
    latest = pd.concat([w, l], ignore_index=True).sort_values("date").drop_duplicates("uid", keep="last")
    archive_names = dict(zip(latest["uid"], latest["name"]))
    names = {u: archive_names.get(u, n.title()) for u, n in names.items()}
    key = f"{tour}:{tournament_id}"
    snaps = stu.build_snapshots({key: {"date0": date0, "field_uids": field_uids}}, df=df,
                                cache_dir=_common.SCRATCH / "snapshot_cache")
    snap = snaps.get(key, {})
    h2h = build_h2h_dict(set(field_uids))
    ptab = build_player_table(field_uids, tour, date0, surface, snap, playstyle_full, age_table)
    matrices = build_round_pred_matrices(field_uids, ptab, tour, surface, 3, level, h2h, date0,
                                         cat_levels, models, STAGE_ORDER[:-1])
    idx = {u: k for k, u in enumerate(field_uids)}
    # TBD plays like the weakest entrant: the row with the lowest mean win prob in the first round matrix
    first = matrices[STAGE_ORDER[0]]
    weakest = int(np.nanargmin(np.nanmean(first, axis=1)))
    log(f"TBD placeholders play like {names[field_uids[weakest]]} (weakest rated entrant)")

    rng = np.random.default_rng(int(f"{args.year}{args.group}") % (2**32))
    counts = {u: {s: 0 for s in STAGE_ORDER} for u in field_uids}
    for _ in range(args.sims):
        bracket = random_bracket_with_byes(field_uids, seed_map, n_slots, n_byes, rng)
        for u, stage in simulate_bracket(bracket, idx, matrices, weakest, rng).items():
            counts[u][stage] += 1
    probs = {}
    for u in field_uids:
        cum, run = {}, 0
        for s in reversed(STAGE_ORDER):
            run += counts[u][s]
            cum[s] = run / args.sims
        probs[u] = cum

    # anchors
    problems = []
    total = sum(probs[u]["W"] for u in field_uids)
    if abs(total - 1.0) > 1e-6:
        problems.append(f"title probabilities sum to {total:.4f}")
    for u, s in seed_map.items():
        if s <= n_byes and probs[u]["R64"] < 1.0 - 1e-9:
            problems.append(f"seed {s} ({names[u]}) has p_r64 {probs[u]['R64']:.3f}, expected 1 (bye)")
            break
    if seed_map:
        top = min(seed_map, key=seed_map.get)
        rank = sorted(field_uids, key=lambda u: -probs[u]["W"]).index(top) + 1
        if rank > 5:
            problems.append(f"top seed {names[top]} ranks {rank} by title probability")
    if any(np.isnan(v) for p in probs.values() for v in p.values()):
        problems.append("NaN in probabilities")
    log("anchors: " + ("PASS" if not problems else "FAIL " + str(problems)))
    if problems:
        raise SystemExit(f"anchors FAILED, refusing to write: {problems}")

    log(f"\n=== {title}: top 12 by P(title), {args.sims:,} sims, {source} ===")
    for i, u in enumerate(sorted(field_uids, key=lambda u: -probs[u]["W"])[:12], 1):
        sd = f"(#{seed_map[u]})" if u in seed_map else ""
        log(f"  {i:2d}. {names[u]:<26} {sd:<6} title {probs[u]['W']:6.1%}  final {probs[u]['F']:6.1%}  "
            f"QF {probs[u]['QF']:6.1%}")

    updated_at = datetime.now(timezone.utc).isoformat()
    rows = [{
        "tournament_id": tournament_id, "tournament_name": title, "year": args.year, "tour": tour,
        "surface": surface, "level": level, "draw_size": draw, "start_date": date0.strftime("%Y-%m-%d"),
        "player_id": u, "player_name": names[u], "seed": float(seed_map[u]) if u in seed_map else np.nan,
        "p_r64": round(probs[u]["R64"], 4), "p_r32": round(probs[u]["R32"], 4), "p_r16": round(probs[u]["R16"], 4),
        "p_qf": round(probs[u]["QF"], 4), "p_sf": round(probs[u]["SF"], 4), "p_f": round(probs[u]["F"], 4),
        "p_title": round(probs[u]["W"], 4),
        "field_status": f"{len(field_uids)}/{draw} entrants, {n_seeds} seeds, {n_tbd} tbd",
        "source": source, "ratings_as_of": ratings_as_of.strftime("%Y-%m-%d"), "updated_at": updated_at,
    } for u in field_uids]
    new = pd.DataFrame(rows).sort_values("p_title", ascending=False)

    out = Path(args.out)
    prev = None
    if not args.no_merge:
        try:
            prev = pd.read_parquet(out) if out.exists() else pd.read_parquet(PUBLIC_URL)
            log(f"merging with existing table: {len(prev):,} rows, events {sorted(prev['tournament_id'].unique())}")
        except Exception as e:                                       # noqa: BLE001
            log(f"no existing table to merge ({type(e).__name__}); writing this event alone")
    if prev is not None and len(prev):
        prev = prev[prev["tournament_id"] != tournament_id]
        new = pd.concat([prev, new], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out, index=False)
    log(f"wrote {out} ({len(new):,} rows, {new['tournament_id'].nunique()} event(s))")


if __name__ == "__main__":
    main()
