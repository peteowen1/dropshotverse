"""Ticket 29 follow-up: dropshotverse's equivalent of every other verse's
build_blog_data.R + build-blog-data.yml (torp/panna/bouncer all follow this
exact shape - see torpdata/scripts/build_blog_data.R and
torpdata/.github/workflows/build-blog-data.yml, read directly before writing
this, not guessed at). Same conventions on purpose: rounded, curated,
site-shaped parquet outputs in a local blog/ folder, `source` tagging so the
site can distinguish genuinely walk-forward predictions from anything
retrodictive, flat R2-ready filenames.

ONE REAL DIFFERENCE FROM THE OTHER VERSES, DELIBERATE: those verses'
workflow downloads SOURCE parquets from GitHub Releases (piggyback) before
aggregating - dropshotverse has no GitHub remote yet (checked: neither
dropshotverse nor dropshotdata has one configured), so there's no release to
download from. This script reads directly from the local artifacts that
already exist in this repo (elo_raw_ratings_full.parquet, the xgb-BT
machinery, etc.) instead of a source/ download step. Once dropshotverse gets
a GitHub remote and a piggyback-equivalent publish step, that gap can be
closed without changing this script's OUTPUT shape - only where it reads
from.

Outputs (written to blog/, ready for an R2 upload step once Cloudflare
credentials exist for this verse - see the draft
.github/workflows/build-blog-data.yml.template alongside this script):
  - ratings.parquet: latest snapshot, one row per active player (match in
    the last 52 weeks), overall/surface Elo + xgb-BT rating.
  - predictions.parquet: per-player stage-reaching + title probabilities
    for the 2026 US Open (both tours) - extensible to future tournaments
    by re-running with a different DATE0/entrants pair.

Does NOT include hier-derived ratings (serve/return split) - Pete's call:
"hier needs lots of work so let's not put that on the site yet."
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import bridge_tennismylife as bridge
import hierarchical_model as hm
import train_best_model as tbm
from simulate_tournament import DD, STAGE_ORDER
import simulate_tournament_updating as stu
from xgb_bt_rating import build_h2h_dict, build_pairwise_features_base, build_player_table, load_age_table, \
    prep_X_d, train_bagged_model

SCRATCH = DD / "tune_scratch"
BLOG_DIR = DD / "blog"
ENTRANTS_DIR = (r"C:\Users\peteo\AppData\Local\Temp\claude\C--dev-dropshotverse"
                r"\43d36b54-7c83-4d41-b4ff-6e4376e6c984\scratchpad")
REFERENCE_SURFACE = "Hard"   # most common surface tour-wide - the ratings.parquet BT context
REFERENCE_ROUND = "R32"      # neutral round, same convention xgb_bt_rating.py's own BT extraction uses


def log(m: str) -> None:
    print(m, flush=True)


RATINGS_POOL_SIZE = 150   # top-N by elo within the active pool - matches real ATP/WTA rankings page depth,
                          # and keeps the O(n^2) BT round-robin tractable (checked: n=459/385 unfiltered
                          # reached rank 300+ journeymen most fans wouldn't recognize, not a real "ratings page")


def build_active_pool(bridged_df: pd.DataFrame, tour: str, as_of: pd.Timestamp,
                      elo_path: str, top_n: int = RATINGS_POOL_SIZE) -> list[str]:
    """Active = at least one tour-level match in the 52 weeks before as_of -
    same "active" spirit elo_baseline.py's own anchor-check filter uses
    (>=15 matches in-year), loosened to a rolling window since this needs to
    work for any as_of date, not just a fixed calendar year. Then capped to
    the top_n by current Elo BEFORE any of the expensive per-player work
    below (hier snapshot, BT round-robin) - ranking on the cheap raw Elo
    lookup first, not after building the full table, is what actually saves
    the O(n^2) cost."""
    window_start = as_of - pd.Timedelta(weeks=52)
    recent = bridged_df[(bridged_df["tour"] == tour) & (bridged_df["level_file"] == "tour") &
                        (bridged_df["date"] >= window_start) & (bridged_df["date"] < as_of)]
    active = sorted(set(recent["winner_uid"]) | set(recent["loser_uid"]))
    if len(active) <= top_n:
        return active

    elo = pd.read_parquet(elo_path)
    elo_t = elo[(elo["tour"] == tour) & (elo["level_file"] == "tour") & (elo["date"] < as_of)]
    long_w = elo_t[["date", "winner_uid", "ro_winner"]].rename(columns={"winner_uid": "uid", "ro_winner": "ro"})
    long_l = elo_t[["date", "loser_uid", "ro_loser"]].rename(columns={"loser_uid": "uid", "ro_loser": "ro"})
    long_ro = pd.concat([long_w, long_l], ignore_index=True)
    long_ro = long_ro[long_ro["uid"].isin(active)].sort_values("date")
    latest_ro = long_ro.drop_duplicates(subset="uid", keep="last").set_index("uid")["ro"]
    ranked = latest_ro.reindex(active).fillna(1500.0).sort_values(ascending=False)
    return list(ranked.index[:top_n])


def build_name_lookup(bridged_df: pd.DataFrame) -> dict[str, str]:
    """uid -> most-recently-seen name, both tours - real archive names
    occasionally vary in spelling across a career (see the accent/hyphenation
    variants found in parse_us_open_draws.py's crosswalk); latest-wins is the
    same convention every other "current state" lookup in this pipeline uses."""
    w = bridged_df[["date", "winner_uid", "winner_name"]].rename(
        columns={"winner_uid": "uid", "winner_name": "name"})
    l = bridged_df[["date", "loser_uid", "loser_name"]].rename(columns={"loser_uid": "uid", "loser_name": "name"})
    long_all = pd.concat([w, l], ignore_index=True).sort_values("date")
    return long_all.drop_duplicates(subset="uid", keep="last").set_index("uid")["name"].to_dict()


def build_ratings_table(models: list, cat_levels: dict, playstyle_full: pd.DataFrame, age_table: pd.DataFrame,
                        bridged_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    name_lookup = build_name_lookup(bridged_df)
    rows = []
    for tour in ("atp", "wta"):
        log(f"  [{tour}] building active player pool...")
        field_uids = build_active_pool(bridged_df, tour, as_of, str(SCRATCH / "elo_raw_ratings_full.parquet"))
        log(f"  [{tour}] {len(field_uids)} players (active in last 52 weeks, capped to top {RATINGS_POOL_SIZE} by elo)")
        if not field_uids:
            continue

        snap = stu.build_snapshots({f"{tour}:ratings": {"date0": as_of, "field_uids": field_uids}}, df=bridged_df)
        snap = snap.get(f"{tour}:ratings", {})
        h2h = build_h2h_dict(set(field_uids))
        ptab = build_player_table(field_uids, tour, as_of, REFERENCE_SURFACE, snap, playstyle_full, age_table)

        # NOTE: deliberately NOT publishing per-surface Elo (ptab's "rs").
        # Checked before shipping it (Sabalenka: rs=1340 on clay despite being
        # a multi-Slam #1, rs=2940 on hard) - real, not a lookup bug, but the
        # frozen elo_baseline.py config uses blend=0.0, meaning `rs` NEVER
        # contributes to any actual prediction this project makes and has
        # never been validated or scrutinized for anything - publishing it as
        # a trustworthy per-surface rating would be misleading. elo_overall
        # (the real, validated number) and bt_rating (single reference
        # surface, no untested 3-way split) are what's actually trustworthy.

        best_of = 5 if tour == "atp" else 3
        log(f"  [{tour}] building BT round-robin ({len(field_uids)*(len(field_uids)-1):,} pairs)...")
        base, pairs_i, pairs_j = build_pairwise_features_base(field_uids, ptab, tour, REFERENCE_SURFACE, best_of,
                                                               "A", h2h, as_of)
        base["round"] = REFERENCE_ROUND
        X = prep_X_d(base, cat_levels)
        dmat = xgb.DMatrix(X, enable_categorical=True)
        preds = np.clip(np.mean([m.predict(dmat) for m in models], axis=0), 1e-9, 1 - 1e-9)
        L = pd.DataFrame({"i": pairs_i, "j": pairs_j, "logit": np.log(preds / (1 - preds))})
        Lp = L.pivot(index="i", columns="j", values="logit").reindex(index=field_uids, columns=field_uids).to_numpy()
        d = (Lp - Lp.T) / 2.0
        np.fill_diagonal(d, 0.0)
        bt_rating = {uid: float(np.nanmean(d[k, :])) for k, uid in enumerate(field_uids)}

        for uid in field_uids:
            rows.append({
                "player_id": uid, "player_name": name_lookup.get(uid, uid), "tour": tour,
                "elo_overall": round(float(ptab.loc[uid, "ro"]), 1),
                "bt_rating": round(bt_rating[uid], 2),
                "age": round(float(ptab.loc[uid, "age"]), 1),
            })
    return pd.DataFrame(rows)


def build_predictions_table() -> pd.DataFrame:
    """Reuses simulate_us_open_2026.py's already-saved predictions rather
    than re-simulating - that script is the source of truth for HOW these
    numbers are produced (model choice, bracket sim, n_sims); this just
    reshapes its output into the curated blog schema."""
    with open(SCRATCH / "us_open_2026_predictions.json", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for tour, data in raw.items():
        names, probs, bt, seed_map = data["names"], data["probs"], data["bt"], data["seed_map"]
        for uid, stage_probs in probs.items():
            rows.append({
                "tournament_id": "2026-usopen", "tournament_name": "US Open", "year": 2026, "tour": tour,
                "player_id": uid, "player_name": names.get(uid, uid),
                "seed": int(seed_map[uid]) if seed_map.get(uid) else None,
                "p_r64": round(stage_probs["R64"], 3), "p_r32": round(stage_probs["R32"], 3),
                "p_r16": round(stage_probs["R16"], 3), "p_qf": round(stage_probs["QF"], 3),
                "p_sf": round(stage_probs["SF"], 3), "p_f": round(stage_probs["F"], 3),
                "p_title": round(stage_probs["W"], 3),
                "bt_rating": round(bt.get(uid, 0.0), 2),
                "source": "locked",   # genuinely walk-forward: only uses data available as of date0
            })
    return pd.DataFrame(rows).sort_values(["tour", "p_title"], ascending=[True, False])


def main() -> None:
    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now(timezone.utc).isoformat()

    log("training/loading model...")
    models, cat_levels, playstyle_full = train_bagged_model()

    log("building bridged archive (real archive + tennismylife gap-fill)...")
    bridged_raw = bridge.load_matches_with_bridge(hm)
    bridged_df, _ = hm.prepare(bridged_raw)
    as_of = bridged_df["date"].max()
    log(f"  ratings as-of: {as_of.date()}")

    log("loading age table...")
    age_table = load_age_table()

    log("\n=== building ratings.parquet ===")
    ratings = build_ratings_table(models, cat_levels, playstyle_full, age_table, bridged_df, as_of)
    ratings["updated_at"] = updated_at
    ratings_path = BLOG_DIR / "ratings.parquet"
    ratings.to_parquet(ratings_path)
    log(f"wrote {ratings_path} ({len(ratings):,} rows)")

    log("\n=== building predictions.parquet ===")
    predictions = build_predictions_table()
    predictions["updated_at"] = updated_at
    predictions_path = BLOG_DIR / "predictions.parquet"
    predictions.to_parquet(predictions_path)
    log(f"wrote {predictions_path} ({len(predictions):,} rows)")

    log("\n=== validate outputs (torp convention: required files must exist and be non-trivial) ===")
    for f, min_rows in ((ratings_path, 50), (predictions_path, 200)):
        if not f.exists():
            raise SystemExit(f"REQUIRED output {f} was not produced")
        n = len(pd.read_parquet(f))
        if n < min_rows:
            raise SystemExit(f"REQUIRED output {f} suspiciously small ({n} rows, expected >={min_rows})")
        log(f"  {f.name}: {n:,} rows OK")


if __name__ == "__main__":
    main()
