"""Pete's question: can we get a single scalar player rating out of the
match-level XGBoost model (train_best_model.py), the way Elo and the
hierarchical model naturally produce one? XGBoost only predicts PAIRWISE
win probabilities, no built-in scalar rating - but a full round-robin of
predictions among a player pool can be collapsed into one via Bradley-Terry:
if the model were perfectly BT-consistent, logit(P(i beats j)) = r_i - r_j
for every pair, so each player's rating is recoverable as the row-mean of
the (antisymmetrized) pairwise log-odds matrix - the exact least-squares
solution when every pair is compared once (derived by hand before writing
this, not assumed).

Trains train_best_model.py's real bagged model ONCE (reusing its functions,
not reimplementing them), then for each of ticket 29's 16 Slam fields, builds
the full round-robin feature matrix and predicts every ordered pair, so the
result is directly comparable to hier_vs_seed_elo_ranking.py's seed/elo/hier
Spearman numbers on the exact same 16-tournament pooled set.

FEATURE FIDELITY per pair (i implicitly "side A", j "side B" - no need for
merge_hier's winner/loser hash-symmetrization since there's no real label to
leak here):
  - elo_overall_diff, elo_surface_diff, elo_logit_pwinner: EXACT, same
    elo_raw_ratings_full.parquet lookup run_one_tournament() uses (surface-
    specific rs uses each player's latest row on the TOURNAMENT'S surface).
  - hier_dmean_diff/hier_dvar_diff/hier_dvar_sum/hier_logit_pwinner: EXACT,
    reuses simulate_tournament_updating's own pre-tournament state snapshots
    and predict_p1_p2/p_match (the same functions the in-sim-update
    simulator uses for its own predictions).
  - weeks_since_last_diff: EXACT, from each player's latest known match date.
  - serve_form_diff_targeted/return_form_diff_targeted: EXACT, each player's
    latest known EWM value before date0 (same targeted mask as training).
  - h2h_diff: EXACT, real prior head-to-head wins between the two specific
    players before date0.
  - age_diff: APPROXIMATE - each player's last known age, aged forward
    linearly to date0.
  - n_prior_diff: PROXY - hierarchical model's own n_matches count (not
    identical to diag_layoff's n_prior_w/l definition, but the same kind of
    quantity - career match volume).
  - fatigue_diff: FIXED at 0 - neutral "first match of a fresh event", the
    correct value for a hypothetical round-robin pairing.
  - surface/tour/best_of/tourney_level/round: FIXED per tournament to its
    REAL context (the tournament's real surface, tour, best_of, "G") except
    round, fixed at "R32" for every pair (a neutral, common round) since
    round has no natural single value across a whole round-robin.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
os.environ.setdefault("DSV_RW_VAR_PER_WEEK", "0.005")
os.environ.setdefault("DSV_SURFACE_VAR_SCALE", "1.0")
os.environ.setdefault("DSV_OU_THETA_PER_WEEK", "0.00001")
os.environ.setdefault("DSV_MAX_SHIFT", "0.125")
os.environ.setdefault("DSV_PHI_MAX", "4.0")
os.environ.setdefault("DSV_LAYOFF_C", "0.06")
os.environ.setdefault("DSV_REGIME_B_N", "0.25")
os.environ.setdefault("DSV_TIER_WEIGHT", "1.0")
os.environ.setdefault("DSV_CLIP_REINFLATE_K", "0.5")

import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

import hierarchical_model as hm
import train_best_model as tbm
from simulate_tournament import DD, STAGE_ORDER, load_tournament_field
import simulate_tournament_updating as stu

TOURNAMENTS = stu.TOURNAMENTS
SCRATCH = DD / "tune_scratch"
# _v2_setgame: separate cache dir from the original match-only model - the
# cache check only verifies file EXISTENCE, not that the feature schema
# matches, so reusing the old path would silently load a model trained on
# the wrong (smaller) feature set once XGB_NUMERIC_D below is added.
MODEL_CACHE_DIR = SCRATCH / "xgb_bagged_model_cache_v2_setgame"

# Ticket 29 CI result (2026-08-29): match+set+game beat match-only with a
# resolved CI on WTA (-0.00170, CI [-0.00321,-0.00006]); ALL and ATP were
# unresolved (CI spans zero) but the ALL point estimate still favoured this
# arm (-0.00058) and Pete's call was to ship it everywhere rather than
# maintain two separate per-tour models - see issue 29's doc for the full
# CI table before assuming this is settled for ATP specifically.
XGB_NUMERIC_D = tbm.XGB_NUMERIC + ["set_elo_diff", "set_elo_logit_pwinner", "game_elo_diff", "game_elo_logit_pwinner"]


def log(m: str) -> None:
    print(m, flush=True)


def merge_granularity_elo(feat: pd.DataFrame, path: str, prefix: str) -> pd.DataFrame:
    """Shared with train_xgb_granularity_variants.py's own copy (kept
    duplicated deliberately - that script is a one-off experiment, this one
    is the live pipeline, and they shouldn't have to stay in lockstep)."""
    se = pd.read_parquet(path)[
        ["date", "tour", "winner_uid", "loser_uid", f"{prefix}_elo_winner", f"{prefix}_elo_loser",
        f"p_winner_{prefix}_elo"]
    ].drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"])
    j = feat.merge(se, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    miss = j[[f"{prefix}_elo_winner", f"{prefix}_elo_loser", f"p_winner_{prefix}_elo"]].isna().any(axis=1)
    log(f"  {len(j):,} -> {int((~miss).sum()):,} after {prefix}-Elo merge ({miss.mean():.4%} missing, dropped)")
    j = j[~miss].reset_index(drop=True)

    y = j["y"].values.astype(int)
    ew, el = j[f"{prefix}_elo_winner"].values.astype(float), j[f"{prefix}_elo_loser"].values.astype(float)
    j[f"{prefix}_elo_diff"] = np.where(y == 1, ew - el, el - ew)
    p_se = j[f"p_winner_{prefix}_elo"].values.astype(float)
    p_sideA = np.clip(np.where(y == 1, p_se, 1 - p_se), 1e-6, 1 - 1e-6)
    j[f"{prefix}_elo_logit_pwinner"] = np.log(p_sideA / (1 - p_sideA))
    return j.drop(columns=[f"{prefix}_elo_winner", f"{prefix}_elo_loser", f"p_winner_{prefix}_elo"])


def prep_X_d(df: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    """Same as train_best_model.prep_X but selecting XGB_NUMERIC_D (match +
    set + game Elo) instead of tbm.XGB_NUMERIC - tbm.prep_X can't be reused
    as-is since it hardcodes the original (smaller) column list."""
    X = df[XGB_NUMERIC_D].copy()
    for c in tbm.XGB_CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def train_bagged_model(use_cache: bool = True):
    """Trains train_best_model.py's real bagged model - the expensive part
    (elo/hier/playstyle feature builds + 5x2528-round training, ~4min) that
    every script in this family (BT rating, tournament-sim predictor, any
    future CI/analysis script) needs identically. Cached to disk since
    TRAIN_YEARS/hyperparameters are frozen for this whole ticket - a second
    script run within the same session norm shouldn't repay this cost."""
    cat_levels_path = MODEL_CACHE_DIR / "cat_levels.pkl"
    model_paths = [MODEL_CACHE_DIR / f"seed{seed}.json" for seed in range(tbm.N_BAGS)]
    if use_cache and cat_levels_path.exists() and all(p.exists() for p in model_paths):
        log("=== loading cached bagged XGBoost model (skip retrain) ===")
        with open(cat_levels_path, "rb") as f:
            cat_levels = pickle.load(f)   # nosec: our own cache, written moments/days ago by this same script
        models = []
        for p in model_paths:
            m = xgb.Booster()
            m.load_model(str(p))
            models.append(m)
        playstyle_full = tbm.build_playstyle_features()   # cheap (~seconds), needed by every caller regardless
        return models, cat_levels, playstyle_full

    log("=== training the real bagged XGBoost model (train_best_model.py's pipeline + set/game Elo, ticket 29) ===")
    train_feat = tbm.build_elo_features(tbm.TRAIN_YEARS)
    train_hier_path = tbm.regen_hier_state(tbm.TRAIN_YEARS, "train_canonical")
    train = tbm.merge_hier(train_feat, train_hier_path)
    playstyle_full = tbm.build_playstyle_features()
    train = tbm.merge_playstyle(train, playstyle_full)
    train = merge_granularity_elo(train, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    train = merge_granularity_elo(train, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")
    holdout = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(holdout[c].dropna().unique()))
                  for c in tbm.XGB_CATEGORICAL}
    X_tr = prep_X_d(train, cat_levels)
    y_tr = train["y"].values.astype(int)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    models = []
    for seed in range(tbm.N_BAGS):
        m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
        models.append(m)
        log(f"  seed {seed} done")

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for m, p in zip(models, model_paths):
        m.save_model(str(p))
    with open(cat_levels_path, "wb") as f:
        pickle.dump(cat_levels, f)
    log(f"  cached model + cat_levels -> {MODEL_CACHE_DIR}")
    return models, cat_levels, playstyle_full


def load_age_table() -> pd.DataFrame:
    """Long (uid, date, age) table across the whole archive, both tours -
    only source of age in this codebase (not in the Elo/hier parquets)."""
    years = list(range(1995, 2027))   # range() stop is exclusive - 2027 needed to include the 2026 file itself
    frames = []
    for tour in ("atp", "wta"):
        for y in years:
            p = DD / "raw/github" / tour / f"{tour}_matches_{y}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, low_memory=False, usecols=lambda c: c in (
                "tourney_date", "winner_id", "loser_id", "winner_age", "loser_age"))
            if not len(df):
                continue
            df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
            df["winner_uid"] = tour + ":" + df["winner_id"].astype("Int64").astype(str)
            df["loser_uid"] = tour + ":" + df["loser_id"].astype("Int64").astype(str)
            frames.append(df[["date", "winner_uid", "winner_age"]].rename(
                columns={"winner_uid": "uid", "winner_age": "age"}))
            frames.append(df[["date", "loser_uid", "loser_age"]].rename(
                columns={"loser_uid": "uid", "loser_age": "age"}))
    long_age = pd.concat(frames, ignore_index=True)
    return long_age[long_age["age"].notna() & long_age["date"].notna()].sort_values(["uid", "date"])


def build_h2h_dict(field_uids_all: set[str]) -> dict[tuple[str, str], list]:
    """Real head-to-head history restricted to the union of all 16 fields
    (full-archive scan would be wasteful - only pool members can ever be
    queried)."""
    h2h: dict[tuple[str, str], list] = {}
    for tour in ("atp", "wta"):
        elo = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet")
        elo = elo[(elo["tour"] == tour) & (elo["level_file"] == "tour")]
        elo = elo[elo["winner_uid"].isin(field_uids_all) & elo["loser_uid"].isin(field_uids_all)]
        for row in elo.itertuples():
            h2h.setdefault((row.winner_uid, row.loser_uid), []).append(row.date)
    return h2h


def h2h_wins_before(h2h: dict, a: str, b: str, date0) -> tuple[int, int]:
    a_wins = sum(1 for d in h2h.get((a, b), []) if d < date0)
    b_wins = sum(1 for d in h2h.get((b, a), []) if d < date0)
    return a_wins, b_wins


def latest_age_at(age_table: pd.DataFrame, uid: str, date0) -> float:
    sub = age_table[(age_table["uid"] == uid) & (age_table["date"] < date0)]
    if not len(sub):
        return 24.0   # archive-wide rough median tour debut age; only hit for a player with zero prior rows
    last = sub.iloc[-1]
    years_elapsed = (date0 - last["date"]).days / 365.25
    return float(last["age"]) + years_elapsed


def _latest_rating_lookup(path: str, tour: str, date0, field_uids: list[str],
                          winner_col: str, loser_col: str) -> pd.DataFrame:
    """Generic 'each player's latest pre-date0 rating' lookup - same pattern
    elo_overall_diff's own ro lookup below uses, reused for set/game Elo
    (both raw-ratings parquets share this date/tour/winner_uid/loser_uid
    schema, see set_elo_raw_for_xgb.py / game_elo_raw_for_xgb.py)."""
    r = pd.read_parquet(path)
    r = r[(r["tour"] == tour) & (r["level_file"] == "tour") & (r["date"] < date0)]
    long_w = r[["date", "winner_uid", winner_col]].rename(columns={"winner_uid": "uid", winner_col: "rating"})
    long_l = r[["date", "loser_uid", loser_col]].rename(columns={"loser_uid": "uid", loser_col: "rating"})
    long_all = pd.concat([long_w, long_l], ignore_index=True)
    long_all = long_all[long_all["uid"].isin(field_uids)].sort_values("date")
    return long_all.drop_duplicates(subset="uid", keep="last").set_index("uid")


def build_player_table(field_uids: list[str], tour: str, date0, surface: str,
                       hier_snapshot: dict, playstyle_full: pd.DataFrame, age_table: pd.DataFrame) -> pd.DataFrame:
    elo = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet")
    elo_t = elo[(elo["tour"] == tour) & (elo["level_file"] == "tour") & (elo["date"] < date0)]

    long_w = elo_t[["date", "winner_uid", "ro_winner"]].rename(columns={"winner_uid": "uid", "ro_winner": "ro"})
    long_l = elo_t[["date", "loser_uid", "ro_loser"]].rename(columns={"loser_uid": "uid", "ro_loser": "ro"})
    long_ro = pd.concat([long_w, long_l], ignore_index=True)
    long_ro = long_ro[long_ro["uid"].isin(field_uids)].sort_values("date")
    latest_ro = long_ro.drop_duplicates(subset="uid", keep="last").set_index("uid")

    latest_set_elo = _latest_rating_lookup(str(SCRATCH / "set_elo_raw_ratings_full.parquet"), tour, date0,
                                           field_uids, "set_elo_winner", "set_elo_loser")
    latest_game_elo = _latest_rating_lookup(str(SCRATCH / "game_elo_raw_ratings_full.parquet"), tour, date0,
                                            field_uids, "game_elo_winner", "game_elo_loser")

    elo_surf = elo_t[elo_t["surface"] == surface]
    long_w2 = elo_surf[["date", "winner_uid", "rs_winner"]].rename(columns={"winner_uid": "uid", "rs_winner": "rs"})
    long_l2 = elo_surf[["date", "loser_uid", "rs_loser"]].rename(columns={"loser_uid": "uid", "rs_loser": "rs"})
    long_rs = pd.concat([long_w2, long_l2], ignore_index=True)
    long_rs = long_rs[long_rs["uid"].isin(field_uids)].sort_values("date")
    latest_rs = long_rs.drop_duplicates(subset="uid", keep="last").set_index("uid")

    ps_t = playstyle_full[playstyle_full["tour"] == tour]
    long_w3 = ps_t[["date", "winner_uid", "serve_form_w", "return_form_w"]].rename(
        columns={"winner_uid": "uid", "serve_form_w": "serve_form", "return_form_w": "return_form"})
    long_l3 = ps_t[["date", "loser_uid", "serve_form_l", "return_form_l"]].rename(
        columns={"loser_uid": "uid", "serve_form_l": "serve_form", "return_form_l": "return_form"})
    long_form = pd.concat([long_w3, long_l3], ignore_index=True)
    long_form = long_form[(long_form["uid"].isin(field_uids)) & (long_form["date"] < date0)].sort_values("date")
    latest_form = long_form.drop_duplicates(subset="uid", keep="last").set_index("uid")

    rows = []
    for uid in field_uids:
        ro = float(latest_ro.loc[uid, "ro"]) if uid in latest_ro.index else 1500.0
        rs = float(latest_rs.loc[uid, "rs"]) if uid in latest_rs.index else 1500.0
        set_elo = float(latest_set_elo.loc[uid, "rating"]) if uid in latest_set_elo.index else 1500.0
        game_elo = float(latest_game_elo.loc[uid, "rating"]) if uid in latest_game_elo.index else 1500.0
        last_date = latest_ro.loc[uid, "date"] if uid in latest_ro.index else None
        weeks_since_last = (date0 - last_date).days / 7.0 if last_date is not None else 52.0
        state = hier_snapshot.get(uid, hm.PlayerState())
        age = latest_age_at(age_table, uid, date0)
        serve_form = float(latest_form.loc[uid, "serve_form"]) if uid in latest_form.index else np.nan
        return_form = float(latest_form.loc[uid, "return_form"]) if uid in latest_form.index else np.nan
        rows.append({"uid": uid, "ro": ro, "rs": rs, "set_elo": set_elo, "game_elo": game_elo,
                    "weeks_since_last": weeks_since_last, "n_matches": state.n_matches, "age": age,
                    "serve_form": serve_form, "return_form": return_form, "state": state})
    return pd.DataFrame(rows).set_index("uid")


def build_pairwise_features_base(field_uids: list[str], ptab: pd.DataFrame, tour: str, surface: str, best_of: int,
                                 tourney_level: str, h2h: dict, date0) -> tuple[pd.DataFrame, list, list]:
    """The full round-robin feature matrix for one tournament's field, EXCEPT
    the `round` categorical (context-dependent - callers decide: a single
    neutral value for a scalar BT rating, or one pass per real round for a
    bracket simulator). Shared by xgb_bt_rating.py's BT extraction and
    simulate_tournament_xgb.py's bracket simulator - was duplicated between
    them until this refactor; keep it that way, don't re-fork it."""
    pairs_i, pairs_j = [], []
    recs = []
    atp_not_grass = (tour == "atp") and (surface != "Grass")
    # Plain dicts instead of ptab.loc[] inside the 128x127 loop: the Series
    # indexing was 6.6M calls and 11% of a live run (ticket 40). Values are
    # the same objects the Series held; verified identical by
    # tune_scratch/verify_perf_live_path.py.
    rows = {uid: dict(zip(ptab.columns, vals)) for uid, vals in zip(ptab.index, ptab.itertuples(index=False, name=None))}
    for i in field_uids:
        pi = rows[i]
        for j in field_uids:
            if i == j:
                continue
            pj = rows[j]
            p1, p2 = stu.predict_p1_p2(pi["state"], pj["state"], surface)
            p1c, p2c = min(max(p1, 1e-9), 1 - 1e-9), min(max(p2, 1e-9), 1 - 1e-9)
            hier_dmean_diff = np.log(p1c / (1 - p1c)) - np.log(p2c / (1 - p2c))
            k_i, vk_i = pi["state"].get_kappa(surface)
            lm_i, vlm_i = pi["state"].get_lam(surface)
            k_j, vk_j = pj["state"].get_kappa(surface)
            lm_j, vlm_j = pj["state"].get_lam(surface)
            v1 = (pi["state"].v_a + vk_i) + (pj["state"].v_b + vlm_j)
            v2 = (pj["state"].v_a + vk_j) + (pi["state"].v_b + vlm_i)
            hier_dvar_diff = v1 - v2
            hier_dvar_sum = v1 + v2
            p_hier = hm.p_match(p1, p2, best_of)
            p_hier_c = min(max(p_hier, 1e-9), 1 - 1e-9)
            hier_logit = np.log(p_hier_c / (1 - p_hier_c))

            h2h_i, h2h_j = h2h_wins_before(h2h, i, j, date0)

            serve_diff = 0.0 if atp_not_grass else (pi["serve_form"] - pj["serve_form"])
            return_diff = 0.0 if atp_not_grass else (pi["return_form"] - pj["return_form"])
            if pd.isna(serve_diff):
                serve_diff = 0.0
            if pd.isna(return_diff):
                return_diff = 0.0

            # elo_logit_pwinner: the standard Elo win-prob formula in logit
            # form - p=1/(1+10^(-d/400)) => logit(p)=(d/400)*ln(10) - exact,
            # not a guess (blend=0.0 in the frozen config means rs never
            # actually contributes to elo_baseline.py's own win probability,
            # only the overall diff does, same as simulate_tournament.py's
            # own frozen-Elo p_a formula).
            # set/game-elo logit form: same log-odds-from-logistic-diff
            # identity as elo_logit_pwinner above, and the exact SCALE=400
            # base used when these ratings were fit
            # (set_elo_raw_for_xgb.py / game_elo_raw_for_xgb.py).
            elo_logit_pwinner = (pi["ro"] - pj["ro"]) / 400.0 * np.log(10.0)
            set_elo_logit_pwinner = (pi["set_elo"] - pj["set_elo"]) / 400.0 * np.log(10.0)
            game_elo_logit_pwinner = (pi["game_elo"] - pj["game_elo"]) / 400.0 * np.log(10.0)
            recs.append({
                "elo_overall_diff": pi["ro"] - pj["ro"], "elo_surface_diff": pi["rs"] - pj["rs"],
                "elo_logit_pwinner": elo_logit_pwinner, "hier_dmean_diff": hier_dmean_diff,
                "hier_dvar_diff": hier_dvar_diff, "hier_dvar_sum": hier_dvar_sum,
                "weeks_since_last_diff": pi["weeks_since_last"] - pj["weeks_since_last"],
                "n_prior_diff": pi["n_matches"] - pj["n_matches"], "h2h_diff": h2h_i - h2h_j,
                "best_of": best_of, "age_diff": pi["age"] - pj["age"], "fatigue_diff": 0.0,
                "hier_logit_pwinner": hier_logit, "serve_form_diff_targeted": serve_diff,
                "return_form_diff_targeted": return_diff,
                "set_elo_diff": pi["set_elo"] - pj["set_elo"], "set_elo_logit_pwinner": set_elo_logit_pwinner,
                "game_elo_diff": pi["game_elo"] - pj["game_elo"], "game_elo_logit_pwinner": game_elo_logit_pwinner,
                "surface": surface, "tourney_level": tourney_level, "tour": tour,
            })
            pairs_i.append(i)
            pairs_j.append(j)

    return pd.DataFrame(recs), pairs_i, pairs_j


def build_pairwise_X(field_uids: list[str], ptab: pd.DataFrame, tour: str, surface: str, best_of: int,
                     tourney_level: str, h2h: dict, date0, cat_levels: dict) -> tuple[pd.DataFrame, list, list]:
    Xdf, pairs_i, pairs_j = build_pairwise_features_base(field_uids, ptab, tour, surface, best_of, tourney_level,
                                                          h2h, date0)
    Xdf["round"] = "R32"   # neutral, common round - no natural single value for a scalar BT rating
    return Xdf, pairs_i, pairs_j


def main() -> None:
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
    log(f"loading age table (full archive, {len(all_field_uids)} pool players)...")
    age_table = load_age_table()
    log("building h2h dict...")
    h2h = build_h2h_dict(all_field_uids)

    rows_out = []
    for tour, tourney_id, year in TOURNAMENTS:
        key = f"{tour}:{tourney_id}"
        meta = tournaments_meta[key]
        field_uids = meta["field_uids"]
        surface = meta["surface"]
        date0 = meta["date0"]
        real_stage = meta["real_stage"]
        seed_map = meta["seed_map"]
        best_of = 5 if tour == "atp" else 3
        snap = snapshots.get(key, {})

        log(f"  [{key}] building player table + round robin ({len(field_uids)} players, "
            f"{len(field_uids)*(len(field_uids)-1):,} pairs)...")
        ptab = build_player_table(field_uids, tour, date0, surface, snap, playstyle_full, age_table)
        Xdf, pairs_i, pairs_j = build_pairwise_X(field_uids, ptab, tour, surface, best_of, "G", h2h, date0, cat_levels)

        X = prep_X_d(Xdf, cat_levels)
        dmat = xgb.DMatrix(X, enable_categorical=True)
        preds = np.mean([m.predict(dmat) for m in models], axis=0)

        L = pd.DataFrame({"i": pairs_i, "j": pairs_j, "p": np.clip(preds, 1e-9, 1 - 1e-9)})
        L["logit"] = np.log(L["p"] / (1 - L["p"]))
        Lp = L.pivot(index="i", columns="j", values="logit").reindex(index=field_uids, columns=field_uids)
        Larr = Lp.to_numpy()
        n = len(field_uids)
        asym = np.nanmean(np.abs(Larr + Larr.T))
        d = (Larr - Larr.T) / 2.0
        np.fill_diagonal(d, 0.0)
        bt_rating = {uid: float(np.nanmean(d[k, :])) for k, uid in enumerate(field_uids)}

        for uid in field_uids:
            rows_out.append({"tour": tour, "tourney_id": tourney_id, "uid": uid,
                            "seed": seed_map.get(uid, np.nan), "bt": bt_rating[uid],
                            "depth": STAGE_ORDER.index(real_stage[uid]), "asym": asym})

    df = pd.DataFrame(rows_out)
    out_path = SCRATCH / "xgb_bt_ratings_16tournaments.parquet"
    df.to_parquet(out_path)
    log(f"\nsaved {len(df):,} rows -> {out_path}")
    log(f"mean |L_ij + L_ji| across all pairs/tournaments (BT-consistency check, 0=perfectly consistent): "
        f"{df.groupby('tourney_id')['asym'].first().mean():.4f}")

    def spearman(x, y):
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])

    rho_bt = spearman(df["bt"].to_numpy(), df["depth"].to_numpy())
    log(f"\nSpearman(xgb-BT rating, real depth), n={len(df)}: {rho_bt:.4f}")


if __name__ == "__main__":
    main()
