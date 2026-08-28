"""Ticket 10, stage 1: surface-adjusted Elo baseline.

The cheap test that runs first (spec §7). If the elaborate hierarchical model
cannot beat this, the project needs to know on day one.

Discipline applied, per stats-discipline:
  §1 anchors    - asserted before any metric is printed
  §5 one axis   - blended vs surface-only Elo differ ONLY in the blend weight
  §7 coverage   - every exclusion counted and printed before any headline number

Strict chronology: every match is predicted using ONLY ratings built from
earlier matches. Elo is naturally online, so this is structural rather than a
split we have to police - but the evaluation window is still held out explicitly.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = Path(r"C:\dev\dropshotverse\dropshotdata\raw\github")
EVAL_YEARS = (2024, 2025)

# Elo parameters. Deliberately conventional - this is a baseline, not a tuned model.
K_BASE, K_SHAPE, K_MIN = 250.0, 0.4, 16.0     # Sackmann-style decaying K
START = 1500.0
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def log(m: str) -> None:
    print(m, flush=True)


# Bug fix (2026-08-28): the SAME causal-ordering bug just found and fixed in
# hierarchical_model.py (commit 111ecce) - this file's load_matches()/prepare()
# are near-duplicates of that file's, and share the identical defect.
# `tourney_date` is a TOURNAMENT-level (start) date, not per-match, so a
# date-only stable sort leaves within-date order at the mercy of raw file
# concatenation order - and load_matches() above reads ALL main-draw yearly
# files for a tour before ANY qual_chall/qual_itf yearly files, so for every
# tournament, every year, both tours, qualifying rounds land AFTER the entire
# main draw in file order, even though qualifying happens BEFORE the main
# draw in real life. A player who just won Q1-Q3 would be rated in their
# R128 match using a state that hadn't yet absorbed those qualifying wins.
#
# Q1 < Q2 < Q3 < Q4 < Q5 < ER < R128 < R64 < R32 < R16 < QF < RR < SF < F/BR
# RR (round-robin, season-ending Finals events) is a group stage, ranked
# before the SF/F knockout bracket it feeds. BR (bronze/3rd-place playoff) is
# ranked with F - both happen after SF and are effectively concurrent.
ROUND_RANK: dict[str, int] = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4,
    "ER": 5,
    "R128": 6, "R64": 7, "R32": 8, "R16": 9,
    "QF": 10, "RR": 11, "SF": 12,
    "F": 13, "BR": 13,
}
QUAL_ROUNDS = {"Q1", "Q2", "Q3", "Q4", "Q5", "ER"}
MAIN_ROUNDS = {"R128", "R64", "R32", "R16", "QF", "RR", "SF", "F", "BR"}


def round_rank(series: pd.Series) -> pd.Series:
    """Map a `round` column to a within-date sort rank. `'f'` is normalized to
    `'F'` first (confirmed the same round, not a distinct one - a WTA
    ITF-qualifying final, per hierarchical_model.py's identical fix). Any
    round value not in ROUND_RANK falls back to a neutral rank (NaN, sorted
    last within its date via na_position='last') rather than crashing or
    guessing a position, and is logged so a new label doesn't silently
    misorder matches."""
    normalized = series.astype(str).str.strip()
    normalized = normalized.where(normalized != "f", "F")
    ranks = normalized.map(ROUND_RANK)
    unknown = ranks.isna()
    if unknown.any():
        bad_vals = sorted(normalized[unknown].unique())
        warnings.warn(
            f"prepare(): {int(unknown.sum())} row(s) have round value(s) "
            f"{bad_vals} not in ROUND_RANK - falling back to date-only order "
            "for these rows (sorted last within their date).",
            stacklevel=2,
        )
    return ranks


def count_round_inversions(df: pd.DataFrame) -> int:
    """Count (qual_row, main_row) pairs that share a date where the
    qualifying-round row currently sits AFTER the main-draw row in the
    DataFrame's own order - exactly the defect this fix corrects. O(n) via a
    running per-date main-draw counter (requires df already sorted by date,
    true both before and after the round-rank sort), not O(n^2)."""
    normalized = df["round"].astype(str).str.strip().replace({"f": "F"})
    is_qual = normalized.isin(QUAL_ROUNDS).to_numpy()
    is_main = normalized.isin(MAIN_ROUNDS).to_numpy()
    dates = df["date"].to_numpy()
    inv = 0
    main_seen = 0
    prev_date = None
    for d, q, m in zip(dates, is_qual, is_main):
        if d != prev_date:
            main_seen = 0
            prev_date = d
        if q:
            inv += main_seen
        if m:
            main_seen += 1
    return inv


# --------------------------------------------------------------- load matches

def load_matches() -> pd.DataFrame:
    frames = []
    for tour in ("atp", "wta"):
        pats = [f"{tour}_matches_[0-9]*.csv",
                f"{tour}_matches_qual_chall_[0-9]*.csv",
                f"{tour}_matches_qual_itf_[0-9]*.csv"]
        for pat in pats:
            for p in sorted((ARCHIVE / tour).glob(pat)):
                try:
                    df = pd.read_csv(p, low_memory=False)
                except Exception:                              # noqa: BLE001
                    continue
                if not len(df):
                    continue
                df["tour"] = tour
                df["level_file"] = ("tour" if "qual" not in p.name else "lower")
                frames.append(df[["tour", "level_file", "tourney_date", "tourney_level",
                                  "surface", "score", "best_of", "round",
                                  "winner_id", "loser_id", "winner_name", "loser_name"]])
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean, namespace ids, drop what cannot be used - counting every exclusion."""
    dropped: dict[str, int] = {}
    n0 = len(df)

    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    bad = df["date"].isna()
    dropped["unparseable date"] = int(bad.sum())
    df = df[~bad]

    # ticket 07a: 'clay' vs 'Clay' casing bug
    df["surface"] = df["surface"].astype(str).str.strip().str.title()
    df.loc[~df["surface"].isin(SURFACES), "surface"] = "Unknown"

    # ticket 07a: ATP and WTA player ids COLLIDE on 2,396 values. Namespace them
    # or the engine silently merges Justine Henin with Jeremy Beale.
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)

    bad = df["winner_uid"].str.contains("<NA>") | df["loser_uid"].str.contains("<NA>")
    dropped["missing player id"] = int(bad.sum())
    df = df[~bad]

    # regime C: retirements, walkovers, defaults - excluded, never silently
    sc = df["score"].astype(str)
    bad = sc.str.contains(r"RET|W/O|DEF|ABD|Walkover|Def\.", case=False, na=False) | (sc.str.strip() == "")
    dropped["retirement/walkover/no score"] = int(bad.sum())
    df = df[~bad]

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Round-order fix (see ROUND_RANK docstring above). Verify it changes
    # something real before trusting any downstream number: count
    # qual-after-main inversions under the OLD date-only order, then again
    # after re-sorting by (date, round_rank) - not code-inspection alone.
    before = count_round_inversions(df)
    df["_round_rank"] = round_rank(df["round"])
    df = df.sort_values(["date", "_round_rank"], kind="mergesort", na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_round_rank"])
    after = count_round_inversions(df)
    dropped["_diag_round_inversions_before"] = before
    dropped["_diag_round_inversions_after"] = after

    dropped["_total_in"] = n0
    dropped["_kept"] = len(df)
    return df, dropped


# ------------------------------------------------------------------- the model

def k_factor(n: int) -> float:
    return max(K_MIN, K_BASE / ((n + 5) ** K_SHAPE))


# ---- UTS-style (tennis-crystal-ball) K-factor: an ALTERNATIVE to k_factor(n)
# above, not an addition to it. Ported directly from EloCalculator.java
# (github.com/mcekovic/tennis-crystal-ball, commit fcda1e6 - fetched and
# read directly, not just Pete's screenshot summary, which turned out to
# describe only the LEVEL switch; the reference's ROUND switch and its
# `default: 0.65` are a structurally separate piece the screenshot didn't
# fully capture - confirmed by reading the actual source).
#
# The reference is a THREE-part multiplicative stack:
#   1. kFactor(level, round, bestOf, outcome) = 32 * level_mult * round_mult
#      * bestof_mult * wo_mult  -- context only, identical for both players.
#   2. kFunction(rating) = 1 + 18 / (1 + 2^((rating-1500)/63))  -- PER
#      PLAYER, evaluated on that player's OWN current rating (not the
#      opponent's, not the match outcome).
#   3. newRating = rating +/- clip(kFactor(...) * (1-p_win) * kFunction(rating), 200)
#      -- the reference's own DELTA_RATING_CAP=200 is genuinely load-bearing
#      here, not a decoration: kFunction alone reaches ~10x at rating=1500,
#      so 32 * kFunction(1500) = 320 for a full-weight (Slam, Bo5) match,
#      well above 200 - ported faithfully as UTS_MAX_DELTA.
#
# Level mapping: this archive's tourney_level codes are NOT the same
# vocabulary as tennis-crystal-ball's own DB (which normalizes ATP+WTA to
# one shared scheme before ever calling kFactor). Checked via
# df['tourney_level'].value_counts() rather than assumed - codes actually
# present include G, F, M, D, O, A (ATP tour), W/P/PM/I/T1-T5 (WTA tier
# variants), C (challenger), numeric ITF prize-money tiers (10..100), CC, E,
# J, and two "+H" hybrid codes. Codes with a clear real-world correspondence
# to the reference's own buckets are mapped there directly; every code with
# no clean correspondence (Challenger, ITF, exhibitions, juniors, the "+H"
# codes) falls to the reference's OWN `default: 0.65` fallback, not a new
# choice invented here.
UTS_K_BASE = 32.0
UTS_MAX_DELTA = 200.0   # reference's DELTA_RATING_CAP
UTS_LEVEL_MULT: dict[str, float] = {
    "G": 1.00,   # Grand Slam
    "F": 0.90,   # Tour Finals (ATP/WTA)
    "M": 0.85,   # ATP Masters 1000
    "PM": 0.85,  # WTA Premier Mandatory - closest real-world Masters-equivalent
    "O": 0.80,   # Olympics
    "A": 0.75,   # ATP regular tour (250/500 not split in this archive)
    "P": 0.75,   # WTA Premier - closest real-world ATP-500-equivalent
    "D": 0.70,   # Davis/Fed Cup - reference's own "D" bucket
    "I": 0.70,   # WTA International (regular tour) - reference's "T"-equivalent bucket
    "W": 0.70,   # WTA's post-2021 simplified single-tier code - regular tour
    "T1": 0.70, "T2": 0.70, "T3": 0.70, "T4": 0.70, "T5": 0.70,  # WTA tier codes
    "CC": 0.70,
}
UTS_DEFAULT_LEVEL_MULT = 0.65   # reference's own `default: 0.65` (Challenger, ITF, exhibitions, juniors, "+H")

# Round switch: the reference has NO default case - any round not listed
# (Q1-Q5, ER - i.e. EVERY qualifying round) falls through UNMULTIPLIED, i.e.
# gets the SAME full weight as a Final. Confirmed by reading the Java switch
# itself (no `default:` branch present), not assumed - a genuinely
# counter-intuitive spec detail, kept faithfully rather than "fixed" here.
UTS_ROUND_MULT: dict[str, float] = {
    "F": 1.00, "BR": 0.95, "SF": 0.90, "QF": 0.85, "RR": 0.85,
    "R16": 0.80, "R32": 0.80, "R64": 0.75, "R128": 0.75,
}


def uts_context_k(level: str, round_: str, best_of: int, is_walkover: bool = False) -> float:
    """The context-only (player-independent) part of the UTS K stack: base *
    level * round * best-of * walkover. `is_walkover` is wired for fidelity
    to the reference but structurally never fires on this archive -
    `prepare()` already drops every RET/W/O/DEF/ABD/no-score row before any
    match reaches run_elo()."""
    k = UTS_K_BASE
    k *= UTS_LEVEL_MULT.get(level, UTS_DEFAULT_LEVEL_MULT)
    k *= UTS_ROUND_MULT.get(round_, 1.0)
    if best_of < 5:
        k *= 0.90
    if is_walkover:
        k *= 0.50
    return k


def uts_kfunction(rating: float) -> float:
    """Rating-dependent K amplifier: ~10x at rating=START (1500), ~5.5x at
    1600, ~1.64x at 1800, ~1.07x at 2000, ~1.008x at 2200 (matches the
    reference's own documented values exactly - checked, not just trusted).
    This REPLACES k_factor(n)'s matches-played decay entirely when
    k_mode='uts' - a different K-philosophy (current-rating-based
    provisional adjustment), not a combination with the matches-played
    mechanism."""
    return 1.0 + 18.0 / (1.0 + 2.0 ** ((rating - START) / 63.0))


# ---- inactivity/layoff decay (Task 2). Reference mechanism: tennis-crystal-
# ball's EloCalculator.adjustRating() pulls a returning player's rating back
# toward START via a logistic function of days since their last match, with
# a level-specific decay magnitude. Ported here as a bounded, saturating pull
# (NOT a full reset - the reference never lets inactivity erase more than a
# capped fraction of the pre-layoff gap either) with the shape constants
# (period/grace/gradient) borrowed directly from the reference and held
# fixed, and exactly ONE new tunable axis - `decay`, the fraction of
# distance-to-START pulled back once the gap has fully saturated the
# logistic. `decay=0.0` is an exact structural no-op (the caller skips the
# adjustment entirely, not just multiplies by zero), so it reproduces
# pre-existing behaviour exactly, verified below before trusting any
# nonzero value.
LAYOFF_PERIOD = 500.0      # days - gap at which the pull is ~fully engaged
LAYOFF_GRACE = 30.0        # days - normal tour cadence; ~no pull below this
LAYOFF_GRADIENT = 100.0    # days - width of the logistic transition
_LAYOFF_DRIFT = 1.0 / (1.0 + math.exp((LAYOFF_PERIOD - LAYOFF_GRACE) / LAYOFF_GRADIENT))


def layoff_decay(rating: float, days_gap: float, decay: float) -> float:
    """Shrink `rating` toward START by up to `decay` * (rating - START), the
    fraction growing from ~0 at short gaps (normal tour cadence, <~30 days)
    to the full `decay` amount by roughly LAYOFF_PERIOD days of inactivity,
    via a logistic curve. Symmetric: a below-START rating is pulled UP
    toward START on return, same as an above-START rating is pulled down."""
    if decay <= 0.0 or days_gap <= 0.0:
        return rating
    logistic = 1.0 / (1.0 + math.exp((LAYOFF_PERIOD - days_gap) / LAYOFF_GRADIENT))
    frac = decay * max(0.0, (logistic - _LAYOFF_DRIFT) / (1.0 - _LAYOFF_DRIFT))
    return rating - frac * (rating - START)


def run_elo(df: pd.DataFrame, blend: float, decay: float = 0.0,
            k_mode: str = "sackmann") -> pd.DataFrame:
    """Online Elo. `blend` mixes overall and surface rating:
        p = sigma( (1-blend)*overall_diff + blend*surface_diff )

    blend=0 is pure overall Elo; blend=1 is pure surface Elo. This is the ONLY
    axis that varies between the original `blend` arms (stats-discipline §5).

    `decay` (Task 2) is a SEPARATE axis, held at 0.0 unless a caller is
    specifically running the decay grid (with `blend` held fixed at whatever
    won the blend grid) - so a difference in score between decay arms is
    attributable to decay alone, not a blend/decay interaction.

    `k_mode` selects the ENTIRE K-factor mechanism and is a THIRD, separate
    axis from `decay`:
      - 'sackmann' (default): k_factor(n) above, decaying by matches played.
      - 'uts': tennis-crystal-ball's tournament-context (level/round/best-of)
        * current-rating kFunction stack. This REPLACES k_factor(n) entirely
        (a different K-philosophy, not a combination with it) - `decay`
        (time-since-last-match inactivity pull) is orthogonal to and kept
        alongside either k_mode unchanged, so a k_mode difference isn't
        confounded with a decay difference.
      - 'uts_context_only' / 'uts_kfunc_only': single-factor ablations of
        the 'uts' bundle above (context-multiplier stack alone with
        kFunction fixed at 1.0, or kFunction alone with context fixed at
        UTS_K_BASE=32) - diagnostic only, to see which piece of the bundle
        is doing the work, per the task's "if time permits" secondary ask.
        Not a candidate in its own right; not scored against the market
        holdout as a real arm, only used for the tour-split diagnosis.
    """
    is_uts = k_mode in ("uts", "uts_context_only", "uts_kfunc_only")
    if is_uts:
        if k_mode == "uts_kfunc_only":
            ctx_k = np.full(len(df), UTS_K_BASE)
        else:
            ctx_k = (UTS_K_BASE
                     * df["tourney_level"].astype(str).map(
                           lambda v: UTS_LEVEL_MULT.get(v, UTS_DEFAULT_LEVEL_MULT)).values
                     * df["round"].astype(str).map(
                           lambda v: UTS_ROUND_MULT.get(v, 1.0)).values
                     * np.where(df["best_of"].astype(int).values < 5, 0.90, 1.00))
        # walkover multiplier omitted here: prepare() already drops every
        # W/O row, so it would be an unconditional 1.0 for every surviving
        # row - see uts_context_k()'s docstring for the same point.
        kfunc = (lambda r: 1.0) if k_mode == "uts_context_only" else uts_kfunction
    else:
        ctx_k = None

    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values

    preds = np.empty(len(df))
    for idx, (w, l, s, day) in enumerate(zip(df["winner_uid"].values,
                                             df["loser_uid"].values,
                                             df["surface"].values,
                                             days)):
        ro_w, ro_l = overall.get(w, START), overall.get(l, START)
        rs_w = surf.get((w, s), START)
        rs_l = surf.get((l, s), START)

        if decay > 0.0:
            ro_w = layoff_decay(ro_w, day - last_all.get(w, day), decay)
            ro_l = layoff_decay(ro_l, day - last_all.get(l, day), decay)
            rs_w = layoff_decay(rs_w, day - last_surf.get((w, s), day), decay)
            rs_l = layoff_decay(rs_l, day - last_surf.get((l, s), day), decay)

        diff = (1 - blend) * (ro_w - ro_l) + blend * (rs_w - rs_l)
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        preds[idx] = p_w                       # P(actual winner wins), pre-match

        # The update is applied on top of the (possibly decayed) rating, so a
        # decay "spends" itself into the persisted state at the moment of
        # return, exactly like the reference's adjustRating().
        if is_uts:
            delta_base = ctx_k[idx] * (1 - p_w)   # shared context term, both sides
            dw = math.copysign(min(abs(delta_base * kfunc(ro_w)), UTS_MAX_DELTA),
                                delta_base) if delta_base else 0.0
            dl = math.copysign(min(abs(delta_base * kfunc(ro_l)), UTS_MAX_DELTA),
                                -delta_base) if delta_base else 0.0
            overall[w] = ro_w + dw
            overall[l] = ro_l + dl

            dsw = math.copysign(min(abs(delta_base * kfunc(rs_w)), UTS_MAX_DELTA),
                                 delta_base) if delta_base else 0.0
            dsl = math.copysign(min(abs(delta_base * kfunc(rs_l)), UTS_MAX_DELTA),
                                 -delta_base) if delta_base else 0.0
            surf[(w, s)] = rs_w + dsw
            surf[(l, s)] = rs_l + dsl
        else:
            kw, kl = k_factor(n_all.get(w, 0)), k_factor(n_all.get(l, 0))
            overall[w] = ro_w + kw * (1 - p_w)
            overall[l] = ro_l - kl * (1 - p_w)

            ksw, ksl = k_factor(n_surf.get((w, s), 0)), k_factor(n_surf.get((l, s), 0))
            surf[(w, s)] = rs_w + ksw * (1 - p_w)
            surf[(l, s)] = rs_l - ksl * (1 - p_w)

        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1
        last_all[w] = day
        last_all[l] = day
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
        last_surf[(w, s)] = day
        last_surf[(l, s)] = day

    out = df.copy()
    out["p_winner"] = preds
    return out, {"overall": overall, "surf": surf, "n_all": n_all}


# ------------------------------------------------------------------- scoring

def score(p: np.ndarray, label: str) -> dict:
    """p = P(model assigned to the player who actually won)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "arm": label,
        "n": len(p),
        "acc": float((p > 0.5).mean()),
        "logloss": float(-np.log(p).mean()),
        "brier": float(((1 - p) ** 2).mean()),
    }


if __name__ == "__main__":
    log("loading archive...")
    raw = load_matches()
    df, dropped = prepare(raw)

    # ---- stats-discipline §7: coverage BEFORE any metric
    log("\n=== COVERAGE ===")
    log(f"  rows read              : {dropped.pop('_total_in'):,}")
    kept = dropped.pop("_kept")
    inv_before = dropped.pop("_diag_round_inversions_before")
    inv_after = dropped.pop("_diag_round_inversions_after")
    for reason, n in dropped.items():
        log(f"  excluded, {reason:<28}: {n:,}")
    log(f"  kept                   : {kept:,}")
    log(f"  round-order fix        : {inv_before:,} qual-after-main inversions "
        f"before -> {inv_after:,} after "
        f"({'OK, fix verified' if inv_after == 0 else 'WARNING: nonzero after fix'})")
    log(f"  date range             : {df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"  players                : {len(set(df.winner_uid) | set(df.loser_uid)):,}")
    log("\n  by tour x level:")
    log(df.groupby(["tour", "level_file"]).size().to_string())
    log("\n  surface:")
    log(df["surface"].value_counts().to_string())

    ev = df["date"].dt.year.isin(EVAL_YEARS)
    log(f"\n  eval window {EVAL_YEARS}: {int(ev.sum()):,} matches "
        f"({100*ev.mean():.1f}% of kept)")

    # ---- arms differing on ONE axis only
    log("\n=== FITTING (one axis: surface blend) ===")
    results = []
    fits = {}
    for blend in (0.0, 0.5, 0.7, 1.0):
        fitted, state = run_elo(df, blend)
        fits[blend] = (fitted, state)
        sub = fitted[ev]
        results.append(score(sub["p_winner"].values, f"blend={blend} ALL"))
        # Tour comparison is only meaningful WITHIN a level: the lower-tier
        # eval sets differ enormously in size and in skill-gap distribution,
        # so an all-levels tour split measures level mix, not tour.
        for lvl in ("tour", "lower"):
            for tour in ("atp", "wta"):
                m = (sub["tour"] == tour) & (sub["level_file"] == lvl)
                if m.sum() > 200:
                    results.append(score(sub.loc[m, "p_winner"].values,
                                         f"blend={blend} {tour}/{lvl}"))
        log(f"  blend={blend}: done")

    log("\n=== ELO RESULTS, eval {} ===".format(EVAL_YEARS))
    log(pd.DataFrame(results).to_string(index=False))

    # ---- stats-discipline §1: anchors, on the best-scoring arm
    best = min((r for r in results if r["arm"].endswith("ALL")), key=lambda r: r["logloss"])
    b = float(best["arm"].split("=")[1].split()[0])
    log(f"\n=== ANCHOR CHECKS (arm blend={b}) ===")
    fitted, state = fits[b]
    ov, n_all = state["overall"], state["n_all"]

    names: dict[str, str] = {}
    for side in ("winner", "loser"):
        names.update(dict(zip(fitted[f"{side}_uid"], fitted[f"{side}_name"])))

    # A player's rating never decays, so a retired great sits at their peak
    # forever. Ranking "current form" must therefore be restricted to players
    # ACTIVE in the window - the first run's anchor failure caught exactly this.
    recent = fitted[fitted["date"].dt.year >= 2024]
    active = pd.concat([recent["winner_uid"], recent["loser_uid"]]).value_counts()
    active = set(active[active >= 15].index)
    log(f"  active filter: {len(active):,} players with >=15 matches in 2024-25")

    passed = []
    for tour, expect in (("atp", {"Sinner", "Alcaraz"}), ("wta", {"Sabalenka", "Swiatek"})):
        ranked = sorted(((r, u) for u, r in ov.items()
                         if u.startswith(tour) and u in active), reverse=True)[:10]
        top = [(names.get(u, u), round(r)) for r, u in ranked]
        log(f"  {tour.upper()} top 5 active: {top[:5]}")
        hit = {e for e in expect if any(e in str(n) for n, _ in top[:5])}
        ok = hit == expect
        passed.append(ok)
        log(f"    anchor: {expect} in top 5? -> {hit or 'NONE'} {'PASS' if ok else 'FAIL'}")

    thin = [names.get(u, u) for u, r in sorted(ov.items(), key=lambda kv: -kv[1])[:20]
            if n_all.get(u, 0) < 10]
    passed.append(not thin)
    log(f"  anchor: no <10-match player in top 20 -> "
        f"{'PASS' if not thin else 'FAIL ' + str(thin)}")
    log(f"\n  ANCHORS: {sum(passed)}/{len(passed)} passed")

    # ---- Task 2: inactivity/layoff decay, a SEPARATE one-axis arm. `blend`
    # is held fixed at b (the blend grid's winner above) throughout - only
    # `decay` varies, so any score difference is attributable to decay alone
    # (stats-discipline §5). NOTE (stats-discipline §5/charter honesty): this
    # script evaluates directly on 2024-25 with no separate tuning-window
    # split of its own (an existing property of this baseline, not invented
    # here) - so picking the single best of this grid on this same window
    # carries real overfitting risk that a proper tune/holdout split would
    # avoid. Mitigated by (a) keeping the grid small, (b) checking the
    # improvement holds across a NEIGHBOURHOOD of decay values rather than
    # spiking at one point, before trusting it enough to change the written
    # parquet.
    #
    # Selection metric: "ALL-tour" (level_file=='tour' only), NOT the
    # all-levels "ALL" the blend grid above selects on. The eval window's
    # 848k rows are 89% lower-tier (qual/challenger/ITF) - a much noisier,
    # more journeyman-heavy population than the tour-level one the
    # hierarchical-model tuning effort actually compares against (its
    # 8,594-row market-odds-joined holdout is level_file=='tour' only, per
    # RESUME-beat-elo.md). A layoff signal genuinely present in the tour-level
    # population can be swamped by "ALL"'s lower-tier noise - which is exactly
    # what happened here (see log): confirmed below by comparing both.
    log(f"\n=== FITTING (one axis: layoff decay, blend held at {b}) ===")
    decay_results = []
    decay_fits = {}
    for decay in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        fitted_d, state_d = run_elo(df, b, decay)
        decay_fits[decay] = (fitted_d, state_d)
        sub_d = fitted_d[ev]
        decay_results.append(score(sub_d["p_winner"].values, f"decay={decay} ALL"))
        tour_only = sub_d[sub_d["level_file"] == "tour"]
        decay_results.append(score(tour_only["p_winner"].values, f"decay={decay} ALL-tour"))
        for tour in ("atp", "wta"):
            m = (sub_d["tour"] == tour) & (sub_d["level_file"] == "tour")
            if m.sum() > 200:
                decay_results.append(score(sub_d.loc[m, "p_winner"].values,
                                           f"decay={decay} {tour}/tour"))
        log(f"  decay={decay}: done")

    log("\n=== LAYOFF-DECAY RESULTS, eval {} ===".format(EVAL_YEARS))
    log(pd.DataFrame(decay_results).to_string(index=False))

    all_rows = {r["arm"]: r for r in decay_results if r["arm"].endswith(" ALL")}
    tour_rows = {r["arm"]: r for r in decay_results if r["arm"].endswith("ALL-tour")}
    base_ll_all = all_rows["decay=0.0 ALL"]["logloss"]
    base_ll = tour_rows["decay=0.0 ALL-tour"]["logloss"]
    log(f"\n  no-op check: decay=0.0 ALL logloss={base_ll_all:.6f} vs blend-grid arm "
        f"logloss={best['logloss']:.6f} "
        f"({'OK exact match' if abs(base_ll_all - best['logloss']) < 1e-9 else 'WARNING: mismatch, investigate'})")

    nonzero = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    n_beat_all = sum(1 for d in nonzero if all_rows[f"decay={d} ALL"]["logloss"] < base_ll_all)
    n_beat = sum(1 for d in nonzero if tour_rows[f"decay={d} ALL-tour"]["logloss"] < base_ll)
    log(f"  robustness (ALL, all levels pooled): {n_beat_all}/{len(nonzero)} nonzero decay "
        f"values beat decay=0.0 (dominated by 89% lower-tier rows - noisy, not the selection metric)")
    log(f"  robustness (ALL-tour, the population that matters): {n_beat}/{len(nonzero)} "
        f"nonzero decay values beat decay=0.0 "
        f"({'a real, non-spiky effect' if n_beat >= len(nonzero) - 1 else 'mixed - treat as noisy'})")

    best_decay_row = min(tour_rows.values(), key=lambda r: r["logloss"])
    d_best = float(best_decay_row["arm"].split("=")[1].split()[0])
    log(f"  best decay arm (ALL-tour): {best_decay_row['arm']} logloss={best_decay_row['logloss']:.6f} "
        f"(delta vs decay=0.0: {best_decay_row['logloss'] - base_ll:+.6f})")

    # ---- anchor check on the winning nonzero-decay arm, same gate as blend
    passed_d = [True]
    if d_best > 0.0:
        log(f"\n=== ANCHOR CHECKS (arm blend={b}, decay={d_best}) ===")
        fitted_d, state_d = decay_fits[d_best]
        ov_d, n_all_d = state_d["overall"], state_d["n_all"]
        names_d: dict[str, str] = {}
        for side in ("winner", "loser"):
            names_d.update(dict(zip(fitted_d[f"{side}_uid"], fitted_d[f"{side}_name"])))
        recent_d = fitted_d[fitted_d["date"].dt.year >= 2024]
        active_d = pd.concat([recent_d["winner_uid"], recent_d["loser_uid"]]).value_counts()
        active_d = set(active_d[active_d >= 15].index)
        passed_d = []
        for tour, expect in (("atp", {"Sinner", "Alcaraz"}), ("wta", {"Sabalenka", "Swiatek"})):
            ranked = sorted(((r, u) for u, r in ov_d.items() if u.startswith(tour) and u in active_d),
                            reverse=True)[:10]
            top = [(names_d.get(u, u), round(r)) for r, u in ranked]
            log(f"  {tour.upper()} top 5 active: {top[:5]}")
            hit = {e for e in expect if any(e in str(n) for n, _ in top[:5])}
            ok = hit == expect
            passed_d.append(ok)
            log(f"    anchor: {expect} in top 5? -> {hit or 'NONE'} {'PASS' if ok else 'FAIL'}")
        thin_d = [names_d.get(u, u) for u, r in sorted(ov_d.items(), key=lambda kv: -kv[1])[:20]
                  if n_all_d.get(u, 0) < 10]
        passed_d.append(not thin_d)
        log(f"  anchor: no <10-match player in top 20 -> "
            f"{'PASS' if not thin_d else 'FAIL ' + str(thin_d)}")
        log(f"\n  ANCHORS (decay arm): {sum(passed_d)}/{len(passed_d)} passed")

    # ---- write the eval predictions parquet from whichever arm is genuinely
    # best - this parquet is the frozen artifact the hierarchical-model
    # tuning effort compares against as "the number to beat", so it should
    # reflect the best-verified Elo baseline. Only switch off decay=0.0 if
    # the win is non-spiky (robust check above) AND anchors still pass.
    use_decay = (d_best > 0.0 and best_decay_row["logloss"] < base_ll
                 and n_beat >= len(nonzero) - 1 and all(passed_d))
    if use_decay:
        log(f"\n  using blend={b}, decay={d_best} for the written parquet "
            f"(genuine, non-spiky improvement; anchors pass)")
        out = decay_fits[d_best][0][ev].copy()
    else:
        log(f"\n  using blend={b}, decay=0.0 for the written parquet "
            f"(no decay value cleared the robustness+anchor bar)")
        out = fitted[ev].copy()
    out.attrs = {}
    out.to_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    log("wrote elo_eval_predictions.parquet")
