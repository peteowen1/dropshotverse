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


def run_elo(df: pd.DataFrame, blend: float) -> pd.DataFrame:
    """Online Elo. `blend` mixes overall and surface rating:
        p = sigma( (1-blend)*overall_diff + blend*surface_diff )

    blend=0 is pure overall Elo; blend=1 is pure surface Elo. This is the ONLY
    axis that varies between arms (stats-discipline §5), so any difference in
    score is attributable to it and nothing else.
    """
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}

    preds = np.empty(len(df))
    for idx, (w, l, s) in enumerate(zip(df["winner_uid"].values,
                                        df["loser_uid"].values,
                                        df["surface"].values)):
        ro_w, ro_l = overall.get(w, START), overall.get(l, START)
        rs_w = surf.get((w, s), START)
        rs_l = surf.get((l, s), START)

        diff = (1 - blend) * (ro_w - ro_l) + blend * (rs_w - rs_l)
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        preds[idx] = p_w                       # P(actual winner wins), pre-match

        kw, kl = k_factor(n_all.get(w, 0)), k_factor(n_all.get(l, 0))
        overall[w] = ro_w + kw * (1 - p_w)
        overall[l] = ro_l - kl * (1 - p_w)
        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1

        ksw, ksl = k_factor(n_surf.get((w, s), 0)), k_factor(n_surf.get((l, s), 0))
        surf[(w, s)] = rs_w + ksw * (1 - p_w)
        surf[(l, s)] = rs_l - ksl * (1 - p_w)
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1

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

    out = fitted[ev].copy()
    out.attrs = {}
    out.to_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    log("wrote elo_eval_predictions.parquet")
