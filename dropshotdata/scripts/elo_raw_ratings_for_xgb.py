"""XGBoost feature-build helper (issue 26).

READ-ONLY reuse of elo_baseline.py's logic: load_matches()/prepare()/
run_elo()'s core loop are copied here verbatim (not imported, not edited in
place in elo_baseline.py) so this script can emit the RAW per-side overall
and surface Elo ratings pre-match (needed to build overall-Elo-diff and
surface-Elo-diff features for XGBoost) - elo_eval_predictions.parquet only
carries the final blended p_winner, which collapses that distinction.

This file does NOT touch elo_baseline.py or hierarchical_model.py, per the
task brief's explicit instruction (another agent may be concurrently editing
elo_baseline.py).

Verification strategy: replicate the exact blend-grid selection and decay=0.3
arm elo_baseline.py already picked, then confirm this script's own p_winner
(recomputed from the raw overall+surface ratings the same way run_elo() does)
matches the frozen elo_eval_predictions.parquet's p_winner on the tour-level
eval rows to high precision, before trusting the raw ratings for anything.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import os

import _common
SURFACE_OWN_EXPECTATION = os.environ.get("DSV_SURFACE_OWN_EXPECTATION", "1") != "0"
ARCHIVE = _common.RAW / "github"
EVAL_YEARS = (2024, 2025)

K_BASE, K_SHAPE, K_MIN = 250.0, 0.4, 16.0
START = 1500.0
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

ROUND_RANK: dict[str, int] = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4,
    "ER": 5,
    "R128": 6, "R64": 7, "R32": 8, "R16": 9,
    "QF": 10, "RR": 11, "SF": 12,
    "F": 13, "BR": 13,
}
QUAL_ROUNDS = {"Q1", "Q2", "Q3", "Q4", "Q5", "ER"}
MAIN_ROUNDS = {"R128", "R64", "R32", "R16", "QF", "RR", "SF", "F", "BR"}

LAYOFF_PERIOD = 500.0
LAYOFF_GRACE = 30.0
LAYOFF_GRADIENT = 100.0
_LAYOFF_DRIFT = 1.0 / (1.0 + math.exp((LAYOFF_PERIOD - LAYOFF_GRACE) / LAYOFF_GRADIENT))


def log(m: str) -> None:
    print(m, flush=True)


def round_rank(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    normalized = normalized.where(normalized != "f", "F")
    ranks = normalized.map(ROUND_RANK)
    unknown = ranks.isna()
    if unknown.any():
        bad_vals = sorted(normalized[unknown].unique())
        warnings.warn(
            f"prepare(): {int(unknown.sum())} row(s) have round value(s) "
            f"{bad_vals} not in ROUND_RANK", stacklevel=2)
    return ranks


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
                except Exception:
                    continue
                if not len(df):
                    continue
                df["tour"] = tour
                df["level_file"] = ("tour" if "qual" not in p.name else "lower")
                frames.append(df[["tour", "level_file", "tourney_date", "tourney_level",
                                  "surface", "score", "best_of", "round",
                                  "winner_id", "loser_id", "winner_name", "loser_name"]])
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df[~df["date"].isna()]
    df["surface"] = df["surface"].astype(str).str.strip().str.title()
    df.loc[~df["surface"].isin(SURFACES), "surface"] = "Unknown"
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    bad = df["winner_uid"].str.contains("<NA>") | df["loser_uid"].str.contains("<NA>")
    df = df[~bad]
    sc = df["score"].astype(str)
    bad = sc.str.contains(r"RET|W/O|DEF|ABD|Walkover|Def\.", case=False, na=False) | (sc.str.strip() == "")
    df = df[~bad]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["_round_rank"] = round_rank(df["round"])
    df = df.sort_values(["date", "_round_rank"], kind="mergesort", na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_round_rank"])
    return df


def k_factor(n: int) -> float:
    return max(K_MIN, K_BASE / ((n + 5) ** K_SHAPE))


def layoff_decay(rating: float, days_gap: float, decay: float) -> float:
    if decay <= 0.0 or days_gap <= 0.0:
        return rating
    logistic = 1.0 / (1.0 + math.exp((LAYOFF_PERIOD - days_gap) / LAYOFF_GRADIENT))
    frac = decay * max(0.0, (logistic - _LAYOFF_DRIFT) / (1.0 - _LAYOFF_DRIFT))
    return rating - frac * (rating - START)


def run_elo_raw(df: pd.DataFrame, blend: float, decay: float):
    """Same as elo_baseline.run_elo() but ALSO records the four pre-match raw
    ratings (winner-overall, loser-overall, winner-surface, loser-surface) so
    overall-diff and surface-diff can be built separately downstream."""
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values

    n = len(df)
    preds = np.empty(n)
    ro_w_arr = np.empty(n); ro_l_arr = np.empty(n)
    rs_w_arr = np.empty(n); rs_l_arr = np.empty(n)
    nw_arr = np.empty(n, dtype=np.int64); nl_arr = np.empty(n, dtype=np.int64)
    nsw_arr = np.empty(n, dtype=np.int64); nsl_arr = np.empty(n, dtype=np.int64)

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
        preds[idx] = p_w
        ro_w_arr[idx] = ro_w; ro_l_arr[idx] = ro_l
        rs_w_arr[idx] = rs_w; rs_l_arr[idx] = rs_l
        nw_arr[idx] = n_all.get(w, 0); nl_arr[idx] = n_all.get(l, 0)
        nsw_arr[idx] = n_surf.get((w, s), 0); nsl_arr[idx] = n_surf.get((l, s), 0)

        kw, kl = k_factor(n_all.get(w, 0)), k_factor(n_all.get(l, 0))
        overall[w] = ro_w + kw * (1 - p_w)
        overall[l] = ro_l - kl * (1 - p_w)
        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1
        last_all[w] = day
        last_all[l] = day

        # The surface rating updates against ITS OWN expectation, not the
        # blended one. Found 2026-09-09 by walking one row: with blend=0.0
        # the old update used overall Elo's p_w, so once overall Elo already
        # expected the result the surface rating barely moved - Alcaraz won
        # 20 straight hard-court matches (Jan-Mar 2026) and rs went 1580 to
        # 1634; Musetti sat at 1160 on hard. rs was a cumulative residual
        # against overall Elo, not a surface rating (corr with ro across the
        # US Open field: 0.26), and elo_surface_diff fed the shipped model.
        # DSV_SURFACE_OWN_EXPECTATION=0 reproduces the old (residual) column
        # exactly, for ablation and for regenerating the parquet the shipped
        # model was trained on.
        p_s = (1.0 / (1.0 + 10 ** (-(rs_w - rs_l) / 400.0))) if SURFACE_OWN_EXPECTATION else p_w
        ksw, ksl = k_factor(n_surf.get((w, s), 0)), k_factor(n_surf.get((l, s), 0))
        surf[(w, s)] = rs_w + ksw * (1 - p_s)
        surf[(l, s)] = rs_l - ksl * (1 - p_s)
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
        last_surf[(w, s)] = day
        last_surf[(l, s)] = day

    out = df.copy()
    out["p_winner_recomputed"] = preds
    out["ro_winner"] = ro_w_arr; out["ro_loser"] = ro_l_arr
    out["rs_winner"] = rs_w_arr; out["rs_loser"] = rs_l_arr
    out["n_all_winner"] = nw_arr; out["n_all_loser"] = nl_arr
    out["n_surf_winner"] = nsw_arr; out["n_surf_loser"] = nsl_arr
    return out


def score(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    log("loading archive...")
    raw = load_matches()
    df = prepare(raw)
    ev = df["date"].dt.year.isin(EVAL_YEARS)
    log(f"kept {len(df):,} rows, eval window {int(ev.sum()):,}")

    # Replicate elo_baseline.py's blend grid selection (decay=0) on the
    # ALL-levels eval metric, exactly as the frozen script did, to recover
    # which blend value b it selected (not persisted anywhere else).
    log("\n=== blend grid (decay=0), for verification only ===")
    best_b, best_ll = None, np.inf
    for blend in (0.0, 0.5, 0.7, 1.0):
        fitted = run_elo_raw(df, blend, 0.0)
        ll = score(fitted.loc[ev, "p_winner_recomputed"].values)
        log(f"  blend={blend}: ALL logloss={ll:.6f}")
        if ll < best_ll:
            best_ll, best_b = ll, blend
    log(f"selected blend b={best_b} (logloss {best_ll:.6f})")

    # Now the decay=0.3 arm elo_baseline.py actually wrote to the parquet.
    log("\n=== decay=0.3 arm (the one actually written to elo_eval_predictions.parquet) ===")
    fitted_final = run_elo_raw(df, best_b, 0.3)

    # ---- verify against the frozen parquet on tour-level eval rows
    frozen = pd.read_parquet(_common.DD / "elo_eval_predictions.parquet")
    frozen_tour = frozen[frozen["level_file"] == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "p_winner"]]
    mine = fitted_final[(fitted_final["level_file"] == "tour") & ev][
        ["date", "tour", "winner_uid", "loser_uid", "p_winner_recomputed"]]
    chk = frozen_tour.merge(mine, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    diff = (chk["p_winner"] - chk["p_winner_recomputed"]).abs()
    log(f"verification join: {len(chk):,} / {len(frozen_tour):,} frozen tour rows matched")
    log(f"max abs p_winner diff: {diff.max():.8f}, mean: {diff.mean():.8f}")
    if diff.max() > 1e-6:
        log("WARNING: recomputed p_winner does not match frozen parquet - "
            "raw ratings below should NOT be trusted until this is resolved.")
    else:
        log("OK: exact match (within float noise) - raw ratings verified faithful.")

    out_cols = ["date", "tour", "level_file", "surface", "round", "best_of",
                "winner_uid", "loser_uid", "winner_name", "loser_name",
                "ro_winner", "ro_loser", "rs_winner", "rs_loser",
                "n_all_winner", "n_all_loser", "n_surf_winner", "n_surf_loser",
                "p_winner_recomputed"]
    out_path = str(_common.SCRATCH / "elo_raw_ratings_full.parquet")
    fitted_final[out_cols].to_parquet(out_path)
    log(f"\nwrote {out_path} ({len(fitted_final):,} rows, all levels/years - caller should "
        f"filter to eval years / tour level as needed)")
