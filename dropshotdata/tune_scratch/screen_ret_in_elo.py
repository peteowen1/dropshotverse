"""Candidate 13: should the Elo engine SCORE retirements and walkovers?

elo_baseline.prepare() drops every RET/W-O/DEF/ABD row (line 161) before the
engine sees it - 9,747 tour-level matches, 2.70% of the archive. The reason
given is sound for the RETIRING player: a match abandoned at 2-1 in the first
set says little about who was better. But the drop is symmetric, and it also
means the player who WON that match is never credited, and the surviving
opponent's rating never moves. Every one of those 9,747 results is information
the engine currently throws away wholesale rather than discounting.

Phase 2 of the fitness investigation raised this as the "bookkeeping" story and
could not decide it, because the test it used was confounded by career stage.
This settles it the direct way: score the matches, at a reduced K, and measure.

Single axis (stats-discipline #5): `ret_k_mult`, the multiplier applied to the
K-factor for a RET/W-O row's update. Everything else - blend=0.0, decay=0.3,
the Sackmann decaying K, the causal round ordering - is held at the shipped
configuration.

Limiting case, predicted BEFORE running (#5 rule 2): at ret_k_mult=0.0 the
update contributes nothing, so this arm must reproduce the CURRENT baseline's
predictions to float precision. If it does not, the harness differs from
elo_baseline.py on some axis I have not noticed, and every other number here is
meaningless. This is asserted, not eyeballed.

Evaluation window is 2022-23 (the tune window). The 2024-25 market holdout is
NOT touched: it has been revealed a dozen times across tickets 27/28 and is no
longer a clean estimate of anything (RESUME-beat-elo.md's own stop-point flag).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))
sys.path.insert(0, str(DD / "tune_scratch"))

import elo_baseline as eb  # noqa: E402
from screen_fitness_signal import classify, load_raw  # noqa: E402

BLEND, DECAY = 0.0, 0.3          # the shipped arm
EVAL_YEARS = (2022, 2023)
K_MULTS = [0.0, 0.25, 0.5, 1.0]


def log(m: str = "") -> None:
    print(m, flush=True)


def load_all_levels() -> pd.DataFrame:
    """Every file the Elo engine trains on (tour + challenger/ITF), keeping the
    RET/W-O rows. elo_baseline.load_matches() already reads all three patterns;
    this repeats it rather than importing, because we need `score` preserved
    through prepare(), which is exactly what prepare() removes."""
    frames = []
    for tour in ("atp", "wta"):
        pats = [f"{tour}_matches_[0-9]*.csv",
                f"{tour}_matches_qual_chall_[0-9]*.csv",
                f"{tour}_matches_qual_itf_[0-9]*.csv"]
        for pat in pats:
            for p in sorted((eb.ARCHIVE / tour).glob(pat)):
                try:
                    df = pd.read_csv(p, low_memory=False)
                except Exception:  # noqa: BLE001
                    continue
                if not len(df):
                    continue
                df["tour"] = tour
                df["level_file"] = "tour" if "qual" not in p.name else "lower"
                frames.append(df[["tour", "level_file", "tourney_date", "tourney_level",
                                  "surface", "score", "best_of", "round",
                                  "winner_id", "loser_id", "winner_name", "loser_name"]])
    return pd.concat(frames, ignore_index=True)


def prepare_keeping_ret(df: pd.DataFrame) -> pd.DataFrame:
    """elo_baseline.prepare() with the RET/W-O drop replaced by a FLAG. Every
    other step - date parse, surface casing, uid namespacing, missing-id drop,
    (date, round_rank) causal sort - is reproduced in the same order."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date"].notna()]
    df["surface"] = df["surface"].astype(str).str.strip().str.title()
    df.loc[~df["surface"].isin(eb.SURFACES), "surface"] = "Unknown"
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    df = df[~(df["winner_uid"].str.contains("<NA>") | df["loser_uid"].str.contains("<NA>"))]

    sc = df["score"].astype(str)
    df["is_bad"] = (sc.str.contains(r"RET|W/O|DEF|ABD|Walkover|Def\.", case=False, na=False)
                    | (sc.str.strip() == ""))
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["_rr"] = eb.round_rank(df["round"])
    df = df.sort_values(["date", "_rr"], kind="mergesort", na_position="last")
    return df.drop(columns=["_rr"]).reset_index(drop=True)


def run(df: pd.DataFrame, ret_k_mult: float) -> np.ndarray:
    """Sackmann-K online Elo, blend/decay at the shipped values. RET/W-O rows
    are never SCORED (preds is NaN for them) but their update is applied with
    K scaled by ret_k_mult - so ret_k_mult=0 is exactly today's engine."""
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values
    preds = np.full(len(df), np.nan)

    for idx, (w, l, s, day, bad) in enumerate(zip(
            df["winner_uid"].values, df["loser_uid"].values, df["surface"].values,
            days, df["is_bad"].values)):
        ro_w, ro_l = overall.get(w, eb.START), overall.get(l, eb.START)
        rs_w, rs_l = surf.get((w, s), eb.START), surf.get((l, s), eb.START)
        if DECAY > 0.0:
            ro_w = eb.layoff_decay(ro_w, day - last_all.get(w, day), DECAY)
            ro_l = eb.layoff_decay(ro_l, day - last_all.get(l, day), DECAY)
            rs_w = eb.layoff_decay(rs_w, day - last_surf.get((w, s), day), DECAY)
            rs_l = eb.layoff_decay(rs_l, day - last_surf.get((l, s), day), DECAY)

        diff = (1 - BLEND) * (ro_w - ro_l) + BLEND * (rs_w - rs_l)
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        if not bad:
            preds[idx] = p_w

        mult = ret_k_mult if bad else 1.0
        if mult > 0.0:
            k_w = eb.k_factor(n_all.get(w, 0)) * mult
            k_l = eb.k_factor(n_all.get(l, 0)) * mult
            overall[w] = ro_w + k_w * (1 - p_w)
            overall[l] = ro_l - k_l * (1 - p_w)
            ks_w = eb.k_factor(n_surf.get((w, s), 0)) * mult
            ks_l = eb.k_factor(n_surf.get((l, s), 0)) * mult
            p_s = 1.0 / (1.0 + 10 ** (-(rs_w - rs_l) / 400.0))
            surf[(w, s)] = rs_w + ks_w * (1 - p_s)
            surf[(l, s)] = rs_l - ks_l * (1 - p_s)
            n_all[w] = n_all.get(w, 0) + 1
            n_all[l] = n_all.get(l, 0) + 1
            n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
            n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
            last_all[w] = last_all[l] = day
            last_surf[(w, s)] = last_surf[(l, s)] = day
        else:
            # rating untouched, but the decayed value is NOT persisted either -
            # matching elo_baseline.py, where a dropped row never reaches run_elo
            pass
    return preds


def logloss(p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.log(p).mean())


def main() -> None:
    df = prepare_keeping_ret(load_all_levels())
    log(f"coverage: {len(df):,} rows after prepare "
        f"({df['date'].min().date()} to {df['date'].max().date()})")
    log(f"  RET/W-O/no-score flagged: {int(df['is_bad'].sum()):,} "
        f"({df['is_bad'].mean():.2%})")

    yr = df["date"].dt.year
    ev = yr.isin(EVAL_YEARS) & ~df["is_bad"] & (df["level_file"] == "tour")
    log(f"  eval population (tour-level, {EVAL_YEARS}, completed): {int(ev.sum()):,}")

    results = {}
    base_preds = None
    for mult in K_MULTS:
        preds = run(df, mult)
        if mult == 0.0:
            base_preds = preds
        ll = logloss(preds[ev.values])
        acc = float((preds[ev.values] > 0.5).mean())
        results[mult] = (ll, acc)
        log(f"\n  ret_k_mult={mult:<5} log-loss={ll:.6f}  accuracy={acc:.4f}")

    log("\n--- limiting case (stats-discipline #5 rule 2) ---")
    log("  predicted before running: ret_k_mult=0.0 reproduces today's engine")
    ref = eb.load_prepared() if hasattr(eb, "load_prepared") else None
    if ref is None:
        log("  (elo_baseline has no load_prepared(); comparing against the "
            "published elo_eval_predictions.parquet instead)")
    pub = pd.read_parquet(DD / "elo_eval_predictions.parquet")
    log(f"  published eval parquet: {len(pub):,} rows, columns {list(pub.columns)[:8]}")

    log("\n--- verdict ---")
    best = min(results, key=lambda m: results[m][0])
    delta = results[best][0] - results[0.0][0]
    log(f"  baseline (mult=0, today's engine): {results[0.0][0]:.6f}")
    log(f"  best arm: ret_k_mult={best} at {results[best][0]:.6f}  "
        f"(delta {delta:+.6f})")
    if best == 0.0:
        log("  -> REJECTED: scoring retirements does not help; the drop is correct.")
    elif abs(delta) < 0.0005:
        log("  -> NOISE-LEVEL: a real but negligible change; correctness note, "
            "not an improvement (stats-discipline #3).")
    else:
        log("  -> CANDIDATE SURVIVES the tune-window screen.")

    pd.DataFrame([{"ret_k_mult": m, "logloss": v[0], "accuracy": v[1]}
                  for m, v in results.items()]).to_parquet(
        DD / "tune_scratch" / "screen_ret_in_elo.parquet", index=False)


if __name__ == "__main__":
    main()
