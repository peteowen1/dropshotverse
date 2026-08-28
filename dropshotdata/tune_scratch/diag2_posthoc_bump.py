"""Diagnostic (read-only): post-hoc, EVAL-TIME-ONLY variance inflation test.
Does NOT touch hierarchical_model.py or refit anything - takes the existing
d1_mean/d1_var/d2_mean/d2_var columns already computed by the v25 fit
(hier_v25_eval_predictions.parquet, via tune_scratch/diag2_enriched.parquet)
and adds a candidate bump directly to d1_var/d2_var for rows where the
winner and/or loser had a 1-12wk gap since their last match, then reruns
mc_average on the MODIFIED variance to see whether more variance AT
PREDICTION TIME (not baked into training via inflate()) helps - separating
"is more variance the right fix" from "is inflate() the right place to add
it".

Trimmed to a SMALL number of mc_average calls (each ~130s on this machine's
24 cores for the full 10,678-row eval set) - a previous version with ~26
configs got killed by an external `timeout` wrapper mid-spawn on Windows
(WinError 5 in a multiprocessing worker, not a real bug) and produced no
usable output. Run this one via run_in_background, no external timeout.
"""
import sys

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import numpy as np
import pandas as pd
from mc_parallel import mc_average

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"
N_MC = 300


def ll(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def acc(p):
    return float((np.asarray(p, dtype=float) > 0.5).mean())


def run_bump(df, bump, scope_mask, label):
    d = df.copy()
    add = np.where(scope_mask, bump, 0.0)
    d["d1_var_mod"] = d["d1_var"] + add
    d["d2_var_mod"] = d["d2_var"] + add
    p = mc_average(d, n_samples=N_MC, d1_var_col="d1_var_mod", d2_var_col="d2_var_mod")
    n_scoped = int(scope_mask.sum())
    print(f"  {label:38s} bump={bump:.3f}  n_scoped={n_scoped:5d}  "
          f"overall_ll={ll(p):.4f}  overall_acc={acc(p):.4f}", flush=True)
    return p


if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)
    print(f"n={len(df):,}", flush=True)

    baseline_mc600 = ll(df.p_winner_hier_mc600)
    print(f"baseline (existing mc600 col, n=600): overall_ll={baseline_mc600:.4f}  elo_ll={ll(df.p_elo):.4f}", flush=True)

    scope_any = df["either_short_layoff"].fillna(False).values
    scope_elite = scope_any & df["is_elite"].fillna(False).values
    scope_nonelite = scope_any & ~df["is_elite"].fillna(False).values
    print(f"rows with either_short_layoff=True: {scope_any.sum():,} / {len(df):,} "
          f"(elite={scope_elite.sum():,}, non-elite={scope_nonelite.sum():,})", flush=True)

    print("\n=== (A) global scope: any winner/loser 1-12wk gap ===", flush=True)
    run_bump(df, 0.0, scope_any, "any_short_layoff (baseline, recomputed)")
    for bump in [0.01, 0.02, 0.05]:
        run_bump(df, bump, scope_any, "any_short_layoff")

    print("\n=== (B) scope restricted to ELITE tourney_level only ===", flush=True)
    run_bump(df, 0.05, scope_elite, "elite_only + short_layoff")
    run_bump(df, 0.15, scope_elite, "elite_only + short_layoff")

    print("\n=== (C) scope restricted to NON-elite ===", flush=True)
    run_bump(df, 0.05, scope_nonelite, "non_elite + short_layoff")

    print("\ndone.", flush=True)
