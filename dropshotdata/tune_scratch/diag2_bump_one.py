"""Diagnostic (read-only): single-config post-hoc variance-bump test, run
directly in the foreground (Windows multiprocessing spawn proved fragile
when nohup'd/backgrounded in this session - twice failed with a WinError 5
pipe-handle error mid-spawn with no useful traceback reaching the log file).
One call = one bump value + one scope, so each invocation finishes well
within a single foreground Bash call's timeout.

Usage: python diag2_bump_one.py <bump> <scope: any|elite|nonelite>
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


if __name__ == "__main__":
    bump = float(sys.argv[1])
    scope_name = sys.argv[2] if len(sys.argv) > 2 else "any"

    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)

    scope_any = df["either_short_layoff"].fillna(False).values
    scope_elite = scope_any & df["is_elite"].fillna(False).values
    scope_nonelite = scope_any & ~df["is_elite"].fillna(False).values
    scope_map = {"any": scope_any, "elite": scope_elite, "nonelite": scope_nonelite}
    scope = scope_map[scope_name]

    add = np.where(scope, bump, 0.0)
    df["d1_var_mod"] = df["d1_var"] + add
    df["d2_var_mod"] = df["d2_var"] + add
    p = mc_average(df, n_samples=N_MC, d1_var_col="d1_var_mod", d2_var_col="d2_var_mod")

    has_mkt = df["p_mkt"].notna().values
    lopsided = has_mkt & (np.maximum(df["p_mkt"].values, 1 - df["p_mkt"].values) > 0.80)

    print(f"RESULT scope={scope_name} bump={bump:.3f} n_scoped={int(scope.sum())} "
          f"overall_ll={ll(p):.4f} overall_acc={acc(p):.4f} "
          f"lopsided_ll={ll(p[lopsided]):.4f} lopsided_conf={np.maximum(p[lopsided],1-p[lopsided]).mean():.4f}",
          flush=True)
