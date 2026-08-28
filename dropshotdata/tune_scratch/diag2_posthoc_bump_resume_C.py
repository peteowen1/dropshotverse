"""Resume-only: finish section (C) of diag2_posthoc_bump.py, which was killed
mid-run by an unrelated OOM (see MEMORY). Sections A and B already completed
and logged in diag2_posthoc_bump.log; their numbers agree with bump_test.log's
independent baseline check, so they are not re-run here.
"""
import pandas as pd

from diag2_posthoc_bump import run_bump

SRC = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag2_enriched.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(SRC)
    df = df.dropna(subset=["p_elo"]).reset_index(drop=True)

    scope_any = df["either_short_layoff"].fillna(False).values
    scope_nonelite = scope_any & ~df["is_elite"].fillna(False).values

    print("\n=== (C) scope restricted to NON-elite ===", flush=True)
    run_bump(df, 0.05, scope_nonelite, "non_elite + short_layoff")
    run_bump(df, 0.15, scope_nonelite, "non_elite + short_layoff")
    print("\ndone.", flush=True)
