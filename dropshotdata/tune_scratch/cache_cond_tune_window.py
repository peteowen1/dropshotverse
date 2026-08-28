"""Cache the 2022-23 tuning window (v25 config, odds-joined) to parquet so
follow-up diagnostics for the conditional var_scale lever (issue 20, round
5/6) don't need to refit each time. Reuses tune_cond_var_scale_2223.py's
fit_and_join exactly."""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\tune_scratch")
from tune_cond_var_scale_2223 import fit_and_join

if __name__ == "__main__":
    tune = fit_and_join((2022, 2023), "condvs_cache")
    out = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\cond_tune_window_2223.parquet"
    tune.to_parquet(out)
    print(f"wrote {out}, n={len(tune)}")
