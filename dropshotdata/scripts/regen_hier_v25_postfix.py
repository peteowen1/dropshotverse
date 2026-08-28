"""Validation task (issue 27 audit): regenerate the v25-config hierarchical
model predictions used as build_xgb_features.py's raw-state feature source
(d1_mean/d1_var/d2_mean/d2_var), using the CURRENT (post round-9
causal-ordering-fix, commit 111ecce) hierarchical_model.py and the CURRENT
(post-fix, 21:56) prepared_matches_cache.pkl.

hier_v25_eval_predictions.parquet (mtime 16:02) and
tune_scratch/v25_tune_preds.parquet (mtime 15:44) both predate the round-9 fix
(21:55) by hours - i.e. their d1_mean/d2_mean posterior state was computed
while same-date multi-round matches (e.g. R32 processed after R16 due to
file-concatenation fallback order) were processed OUT of true chronological
order, exactly the bug class already caught in diag_layoff.parquet. Since
run() in hierarchical_model.py updates each player's posterior state
sequentially in df order, this is a plausible genuine leakage path into the
`hier_dmean_diff` feature (issue 27's #2 feature by gain, 3487.2).

This script does NOT edit hierarchical_model.py - it re-runs it as a
subprocess (same pattern as tune_scratch/reveal_v26_cond_var_scale.py), same
V25 hyperparameter env as the frozen snapshot, writing to NEW output files so
the original frozen hier_v25_eval_predictions.parquet / v25_tune_preds.parquet
are left untouched for comparison.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")

V25 = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
       "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
       "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
       "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}


def log(m):
    print(m, flush=True)


def run_window(years: str, tag: str) -> Path:
    preds_path = SCRATCH / f"v25_postfix_{tag}_preds.parquet"
    state_path = SCRATCH / f"v25_postfix_{tag}_state.pkl"
    env = {**V25, "DSV_EVAL_YEARS": years,
           "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    full_env = {**os.environ, **env}
    log(f"=== fitting v25 config (post-fix code+cache), eval years {years} -> {preds_path.name} ===")
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                        capture_output=True, text=True, timeout=1200)
    log(f"  exit={r.returncode}  elapsed={time.time()-t0:.0f}s")
    if r.returncode != 0:
        log("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-40:]))
        log("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-40:]))
        raise SystemExit(f"hierarchical_model.py failed for {tag}")
    log("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-15:]))
    return preds_path


if __name__ == "__main__":
    holdout_path = run_window("2024,2025", "holdout")
    tune_path = run_window("2022,2023", "tune")
    log(f"\nwrote {holdout_path}")
    log(f"wrote {tune_path}")
