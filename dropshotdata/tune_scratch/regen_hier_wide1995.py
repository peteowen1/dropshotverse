"""Ticket 28 round 12 continued: extend hierarchical_model.py's ALREADY-
VALIDATED v25 config (frozen hyperparameters, no retuning) to export
posterior state for 2015-2023, not just 2022-23. The model already replays
the full archive sequentially to reach any requested year - this only
changes which years its predictions get EXPORTED for, via the existing
DSV_EVAL_YEARS mechanism regen_hier_v25_postfix.py already established.
Read-only wrt hierarchical_model.py - runs it as a subprocess, same pattern.
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


years = ",".join(str(y) for y in range(1995, 2024))
preds_path = SCRATCH / "v25_postfix_wide1995_preds.parquet"
state_path = SCRATCH / "v25_postfix_wide1995_state.pkl"
env = {**V25, "DSV_EVAL_YEARS": years, "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
full_env = {**os.environ, **env}
log(f"=== fitting v25 config (frozen hyperparameters, no retuning), eval years {years} -> {preds_path.name} ===")
t0 = time.time()
r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env, capture_output=True, text=True, timeout=1800)
log(f"exit={r.returncode}  elapsed={time.time()-t0:.0f}s")
log("STDOUT tail:\n" + "\n".join(r.stdout.splitlines()[-30:]))
if r.returncode != 0:
    log("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-40:]))
    raise SystemExit("hierarchical_model.py failed")
log(f"\nwrote {preds_path}")
