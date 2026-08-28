"""Run exactly one DSV_REGIME_B_N candidate under FULL_DP=1 and print the
result. Split out from tune_ticket21_fulldp.py so each candidate can be run
as its own foreground process with tight visibility, after that script's
step 2 stalled silently (subprocess.run hung with 0 CPU progress, caught via
CPU-delta check, killed). Reuses run_candidate() unchanged - only the
driver loop changed, not the tuning logic itself.
"""
import sys

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from tune_ticket21_fulldp import run_candidate, TUNE_YEARS, log

n = sys.argv[1]
tag = f"fulldp_n_{n}".replace(".", "p")
log(f"running REGIME_B_N={n} ...")
r = run_candidate({"DSV_REGIME_B_FULL_DP": "1", "DSV_REGIME_B_N": n}, tag, TUNE_YEARS, n_mc=15)
if r is None:
    log(f"RESULT n={n}: REJECTED (gate failure or no data)")
else:
    log(f"RESULT n={n}: {r}")
