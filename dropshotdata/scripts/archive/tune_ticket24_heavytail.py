"""Ticket 24: tune DSV_STUDENT_T_NU (heavy-tailed process noise degrees of
freedom) on top of v25, through the SAME weighted/lopsided metric v25 itself
was selected on (tune_v3_lopsided.py's metric), on the 2022-23 tuning window
only. Same charter, same corruption gate, same tune/holdout split.

Step 1: DSV_HEAVY_TAILED_NOISE=1 at the default DSV_STUDENT_T_NU=4.0 vs
DSV_HEAVY_TAILED_NOISE=0 (v25 baseline) - does turning the mechanism on at
all move the needle, before spending time scanning nu?
Step 2: if step 1 shows ANY signal (even a small one, not clearly noise),
scan nu across a real range (2, 3, 4, 6, 10, 20, 30) to find the tuning-
window optimum. If step 1 is flat/negative, report that honestly and do not
manufacture a scan result.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from tune_v3_lopsided import run_candidate, log, TUNE_YEARS

# v25 config (RESUME-beat-elo.md, round 3) - the actual current best, not
# tune_v3_lopsided.py's own stale DEFAULTS (that module's DEFAULTS predate
# ticket 21/the CLIP_REINFLATE_K era - only its run_candidate/metric/gate
# machinery is reused here, not its hyperparameter defaults).
V25 = {
    "DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
    "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
    "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
    "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5",
}

SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")

if __name__ == "__main__":
    log("=== Ticket 24: DSV_STUDENT_T_NU tuning on 2022-23 (weighted/lopsided metric) ===")
    t0 = time.time()

    baseline = run_candidate(V25, "t24_v25_baseline", TUNE_YEARS, n_mc=15)
    log(f"v25 baseline (HEAVY_TAILED_NOISE=0): {baseline}")

    step1_env = {**V25, "DSV_HEAVY_TAILED_NOISE": "1"}   # DSV_STUDENT_T_NU defaults to 4.0
    step1 = run_candidate(step1_env, "t24_heavytail_on_default_nu", TUNE_YEARS, n_mc=15)
    log(f"heavy-tailed ON, nu=4.0 (default): {step1}")

    results = {"baseline": baseline, "step1_default_nu": step1}

    if baseline is None or step1 is None:
        log("\n!! baseline or step1 candidate failed the gate or fit - aborting, nothing to report.")
        Path(SCRATCH / "ticket24_tuning_result.json").write_text(json.dumps(results, indent=1))
        sys.exit(1)

    delta = step1["weighted_logloss"] - baseline["weighted_logloss"]
    log(f"\nstep1 delta (heavy-tailed - baseline) weighted_logloss = {delta:+.4f}")

    # Scan nu regardless of step1's sign/size, so the tuning-window optimum
    # (if any) is found properly rather than eyeballing one point - but the
    # write-up will be honest if this whole family turns out flat, same
    # precedent as ticket 21.
    NU_GRID = ["2.0", "3.0", "4.0", "6.0", "10.0", "20.0", "30.0"]
    scan = []
    best = step1
    best_nu = "4.0"
    for nu in NU_GRID:
        if nu == "4.0":
            scan.append({"nu": nu, **step1})
            continue
        env = {**V25, "DSV_HEAVY_TAILED_NOISE": "1", "DSV_STUDENT_T_NU": nu}
        tag = f"t24_nu_{nu.replace('.', 'p')}"
        r = run_candidate(env, tag, TUNE_YEARS, n_mc=15)
        if r is None:
            log(f"  nu={nu}: REJECTED (gate or fit failure)")
            continue
        log(f"  nu={nu}: weighted={r['weighted_logloss']} plain={r['plain_logloss']} "
            f"lopsided={r['lopsided_logloss']} corruption={r['corruption_pct']}% fit={r['fit_s']}s")
        scan.append({"nu": nu, **r})
        if r["weighted_logloss"] < best["weighted_logloss"]:
            best = r
            best_nu = nu

    results["nu_scan"] = scan
    results["best_nu"] = best_nu
    results["best"] = best
    results["baseline_weighted_logloss"] = baseline["weighted_logloss"]
    results["best_vs_baseline_delta"] = round(best["weighted_logloss"] - baseline["weighted_logloss"], 4)

    log(f"\n=== SELECTED nu={best_nu}: weighted={best['weighted_logloss']} "
        f"(baseline {baseline['weighted_logloss']}, delta {results['best_vs_baseline_delta']:+.4f}) ===")
    log(f"total time: {time.time()-t0:.0f}s")

    Path(SCRATCH / "ticket24_tuning_result.json").write_text(json.dumps(results, indent=1))
    log("wrote tune_scratch/ticket24_tuning_result.json")
