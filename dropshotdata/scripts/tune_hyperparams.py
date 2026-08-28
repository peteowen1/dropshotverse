"""Ticket 17: cross-validate the three hyperparameters ticket 10 flagged as
plausibility-set. Coordinate search, selection strictly on the 2022-2023
tuning window; 2024-2025 is never read until the chosen configuration's final
run, per the locked validation charter in issue 17.

Each candidate is a fresh subprocess of hierarchical_model.py (env-configured,
see that file's DSV_* overrides) - no shared state between runs, and each
candidate's predictions land in its own file.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from hierarchical_model import p_match, sigmoid
from market_compare import key_from_full, key_from_short, devig_power, load_odds

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
TUNE_YEARS = (2022, 2023)
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")
SCRATCH.mkdir(exist_ok=True)

# defaults = ticket 10's plausibility-set values, always included as a candidate
# so tuning can never silently do WORSE than what's already been reported
DEFAULTS = {"DSV_RW_VAR_PER_WEEK": "0.010", "DSV_SURFACE_VAR_SCALE": "1.0",
           "DSV_OU_THETA_PER_WEEK": "0.0025"}

GRID = {
    "DSV_RW_VAR_PER_WEEK": ["0.005", "0.010", "0.020", "0.040"],
    "DSV_SURFACE_VAR_SCALE": ["0.5", "1.0", "2.0", "4.0"],
    "DSV_OU_THETA_PER_WEEK": ["0.0010", "0.0025", "0.0050", "0.0100"],
}


def log(m): print(m, flush=True)


def run_candidate(env_overrides: dict, tag: str, years: tuple, n_mc: int) -> dict:
    """Run one fit + posterior-averaged score. Returns the metrics dict."""
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    env = {**DEFAULTS, **env_overrides,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path),
          "DSV_OUT_STATE": str(SCRATCH / f"state_{tag}.pkl")}
    full_env = {**__import__("os").environ, **env}

    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log(f"  !! fit failed for {tag}: {r.stderr[-800:]}")
        return None
    fit_s = time.time() - t0

    p = pd.read_parquet(preds_path)
    p = p[p.level_file == "tour"] if "level_file" in p.columns else p
    if len(p) == 0:
        log(f"  !! no eval rows for {tag}")
        return None

    # LEAKAGE CHECK, per the locked charter: assert no row in this scoring
    # pass is dated 2024 or later unless we are explicitly in the final reveal.
    if years == TUNE_YEARS:
        bad = p["date"].dt.year >= 2024
        assert not bad.any(), f"LEAKAGE: {tag} tuning pass touched {bad.sum()} rows >= 2024"

    p["k_w"] = p["winner_name"].map(key_from_full)
    p["k_l"] = p["loser_name"].map(key_from_full)
    p["pair"] = p.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=years)
    odds = odds.dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    m = p.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
               on=["pair", "tour"], how="left", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
    m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    m = m[(m["k_w_o"] == m["k_w"]) & m[["PSW", "PSL"]].notna().all(axis=1)]
    if len(m) < 200:
        log(f"  !! only {len(m)} odds-joined rows for {tag}, too few to trust")
        return None

    rng = np.random.default_rng(0)
    mc = np.empty(len(m))
    for i, row in enumerate(m.itertuples()):
        d1 = rng.normal(row.d1_mean, np.sqrt(max(row.d1_var, 1e-6)), n_mc)
        d2 = rng.normal(row.d2_mean, np.sqrt(max(row.d2_var, 1e-6)), n_mc)
        ps = [p_match(sigmoid(a), sigmoid(b), int(row.best_of)) for a, b in zip(d1, d2)]
        mc[i] = np.mean(ps)

    pr = np.clip(mc, 1e-9, 1 - 1e-9)
    logloss = float(-np.log(pr).mean())
    acc = float((pr > 0.5).mean())
    scratch_files = [preds_path, SCRATCH / f"state_{tag}.pkl"]
    for f in scratch_files:
        f.unlink(missing_ok=True)   # candidates are disposable; keep disk clean
    return {"tag": tag, "n": len(m), "logloss": round(logloss, 4),
           "acc": round(acc, 4), "fit_s": round(fit_s, 1)}


if __name__ == "__main__":
    log(f"=== TUNING on {TUNE_YEARS} only (charter: issues/17) ===")
    best = dict(DEFAULTS)
    baseline = run_candidate({}, "baseline", TUNE_YEARS, n_mc=15)
    log(f"baseline (ticket 10 defaults): {baseline}")
    best_logloss = baseline["logloss"]

    for param, values in GRID.items():
        log(f"\n--- coordinate: {param} (holding others at {best}) ---")
        for v in values:
            if v == best[param]:
                continue   # already scored as part of baseline or a prior step
            tag = f"{param}_{v}".replace(".", "p")
            trial = dict(best)
            trial[param] = v
            r = run_candidate(trial, tag, TUNE_YEARS, n_mc=15)
            if r is None:
                continue
            log(f"  {param}={v}: logloss={r['logloss']} acc={r['acc']} "
                f"n={r['n']} fit={r['fit_s']}s")
            if r["logloss"] < best_logloss:
                best_logloss = r["logloss"]
                best[param] = v
                log(f"    -> new best ({param}={v})")

    log(f"\n=== SELECTED CONFIGURATION (chosen using ONLY {TUNE_YEARS}) ===")
    log(f"{best}")
    log(f"tuning-window logloss: {best_logloss} (baseline was {baseline['logloss']})")

    import json
    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\selected.json").write_text(
        json.dumps({"best": best, "tuning_logloss": best_logloss,
                   "baseline_logloss": baseline["logloss"]}, indent=1))
    log("\nwrote tune_scratch/selected.json")
