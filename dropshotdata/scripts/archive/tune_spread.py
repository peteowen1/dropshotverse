"""Ticket 18: widen ability spread (OU_THETA down, MAX_SHIFT up) while tuning
the injury-specific parameters (PHI_MAX, LAYOFF_C) in their own right - every
candidate gated by stress_test_safety.py first. Same tune(2022-23)/holdout
(2024-25) discipline as ticket 17; RW_VAR_PER_WEEK=0.005 and
SURFACE_VAR_SCALE=4.0 (ticket 17's winners) held fixed.
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
GATE = r"C:\dev\dropshotverse\dropshotdata\scripts\stress_test_safety.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
TUNE_YEARS = (2022, 2023)
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")
SCRATCH.mkdir(exist_ok=True)

# ticket 17's winners, held fixed here
FIXED = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "4.0"}
DEFAULTS = {**FIXED, "DSV_OU_THETA_PER_WEEK": "0.0010", "DSV_MAX_SHIFT": "1.0",
           "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06"}

GRID = {
    "DSV_OU_THETA_PER_WEEK": ["0.0000", "0.0002", "0.0005", "0.0010"],
    "DSV_MAX_SHIFT": ["1.0", "1.5", "2.0", "3.0"],
    "DSV_PHI_MAX": ["2.0", "4.0", "8.0"],
    "DSV_LAYOFF_C": ["0.03", "0.06", "0.12"],
}


def log(m): print(m, flush=True)


def gate_passes(env: dict) -> bool:
    full_env = {**__import__("os").environ, **env}
    r = subprocess.run([sys.executable, GATE], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log(f"    GATE REJECTED this candidate:\n" +
            "\n".join(f"      {l}" for l in r.stdout.splitlines() if "FAIL" in l or "FAILURES" in l))
    return r.returncode == 0


def run_candidate(env_overrides: dict, tag: str, years: tuple, n_mc: int) -> dict:
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    env = {**DEFAULTS, **env_overrides,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path),
          "DSV_OUT_STATE": str(SCRATCH / f"state_{tag}.pkl")}

    if not gate_passes(env):
        return None

    full_env = {**__import__("os").environ, **env}
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log(f"  !! fit failed for {tag}: {r.stderr[-500:]}")
        return None
    fit_s = time.time() - t0

    p = pd.read_parquet(preds_path)
    if len(p) == 0:
        return None
    if years == TUNE_YEARS:
        bad = p["date"].dt.year >= 2024
        assert not bad.any(), f"LEAKAGE: {tag} touched {bad.sum()} rows >= 2024"

    p["k_w"] = p["winner_name"].map(key_from_full)
    p["k_l"] = p["loser_name"].map(key_from_full)
    p["pair"] = p.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=years).dropna(subset=["Winner", "Loser", "Date"])
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
        return None

    rng = np.random.default_rng(0)
    mc = np.empty(len(m))
    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m["p_fav_mkt"] = np.where(m.p_mkt >= .5, m.p_mkt, 1 - m.p_mkt)
    for i, row in enumerate(m.itertuples()):
        d1 = rng.normal(row.d1_mean, np.sqrt(max(row.d1_var, 1e-6)), n_mc)
        d2 = rng.normal(row.d2_mean, np.sqrt(max(row.d2_var, 1e-6)), n_mc)
        ps = [p_match(sigmoid(a), sigmoid(b), int(row.best_of)) for a, b in zip(d1, d2)]
        mc[i] = np.mean(ps)

    pr = np.clip(mc, 1e-9, 1 - 1e-9)
    logloss = float(-np.log(pr).mean())
    # lopsided-bucket diagnostic (per the charter, reported not selected on)
    lop = m[m.p_fav_mkt > 0.80]
    lop_pr = np.clip(mc[m.p_fav_mkt.values > 0.80], 1e-9, 1 - 1e-9)
    lop_ll = float(-np.log(lop_pr).mean()) if len(lop) else float("nan")

    for f in [preds_path, SCRATCH / f"state_{tag}.pkl"]:
        f.unlink(missing_ok=True)
    return {"tag": tag, "n": len(m), "logloss": round(logloss, 4),
           "lopsided_n": len(lop), "lopsided_logloss": round(lop_ll, 4),
           "fit_s": round(fit_s, 1)}


if __name__ == "__main__":
    log(f"=== TUNING spread+injury on {TUNE_YEARS} only (charter: issues/18) ===")
    best = dict(DEFAULTS)
    baseline = run_candidate({}, "spread_baseline", TUNE_YEARS, n_mc=15)
    log(f"baseline (ticket 17 config): {baseline}")
    best_logloss = baseline["logloss"]

    for param, values in GRID.items():
        log(f"\n--- coordinate: {param} (holding others at {best}) ---")
        for v in values:
            if v == best[param]:
                continue
            tag = f"{param}_{v}".replace(".", "p")
            trial = dict(best)
            trial[param] = v
            r = run_candidate(trial, tag, TUNE_YEARS, n_mc=15)
            if r is None:
                continue
            log(f"  {param}={v}: logloss={r['logloss']} lopsided_ll={r['lopsided_logloss']} "
                f"(n={r['lopsided_n']}) fit={r['fit_s']}s")
            if r["logloss"] < best_logloss:
                best_logloss = r["logloss"]
                best[param] = v
                log(f"    -> new best ({param}={v})")

    log(f"\n=== SELECTED (chosen using ONLY {TUNE_YEARS}) ===")
    log(f"{best}")
    log(f"tuning-window logloss: {best_logloss} (baseline was {baseline['logloss']})")

    import json
    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\selected_spread.json").write_text(
        json.dumps({"best": best, "tuning_logloss": best_logloss,
                   "baseline_logloss": baseline["logloss"]}, indent=1))
    log("\nwrote tune_scratch/selected_spread.json")
