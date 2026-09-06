"""Ticket 19: re-optimize hyperparameters against a lopsided-match-weighted
log-loss, not plain overall log-loss (ticket 18's tune_v2.py). Same charter,
same corruption gate, same tune(2022-23)/holdout(2024-25) split - only the
scoring function changes, per the locked charter in issues/19.
"""
import json
import pickle
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

LOPSIDED_THRESHOLD = 0.80
LOPSIDED_WEIGHT = 3.0

BASELINE_CORRUPTION = 0.0067 / 100
CORRUPTION_TOLERANCE = 3.0

# warm-started from ticket 18's overall-log-loss optimum, not the original defaults
DEFAULTS = {"DSV_RW_VAR_PER_WEEK": "0.008", "DSV_SURFACE_VAR_SCALE": "2.0",
           "DSV_OU_THETA_PER_WEEK": "0.0010", "DSV_MAX_SHIFT": "0.2",
           "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06"}

# MAX_SHIFT grid extended upward - the whole point is testing whether a bigger
# cap (more confidence, more spread) helps once we're scoring for it
GRID = {
    "DSV_MAX_SHIFT": ["0.2", "0.3", "0.4", "0.5", "0.6"],
    "DSV_RW_VAR_PER_WEEK": ["0.005", "0.008", "0.012"],
    "DSV_OU_THETA_PER_WEEK": ["0.0005", "0.0010", "0.0020"],
    "DSV_SURFACE_VAR_SCALE": ["1.0", "2.0", "4.0"],
}


def log(m): print(m, flush=True)


def gate_passes(env: dict) -> bool:
    full_env = {**__import__("os").environ, **env}
    r = subprocess.run([sys.executable, GATE], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log("    SYNTHETIC GATE REJECTED: " +
            "; ".join(l.strip() for l in r.stdout.splitlines() if "FAIL" in l))
    return r.returncode == 0


def corruption_rate(state_path: str) -> float:
    # pickle load is safe: state_path was written moments ago by our own
    # hierarchical_model.py subprocess in this same tuning run
    with open(state_path, "rb") as f:
        st = pickle.load(f)
    vals = [v[0] for v in st["players"].values()] + [v[2] for v in st["players"].values()]
    return sum(1 for x in vals if abs(x) > 2.5) / len(vals)


def run_candidate(env_overrides: dict, tag: str, years: tuple, n_mc: int) -> dict:
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    state_path = SCRATCH / f"state_{tag}.pkl"
    env = {**DEFAULTS, **env_overrides,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}

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

    corrupt = corruption_rate(str(state_path))
    if corrupt > BASELINE_CORRUPTION * CORRUPTION_TOLERANCE:
        log(f"    POPULATION GATE REJECTED {tag}: corruption={corrupt*100:.4f}%")
        preds_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None

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

    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m = m.dropna(subset=["p_mkt"])
    m["mkt_fav"] = np.where(m.p_mkt >= .5, m.p_mkt, 1 - m.p_mkt)
    weight = np.where(m["mkt_fav"].values > LOPSIDED_THRESHOLD, LOPSIDED_WEIGHT, 1.0)

    rng = np.random.default_rng(0)
    mc = np.empty(len(m))
    for i, row in enumerate(m.itertuples()):
        d1 = rng.normal(row.d1_mean, np.sqrt(max(row.d1_var, 1e-6)), n_mc)
        d2 = rng.normal(row.d2_mean, np.sqrt(max(row.d2_var, 1e-6)), n_mc)
        ps = [p_match(sigmoid(a), sigmoid(b), int(row.best_of)) for a, b in zip(d1, d2)]
        mc[i] = np.mean(ps)

    pr = np.clip(mc, 1e-9, 1 - 1e-9)
    ll = -np.log(pr)
    weighted_logloss = float(np.sum(weight * ll) / np.sum(weight))
    plain_logloss = float(ll.mean())
    lop_mask = m["mkt_fav"].values > LOPSIDED_THRESHOLD
    lop_logloss = float(ll[lop_mask].mean()) if lop_mask.any() else float("nan")

    for f in [preds_path, state_path]:
        f.unlink(missing_ok=True)
    return {"tag": tag, "n": len(m), "weighted_logloss": round(weighted_logloss, 4),
           "plain_logloss": round(plain_logloss, 4), "lopsided_logloss": round(lop_logloss, 4),
           "corruption_pct": round(corrupt * 100, 5), "fit_s": round(fit_s, 1)}


if __name__ == "__main__":
    log(f"=== TUNING v3 (lopsided-weighted) on {TUNE_YEARS} only ===")
    log(f"metric: weight={LOPSIDED_WEIGHT}x for market-favourite > {LOPSIDED_THRESHOLD}")
    log(f"corruption gate: reject above {BASELINE_CORRUPTION*CORRUPTION_TOLERANCE*100:.4f}%")
    best = dict(DEFAULTS)
    baseline = run_candidate({}, "v3_baseline", TUNE_YEARS, n_mc=15)
    log(f"baseline (ticket 18's overall-optimum config): {baseline}")
    best_score = baseline["weighted_logloss"]

    for param, values in GRID.items():
        log(f"\n--- coordinate: {param} (holding others at {best}) ---")
        for v in values:
            if v == best[param]:
                continue
            tag = f"v3_{param}_{v}".replace(".", "p")
            trial = dict(best)
            trial[param] = v
            r = run_candidate(trial, tag, TUNE_YEARS, n_mc=15)
            if r is None:
                continue
            log(f"  {param}={v}: weighted={r['weighted_logloss']} plain={r['plain_logloss']} "
                f"lopsided={r['lopsided_logloss']} corruption={r['corruption_pct']}% fit={r['fit_s']}s")
            if r["weighted_logloss"] < best_score:
                best_score = r["weighted_logloss"]
                best[param] = v
                log(f"    -> new best ({param}={v})")

    log(f"\n=== SELECTED (lopsided-weighted metric, chosen using ONLY {TUNE_YEARS}) ===")
    log(f"{best}")
    log(f"tuning-window weighted logloss: {best_score} (baseline was {baseline['weighted_logloss']})")

    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\selected_v3.json").write_text(
        json.dumps({"best": best, "tuning_weighted_logloss": best_score,
                   "baseline_weighted_logloss": baseline["weighted_logloss"]}, indent=1))
    log("\nwrote tune_scratch/selected_v3.json")
