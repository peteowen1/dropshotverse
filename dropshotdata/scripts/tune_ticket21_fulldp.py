"""Ticket 21 (full-DP): does the richer set-by-set-scoreline likelihood for
regime B (DSV_REGIME_B_FULL_DP=1, built and stress-tested in the prior
commit) beat the cheap games-won-count approximation v25 already ships with
(DSV_REGIME_B_FULL_DP=0)? Same charter as every prior round: tune ONLY on
2022-23, through the same weighted/lopsided metric v25 was selected on, gated
by stress_test_safety.py + corruption_rate() before any number is trusted.

Step 1: baseline (FULL_DP=0, v25 exactly) vs FULL_DP=1 at v25's own
REGIME_B_N=0.25 - does simply turning it on help or hurt at the SAME evidence
weight?
Step 2 (only if step 1 isn't a clean rejection): re-scan REGIME_B_N under
FULL_DP=1, since the richer likelihood's evidence strength doesn't
necessarily peak at the same value the games-won-count approximation did -
and unlike that approximation, a real per-outcome likelihood isn't obviously
maximally overconfident at N>1, so the previously-closed-off upper range is
worth re-checking too (still corruption-gated, same as ever).
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

# v25, the current best config (RESUME-beat-elo.md)
V25 = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
      "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
      "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
      "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}


def log(m): print(m, flush=True)


def gate_passes(env):
    full_env = {**__import__("os").environ, **env}
    r = subprocess.run([sys.executable, GATE], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log("    SYNTHETIC GATE REJECTED: " +
            "; ".join(l.strip() for l in r.stdout.splitlines() if "FAIL" in l))
    return r.returncode == 0


def corruption_rate(state_path: str) -> float:
    # pickle load is safe: state_path was written moments ago by our own
    # hierarchical_model.py subprocess in this same tuning run (same
    # convention as tune_v2.py/tune_v3_lopsided.py/tune_regime_b.py).
    with open(state_path, "rb") as f:
        st = pickle.load(f)
    vals = [v[0] for v in st["players"].values()] + [v[2] for v in st["players"].values()]
    return sum(1 for x in vals if abs(x) > 2.5) / len(vals)


def run_candidate(env_overrides: dict, tag: str, years: tuple, n_mc: int) -> dict | None:
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    state_path = SCRATCH / f"state_{tag}.pkl"
    env = {**V25, **env_overrides,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}

    if not gate_passes(env):
        return None

    full_env = {**__import__("os").environ, **env}
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                      capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        log(f"  !! fit failed for {tag}: {r.stderr[-800:]}")
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
    log(f"=== Ticket 21 full-DP: does DSV_REGIME_B_FULL_DP=1 beat v25 on {TUNE_YEARS}? ===")

    baseline = run_candidate({"DSV_REGIME_B_FULL_DP": "0"}, "fulldp_baseline_v25", TUNE_YEARS, n_mc=15)
    log(f"baseline (v25, FULL_DP=0): {baseline}")

    step1 = run_candidate({"DSV_REGIME_B_FULL_DP": "1"}, "fulldp_on_default_n", TUNE_YEARS, n_mc=15)
    log(f"FULL_DP=1 at v25's REGIME_B_N=0.25: {step1}")

    results = {"baseline": baseline, "step1_default_n": step1}

    log("\n--- step 2: re-scan DSV_REGIME_B_N under FULL_DP=1 ---")
    grid = ["0.1", "0.25", "0.5", "1.0", "2.0", "4.0"]
    best_tag, best_score, best_n = None, baseline["weighted_logloss"], None
    scan = []
    for n in grid:
        tag = f"fulldp_n_{n}".replace(".", "p")
        r = run_candidate({"DSV_REGIME_B_FULL_DP": "1", "DSV_REGIME_B_N": n}, tag, TUNE_YEARS, n_mc=15)
        if r is None:
            log(f"  REGIME_B_N={n}: rejected by a gate")
            continue
        log(f"  REGIME_B_N={n}: weighted={r['weighted_logloss']} plain={r['plain_logloss']} "
            f"lopsided={r['lopsided_logloss']} corruption={r['corruption_pct']}% fit={r['fit_s']}s")
        scan.append({"n": n, **r})
        if r["weighted_logloss"] < best_score:
            best_score = r["weighted_logloss"]
            best_tag = tag
            best_n = n

    results["scan"] = scan
    results["best_n"] = best_n
    results["best_weighted_logloss"] = best_score

    log(f"\n=== SUMMARY ===")
    log(f"baseline (v25, FULL_DP=0): weighted={baseline['weighted_logloss']} "
        f"plain={baseline['plain_logloss']} lopsided={baseline['lopsided_logloss']}")
    if best_n is not None:
        log(f"best FULL_DP=1 candidate: REGIME_B_N={best_n} weighted={best_score} "
            f"(vs baseline {baseline['weighted_logloss']})")
    else:
        log("no FULL_DP=1 candidate beat baseline on the tuning window")

    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\selected_fulldp.json").write_text(
        json.dumps(results, indent=1))
    log("\nwrote tune_scratch/selected_fulldp.json")
