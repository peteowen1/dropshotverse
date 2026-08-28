"""Issue 20 round 4, attempt 17: generic single-hyperparameter re-scan on the
2022-23 tuning window, scored POOLED / ATP-only / WTA-only, to test whether
any pooled hyperparameter's optimum is being dragged away from WTA's true
optimum by ATP (or vice versa). WTA_LOWER_WEIGHT scan (attempt 16) already
REJECTED the "discount noisy WTA lower-tier evidence" hypothesis - monotonic
regression on WTA as weight decreased, ATP unaffected as expected (isolated
correctly). This is the next candidate: is a globally-flat-looking
hyperparameter actually two tour-specific optima that happen to average out?

Usage: python tune_by_tour_generic.py <ENV_VAR_NAME> <comma-separated grid>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import tune_ticket21 as t

# v25 config (RESUME-beat-elo.md)
t.DEFAULTS = {**t.DEFAULTS,
             "DSV_MAX_SHIFT": "0.125", "DSV_REGIME_B_N": "0.25",
             "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_CLIP_REINFLATE_K": "0.5"}


def run_candidate(var_name, val, tag, years, n_mc):
    preds_path = t.SCRATCH / f"preds_{tag}.parquet"
    state_path = t.SCRATCH / f"state_{tag}.pkl"
    env = {**t.DEFAULTS, var_name: val,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    if not t.gate_passes(env):
        return None
    full_env = {**os.environ, **env}
    t0c = time.time()
    r = subprocess.run([sys.executable, t.SCRIPT], cwd=t.WORKDIR, env=full_env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        t.log(f"  !! fit failed for {tag}: {r.stderr[-800:]}")
        return None
    fit_s = time.time() - t0c
    corrupt = t.corruption_rate(str(state_path))
    if corrupt > t.BASELINE_CORRUPTION * t.CORRUPTION_TOLERANCE:
        t.log(f"    POPULATION GATE REJECTED {tag}: corruption={corrupt*100:.4f}%")
        preds_path.unlink(missing_ok=True); state_path.unlink(missing_ok=True)
        return None

    import pandas as pd, numpy as np
    from market_compare import key_from_full, key_from_short, devig_power, load_odds
    from mc_parallel import mc_average
    p = pd.read_parquet(preds_path)
    if len(p) == 0:
        return None
    if years == t.TUNE_YEARS:
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
    mc = mc_average(m, n_samples=n_mc)
    m["mc"] = mc

    def score(sub):
        weight_v = np.where(sub["mkt_fav"].values > t.LOPSIDED_THRESHOLD, t.LOPSIDED_WEIGHT, 1.0)
        pr = np.clip(sub["mc"].values, 1e-9, 1 - 1e-9)
        ll = -np.log(pr)
        weighted_logloss = float(np.sum(weight_v * ll) / np.sum(weight_v))
        plain_logloss = float(ll.mean())
        lop_mask = sub["mkt_fav"].values > t.LOPSIDED_THRESHOLD
        lop_logloss = float(ll[lop_mask].mean()) if lop_mask.any() else float("nan")
        return {"n": len(sub), "weighted": round(weighted_logloss, 4),
               "plain": round(plain_logloss, 4), "lopsided": round(lop_logloss, 4)}

    out = {"pooled": score(m)}
    for tour, g in m.groupby("tour"):
        out[tour] = score(g)

    for f in [preds_path, state_path]:
        f.unlink(missing_ok=True)
    return {"tag": tag, "corruption_pct": round(corrupt * 100, 5), "fit_s": round(fit_s, 1), **out}


if __name__ == "__main__":
    var_name = sys.argv[1]
    grid = sys.argv[2].split(",")
    results = []
    t0 = time.time()
    t.log(f"=== {var_name} scan (by-tour), v25 base config, {t.TUNE_YEARS} tuning window ===")
    for val in grid:
        tag = f"btg_{var_name.replace('DSV_','').lower()}_{val.replace('.', 'p').replace('-','m')}"
        r = run_candidate(var_name, val, tag, t.TUNE_YEARS, n_mc=20)
        if r is None:
            t.log(f"  {var_name}={val}: REJECTED or failed")
            continue
        results.append({var_name: val, **r})
        t.log(f"  {var_name}={val}: corruption={r['corruption_pct']}% fit={r['fit_s']}s "
             f"[{time.time()-t0:.0f}s elapsed]")
        t.log(f"    pooled={r['pooled']}")
        t.log(f"    atp={r.get('atp')}")
        t.log(f"    wta={r.get('wta')}")
    outpath = Path(rf"C:\dev\dropshotverse\dropshotdata\tune_scratch\bytour_{var_name.replace('DSV_','').lower()}_scan.json")
    outpath.write_text(json.dumps(results, indent=1))
    t.log(f"\ndone in {time.time()-t0:.0f}s, wrote {outpath}")
