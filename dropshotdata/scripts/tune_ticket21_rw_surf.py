"""Ticket 21 follow-up: re-scan RW_VAR_PER_WEEK and SURFACE_VAR_SCALE, untouched
since regime B's evidence mechanism fundamentally changed (ticket 21) and
MAX_SHIFT/CLIP_REINFLATE_K were re-tuned around it. Fixed: REGIME_B_N=0.25,
OU_THETA_PER_WEEK=0.00001, MAX_SHIFT=0.125, CLIP_REINFLATE_K=0.4.
"""
import sys
import time

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import tune_ticket21 as t

t.DEFAULTS = {**t.DEFAULTS, "DSV_REGIME_B_N": "0.25", "DSV_OU_THETA_PER_WEEK": "0.00001",
             "DSV_MAX_SHIFT": "0.125", "DSV_CLIP_REINFLATE_K": "0.4"}


def run_one(env_var, val, tag, n_mc=20):
    preds_path = t.SCRATCH / f"preds_{tag}.parquet"
    state_path = t.SCRATCH / f"state_{tag}.pkl"
    env = {**t.DEFAULTS, env_var: val,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in t.TUNE_YEARS),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    if not t.gate_passes(env):
        return None
    import subprocess, os
    full_env = {**os.environ, **env}
    t0c = time.time()
    r = subprocess.run([sys.executable, t.SCRIPT], cwd=t.WORKDIR, env=full_env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        t.log(f"  !! fit failed for {tag}: {r.stderr[-500:]}")
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
    bad = p["date"].dt.year >= 2024
    assert not bad.any(), f"LEAKAGE: {tag} touched {bad.sum()} rows >= 2024"
    p["k_w"] = p["winner_name"].map(key_from_full)
    p["k_l"] = p["loser_name"].map(key_from_full)
    p["pair"] = p.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=t.TUNE_YEARS).dropna(subset=["Winner", "Loser", "Date"])
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
    weight = np.where(m["mkt_fav"].values > t.LOPSIDED_THRESHOLD, t.LOPSIDED_WEIGHT, 1.0)
    mc = mc_average(m, n_samples=n_mc)
    pr = np.clip(mc, 1e-9, 1 - 1e-9)
    ll = -np.log(pr)
    weighted_logloss = float(np.sum(weight * ll) / np.sum(weight))
    plain_logloss = float(ll.mean())
    lop_mask = m["mkt_fav"].values > t.LOPSIDED_THRESHOLD
    lop_logloss = float(ll[lop_mask].mean()) if lop_mask.any() else float("nan")
    for f in [preds_path, state_path]:
        f.unlink(missing_ok=True)
    return {"tag": tag, "n": len(m), "weighted_logloss": round(weighted_logloss, 4),
           "plain_logloss": round(plain_logloss, 4), "lopsided_logloss": round(lop_logloss, 4),
           "corruption_pct": round(corrupt * 100, 5), "fit_s": round(fit_s, 1)}
