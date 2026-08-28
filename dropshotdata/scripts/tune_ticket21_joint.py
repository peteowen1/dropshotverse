"""Joint scan of DSV_MAX_SHIFT x DSV_CLIP_REINFLATE_K - these interact
(reinflation only fires on a clamp, and MAX_SHIFT controls how often clamps
happen), and univariate scans of each found a different optimum than the
other's default. REGIME_B_N=0.25, OU_THETA_PER_WEEK=0.00001 fixed (ticket 21).
Uses the cached prepared archive + parallel MC (mc_parallel.py) - each
candidate now costs ~60-90s instead of ~230s.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import tune_ticket21 as t

t.DEFAULTS = {**t.DEFAULTS, "DSV_REGIME_B_N": "0.25", "DSV_OU_THETA_PER_WEEK": "0.00001"}

MAX_SHIFT_GRID = ["0.15", "0.175", "0.2"]
K_GRID = ["1.5", "2.0", "2.5"]


def run_joint_candidate(ms, k, tag, years, n_mc):
    """Like tune_ticket21.run_candidate, but does NOT hardcode DSV_REGIME_B_N
    from its args - this scan varies MAX_SHIFT and CLIP_REINFLATE_K only,
    with REGIME_B_N fixed via t.DEFAULTS."""
    preds_path = t.SCRATCH / f"preds_{tag}.parquet"
    state_path = t.SCRATCH / f"state_{tag}.pkl"
    env = {**t.DEFAULTS, "DSV_MAX_SHIFT": ms, "DSV_CLIP_REINFLATE_K": k,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    if not t.gate_passes(env):
        return None
    import subprocess
    full_env = {**__import__("os").environ, **env}
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


if __name__ == "__main__":
    results = []
    t0 = time.time()
    for ms in MAX_SHIFT_GRID:
        for k in K_GRID:
            tag = f"joint_ms{ms.replace('.','p')}_k{k.replace('.','p')}"
            r = run_joint_candidate(ms, k, tag, t.TUNE_YEARS, n_mc=20)
            if r is None:
                t.log(f"  MAX_SHIFT={ms} K={k}: REJECTED or failed")
                continue
            results.append({"MAX_SHIFT": ms, "CLIP_REINFLATE_K": k, **r})
            t.log(f"  MAX_SHIFT={ms} K={k}: weighted={r['weighted_logloss']} "
                 f"plain={r['plain_logloss']} lopsided={r['lopsided_logloss']} "
                 f"corruption={r['corruption_pct']}% fit={r['fit_s']}s "
                 f"[{time.time()-t0:.0f}s elapsed]")
    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\ticket21_joint_scan.json").write_text(
        json.dumps(results, indent=1))
    t.log(f"\ndone in {time.time()-t0:.0f}s, wrote tune_scratch/ticket21_joint_scan.json")
