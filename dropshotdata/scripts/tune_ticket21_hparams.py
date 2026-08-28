"""Ticket 21 follow-up: RW_VAR_PER_WEEK/OU_THETA_PER_WEEK/SURFACE_VAR_SCALE/
MAX_SHIFT were coordinate-searched (tickets 17/19) against the OLD fake
all-or-nothing regime-B evidence. Now that regime B carries real games-per-set
evidence (ticket 21, DSV_REGIME_B_N=0.25), those optima may have shifted -
stats-discipline 5.4: a hyperparameter tuned inside one arm doesn't
necessarily transfer to a materially different arm. Re-scan one axis at a
time (coordinate descent, same style as ticket 17), holding REGIME_B_N=0.25.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import tune_ticket21 as t

t.DEFAULTS = {**t.DEFAULTS, "DSV_REGIME_B_N": "0.25"}


def scan(env_var, grid, label):
    t.log(f"=== scanning {env_var} (ticket 21 re-tune, REGIME_B_N=0.25 fixed) ===")
    results = []
    for val in grid:
        env_backup = t.DEFAULTS.pop(env_var, None)
        tag = f"{label}_{val.replace('.', 'p').replace('-', 'm')}"
        old_run_candidate = t.run_candidate

        def run_with_override(n_val, tag, years, n_mc, _val=val, _env_var=env_var):
            preds_path = t.SCRATCH / f"preds_{tag}.parquet"
            state_path = t.SCRATCH / f"state_{tag}.pkl"
            env = {**t.DEFAULTS, _env_var: _val, "DSV_REGIME_B_N": "0.25",
                  "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
                  "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
            if not t.gate_passes(env):
                return None
            import subprocess, time
            full_env = {**__import__("os").environ, **env}
            t0 = time.time()
            r = subprocess.run([sys.executable, t.SCRIPT], cwd=t.WORKDIR, env=full_env,
                              capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                t.log(f"  !! fit failed for {tag}: {r.stderr[-500:]}")
                return None
            fit_s = time.time() - t0
            corrupt = t.corruption_rate(str(state_path))
            if corrupt > t.BASELINE_CORRUPTION * t.CORRUPTION_TOLERANCE:
                t.log(f"    POPULATION GATE REJECTED {tag}: corruption={corrupt*100:.4f}%")
                preds_path.unlink(missing_ok=True); state_path.unlink(missing_ok=True)
                return None
            import pandas as pd, numpy as np
            from market_compare import key_from_full, key_from_short, devig_power, load_odds
            from hierarchical_model import p_match, sigmoid
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
            from mc_parallel import mc_average   # speed fix - see mc_parallel.py
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

        r = run_with_override(None, tag, t.TUNE_YEARS, 15)
        if env_backup is not None:
            t.DEFAULTS[env_var] = env_backup
        if r is None:
            t.log(f"  {env_var}={val}: REJECTED or failed")
            continue
        results.append({env_var: val, **r})
        t.log(f"  {env_var}={val}: weighted={r['weighted_logloss']} plain={r['plain_logloss']} "
              f"lopsided={r['lopsided_logloss']} corruption={r['corruption_pct']}% fit={r['fit_s']}s")
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ou"
    if which == "ou":
        res = scan("DSV_OU_THETA_PER_WEEK", ["0.0002", "0.0005", "0.001", "0.002", "0.005"], "ou")
    elif which == "rw":
        res = scan("DSV_RW_VAR_PER_WEEK", ["0.0025", "0.005", "0.01", "0.02"], "rw")
    elif which == "surf":
        res = scan("DSV_SURFACE_VAR_SCALE", ["0.5", "1.0", "2.0", "4.0"], "surf")
    elif which == "shift":
        res = scan("DSV_MAX_SHIFT", ["0.1", "0.2", "0.3", "0.4"], "shift")
    else:
        raise SystemExit(f"unknown axis {which}")
    Path(rf"C:\dev\dropshotverse\dropshotdata\tune_scratch\ticket21_hparam_{which}.json").write_text(
        json.dumps(res, indent=1))
