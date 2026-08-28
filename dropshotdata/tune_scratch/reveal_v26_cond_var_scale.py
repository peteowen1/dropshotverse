"""Issue 20, round 5/6: ONE holdout reveal of the conditional var_scale
candidate (threshold=0.85, scale=0.7 - the genuine, non-boundary tuning-
window optimum found by tune_cond_var_scale_2223.py + tune_cond_var_scale_fine.py).
Exact join/scoring pattern from RESUME-beat-elo.md's verification harness,
split by tour (ATP/WTA) the way round 4/5 did, plus midband/lopsided.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
from market_compare import key_from_full, key_from_short, devig_power, load_odds
from mc_parallel import mc_average

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"
SCRATCH = Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch")

V25 = {"DSV_RW_VAR_PER_WEEK": "0.005", "DSV_SURFACE_VAR_SCALE": "1.0",
      "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_MAX_SHIFT": "0.125",
      "DSV_PHI_MAX": "4.0", "DSV_LAYOFF_C": "0.06", "DSV_REGIME_B_N": "0.25",
      "DSV_TIER_WEIGHT": "1.0", "DSV_CLIP_REINFLATE_K": "0.5"}

THRESHOLD = 0.85
SCALE = 0.7


def log(m): print(m, flush=True)


def ll(x):
    pr = np.clip(x, 1e-9, 1 - 1e-9)
    return float(-np.log(pr).mean())


if __name__ == "__main__":
    preds_path = SCRATCH / "preds_v26cand_holdout.parquet"
    state_path = SCRATCH / "state_v26cand_holdout.pkl"
    env = {**V25, "DSV_EVAL_YEARS": "2024,2025",
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    full_env = {**os.environ, **env}
    log("=== fitting v25 config on full history, evaluating on 2024-25 holdout ===")
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fit failed: {r.stderr[-2000:]}")
    log(f"  fit done in {time.time()-t0:.0f}s")

    p = pd.read_parquet(preds_path)
    log(f"holdout preds n={len(p)}")

    log("=== MC-averaging: baseline (var_scale=1.0, no-op) and candidate (threshold=0.85, scale=0.7), N=60 ===")
    t0 = time.time()
    mc_baseline = mc_average(p, n_samples=60, seed_base=0)
    log(f"  baseline mc_average done in {time.time()-t0:.0f}s")
    t0 = time.time()
    mc_cand = mc_average(p, n_samples=60, seed_base=0,
                         var_scale_threshold=THRESHOLD, var_scale_high_scale=SCALE)
    log(f"  candidate mc_average done in {time.time()-t0:.0f}s")

    p["p_winner_hier_mc_baseline"] = mc_baseline
    p["p_winner_hier_mc_cand"] = mc_cand

    log("=== joining to Elo + market odds (exact RESUME-beat-elo.md pattern) ===")
    elo = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    elo = elo[elo.level_file == "tour"][["date", "tour", "winner_uid", "loser_uid",
                                          "winner_name", "loser_name", "p_winner"]].rename(
        columns={"p_winner": "p_elo"})
    hier = p[["date", "tour", "winner_uid", "loser_uid", "best_of",
             "p_winner_hier_mc_baseline", "p_winner_hier_mc_cand"]]
    m = elo.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    m["k_w"] = m["winner_name"].map(key_from_full)
    m["k_l"] = m["loser_name"].map(key_from_full)
    m["pair"] = m.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    j = m.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]],
               on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    j["p_mkt"] = devig_power(j["PSW"].values.astype(float), j["PSL"].values.astype(float))
    j = j.dropna(subset=["p_mkt"])
    j["mkt_fav"] = np.where(j.p_mkt >= .5, j.p_mkt, 1 - j.p_mkt)
    log(f"joined holdout n={len(j)}")
    j.to_parquet(SCRATCH / "real_holdout_v26_cond_var_scale.parquet")

    def report(sub, label):
        log(f"\n--- {label} (n={len(sub)}) ---")
        for col, name in [("p_elo", "Elo"), ("p_winner_hier_mc_baseline", "Hier v25 (baseline)"),
                          ("p_winner_hier_mc_cand", "Hier v25+cond_var_scale(t=0.85,s=0.7)")]:
            pr = np.clip(sub[col].values, 1e-9, 1 - 1e-9)
            acc = float((pr > 0.5).mean())
            logloss = ll(pr)
            lop_mask = sub["mkt_fav"].values > 0.80
            lop_ll = ll(sub.loc[lop_mask, col].values) if lop_mask.any() else float("nan")
            mid_mask = (sub["mkt_fav"].values > 0.5) & (sub["mkt_fav"].values <= 0.7)
            mid_ll = ll(sub.loc[mid_mask, col].values) if mid_mask.any() else float("nan")
            log(f"  {name:38s} acc={acc:.4f}  logloss={logloss:.4f}  lopsided={lop_ll:.4f}  midband={mid_ll:.4f}")

    report(j, "OVERALL")
    report(j[j.tour == "atp"], "ATP")
    report(j[j.tour == "wta"], "WTA")

    preds_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    log("\ndone.")
