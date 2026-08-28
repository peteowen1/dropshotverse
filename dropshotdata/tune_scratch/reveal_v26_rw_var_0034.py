"""Issue 20 round 5: ONE holdout reveal of the RW_VAR_PER_WEEK candidate that
survived both gates (stress_test_safety.py's new stale-but-confident checks,
AND corruption_rate()) on the 2022-23 tuning window. Candidate: v25 config
with DSV_RW_VAR_PER_WEEK=0.0034 (the best-scoring value that still passes the
new variance-sufficiency floors; 0.0033 and below fail the 5yr-layoff floor).

Per charter: reveal exactly once on the untouched 2024-25 holdout, using the
exact join pattern from RESUME-beat-elo.md's verification harness section.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

SCRIPT = r"C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py"
GATE = r"C:\dev\dropshotverse\dropshotdata\scripts\stress_test_safety.py"
WORKDIR = r"C:\dev\dropshotverse\dropshotdata\scripts"

V25_BASE = {
    "DSV_RW_VAR_PER_WEEK": "0.0034",   # candidate under test, was 0.005
    "DSV_SURFACE_VAR_SCALE": "1.0",
    "DSV_OU_THETA_PER_WEEK": "0.00001",
    "DSV_MAX_SHIFT": "0.125",
    "DSV_PHI_MAX": "4.0",
    "DSV_LAYOFF_C": "0.06",
    "DSV_REGIME_B_N": "0.25",
    "DSV_TIER_WEIGHT": "1.0",
    "DSV_CLIP_REINFLATE_K": "0.5",
}

PREDS = r"C:\dev\dropshotverse\dropshotdata\hier_v26cand_eval_predictions.parquet"
STATE = r"C:\dev\dropshotverse\dropshotdata\hier_v26cand_final_state.pkl"


def main():
    env = {**os.environ, **V25_BASE,
          "DSV_EVAL_YEARS": "2024,2025",
          "DSV_OUT_PREDS": PREDS, "DSV_OUT_STATE": STATE}

    print("=== gate check on holdout config ===", flush=True)
    r = subprocess.run([sys.executable, GATE], cwd=WORKDIR, env=env,
                       capture_output=True, text=True, timeout=60)
    print(r.stdout)
    if r.returncode != 0:
        print("GATE FAILED, aborting reveal"); sys.exit(1)

    print("=== fitting v26 candidate on full history, eval 2024-2025 ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=env,
                       capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("FIT FAILED:", r.stderr[-2000:]); sys.exit(1)
    print(f"fit done in {time.time()-t0:.1f}s", flush=True)

    import pickle
    # pickle load is safe: STATE was written moments ago by our own
    # hierarchical_model.py subprocess in this same run (same pattern as
    # every tune_*.py script's corruption_rate()).
    with open(STATE, "rb") as f:
        st = pickle.load(f)
    vals = [v[0] for v in st["players"].values()] + [v[2] for v in st["players"].values()]
    corrupt = sum(1 for x in vals if abs(x) > 2.5) / len(vals)
    print(f"corruption on full-history fit: {corrupt*100:.5f}% (gate: 0.0201%)", flush=True)
    if corrupt > 0.0067 / 100 * 3.0:
        print("CORRUPTION GATE FAILED on full-history fit, aborting"); sys.exit(1)

    print("=== MC-averaging (parallel, N=60) ===", flush=True)
    import pandas as pd
    from mc_parallel import mc_average
    p = pd.read_parquet(PREDS)
    t0 = time.time()
    mc = mc_average(p, n_samples=60)
    p = p.assign(p_winner_hier_mc=mc)
    p.to_parquet(PREDS)
    print(f"mc_average done in {time.time()-t0:.1f}s, {len(p)} rows", flush=True)

    print("=== join to Elo + market odds, score pooled and by tour ===", flush=True)
    import numpy as np
    from market_compare import key_from_full, key_from_short, devig_power, load_odds

    elo = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    elo = elo[elo.level_file == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name", "p_winner"]
    ].rename(columns={"p_winner": "p_elo"})
    hier = pd.read_parquet(PREDS)[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc"]]
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
    j.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\real_holdout_v26cand_joined.parquet")

    def score(sub, label):
        print(f"\n-- {label} (n={len(sub)}) --")
        for col, name in [("p_elo", "Elo"), ("p_winner_hier_mc", "Hierarchical")]:
            pr = np.clip(sub[col].values, 1e-9, 1 - 1e-9)
            ll = float(-np.log(pr).mean())
            acc = float((pr > 0.5).mean())
            print(f"  {name:15s} acc={acc:.4f}  logloss={ll:.4f}")
        e = -np.log(np.clip(sub["p_elo"].values, 1e-9, 1 - 1e-9)).mean()
        h = -np.log(np.clip(sub["p_winner_hier_mc"].values, 1e-9, 1 - 1e-9)).mean()
        print(f"  gap (hier - elo) = {h - e:+.4f}")

    score(j, "OVERALL")
    for tour, g in j.groupby("tour"):
        score(g, tour.upper())

    lop = j[np.where(j.p_mkt >= .5, j.p_mkt, 1 - j.p_mkt) > 0.80]
    score(lop, "LOPSIDED (mkt_fav>0.80)")


if __name__ == "__main__":
    main()
