"""Round-order-fix HOLDOUT reveal (2024-25) - exact harness from
RESUME-beat-elo.md sections 3/4. Run exactly once per charter."""
import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")

import pickle
import numpy as np
import pandas as pd
from mc_parallel import mc_average
from market_compare import key_from_full, key_from_short, devig_power, load_odds

if __name__ == "__main__":
    preds_path = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_holdout_predictions.parquet"
    state_path = r"C:\dev\dropshotverse\dropshotdata\hier_roundorder_holdout_state.pkl"

    # corruption gate first
    with open(state_path, "rb") as f:
        st = pickle.load(f)   # nosec: our own state file, written moments ago
    vals = [v[0] for v in st["players"].values()] + [v[2] for v in st["players"].values()]
    corrupt = sum(1 for x in vals if abs(x) > 2.5) / len(vals)
    print(f"corruption_rate = {corrupt*100:.5f}% (gate: <= 0.0201%)")

    p = pd.read_parquet(preds_path)
    print(f"predictions rows: {len(p)}")
    bad = p["date"].dt.year < 2024
    assert not bad.any(), f"LEAKAGE: touched {bad.sum()} rows < 2024"

    mc = mc_average(p, n_samples=60)
    p = p.assign(p_winner_hier_mc=mc)
    p.to_parquet(preds_path)

    elo = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\elo_eval_predictions.parquet")
    elo = elo[elo.level_file == "tour"][
        ["date", "tour", "winner_uid", "loser_uid", "winner_name", "loser_name", "p_winner"]
    ].rename(columns={"p_winner": "p_elo"})
    hier = p[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc"]]
    m = elo.merge(hier, on=["date", "tour", "winner_uid", "loser_uid"], how="inner")
    print(f"elo x hier merged rows: {len(m)}")
    m["k_w"] = m["winner_name"].map(key_from_full)
    m["k_l"] = m["loser_name"].map(key_from_full)
    m["pair"] = m.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)

    j = m.merge(odds[["pair", "tour", "date", "k_w", "k_l", "PSW", "PSL"]], on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    j["p_mkt"] = devig_power(j["PSW"].values.astype(float), j["PSL"].values.astype(float))
    j = j.dropna(subset=["p_mkt"])

    print(f"joined rows (odds-joined holdout): {len(j)}")

    print("\n=== OVERALL (2024-25 holdout) ===")
    lls = {}
    for col, name in [("p_elo", "Elo"), ("p_winner_hier_mc", "Hierarchical"), ("p_mkt", "Market")]:
        pr = np.clip(j[col].values, 1e-9, 1 - 1e-9)
        acc = float((pr > 0.5).mean())
        ll = float(-np.log(pr).mean())
        lls[name] = ll
        print(f"{name:15s} acc={acc:.4f}  logloss={ll:.4f}")

    j["mkt_fav"] = np.where(j["p_mkt"] >= 0.5, j["p_mkt"], 1 - j["p_mkt"])
    lopsided = j[j["mkt_fav"] > 0.80]
    print(f"\n=== LOPSIDED (mkt_fav>0.80, n={len(lopsided)}) ===")
    for col, name in [("p_elo", "Elo"), ("p_winner_hier_mc", "Hierarchical"), ("p_mkt", "Market")]:
        pr = np.clip(lopsided[col].values, 1e-9, 1 - 1e-9)
        ll = float(-np.log(pr).mean())
        print(f"{name:15s} logloss={ll:.4f}")

    print(f"\nGap to Elo (overall): {lls['Hierarchical'] - lls['Elo']:+.4f}  (pre-fix v25 gap was +0.0114)")

    # ATP/WTA split, per round 4/6's finding that the gap is concentrated in WTA
    print("\n=== ATP / WTA split ===")
    for tour in ("atp", "wta"):
        jt = j[j["tour"] == tour]
        pr_h = np.clip(jt["p_winner_hier_mc"].values, 1e-9, 1 - 1e-9)
        pr_e = np.clip(jt["p_elo"].values, 1e-9, 1 - 1e-9)
        ll_h = float(-np.log(pr_h).mean())
        ll_e = float(-np.log(pr_e).mean())
        print(f"{tour.upper():5s} n={len(jt):5d}  Hier={ll_h:.4f}  Elo={ll_e:.4f}  gap={ll_h-ll_e:+.4f}")
