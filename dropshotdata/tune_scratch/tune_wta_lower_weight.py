"""Issue 20 round 4, attempt 16: tour-conditioned WTA lower-tier (qual/ITF)
regime-B evidence discount (DSV_WTA_LOWER_WEIGHT, hierarchical_model.py).

Confirmed on the real 8,594-row odds-joined 2024-25 holdout this round: the
entire hier-vs-Elo gap is WTA (atp gap ~0, wta gap +0.0185 to +0.0223
depending on MC sample count). Mechanism hypothesis: WTA's "lower" tier
(qual/ITF/$25k-$100k circuit) archive volume is ~3.8x WTA main-tour volume
(605k vs 158k rows, 2022-25: 131,556 lower-tier rows) vs ATP's ~1.2x (232k vs
194k), and only 27.9% of WTA lower-tier rows have point-level data (vs 93-99.8%
elsewhere) - so WTA main-tour player states are disproportionately built from
high-volume, low-instrumentation, weak-field regime-B evidence relative to
every other tour/level population. TIER_WEIGHT (ticket 22) already tried a
similar discount but is tour-SYMMETRIC (keys only on tourney_level, so it
discounts ATP Challenger exactly as much as WTA ITF) and didn't transfer -
this is a genuinely new, tour-conditioned angle: discount ONLY WTA's lower
tier, leave ATP Challenger at full weight.

Tunes ONLY on 2022-23 per charter. Scores POOLED, ATP-only, and WTA-only so
we can see the tradeoff directly instead of just a pooled number.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import tune_ticket21 as t

# v25 config (RESUME-beat-elo.md)
t.DEFAULTS = {**t.DEFAULTS,
             "DSV_MAX_SHIFT": "0.125", "DSV_REGIME_B_N": "0.25",
             "DSV_OU_THETA_PER_WEEK": "0.00001", "DSV_CLIP_REINFLATE_K": "0.5"}

GRID = ["1.0", "0.5", "0.25", "0.1", "0.05", "0.02", "0.0"]


def run_candidate(weight, tag, years, n_mc):
    preds_path = t.SCRATCH / f"preds_{tag}.parquet"
    state_path = t.SCRATCH / f"state_{tag}.pkl"
    env = {**t.DEFAULTS, "DSV_WTA_LOWER_WEIGHT": weight,
          "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    if not t.gate_passes(env):
        return None
    import subprocess, os
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
    results = []
    t0 = time.time()
    t.log(f"=== DSV_WTA_LOWER_WEIGHT scan, v25 base config, {t.TUNE_YEARS} tuning window ===")
    for weight in GRID:
        tag = f"wlw_{weight.replace('.', 'p')}"
        r = run_candidate(weight, tag, t.TUNE_YEARS, n_mc=20)
        if r is None:
            t.log(f"  WTA_LOWER_WEIGHT={weight}: REJECTED or failed")
            continue
        results.append({"WTA_LOWER_WEIGHT": weight, **r})
        t.log(f"  WTA_LOWER_WEIGHT={weight}: corruption={r['corruption_pct']}% fit={r['fit_s']}s "
             f"[{time.time()-t0:.0f}s elapsed]")
        t.log(f"    pooled={r['pooled']}")
        t.log(f"    atp={r.get('atp')}")
        t.log(f"    wta={r.get('wta')}")
    Path(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\wta_lower_weight_scan.json").write_text(
        json.dumps(results, indent=1))
    t.log(f"\ndone in {time.time()-t0:.0f}s, wrote tune_scratch/wta_lower_weight_scan.json")
