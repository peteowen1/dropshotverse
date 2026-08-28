"""Issue 20 round 4, attempt 18: tour-conditioned isotonic recalibration.

Two pooled recalibration attempts (linear Platt, isotonic - both boundary-safe)
were already tried and REJECTED earlier this session: both improved overall
log-loss but traded away lopsided-bucket confidence, because the pooled
miscalibration curve is non-monotonic (overconfident in the middle, mildly
under/well-calibrated at the extremes) and a pooled corrector can't serve
both tours' curves at once if they have genuinely different shapes.

This round found (real 8,594-row holdout, split by tour):
  - WTA's own calibration hump is MORE SEVERE than ATP's: WTA Hier error in
    the (0.5,0.6] mkt_fav band is -0.09, in the (0.8,1.0] band is +0.10-0.11
    (vs Elo's -0.02/+0.04 and Market's ~0 in the same bands) - both ends
    worse than the pooled numbers the earlier Platt/isotonic attempts were
    built against.
  - ATP's hier actually BEATS Elo on non-lopsided matches (gap -0.0103);
    WTA's hier LOSES to Elo even on non-lopsided matches (gap +0.0114).

This is a genuinely new angle vs the twice-rejected pooled calibration: fit
isotonic SEPARATELY per tour (WTA-only 2022-23 tuning window) instead of
pooled, so the corrector only has to match WTA's own (more extreme) curve
shape, and apply it ONLY to WTA holdout rows - ATP stays completely
untouched (identity transform), unlike the pooled attempt which touched both
tours with one compromise curve.

Boundary-safe isotonic per the earlier bugfix: calibrate in favourite-prob
space (p_fav = max(p,1-p) in [0.5,1]), y_min=0.5 constraint, decision
boundary provably fixed at 0.5.
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


def log(m): print(m, flush=True)


def fit_and_join(years, tag, n_mc):
    preds_path = SCRATCH / f"preds_{tag}.parquet"
    state_path = SCRATCH / f"state_{tag}.pkl"
    env = {**V25, "DSV_EVAL_YEARS": ",".join(str(y) for y in years),
          "DSV_OUT_PREDS": str(preds_path), "DSV_OUT_STATE": str(state_path)}
    full_env = {**os.environ, **env}
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT], cwd=WORKDIR, env=full_env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fit failed: {r.stderr[-1000:]}")
    log(f"  fit {tag} done in {time.time()-t0:.0f}s")

    p = pd.read_parquet(preds_path)
    if years == (2022, 2023):
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
    m["p_mkt"] = devig_power(m["PSW"].values.astype(float), m["PSL"].values.astype(float))
    m = m.dropna(subset=["p_mkt"])
    mc = mc_average(m, n_samples=n_mc)
    m["p_hier_mc"] = mc
    preds_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    return m


def ll(x):
    pr = np.clip(x, 1e-9, 1 - 1e-9)
    return float(-np.log(pr).mean())


def score(df, label):
    e = ll(df.p_elo.values)
    h_raw = ll(df.p_winner_hier_mc.values) if "p_winner_hier_mc" in df else ll(df.p_hier_mc.values)
    log(f"  {label}: n={len(df)} elo={e:.4f} hier={h_raw:.4f} gap={h_raw-e:+.4f}")


if __name__ == "__main__":
    log("=== fitting v25 on 2022-23 tuning window ===")
    tune = fit_and_join((2022, 2023), "iso_tune", n_mc=30)
    tune["mkt_fav"] = np.where(tune.p_mkt >= .5, tune.p_mkt, 1 - tune.p_mkt)

    from sklearn.isotonic import IsotonicRegression
    wta_tune = tune[tune.tour == "wta"].copy()
    wta_tune["p_fav"] = np.where(wta_tune.p_hier_mc >= .5, wta_tune.p_hier_mc, 1 - wta_tune.p_hier_mc)
    wta_tune["y_correct"] = (wta_tune.p_hier_mc >= .5).astype(int)  # did the model's favourite win
    # Boundary-safe: calibrate p_fav in [0.5,1] -> P(favourite correct), y_min=0.5
    iso = IsotonicRegression(y_min=0.5, y_max=1.0, out_of_bounds="clip")
    iso.fit(wta_tune["p_fav"].values, wta_tune["y_correct"].values)

    def apply_iso(p_raw):
        p_raw = np.asarray(p_raw, dtype=float)
        fav = np.where(p_raw >= .5, p_raw, 1 - p_raw)
        side = np.where(p_raw >= .5, 1, -1)
        cal_fav = iso.predict(fav)
        # map back: cal_fav is P(favourite side correct); reconstruct P(original winner side)
        return np.where(side == 1, cal_fav, 1 - cal_fav)

    log("\n=== in-sample check on 2022-23 tuning window (WTA only) - no Elo comparison, "
        "elo_eval_predictions.parquet only covers 2024-25 ===")
    wta_tune["p_hier_cal"] = apply_iso(wta_tune["p_hier_mc"].values)
    log(f"  WTA raw hier logloss: {ll(wta_tune.p_hier_mc):.4f}")
    log(f"  WTA cal hier logloss: {ll(wta_tune.p_hier_cal):.4f}")
    lop = wta_tune[wta_tune.mkt_fav > 0.80]
    log(f"  WTA lopsided raw: hier={ll(lop.p_hier_mc):.4f}  n={len(lop)}")
    log(f"  WTA lopsided cal: hier={ll(lop.p_hier_cal):.4f}")

    log("\n=== REVEAL on real 2024-25 holdout (join pattern from RESUME-beat-elo.md) ===")
    j = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\real_holdout_v25_joined.parquet")
    j["mkt_fav"] = np.where(j.p_mkt >= .5, j.p_mkt, 1 - j.p_mkt)
    j["p_hier_cal"] = j["p_winner_hier_mc"].values
    wta_mask = j.tour == "wta"
    j.loc[wta_mask, "p_hier_cal"] = apply_iso(j.loc[wta_mask, "p_winner_hier_mc"].values)

    for label, sub in [("OVERALL", j), ("ATP", j[j.tour == "atp"]), ("WTA", j[j.tour == "wta"])]:
        e = ll(sub.p_elo.values)
        h_raw = ll(sub.p_winner_hier_mc.values)
        h_cal = ll(sub.p_hier_cal.values)
        mkt = ll(sub.p_mkt.values)
        log(f"  {label}: n={len(sub)}  elo={e:.4f}  hier_raw={h_raw:.4f} (gap {h_raw-e:+.4f})  "
            f"hier_calibrated={h_cal:.4f} (gap {h_cal-e:+.4f})  mkt={mkt:.4f}")

    lop = j[j.mkt_fav > 0.80]
    log(f"\n  lopsided-overall: raw={ll(lop.p_winner_hier_mc):.4f} cal={ll(lop.p_hier_cal):.4f} n={len(lop)}")
    lop_wta = j[(j.tour == "wta") & (j.mkt_fav > 0.80)]
    log(f"  lopsided-WTA: raw={ll(lop_wta.p_winner_hier_mc):.4f} cal={ll(lop_wta.p_hier_cal):.4f} n={len(lop_wta)}")
