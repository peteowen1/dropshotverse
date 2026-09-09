"""Four-arm comparison of the two input fixes found 2026-09-09, screened on
the 2022-23 tune window only (the 2024-25 holdout is burnt, 2026 is spent).
Same pipeline as xgb_bt_rating.train_bagged_model: train 1995-2021, frozen
v25 hyperparameters, 5-seed bag. Score on 2022-23 tour-level rows.

  arm      Elo parquet            SURF_RW_SCALE
  old      legacy (residual rs)   1.0
  elo      surface-expectation    1.0
  hier     legacy                 <scale>
  both     surface-expectation    <scale>

Usage: python compare_input_fixes.py <arm> <scale>
Env is set by the runner so each arm is a fresh process (the ELO path and
SURF_RW_SCALE are read at import time by the scripts under test)."""
import os, sys, time
from pathlib import Path
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T.parent / "scripts"))
arm, scale = sys.argv[1], sys.argv[2]
import numpy as np, pandas as pd, xgboost as xgb
import train_best_model as tbm
from xgb_bt_rating import merge_granularity_elo, prep_X_d, SCRATCH
log = tbm.log
TRAIN = list(range(1995, 2022)); EVAL = [2022, 2023]
def ll(y, p): p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

t0 = time.time()
log(f"=== arm={arm} scale={scale} ELO_RAW_PATH={tbm.ELO_RAW_PATH.name} DSV_SURF_RW_SCALE={os.environ.get('DSV_SURF_RW_SCALE')} ===")
def build(years, tag):
    feat = tbm.build_elo_features(years)
    feat = tbm.merge_hier(feat, tbm.regen_hier_state(years, tag))
    feat = tbm.merge_playstyle(feat, tbm.build_playstyle_features())
    feat = merge_granularity_elo(feat, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
    feat = merge_granularity_elo(feat, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")
    return feat
train = build(TRAIN, f"cmp_{arm}_train")
ev = build(EVAL, f"cmp_{arm}_eval")
log(f"train rows {len(train):,}, eval rows {len(ev):,}; elo_surface_diff coverage train "
    f"{train['elo_surface_diff'].notna().mean():.3f} eval {ev['elo_surface_diff'].notna().mean():.3f}")
cat_levels = {c: sorted(set(train[c].dropna().unique()) | set(ev[c].dropna().unique())) for c in tbm.XGB_CATEGORICAL}
dtrain = xgb.DMatrix(prep_X_d(train, cat_levels), label=train["y"].values.astype(int), enable_categorical=True)
dev = xgb.DMatrix(prep_X_d(ev, cat_levels), enable_categorical=True)
preds = []
for seed in range(tbm.N_BAGS):
    m = xgb.train({**tbm.XGB_BASE_PARAMS, "seed": seed}, dtrain, num_boost_round=tbm.NUM_BOOST_ROUND)
    preds.append(m.predict(dev)); log(f"  seed {seed} done ({time.time()-t0:.0f}s)")
p = np.mean(preds, axis=0); y = ev["y"].values.astype(int)
out = ev[["date", "tour", "surface", "winner_uid", "loser_uid", "y"]].copy(); out["p"] = p
out.to_parquet(T / f"cmp_{arm}_{scale}_eval_preds.parquet", index=False)
mkt = pd.read_parquet(T / "cond_tune_window_2223.parquet")[["date", "winner_uid", "loser_uid", "p_mkt"]].drop_duplicates(["date", "winner_uid", "loser_uid"])
j = out.merge(mkt, on=["date", "winner_uid", "loser_uid"], how="left")
log("\nRESULTS (log-loss, lower is better; 'mkt rows' = subset with market odds, market scored on same rows)")
for label, mask in (("ALL", np.ones(len(j), bool)), ("ATP", (j.tour == "atp").values), ("WTA", (j.tour == "wta").values),
                    ("Hard", (j.surface == "Hard").values), ("Clay", (j.surface == "Clay").values), ("Grass", (j.surface == "Grass").values)):
    mm = mask & j.p_mkt.notna().values
    pm = np.where(j.y.values[mm] == 1, j.p_mkt.values[mm], 1 - j.p_mkt.values[mm])
    log(f"{arm:5s} {label:6s} n={mask.sum():6,d} model={ll(y[mask], p[mask]):.5f} | mkt rows n={mm.sum():5,d} model={ll(y[mm], p[mm]):.5f} market={ll(np.ones(mm.sum(),int), pm):.5f}")
log(f"done in {time.time()-t0:.0f}s")
