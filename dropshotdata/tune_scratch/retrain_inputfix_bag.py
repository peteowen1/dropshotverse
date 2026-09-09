"""Retrain the shipped 5-seed bag with both 2026-09-09 input fixes into a
NEW cache dir, so the live task keeps loading the old bag until the swap.
Env: DSV_ELO_RAW_PATH=<SURFACEFIX parquet>, DSV_SURF_RW_SCALE=0.0."""
import os, sys, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import xgb_bt_rating as xbr
assert os.environ.get("DSV_SURF_RW_SCALE") == "0.0" and "SURFACEFIX" in os.environ.get("DSV_ELO_RAW_PATH", ""), "env not set"
xbr.MODEL_CACHE_DIR = xbr.SCRATCH / "xgb_bagged_model_cache_v3_inputfix"
t0 = time.time(); models, cat_levels, _ = xbr.train_bagged_model(use_cache=False)
print(f"trained {len(models)} seeds into {xbr.MODEL_CACHE_DIR} in {time.time()-t0:.0f}s")
