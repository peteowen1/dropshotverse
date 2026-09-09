#!/bin/bash
# One-shot cutover for the 2026-09-09 input fixes. Run between two live-task
# runs (task fires at :04 and :34, takes ~90s). Every step is reversible:
# the legacy parquet and bag are kept beside the new ones.
set -euo pipefail
cd /c/dev/dropshotverse/dropshotdata
S=tune_scratch
test -f $S/xgb_bagged_model_cache_v3_inputfix/seed4.json
test -f $S/elo_raw_ratings_full.SURFACEFIX.parquet
# 1. Elo parquet: legacy (surface residual) kept, fixed one becomes canonical
cp $S/elo_raw_ratings_full.parquet $S/elo_raw_ratings_full.LEGACY_SURFACE_RESIDUAL.parquet
cp $S/elo_raw_ratings_full.SURFACEFIX.parquet $S/elo_raw_ratings_full.parquet
# 2. bag: old dir kept under a new name, new dir takes the canonical name
mv $S/xgb_bagged_model_cache_v2_setgame $S/xgb_bagged_model_cache_v2_setgame.LEGACY_2026-09-09
mv $S/xgb_bagged_model_cache_v3_inputfix $S/xgb_bagged_model_cache_v2_setgame
# 3. point-model default -> 0.0 (the bag above was trained with it)
sed -i 's/SURF_RW_SCALE = float(os.environ.get("DSV_SURF_RW_SCALE", "1.0"))/SURF_RW_SCALE = float(os.environ.get("DSV_SURF_RW_SCALE", "0.0"))/' scripts/hierarchical_model.py
grep -q 'DSV_SURF_RW_SCALE", "0.0"' scripts/hierarchical_model.py
echo "cutover done $(date)"
