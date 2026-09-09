"""Gain and mean |SHAP| for the shipped bag (both 2026-09-09 fixes). SHAP via
xgboost pred_contribs on a 2022-23 feature table rebuilt the same way the
comparison arm built it (env: DSV_ELO_RAW_PATH unset -> canonical fixed
parquet, DSV_SURF_RW_SCALE default 0.0)."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
T = Path(__file__).resolve().parent; sys.path.insert(0, str(T.parent / "scripts"))
import train_best_model as tbm
from xgb_bt_rating import merge_granularity_elo, prep_X_d, train_bagged_model, SCRATCH
models, cat_levels, _ = train_bagged_model()
feat = tbm.build_elo_features([2022, 2023])
feat = tbm.merge_hier(feat, tbm.regen_hier_state([2022, 2023], "importance_2223"))
feat = tbm.merge_playstyle(feat, tbm.build_playstyle_features())
feat = merge_granularity_elo(feat, str(SCRATCH / "set_elo_raw_ratings_full.parquet"), "set")
feat = merge_granularity_elo(feat, str(SCRATCH / "game_elo_raw_ratings_full.parquet"), "game")
X = prep_X_d(feat, cat_levels); dm = xgb.DMatrix(X, enable_categorical=True)
gain = pd.DataFrame([m.get_score(importance_type="gain") for m in models]).mean().rename("gain")
cover = pd.DataFrame([m.get_score(importance_type="total_gain") for m in models]).mean().rename("total_gain")
contrib = np.mean([m.predict(dm, pred_contribs=True) for m in models], axis=0)[:, :-1]
shap = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns, name="mean_abs_shap")
out = pd.concat([shap, gain, cover], axis=1).fillna(0)
out["shap_share"] = out.mean_abs_shap / out.mean_abs_shap.sum(); out["gain_share"] = out.total_gain / out.total_gain.sum()
out = out.sort_values("mean_abs_shap", ascending=False)
print(f"n rows = {len(X):,} (2022-23). mean|SHAP| in logits; shares sum to 1.")
print(out[["mean_abs_shap","shap_share","gain_share"]].round(4).to_string())
