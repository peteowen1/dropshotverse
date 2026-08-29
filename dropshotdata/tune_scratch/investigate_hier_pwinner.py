"""Investigate before trusting round 13's screen_hier_extra_state.py result
(Δ-0.00877, 30x bigger than anything else this session, hit the 2000-round
CV cap without early stopping - both classic red flags per stats-discipline
#2). Three checks: (1) is p_winner_hier's OWN standalone log-loss consistent
with ticket 10's known finding that the raw (non-MC-averaged) point estimate
is badly overconfident (~0.928)? (2) does the improvement plateau at a much
lower round count with a tighter cap, or does it genuinely keep improving?
(3) is hier's eval-row population (has_points==True) actually a DIFFERENT,
narrower population than Arm E's full training set implied?
"""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
hier_raw = pd.read_parquet(rf"{DD}\tune_scratch\v25_postfix_wide1995_preds.parquet")
print(f"v25_postfix_wide1995_preds.parquet: {len(hier_raw):,} rows total")
print("has_points-implied population - is this ALL matches or only point-stat ones?")

train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_widedata_hier_1995_2023.parquet")
print(f"xgb_features_widedata_hier_1995_2023.parquet (Arm E's training set): {len(train):,} rows")
print(f"v25_postfix_wide1995_preds row count == Arm E train row count? "
      f"{len(hier_raw) == len(train)} ({len(hier_raw):,} vs {len(train):,})")

# check 1: standalone calibration of p_winner_hier (winner-relative convention,
# so accuracy should be ~100% by construction - the real check is log-loss,
# which reveals overconfidence)
p = np.clip(hier_raw["p_winner_hier"].values, 1e-9, 1 - 1e-9)
ll_standalone = float(-np.log(p).mean())
print(f"\np_winner_hier standalone log-loss (P(actual winner), all eval rows): {ll_standalone:.4f}")
print("(ticket 10's known finding for the pre-MC-averaging point estimate: ~0.928 - "
      "badly overconfident but not necessarily leaky)")
print(f"min={p.min():.6f}  max={p.max():.6f}  mean={p.mean():.4f}")

# check 2: tighter CV re-run
extra = hier_raw[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])
j = train.merge(extra, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
p_hier_sideA = np.clip(np.where(y == 1, j["p_winner_hier"].values, 1 - j["p_winner_hier"].values), 1e-6, 1 - 1e-6)
j["hier_logit_pwinner"] = np.log(p_hier_sideA / (1 - p_hier_sideA))
print(f"\nmerge coverage: {j['hier_logit_pwinner'].notna().mean():.4f}")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": -1}


def cv_ll(numeric_cols, cap=400, esr=30):
    sub = j.dropna(subset=numeric_cols + CAT).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    hit_cap = (best + 1) >= cap
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), hit_cap


base_same, r0, n0, cap0 = cv_ll(BASE, cap=400, esr=30)
print(f"\nBASE (cap=400)                  round={r0:4d} (hit_cap={cap0}) cv_logloss={base_same:.5f}  n={n0}")
ll, r, n, cap1 = cv_ll(BASE + ["hier_logit_pwinner"], cap=400, esr=30)
print(f"BASE+hier_logit_pwinner (cap=400) round={r:4d} (hit_cap={cap1}) cv_logloss={ll:.5f}  "
      f"delta={ll - base_same:+.5f}  n={n}")
