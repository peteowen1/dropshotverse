"""Round 17: height was rejected in round 3 bundled with hand-mismatch, but
that rejection was driven by WTA's height missingness (~15-36%), not
necessarily a real ATP signal problem (ATP height missingness is <1%).
Height correlates directly with serve mechanical advantage, and grass is
exactly the surface where serve dominance matters most (round 15's own
diagnosis). Testing height_diff in isolation, targeted-masked to ATP-Grass
the same way serve/return form was (round 15's winning pattern) rather than
applied globally (which round 3 already showed fails due to WTA missingness)."""
import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")


def load_height(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"{DD}\raw\github\{tour}\{tour}_matches_{y}.csv", low_memory=False,
                         usecols=["tourney_date", "winner_id", "loser_id", "winner_ht", "loser_ht"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["tour"] = tour
    d["date"] = pd.to_datetime(d["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        d[f"{side}_uid"] = d["tour"] + ":" + d[f"{side}_id"].astype("Int64").astype(str)
    return d[["date", "tour", "winner_uid", "loser_uid", "winner_ht", "loser_ht"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])


years = list(range(1995, 2024))
ht = pd.concat([load_height("atp", years), load_height("wta", years)], ignore_index=True)
print(f"ATP height missing: {ht.loc[ht['tour']=='atp','winner_ht'].isna().mean():.4f}  "
      f"WTA height missing: {ht.loc[ht['tour']=='wta','winner_ht'].isna().mean():.4f}")

j = train.merge(ht, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
hw, hl = j["winner_ht"].values.astype(float), j["loser_ht"].values.astype(float)
ht_diff = np.where(y == 1, hw - hl, hl - hw)

atp_grass = (j["tour"] == "atp") & (j["surface"] == "Grass")
j["ht_diff_atp_grass"] = np.where(atp_grass, ht_diff, 0.0)
print(f"ATP-Grass rows: {atp_grass.sum()}  ht_diff coverage there: "
      f"{(~np.isnan(ht_diff[atp_grass.values])).mean():.4f}")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff", "hier_logit_pwinner",
        "serve_form_diff_targeted", "return_form_diff_targeted"]
CAT = ["surface", "round", "tourney_level", "tour"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.02,
          "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 5.0, "gamma": 1.0,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "seed": 0, "nthread": 8}


def cv_ll(numeric_cols, restrict_cols=None, cap=4000, esr=50):
    dropna_cols = numeric_cols + CAT if restrict_cols is None else numeric_cols + restrict_cols + CAT
    sub = j.dropna(subset=dropna_cols).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    cv = xgb.cv(PARAMS, d, num_boost_round=cap, nfold=5, early_stopping_rounds=esr,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub), (best + 1) >= cap


restrict = ["ht_diff_atp_grass"]
base_ll, r0, n0, cap0 = cv_ll(BASE, restrict_cols=restrict)
print(f"BASE (same rows)          round={r0:5d}(cap={cap0}) cv_logloss={base_ll:.5f}  n={n0:,}", flush=True)
ll1, r1, n1, cap1 = cv_ll(BASE + restrict)
print(f"BASE+ht_diff_atp_grass    round={r1:5d}(cap={cap1}) cv_logloss={ll1:.5f}  delta={ll1 - base_ll:+.5f}  n={n1:,}",
      flush=True)
