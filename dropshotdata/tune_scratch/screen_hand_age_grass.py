import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
train = pd.read_parquet(rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")


def load_hand(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"{DD}\raw\github\{tour}\{tour}_matches_{y}.csv", low_memory=False,
                         usecols=["tourney_date", "winner_id", "loser_id", "winner_hand", "loser_hand"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["tour"] = tour
    d["date"] = pd.to_datetime(d["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        d[f"{side}_uid"] = d["tour"] + ":" + d[f"{side}_id"].astype("Int64").astype(str)
    return d[["date", "tour", "winner_uid", "loser_uid", "winner_hand", "loser_hand"]].drop_duplicates(
        subset=["date", "tour", "winner_uid", "loser_uid"])


years = list(range(1995, 2024))
hand = pd.concat([load_hand("atp", years), load_hand("wta", years)], ignore_index=True)
# xgb_features_v10_train.parquet already carries stale winner_hand/loser_hand
# columns from earlier round-14 work - drop before merging to avoid a
# _x/_y suffix collision.
train = train.drop(columns=[c for c in ("winner_hand", "loser_hand") if c in train.columns])
j = train.merge(hand, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)

hand_w = j["winner_hand"].fillna("U")
hand_l = j["loser_hand"].fillna("U")
sideA_left = np.where(y == 1, (hand_w == "L").astype(int), (hand_l == "L").astype(int))
sideB_left = np.where(y == 1, (hand_l == "L").astype(int), (hand_w == "L").astype(int))
mismatch = (sideA_left != sideB_left).astype(int)

atp_grass = ((j["tour"] == "atp") & (j["surface"] == "Grass")).values
j["lefty_mismatch_atp_grass"] = np.where(atp_grass, mismatch, 0.0)
j["sideA_left_atp_grass"] = np.where(atp_grass, sideA_left, 0.0)
j["sideB_left_atp_grass"] = np.where(atp_grass, sideB_left, 0.0)

j["age_diff_atp_grass"] = np.where(atp_grass, j["age_diff"].values, 0.0)

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


CANDS = {
    "hand_atp_grass": ["lefty_mismatch_atp_grass", "sideA_left_atp_grass", "sideB_left_atp_grass"],
    "age_atp_grass_extra": ["age_diff_atp_grass"],
}
for name, cols in CANDS.items():
    base_ll, r0, n0, cap0 = cv_ll(BASE, restrict_cols=cols)
    ll1, r1, n1, cap1 = cv_ll(BASE + cols)
    print(f"BASE+{name:24s} n={n1:7,d} round={r1:5d}(cap={cap1}) cv_logloss={ll1:.5f}  "
          f"vs same-rows base={base_ll:.5f}  delta={ll1 - base_ll:+.5f}", flush=True)
