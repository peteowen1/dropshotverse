"""Issue 26: xgb.cv on the 2022-23 tune window (num_boost_round selection via
early stopping only - no hyperparameter grid search, per Pete's own framing:
"chosen default parameters tuned to optimal nrounds"), one final fit on all
of 2022-23 with the selected round count, scored exactly ONCE on the
2024-25 market-odds-joined holdout (n=8,594, same rows as every other
candidate this session).

Label-equivalence note: y=1 means "side A (the outcome-blind hash-assigned
side) actually won". Standard binary log-loss/accuracy/Brier computed
directly against (y, p_pred) is mathematically IDENTICAL to Elo/hier's own
"P(assigned to the actual winner)" convention - both are just
-mean(y*log(p)+(1-y)*log(1-p)) under a relabeling - so no extra conversion
step is needed for the comparison to be apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DD = r"C:\dev\dropshotverse\dropshotdata"
TUNE = rf"{DD}\tune_scratch\xgb_features_tune.parquet"
HOLDOUT = rf"{DD}\tune_scratch\xgb_features_holdout.parquet"
DIAG2 = rf"{DD}\tune_scratch\diag2_enriched.parquet"       # source of the odds-joined n=8,594 key set
ELO_EVAL = rf"{DD}\elo_eval_predictions.parquet"            # post-fix Elo, 0.6120 basis

NUMERIC = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
           "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
           "h2h_diff", "best_of"]
CATEGORICAL = ["surface", "round", "tourney_level"]
BOOL_AS_INT = ["is_elite"]

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 0,
    "nthread": -1,
}
NUM_BOOST_ROUND_CAP = 3000
EARLY_STOPPING_ROUNDS = 50
NFOLD = 5


def log(m: str) -> None:
    print(m, flush=True)


def ll(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


def prep_X(df: pd.DataFrame, cat_levels: dict[str, list] | None = None) -> pd.DataFrame:
    """cat_levels, when given, fixes each categorical column's category set to
    a shared union (computed once across tune+holdout) so a level present
    only in the holdout (e.g. a tourney_level value absent from 2022-23) is
    still a valid category at predict time rather than raising."""
    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        if cat_levels is not None:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        else:
            X[c] = df[c].astype("category")
    for c in BOOL_AS_INT:
        X[c] = df[c].astype(int)
    return X


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "n": len(y),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def main() -> None:
    log("loading tune-window features (2022-2023)...")
    tune = pd.read_parquet(TUNE)
    holdout_raw = pd.read_parquet(HOLDOUT)
    # shared categorical levels across tune+holdout, computed once, so a
    # tourney_level/round/surface value seen only in one window doesn't
    # crash predict() on the other.
    cat_levels = {c: sorted(set(tune[c].dropna().unique()) | set(holdout_raw[c].dropna().unique()))
                  for c in CATEGORICAL}
    X_tune = prep_X(tune, cat_levels)
    y_tune = tune["y"].values.astype(int)
    log(f"  tune rows: {len(tune):,}  columns: {list(X_tune.columns)}")

    dtrain = xgb.DMatrix(X_tune, label=y_tune, enable_categorical=True)

    log(f"\nrunning xgb.cv (nfold={NFOLD}, early_stopping_rounds={EARLY_STOPPING_ROUNDS}, "
        f"cap={NUM_BOOST_ROUND_CAP})...")
    cv_results = xgb.cv(
        params=PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND_CAP,
        nfold=NFOLD,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        metrics=("logloss",),
        seed=0,
        as_pandas=True,
        verbose_eval=False,
    )
    best_round = int(cv_results["test-logloss-mean"].idxmin()) + 1  # 1-indexed round count
    best_cv_ll = float(cv_results["test-logloss-mean"].iloc[best_round - 1])
    log(f"  cv ran {len(cv_results)} rounds (stopped by early stopping or cap)")
    log(f"  selected num_boost_round={best_round}  (cv test-logloss={best_cv_ll:.5f})")

    log(f"\nfitting final model on ALL of tune (2022-23) with num_boost_round={best_round}...")
    final_model = xgb.train(PARAMS, dtrain, num_boost_round=best_round)

    log("\nrestricting holdout-window features (2024-2025) to the "
        "market-odds-joined n=8,594 subset (same rows used throughout this session)...")
    holdout = holdout_raw
    diag2 = pd.read_parquet(DIAG2)
    odds_key = diag2.loc[diag2["p_mkt"].notna(), ["date", "tour", "winner_uid", "loser_uid"]].drop_duplicates()
    odds_key["_in_odds"] = True
    holdout_j = holdout.merge(odds_key, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    holdout_j = holdout_j[holdout_j["_in_odds"] == True].drop(columns=["_in_odds"])  # noqa: E712
    holdout_j = holdout_j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    log(f"  holdout rows after odds-join restriction: {len(holdout_j):,} "
        f"({'OK matches n=8,594' if len(holdout_j) == 8594 else 'WARNING: does not match expected 8,594'})")

    X_hold = prep_X(holdout_j, cat_levels)
    dhold = xgb.DMatrix(X_hold, enable_categorical=True)
    p_xgb = final_model.predict(dhold)
    y_hold = holdout_j["y"].values.astype(int)

    # ---- comparison basis: Elo (post-fix, 0.6120) and hier (v25+round9, 0.6275)
    # on the identical rows, joined the same way this session's harness does.
    elo = pd.read_parquet(ELO_EVAL)
    elo = elo[elo["level_file"] == "tour"][["date", "tour", "winner_uid", "loser_uid", "p_winner"]]
    elo = elo.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    cmp_df = holdout_j[["date", "tour", "winner_uid", "loser_uid", "y"]].merge(
        elo, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cmp_df = cmp_df.merge(
        diag2[["date", "tour", "winner_uid", "loser_uid", "p_winner_hier_mc"]].drop_duplicates(
            subset=["date", "tour", "winner_uid", "loser_uid"]),
        on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cmp_df["p_xgb"] = p_xgb
    log(f"  elo/hier comparison coverage: elo notna={cmp_df['p_winner'].notna().sum():,}, "
        f"hier notna={cmp_df['p_winner_hier_mc'].notna().sum():,} / {len(cmp_df):,}")

    # elo/hier's p_winner/p_winner_hier_mc are already "P(assigned to actual
    # winner)" - convert to the same y-relative convention as p_xgb for a
    # uniform scoring function: p_actual_winner if y==1 means side A (== the
    # actual winner here) so p_for_y1 = p_winner as-is when y==1... simpler:
    # elo/hier predictions ARE p(actual winner), and our y is 1 exactly when
    # side A is the actual winner - so p_model_for_y1 = p_winner when y==1,
    # and = 1-p_winner when y==0 (side B is the actual winner, so P(side A
    # wins) = 1 - P(actual winner wins)).
    y_arr = cmp_df["y"].values.astype(int)
    p_elo_sideA = np.where(y_arr == 1, cmp_df["p_winner"].values, 1 - cmp_df["p_winner"].values)
    p_hier_sideA = np.where(y_arr == 1, cmp_df["p_winner_hier_mc"].values, 1 - cmp_df["p_winner_hier_mc"].values)

    def report(label: str, mask: np.ndarray) -> None:
        n = int(mask.sum())
        log(f"\n  --- {label} (n={n:,}) ---")
        for name, arr in (("XGBoost", p_xgb[mask]), ("Elo (post-fix)", p_elo_sideA[mask]),
                          ("Hierarchical (v25/mc60)", p_hier_sideA[mask])):
            s = score(y_arr[mask], arr)
            log(f"    {name:26s} logloss={s['logloss']:.4f}  acc={s['accuracy']:.4f}  brier={s['brier']:.4f}")

    tour_arr = cmp_df["tour"].values
    log("\n=== HOLDOUT RESULTS (2024-2025, market-odds-joined, n={:,}) ===".format(len(cmp_df)))
    report("ALL", np.ones(len(cmp_df), dtype=bool))
    report("ATP", tour_arr == "atp")
    report("WTA", tour_arr == "wta")

    # ---- feature importance (gain-based)
    log("\n=== FEATURE IMPORTANCE (gain) ===")
    gain = final_model.get_score(importance_type="gain")
    total_gain = final_model.get_score(importance_type="total_gain")
    weight = final_model.get_score(importance_type="weight")
    imp = pd.DataFrame({
        "feature": list(gain.keys()),
        "avg_gain": list(gain.values()),
    }).set_index("feature")
    imp["total_gain"] = pd.Series(total_gain)
    imp["n_splits"] = pd.Series(weight)
    imp = imp.sort_values("total_gain", ascending=False)
    log(imp.to_string())

    final_model.save_model(rf"{DD}\tune_scratch\xgb_v1_model.json")
    holdout_j.assign(p_xgb=p_xgb).to_parquet(rf"{DD}\tune_scratch\xgb_v1_holdout_predictions.parquet")
    imp.to_csv(rf"{DD}\tune_scratch\xgb_v1_feature_importance.csv")
    log(f"\nsaved model, holdout predictions, and feature importance to tune_scratch\\xgb_v1_*")


if __name__ == "__main__":
    main()
