"""Does either Elo win survive into the model? (rounds 13/14, downstream test)

Two changes to the Elo engine measured on the 2022-23 tune window:
  round 13  score retirements at half K        -0.00044 log-loss
  round 14  faster-decaying K (the ridge)      -0.00062 log-loss

Both are improvements to ELO. The shipped predictor is XGBoost, which consumes
Elo as three features among nineteen and has already absorbed most of Elo's own
signal - ticket 28 round 8 found an XGBoost-Elo logit blend worth only
Delta-0.0001, an interior optimum that still did not transfer. So an Elo gain
does not imply a model gain, and this is the only measurement that decides
whether either result ships.

METHOD, and why it is a swap rather than a rebuild
Both arms use the EXACT same feature file, same rows, same folds, same
hyperparameters. The only thing that differs is the three Elo-derived columns
(elo_overall_diff, elo_surface_diff, elo_logit_pwinner) plus n_prior_diff, each
recomputed from the compact harness under the arm's Elo configuration. Every
non-Elo feature - hier state, layoff, h2h, surface/round/level - is byte
identical between arms, so a difference in CV score is attributable to the Elo
change alone (stats-discipline #5).

LIMITING CASE, asserted before any arm is compared (#5 rule 2)
Rebuilding the Elo columns at the SHIPPED configuration must reproduce
xgb_features_tune.parquet's own columns. If it does not, my side-assignment or
state extraction differs from build_xgb_features.py and every delta below is
meaningless. Checked to 1e-9 on all four columns; the run aborts otherwise.

SCOPE NOTE, stated rather than buried
xgb_features_tune.parquet is the v1 19-feature set, not shipped config D (which
adds set-Elo, game-Elo, age_diff, fatigue_diff and serve/return form). Config D
carries set- and game-level Elo that would absorb some of the same signal, so a
gain measured here is an UPPER BOUND on what config D would see, not an
estimate of it. A null here is therefore decisive; a win here is not.

Tune window only. The 2024-25 holdout is retired (docs/DECISIONS.md D-10).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import build_compact, run_elo_fast, KIND_OK  # noqa: E402

FEAT = DD / "tune_scratch" / "xgb_features_tune.parquet"
OUT = DD / "tune_scratch" / "screen_xgb_downstream.parquet"

ELO_COLS = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "n_prior_diff"]
CATS = ["surface", "round", "tourney_level", "tour"]

PARAMS = dict(objective="binary:logistic", eval_metric="logloss",
              max_depth=3, eta=0.02, subsample=0.7, colsample_bytree=0.9,
              min_child_weight=5.0, gamma=1.0, reg_alpha=0.1, reg_lambda=1.0,
              tree_method="hist", nthread=4)
NROUND, NFOLD, SEED = 3000, 5, 20260905

ARMS = {
    "baseline (shipped Elo)":      dict(ret_update=0.0, k_base=250.0, k_shape=0.4, k_min=16.0),
    "round13 retirements @0.5":    dict(ret_update=0.5, k_base=250.0, k_shape=0.4, k_min=16.0),
    "round14 K ridge 350/0.5/16":  dict(ret_update=0.0, k_base=350.0, k_shape=0.5, k_min=16.0),
    "both":                        dict(ret_update=0.5, k_base=350.0, k_shape=0.5, k_min=16.0),
}


def log(m: str = "") -> None:
    print(m, flush=True)


def side_a_is_winner(date: pd.Timestamp, w: str, l: str) -> bool:
    """Byte-for-byte the rule in build_xgb_features.py - outcome-blind hash of
    (date, sorted uid pair)."""
    key = f"{date.date()}|{'|'.join(sorted([w, l]))}"
    return (hashlib.sha256(key.encode()).digest()[0] % 2) == 0


def elo_columns(d: dict, arm: dict, feat: pd.DataFrame) -> pd.DataFrame:
    """Recompute the four Elo-derived features for `feat`'s rows under `arm`."""
    st = run_elo_fast(d, blend=0.0, decay=0.3, return_state=True, **arm)
    ok = d["kind"] == KIND_OK
    uids = d["uids"]
    src = pd.DataFrame({
        "date": d["day"][ok].astype("datetime64[D]"),
        "winner_uid": uids[d["w"][ok]],
        "loser_uid": uids[d["l"][ok]],
        "ro_w": st["ro_w"][ok], "ro_l": st["ro_l"][ok],
        "rs_w": st["rs_w"][ok], "rs_l": st["rs_l"][ok],
        "n_w": st["n_all_w"][ok], "n_l": st["n_all_l"][ok],
        "p_w": st["p_winner"][ok],
    })
    key = ["date", "winner_uid", "loser_uid"]
    src = src[~src.duplicated(key, keep=False)]
    j = feat[key].merge(src, on=key, how="left")

    sa = np.array([side_a_is_winner(dt, w, l)
                   for dt, w, l in zip(j["date"], j["winner_uid"], j["loser_uid"])])
    p = np.clip(np.where(sa, j["p_w"], 1 - j["p_w"]), 1e-6, 1 - 1e-6)
    return pd.DataFrame({
        "elo_overall_diff": np.where(sa, j["ro_w"] - j["ro_l"], j["ro_l"] - j["ro_w"]),
        "elo_surface_diff": np.where(sa, j["rs_w"] - j["rs_l"], j["rs_l"] - j["rs_w"]),
        "elo_logit_pwinner": np.log(p / (1 - p)),
        "n_prior_diff": np.where(sa, j["n_w"] - j["n_l"], j["n_l"] - j["n_w"]).astype(float),
    })


def cv(feat: pd.DataFrame) -> tuple[float, int]:
    X = feat.drop(columns=["date", "tour", "window", "winner_uid", "loser_uid", "y"],
                  errors="ignore").copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    dm = xgb.DMatrix(X, label=feat["y"].values, enable_categorical=True)
    r = xgb.cv(PARAMS, dm, num_boost_round=NROUND, nfold=NFOLD, seed=SEED,
               early_stopping_rounds=100, verbose_eval=False, shuffle=False)
    return float(r["test-logloss-mean"].min()), int(r["test-logloss-mean"].idxmin())


def main() -> None:
    feat = pd.read_parquet(FEAT)
    feat["tour"] = feat["tour"].astype(str)
    log(f"coverage: {len(feat):,} rows, {feat.date.min().date()} to {feat.date.max().date()}, "
        f"y mean {feat.y.mean():.4f}")

    d = build_compact()

    # ---- limiting case, before any comparison
    log("\n--- limiting case: rebuild the shipped Elo columns and compare ---")
    base_cols = elo_columns(d, ARMS["baseline (shipped Elo)"], feat)
    worst = 0.0
    for c in ELO_COLS:
        delta = (base_cols[c].values - feat[c].values)
        finite = np.isfinite(delta)
        m = float(np.nanmax(np.abs(delta[finite]))) if finite.any() else np.nan
        worst = max(worst, m)
        log(f"  {c:<22} max|delta| = {m:.3e}   (n_finite {int(finite.sum()):,})")
    log(f"  -> {'PASS' if worst < 1e-9 else 'FAIL'}")
    if not (worst < 1e-9):
        log("  Side assignment or state extraction differs from "
            "build_xgb_features.py. Every delta below would be meaningless.")
        sys.exit(1)

    # ---- arms
    log("\n--- arms (identical rows, folds, params; only Elo columns differ) ---")
    rows = []
    for name, arm in ARMS.items():
        f = feat.copy()
        f[ELO_COLS] = elo_columns(d, arm, feat)[ELO_COLS].values
        ll, best_it = cv(f)
        rows.append(dict(arm=name, cv_logloss=ll, best_iter=best_it, **arm))
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        log(f"  {name:<30} cv={ll:.6f}  (best_iter {best_it})")

    tab = pd.DataFrame(rows)
    b = float(tab.loc[tab.arm.str.startswith("baseline"), "cv_logloss"].iloc[0])
    tab["delta"] = tab["cv_logloss"] - b
    log("\n--- verdict (delta vs baseline) ---")
    log(tab[["arm", "cv_logloss", "delta"]].to_string(index=False,
        float_format=lambda x: f"{x:+.6f}"))
    log("\n  reference: the same changes were worth -0.00044 (round 13) and")
    log("  -0.00062 (round 14) on Elo's OWN log-loss. Attenuation here is the")
    log("  expected result, not a surprise - the question is whether anything")
    log("  survives at all.")
    log("DOWNSTREAM_COMPLETE")


if __name__ == "__main__":
    main()
