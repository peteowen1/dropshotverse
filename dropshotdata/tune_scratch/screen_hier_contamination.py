"""Issue #1: does the hierarchical model's tail-calibration bug reach the
shipped XGBoost model through `hier_logit_pwinner`?

Standalone, hierarchical_model.py assigns the eventual winner < 1% in 0.78% of
2024-25 holdout matches and scores worse than random (0.72388 vs 0.69315).
XGBoost consumes its output as one feature of 19. Never tested directly whether
the defect survives the fit or is absorbed. Two questions, two tests
(stats-discipline #6):

  A. Does the shipped model's OUTPUT have the tail defect? Same 2024-25 rows,
     hier standalone vs the shipped 5-seed bag: share of winners assigned
     < 10% and < 1%, and log-loss share of the worst 0.5%. This is the
     contamination question, answered on identical rows.
  B. Does the shipped model NEED the feature? Paired 5 chronological folds on
     the shipped 1995-2023 train table, rounds pinned at 2528 (the shipped
     value, so stopping point is not a free axis), arms: all features / drop
     hier_logit_pwinner / drop all four hier columns. Per-fold log-loss and
     per-fold tail shares, differences vs the full arm.

Anchors, written before running:
  A1. hier standalone log-loss on the joined rows ~ 0.7239 (issue #1's number)
  A2. shipped bag log-loss on all holdout rows within 0.01 of 0.5932 (that
      number was on the 8,294-row odds-joined subset)
  B1. dropping hier_logit_pwinner costs between 0 and +0.010 (round 13 put it
      at -0.00941 when added)
Predicted: if the fit absorbs the defect, winners < 1% under the shipped model
are well under 0.1% and do not move across arms.

Caveat on B: contiguous folds mean early folds are scored by a model trained
partly on later years. Same design as screen_xgb_paired_folds.py; fine for a
paired ablation, not a walk-forward estimate of absolute error.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
SCRATCH = DD / "tune_scratch"
sys.path.insert(0, str(DD / "scripts"))
from train_best_model import (NUM_BOOST_ROUND, XGB_BASE_PARAMS,  # noqa: E402
                              XGB_CATEGORICAL, XGB_NUMERIC)

KEY = ["date", "tour", "winner_uid", "loser_uid"]
HIER_COLS = ["hier_dmean_diff", "hier_dvar_diff", "hier_dvar_sum", "hier_logit_pwinner"]
ARMS = {
    "full (shipped)": [],
    "drop hier_logit_pwinner": ["hier_logit_pwinner"],
    "drop all 4 hier cols": HIER_COLS,
}
NFOLD = 5
OUT = SCRATCH / "screen_hier_contamination.parquet"


def log(m: str = "") -> None:
    print(m, flush=True)


def ll(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tail(y, p) -> dict:
    """p is P(side A wins); y is 1 if side A won. Winner's assigned prob."""
    pw = np.where(y == 1, p, 1 - p)
    pw = np.clip(pw, 1e-9, 1)
    contrib = -np.log(pw)
    k = max(1, int(round(0.005 * len(pw))))
    worst = np.sort(contrib)[-k:].sum() / contrib.sum()
    return dict(n=len(pw), ll=float(contrib.mean()),
                lt10=float((pw < 0.10).mean()), lt01=float((pw < 0.01).mean()),
                worst_half_pct_share=float(worst))


def fmt(t: dict) -> str:
    return (f"n={t['n']:,} ll={t['ll']:.5f} winner<10%={t['lt10']:.2%} "
            f"winner<1%={t['lt01']:.2%} worst0.5%-share={t['worst_half_pct_share']:.1%}")


# ---------------------------------------------------------------- Part A
def part_a() -> list[dict]:
    log("=== A. shipped model output vs hier standalone, identical 2024-25 rows ===")
    hold = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")[KEY + ["y"]]
    bag = pd.read_parquet(SCRATCH / "bag_v10_seed_preds.parquet")
    hier = pd.read_parquet(SCRATCH / "v25_postfix_holdout_preds.parquet")[KEY + ["p_winner_hier"]]
    for d in (hold, bag, hier):
        d["date"] = pd.to_datetime(d["date"])
        d["tour"] = d["tour"].astype(str)
    log(f"  coverage: holdout {len(hold):,}, bag preds {len(bag):,}, hier preds {len(hier):,}")
    bag["p_bag"] = bag[[f"p_seed{i}" for i in range(5)]].mean(axis=1)
    j = hold.merge(bag[KEY + ["p_bag"]], on=KEY, how="inner")
    log(f"  holdout x bag: {len(j):,} rows")
    j = j.merge(hier, on=KEY, how="inner").drop_duplicates(subset=KEY)
    log(f"  ... x hier: {len(j):,} rows (dropped = not in one of the three tables)")
    y = j["y"].values.astype(int)
    # p_winner_hier is already the WINNER's probability; convert to side-A form
    p_hier_a = np.where(y == 1, j["p_winner_hier"].values, 1 - j["p_winner_hier"].values)
    t_bag, t_hier = tail(y, j["p_bag"].values), tail(y, p_hier_a)
    log(f"  hier standalone : {fmt(t_hier)}")
    log(f"  shipped 5-bag   : {fmt(t_bag)}")
    log(f"  anchor A1 hier ll ~0.7239   -> {t_hier['ll']:.5f} "
        f"{'PASS' if abs(t_hier['ll'] - 0.7239) < 0.01 else 'FAIL'}")
    log(f"  anchor A2 bag ll ~0.5932+-0.01 -> {t_bag['ll']:.5f} "
        f"{'PASS' if abs(t_bag['ll'] - 0.5932) < 0.01 else 'FAIL'}")
    # do the two models' worst matches coincide? if XGB inherited the defect,
    # hier's worst-50 should also be XGB's worst.
    j["c_hier"] = -np.log(np.clip(j["p_winner_hier"], 1e-9, 1))
    j["c_bag"] = -np.log(np.clip(np.where(y == 1, j["p_bag"], 1 - j["p_bag"]), 1e-9, 1))
    w = j.nlargest(50, "c_hier")
    log(f"  hier's 50 worst matches: hier gives winner mean {np.exp(-w['c_hier']).mean():.3%}, "
        f"shipped model gives them mean {np.exp(-w['c_bag']).mean():.1%} "
        f"(min {np.exp(-w['c_bag']).min():.1%})")
    log(f"  rank correlation of per-match log-loss contribution (spearman): "
        f"{j['c_hier'].corr(j['c_bag'], method='spearman'):.3f}")
    return [dict(part="A", arm="hier standalone", **t_hier),
            dict(part="A", arm="shipped 5-bag", **t_bag)]


# ---------------------------------------------------------------- Part B
def part_b() -> list[dict]:
    log("\n=== B. paired 5-fold ablation on the shipped train table ===")
    tr = pd.read_parquet(SCRATCH / "xgb_features_v10_train.parquet")
    tr = tr.sort_values("date").reset_index(drop=True)
    hold = pd.read_parquet(SCRATCH / "xgb_features_v10_holdout.parquet")
    cat_levels = {c: sorted(set(tr[c].dropna().unique()) | set(hold[c].dropna().unique()))
                  for c in XGB_CATEGORICAL}
    y = tr["y"].values.astype(int)
    n = len(tr)
    log(f"  coverage: {n:,} rows {tr['date'].min().date()}..{tr['date'].max().date()}, "
        f"tours {tr['tour'].value_counts().to_dict()}, y mean {y.mean():.3f}")
    log(f"  rounds pinned at {NUM_BOOST_ROUND}, seed 0, params = shipped")
    bounds = [(i * n // NFOLD, (i + 1) * n // NFOLD) for i in range(NFOLD)]
    for i, (lo, hi) in enumerate(bounds):
        log(f"  fold {i}: test rows {lo:,}-{hi:,} = {tr['date'].iloc[lo].date()}..{tr['date'].iloc[hi-1].date()}")

    per_fold = {}
    oof = {}
    for arm, drop in ARMS.items():
        cols = [c for c in XGB_NUMERIC if c not in drop]
        X = tr[cols].copy()
        for c in XGB_CATEGORICAL:
            X[c] = pd.Categorical(tr[c], categories=cat_levels[c])
        p_oof = np.full(n, np.nan)
        scores = []
        for i, (lo, hi) in enumerate(bounds):
            te = np.zeros(n, dtype=bool)
            te[lo:hi] = True
            t0 = time.time()
            dtr = xgb.DMatrix(X[~te], label=y[~te], enable_categorical=True)
            dte = xgb.DMatrix(X[te], enable_categorical=True)
            bst = xgb.train({**XGB_BASE_PARAMS, "seed": 0}, dtr,
                            num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)
            p_oof[te] = bst.predict(dte)
            scores.append(ll(y[te], p_oof[te]))
            log(f"    {arm:<26} fold {i}: ll {scores[-1]:.5f}  ({time.time()-t0:.0f}s)")
        per_fold[arm] = scores
        oof[arm] = p_oof
        log(f"  {arm:<26} OOF: {fmt(tail(y, p_oof))}")

    base = np.array(per_fold["full (shipped)"])
    recs = []
    log("\n--- paired per-fold deltas vs full (positive = worse without) ---")
    for arm in ARMS:
        s = np.array(per_fold[arm])
        d = s - base
        t = tail(y, oof[arm])
        rec = dict(part="B", arm=arm, mean_ll=float(s.mean()), mean_delta=float(d.mean()),
                   sd_delta=float(d.std(ddof=1)) if arm != "full (shipped)" else 0.0,
                   folds_worse=int((d > 0).sum()), **t)
        recs.append(rec)
        if arm == "full (shipped)":
            log(f"  {arm:<26} folds " + " ".join(f"{v:.5f}" for v in s))
            continue
        se = rec["sd_delta"] / np.sqrt(NFOLD)
        log(f"  {arm:<26} delta " + " ".join(f"{v:+.5f}" for v in d)
            + f"  mean {d.mean():+.6f} se {se:.6f} t {d.mean()/se if se else float('nan'):+.2f} "
              f"folds worse {rec['folds_worse']}/{NFOLD}")
    d1 = recs[1]["mean_delta"]
    log(f"  anchor B1 drop hier_logit_pwinner costs 0..+0.010 -> {d1:+.5f} "
        f"{'PASS' if 0 <= d1 <= 0.010 else 'FAIL'}")
    return recs


def main() -> None:
    recs = part_a()
    recs += part_b()
    pd.DataFrame(recs).to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")
    log("HIER_CONTAMINATION_COMPLETE")


if __name__ == "__main__":
    main()
