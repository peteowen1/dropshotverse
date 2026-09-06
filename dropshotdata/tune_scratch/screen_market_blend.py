"""W1: does conditioning on the bookmaker price actually help, and where?

The critique's claim is that the repo holds prices it only ever uses as a
scorecard, and that blending them in is the shortest path to the project's
stated goal of calibration parity with the market.

Tested under docs/plans/validation-charter.md, on the charter's own folds and
rows, so the answer is comparable to the shipped config's scorecard.

DESIGN, and why the first testable year is 2020 rather than 2019.
The blend weight w must be fitted on model predictions the model did not train
on, or w is fitted to memorised rows and will not survive. So this runs the
charter's walk-forward first to collect OUT-OF-SAMPLE predictions for every
year 2019-2025, then for each test year Y fits w on the pooled out-of-sample
rows of years < Y and applies it to Y. 2019 has no earlier out-of-sample year,
so it is a fit year only. Six test folds.

  logit(p_blend) = w * logit(p_market) + (1 - w) * logit(p_model)

Reported per year: the fitted w, and log-loss for the model alone, the market
alone and the blend. Two pre-registered rules are applied, both written before
this ran:
  - the critique's own W1 rule: blend beats MODEL by >= 0.015 on 5 of 7 folds
    and is never worse than the MARKET by more than 0.003; under 0.005 means
    the two signals are near-collinear and the work is not worth doing.
  - the charter's standing rule, for comparability with every other candidate.

It also answers the question the critique raised but did not size: what share
of the pairs the LIVE simulator needs could ever carry a price.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import _common                                                          # noqa: E402
from train_best_model import (XGB_BASE_PARAMS, XGB_CATEGORICAL,         # noqa: E402
                              XGB_NUMERIC, NUM_BOOST_ROUND)
from walk_forward_yearly import (TEST_YEARS, design, ece, join_market,   # noqa: E402
                                 load_features, logloss)

OUT = _common.SCRATCH / "screen_market_blend.parquet"
EPS = 1e-9


def log(m: str = "") -> None:
    print(m, flush=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def blend(p_model, p_mkt, w):
    return 1.0 / (1.0 + np.exp(-((1 - w) * logit(p_model) + w * logit(p_mkt))))


def fit_w(p_model, p_mkt, y) -> float:
    """w in [0,1] minimising log-loss on the fitting rows."""
    r = minimize_scalar(lambda w: logloss(blend(p_model, p_mkt, w), y),
                        bounds=(0.0, 1.0), method="bounded")
    return float(r.x)


def main() -> None:
    log("=== W1: is the bookmaker price worth conditioning on? ===")
    df = load_features()
    mkt = join_market(df)
    df = df.merge(mkt, on=["date", "tour", "winner_uid", "loser_uid"], how="left")

    cols = XGB_NUMERIC + XGB_CATEGORICAL
    cat_levels = {c: sorted(df[c].dropna().unique()) for c in XGB_CATEGORICAL}

    # 1. out-of-sample predictions for every charter year
    oos = []
    for Y in TEST_YEARS:
        tr = df[df["year"] < Y]
        te = df[(df["year"] == Y) & df["p_mkt"].notna()]
        dtr = xgb.DMatrix(design(tr, cols, cat_levels), label=tr["y"].values.astype(int),
                          enable_categorical=True)
        bst = xgb.train({**XGB_BASE_PARAMS, "seed": 0}, dtr, num_boost_round=NUM_BOOST_ROUND)
        p = bst.predict(xgb.DMatrix(design(te, cols, cat_levels), enable_categorical=True))
        oos.append(pd.DataFrame({"year": Y, "y": te["y"].values.astype(int),
                                 "p_model": p, "p_mkt": te["p_mkt"].values}))
        log(f"  {Y}: {len(te):,} out-of-sample rows")
    oos = pd.concat(oos, ignore_index=True)

    # 2. walk the blend forward: fit w on years < Y, apply to Y
    rows = []
    for Y in TEST_YEARS[1:]:
        fit = oos[oos["year"] < Y]
        te = oos[oos["year"] == Y]
        w = fit_w(fit["p_model"].values, fit["p_mkt"].values, fit["y"].values)
        pb = blend(te["p_model"].values, te["p_mkt"].values, w)
        rec = dict(year=Y, n=len(te), w=w,
                   ll_model=logloss(te["p_model"], te["y"]),
                   ll_market=logloss(te["p_mkt"], te["y"]),
                   ll_blend=logloss(pb, te["y"]),
                   ece_model=ece(te["p_model"], te["y"]), ece_blend=ece(pb, te["y"]),
                   # the market's OWN calibration. Without it, "the blend beats
                   # the market" cannot be told apart from "the de-vigged market
                   # is slightly miscalibrated and the blend corrects it".
                   ece_market=ece(te["p_mkt"], te["y"]))
        rec["gain_vs_model"] = rec["ll_model"] - rec["ll_blend"]
        rec["gap_vs_market"] = rec["ll_blend"] - rec["ll_market"]
        rows.append(rec)
        log(f"\n{Y}: w fitted on {len(fit):,} earlier out-of-sample rows = {w:.3f}")
        log(f"  log-loss  model {rec['ll_model']:.5f}  market {rec['ll_market']:.5f}  "
            f"blend {rec['ll_blend']:.5f}")
        log(f"  blend gains {rec['gain_vs_model']:+.5f} on the model, "
            f"sits {rec['gap_vs_market']:+.5f} vs the market")

    res = pd.DataFrame(rows)
    log("\n=== blend scorecard ===")
    log(res[["year", "n", "w", "ll_model", "ll_market", "ll_blend",
             "gain_vs_model", "gap_vs_market", "ece_model", "ece_market",
             "ece_blend"]].to_string(index=False))
    mean_gain = float(res["gain_vs_model"].mean())
    log(f"\nmean w {res['w'].mean():.3f}  |  mean gain on the model {mean_gain:+.5f}  |  "
        f"mean gap to market {res['gap_vs_market'].mean():+.5f}")
    log(f"beats the model in {int((res['gain_vs_model'] > 0).sum())}/{len(res)} years; "
        f"beats the market in {int((res['gap_vs_market'] < 0).sum())}/{len(res)}")
    log(f"mean ECE: model {res['ece_model'].mean():.4f}  market {res['ece_market'].mean():.4f}  "
        f"blend {res['ece_blend'].mean():.4f}")
    # 2020 is COVID-shortened (n=1,931) and is the biggest single contributor to
    # the market-beating mean; the shipped-config scorecard already flags it as a
    # regime rather than a triumph, so the same scrutiny applies here.
    ex = res[res["year"] != 2020]
    log(f"leaving 2020 out: mean gain on model {ex['gain_vs_model'].mean():+.5f}, "
        f"mean gap to market {ex['gap_vs_market'].mean():+.5f} (n={len(ex)} folds)")

    log("\n=== pre-registered rules ===")
    won = int((res["gain_vs_model"] >= 0.015).sum())
    never_worse = bool((res["gap_vs_market"] <= 0.003).all())
    log(f"  critique W1: gains >=0.015 in {won}/{len(res)} years (needs 5), "
        f"never worse than market by >0.003: {never_worse}")
    log(f"  critique W1 no-go line: mean gain {mean_gain:.5f} "
        f"{'UNDER' if mean_gain < 0.005 else 'over'} 0.005")
    log(f"  charter rule: beats shipped in {int((res['gain_vs_model'] > 0).sum())}/{len(res)} "
        f"(needs 5) and mean gain {mean_gain:.5f} (needs 0.002)")
    verdict = "SHIP" if (won >= 5 and never_worse) else "DO NOT SHIP as a blend"
    log(f"  VERDICT: {verdict}")

    # 3. the serving question the critique raised but did not size
    log("\n=== could the live simulator use this? ===")
    n_players = 128
    pairs = n_players * (n_players - 1) // 2
    log(f"  a 128 draw needs p(A beats B) for {pairs:,} unordered pairs across the bracket")
    log(f"  matches that actually get played: 127, of which round 1 is 64")
    log(f"  so a price can exist for at most {64/pairs:.2%} of the pairs the simulator asks for,")
    log(f"  and only for round 1, before any later-round pairing is known.")
    log("  A market-conditioned pre-match model therefore cannot carry the draw simulation;")
    log("  it can only improve the single-match product.")

    if OUT.exists():
        log(f"\n{OUT.name} already exists; this measurement informs a product call "
            f"and is scored once. Delete it deliberately to re-measure.")
        raise SystemExit(2)
    res.to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")
    log("MARKET_BLEND_COMPLETE")


if __name__ == "__main__":
    main()
