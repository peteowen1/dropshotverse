"""The 2026 forward test: the one population no decision in this project has touched.

WHY THIS EXISTS
The 2024-25 holdout has been revealed at least a dozen times across tickets
27/28 and is retired from screening (docs/DECISIONS.md D-10). Round 14's K
result (-0.001393 downstream, 5/5 folds) was measured on the 2022-23 tune
window, and the K point was SELECTED on that same window - so its magnitude is
optimistically biased and only its direction is supported. 2026 is different:
Pinnacle odds exist for Jan-Aug (1,976 ATP + 1,921 WTA rows), the archive plus
the tennismylife bridge carries results to 2026-08-24, and no tuning, screening
or feature decision in this project has ever looked at any of it.

TWO QUESTIONS, IN PRIORITY ORDER
  1. Does the K change hold OUT OF SAMPLE? Arm vs arm, identical everything
     else. This is the question the test is designed for and can answer cleanly.
  2. How does the model compare to the market on untouched data, and does the
     disagreement-accuracy finding still hold? Ticket 28 round 7 found the model
     right only 44.6% of the time when it picked a different favourite than the
     market (CI [41.5, 47.7]), measured at v2. Never re-measured since.

SCOPE, STATED NOT BURIED
This uses an ELO-FEATURE model: the v1 feature set minus the three hierarchical
columns, because regenerating hier state for 2026 is a separate and much larger
job. So question 1 is answered properly - both arms differ only in K - but
question 2 is answered by a WEAKER model than shipped config D, which also
carries hier state, set/game-level Elo, age and fatigue. Any market gap reported
here is an UPPER BOUND on config D's gap, not an estimate of it. A model that
beat the market here would be a strong claim; one that trails is not evidence
that config D trails by the same amount.

TRAIN/TEST
  train  everything up to 2025-12-31
  test   2026 tour-level completed matches joined to Pinnacle closing odds
No 2026 row influences training, feature construction, or the K choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))
sys.path.insert(0, str(DD / "tune_scratch"))

from elo_compact import (KIND_OK, LEVELS, ROUNDS, SURFACES,  # noqa: E402
                         build_compact, run_elo_fast)
from market_compare import devig_power, key_from_full, key_from_short  # noqa: E402

OUT = DD / "tune_scratch" / "forward_test_2026.parquet"
ODDS = DD / "raw" / "odds"

ARMS = {
    "baseline (shipped K)": dict(k_base=250.0, k_shape=0.4, k_min=16.0),
    "round14 K ridge":      dict(k_base=350.0, k_shape=0.5, k_min=16.0),
}
PARAMS = dict(objective="binary:logistic", eval_metric="logloss",
              max_depth=3, eta=0.02, subsample=0.7, colsample_bytree=0.9,
              min_child_weight=5.0, gamma=1.0, reg_alpha=0.1, reg_lambda=1.0,
              tree_method="hist", nthread=4)
N_ROUND = 217          # pinned, as in the paired-fold test - not a free axis
CATS = ["surface", "round", "level", "tour"]


def log(m: str = "") -> None:
    print(m, flush=True)


def context_walk(d: dict):
    """h2h and days-since-last, computed once. Both are arm-invariant here: the
    K arms change rating VALUES, never which rows count or their order."""
    n = len(d["day"])
    h2h_w = np.zeros(n, dtype=np.int16)
    h2h_l = np.zeros(n, dtype=np.int16)
    gap_w = np.full(n, -1, dtype=np.int32)
    gap_l = np.full(n, -1, dtype=np.int32)
    wins: dict[tuple[int, int], int] = {}
    last: dict[int, int] = {}
    W, L, DAY, K = d["w"], d["l"], d["day"], d["kind"]
    for i in range(n):
        w, l, day = int(W[i]), int(L[i]), int(DAY[i])
        h2h_w[i] = wins.get((w, l), 0)
        h2h_l[i] = wins.get((l, w), 0)
        if w in last:
            gap_w[i] = day - last[w]
        if l in last:
            gap_l[i] = day - last[l]
        if K[i] == KIND_OK:
            wins[(w, l)] = wins.get((w, l), 0) + 1
        last[w] = last[l] = day
    return h2h_w, h2h_l, gap_w, gap_l


def build_frame(d: dict, arm: dict, ctx) -> pd.DataFrame:
    """One row per completed match, side-A/side-B differentials.

    Side assignment here is by PARITY OF ROW INDEX rather than the hash used in
    build_xgb_features.py. Either is outcome-blind, which is the only property
    that matters (it must not depend on who won); parity is used because the
    compact arrays carry no uid strings to hash. Both arms get the identical
    assignment, so it cannot favour one."""
    st = run_elo_fast(d, blend=0.0, decay=0.3, return_state=True, **arm)
    ok = d["kind"] == KIND_OK
    h2h_w, h2h_l, gap_w, gap_l = ctx
    sa = (np.arange(len(d["day"])) % 2 == 0)[ok]        # side A is the winner
    p_w = np.clip(st["p_winner"][ok], 1e-6, 1 - 1e-6)
    p_a = np.where(sa, p_w, 1 - p_w)

    def diff(a, b):
        a, b = a[ok], b[ok]
        return np.where(sa, a - b, b - a).astype(float)

    return pd.DataFrame({
        "day": d["day"][ok],
        "y": sa.astype(int),
        "elo_overall_diff": diff(st["ro_w"], st["ro_l"]),
        "elo_surface_diff": diff(st["rs_w"], st["rs_l"]),
        "elo_logit_pwinner": np.log(p_a / (1 - p_a)),
        "n_prior_diff": diff(st["n_all_w"], st["n_all_l"]),
        "h2h_diff": diff(h2h_w, h2h_l),
        "weeks_since_last_diff": diff(gap_w, gap_l) / 7.0,
        "surface": pd.Categorical([SURFACES[i] for i in d["surf"][ok]],
                                  categories=list(SURFACES)),
        "round": pd.Categorical([ROUNDS[i] for i in d["round"][ok]],
                                categories=list(ROUNDS)),
        "level": pd.Categorical([LEVELS[i] for i in d["level"][ok]],
                                categories=list(LEVELS)),
        "tour": pd.Categorical(np.where(d["tour"][ok] == 0, "atp", "wta"),
                               categories=["atp", "wta"]),
        "best_of": d["best_of"][ok].astype(float),
        "lvl": d["lvl"][ok],
        "p_elo_sideA": p_a,
        "_sa": sa,
    })


def load_2026_odds() -> pd.DataFrame:
    frames = []
    for tour in ("atp", "wta"):
        f = ODDS / tour / "2026.xlsx"
        df = pd.read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        df["tour"] = tour
        frames.append(df)
    o = pd.concat(frames, ignore_index=True)

    # COVERAGE, not presence (CLAUDE.md). Pinnacle is PRESENT as a column in the
    # 2026 files and 96% EMPTY: PSW stops on 2026-01-13 (ATP 71/1,976 = 3.6%,
    # WTA 101/1,921 = 5.3%). Every earlier statement in this project that 2026
    # holds ~3,900 matches "with Pinnacle odds" was reading the header, not the
    # data. AvgW/AvgL are 100% populated and run to 2026-08-24, so the market
    # benchmark here is the BOOKMAKER AVERAGE.
    #
    # That is a weaker benchmark than Pinnacle closing: the average carries more
    # vig and is less sharp, so a gap measured against it UNDERSTATES the gap to
    # Pinnacle. Beating the average is not beating the closing line.
    for c in ("PSW", "AvgW", "B365W"):
        if c in o.columns:
            log(f"  odds coverage {c}: {o[c].notna().sum():,}/{len(o):,} "
                f"({o[c].notna().mean():.1%})")
    o = o.dropna(subset=["Winner", "Loser", "Date", "AvgW", "AvgL"])
    o["date"] = pd.to_datetime(o["Date"], errors="coerce")
    o = o.dropna(subset=["date"])
    o["k_w"] = o["Winner"].map(key_from_short)
    o["k_l"] = o["Loser"].map(key_from_short)
    o["pair"] = [("|".join(sorted([a, b]))) for a, b in zip(o["k_w"], o["k_l"])]
    o["p_mkt_winner"] = devig_power(o["AvgW"].values.astype(float),
                                    o["AvgL"].values.astype(float))
    return o


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    d = build_compact()
    log("computing h2h / layoff walk (arm-invariant, once)...")
    ctx = context_walk(d)

    # uid -> name, needed only for the odds join
    names = pd.read_parquet(DD / "tune_scratch" / "elo_raw_ratings_full.parquet",
                            columns=["winner_uid", "winner_name", "loser_uid",
                                     "loser_name"])
    uid2name = dict(zip(names["winner_uid"], names["winner_name"]))
    uid2name.update(dict(zip(names["loser_uid"], names["loser_name"])))
    del names
    log(f"uid->name map: {len(uid2name):,}")

    uids = d["uids"]
    ok = d["kind"] == KIND_OK
    dates = d["day"][ok].astype("datetime64[D]")
    yr = pd.to_datetime(dates).year

    results = []
    frames = {}
    for name, arm in ARMS.items():
        f = build_frame(d, arm, ctx)
        frames[name] = f
        tr = (yr <= 2025) & (f["lvl"].values >= 0)
        te = (yr == 2026) & (f["lvl"].values == 0)
        log(f"\n{name}: train {int(tr.sum()):,}  test(2026 tour) {int(te.sum()):,}")

        drop = ["day", "y", "lvl", "p_elo_sideA", "_sa"]
        X = f.drop(columns=drop)
        dtr = xgb.DMatrix(X[tr], label=f["y"].values[tr], enable_categorical=True)
        dte = xgb.DMatrix(X[te], label=f["y"].values[te], enable_categorical=True)
        bst = xgb.train(PARAMS, dtr, num_boost_round=N_ROUND, verbose_eval=False)
        p = bst.predict(dte)
        ll = logloss(p, f["y"].values[te])
        acc = float(((p > 0.5).astype(int) == f["y"].values[te]).mean())
        log(f"  2026 log-loss {ll:.6f}   accuracy {acc:.4f}")
        results.append(dict(arm=name, n_test=int(te.sum()), logloss=ll, acc=acc))
        frames[name] = (f, te, p)

    tab = pd.DataFrame(results)
    base = float(tab.loc[tab.arm.str.startswith("baseline"), "logloss"].iloc[0])
    tab["delta"] = tab["logloss"] - base
    log("\n=== QUESTION 1: does the K change hold out of sample? ===")
    log(tab[["arm", "n_test", "logloss", "acc", "delta"]]
        .to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    log(f"\n  tune-window paired result was -0.001393 (5/5 folds).")
    log(f"  2026 out-of-sample delta: {float(tab['delta'].iloc[-1]):+.6f}")

    # ---------------- market comparison on the same rows
    log("\n=== QUESTION 2: versus the market, on rows never used for anything ===")
    f, te, p = frames["round14 K ridge"]
    sub = f[te].copy()
    sub["p_model_sideA"] = p
    sub["date"] = pd.to_datetime(dates[te])
    w_uid = uids[d["w"][ok]][te]
    l_uid = uids[d["l"][ok]][te]
    sub["k_w"] = [key_from_full(uid2name.get(u, "")) for u in w_uid]
    sub["k_l"] = [key_from_full(uid2name.get(u, "")) for u in l_uid]
    sub["pair"] = [("|".join(sorted([a, b]))) for a, b in zip(sub["k_w"], sub["k_l"])]
    sub["tour_s"] = sub["tour"].astype(str)

    odds = load_2026_odds()
    log(f"  2026 odds rows with Pinnacle prices: {len(odds):,}")
    m = sub.merge(odds[["pair", "tour", "date", "k_w", "p_mkt_winner"]],
                  left_on=["pair", "tour_s"], right_on=["pair", "tour"],
                  how="inner", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= -3) & (m["gap"] <= 21)]
    m = m.sort_values("gap").drop_duplicates(subset=["pair", "tour_s", "date"],
                                             keep="first")
    log(f"  joined to model rows: {len(m):,} of {int(te.sum()):,} "
        f"({len(m) / max(int(te.sum()), 1):.1%})")

    # The odds file's "winner" is the actual winner, same as ours; convert the
    # market's winner-probability to side A.
    m["p_mkt_sideA"] = np.where(m["_sa"], m["p_mkt_winner"], 1 - m["p_mkt_winner"])
    m = m[m["p_mkt_sideA"].notna()]
    y = m["y"].values
    ll_model = logloss(m["p_model_sideA"].values, y)
    ll_mkt = logloss(m["p_mkt_sideA"].values, y)
    log(f"\n  n = {len(m):,}")
    log(f"  model  log-loss {ll_model:.5f}")
    log(f"  market log-loss {ll_mkt:.5f}")
    log(f"  gap             {ll_model - ll_mkt:+.5f}   "
        f"(negative = model beats market)")

    # disagreement accuracy - ticket 28 round 7, re-measured
    fav_model = m["p_model_sideA"].values > 0.5
    fav_mkt = m["p_mkt_sideA"].values > 0.5
    dis = fav_model != fav_mkt
    if dis.sum():
        right = float((fav_model[dis].astype(int) == y[dis]).mean())
        n_d = int(dis.sum())
        se = np.sqrt(right * (1 - right) / n_d)
        log(f"\n  disagreements with the market: {n_d:,} of {len(m):,} "
            f"({dis.mean():.1%})")
        log(f"  model right when it disagrees: {right:.1%}  "
            f"95% CI [{right - 1.96 * se:.1%}, {right + 1.96 * se:.1%}]")
        log(f"  ticket 28 round 7 (v2, 2024-25): 44.6% CI [41.5%, 47.7%]")

    pd.DataFrame(results).to_parquet(OUT, index=False)
    log("\nFORWARD_COMPLETE")


if __name__ == "__main__":
    main()
