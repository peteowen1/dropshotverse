"""The standing scorecard: expanding walk-forward by year, 2019-2025.

Implements `docs/plans/validation-charter.md`, which was locked before this ran.
Answers W2 of docs/reviews/FABLE-critique-2026-09-07.md.

WHY: both previous test windows are spent. The 2022-23 tune window was scored
with random-fold xgb.cv, which lets a fold fit on matches played after the ones
it predicts (round 14 passed 5/5 folds there and came back +0.000029 out of
sample). The 2024-25 holdout has been revealed a dozen times. The 2026 forward
test was the one clean shot and has been fired. So there was no way to accept or
reject W1/W3/W4 honestly, and F4 and F5 had to be argued on mechanism.

WHAT IT DOES: for each Y in 2019..2025, fit the shipped configuration on every
tour-level match before 1 January Y and score the matches of year Y that join to
a bookmaker price. Ratings and every other feature are already as-of, so the
only thing the fold boundary has to enforce is that the BOOSTER never sees a
later row - which random folds did not.

Baselines on identical rows: the de-vigged bookmaker average, Elo alone, and a
constant 0.5. Reported per year and never aggregated away.

Usage
  python walk_forward_yearly.py                     # score the shipped config
  python walk_forward_yearly.py --drop h2h_diff     # an ablation candidate
  python walk_forward_yearly.py --label "no-hier" --drop hier_logit_pwinner

Each candidate is scored ONCE, all seven years at once, and the result is
written whether or not it is liked (charter, "reruns and versioning").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import _common                                                          # noqa: E402
from market_compare import devig_power, key_from_full, key_from_short   # noqa: E402
from train_best_model import (XGB_BASE_PARAMS, XGB_CATEGORICAL,         # noqa: E402
                              XGB_NUMERIC, NUM_BOOST_ROUND)

DD = _common.DD            # honours DROPSHOT_DATA_DIR, so a worktree copy of
SCRATCH = _common.SCRATCH  # this script can read the real checkout's data
TEST_YEARS = list(range(2019, 2026))
GAP_LO, GAP_HI = -3, 21     # odds date minus tournament start date, days
OUT = SCRATCH / "walk_forward_yearly.parquet"

# Charter, locked 2026-09-07. Changing these makes a new charter version.
RULE_MIN_YEARS_WON = 5
RULE_MIN_MEAN_GAIN = 0.002
RULE_MAX_YEAR_LOSS = 0.01
RULE_MAX_ECE_WORSENING = 0.005


def log(m: str = "") -> None:
    print(m, flush=True)


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins (charter)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(total)


# ------------------------------------------------------------------ data
def load_features() -> pd.DataFrame:
    """The shipped feature tables, concatenated: 1995-2023 train + 2024-25 holdout."""
    # dict.fromkeys: `tour` is both a model categorical and a join key, and
    # selecting it twice makes df["tour"] a DataFrame rather than a Series.
    keep = list(dict.fromkeys(XGB_NUMERIC + XGB_CATEGORICAL
                              + ["date", "tour", "winner_uid", "loser_uid", "y"]))
    frames = []
    for name in ("xgb_features_v10_train.parquet", "xgb_features_v10_holdout.parquet"):
        d = pd.read_parquet(SCRATCH / name)
        missing = [c for c in keep if c not in d.columns]
        if missing:
            raise SystemExit(f"{name} is missing {missing}")
        frames.append(d[keep])
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    if df.duplicated(subset=["date", "tour", "winner_uid", "loser_uid"]).any():
        n = int(df.duplicated(subset=["date", "tour", "winner_uid", "loser_uid"]).sum())
        log(f"  WARNING: {n} duplicate (date, tour, winner, loser) rows across the two tables")
    return df


def join_market(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the de-vigged bookmaker average. Coverage is reported per year,
    never assumed (CLAUDE.md: assert coverage, not presence)."""
    from market_compare import load_odds
    names = pd.read_parquet(SCRATCH / "elo_raw_ratings_full.parquet",
                            columns=["winner_uid", "winner_name", "loser_uid", "loser_name"])
    uid2name = dict(zip(names["winner_uid"], names["winner_name"]))
    uid2name.update(dict(zip(names["loser_uid"], names["loser_name"])))
    del names

    d = df.copy()
    kw = d["winner_uid"].map(lambda u: key_from_full(uid2name.get(u, "")))
    kl = d["loser_uid"].map(lambda u: key_from_full(uid2name.get(u, "")))
    d["pair"] = ["|".join(sorted([a, b])) for a, b in zip(kw, kl)]
    d["tour_s"] = d["tour"].astype(str)

    o = load_odds(years=TEST_YEARS)
    o.columns = [str(c).strip() for c in o.columns]
    for c in ("PSW", "AvgW", "B365W"):
        if c in o.columns:
            log(f"  odds column {c}: {o[c].notna().sum():,}/{len(o):,} populated "
                f"({o[c].notna().mean():.1%})")
    if not {"AvgW", "AvgL"} <= set(o.columns):
        raise SystemExit("odds files have no AvgW/AvgL; the charter's benchmark is the bookmaker average")
    o = o.dropna(subset=["Winner", "Loser", "Date", "AvgW", "AvgL"])
    o["date"] = pd.to_datetime(o["Date"], errors="coerce")
    o = o.dropna(subset=["date"])
    o["pair"] = ["|".join(sorted([key_from_short(a), key_from_short(b)]))
                 for a, b in zip(o["Winner"], o["Loser"])]
    o["p_mkt_winner"] = devig_power(o["AvgW"].values.astype(float), o["AvgL"].values.astype(float))
    o = o[o["p_mkt_winner"].notna()]
    log(f"  odds rows usable: {len(o):,}")

    o["k_w_o"] = o["Winner"].map(key_from_short)
    m = d.merge(o[["pair", "tour", "date", "p_mkt_winner", "k_w_o"]],
                left_on=["pair", "tour_s"], right_on=["pair", "tour"],
                how="inner", suffixes=("", "_o"))
    m["gap"] = (m["date_o"] - m["date"]).dt.days
    m = m[(m["gap"] >= GAP_LO) & (m["gap"] <= GAP_HI)]
    m = m.sort_values("gap").drop_duplicates(
        subset=["date", "tour_s", "winner_uid", "loser_uid"], keep="first")
    # Does the odds row name the same WINNER as the archive row? The join is on
    # an unordered surname+initial pair within a 25-day window, so it can attach
    # the wrong meeting when two players play twice inside one event (a round
    # robin then a final) or when two players share a surname key. Disagreement
    # is a bad join, not a bad model, and silently keeping it corrupts the very
    # baseline the charter measures against. market_compare.py's own main() has
    # made this check since ticket 04; the first version of this script omitted
    # it, which a review caught.
    m["k_w"] = m["winner_uid"].map(lambda u: key_from_full(uid2name.get(u, "")))
    agrees = m["k_w_o"] == m["k_w"]
    log(f"  winner agrees: {int(agrees.sum()):,}/{len(m):,} ({agrees.mean():.1%}); "
        f"dropping {int((~agrees).sum()):,} mis-joined row(s)")
    m = m[agrees]
    # The odds file's "Winner" is the actual winner; express its price for SIDE A.
    # This is a relabelling, not information: the market's price for a given
    # player is fixed, and y only says which player we called side A.
    m["p_mkt"] = np.where(m["y"] == 1, m["p_mkt_winner"], 1 - m["p_mkt_winner"])
    return m[["date", "tour_s", "winner_uid", "loser_uid", "p_mkt"]].rename(columns={"tour_s": "tour"})


def design(d: pd.DataFrame, cols: list[str], cat_levels: dict) -> pd.DataFrame:
    X = d[cols].copy()
    for c in XGB_CATEGORICAL:
        if c in X.columns:
            X[c] = pd.Categorical(d[c], categories=cat_levels[c])
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="", help="comma-separated features to drop (an ablation candidate)")
    ap.add_argument("--label", default=None, help="name for this candidate in the output")
    ap.add_argument("--rounds", type=int, default=NUM_BOOST_ROUND)
    ap.add_argument("--rescore", action="store_true",
                    help="overwrite an existing result for this label. The charter says a candidate "
                         "is scored ONCE; re-scoring after seeing the number makes it exploratory, "
                         "so this flag exists to make that deliberate and visible, not easy.")
    args = ap.parse_args()
    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    label = args.label or ("shipped" if not drop else "drop:" + ",".join(drop))

    log(f"=== walk-forward scorecard: {label} ===")
    log(f"charter: docs/plans/validation-charter.md | rounds {args.rounds} | "
        f"rule: beat shipped in >={RULE_MIN_YEARS_WON}/7 years AND mean gain >={RULE_MIN_MEAN_GAIN}")

    df = load_features()
    log(f"features: {len(df):,} rows {df['date'].min().date()}..{df['date'].max().date()}")

    log("\njoining the bookmaker average ...")
    mkt = join_market(df)
    df = df.merge(mkt, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    cov = df[df["year"].isin(TEST_YEARS)].groupby("year")["p_mkt"].agg(["size", "count"])
    cov["coverage"] = cov["count"] / cov["size"]
    log("  market coverage by test year:")
    for y, r in cov.iterrows():
        log(f"    {y}: {int(r['count']):,}/{int(r['size']):,} ({r['coverage']:.1%})")
    if (cov["coverage"] < 0.5).any():
        log("  WARNING: a test year is under 50% covered; its number rests on a subset")

    unknown = [c for c in drop if c not in XGB_NUMERIC + XGB_CATEGORICAL]
    if unknown:
        raise SystemExit(f"--drop names features that are not in the shipped set: {unknown}")
    cols = [c for c in XGB_NUMERIC + XGB_CATEGORICAL if c not in drop]
    cat_levels = {c: sorted(df[c].dropna().unique()) for c in XGB_CATEGORICAL}

    rows = []
    for Y in TEST_YEARS:
        tr = df[df["year"] < Y]
        te = df[(df["year"] == Y) & df["p_mkt"].notna()]
        if not len(te):
            log(f"\n{Y}: no market-joined rows, skipped")
            continue
        # charter leakage checks, asserted per fold
        assert tr["date"].max() < te["date"].min(), f"{Y}: train overlaps test"
        assert not (set(map(tuple, tr[["date", "tour", "winner_uid", "loser_uid"]].values))
                    & set(map(tuple, te[["date", "tour", "winner_uid", "loser_uid"]].values))), \
            f"{Y}: a match appears in both train and test"

        dtr = xgb.DMatrix(design(tr, cols, cat_levels), label=tr["y"].values.astype(int),
                          enable_categorical=True)
        bst = xgb.train({**XGB_BASE_PARAMS, "seed": 0}, dtr, num_boost_round=args.rounds)
        p = bst.predict(xgb.DMatrix(design(te, cols, cat_levels), enable_categorical=True))
        y = te["y"].values.astype(int)
        p_elo = 1.0 / (1.0 + np.exp(-te["elo_logit_pwinner"].values))
        rec = dict(
            label=label, year=Y, n_train=len(tr), n_test=len(te),
            train_to=str(tr["date"].max().date()), test_from=str(te["date"].min().date()),
            ll_model=logloss(p, y), ll_market=logloss(te["p_mkt"].values, y),
            ll_elo=logloss(p_elo, y), ll_const=logloss(np.full(len(y), 0.5), y),
            ece_model=ece(p, y), ece_market=ece(te["p_mkt"].values, y),
            brier_model=brier(p, y), acc_model=float(((p > 0.5).astype(int) == y).mean()),
        )
        rec["gap_to_market"] = rec["ll_model"] - rec["ll_market"]
        rows.append(rec)
        log(f"\n{Y}: train {len(tr):,} rows to {rec['train_to']} | test {len(te):,} from {rec['test_from']}")
        log(f"  log-loss  model {rec['ll_model']:.5f}  market {rec['ll_market']:.5f}  "
            f"elo {rec['ll_elo']:.5f}  const {rec['ll_const']:.5f}")
        log(f"  gap to market {rec['gap_to_market']:+.5f}   ECE model {rec['ece_model']:.4f} "
            f"market {rec['ece_market']:.4f}   acc {rec['acc_model']:.3f}")

    res = pd.DataFrame(rows)
    log("\n=== scorecard ===")
    log(res[["year", "n_test", "ll_model", "ll_market", "ll_elo", "gap_to_market",
             "ece_model", "acc_model"]].to_string(index=False))
    log(f"\nunweighted mean over {len(res)} years: model {res['ll_model'].mean():.5f}, "
        f"market {res['ll_market'].mean():.5f}, gap {res['gap_to_market'].mean():+.5f}")
    log(f"years the model beats the market: {int((res['gap_to_market'] < 0).sum())}/{len(res)}")

    if OUT.exists():
        prev = pd.read_parquet(OUT)
        if label in set(prev["label"]) and not args.rescore:
            log(f"\n{OUT.name} already holds a result for '{label}'. The charter scores a candidate "
                f"ONCE; re-scoring after seeing the number makes it exploratory. Nothing was written. "
                f"Pass --rescore to overwrite deliberately, or use a new --label.")
            raise SystemExit(2)
        if label in set(prev["label"]):
            log(f"  --rescore: replacing the previous '{label}' result (now exploratory)")
        prev = prev[prev["label"] != label]
        res_out = pd.concat([prev, res], ignore_index=True)
    else:
        res_out = res
    res_out.to_parquet(OUT, index=False)
    log(f"\nwrote {OUT} ({len(res_out):,} rows, {res_out['label'].nunique()} candidate(s))")

    if label != "shipped" and (OUT.exists() and "shipped" in set(res_out["label"])):
        base = res_out[res_out["label"] == "shipped"].set_index("year")
        cand = res.set_index("year")
        common = base.index.intersection(cand.index)
        delta = cand.loc[common, "ll_model"] - base.loc[common, "ll_model"]
        won = int((delta < 0).sum())
        mean_gain = float(-delta.mean())
        d_ece = float(cand.loc[common, "ece_model"].mean() - base.loc[common, "ece_model"].mean())
        worst = float(delta.max())
        log("\n=== charter rule ===")
        log(f"  beats shipped in {won}/{len(common)} years (need >={RULE_MIN_YEARS_WON})")
        log(f"  mean gain {mean_gain:+.5f} (need >={RULE_MIN_MEAN_GAIN})")
        log(f"  worst single year {worst:+.5f} (fails outright above {RULE_MAX_YEAR_LOSS})")
        log(f"  ECE change {d_ece:+.5f} (may not worsen by more than {RULE_MAX_ECE_WORSENING})")
        ship = (won >= RULE_MIN_YEARS_WON and mean_gain >= RULE_MIN_MEAN_GAIN
                and worst <= RULE_MAX_YEAR_LOSS and d_ece <= RULE_MAX_ECE_WORSENING)
        log(f"  VERDICT: {'SHIP' if ship else 'DO NOT SHIP'}")
    log("WALK_FORWARD_COMPLETE")


if __name__ == "__main__":
    main()
