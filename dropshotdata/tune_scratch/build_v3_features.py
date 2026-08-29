"""Ticket 28 round 9: add serve_form_diff + return_form_diff (screened
together, tune-only, Delta -0.00033 CV) to v2's feature set, for both
windows. Read-only wrt build_v2_features.py."""
import pickle

import numpy as np
import pandas as pd


def build_serve_return_form():
    with open(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\prepared_matches_cache.pkl", "rb") as f:
        arc, _ = pickle.load(f)
    arc = arc[arc["level_file"] == "tour"].reset_index(drop=True).copy()
    arc["orig_idx"] = arc.index
    arc["serve_pct_w"] = np.where(arc["has_points"] & (arc["w_svpt"] > 0),
                                   (arc["w_1stWon"] + arc["w_2ndWon"]) / arc["w_svpt"], np.nan)
    arc["serve_pct_l"] = np.where(arc["has_points"] & (arc["l_svpt"] > 0),
                                   (arc["l_1stWon"] + arc["l_2ndWon"]) / arc["l_svpt"], np.nan)
    arc["return_pct_w"] = 1.0 - arc["serve_pct_l"]
    arc["return_pct_l"] = 1.0 - arc["serve_pct_w"]

    long_w = arc[["orig_idx", "winner_uid", "serve_pct_w", "return_pct_w"]].rename(
        columns={"winner_uid": "uid", "serve_pct_w": "serve_pct", "return_pct_w": "return_pct"})
    long_l = arc[["orig_idx", "loser_uid", "serve_pct_l", "return_pct_l"]].rename(
        columns={"loser_uid": "uid", "serve_pct_l": "serve_pct", "return_pct_l": "return_pct"})
    long_all = pd.concat([long_w, long_l], ignore_index=True).sort_values(["uid", "orig_idx"], kind="stable")
    for col in ("serve_pct", "return_pct"):
        long_all[f"{col}_form"] = long_all.groupby("uid")[col].transform(
            lambda s: s.shift(1).ewm(span=15, min_periods=3).mean())

    fw = long_all.rename(columns={"uid": "winner_uid", "serve_pct_form": "serve_form_w",
                                   "return_pct_form": "return_form_w"})[
        ["orig_idx", "winner_uid", "serve_form_w", "return_form_w"]]
    fl = long_all.rename(columns={"uid": "loser_uid", "serve_pct_form": "serve_form_l",
                                   "return_pct_form": "return_form_l"})[
        ["orig_idx", "loser_uid", "serve_form_l", "return_form_l"]]
    arc = arc.merge(fw, on=["orig_idx", "winner_uid"], how="left").merge(fl, on=["orig_idx", "loser_uid"], how="left")
    return arc[["date", "tour", "winner_uid", "loser_uid",
                "serve_form_w", "serve_form_l", "return_form_w", "return_form_l"]]


def build_window(feat_path, sr):
    feat = pd.read_parquet(feat_path)
    n0 = len(feat)
    j = feat.merge(sr, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")

    y = j["y"].values.astype(int)
    sw, sl = j["serve_form_w"].values.astype(float), j["serve_form_l"].values.astype(float)
    rw, rl = j["return_form_w"].values.astype(float), j["return_form_l"].values.astype(float)
    j["serve_form_diff"] = np.where(y == 1, sw - sl, sl - sw)
    j["return_form_diff"] = np.where(y == 1, rw - rl, rl - rw)

    miss = j["serve_form_diff"].isna() | j["return_form_diff"].isna()
    print(f"  {feat_path}: {n0} -> {len(j)} rows after merge, {int(miss.sum())} missing "
          f"serve/return form ({miss.mean():.4%}), dropped")
    j = j[~miss].drop(columns=["serve_form_w", "serve_form_l", "return_form_w", "return_form_l"])
    return j.reset_index(drop=True)


sr = build_serve_return_form()
tune = build_window(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_tune.parquet", sr)
holdout = build_window(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_holdout.parquet", sr)

tune.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v3_tune.parquet")
holdout.to_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v3_holdout.parquet")
print(f"tune: {len(tune):,} rows, holdout: {len(holdout):,} rows")
