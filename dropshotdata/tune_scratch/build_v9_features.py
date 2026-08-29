import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
ps = pd.read_parquet(rf"{DD}\tune_scratch\playstyle_full.parquet")


def add_playstyle(feat_path, out_path):
    feat = pd.read_parquet(feat_path)
    j = feat.merge(ps, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
    j = j.drop_duplicates(subset=["date", "tour", "winner_uid", "loser_uid"], keep="first")
    n0 = len(j)
    miss = j[["serve_form_w", "serve_form_l", "return_form_w", "return_form_l"]].isna().any(axis=1)
    print(f"  {feat_path}: {n0} -> {int((~miss).sum())} after dropping {int(miss.sum())} missing "
          f"({miss.mean():.4%})")
    j = j[~miss].reset_index(drop=True)
    y = j["y"].values.astype(int)
    sw, sl = j["serve_form_w"].values.astype(float), j["serve_form_l"].values.astype(float)
    rw, rl = j["return_form_w"].values.astype(float), j["return_form_l"].values.astype(float)
    j["serve_form_diff"] = np.where(y == 1, sw - sl, sl - sw)
    j["return_form_diff"] = np.where(y == 1, rw - rl, rl - rw)
    j = j.drop(columns=["serve_form_w", "serve_form_l", "return_form_w", "return_form_l"])
    j.to_parquet(out_path)
    print(f"  wrote {out_path} ({len(j):,} rows)")


print("training window:")
add_playstyle(rf"{DD}\tune_scratch\xgb_features_v7_train.parquet", rf"{DD}\tune_scratch\xgb_features_v9_train.parquet")
print("\nholdout window:")
add_playstyle(rf"{DD}\tune_scratch\xgb_features_v7_holdout.parquet", rf"{DD}\tune_scratch\xgb_features_v9_holdout.parquet")
