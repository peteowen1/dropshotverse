import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"


def add_targeted_mask(in_path, out_path):
    df = pd.read_parquet(in_path)
    atp_not_grass = (df["tour"] == "atp") & (df["surface"] != "Grass")
    df["serve_form_diff_targeted"] = np.where(atp_not_grass, 0.0, df["serve_form_diff"].values)
    df["return_form_diff_targeted"] = np.where(atp_not_grass, 0.0, df["return_form_diff"].values)
    df.to_parquet(out_path)
    print(f"wrote {out_path} ({len(df):,} rows)")


add_targeted_mask(rf"{DD}\tune_scratch\xgb_features_v9_train.parquet", rf"{DD}\tune_scratch\xgb_features_v10_train.parquet")
add_targeted_mask(rf"{DD}\tune_scratch\xgb_features_v9_holdout.parquet", rf"{DD}\tune_scratch\xgb_features_v10_holdout.parquet")
