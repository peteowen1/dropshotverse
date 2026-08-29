import pandas as pd
d = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\v25_postfix_wide_preds.parquet")
print(d.columns.tolist())
print(d.shape)
print(d["date"].dt.year.value_counts().sort_index())
