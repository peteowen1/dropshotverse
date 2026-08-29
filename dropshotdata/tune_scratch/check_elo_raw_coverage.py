import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata"
elo_raw = pd.read_parquet(rf"{DD}\tune_scratch\elo_raw_ratings_full.parquet")
elo_raw = elo_raw[elo_raw["level_file"] == "tour"]
print("elo_raw_ratings_full.parquet, tour-level, date range:", elo_raw["date"].min(), "to", elo_raw["date"].max())
print("rows by year:")
print(elo_raw["date"].dt.year.value_counts().sort_index().to_string())

cache_path = rf"{DD}\tune_scratch\prepared_matches_cache.pkl"
import pickle
with open(cache_path, "rb") as f:
    arc, _ = pickle.load(f)
arc = arc[arc["level_file"] == "tour"]
print("\nprepared_matches_cache.pkl, tour-level, date range:", arc["date"].min(), "to", arc["date"].max())
