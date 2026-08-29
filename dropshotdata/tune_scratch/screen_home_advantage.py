"""Ticket 28 round 10 (Pete's own suggestion): home-court advantage. No
tournament venue country exists in this archive's schema - only free-text
`tourney_name`. Built a hand-mapped city/event -> IOC-country lookup covering
the tour's recurring annual stops (not exhaustive - unmapped/rare event names
default to "no home player", a safe non-leaky default since that's true for
the large majority of matches anyway, not an excluded-row situation).
Screened tune-window (2022-23) only via 5-fold xgb.cv, holdout untouched.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

# Country code convention matches winner_ioc/loser_ioc (IOC 3-letter codes).
TOURNEY_COUNTRY = {
    "Australian Open": "AUS", "Adelaide 1": "AUS", "Adelaide 2": "AUS", "Adelaide": "AUS",
    "Sydney": "AUS", "Brisbane": "AUS", "Melbourne": "AUS", "Hobart": "AUS",
    "Roland Garros": "FRA", "Lyon": "FRA", "Marseille": "FRA", "Montpellier": "FRA",
    "Strasbourg": "FRA", "Paris Masters": "FRA", "Paris": "FRA", "Metz": "FRA", "Rouen": "FRA",
    "Wimbledon": "GBR", "Eastbourne": "GBR", "Queen's Club": "GBR", "Nottingham": "GBR",
    "Birmingham": "GBR", "Liverpool": "GBR", "Glasgow": "GBR", "London": "GBR",
    "Us Open": "USA", "Indian Wells Masters": "USA", "Indian Wells": "USA",
    "Miami Masters": "USA", "Miami": "USA", "Washington": "USA", "Cincinnati Masters": "USA",
    "Cincinnati": "USA", "Charleston": "USA", "Winston-Salem": "USA", "San Diego": "USA",
    "Cleveland": "USA", "Dallas": "USA", "Houston": "USA", "Delray Beach": "USA",
    "Atlanta": "USA", "Newport": "USA", "New York": "USA", "Indianapolis": "USA",
    "Madrid Masters": "ESP", "Madrid": "ESP", "Barcelona": "ESP", "Mallorca": "ESP",
    "Marbella": "ESP", "Valencia": "ESP", "Santander": "ESP",
    "Rome Masters": "ITA", "Rome": "ITA", "Palermo": "ITA", "Parma": "ITA", "Milan": "ITA",
    "Florence": "ITA", "Turin": "ITA", "Naples": "ITA",
    "Dubai": "UAE", "Abu Dhabi": "UAE",
    "Doha": "QAT",
    "Hamburg": "GER", "Stuttgart": "GER", "Munich": "GER", "Halle": "GER",
    "Berlin": "GER", "Bad Homburg": "GER", "Cologne": "GER",
    "s Hertogenbosch": "NED", "Rotterdam": "NED", "Amsterdam": "NED",
    "Tokyo": "JPN", "Osaka": "JPN",
    "Monte Carlo Masters": "MON",
    "Canada Masters": "CAN", "Toronto": "CAN", "Montreal": "CAN",
    "Shanghai Masters": "CHN", "Beijing": "CHN", "Zhuhai": "CHN", "Shenzhen": "CHN",
    "Guangzhou": "CHN", "Nanchang": "CHN", "Hangzhou": "CHN", "Chengdu": "CHN",
    "Seoul": "KOR",
    "Guadalajara": "MEX", "Guadalajara 2": "MEX", "Acapulco": "MEX", "Monterrey": "MEX", "Los Cabos": "MEX",
    "Basel": "SUI", "Geneva": "SUI", "Gstaad": "SUI", "Lausanne": "SUI",
    "Vienna": "AUT", "Kitzbuhel": "AUT",
    "Bogota": "COL", "Barranquilla": "COL",
    "Rabat": "MAR", "Marrakech": "MAR",
    "Warsaw": "POL",
    "Budapest": "HUN",
    "Prague": "CZE", "Ostrava": "CZE",
    "Cluj Napoca": "ROU", "Bucharest": "ROU",
    "Astana": "KAZ", "Nur-Sultan": "KAZ",
    "Auckland": "NZL",
    "Cordoba": "ARG", "Buenos Aires": "ARG",
    "Pune": "IND", "Chennai": "IND",
    "Santiago": "CHI",
    "Estoril": "POR",
    "United Cup": "AUS",  # hosted in Australia (Sydney/Perth)
    "Atp Cup": "AUS",
    "Antwerp": "BEL", "Brussels": "BEL",
    "Sofia": "BUL",
    "Belgrade": "SRB",
    "Zagreb": "CRO", "Umag": "CRO",
    "Stockholm": "SWE", "Bastad": "SWE",
    "Tel Aviv": "ISR",
    "Metz": "FRA",
    "Antalya": "TUR", "Istanbul": "TUR",
    "Doha": "QAT",
    "Los Angeles": "USA", "Chicago": "USA",
    "Wuhan": "CHN", "Ningbo": "CHN", "Jiujiang": "CHN",
    "Hong Kong": "HKG",
    "Merida": "MEX",
    "Charleston 2": "USA",
    "San Jose": "CRC",
}


def load_raw(tour, years):
    frames = []
    for y in years:
        d = pd.read_csv(rf"C:\dev\dropshotverse\dropshotdata\raw\github\{tour}\{tour}_matches_{y}.csv",
                         low_memory=False, usecols=["tourney_name", "tourney_date", "winner_id", "loser_id",
                                                     "winner_ioc", "loser_ioc"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    return df


raw = pd.concat([load_raw("atp", [2022, 2023]), load_raw("wta", [2022, 2023])], ignore_index=True)
raw["tourney_country"] = raw["tourney_name"].map(TOURNEY_COUNTRY)
mapped = raw["tourney_country"].notna()
print(f"tournament-name mapping coverage: {mapped.mean():.4f} ({mapped.sum():,} / {len(raw):,} rows)")
print("top 15 UNMAPPED tourney names by row count:")
print(raw.loc[~mapped, "tourney_name"].value_counts().head(15).to_string())

raw["winner_home"] = (raw["winner_ioc"] == raw["tourney_country"]).astype(int)
raw["loser_home"] = (raw["loser_ioc"] == raw["tourney_country"]).astype(int)
print(f"\nhome-player rate (of mapped rows): winner={raw.loc[mapped,'winner_home'].mean():.4f}  "
      f"loser={raw.loc[mapped,'loser_home'].mean():.4f}")

raw_small = raw[["date", "tour", "winner_uid", "loser_uid", "winner_home", "loser_home"]].drop_duplicates(
    subset=["date", "tour", "winner_uid", "loser_uid"])

tune = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\xgb_features_v2_tune.parquet")
j = tune.merge(raw_small, on=["date", "tour", "winner_uid", "loser_uid"], how="left")
y = j["y"].values.astype(int)
hw, hl = j["winner_home"].values.astype(float), j["loser_home"].values.astype(float)
j["home_diff"] = np.where(y == 1, hw - hl, hl - hw)   # +1 side A home & B not, -1 reverse, 0 else
print(f"\nhome_diff nonzero rate in tune window: {(j['home_diff'] != 0).mean():.4f}  "
      f"(n={(j['home_diff'] != 0).sum()} matches with exactly one home player)")

BASE = ["elo_overall_diff", "elo_surface_diff", "elo_logit_pwinner", "hier_dmean_diff",
        "hier_dvar_diff", "hier_dvar_sum", "weeks_since_last_diff", "n_prior_diff",
        "h2h_diff", "age_diff", "fatigue_diff"]
CAT = ["surface", "round", "tourney_level"]
BOOLI = ["is_elite"]
TUNED = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.9,
         "min_child_weight": 5.0, "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0}


def cv_ll(numeric_cols):
    sub = j.dropna(subset=numeric_cols + CAT).copy()
    X = sub[numeric_cols].copy()
    for c in CAT:
        X[c] = sub[c].astype("category")
    for c in BOOLI:
        X[c] = sub[c].astype(int)
    y2 = sub["y"].values.astype(int)
    d = xgb.DMatrix(X, label=y2, enable_categorical=True)
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": -1, **TUNED}
    cv = xgb.cv(p, d, num_boost_round=1000, nfold=5, early_stopping_rounds=50,
                metrics=("logloss",), seed=0, as_pandas=True, verbose_eval=False)
    best = int(cv["test-logloss-mean"].idxmin())
    return float(cv["test-logloss-mean"].iloc[best]), best + 1, len(sub)


base_ll, r0, n0 = cv_ll(BASE)
ll, r, n = cv_ll(BASE + ["home_diff"])
print(f"\nBASE            n={n0:6d} round={r0:4d} cv_logloss={base_ll:.5f}")
print(f"BASE+home_diff  n={n:6d} round={r:4d} cv_logloss={ll:.5f}  delta={ll - base_ll:+.5f}")

# direct win-rate check on home_diff!=0 rows (cheap sanity, not a CV metric)
sub = j[j["home_diff"] != 0]
if len(sub):
    home_win_rate = (sub["home_diff"] == 1).mean()
    print(f"\nsanity: among matches with exactly one home player, that player won "
          f"{home_win_rate:.4f} of the time (n={len(sub)})")
