"""Closes the loop stats-discipline #10 calls for: simulate_tournament_xgb.py's
pooled log-loss beat Elo at 6/7 stages on point estimates alone (SF was a
wash, +0.0013, no CI). Tournament-block bootstrap (paired - same 16
tournaments, matching this session's established pattern) on the delta at
every stage."""
import numpy as np
import pandas as pd

DD = r"C:\dev\dropshotverse\dropshotdata\tune_scratch"

elo = pd.read_parquet(f"{DD}/simulate_tournament_yp_elo.parquet")
xgb = pd.read_parquet(f"{DD}/simulate_tournament_yp_xgb.parquet")

STAGE_ORDER_TAIL = ["R64", "R32", "R16", "QF", "SF", "F", "W"]


for df in (elo, xgb):
    df["key"] = df["tour"] + ":" + df["tourney_id"]   # tourney_id alone collides across tours (e.g. "2024-580"
                                                       # is both ATP and WTA's Australian Open) - would silently
                                                       # halve the block count from 16 to 8 if used bare.


def ll_by_tournament(df: pd.DataFrame, stage: str) -> pd.Series:
    """log-loss per (tour, tourney_id) key for one stage - the block unit for bootstrap."""
    sub = df[df["stage"] == stage]
    out = {}
    for key, g in sub.groupby("key"):
        y = g["y"].to_numpy()
        p = np.clip(g["p"].to_numpy(), 1e-9, 1 - 1e-9)
        out[key] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return pd.Series(out)


tids = sorted(elo["key"].unique())
assert sorted(xgb["key"].unique()) == tids, "tournament sets don't match between arms"

print(f"n={len(tids)} tournaments, paired tournament-block bootstrap (2000x)\n")
rng = np.random.default_rng(0)
for stage in STAGE_ORDER_TAIL:
    ll_e = ll_by_tournament(elo, stage)
    ll_x = ll_by_tournament(xgb, stage)
    d_by_t = (ll_x - ll_e).reindex(tids)   # xgb - elo per tournament (block)

    point_delta = d_by_t.mean()
    boot = []
    for _ in range(2000):
        sample = rng.choice(tids, size=len(tids), replace=True)
        boot.append(d_by_t.reindex(sample).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "xgb wins" if hi < 0 else ("elo wins" if lo > 0 else "unresolved (CI spans zero)")
    print(f"  >= {stage:4s}  mean(xgb-elo) per-tournament-avg-logloss delta = {point_delta:+.4f}  "
        f"CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")
