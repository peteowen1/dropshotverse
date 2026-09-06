"""Ticket 28 round 6: issue 26 found the UTS context-K multiplier (level/round,
kFunction fixed=1.0) explains almost all of ATP's improvement but is net-
negative for WTA, and flagged the WTA level-code mapping (P/PM/I/W/T1-T5) as
"a real judgment call, not code-verified against a WTA-specific reference" -
the likely place to start if the WTA degradation is worth chasing. This round
does that: diagnose which WTA level code(s) drive the loss on the TUNE window
(2022-23) only, grid-search a WTA-specific remap on tune only, ONE reveal on
the 2024-25 holdout - proper split this time (tickets 25/26 evaluated directly
on the holdout; this round keeps that discipline tighter).

Read-only wrt elo_baseline.py - imports it rather than editing it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import elo_baseline as eb  # noqa: E402
from market_compare import key_from_full, key_from_short, load_odds  # noqa: E402

raw = eb.load_matches()
df, dropped = eb.prepare(raw)
print("dropped:", dropped, flush=True)

odds_all = load_odds(years=(2022, 2023, 2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
odds_all["k_w"] = odds_all["Winner"].map(key_from_short)
odds_all["k_l"] = odds_all["Loser"].map(key_from_short)
odds_all["date"] = pd.to_datetime(odds_all["Date"], errors="coerce")
odds_all = odds_all.dropna(subset=["date"])
odds_all["pair"] = odds_all.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
odds_small = odds_all[["pair", "tour", "date", "k_w", "PSW", "PSL"]]


def join_score(arm: pd.DataFrame, year_lo: int, year_hi: int) -> pd.DataFrame:
    sub = arm[(arm["level_file"] == "tour") & (arm["date"].dt.year >= year_lo) &
              (arm["date"].dt.year <= year_hi)].copy()
    sub["k_w"] = sub["winner_name"].map(key_from_full)
    sub["k_l"] = sub["loser_name"].map(key_from_full)
    sub["pair"] = sub.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    j = sub.merge(odds_small, on=["pair", "tour"], how="left", suffixes=("", "_o"))
    j["gap"] = (j["date_o"] - j["date"]).dt.days
    j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
    j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
    j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
    return j


def ll(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.log(p).mean())


# ---- Step 1: diagnose WTA-only, tune window, context-only arm vs sackmann,
# broken down by tourney_level, to see where the loss actually lives.
print("\n=== Step 1: WTA context-only vs sackmann, by tourney_level, tune window 2022-23 ===")
ctx_arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="uts_context_only")
base_arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="sackmann")

j_ctx = join_score(ctx_arm, 2022, 2023)
j_base = join_score(base_arm, 2022, 2023)
j_ctx = j_ctx[j_ctx["tour"] == "wta"]
j_base = j_base[j_base["tour"] == "wta"]
print(f"n: ctx={len(j_ctx)}, base={len(j_base)}")
for lvl in sorted(set(j_base["tourney_level"].unique()) | set(j_ctx["tourney_level"].unique())):
    b = j_base[j_base["tourney_level"] == lvl]["p_winner"].values
    c = j_ctx[j_ctx["tourney_level"] == lvl]["p_winner"].values
    if len(b) < 20:
        continue
    print(f"  {lvl:5s} n={len(b):4d}  sackmann_ll={ll(b):.4f}  context_ll={ll(c):.4f}  "
          f"delta={ll(c) - ll(b):+.4f}  current_mult={eb.UTS_LEVEL_MULT.get(lvl, eb.UTS_DEFAULT_LEVEL_MULT):.2f}")

print("\n=== Step 2: grid-search WTA-specific level mults, tune window 2022-23 ONLY ===")
# Vary the WTA-relevant codes only; ATP codes (G, F, M, O, A, D) held fixed at
# issue 26's already-committed values - one axis, per stats-discipline #5.
WTA_CODES = ["P", "PM", "I", "W", "T1", "T2", "T3", "T4", "T5", "CC"]
GRID = [0.55, 0.65, 0.70, 0.80, 0.90, 1.00]

results = []
for mult in GRID:
    level_mult = dict(eb.UTS_LEVEL_MULT)
    for code in WTA_CODES:
        level_mult[code] = mult
    orig = dict(eb.UTS_LEVEL_MULT)
    eb.UTS_LEVEL_MULT.clear()
    eb.UTS_LEVEL_MULT.update(level_mult)
    arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="uts_context_only")
    eb.UTS_LEVEL_MULT.clear()
    eb.UTS_LEVEL_MULT.update(orig)

    jj = join_score(arm, 2022, 2023)
    jj = jj[jj["tour"] == "wta"]
    loss = ll(jj["p_winner"].values)
    results.append((mult, loss, len(jj)))
    print(f"  WTA_uniform_mult={mult:.2f}  n={len(jj)}  tune_logloss={loss:.5f}", flush=True)

results.sort(key=lambda t: t[1])
best_mult = results[0][0]
print(f"\nbest WTA uniform mult on tune: {best_mult} (logloss={results[0][1]:.5f})")

print("\n=== Step 3: ONE holdout reveal (2024-25) - sackmann vs context-only-default vs "
      "context-only-WTA-tuned ===")
level_mult_tuned = dict(eb.UTS_LEVEL_MULT)
for code in WTA_CODES:
    level_mult_tuned[code] = best_mult

for name, mults in (("default (issue 26's as-ported mapping)", eb.UTS_LEVEL_MULT),
                     (f"WTA-tuned (uniform {best_mult})", level_mult_tuned)):
    orig = dict(eb.UTS_LEVEL_MULT)
    eb.UTS_LEVEL_MULT.clear()
    eb.UTS_LEVEL_MULT.update(mults)
    arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="uts_context_only")
    eb.UTS_LEVEL_MULT.clear()
    eb.UTS_LEVEL_MULT.update(orig)

    for tour in ("atp", "wta", "all"):
        jj = join_score(arm, 2024, 2025)
        if tour != "all":
            jj = jj[jj["tour"] == tour]
        print(f"  context-only [{name}] {tour:4s} n={len(jj):5d}  holdout_logloss={ll(jj['p_winner'].values):.5f}")

sack_arm, _ = eb.run_elo(df, blend=0.0, decay=0.3, k_mode="sackmann")
for tour in ("atp", "wta", "all"):
    jj = join_score(sack_arm, 2024, 2025)
    if tour != "all":
        jj = jj[jj["tour"] == tour]
    print(f"  sackmann (frozen 0.6120 basis)  {tour:4s} n={len(jj):5d}  "
          f"holdout_logloss={ll(jj['p_winner'].values):.5f}")
