"""Candidate 13b: split the axis. RET and W/O are different events.

screen_ret_in_elo.py treated "RET or W/O or DEF or ABD or blank score" as one
category and found ret_k_mult=0.5 worth -0.000514 on the 2022-23 tune window
(limiting case verified exactly: verify_ret_harness_identity.py, max |delta|
0.000e+00 on 89,678 rows). That bundles two mechanisms that have no reason to
behave alike (stats-discipline #5):

  RET  a real match was played and abandoned mid-way. The winner was ahead, or
       at least on court. There IS evidence here, just noisier than a
       completed match.
  W/O  the loser withdrew BEFORE play. The winner never hit a ball. Crediting
       them with a rating gain is crediting a result that did not happen.

If the whole -0.000514 comes from RET and W/O is flat-to-harmful, the shipped
fix should score retirements and keep dropping walkovers. If both contribute,
the simple version is fine. Either way the bundled number should not be shipped
without knowing which half it came from.

Grid over (ret_mult, wo_mult) on the 2022-23 tune window. The 2024-25 market
holdout stays untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))
sys.path.insert(0, str(DD / "tune_scratch"))

import elo_baseline as eb  # noqa: E402
from screen_ret_in_elo import load_all_levels, prepare_keeping_ret, logloss  # noqa: E402

BLEND, DECAY = 0.0, 0.3
EVAL_YEARS = (2022, 2023)
RET_PAT = r"RET|Retired"
WO_PAT = r"W/O|DEF|ABD|Walkover|Def\."


def log(m: str = "") -> None:
    print(m, flush=True)


def run2(df: pd.DataFrame, ret_mult: float, wo_mult: float) -> np.ndarray:
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values
    preds = np.full(len(df), np.nan)
    kind = df["kind"].values          # 'ok' | 'ret' | 'wo'

    for idx, (w, l, s, day, kd) in enumerate(zip(
            df["winner_uid"].values, df["loser_uid"].values, df["surface"].values,
            days, kind)):
        ro_w, ro_l = overall.get(w, eb.START), overall.get(l, eb.START)
        rs_w, rs_l = surf.get((w, s), eb.START), surf.get((l, s), eb.START)
        if DECAY > 0.0:
            ro_w = eb.layoff_decay(ro_w, day - last_all.get(w, day), DECAY)
            ro_l = eb.layoff_decay(ro_l, day - last_all.get(l, day), DECAY)
            rs_w = eb.layoff_decay(rs_w, day - last_surf.get((w, s), day), DECAY)
            rs_l = eb.layoff_decay(rs_l, day - last_surf.get((l, s), day), DECAY)

        diff = (1 - BLEND) * (ro_w - ro_l) + BLEND * (rs_w - rs_l)
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        if kd == "ok":
            preds[idx] = p_w

        mult = 1.0 if kd == "ok" else (ret_mult if kd == "ret" else wo_mult)
        if mult > 0.0:
            k_w = eb.k_factor(n_all.get(w, 0)) * mult
            k_l = eb.k_factor(n_all.get(l, 0)) * mult
            overall[w] = ro_w + k_w * (1 - p_w)
            overall[l] = ro_l - k_l * (1 - p_w)
            ks_w = eb.k_factor(n_surf.get((w, s), 0)) * mult
            ks_l = eb.k_factor(n_surf.get((l, s), 0)) * mult
            p_s = 1.0 / (1.0 + 10 ** (-(rs_w - rs_l) / 400.0))
            surf[(w, s)] = rs_w + ks_w * (1 - p_s)
            surf[(l, s)] = rs_l - ks_l * (1 - p_s)
            n_all[w] = n_all.get(w, 0) + 1
            n_all[l] = n_all.get(l, 0) + 1
            n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
            n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
            last_all[w] = last_all[l] = day
            last_surf[(w, s)] = last_surf[(l, s)] = day
    return preds


def main() -> None:
    df = prepare_keeping_ret(load_all_levels())
    sc = df["score"].astype(str)
    is_ret = sc.str.contains(RET_PAT, case=False, na=False)
    is_wo = sc.str.contains(WO_PAT, case=False, na=False) | (sc.str.strip() == "")
    df["kind"] = np.where(is_ret, "ret", np.where(is_wo, "wo", "ok"))
    log(f"coverage: {len(df):,} rows   "
        f"ret={int((df['kind'] == 'ret').sum()):,}  "
        f"wo/blank={int((df['kind'] == 'wo').sum()):,}  "
        f"ok={int((df['kind'] == 'ok').sum()):,}")

    yr = df["date"].dt.year
    ev = (yr.isin(EVAL_YEARS) & (df["kind"] == "ok")
          & (df["level_file"] == "tour")).values
    log(f"  eval population: {int(ev.sum()):,} tour-level completed {EVAL_YEARS}")

    grid = [(r, w) for r in (0.0, 0.25, 0.5, 0.75) for w in (0.0, 0.25, 0.5)]
    rows = []
    for r, w in grid:
        ll = logloss(run2(df, r, w)[ev])
        rows.append(dict(ret_mult=r, wo_mult=w, logloss=ll))
        log(f"  ret={r:<5} wo={w:<5} log-loss={ll:.6f}")

    tab = pd.DataFrame(rows)
    base = float(tab[(tab.ret_mult == 0) & (tab.wo_mult == 0)]["logloss"].iloc[0])
    tab["delta"] = tab["logloss"] - base
    log("\n--- grid, delta vs today's engine (ret=0, wo=0) ---")
    log(tab.pivot(index="ret_mult", columns="wo_mult", values="delta")
        .to_string(float_format=lambda x: f"{x:+.6f}"))

    best = tab.loc[tab["logloss"].idxmin()]
    log(f"\n  best: ret={best.ret_mult}, wo={best.wo_mult}, "
        f"delta {best.delta:+.6f}")

    # Which half carries it? Marginal effect of each axis with the other at 0.
    ret_only = tab[(tab.wo_mult == 0)].set_index("ret_mult")["delta"]
    wo_only = tab[(tab.ret_mult == 0)].set_index("wo_mult")["delta"]
    log(f"\n  RET axis alone (wo=0): "
        + ", ".join(f"{k}={v:+.6f}" for k, v in ret_only.items()))
    log(f"  W/O axis alone (ret=0): "
        + ", ".join(f"{k}={v:+.6f}" for k, v in wo_only.items()))
    log("\n  reading: whichever axis moves the metric on its own is the one "
        "carrying the bundled result.")
    tab.to_parquet(DD / "tune_scratch" / "screen_ret_vs_wo_split.parquet", index=False)


if __name__ == "__main__":
    main()
