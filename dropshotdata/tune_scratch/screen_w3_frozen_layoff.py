"""W3: size the frozen layoff before deciding whether it is worth fixing.

`build_player_table` computes `weeks_since_last` once, from DATE0, and the
simulator reuses that number for every round. So a player who arrived having
not played for four weeks is still four weeks idle in the model's eyes when it
simulates the final, two weeks later, by which point they would in fact have
played six matches.

Whether that matters is a question about the DIFFERENCE, not the level. The
feature the model sees is `weeks_since_last_diff`. In round one it is exactly
right. From round two on, both players in any given match have just played
their previous round, so the true difference collapses toward zero - while the
simulator keeps feeding it the pre-tournament gap.

This measures the effect the way it reaches the site: rebuild the real US Open
field's pairwise matrices with the layoff difference zeroed from round two
onward, everything else held identical, and compare the probabilities that
would have been published.

Reported: the spread of the frozen feature across the field, the resulting move
in p(a beats b) per round, and the effect on each player's title-relevant row
mean. Writes nothing to blog/.

Sized before built, per the effect-size rule: removing an error component of
spread s from a total spread sigma buys roughly s^2/2sigma^2. A mechanism being
real says nothing about it being worth a day.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import _common                                                        # noqa: E402
import bridge_tennismylife as bridge                                  # noqa: E402
import hierarchical_model as hm                                       # noqa: E402
import simulate_tournament_updating as stu                            # noqa: E402
import usopen_live_draw as draw_mod                                   # noqa: E402
import xgb_bt_rating as xbr                                           # noqa: E402
from simulate_tournament import STAGE_ORDER                           # noqa: E402
from simulate_tournament_xgb import build_round_pred_matrices         # noqa: E402

DATE0 = pd.Timestamp("2026-08-30")
SURFACE = "Hard"
OUT = _common.SCRATCH / "screen_w3_frozen_layoff.parquet"


def log(m: str = "") -> None:
    print(m, flush=True)


def main() -> None:
    log("loading model and bridged archive ...")
    models, cat_levels, playstyle_full = xbr.train_bagged_model()
    age_table = xbr.load_age_table()
    df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
    n2u = draw_mod.build_name_to_uid(df)

    rows = []
    for tour in ("atp", "wta"):
        d = draw_mod.load_live_draw(tour, n2u)
        field = sorted(set(d["bracket"]))
        key = f"{tour}:2026-usopen"
        snap = stu.build_snapshots({key: {"date0": DATE0, "field_uids": field}}, df=df,
                                   cache_dir=_common.SCRATCH / "snapshot_cache").get(key, {})
        best_of = 5 if tour == "atp" else 3

        ptab = xbr.build_player_table(field, tour, DATE0, SURFACE, snap,
                                      playstyle_full, age_table)
        # ptab is a DataFrame indexed by uid, not a dict of dicts.
        wsl = ptab.loc[field, "weeks_since_last"].astype(float)
        log(f"\n[{tour}] weeks_since_last across the field: "
            f"median {wsl.median():.2f}, mean {wsl.mean():.2f}, "
            f"p90 {wsl.quantile(0.90):.2f}, max {wsl.max():.2f}")
        # what the model is actually fed: the pairwise DIFFERENCE
        v = wsl.values
        diff = np.abs(v[:, None] - v[None, :])
        off = ~np.eye(len(field), dtype=bool)
        log(f"[{tour}] |weeks_since_last_diff| over every pair: "
            f"median {np.median(diff[off]):.2f}, mean {diff[off].mean():.2f}, "
            f"p99 {np.percentile(diff[off], 99):.2f}")

        # the counterfactual: from round two on, both players have just played,
        # so the difference is zero. Level held at the field median so only the
        # DIFFERENCE changes and this cannot move through the level as well.
        ptab_now = ptab.copy()
        ptab_now["weeks_since_last"] = float(wsl.median())

        m_frozen = build_round_pred_matrices(field, ptab, tour, SURFACE, best_of, "G",
                                             xbr.build_h2h_dict(set(field)), DATE0,
                                             cat_levels, models, STAGE_ORDER[:-1])
        m_now = build_round_pred_matrices(field, ptab_now, tour, SURFACE, best_of, "G",
                                          xbr.build_h2h_dict(set(field)), DATE0,
                                          cat_levels, models, STAGE_ORDER[:-1])

        for rnd in ("R128", "R32", "QF", "SF", "F"):
            if rnd not in m_frozen:
                continue
            delta = np.abs(m_frozen[rnd] - m_now[rnd])[off]
            log(f"[{tour}] {rnd}: mean |dp| {np.nanmean(delta):.5f}, "
                f"median {np.nanmedian(delta):.5f}, "
                f"p99 {np.nanpercentile(delta, 99):.4f}, max {np.nanmax(delta):.4f}")
            rows.append(dict(tour=tour, round=rnd, mean_abs_dp=float(np.nanmean(delta)),
                             p99=float(np.nanpercentile(delta, 99)),
                             max=float(np.nanmax(delta))))

        # per-player: the row mean is what the sim's edge comes from
        eye = np.eye(len(field), dtype=bool)
        rm_f = np.nanmean(np.where(eye, np.nan, m_frozen["QF"]), axis=1)
        rm_n = np.nanmean(np.where(eye, np.nan, m_now["QF"]), axis=1)
        shift = pd.Series(rm_n - rm_f, index=field).sort_values()
        names = d["names"]
        log(f"[{tour}] per-player mean win prob shift at QF: "
            f"max down {shift.iloc[0]:+.4f} ({names.get(shift.index[0])}), "
            f"max up {shift.iloc[-1]:+.4f} ({names.get(shift.index[-1])})")

    res = pd.DataFrame(rows)
    res.to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")

    worst = res["mean_abs_dp"].max()
    log("\n=== verdict ===")
    log(f"largest mean |dp| in any round: {worst:.5f}")
    log("A published match probability moving by less than about 0.005 on")
    log("average is a correctness note, not a fix worth shipping: it is far")
    log("below the model's own calibration error (ECE ~0.03) and nothing")
    log("downstream could distinguish it.")
    log(f"VERDICT: {'WORTH FIXING' if worst >= 0.005 else 'NOT WORTH FIXING'}")
    log("W3_LAYOFF_COMPLETE")


if __name__ == "__main__":
    main()
