"""F4: size the train/serve head-to-head skew before deciding to fix it.

`train_best_model.py:153` builds the training `h2h_diff` from the UNFILTERED
elo parquet ("all levels, for the fullest possible H2H history"), while
`xgb_bt_rating.build_h2h_dict` serves it from `level_file == "tour"` rows only.
70.3% of the parquet's rows are non-tour, so the model is fed a systematically
smaller count than it learned on.

That share is not the effect. This measures the effect the way it actually
reaches the site: rebuild the real US Open field's pairwise matrices with each
head-to-head dictionary, everything else held identical, and compare the
probabilities that would have been published.

Reported: how many pairs' h2h_diff changes, by how much, and the resulting
move in p(a beats b) - mean, median, tail, and the effect on each player's
title-relevant row mean. Writes nothing to blog/.
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


def log(m: str = "") -> None:
    print(m, flush=True)


def h2h_dict(field: set[str], tour_only: bool) -> dict:
    """build_h2h_dict, with the level filter as a switch."""
    out: dict[tuple[str, str], list] = {}
    elo = pd.read_parquet(_common.SCRATCH / "elo_raw_ratings_full.parquet")
    if tour_only:
        elo = elo[elo["level_file"] == "tour"]
    elo = elo[elo["winner_uid"].isin(field) & elo["loser_uid"].isin(field)]
    for row in elo.itertuples():
        out.setdefault((row.winner_uid, row.loser_uid), []).append(row.date)
    return out


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

        h_tour = h2h_dict(set(field), tour_only=True)     # what is served today
        h_all = h2h_dict(set(field), tour_only=False)     # what training counted
        n_pairs_tour = sum(len(v) for v in h_tour.values())
        n_pairs_all = sum(len(v) for v in h_all.values())
        log(f"\n[{tour}] meetings among the field: tour-only {n_pairs_tour:,}, "
            f"all levels {n_pairs_all:,} (+{n_pairs_all - n_pairs_tour:,})")

        ptab = xbr.build_player_table(field, tour, DATE0, SURFACE, snap, playstyle_full, age_table)
        best_of = 5 if tour == "atp" else 3

        # h2h_diff both ways, over the exact ordered pairs the simulator uses
        diffs = []
        for i in field:
            for j in field:
                if i == j:
                    continue
                a1, b1 = xbr.h2h_wins_before(h_tour, i, j, DATE0)
                a2, b2 = xbr.h2h_wins_before(h_all, i, j, DATE0)
                diffs.append(((a1 - b1), (a2 - b2)))
        dd = pd.DataFrame(diffs, columns=["served", "trained"])
        changed = dd["served"] != dd["trained"]
        log(f"[{tour}] ordered pairs: {len(dd):,}; h2h_diff differs on {int(changed.sum()):,} "
            f"({changed.mean():.2%}); max |delta| {int((dd['trained'] - dd['served']).abs().max())}")

        # the probabilities each would publish, everything else identical
        m_tour = build_round_pred_matrices(field, ptab, tour, SURFACE, best_of, "G", h_tour,
                                           DATE0, cat_levels, models, STAGE_ORDER[:-1])
        m_all = build_round_pred_matrices(field, ptab, tour, SURFACE, best_of, "G", h_all,
                                          DATE0, cat_levels, models, STAGE_ORDER[:-1])
        for rnd in ("R128", "QF"):
            A, B = m_tour[rnd], m_all[rnd]
            delta = np.abs(A - B)
            off = ~np.eye(len(field), dtype=bool)
            v = delta[off]
            log(f"[{tour}] {rnd}: mean |dp| {np.nanmean(v):.5f}, median {np.nanmedian(v):.5f}, "
                f"p99 {np.nanpercentile(v, 99):.4f}, max {np.nanmax(v):.4f}")
            rows.append(dict(tour=tour, round=rnd, mean_abs_dp=float(np.nanmean(v)),
                             p99=float(np.nanpercentile(v, 99)), max=float(np.nanmax(v)),
                             pairs_changed=float(changed.mean())))

        # per-player: the row mean is what bt_rating and the sim's edge come from
        rm_t = np.nanmean(np.where(np.eye(len(field), dtype=bool), np.nan, m_tour["R128"]), axis=1)
        rm_a = np.nanmean(np.where(np.eye(len(field), dtype=bool), np.nan, m_all["R128"]), axis=1)
        shift = pd.Series(rm_a - rm_t, index=field).sort_values()
        names = d["names"]
        log(f"[{tour}] per-player mean win prob shift: max down "
            f"{shift.iloc[0]:+.4f} ({names.get(shift.index[0])}), max up "
            f"{shift.iloc[-1]:+.4f} ({names.get(shift.index[-1])})")

    out = _common.SCRATCH / "screen_h2h_train_serve_skew.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    log(f"\nwrote {out}")
    log("H2H_SKEW_COMPLETE")


if __name__ == "__main__":
    main()
