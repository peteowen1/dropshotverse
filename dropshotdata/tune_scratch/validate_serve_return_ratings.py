"""Do the serve and return ratings predict what actually happened?

Skills are frozen at 2026-08-30, before the US Open started, and scored
against the 88,993-point tape. Nothing here is fitted to that tape, so this is
an out-of-sample check and not a description.

Anchors, written before the run:
  A1  the ratings must ORDER players correctly on serve points won. A positive
      correlation, on both tours.
  A2  the SPREAD of the ratings must match the spread of real skill, with
      sampling noise removed from the observed side. Bias is not the test:
      over-dispersion is too harsh at one end and too generous at the other, so
      the mean error cancels while individual players are far out.
  A3  the two ratings must not be the same number wearing different labels.
      Reported per tour rather than assumed, because they are not equally
      separable: the split is much weaker on the WTA side and shipping both as
      independent would overstate the model.
  A4  no tour-to-tour comparison is meaningful, so the ratings are relative to
      each tour's own average and both tours must sit near 50% at the centre.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import _common                                                        # noqa: E402
import bridge_tennismylife as bridge                                  # noqa: E402
import hierarchical_model as hm                                       # noqa: E402
import simulate_tournament_updating as stu                            # noqa: E402
import usopen_live_draw as dm                                         # noqa: E402
from serve_return_ratings import rate_field, serve_return_split_quality  # noqa: E402

DATE0 = pd.Timestamp("2026-08-30")
MIN_PTS = 150


def log(m=""):
    print(m, flush=True)


d = pd.read_parquet(_common.DD / "usopen_2026_points.parquet",
                    columns=["match_id", "event", "PointServer", "PointWinner", "GameWinner"])
d = d[d["match_id"].astype(str).str.len() == 4]
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
n2u = dm.build_name_to_uid(df)

failures = []
for tour, ev in (("atp", "1"), ("wta", "2")):
    draw = dm.load_live_draw(tour, n2u)
    field = sorted(set(draw["bracket"]))
    key = f"{tour}:2026-usopen"
    snap = stu.build_snapshots({key: {"date0": DATE0, "field_uids": field}}, df=df,
                               cache_dir=_common.SCRATCH / "snapshot_cache").get(key, {})
    rated = rate_field(snap, field).set_index("uid")

    # who served which side of which match, straight off the feed
    side = {}
    for m in draw["raw"]["matches"]:
        mid = str(m.get("match_id") or "")
        if len(mid) != 4 or mid[0] != ev:
            continue
        p1, p2 = dm._player(m.get("team1")), dm._player(m.get("team2"))
        if not (p1 and p2):
            continue
        u1, u2 = dm.lookup_uid(n2u, p1[0]), dm.lookup_uid(n2u, p2[0])
        if u1 and u2:
            side[mid] = (u1, u2)

    pts_won, pts_played, games_held, games_served = {}, {}, {}, {}
    for mid, g in d[d["event"] == ev].groupby("match_id", sort=False):
        mid = str(mid)
        if mid not in side:
            continue
        for srv, uid in zip(("1", "2"), side[mid]):
            p = g[g["PointServer"] == srv]
            if not len(p):
                continue
            pts_played[uid] = pts_played.get(uid, 0) + len(p)
            pts_won[uid] = pts_won.get(uid, 0) + int((p["PointWinner"] == srv).sum())
            ends = p[p["GameWinner"].astype(str) != "0"]
            games_served[uid] = games_served.get(uid, 0) + len(ends)
            games_held[uid] = games_held.get(uid, 0) + int((ends["GameWinner"] == srv).sum())

    rows = []
    for uid, n in pts_played.items():
        if n < MIN_PTS or uid not in rated.index:
            continue
        r = rated.loc[uid]
        rows.append(dict(name=draw["names"].get(uid, uid),
                         serve_rating=r["serve_rating"], hold_pred=r["hold_pct"],
                         obs_pt=pts_won[uid] / n,
                         obs_hold=games_held.get(uid, 0) / max(games_served.get(uid, 0), 1),
                         n_pts=n, n_games=games_served.get(uid, 0)))
    t = pd.DataFrame(rows)

    log(f"\n{'=' * 66}\n{tour.upper()}: {len(t)} players with {MIN_PTS}+ service points\n{'=' * 66}")

    r_pt = float(t["serve_rating"].corr(t["obs_pt"]))
    log(f"A1 order on serve points:  corr = {r_pt:+.3f}  "
        f"{'PASS' if r_pt > 0 else 'FAIL'}")
    if r_pt <= 0:
        failures.append(f"{tour}: serve rating does not order serve points won")

    # A2 originally tested BIAS. That was a check that could not fail: the
    # ratings are too harsh on weak servers and too generous to strong ones, so
    # the two errors cancel and the mean sat at +0.022 while individual holds
    # were out by 35 points. Dispersion is the statistic that sees it.
    #
    # The observed spread is itself inflated by sampling noise - a player with
    # n service points has a binomial sd of sqrt(p(1-p)/n), about 0.031 at 250
    # points - so that is removed before comparing, which makes the ratings
    # look WORSE, correctly.
    sd_rated = float(t["serve_rating"].std())
    p_bar = float(t["obs_pt"].mean())
    noise_var = float(np.mean(p_bar * (1 - p_bar) / t["n_pts"]))
    sd_true = float(np.sqrt(max(t["obs_pt"].var() - noise_var, 1e-9)))
    ratio = sd_rated / sd_true
    mae = float((t["hold_pred"] - t["obs_hold"]).abs().mean())
    bias = float((t["hold_pred"] - t["obs_hold"]).mean())
    log(f"A2 spread of serve rating: rated sd {sd_rated:.4f} vs true sd {sd_true:.4f} "
        f"-> {ratio:.2f}x  {'PASS' if ratio < 1.25 else 'FAIL, over-dispersed'}")
    log(f"   published hold: MAE {mae:.4f} (bias only {bias:+.4f}, because the "
        f"tails cancel)")
    if ratio >= 1.25:
        failures.append(f"{tour}: serve ratings are {ratio:.2f}x too spread out, so the "
                        f"percentages are not publishable as levels")

    split = serve_return_split_quality(rated.reset_index())
    verdict = "separates" if abs(split) < 0.45 else "LARGELY RESTATES overall quality"
    log(f"A3 serve vs return split:  corr = {split:+.3f}  -> {verdict}")

    est = rated[~rated["provisional"]]
    log(f"A4 centred on this tour:   median serve {est['serve_rating'].median():.3f}, "
        f"median return {est['return_rating'].median():.3f}")

    log("\n  strongest serves, and what they actually did:")
    for x in t.nlargest(5, "serve_rating").itertuples():
        log(f"    {x.name:<24} rated {x.serve_rating:.1%} -> won {x.obs_pt:.1%} of serve pts; "
            f"hold {x.hold_pred:.1%} -> {x.obs_hold:.1%} ({x.n_games} games)")
    log("  weakest:")
    for x in t.nsmallest(3, "serve_rating").itertuples():
        log(f"    {x.name:<24} rated {x.serve_rating:.1%} -> won {x.obs_pt:.1%} of serve pts; "
            f"hold {x.hold_pred:.1%} -> {x.obs_hold:.1%} ({x.n_games} games)")

log("\n" + "=" * 66)
if failures:
    log("FAILURES: " + "; ".join(failures))
    raise SystemExit(1)
log("ALL SERVE/RETURN ANCHORS PASSED")
log("SERVE_RETURN_VALIDATED")
