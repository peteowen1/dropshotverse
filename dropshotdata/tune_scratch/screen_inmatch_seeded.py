"""W4 arm C: seed the in-match recursion with the model's pre-match skill.

Ticket 37 scored two arms on the 2026 US Open point tape and left the obvious
third one open because the tape carries "P1"/"P2" and names in prose, not uids:

  A  both players serve at the tour average. Measures the RECURSION alone.
     ECE 0.0471, log-loss 0.56916.
  B  each player's serve rate estimated online from this match so far.
     ECE 0.0737, worse - it learns slowly and early points are noisy.
  C  (this) both players' serve rates chosen so that the recursion's FRESH
     match probability equals the shipped model's pre-match probability for
     that pairing. All the pre-match skill, none of the in-match noise.

The crosswalk ticket 37 lacked now exists: the point tape's `match_id` is the
draw feed's own id, and usopen_live_draw resolves the feed's players to uids
(and since 2026-09-07 resolves the two it used to miss).

SOLVE. Hold the pair's serve sum at twice the tour average, so arm C differs
from arm A on the DIFFERENCE in skill only and cannot win by making both
players better servers. Then bisect the difference until
p_match(p1, p2, best_of) equals the model's pre-match number.

METRIC, and the bar, both from the critique's W4 before this ran:
go if ECE < 0.03 and log-loss < 0.55, against arm A's 0.0471 / 0.56916,
on the same points and the same matches.

ANCHORS, fixed before looking at output
  C1  the tape's P1/P2 must be the feed's team1/team2. Checked by comparing
      the tape's MatchWinner with the feed's own winner, per match. Any
      disagreement means the mapping is inverted and every number below would
      be confidently backwards.
  C2  at 0-0, arm C's stated probability must equal the model's pre-match
      probability to within 0.005 - that is what the solve is for. This anchor
      earned its place: the first run failed it at 0.496 because the bisection
      searched only positive skill differences, so every match with an
      underdog P1 was silently published as a coin flip while the aggregate
      log-loss still beat the baseline.
  C3  on the final point, the eventual winner must be above 0.5 in over 90%
      of matches (ticket 37's own anchor, so the arms are comparable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import _common                                                       # noqa: E402
import bridge_tennismylife as bridge                                 # noqa: E402
import hierarchical_model as hm                                      # noqa: E402
import simulate_tournament_updating as stu                           # noqa: E402
import usopen_live_draw as draw_mod                                  # noqa: E402
import xgb_bt_rating as xbr                                          # noqa: E402
from simulate_tournament import STAGE_ORDER                          # noqa: E402
from simulate_tournament_xgb import build_round_pred_matrices        # noqa: E402
from inmatch_seeding import SERVE_BASE, solve_serves, wp             # noqa: E402
from ticket09_live_modes import point_count, set_is_decided          # noqa: E402

PTS = _common.DD / "usopen_2026_points.parquet"
OUT = _common.SCRATCH / "screen_inmatch_seeded.parquet"
DATE0 = pd.Timestamp("2026-08-30")
SURFACE = "Hard"
EVENT_TOUR = {"1": "atp", "2": "wta"}
ARM_A_ECE, ARM_A_LL = 0.0471, 0.56916      # ticket 37, the number to beat
GO_ECE, GO_LL = 0.03, 0.55                 # the critique's W4 bar


def log(m: str = "") -> None:
    print(m, flush=True)


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
                     for m in (idx == b for b in range(bins)) if m.sum()))


def main() -> None:
    d = pd.read_parquet(PTS)
    d["best_of"] = np.where(d["event"] == "1", 5, 3)
    # D-58: tour_avg for the SERVE_BASE drift check must be MAIN DRAW ONLY.
    # 5-character match ids are qualifying/doubles/juniors (rescore_after_final.py's
    # round_coverage() already knows this) and measurably contaminate the average -
    # confirmed on real players, e.g. men's names under a "2" (women's) 5-char id.
    # On the 2026 tape this moved the ATP figure 0.6109 -> 0.6276, a real 0.017 shift,
    # not noise. The scoring below is unaffected (matches without a live-draw
    # pre-match number are already skipped), only this drift check was contaminated.
    main_draw = d[d["match_id"].astype(str).str.len() == 4]
    tour_avg = main_draw.groupby("event").apply(
        lambda g: float((g["PointServer"] == g["PointWinner"]).mean()), include_groups=False)
    # The shipped path cannot recompute this - the runner has no point tape -
    # so it carries the same rates as a constant. Assert they still agree, or
    # the thing measured here stops being the thing that runs.
    for ev, tour in EVENT_TOUR.items():
        drift = abs(float(tour_avg[ev]) - SERVE_BASE[tour])
        if drift > 5e-4:
            raise SystemExit(f"{tour}: tape serve rate {tour_avg[ev]:.6f} has drifted "
                             f"{drift:.6f} from the shipped SERVE_BASE {SERVE_BASE[tour]:.6f}; "
                             f"update inmatch_seeding.SERVE_BASE before trusting this run")
    log(f"points {len(d):,} across {d['match_id'].nunique():,} matches; "
        f"tour-average serve win " + ", ".join(f"{k}={v:.4f}" for k, v in tour_avg.items()))

    log("\nbuilding the model's pre-match matrix (the same one the live sim publishes) ...")
    models, cat_levels, playstyle_full = xbr.train_bagged_model()
    age_table = xbr.load_age_table()
    df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
    n2u = draw_mod.build_name_to_uid(df)

    pre: dict[str, tuple[float, str, str]] = {}   # match_id -> (p(P1 beats P2), uid1, uid2)
    feed_winner: dict[str, int] = {}              # match_id -> 1 or 2, per the feed
    for ev, tour in EVENT_TOUR.items():
        draw = draw_mod.load_live_draw(tour, n2u)
        field = sorted(set(draw["bracket"]))
        key = f"{tour}:2026-usopen"
        snap = stu.build_snapshots({key: {"date0": DATE0, "field_uids": field}}, df=df,
                                   cache_dir=_common.SCRATCH / "snapshot_cache").get(key, {})
        h2h = xbr.build_h2h_dict(set(field))
        ptab = xbr.build_player_table(field, tour, DATE0, SURFACE, snap, playstyle_full, age_table)
        bo = 5 if tour == "atp" else 3
        mats = build_round_pred_matrices(field, ptab, tour, SURFACE, bo, "G", h2h, DATE0,
                                         cat_levels, models, STAGE_ORDER[:-1])
        idx = {u: k for k, u in enumerate(field)}
        for m in draw["raw"]["matches"]:
            mid = str(m.get("match_id") or "")
            if not mid or mid[0] != ev:
                continue
            p1, p2 = draw_mod._player(m.get("team1")), draw_mod._player(m.get("team2"))
            if not (p1 and p2):
                continue
            u1 = draw_mod.lookup_uid(n2u, p1[0]) or f"{tour}:NEW_{p1[0].replace(' ', '_')}"
            u2 = draw_mod.lookup_uid(n2u, p2[0]) or f"{tour}:NEW_{p2[0].replace(' ', '_')}"
            if u1 not in idx or u2 not in idx:
                continue
            rnd = STAGE_ORDER[draw_mod.parse_match_id(mid)[0] - 1]
            pre[mid] = (float(mats[rnd][idx[u1], idx[u2]]), u1, u2)
            w = m.get("winner")
            if w in ("1", "2"):
                feed_winner[mid] = int(w)
        log(f"  [{tour}] pre-match probabilities for {sum(1 for k in pre if k[0] == ev):,} feed matches")

    # ---- anchor C1: is the tape's P1 the feed's team1?
    agree = dis = 0
    for mid, g in d.groupby("match_id", sort=False):
        mw = g["MatchWinner"].iloc[-1]
        if mid in feed_winner and mw in ("1", "2"):
            agree += int(int(mw) == feed_winner[mid])
            dis += int(int(mw) != feed_winner[mid])
    log(f"\nanchor C1 tape P1 == feed team1: {agree} agree, {dis} disagree "
        f"({agree / max(agree + dis, 1):.1%})")
    if agree + dis and agree / (agree + dis) < 0.95:
        raise SystemExit("C1 FAILED: the tape's P1/P2 is not the feed's team1/team2; "
                         "every probability below would be backwards")

    rows = []
    skipped = 0
    for mid, g in d.groupby("match_id", sort=False):
        if mid not in pre:
            skipped += 1
            continue
        g = g.sort_values("point_no")
        mw = g["MatchWinner"].iloc[-1]
        if mw not in ("1", "2"):
            skipped += 1
            continue
        y1 = int(mw == "1")
        bo = int(g["best_of"].iloc[0])
        base = float(tour_avg[g["event"].iloc[0]])
        p_pre = pre[mid][0]
        first_server_is_p1 = bool(g["PointServer"].iloc[0] == "1")
        c1, c2 = solve_serves(p_pre, base, bo, first_server_is_p1)
        # ---- reference frame ------------------------------------------
        # Every STATE column on the tape is recorded AFTER the point: at point
        # 1 the score already reads 15-0, and on the last point of match 1101
        # P1SetsWon is 3, i.e. the match is over. `PointServer`, by contrast,
        # is the server OF that point, which is a before-the-point fact.
        #
        # Scoring the post-point state against the pre-point server mixes the
        # two, and it is wrong precisely on the points where it matters: when a
        # point ends a game the serve passes, so the recursion was told the
        # wrong player serves next on every game-ending point.
        #
        # It also made anchor C3 - "the eventual winner is above 0.5 on the
        # final point" - impossible to fail, because the final point's
        # post-state IS the finished match and the recursion returns exactly
        # 1.0. It reported 100% for arm A and arm C alike; it would report 100%
        # for any model at all.
        #
        # So the state is shifted by one within the match: what a point is
        # scored from is the state as that point BEGINS, which is the previous
        # point's post-state, and 0-0 0-0 for the first point. That pairs
        # correctly with PointServer, makes the first point's probability the
        # pre-match number by construction (anchor C2), and makes the last
        # point a real forecast again.
        #
        # The post-point probability is kept alongside it. The difference
        # between the two is what winning that point was worth.
        st = g[["P1SetsWon", "P2SetsWon", "P1GamesWon", "P2GamesWon"]].fillna(0).astype(int)
        # The WITHIN-GAME score, which the recursion ignored until now. The
        # tape resets it to 0-0 on the point that ends a game, so shifting it
        # by one alongside the other state columns lands the right thing: the
        # first point of a game sees 0-0, and a point mid-game sees the score
        # it was actually played at.
        tb = [bool(x) for x in g["is_tb"]]
        pts = [(point_count(a, t), point_count(b, t))
               for a, b, t in zip(g["P1Score"], g["P2Score"], tb)]
        # Which rows begin a new game, from the tape's OWN numbering. The score
        # is normally reset to 0-0 on the row that ends a game, so shifting it
        # lands 0-0 on the next row by itself - but not always: 3 of the 13,203
        # game boundaries have rows missing or a live score correction, and
        # there a stale score would leak into the first point of a new game.
        # Reading (SetNo, GameNo) makes it exact rather than usually right.
        keys = list(zip(g["SetNo"].astype(str), g["GameNo"].astype(str)))
        starts_game = [True] + [keys[i] != keys[i - 1] for i in range(1, len(keys))]
        # NOT named `pre`: that is the dict of pre-match probabilities this
        # function is built around, and shadowing it silently skipped every
        # match after the first.
        post_state = list(st.itertuples(index=False, name=None))
        pre_state = [(0, 0, 0, 0)] + post_state[:-1]
        post_pts = pts
        pre_pts = [(0, 0) if starts_game[i] else pts[i - 1] if i else (0, 0)
                   for i in range(len(pts))]
        servers = [x == "1" for x in g["PointServer"]]

        for k, r in enumerate(g.itertuples()):
            a1, a2, ag1, ag2 = pre_state[k]
            b1, b2, bg1, bg2 = post_state[k]
            # The set-ending row carries the new set count AND the finished
            # set's games at once (match 1101 point 70: sets 1-0, games 6-4),
            # so those games are a leftover the set count already reflects.
            # True of the shifted state exactly as it was of the raw one.
            if set_is_decided(ag1, ag2):
                ag1 = ag2 = 0
            if set_is_decided(bg1, bg2):
                bg1 = bg2 = 0
            srv = servers[k]
            # who serves the next point, for the post-point frame: the same
            # player unless this point ended a game, in which case it passes
            srv_next = (not srv) if str(getattr(r, "GameWinner", "0")) != "0" else srv
            ap1, ap2 = pre_pts[k]
            bp1, bp2 = post_pts[k]
            rows.append((mid, r.point_no, y1,
                         wp(base, base, a1, a2, ag1, ag2, srv, bo, ap1, ap2),
                         wp(c1, c2, a1, a2, ag1, ag2, srv, bo, ap1, ap2),
                         p_pre,
                         wp(base, base, b1, b2, bg1, bg2, srv_next, bo, bp1, bp2),
                         wp(c1, c2, b1, b2, bg1, bg2, srv_next, bo, bp1, bp2)))

    log(f"scored {len({r[0] for r in rows}):,} matches, {len(rows):,} points "
        f"({skipped} matches skipped: no pre-match number or no clean result)")

    res = pd.DataFrame(rows, columns=["match_id", "point_no", "y1", "p_a", "p_c", "p_pre",
                                      "p_a_post", "p_c_post"])
    # What winning the point was worth, in match-win-probability terms. The
    # post-point number of one point is the pre-point number of the next, so
    # this is the per-point increment the live product would actually show.
    res["swing_c"] = res["p_c_post"] - res["p_c"]

    # ---- anchor C2: at 0-0 arm C must restate the pre-match number
    first = res.sort_values("point_no").groupby("match_id").first()
    dev = float((first["p_c"] - first["p_pre"]).abs().max())
    log(f"anchor C2 max |arm C at 0-0 minus pre-match| = {dev:.5f} "
        f"{'PASS' if dev < 0.005 else 'FAIL'}")

    # ---- anchor C3: winner above 0.5 on the final point
    last = res.sort_values("point_no").groupby("match_id").last()
    share = float((np.where(last["y1"] == 1, last["p_c"], 1 - last["p_c"]) > 0.5).mean())
    log(f"anchor C3 winner above 0.5 on the final point: {share:.1%} "
        f"{'PASS' if share > 0.90 else 'FAIL'}")

    sw = res["swing_c"].abs()
    log(f"\nper-point swing in arm C: mean |delta| {sw.mean():.4f}, "
        f"median {sw.median():.4f}, p99 {sw.quantile(0.99):.4f}, max {sw.max():.4f}")
    log("  (the pre-point number of a point is the post-point number of the one "
        "before it, so this is the increment the live product would show)")

    log("\n=== calibration over every point, scored as each point BEGAN ===")
    out = []
    for arm, col in (("A fixed skill (ticket 37)", "p_a"), ("C model-seeded", "p_c")):
        e, l = ece(res[col], res["y1"]), logloss(res[col], res["y1"])
        acc = float(((res[col] > 0.5).astype(int) == res["y1"]).mean())
        log(f"  {arm:<26} ECE {e:.4f}   log-loss {l:.5f}   acc {acc:.3f}")
        out.append(dict(arm=arm, ece=e, logloss=l, acc=acc, n_points=len(res),
                        n_matches=int(res['match_id'].nunique())))
    e_c, l_c = out[1]["ece"], out[1]["logloss"]
    log(f"\n  ticket 37 arm A reference: ECE {ARM_A_ECE:.4f}, log-loss {ARM_A_LL:.5f}")
    log(f"  critique W4 bar: ECE < {GO_ECE}, log-loss < {GO_LL}")
    log(f"  VERDICT: {'GO' if (e_c < GO_ECE and l_c < GO_LL) else 'NO-GO'} "
        f"(ECE {e_c:.4f}, log-loss {l_c:.5f})")

    pd.DataFrame(out).to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")
    # Per-point frame too: recalibration candidates need arm C's predictions
    # and must not pay the model + matrix rebuild again to get them.
    pts_out = _common.SCRATCH / "screen_inmatch_seeded_points.parquet"
    res.to_parquet(pts_out, index=False)
    log(f"wrote {pts_out} ({len(res):,} points)")
    log("INMATCH_SEEDED_COMPLETE")


if __name__ == "__main__":
    main()
