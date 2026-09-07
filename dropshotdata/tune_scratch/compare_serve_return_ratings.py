"""Three serve ratings, one tape, one question: which one is publishable?

Candidates, all frozen before the 2026 US Open and scored against its 88,993
points, so none of them has seen this data:

  hier         the hierarchical model's m_a, converted through its own sigmoid
  elo-point    paired serve/return Elo fed on share of service POINTS won
  elo-game     the same, fed on share of service GAMES held

The test that matters is DISPERSION, not correlation. All three order players
sensibly; the reason hier cannot be published is that its spread is roughly
twice the real spread on ATP and nearly five times on WTA, so the percentages
are confidently wrong at both ends even though the ranking is right.

Observed spread is inflated by sampling noise - a player with n service points
has a binomial sd of about sqrt(p(1-p)/n) - so that is removed before
comparing. Leaving it in flatters every candidate.

A ratio near 1.0 means the rating claims exactly as much spread as really
exists. Above 1.0 is over-confident, below is over-shrunk.
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
from serve_return_elo import expected, published_rate                 # noqa: E402
from serve_return_ratings import rate_field                           # noqa: E402

DATE0 = pd.Timestamp("2026-08-30")
MIN_PTS = 150


def log(m=""):
    print(m, flush=True)


def observed(tour: str, ev: str, draw, n2u):
    """Each player's share of their own service points and games at this event."""
    d = pd.read_parquet(_common.DD / "usopen_2026_points.parquet",
                        columns=["match_id", "event", "PointServer", "PointWinner", "GameWinner"])
    d = d[(d["match_id"].astype(str).str.len() == 4) & (d["event"] == ev)]
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
    pw, pn, gh, gn = {}, {}, {}, {}
    for mid, g in d.groupby("match_id", sort=False):
        mid = str(mid)
        if mid not in side:
            continue
        for srv, uid in zip(("1", "2"), side[mid]):
            p = g[g["PointServer"] == srv]
            if not len(p):
                continue
            pn[uid] = pn.get(uid, 0) + len(p)
            pw[uid] = pw.get(uid, 0) + int((p["PointWinner"] == srv).sum())
            ends = p[p["GameWinner"].astype(str) != "0"]
            gn[uid] = gn.get(uid, 0) + len(ends)
            gh[uid] = gh.get(uid, 0) + int((ends["GameWinner"] == srv).sum())
    return pw, pn, gh, gn


def dispersion(rated: pd.Series, obs: pd.Series, n: pd.Series):
    """(rated sd, true sd, ratio) - ratio is None when it is not measurable.

    The noise correction subtracts sampling variance from observed variance,
    and with few trials per player the remainder can be at or below zero: the
    spread that was observed is entirely explained by chance, so there is no
    measurable true spread to compare against. Dividing by that produced 1775.59
    for WTA hold rates on about 25 service games a player - a number that reads
    like a catastrophic finding and is a division by almost zero. It is now
    reported as not measurable, which is what it is.
    """
    p_bar = float(obs.mean())
    noise = float(np.mean(p_bar * (1 - p_bar) / n))
    resid = float(obs.var() - noise)
    sd_rated = float(rated.std())
    if resid <= 0.15 * float(obs.var()):
        # the observed spread is within, or barely above, what chance alone
        # would produce at this sample size
        return sd_rated, float(np.sqrt(max(resid, 0.0))), None
    sd_true = float(np.sqrt(resid))
    return sd_rated, sd_true, sd_rated / sd_true


df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
n2u = dm.build_name_to_uid(df)
elo = {m: pd.read_parquet(_common.SCRATCH / f"serve_return_elo_final_{m}.parquet")
       for m in ("point", "game")}

log(f"{'tour':>5} {'candidate':>12} {'unit':>12} {'rated sd':>9} {'true sd':>9} "
    f"{'ratio':>7} {'corr':>7} {'n':>4}")
for tour, ev in (("atp", "1"), ("wta", "2")):
    draw = dm.load_live_draw(tour, n2u)
    field = sorted(set(draw["bracket"]))
    key = f"{tour}:2026-usopen"
    snap = stu.build_snapshots({key: {"date0": DATE0, "field_uids": field}}, df=df,
                               cache_dir=_common.SCRATCH / "snapshot_cache").get(key, {})
    hier = rate_field(snap, field).set_index("uid")
    pw, pn, gh, gn = observed(tour, ev, draw, n2u)

    for mode in ("point", "game"):
        e = elo[mode]
        e = e[e["tour"] == tour].set_index("uid")
        est = e[e["n_matches"] >= 20]
        r_avg = float(est["return_elo"].mean())
        rows = []
        for uid, n in pn.items():
            if n < MIN_PTS or uid not in e.index:
                continue
            # the shrunk number, which is what would actually be shown
            rated = published_rate(float(e.loc[uid, "serve_elo"]), r_avg, tour, mode)
            obs = (pw[uid] / n) if mode == "point" else (gh.get(uid, 0) / max(gn.get(uid, 0), 1))
            trials = n if mode == "point" else max(gn.get(uid, 0), 1)
            rows.append(dict(rated=rated, obs=obs, n=trials))
        t = pd.DataFrame(rows)
        if len(t) < 10:
            continue
        sd_r, sd_t, ratio = dispersion(t["rated"], t["obs"], t["n"])
        unit = "serve points" if mode == "point" else "games held"
        rs = f"{ratio:>7.2f}" if ratio is not None else f"{'n/a':>7}"
        log(f"{tour:>5} {'elo-' + mode:>12} {unit:>12} {sd_r:>9.4f} {sd_t:>9.4f} "
            f"{rs} {t['rated'].corr(t['obs']):>+7.3f} {len(t):>4}")

    rows = []
    for uid, n in pn.items():
        if n < MIN_PTS or uid not in hier.index:
            continue
        rows.append(dict(rated=float(hier.loc[uid, "serve_rating"]), obs=pw[uid] / n, n=n))
    t = pd.DataFrame(rows)
    sd_r, sd_t, ratio = dispersion(t["rated"], t["obs"], t["n"])
    rs = f"{ratio:>7.2f}" if ratio is not None else f"{'n/a':>7}"
    log(f"{tour:>5} {'hier':>12} {'serve points':>12} {sd_r:>9.4f} {sd_t:>9.4f} "
        f"{rs} {t['rated'].corr(t['obs']):>+7.3f} {len(t):>4}")

log("\nratio near 1.00 is the publishable one; above 1.25 is over-confident.")
log("COMPARE_COMPLETE")
