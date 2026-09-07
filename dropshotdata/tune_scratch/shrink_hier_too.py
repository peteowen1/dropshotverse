"""Can the same shrink fix hier? Yes - and here it is done without cheating.

The shrink does not care where a rating came from. It rescales deviations from
the mean, so it can drive any candidate's dispersion to 1.00 while leaving its
ordering untouched. hier's only real weakness on the 2026 tape is a dispersion
ratio of 1.96 against a correlation of +0.478 that is the best of any candidate
- so if the ratio is free to fix, hier is the rating to publish.

THE TRAP THIS AVOIDS. That 1.96 was measured on the tape. Deriving the shrink
factor from it and then reporting a ratio of 1.00 on the same tape would be
circular - of course it lands on 1.00, it was fitted there. So the players are
split in half: the factor is derived on one half and applied to the other,
which the factor has never seen. A shrink that only works on the half it was
fitted to is not a fix.

Split is on a hash of the uid, so it is deterministic and has nothing to do
with how well anyone served.
"""
from __future__ import annotations

import hashlib
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
from serve_return_elo import expected                                 # noqa: E402
from serve_return_ratings import rate_field                           # noqa: E402

DATE0 = pd.Timestamp("2026-08-30")
MIN_PTS = 150


def log(m=""):
    print(m, flush=True)


def half(uid: str) -> int:
    return hashlib.sha256(uid.encode()).digest()[0] % 2


def spreads(rated, obs, n):
    p = float(np.mean(obs))
    noise = float(np.mean(p * (1 - p) / n))
    resid = float(np.var(obs) - noise)
    if resid <= 0.15 * float(np.var(obs)):
        return None
    return float(np.std(rated)) / float(np.sqrt(resid))


d = pd.read_parquet(_common.DD / "usopen_2026_points.parquet",
                    columns=["match_id", "event", "PointServer", "PointWinner"])
d = d[d["match_id"].astype(str).str.len() == 4]
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
n2u = dm.build_name_to_uid(df)

for tour, ev in (("atp", "1"), ("wta", "2")):
    draw = dm.load_live_draw(tour, n2u)
    field = sorted(set(draw["bracket"]))
    key = f"{tour}:2026-usopen"
    snap = stu.build_snapshots({key: {"date0": DATE0, "field_uids": field}}, df=df,
                               cache_dir=_common.SCRATCH / "snapshot_cache").get(key, {})
    hier = rate_field(snap, field).set_index("uid")
    elo = pd.read_parquet(_common.SCRATCH / "serve_return_elo_final_point.parquet")
    elo = elo[elo["tour"] == tour].set_index("uid")
    r_avg = float(elo[elo["n_matches"] >= 20]["return_elo"].mean())

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
    won, tot = {}, {}
    for mid, g in d[d["event"] == ev].groupby("match_id", sort=False):
        mid = str(mid)
        if mid not in side:
            continue
        for srv, uid in zip(("1", "2"), side[mid]):
            p = g[g["PointServer"] == srv]
            if not len(p):
                continue
            tot[uid] = tot.get(uid, 0) + len(p)
            won[uid] = won.get(uid, 0) + int((p["PointWinner"] == srv).sum())

    log(f"\n{'=' * 70}\n{tour.upper()}\n{'=' * 70}")
    log(f"{'candidate':>10} {'half':>6} {'players':>8} {'ratio raw':>10} "
        f"{'ratio shrunk':>13} {'corr raw':>9} {'corr shrunk':>12}")
    for name, getter in (("hier", lambda u: hier["serve_rating"].get(u)),
                         ("elo", lambda u: expected(elo["serve_elo"].get(u, np.nan), r_avg))):
        rows = []
        for uid, n in tot.items():
            if n < MIN_PTS:
                continue
            v = getter(uid)
            if v is None or not np.isfinite(v):
                continue
            rows.append(dict(uid=uid, rated=v, obs=won[uid] / n, n=n, h=half(uid)))
        t = pd.DataFrame(rows)
        if len(t) < 20:
            continue
        a, b = t[t["h"] == 0], t[t["h"] == 1]
        # derive the factor on half A only
        ratio_a = spreads(a["rated"].values, a["obs"].values, a["n"].values)
        if ratio_a is None:
            log(f"{name:>10} {'A':>6} {len(a):>8}  ratio not measurable on this half")
            continue
        f = 1.0 / ratio_a
        for lbl, part in (("A (fit)", a), ("B (held out)", b)):
            if len(part) < 10:
                continue
            raw = part["rated"].values
            shr = raw.mean() + f * (raw - raw.mean())
            r_raw = spreads(raw, part["obs"].values, part["n"].values)
            r_shr = spreads(shr, part["obs"].values, part["n"].values)
            c_raw = float(np.corrcoef(raw, part["obs"])[0, 1])
            c_shr = float(np.corrcoef(shr, part["obs"])[0, 1])
            fr = f"{r_raw:.2f}" if r_raw else "n/a"
            fs = f"{r_shr:.2f}" if r_shr else "n/a"
            log(f"{name:>10} {lbl:>12} {len(part):>4} {fr:>10} {fs:>13} "
                f"{c_raw:>+9.3f} {c_shr:>+12.3f}")
        log(f"{'':>10} factor {f:.3f}, derived on half A only")

log("\nIf the shrunk ratio lands near 1.00 on half B - which the factor never saw -")
log("the shrink genuinely fixes calibration rather than fitting it.")
log("SHRINK_HIER_COMPLETE")
