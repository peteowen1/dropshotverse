"""Elo with TWO ratings per player: one for serving, one for returning.

`point_elo.py` deliberately did not do this, and said why: serve% and return%
"aren't complementary (both players can independently hold serve at very
different rates), which would need two separate ratings per player (a
serve/return split, hier's own approach) to update cleanly". That is right
about a SINGLE rating. It is not an argument against a paired one.

The complementary quantity is there, it is just not player-versus-player. On
any given point somebody is serving and somebody is returning, and the server
winning it IS the returner losing it. So rate the exchange:

    E = 1 / (1 + 10 ** (-(serve[server] - ret[returner]) / 400))
    A = that player's share of their own service points in this match
    serve[server] += K * (A - E)
    ret[returner] -= K * (A - E)

Zero-sum per exchange, one K-factor, the same decaying `eb.k_factor` the
match-level engine uses. Each match supplies two updates, one per server.

WHY BOTHER, given the hierarchical model already splits serve and return.
Because hier's split is badly over-dispersed - measured against the 2026 US
Open tape with sampling noise removed, its serve skills are 1.96x too spread
out on ATP and 4.76x on WTA, which is why its percentages are not publishable
(D-30). Dispersion is exactly what a K-factor controls and what a posterior
does not: K sets how far one match can move a rating, so the spread is a dial
here rather than an emergent property.

The ratings equilibrate rather than starting correct. Everyone begins at 1500,
so the first expected value is 0.5 while real serve-point win rates sit near
0.61 - serve ratings drift up and return ratings drift down until the average
gap reproduces the tour's own average. That is self-correcting, but it means
the first few hundred matches are burn-in and that serve and return ratings sit
on DIFFERENT absolute levels. Never compare a serve rating to a return rating.

Fit per tour, never pooled: the two tours' serve baselines genuinely differ
(60.9% against 57.0% on the same tape), and pooling would let one tour's
average set the other's zero (D-32).

TWO EVIDENCE MODES, because they answer slightly different questions.

  point  share of a player's own SERVICE POINTS won. More trials per match, so
         less noise per update, and it is the quantity the point recursion
         consumes.
  game   share of a player's own SERVICE GAMES HELD. Fewer trials, but it is
         the number a reader actually wants to see next to a name, and it
         weights a tight hold the same as a love hold, which is what "did you
         hold" means.

Holds are not in the prepared archive - it keeps serve points and drops the
rest - but they ARE in the raw Sackmann CSVs, and exactly:

    holds = SvGms - (bpFaced - bpSaved)

because a service game that is lost contributes exactly one break point that
was faced and not saved, however many were faced in it. Multiple break points
in one game do not double-count.

Writes serve_return_elo_ratings_{mode}.parquet: one row per match with each
player's PRE-match serve and return rating, so it can be joined as an as-of
feature exactly like elo_raw_ratings_full.parquet, plus the final ratings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common
import elo_baseline as eb
import hierarchical_model as hm

SCALE = 400.0
START = 1500.0

# K chosen by sweeping on the 2022-23 tune window ONLY, picking the value whose
# rating spread matched the real spread with sampling noise removed
# (tune_scratch/sweep_serve_return_elo_k.py). The 2026 tape was kept out of that
# choice so it can confirm it once.
#
#   tour mode   k     ratio  corr
#   atp  point  0.15  0.99   +0.645
#   atp  game   0.10  1.07   +0.621
#   wta  point  0.50  1.06   +0.713
#   wta  game   0.20  0.98   +0.797
#
# ATP wants a SMALLER K than WTA despite four times the usable matches, which is
# the right way round: more updates per player means the same K accumulates more
# movement, so a well-covered tour needs less of it. K is per tour for the same
# reason the ratings are.
#
# These are CALIBRATION values. A rating used as an XGBoost feature needs
# ordering, not calibrated spread, and ordering improves monotonically with K
# (+0.645 at 0.15 against +0.771 at 1.0 on ATP points), so a feature build
# should choose its own K rather than inherit these.
K_BY_TOUR_MODE = {("atp", "point"): 0.15, ("atp", "game"): 0.10,
                  ("wta", "point"): 0.50, ("wta", "game"): 0.20}
K_FOR_FEATURES = 1.0
RAW = _common.DD / "raw" / "github"
MODES = ("point", "game")


def out_rows(mode: str):
    return _common.SCRATCH / f"serve_return_elo_ratings_{mode}.parquet"


def out_final(mode: str):
    return _common.SCRATCH / f"serve_return_elo_final_{mode}.parquet"


def load_hold_columns() -> pd.DataFrame:
    """SvGms and break points from the raw CSVs, which the prepared archive drops.

    Merged back on (tourney_date, winner_id, loser_id) rather than recomputed,
    so the two views of a match cannot disagree about which match it is.
    """
    keep = ["tourney_date", "winner_id", "loser_id",
            "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_SvGms", "l_bpSaved", "l_bpFaced"]
    frames = []
    for f in sorted(RAW.rglob("*_matches_*.csv")):
        try:
            d = pd.read_csv(f, usecols=lambda c: c in keep, low_memory=False)
        except Exception:                                        # noqa: BLE001
            continue
        if set(keep) - set(d.columns):
            continue
        frames.append(d)
    if not frames:
        raise SystemExit(f"no raw CSVs with SvGms/bp columns under {RAW}")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    raw["tourney_date"] = pd.to_numeric(raw["tourney_date"], errors="coerce")
    for c in keep[3:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    # holds, exactly: a lost service game is one break point faced and not saved
    raw["holds_w"] = raw["w_SvGms"] - (raw["w_bpFaced"] - raw["w_bpSaved"])
    raw["holds_l"] = raw["l_SvGms"] - (raw["l_bpFaced"] - raw["l_bpSaved"])
    # Two different reasons a row is unusable, reported separately. Only 18.9%
    # of the archive records serve stats at all - the rest is old, ITF and
    # challenger matches with none - and lumping that in with bad arithmetic
    # would tell the next reader the holds formula is suspect when it is not.
    have = raw[["w_SvGms", "w_bpSaved", "w_bpFaced",
                "l_SvGms", "l_bpSaved", "l_bpFaced"]].notna().all(axis=1)
    ok = (have
          & raw["holds_w"].between(0, raw["w_SvGms"])
          & raw["holds_l"].between(0, raw["l_SvGms"])
          & (raw["w_SvGms"] > 0) & (raw["l_SvGms"] > 0))
    impossible = int((have & ~ok).sum())
    log(f"  {len(raw):,} raw rows; {int(have.sum()):,} ({have.mean():.1%}) record "
        f"serve stats at all")
    log(f"  of those, {int(ok.sum()):,} ({ok.sum() / max(int(have.sum()), 1):.2%}) give "
        f"holds inside [0, SvGms]; {impossible:,} are impossible and dropped")
    return raw.loc[ok, ["tourney_date", "winner_id", "loser_id",
                        "holds_w", "w_SvGms", "holds_l", "l_SvGms"]]


def evidence(mode: str, row) -> tuple[float, float] | None:
    """(winner's share on own serve, loser's share on own serve), or None."""
    if mode == "point":
        if not getattr(row, "has_points", False) or row.w_svpt <= 0 or row.l_svpt <= 0:
            return None
        return row.w_won / row.w_svpt, row.l_won / row.l_svpt
    if mode == "game":
        gw, gl = getattr(row, "w_SvGms", None), getattr(row, "l_SvGms", None)
        if gw is None or gl is None or pd.isna(gw) or pd.isna(gl) or gw <= 0 or gl <= 0:
            return None
        return row.holds_w / gw, row.holds_l / gl
    raise ValueError(mode)


def log(m: str = "") -> None:
    print(m, flush=True)


def expected(serve_rating: float, return_rating: float) -> float:
    """Share of service points the server is expected to win."""
    return 1.0 / (1.0 + 10 ** (-(serve_rating - return_rating) / SCALE))


def start_gap(df: pd.DataFrame, mode: str) -> float:
    """The serve-minus-return gap that reproduces this tour's average.

    Without it every player starts at an expected 0.500 while real serve-point
    win rates sit near 0.61, so EVERY rating drifts upward on serve and
    downward on return for as long as that player keeps playing. The drift is
    shared, so it cancels out of a head-to-head prediction - but it does not
    cancel out of a PUBLISHED rating, and it made serve rating correlate +0.758
    with match count. A journeyman with 500 matches outranked a better newcomer
    on serve for having been around longer.

    Starting at the tour's own average removes the thing there is to drift
    towards, so a rating moves on merit from the first match.
    """
    vals = [v for row in df.itertuples(index=False)
            if (ev := evidence(mode, row)) is not None for v in ev]
    if not vals:
        return 0.0
    p_bar = float(np.mean(vals))
    p_bar = min(max(p_bar, 1e-3), 1 - 1e-3)
    return SCALE * np.log10(p_bar / (1 - p_bar))


def run(df: pd.DataFrame, mode: str = "point", k_mult: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Walk the archive once, updating both ratings per served exchange.

    `k_mult` scales the K-factor. It is the dispersion dial: a smaller K moves
    ratings less per match and produces a tighter spread. Left at 1.0 here and
    swept by the screen, because picking it by eye is how you get a rating that
    looks confident and is not.
    """
    gap = start_gap(df, mode)
    s0, r0 = START + gap / 2.0, START - gap / 2.0
    log(f"    starting serve {s0:.1f} / return {r0:.1f} "
        f"(gap {gap:.1f}, the tour's own average) so nothing drifts")
    serve: dict[str, float] = {}
    ret: dict[str, float] = {}
    n_serve: dict[str, int] = {}
    rows = []
    used = skipped = 0

    for row in df.itertuples(index=False):
        w, l = row.winner_uid, row.loser_uid
        ev = evidence(mode, row)
        if ev is None:
            skipped += 1
            continue
        act_w, act_l = ev
        sw = serve.get(w, s0); rw = ret.get(w, r0)
        sl = serve.get(l, s0); rl = ret.get(l, r0)

        # recorded BEFORE any update, so a join on this row is strictly as-of
        rows.append({"date": row.date, "tour": row.tour,
                     "winner_uid": w, "loser_uid": l,
                     "serve_w": sw, "return_w": rw, "serve_l": sl, "return_l": rl})

        exp_w = expected(sw, rl)
        exp_l = expected(sl, rw)
        k_w = eb.k_factor(n_serve.get(w, 0)) * k_mult
        k_l = eb.k_factor(n_serve.get(l, 0)) * k_mult

        # each exchange is zero-sum: what the server gains, the returner loses
        d_w = k_w * (act_w - exp_w)
        d_l = k_l * (act_l - exp_l)
        serve[w] = sw + d_w
        ret[l] = rl - d_w
        serve[l] = sl + d_l
        ret[w] = rw - d_l
        n_serve[w] = n_serve.get(w, 0) + 1
        n_serve[l] = n_serve.get(l, 0) + 1
        used += 1

    log(f"    {used:,} matches scored, {skipped:,} skipped for no {mode} data "
        f"({used / max(used + skipped, 1):.1%} usable)")
    final = {"serve": serve, "return": ret, "n": n_serve}
    return pd.DataFrame(rows), final


def main() -> None:
    log("loading prepared archive...")
    df, _ = hm.load_prepared(use_cache=True)
    log(f"  {len(df):,} rows")

    log("loading SvGms and break points from the raw CSVs...")
    holds = load_hold_columns()
    before = len(df)
    df = df.merge(holds, on=["tourney_date", "winner_id", "loser_id"], how="left")
    log(f"  {before:,} prepared rows, {df['holds_w'].notna().sum():,} "
        f"({df['holds_w'].notna().mean():.1%}) matched a raw row with holds")

    for mode in MODES:
        log(f"\n{'#' * 60}\n# {mode}-level evidence\n{'#' * 60}")
        run_mode_all(df, mode)


def run_mode_all(df: pd.DataFrame, mode: str) -> None:
    all_rows, finals = [], []
    for tour in ("atp", "wta"):
        sub = df[df["tour"] == tour].sort_values("date")
        log(f"\n=== {tour} ({len(sub):,} matches), k_mult {K_BY_TOUR_MODE[(tour, mode)]} ===")
        k = K_BY_TOUR_MODE[(tour, mode)]
        rows, final = run(sub, mode, k_mult=k)
        all_rows.append(rows)
        t = pd.DataFrame({"uid": list(final["serve"]),
                          "serve_elo": [final["serve"][u] for u in final["serve"]],
                          "return_elo": [final["return"].get(u, np.nan) for u in final["serve"]],
                          "n_matches": [final["n"].get(u, 0) for u in final["serve"]]})
        t["tour"] = tour
        est = t[t["n_matches"] >= 20]
        log(f"    {len(t):,} players, {len(est):,} with 20+ matches")
        log(f"    serve elo  median {est['serve_elo'].median():.0f}, "
            f"sd {est['serve_elo'].std():.0f}")
        log(f"    return elo median {est['return_elo'].median():.0f}, "
            f"sd {est['return_elo'].std():.0f}")
        unit = "serve-point win rate" if mode == "point" else "hold rate"
        log(f"    implied {unit} vs an average returner: "
            f"{expected(est['serve_elo'].median(), est['return_elo'].median()):.3f}")
        log(f"    corr(serve, return) = "
            f"{est['serve_elo'].corr(est['return_elo']):+.3f}")
        for label, col in (("strongest serves", "serve_elo"), ("strongest returns", "return_elo")):
            top = est.nlargest(5, col)
            log(f"    {label}: " + ", ".join(f"{u.split(':')[-1]} {v:.0f}"
                                             for u, v in zip(top["uid"], top[col])))
        finals.append(t)

    pd.concat(all_rows, ignore_index=True).to_parquet(out_rows(mode), index=False)
    pd.concat(finals, ignore_index=True).to_parquet(out_final(mode), index=False)
    log(f"\n  wrote {out_rows(mode).name} and {out_final(mode).name}")


if __name__ == "__main__":
    main()
    log("\nSERVE_RETURN_ELO_COMPLETE")
