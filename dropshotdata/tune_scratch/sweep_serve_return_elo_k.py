"""Sweep the K-factor until the ratings claim as much spread as really exists.

Two problems, one cause and one dial.

The WTA ratings came out 2.30x over-dispersed on serve points where ATP was
1.15x. The obvious suspect is data volume: 101,669 of 765,582 WTA matches carry
point statistics (13.3%) against 227,956 of 426,169 for ATP (53.5%). Fewer
updates per player means noisier ratings, and a noisy rating is a wide one.
K is what governs how far a single match moves a rating, so it is the dial for
exactly this, and it was left at 1.0 without ever being swept.

The WTA hold-rate check separately came back "not measurable", which was never
about the rating - it was about testing it on ONE tournament, where 25 service
games a player leaves the observed spread indistinguishable from chance. Scored
across a whole season instead, each player has hundreds of service games and
the spread becomes measurable.

WHERE THIS IS SCORED, and why it is not the 2026 tape. The repo's rule is to
screen on the 2022-23 tune window and keep 2026 as the one clean forward test.
Choosing K on the tape would spend that. So K is chosen here, on 2022-23, and
the tape confirms it once afterwards - a number chosen on one sample and
checked on another, rather than a number chosen and reported on the same one.

Dispersion is measured with sampling noise removed from the observed side,
because leaving it in flatters every candidate and is what made a divide-by-
almost-zero read as 1775.59 earlier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import _common                                                        # noqa: E402
import hierarchical_model as hm                                       # noqa: E402
import serve_return_elo as sre                                        # noqa: E402

TUNE_YEARS = (2022, 2023)
K_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
MIN_TRIALS = 200          # service points, or games, a player needs in the window


def log(m=""):
    print(m, flush=True)


def dispersion(rated: np.ndarray, obs: np.ndarray, n: np.ndarray):
    p_bar = float(obs.mean())
    noise = float(np.mean(p_bar * (1 - p_bar) / n))
    resid = float(obs.var() - noise)
    sd_rated = float(np.std(rated))
    if resid <= 0.15 * float(obs.var()):
        return sd_rated, float(np.sqrt(max(resid, 0.0))), None
    return sd_rated, float(np.sqrt(resid)), sd_rated / float(np.sqrt(resid))


log("loading prepared archive and the raw hold columns ...")
df, _ = hm.load_prepared(use_cache=True)
holds = sre.load_hold_columns()
df = df.merge(holds, on=["tourney_date", "winner_id", "loser_id"], how="left")
log(f"  {len(df):,} rows, {df['holds_w'].notna().sum():,} with holds")

log(f"\n{'tour':>5} {'mode':>6} {'k':>5} {'rated sd':>9} {'true sd':>9} {'ratio':>7} "
    f"{'corr':>7} {'players':>8}")
best: dict[tuple[str, str], tuple[float, float]] = {}
for mode in ("point", "game"):
    for tour in ("atp", "wta"):
        sub = df[df["tour"] == tour].sort_values("date")
        for k in K_GRID:
            rows, _final = sre.run(sub, mode, k_mult=k)
            if rows.empty:
                continue
            rows["year"] = pd.to_datetime(rows["date"]).dt.year
            ev = rows[rows["year"].isin(TUNE_YEARS)]
            if ev.empty:
                continue
            # pair each PRE-match rating with what that player then did on serve
            src = sub.set_index(["date", "winner_uid", "loser_uid"])
            pred: dict[str, list] = {}
            won: dict[str, float] = {}
            tot: dict[str, float] = {}
            r_avg = float(pd.concat([ev["return_w"], ev["return_l"]]).mean())
            for r in ev.itertuples(index=False):
                try:
                    m = src.loc[(r.date, r.winner_uid, r.loser_uid)]
                except KeyError:
                    continue
                if isinstance(m, pd.DataFrame):
                    m = m.iloc[0]
                pairs = ((r.winner_uid, r.serve_w, "w"), (r.loser_uid, r.serve_l, "l"))
                for uid, s_elo, sidetag in pairs:
                    if mode == "point":
                        n_i = getattr(m, f"{sidetag}_svpt", np.nan)
                        w_i = getattr(m, f"{sidetag}_won", np.nan)
                    else:
                        n_i = m.get(f"{sidetag}_SvGms", np.nan)
                        w_i = m.get(f"holds_{sidetag}", np.nan)
                    if not np.isfinite(n_i) or n_i <= 0 or not np.isfinite(w_i):
                        continue
                    pred.setdefault(uid, []).append(sre.expected(s_elo, r_avg))
                    won[uid] = won.get(uid, 0.0) + float(w_i)
                    tot[uid] = tot.get(uid, 0.0) + float(n_i)
            keep = [u for u in pred if tot.get(u, 0) >= MIN_TRIALS]
            if len(keep) < 30:
                continue
            rated = np.array([np.mean(pred[u]) for u in keep])
            obs = np.array([won[u] / tot[u] for u in keep])
            n = np.array([tot[u] for u in keep])
            sd_r, sd_t, ratio = dispersion(rated, obs, n)
            corr = float(np.corrcoef(rated, obs)[0, 1])
            rs = f"{ratio:>7.2f}" if ratio is not None else f"{'n/a':>7}"
            log(f"{tour:>5} {mode:>6} {k:>5.2f} {sd_r:>9.4f} {sd_t:>9.4f} {rs} "
                f"{corr:>+7.3f} {len(keep):>8}")
            if ratio is not None:
                cur = best.get((tour, mode))
                if cur is None or abs(ratio - 1.0) < abs(cur[1] - 1.0):
                    best[(tour, mode)] = (k, ratio)

log("\nK closest to a ratio of 1.00, chosen on 2022-23 only:")
for (tour, mode), (k, ratio) in sorted(best.items()):
    log(f"  {tour} {mode:<6} k_mult {k:.2f}  ratio {ratio:.2f}")
log("\nConfirm these once on the 2026 tape, and do not re-tune there.")
log("SWEEP_COMPLETE")
