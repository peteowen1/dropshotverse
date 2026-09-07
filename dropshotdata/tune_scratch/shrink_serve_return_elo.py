"""Fit K for ordering, then recover calibration with a shrink that costs nothing.

Pete's observation, which reframed this whole exercise: a dispersion ratio of
1.00 is nearly free - any rating can be scaled to it - so choosing K to hit it
spends a parameter on something a post-hoc rescale gives away. Ordering is the
part that cannot be faked.

The arithmetic behind that is a standard identity: correlation is invariant
under a positive linear transform, corr(a + b*x, y) == corr(x, y) for b > 0.
Shrinking each rating's deviation from the tour mean by a constant factor is
exactly such a transform, so it can move the spread to wherever we want while
leaving the ordering untouched.

This DEMONSTRATES that on the actual ratings rather than asserting it, because
an identity that is true in general can still be got wrong in code - the
shrink has to be applied in the same units the ratio was measured in, and it
is easy to shrink the Elo points instead of the probabilities and quietly
change both numbers.

Recipe, per tour and mode:
  1. fit at the K that maximised correlation on the 2022-23 window
  2. measure the dispersion ratio there
  3. shrink the PROBABILITIES by 1/ratio
  4. check correlation is unchanged and the ratio is now 1.00
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
MIN_TRIALS = 200
# chosen by CORRELATION on the 2022-23 window (sweep_serve_return_elo_k.py)
K_BY_CORR = {("atp", "point"): 4.0, ("atp", "game"): 4.0,
             ("wta", "point"): 4.0, ("wta", "game"): 3.0}


def log(m=""):
    print(m, flush=True)


def spreads(rated, obs, n):
    p_bar = float(np.mean(obs))
    noise = float(np.mean(p_bar * (1 - p_bar) / n))
    resid = float(np.var(obs) - noise)
    if resid <= 0.15 * float(np.var(obs)):
        return float(np.std(rated)), float(np.sqrt(max(resid, 0.0))), None
    return float(np.std(rated)), float(np.sqrt(resid)), float(np.std(rated)) / float(np.sqrt(resid))


log("loading archive and hold columns ...")
df, _ = hm.load_prepared(use_cache=True)
df = df.merge(sre.load_hold_columns(), on=["tourney_date", "winner_id", "loser_id"], how="left")

log(f"\n{'tour':>5} {'mode':>6} {'k':>5} {'stage':>10} {'ratio':>7} {'corr':>8} {'players':>8}")
shrinks = {}
for (tour, mode), k in sorted(K_BY_CORR.items()):
    sub = df[df["tour"] == tour].sort_values("date")
    rows, _f = sre.run(sub, mode, k_mult=k)
    rows["year"] = pd.to_datetime(rows["date"]).dt.year
    ev = rows[rows["year"].isin(TUNE_YEARS)]
    src = sub.set_index(["date", "winner_uid", "loser_uid"])
    r_avg = float(pd.concat([ev["return_w"], ev["return_l"]]).mean())
    pred, won, tot = {}, {}, {}
    for r in ev.itertuples(index=False):
        try:
            m = src.loc[(r.date, r.winner_uid, r.loser_uid)]
        except KeyError:
            continue
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        for uid, s_elo, tag in ((r.winner_uid, r.serve_w, "w"), (r.loser_uid, r.serve_l, "l")):
            if mode == "point":
                n_i, w_i = getattr(m, f"{tag}_svpt", np.nan), getattr(m, f"{tag}_won", np.nan)
            else:
                n_i, w_i = m.get(f"{tag}_SvGms", np.nan), m.get(f"holds_{tag}", np.nan)
            if not np.isfinite(n_i) or n_i <= 0 or not np.isfinite(w_i):
                continue
            pred.setdefault(uid, []).append(sre.expected(s_elo, r_avg))
            won[uid] = won.get(uid, 0.0) + float(w_i)
            tot[uid] = tot.get(uid, 0.0) + float(n_i)
    keep = [u for u in pred if tot.get(u, 0) >= MIN_TRIALS]
    rated = np.array([np.mean(pred[u]) for u in keep])
    obs = np.array([won[u] / tot[u] for u in keep])
    n = np.array([tot[u] for u in keep])

    _sr, _st, ratio = spreads(rated, obs, n)
    corr = float(np.corrcoef(rated, obs)[0, 1])
    log(f"{tour:>5} {mode:>6} {k:>5.1f} {'raw':>10} "
        f"{ratio if ratio else float('nan'):>7.2f} {corr:>+8.3f} {len(keep):>8}")

    if ratio is None:
        log(f"{'':>5} {'':>6} {'':>5} {'shrink':>10} {'n/a':>7} - ratio not measurable")
        continue
    # shrink the PROBABILITIES toward their own mean, which is the space the
    # ratio was measured in. Shrinking the Elo points instead would move
    # through the sigmoid and change the ordering slightly.
    f = 1.0 / ratio
    shrunk = rated.mean() + f * (rated - rated.mean())
    _sr2, _st2, ratio2 = spreads(shrunk, obs, n)
    corr2 = float(np.corrcoef(shrunk, obs)[0, 1])
    shrinks[(tour, mode)] = f
    log(f"{'':>5} {'':>6} {'':>5} {'shrunk':>10} {ratio2:>7.2f} {corr2:>+8.3f} "
        f"  factor {f:.3f}")
    assert abs(corr2 - corr) < 1e-9, "the shrink moved the correlation - it must not"
    assert abs(ratio2 - 1.0) < 0.02, f"the shrink did not land on 1.00 ({ratio2:.3f})"

log("\nBoth assertions held for every candidate: correlation identical to 1e-9,")
log("dispersion driven to 1.00. So K buys ordering and the shrink buys")
log("calibration, and they do not trade against each other.")
log("\nshrink factors to apply at serve time:")
for (tour, mode), f in sorted(shrinks.items()):
    log(f"  {tour} {mode:<6} k={K_BY_CORR[(tour, mode)]:.1f}  shrink {f:.4f}")
log("SHRINK_COMPLETE")
