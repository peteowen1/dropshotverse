"""W4 arm D: arm C with its overconfidence taken out, fitted out of sample.

W4 measured arm C (the in-match recursion seeded with the model's own pre-match
skill) at log-loss 0.53188 and accuracy 0.779 against arm A's 0.57166 and 0.720
- much sharper - but ECE 0.0507 against arm A's 0.0342 and a 0.03 bar. It
failed the pre-registered bar on calibration, not on discrimination, and the
charter's answer to "recalibrate it and re-run until it passes" is no: that is
a new candidate with its own out-of-sample discipline, which is this file.

  logit(p_D) = logit(p_C) / T,  T > 1 shrinking confidence toward 0.5

WHERE THE FIT DATA COMES FROM, AND WHY IT IS WEAKER THAN THE CHARTER WANTS.
The charter fits on earlier events and scores the target once. There are no
earlier events: `usopen_2026_points.parquet` is the only point tape this
project has ever captured, so a temperature fitted on an earlier tournament is
not available at any price. The fallback is a CROSS-FIT by tour: T is fitted on
the men's matches and scored on the women's, and vice versa, so no match ever
contributes to the temperature that scores it. That is a real out-of-sample
split across different players, matches and best-of formats, and it is still
weaker than the charter's standard because both halves come from one
tournament, one surface and one fortnight. Labelled as such rather than
presented as equivalent.

A by-round split (fit early rounds, score late) runs alongside as a robustness
check, because calibration can drift as the field thins.

THE BAR IS THE SAME ONE W4 FAILED, unchanged and set before this ran:
ECE < 0.03 and log-loss < 0.55, against arm A's 0.0342 / 0.57166.

ANCHORS, fixed before looking at output
  D1  T must exceed 1. Arm C is overconfident; a fitted T below 1 would mean
      the opposite and the diagnosis was wrong.
  D2  recalibration may not destroy discrimination: accuracy must stay within
      0.005 of arm C's, since a monotone transform of the probability cannot
      change the ordering and therefore cannot change accuracy at the 0.5
      threshold except where it moves a value across 0.5.
  D3  the two cross-fit temperatures must be within 0.15 of each other. Wildly
      different values would mean the tours need separate treatment and one
      pooled number would be misleading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import _common                                                  # noqa: E402

PTS = _common.SCRATCH / "screen_inmatch_seeded_points.parquet"
OUT = _common.SCRATCH / "screen_inmatch_recalibrated.parquet"
ARM_A_ECE, ARM_A_LL, ARM_A_ACC = 0.0342, 0.57166, 0.720
GO_ECE, GO_LL = 0.03, 0.55
EPS = 1e-9


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


def temper(p, T: float):
    z = np.log(np.clip(np.asarray(p, float), EPS, 1 - EPS) /
               (1 - np.clip(np.asarray(p, float), EPS, 1 - EPS)))
    return 1.0 / (1.0 + np.exp(-z / T))


def fit_T(p, y) -> float:
    r = minimize_scalar(lambda T: logloss(temper(p, T), y), bounds=(0.2, 5.0), method="bounded")
    return float(r.x)


def report(name: str, p, y) -> dict:
    e, l = ece(p, y), logloss(p, y)
    a = float(((np.asarray(p) > 0.5).astype(int) == np.asarray(y)).mean())
    log(f"  {name:<34} ECE {e:.4f}   log-loss {l:.5f}   acc {a:.3f}")
    return dict(arm=name, ece=e, logloss=l, acc=a, n=len(p))


def main() -> None:
    if not PTS.exists():
        raise SystemExit(f"{PTS} not found - run screen_inmatch_seeded.py first")
    d = pd.read_parquet(PTS)
    d["tour"] = np.where(d["match_id"].astype(str).str[0] == "1", "atp", "wta")
    d["round_no"] = d["match_id"].astype(str).str[1].astype(int)
    log(f"points {len(d):,} across {d['match_id'].nunique():,} matches "
        f"({d.groupby('tour')['match_id'].nunique().to_dict()})")

    rows = [report("A fixed skill (ticket 37)", d["p_a"], d["y1"]),
            report("C model-seeded (W4)", d["p_c"], d["y1"])]

    # ---- cross-fit by tour: fit on one tour, score the other
    log("\n=== arm D: temperature cross-fitted by tour ===")
    p_d = pd.Series(index=d.index, dtype=float)
    temps = {}
    for tour in ("atp", "wta"):
        fit = d[d["tour"] != tour]
        T = fit_T(fit["p_c"].values, fit["y1"].values)
        temps[tour] = T
        m = d["tour"] == tour
        p_d[m] = temper(d.loc[m, "p_c"].values, T)
        log(f"  score {tour}: T = {T:.4f}, fitted on {len(fit):,} points of the other tour")
    rows.append(report("D recalibrated (cross-fit)", p_d, d["y1"]))

    log("\n  anchors")
    t_ok = all(t > 1.0 for t in temps.values())
    log(f"    D1 both temperatures above 1 (arm C is overconfident): "
        f"{ {k: round(v, 3) for k, v in temps.items()} } {'PASS' if t_ok else 'FAIL'}")
    d_acc = abs(rows[-1]["acc"] - rows[1]["acc"])
    log(f"    D2 accuracy preserved within 0.005: moved {d_acc:.4f} "
        f"{'PASS' if d_acc <= 0.005 else 'FAIL'}")
    spread = abs(temps["atp"] - temps["wta"])
    log(f"    D3 the two temperatures agree within 0.15: spread {spread:.3f} "
        f"{'PASS' if spread <= 0.15 else 'FAIL'}")

    # ---- robustness: fit on early rounds, score late rounds
    log("\n=== robustness: fit on rounds 1-3, score rounds 4-7 ===")
    early, late = d[d["round_no"] <= 3], d[d["round_no"] >= 4]
    if len(late):
        T_e = fit_T(early["p_c"].values, early["y1"].values)
        log(f"  T fitted on {len(early):,} early-round points = {T_e:.4f}")
        report("  C on late rounds", late["p_c"], late["y1"])
        report("  D on late rounds", temper(late["p_c"].values, T_e), late["y1"])

    e_d, l_d = rows[-1]["ece"], rows[-1]["logloss"]
    log(f"\n=== the same bar W4 failed, unchanged ===")
    log(f"  arm A reference: ECE {ARM_A_ECE:.4f}, log-loss {ARM_A_LL:.5f}, acc {ARM_A_ACC:.3f}")
    log(f"  bar: ECE < {GO_ECE}, log-loss < {GO_LL}")
    go = e_d < GO_ECE and l_d < GO_LL
    log(f"  VERDICT: {'GO' if go else 'NO-GO'} (ECE {e_d:.4f}, log-loss {l_d:.5f})")
    if go:
        log("  Caveat that travels with a GO: the temperature was cross-fitted within one")
        log("  tournament because no earlier point tape exists. Before this ships, it should")
        log("  be refitted on the first event that gives a second tape, and re-scored.")

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    log(f"\nwrote {OUT}")
    log("INMATCH_RECAL_COMPLETE")


if __name__ == "__main__":
    main()
