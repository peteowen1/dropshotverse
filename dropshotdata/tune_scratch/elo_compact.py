"""Memory-cheap Elo replay harness.

The round 12/13 screens were killed mid-run by the OS for memory. The cause was
not the arithmetic: they held 1.23M rows as a pandas DataFrame with STRING uids
("atp:104925") and then keyed Python dicts on those strings. Millions of small
string objects plus dict overhead is where the gigabytes went.

This module does the identical computation on int32-coded players and flat numpy
arrays. Peak footprint is a few hundred MB instead of several GB, and it is also
faster, so more grid points fit in the same wall-clock.

Two things it must preserve exactly, or every number built on it is worthless:

  1. The CAUSAL ORDER of elo_baseline.prepare() - (date, round_rank) mergesort,
     not date alone. Tickets 9/25 traced real feature corruption to this.
  2. Bit-identical output to elo_baseline.py in the limiting case. This is
     asserted by tune_scratch/verify_ret_harness_identity.py, which already
     passed at max |delta| = 0.000e+00 on 89,678 rows for the SLOW harness;
     verify_compact_identity.py re-runs the same check against this one.

Files are reduced one at a time and never concatenated as DataFrames, so peak
memory is one year-file plus the growing compact arrays. The result is cached to
an .npz so later runs skip CSV parsing entirely.
"""
from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(r"C:\dev\dropshotverse\dropshotdata")
sys.path.insert(0, str(DD / "scripts"))

import elo_baseline as eb  # noqa: E402

CACHE = DD / "tune_scratch" / "elo_compact_cache.npz"
SURFACES = ("Hard", "Clay", "Grass", "Carpet", "Unknown")
RET_PAT = r"RET|Retired"
WO_PAT = r"W/O|DEF|ABD|Walkover|Def\."

KIND_OK, KIND_RET, KIND_WO = 0, 1, 2
USE_COLS = ["tourney_date", "surface", "score", "round",
            "winner_id", "loser_id"]


def log(m: str = "") -> None:
    print(m, flush=True)


def build_compact(force: bool = False) -> dict:
    """Reduce every archive CSV to int32 arrays, one file at a time."""
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}

    uid_to_code: dict[str, int] = {}
    parts: list[tuple] = []
    for tour_i, tour in enumerate(("atp", "wta")):
        for pat, lvl in ((f"{tour}_matches_[0-9]*.csv", 0),
                         (f"{tour}_matches_qual_chall_[0-9]*.csv", 1),
                         (f"{tour}_matches_qual_itf_[0-9]*.csv", 1)):
            for p in sorted((eb.ARCHIVE / tour).glob(pat)):
                try:
                    df = pd.read_csv(p, low_memory=False, usecols=USE_COLS)
                except Exception:  # noqa: BLE001
                    continue
                if not len(df):
                    continue
                date = pd.to_datetime(df["tourney_date"], format="%Y%m%d",
                                      errors="coerce")
                keep = date.notna() & df["winner_id"].notna() & df["loser_id"].notna()
                df, date = df[keep], date[keep]
                if not len(df):
                    continue

                surf = df["surface"].astype(str).str.strip().str.title()
                surf = surf.where(surf.isin(SURFACES[:4]), "Unknown")
                sc = df["score"].astype(str)
                kind = np.where(sc.str.contains(RET_PAT, case=False, na=False), KIND_RET,
                                np.where(sc.str.contains(WO_PAT, case=False, na=False)
                                         | (sc.str.strip() == ""), KIND_WO, KIND_OK))

                codes = []
                for side in ("winner", "loser"):
                    ids = df[f"{side}_id"].astype("Int64").astype(str)
                    arr = np.empty(len(df), dtype=np.int32)
                    for i, v in enumerate(ids):
                        key = f"{tour}:{v}"
                        c = uid_to_code.get(key)
                        if c is None:
                            c = len(uid_to_code)
                            uid_to_code[key] = c
                        arr[i] = c
                    codes.append(arr)

                parts.append((
                    (date.values.astype("datetime64[D]").astype(np.int32)),
                    codes[0], codes[1],
                    np.array([SURFACES.index(s) for s in surf], dtype=np.int8),
                    eb.round_rank(df["round"]).to_numpy(dtype=np.float32),
                    kind.astype(np.int8),
                    np.full(len(df), tour_i, dtype=np.int8),
                    np.full(len(df), lvl, dtype=np.int8),
                ))
                del df, date, surf, sc
                gc.collect()

    out = {name: np.concatenate([p[i] for p in parts])
           for i, name in enumerate(["day", "w", "l", "surf", "rr", "kind",
                                     "tour", "lvl"])}
    del parts
    gc.collect()

    # elo_baseline.prepare()'s exact ordering: stable sort on date, then a
    # stable sort on (date, round_rank). np.lexsort is stable and sorts by the
    # LAST key first, so keys are given as (round_rank, day).
    rr = np.nan_to_num(out["rr"], nan=np.inf)
    order = np.lexsort((rr, out["day"]))
    out = {k: v[order] for k, v in out.items()}
    out["n_players"] = np.array([len(uid_to_code)], dtype=np.int64)
    # Keep the code -> uid decode table so the identity check can join these
    # rows back to elo_baseline.py's published predictions. 44,913 short
    # strings, negligible next to the arrays.
    decode = np.empty(len(uid_to_code), dtype=object)
    for k, v in uid_to_code.items():
        decode[v] = k
    out["uids"] = decode.astype("U16")

    np.savez_compressed(CACHE, **out)
    log(f"  cached {CACHE.name}: {len(out['day']):,} rows, "
        f"{int(out['n_players'][0]):,} players, "
        f"{CACHE.stat().st_size / 1e6:.1f} MB on disk")
    return out


def run_elo_fast(d: dict, *, blend: float = 0.0, decay: float = 0.3,
                 k_base: float = 250.0, k_shape: float = 0.4, k_min: float = 16.0,
                 ret_update: float = 0.0, wo_update: float = 0.0,
                 ret_counts: bool | None = None,
                 wo_counts: bool | None = None) -> np.ndarray:
    """Sackmann-K online Elo on compact arrays.

    `ret_update` / `wo_update` scale the K applied to a RET / W-O row's rating
    delta. `ret_counts` / `wo_counts` control SEPARATELY whether such a row
    advances the match counters that drive K's decay - the axis that was
    silently bundled into screen_ret_in_elo.py's `ret_k_mult`. They default to
    (multiplier > 0), reproducing the bundled behaviour, so the old result can
    still be regenerated for comparison.
    """
    if ret_counts is None:
        ret_counts = ret_update > 0.0
    if wo_counts is None:
        wo_counts = wo_update > 0.0

    n_p = int(d["n_players"][0])
    n_s = len(SURFACES)
    ro = np.full(n_p, eb.START)
    rs = np.full((n_p, n_s), eb.START)
    n_all = np.zeros(n_p, dtype=np.int32)
    n_surf = np.zeros((n_p, n_s), dtype=np.int32)
    last_all = np.zeros(n_p, dtype=np.int32)
    last_surf = np.zeros((n_p, n_s), dtype=np.int32)
    seen = np.zeros(n_p, dtype=bool)
    seen_s = np.zeros((n_p, n_s), dtype=bool)

    day, W, L, S, K = d["day"], d["w"], d["l"], d["surf"], d["kind"]
    preds = np.full(len(day), np.nan)

    LP, LG, DR = eb.LAYOFF_PERIOD, eb.LAYOFF_GRADIENT, eb._LAYOFF_DRIFT
    START = eb.START

    def decayed(r: float, gap: int) -> float:
        if decay <= 0.0 or gap <= 0:
            return r
        logistic = 1.0 / (1.0 + math.exp((LP - gap) / LG))
        frac = decay * max(0.0, (logistic - DR) / (1.0 - DR))
        return r - frac * (r - START)

    for i in range(len(day)):
        w, l, s, dy, kd = W[i], L[i], S[i], day[i], K[i]
        ro_w = decayed(ro[w], dy - last_all[w]) if seen[w] else ro[w]
        ro_l = decayed(ro[l], dy - last_all[l]) if seen[l] else ro[l]
        rs_w = decayed(rs[w, s], dy - last_surf[w, s]) if seen_s[w, s] else rs[w, s]
        rs_l = decayed(rs[l, s], dy - last_surf[l, s]) if seen_s[l, s] else rs[l, s]

        diff = (1 - blend) * (ro_w - ro_l) + blend * (rs_w - rs_l)
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        if kd == KIND_OK:
            preds[i] = p_w

        if kd == KIND_OK:
            mult, counts = 1.0, True
        elif kd == KIND_RET:
            mult, counts = ret_update, ret_counts
        else:
            mult, counts = wo_update, wo_counts

        if mult > 0.0:
            kw = max(k_min, k_base / ((n_all[w] + 5) ** k_shape)) * mult
            kl = max(k_min, k_base / ((n_all[l] + 5) ** k_shape)) * mult
            ro[w] = ro_w + kw * (1 - p_w)
            ro[l] = ro_l - kl * (1 - p_w)
            p_s = 1.0 / (1.0 + 10 ** (-(rs_w - rs_l) / 400.0))
            ksw = max(k_min, k_base / ((n_surf[w, s] + 5) ** k_shape)) * mult
            ksl = max(k_min, k_base / ((n_surf[l, s] + 5) ** k_shape)) * mult
            rs[w, s] = rs_w + ksw * (1 - p_s)
            rs[l, s] = rs_l - ksl * (1 - p_s)
        if counts:
            n_all[w] += 1
            n_all[l] += 1
            n_surf[w, s] += 1
            n_surf[l, s] += 1
            last_all[w] = last_all[l] = dy
            last_surf[w, s] = last_surf[l, s] = dy
            seen[w] = seen[l] = True
            seen_s[w, s] = seen_s[l, s] = True
    return preds


def eval_mask(d: dict, years=(2022, 2023)) -> np.ndarray:
    yr = (d["day"].astype("datetime64[D]").astype("datetime64[Y]")
          .astype(int) + 1970)
    return np.isin(yr, years) & (d["kind"] == KIND_OK) & (d["lvl"] == 0)


def logloss(p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.log(p).mean())


if __name__ == "__main__":
    d = build_compact(force="--force" in sys.argv)
    log(f"rows={len(d['day']):,}  players={int(d['n_players'][0]):,}  "
        f"eval={int(eval_mask(d).sum()):,}")
