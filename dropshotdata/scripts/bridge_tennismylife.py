"""Fixes the stale-ratings gap Pete flagged: the archive's own source
(Sackmann's tennis_atp/tennis_wta) went 404 on 2026-08-27, freezing our
data at 2026-05-25. stats.tennismylife.org is a live, daily-updated ATP+WTA
replacement (schema checked directly against known tournaments - matches
Sackmann's column names and round/surface vocabulary almost exactly; the one
real difference, tourney_level using "250"/"500" instead of Sackmann's "A",
turns out not to matter under our frozen hyperparameters: k_mode='sackmann'
never reads tourney_level for K-factor, and DSV_TIER_WEIGHT=1.0 makes hier's
elite/non-elite split numerically identical either way).

Scoped-down crosswalk (not a full 50-year migration): only players active
since 2026-05-25 need mapping (~216/tour), since the archive already covers
everything before that. Read-only wrt elo_baseline.py/hierarchical_model.py -
this only builds a BRIDGE dataframe with their exact expected schema and
feeds it in via pd.concat before their own prepare()/load_prepared(), never
edits their code.

New players (genuinely absent from the archive under any name checked) get
a synthetic large numeric id (9e8 + stable hash) - a real Sackmann id in
this archive is 5-6 digits, so there's no realistic collision risk.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import _common
DD = _common.DD
TML_URLS = {"atp": "https://stats.tennismylife.org/data/2026.csv",
           "wta": "https://stats.tennismylife.org/data/2026_wta.csv"}
BRIDGE_CUTOFF = pd.Timestamp("2026-05-25")   # archive's own last known date - only bridge matches strictly after this

# Hand-verified overrides for real name-order/hyphenation mismatches (same
# discipline as parse_us_open_draws.py's - checked against the archive
# directly, not guessed). Keys are TML spellings (post accent-strip/lower).
MANUAL_NAME_OVERRIDES = {
    "felix auger-aliassime": "felix auger aliassime",
    "jan-lennard struff": "jan lennard struff",
    "j j wolf": "j j wolf",
}


def log(m: str) -> None:
    print(m, flush=True)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(name: str) -> str:
    return strip_accents(str(name)).lower().strip()


def synthetic_id(name: str) -> int:
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 99_999_999
    return 900_000_000 + h


def build_name_to_id(tour: str) -> dict[str, int]:
    """Archive's own winner_name/loser_name -> winner_id/loser_id, tour-level
    files only (matches TML's 2026.csv scope) - same source of truth
    parse_us_open_draws.py's crosswalk used."""
    # Vectorised 2026-09-06 (ticket 40): the previous iterrows() over ~930K
    # rows was 38% of a live run. Semantics kept exactly: files in the same
    # order, rows in order, winner before loser within a row, and the LAST
    # occurrence of a normalised name wins. Verified identical by
    # tune_scratch/verify_perf1_name_to_id.py.
    frames = []
    for suffix in ("", "_qual_chall", "_qual_itf"):
        for year in range(1995, 2027):
            p = DD / f"raw/github/{tour}/{tour}_matches{suffix}_{year}.csv"
            if not p.exists():
                continue
            frames.append(pd.read_csv(p, low_memory=False,
                                      usecols=["winner_id", "winner_name", "loser_id", "loser_name"]))
    name_to_id: dict[str, int] = {}
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # interleave winner/loser per row so the order matches the row loop
        long = pd.DataFrame({
            "name": np.column_stack([df["winner_name"].values, df["loser_name"].values]).ravel(),
            "id": np.column_stack([df["winner_id"].values, df["loser_id"].values]).ravel(),
        })
        long = long[pd.notna(long["id"])]
        uniq = pd.Series(long["name"].unique())
        key_of = dict(zip(uniq, uniq.map(norm)))
        long["key"] = long["name"].map(key_of)
        last = long.drop_duplicates(subset="key", keep="last")
        name_to_id = dict(zip(last["key"], last["id"].astype("int64").map(int)))
    for wiki_key, archive_key in MANUAL_NAME_OVERRIDES.items():
        if archive_key in name_to_id:
            name_to_id[wiki_key] = name_to_id[archive_key]
    return name_to_id


def build_bridge(tour: str) -> pd.DataFrame:
    log(f"[{tour}] downloading tennismylife 2026 data...")
    tml = pd.read_csv(TML_URLS[tour], low_memory=False)
    tml["date"] = pd.to_datetime(tml["tourney_date"], format="%Y%m%d", errors="coerce")
    tml = tml[tml["date"] > BRIDGE_CUTOFF].copy()
    log(f"  {len(tml)} rows after {BRIDGE_CUTOFF.date()}")

    name_to_id = build_name_to_id(tour)
    n_new = 0
    for side in ("winner", "loser"):
        ids = []
        for name in tml[f"{side}_name"]:
            key = norm(name)
            if key in name_to_id:
                ids.append(name_to_id[key])
            else:
                ids.append(synthetic_id(key))
                n_new += 1
        tml[f"{side}_id"] = ids
    log(f"  {n_new} winner/loser slots resolved to a synthetic new-player id "
        f"(genuinely absent from the archive under any spelling checked)")

    # Match hierarchical_model.py's load_matches() column set exactly (the
    # stricter of the two schemas - elo_baseline.py's is a subset of these).
    cols = ["tourney_date", "tourney_level", "surface", "score", "best_of", "round",
           "winner_id", "loser_id", "winner_name", "loser_name",
           "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    out = tml[cols].copy()
    # TML has at least one corrupted stat cell in the wild (a literal
    # "-35.9%" string found in l_2ndWon during this build, which silently
    # turns pandas' WHOLE column to object dtype and crashes hierarchical_
    # model.py's arithmetic downstream) - coerce numeric stat columns
    # explicitly so one bad cell becomes a clean NaN (falls back to regime B
    # for that match, the correct/safe treatment) instead of poisoning the
    # column's dtype for every row.
    stat_cols = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    for c in stat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["tour"] = tour
    out["level_file"] = "tour"   # TML's 2026.csv is the tour-level file, same scope as *_matches_2026.csv
    return out


def load_matches_with_bridge(loader_module) -> pd.DataFrame:
    """loader_module is elo_baseline or hierarchical_model - calls its OWN
    unmodified load_matches(), then appends the tennismylife bridge for both
    tours. Callers pass the result straight into that same module's
    prepare()/load_prepared(), unchanged."""
    archive = loader_module.load_matches()
    bridges = [build_bridge(tour) for tour in ("atp", "wta")]
    combined = pd.concat([archive, *bridges], ignore_index=True)
    log(f"combined: {len(archive):,} archive rows + {sum(len(b) for b in bridges):,} bridge rows "
        f"= {len(combined):,}")
    return combined
