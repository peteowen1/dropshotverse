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

import datetime as dt
import hashlib
import re
import unicodedata
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

import _common
DD = _common.DD
# One file per season per tour: {year}.csv (ATP) and {year}_wta.csv (WTA).
# The year used to be hard-coded to 2026, which would have frozen the ratings
# at the end of 2026 with every check still green. Every season from
# FIRST_TML_SEASON to the current one is read; BRIDGE_CUTOFF still decides
# which rows count, so the older seasons only add history, never duplicates.
FIRST_TML_SEASON = 2026
TML_URL_PATTERNS = {"atp": "https://stats.tennismylife.org/data/{year}.csv",
                    "wta": "https://stats.tennismylife.org/data/{year}_wta.csv"}
BRIDGE_CUTOFF = pd.Timestamp("2026-05-25")   # archive's own last known date - only bridge matches strictly after this


def tml_seasons(today: dt.date | None = None) -> list[int]:
    """Every season the bridge reads: FIRST_TML_SEASON up to today's year."""
    today = today or dt.date.today()
    return list(range(FIRST_TML_SEASON, max(today.year, FIRST_TML_SEASON) + 1))


def tml_url(tour: str, year: int) -> str:
    return TML_URL_PATTERNS[tour].format(year=year)


def read_tml(tour: str, today: dt.date | None = None, reader=pd.read_csv) -> pd.DataFrame:
    """Every season's file for `tour`, stacked. A new season's file that is
    not published yet (404 in January) is skipped with a warning; any other
    missing season is an error, because that is the source going away."""
    today = today or dt.date.today()
    frames = []
    for year in tml_seasons(today):
        url = tml_url(tour, year)
        try:
            frames.append(reader(url, low_memory=False))
        except urllib.error.HTTPError as e:
            if e.code == 404 and year == today.year and today.month == 1:
                log(f"  WARN: {url} is not published yet (404); the new season "
                    f"has no matches in the bridge until it is")
                continue
            raise
    return pd.concat(frames, ignore_index=True)

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


# A looser match is only trusted for an archive id that has played this recently.
# The bridge exists for players active since BRIDGE_CUTOFF, so a real match for
# one of them has recent archive history; a retired homonym from 1998 does not.
LOOSE_MIN_YEAR = 2024


def _tokens(name: str) -> list[str]:
    """Accent-stripped, lower-cased words, letters only ("J.J. Wolf" -> j, j, wolf)."""
    return [t for t in re.split(r"[^a-z]+", strip_accents(str(name)).lower()) if t]


def loose_keys(name: str) -> list[str]:
    """Keys tried, strictest first, when the exact `norm` key misses.

    The archive and TML spell the same player differently and `norm` keeps
    punctuation and word order, so each pair below missed and the player got a
    synthetic id with no history (issue #45):
      letters     "Elena Gabriela Ruse" / "Elena-Gabriela Ruse", "J.J. Wolf" / "J. J. Wolf"
      sorted      "Xinyu Wang" / "Wang Xinyu" (surname first)
      first+last  "Gabriela Knutson" / "Gabriela Andrea Knutson" (middle name)
    One key namespace, not one per kind: a two-word name's first+last key is
    its letters key, and that is how "Gabriela Knutson" reaches the archive's
    three-word entry. Same three ideas as usopen_live_draw._alt_keys, which the
    draw already uses."""
    toks = _tokens(name)
    if not toks:
        return []
    keys = ["".join(toks)]
    if len(toks) >= 2:
        keys.append("".join(sorted(toks)))
        keys.append(toks[0] + toks[-1])
    return list(dict.fromkeys(keys))


def synthetic_id(name: str) -> int:
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 99_999_999
    return 900_000_000 + h


def _archive_names(tour: str) -> pd.DataFrame:
    """Every (name, id, year) the archive's tour-level, qualifying and ITF
    files give a winner or loser, in file order then row order, winner before
    loser within a row. Empty when no file exists."""
    frames = []
    for suffix in ("", "_qual_chall", "_qual_itf"):
        for year in range(1995, 2027):
            p = DD / f"raw/github/{tour}/{tour}_matches{suffix}_{year}.csv"
            if not p.exists():
                continue
            f = pd.read_csv(p, low_memory=False,
                            usecols=["winner_id", "winner_name", "loser_id", "loser_name"])
            f["year"] = year
            frames.append(f)
    if not frames:
        return pd.DataFrame({"name": [], "id": [], "year": []})
    df = pd.concat(frames, ignore_index=True)
    # interleave winner/loser per row so the order matches the row loop
    long = pd.DataFrame({
        "name": np.column_stack([df["winner_name"].values, df["loser_name"].values]).ravel(),
        "id": np.column_stack([df["winner_id"].values, df["loser_id"].values]).ravel(),
        "year": np.repeat(df["year"].values, 2),
    })
    return long[pd.notna(long["id"])]


def build_name_to_id(tour: str) -> dict[str, int]:
    """Archive's own winner_name/loser_name -> winner_id/loser_id, tour-level
    files only (matches TML's 2026.csv scope) - same source of truth
    parse_us_open_draws.py's crosswalk used."""
    # Vectorised 2026-09-06 (ticket 40): the previous iterrows() over ~930K
    # rows was 38% of a live run. Semantics kept exactly: files in the same
    # order, rows in order, winner before loser within a row, and the LAST
    # occurrence of a normalised name wins. Verified identical by
    # tune_scratch/verify_perf1_name_to_id.py.
    long = _archive_names(tour)
    name_to_id: dict[str, int] = {}
    if len(long):
        uniq = pd.Series(long["name"].unique())
        key_of = dict(zip(uniq, uniq.map(norm)))
        long = long.assign(key=long["name"].map(key_of))
        last = long.drop_duplicates(subset="key", keep="last")
        name_to_id = dict(zip(last["key"], last["id"].astype("int64").map(int)))
    for wiki_key, archive_key in MANUAL_NAME_OVERRIDES.items():
        if archive_key in name_to_id:
            name_to_id[wiki_key] = name_to_id[archive_key]
    return name_to_id


def build_loose_name_to_id(tour: str) -> dict[str, int]:
    """`loose_keys` key -> id, over archive ids seen in LOOSE_MIN_YEAR or later.
    A key that reaches more than one id is dropped, not resolved: attaching one
    real player's history to another is worse than the synthetic id this is the
    fallback for, because it looks right."""
    long = _archive_names(tour)
    long = long[long["year"] >= LOOSE_MIN_YEAR].drop_duplicates(subset=["name", "id"])
    rows = [(key, int(i)) for name, i in zip(long["name"], long["id"]) for key in loose_keys(name)]
    if not rows:
        return {}
    keyed = pd.DataFrame(rows, columns=["key", "id"]).drop_duplicates()
    unique = keyed[keyed.groupby("key")["id"].transform("nunique") == 1]
    return dict(zip(unique["key"], unique["id"]))


def build_bridge(tour: str) -> pd.DataFrame:
    seasons = tml_seasons()
    log(f"[{tour}] downloading tennismylife data for {seasons[0]}-{seasons[-1]}...")
    tml = read_tml(tour)
    tml["date"] = pd.to_datetime(tml["tourney_date"], format="%Y%m%d", errors="coerce")
    tml = tml[tml["date"] > BRIDGE_CUTOFF].copy()
    log(f"  {len(tml)} rows after {BRIDGE_CUTOFF.date()}")

    name_to_id = build_name_to_id(tour)
    loose = build_loose_name_to_id(tour)
    n_new = n_loose = 0
    loose_hits: dict[str, int] = {}
    for side in ("winner", "loser"):
        ids = []
        for name in tml[f"{side}_name"]:
            key = norm(name)
            if key in name_to_id:
                ids.append(name_to_id[key])
                continue
            hit = next((loose[k] for k in loose_keys(name) if k in loose), None)
            if hit is not None:
                ids.append(hit)
                n_loose += 1
                loose_hits[name] = hit
            else:
                ids.append(synthetic_id(key))
                n_new += 1
        tml[f"{side}_id"] = ids
    if loose_hits:
        log(f"  {n_loose} slots matched the archive by a looser name key: "
            + ", ".join(f"{n} -> {i}" for n, i in sorted(loose_hits.items())))
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
