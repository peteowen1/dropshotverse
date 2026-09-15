"""D-42 follow-up, step 2 of 2: resolve the abbreviated names
extract_usopen_player_names.py recovered ("P. Carreno Busta") to the
archive's own uid, tour-scoped (event 1 = ATP, event 2 = WTA) since the
crosswalk must not let a same-named ATP/WTA pair collide.

Matches on (first initial, surname), where surname is the full archive name
minus its first token - the same shape the tape's abbreviation uses for
compound surnames ("Pablo Carreno Busta" -> "P." + "Carreno Busta"). Reports
three outcomes per name, not just a resolved count, because a name that
resolves to the WRONG player is worse than one that fails cleanly:
  unique   - exactly one archive player matches (first_initial, surname)
  collide  - 2+ archive players share it (e.g. two players with the same
             surname and first initial) - NOT guessed, left unresolved
  missing  - no archive player matches at all (accent/hyphenation mismatch,
             or a player with no ATP/WTA tour-level match on file)
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common

DD = _common.DD


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return strip_accents(str(s)).lower().strip()


def build_archive_index(tour: str) -> tuple[dict, dict]:
    """(initial, surname) -> {ids}, and (initial, surname) -> most-recent full name (for display)."""
    frames = []
    for suffix in ("", "_qual_chall", "_qual_itf"):
        for year in range(1968, 2027):
            p = DD / f"raw/github/{tour}/{tour}_matches{suffix}_{year}.csv"
            if p.exists():
                frames.append(pd.read_csv(p, low_memory=False,
                                          usecols=["winner_id", "winner_name", "loser_id", "loser_name"]))
    if not frames:
        raise SystemExit(f"no archive CSVs found for {tour}")
    df = pd.concat(frames, ignore_index=True)
    long = pd.DataFrame({
        "name": np.concatenate([df["winner_name"].values, df["loser_name"].values]),
        "id": np.concatenate([df["winner_id"].values, df["loser_id"].values]),
    }).dropna()
    long["name"] = long["name"].astype(str)
    long = long[long["name"].str.contains(" ")]
    parts = long["name"].str.split(" ", n=1, expand=True)
    long["initial"] = parts[0].map(lambda s: norm(s)[:1] if s else "")
    long["surname"] = parts[1].map(norm)
    ids_by_key: dict[tuple, set] = {}
    name_by_key: dict[tuple, str] = {}
    for row in long.itertuples(index=False):
        key = (row.initial, row.surname)
        ids_by_key.setdefault(key, set()).add(int(row.id))
        name_by_key[key] = row.name
    return ids_by_key, name_by_key


def resolve_one(abbrev: str, ids_by_key: dict) -> tuple[str, int | None]:
    """Returns (outcome, uid). abbrev is like 'P. Carreno Busta'."""
    clean = abbrev.replace(".", "").strip()
    if " " not in clean:
        return "missing", None
    initial, surname = clean.split(" ", 1)
    key = (norm(initial)[:1], norm(surname))
    ids = ids_by_key.get(key, set())
    if len(ids) == 1:
        return "unique", next(iter(ids))
    if len(ids) > 1:
        return "collide", None
    return "missing", None


def main() -> None:
    names = pd.read_parquet(_common.SCRATCH / "usopen_player_names_by_match.parquet")
    names = names[names["p1_name"].notna()].copy()
    names["mid"] = names["match_id"].astype(str)
    # 5-digit match_ids are qualifying-round entries and do NOT follow the
    # "first digit = tour" convention 4-digit main-draw ids do (confirmed:
    # ATP men under "21109"-style ids, WTA women under "12113"-style ids -
    # this is a pre-existing gap in the shared `event = match_id[0]`
    # convention build_point_table.py also uses, not something to fix here).
    # Main draw is what the shrink-factor comparison needs anyway.
    n_before = len(names)
    names = names[names["mid"].str.len() == 4]
    print(f"restricting to 4-digit main-draw match ids: {len(names):,} of {n_before:,} resolved matches")

    for tour_ev, tour in (("1", "atp"), ("2", "wta")):
        ids_by_key, name_by_key = build_archive_index(tour)
        sub = names[names["mid"].str[0] == tour_ev]
        all_names = pd.unique(pd.concat([sub["p1_name"], sub["p2_name"]]))
        outcomes = {"unique": [], "collide": [], "missing": []}
        for n in all_names:
            outcome, uid = resolve_one(n, ids_by_key)
            outcomes[outcome].append((n, uid))
        total = len(all_names)
        print(f"\n=== {tour.upper()} ({total} distinct names across {sub['year'].nunique()} years) ===")
        for k in ("unique", "collide", "missing"):
            print(f"  {k}: {len(outcomes[k])}")
        if outcomes["collide"]:
            print("  collisions:", [n for n, _ in outcomes["collide"][:10]])
        if outcomes["missing"]:
            print("  missing (sample):", [n for n, _ in outcomes["missing"][:15]])


if __name__ == "__main__":
    main()
