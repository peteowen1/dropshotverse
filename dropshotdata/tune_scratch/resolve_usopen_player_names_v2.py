"""v2 of resolve_usopen_player_names.py: restrict candidates to players
ACTUALLY IN that year's US Open field, instead of the whole archive.

Found by checking why top seeds were failing to resolve (Pete asked "does it
work for the top 20"): several were genuine collisions, not typos - Casper
Ruud collides with his father Christian Ruud (ATP 1980s-90s), Tommy Paul
with an unrelated Todd Paul, Ben Shelton with his father Bryan Shelton (also
a former pro). A name-only, archive-wide match can never tell these apart.
The field for a specific year's US Open can: the retired namesakes were not
IN that draw, so restricting candidates to the ~250 real entrants (from the
archive's own match rows for that tourney_id) resolves them outright.

Also normalizes hyphens to spaces on both sides before matching (tape says
"F. Auger-Aliassime", archive says "Felix Auger Aliassime") - the same class
of variant bridge_tennismylife.py's MANUAL_NAME_OVERRIDES exists for.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common

DD = _common.DD
TOURNEY_ID = "560"   # US Open, both tours


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return strip_accents(str(s)).lower().strip().replace("-", " ")


def field_index(tour: str, year: int) -> tuple[dict, dict]:
    """(initial, surname) -> {uid}, restricted to players who appear
    ANYWHERE in that year's US Open (winner or loser, any round) - so
    qualifiers/lucky losers are included, not just the R128 seed rows."""
    raw = pd.read_csv(DD / f"raw/github/{tour}/{tour}_matches_{year}.csv", low_memory=False)
    tmatch = raw[raw["tourney_id"] == f"{year}-{TOURNEY_ID}"]
    if tmatch.empty:
        raise SystemExit(f"no {tour} US Open rows in {year} archive")
    long = pd.concat([
        tmatch[["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        tmatch[["loser_id", "loser_name"]].rename(columns={"loser_id": "id", "loser_name": "name"}),
    ]).dropna(subset=["id"])
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


def resolve_one_v2(abbrev: str, ids_by_key: dict) -> tuple[str, int | None]:
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
    names = names[names["mid"].str.len() == 4]

    for tour_ev, tour in (("1", "atp"), ("2", "wta")):
        print(f"\n=== {tour.upper()} ===")
        for year in (2022, 2023, 2024, 2025):
            ids_by_key, name_by_key = field_index(tour, year)
            sub = names[(names["year"] == year) & (names["mid"].str[0] == tour_ev)]
            all_names = pd.unique(pd.concat([sub["p1_name"], sub["p2_name"]]))
            unique = collide = missing = 0
            miss_examples, collide_examples = [], []
            for n in all_names:
                outcome, uid = resolve_one_v2(n, ids_by_key)
                if outcome == "unique":
                    unique += 1
                elif outcome == "collide":
                    collide += 1
                    collide_examples.append(n)
                else:
                    missing += 1
                    miss_examples.append(n)
            print(f"  {year}: {len(all_names)} names -> unique {unique}, collide {collide}, missing {missing}")
            if collide_examples:
                print(f"    still colliding: {collide_examples}")
            if miss_examples:
                print(f"    still missing: {miss_examples[:10]}")


if __name__ == "__main__":
    main()
