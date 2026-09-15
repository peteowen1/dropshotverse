"""D-42 follow-up, step 1 of 2: recover which real player is P1/P2 in each
2022-25 US Open point-tape match, since no draw/bracket feed with team1/team2
was captured for those years (only the draws.json category INDEX, not the
per-event match list). 2026 has this via draw_mod.load_live_draw instead and
does not need this script.

The tape's own Sentence field names a player on almost every row ("D.
Medvedev wins the point...", "S. Kozlov loses the break..."), so identity is
recoverable from the tape itself. Two things must be validated empirically,
not assumed, before trusting it (stats-discipline):

  1. Which verb implies the SENTENCE SUBJECT is this row's PointWinner, and
     which implies they are the point's LOSER. Reasoned once from tennis
     semantics ("wins"/"saves"/"hits" -> subject won the point; "loses"/
     "fails to convert"/"commits [double fault]" -> subject lost it), then
     checked: for a genuine match, every row's (subject, verb) pair should
     agree on a SINGLE name<->slot mapping. Rows that disagree are counted
     and reported per match, not silently outvoted away.
  2. Doubles contamination. A "/" in the extracted name (partner pairs, e.g.
     "A. Arystanbekova/K. Efremova") means this is a doubles point despite
     living under a singles event id - excluded, not guessed at.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common

DD = _common.DD
YEARS = [2022, 2023, 2024, 2025]

VERB_RE = re.compile(r"^(.*?)\s+(wins|loses|commits|saves|fails|hits|double faults)\b")
WINNER_VERBS = {"wins", "saves", "hits"}
LOSER_VERBS = {"loses", "fails", "commits", "double faults"}


def log(m: str = "") -> None:
    print(m, flush=True)


def name_and_verb(sentence: str) -> tuple[str, str] | tuple[None, None]:
    m = VERB_RE.match(sentence)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2)


def resolve_match(g: pd.DataFrame) -> dict:
    """One match's real points (any subset with a Sentence). Returns a dict
    with p1_name, p2_name, n_rows, n_agree, n_disagree, doubles (bool)."""
    votes: dict[str, dict[int, int]] = {}   # name -> {slot: count}
    doubles = False
    n_used = 0
    for sentence, pw in zip(g["Sentence"], g["PointWinner"]):
        name, verb = name_and_verb(sentence)
        if name is None or pw not in ("1", "2"):
            continue
        if "/" in name:
            doubles = True
            continue
        if verb in WINNER_VERBS:
            slot = int(pw)
        elif verb in LOSER_VERBS:
            slot = 3 - int(pw)
        else:
            continue
        votes.setdefault(name, {1: 0, 2: 0})[slot] += 1
        n_used += 1

    if doubles or len(votes) != 2:
        return {"p1_name": None, "p2_name": None, "n_rows": n_used,
                "n_agree": 0, "n_disagree": 0, "doubles": doubles,
                "n_distinct_names": len(votes)}

    names = list(votes.keys())
    # each name's slot = whichever slot it voted for MORE, ties broken by
    # total votes (arbitrary but rare - flagged via n_disagree either way)
    slot_of = {n: (1 if votes[n][1] >= votes[n][2] else 2) for n in names}
    if slot_of[names[0]] == slot_of[names[1]]:
        # both names majority-voted the same slot - cannot disambiguate
        return {"p1_name": None, "p2_name": None, "n_rows": n_used,
                "n_agree": 0, "n_disagree": n_used, "doubles": False,
                "n_distinct_names": 2}
    p1_name = names[0] if slot_of[names[0]] == 1 else names[1]
    p2_name = names[1] if p1_name == names[0] else names[0]
    n_agree = votes[p1_name][1] + votes[p2_name][2]
    n_disagree = votes[p1_name][2] + votes[p2_name][1]
    return {"p1_name": p1_name, "p2_name": p2_name, "n_rows": n_used,
            "n_agree": n_agree, "n_disagree": n_disagree, "doubles": False,
            "n_distinct_names": 2}


def main() -> None:
    all_rows = []
    for year in YEARS:
        d = pd.read_parquet(DD / f"usopen_{year}_points.parquet", columns=["match_id", "Sentence", "PointWinner"])
        clean = ambiguous = doubles = 0
        disagree_rates = []
        for mid, g in d.groupby("match_id", sort=False):
            r = resolve_match(g)
            r["year"] = year
            r["match_id"] = mid
            all_rows.append(r)
            if r["doubles"]:
                doubles += 1
            elif r["p1_name"] is None:
                ambiguous += 1
            else:
                clean += 1
                if r["n_rows"]:
                    disagree_rates.append(r["n_disagree"] / r["n_rows"])
        n_total = clean + ambiguous + doubles
        worst = max(disagree_rates) if disagree_rates else 0.0
        log(f"{year}: {n_total} matches -> {clean} resolved, {ambiguous} ambiguous, "
            f"{doubles} doubles-contaminated. Worst disagreement rate among resolved: {worst:.2%}")

    out = pd.DataFrame(all_rows)
    out_path = _common.SCRATCH / "usopen_player_names_by_match.parquet"
    out.to_parquet(out_path, index=False)
    log(f"\nwrote {out_path} ({len(out):,} rows)")

    resolved = out[out["p1_name"].notna()]
    log(f"\ntotal resolved: {len(resolved):,} of {len(out):,} matches across {YEARS}")
    log(f"disagreement rate distribution (resolved matches): "
        f"p50={((resolved['n_disagree']/resolved['n_rows'].clip(lower=1)).median()):.4%}  "
        f"max={((resolved['n_disagree']/resolved['n_rows'].clip(lower=1)).max()):.4%}")
    log("\nsample of resolved matches:")
    log(resolved[["year", "match_id", "p1_name", "p2_name", "n_rows", "n_disagree"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
