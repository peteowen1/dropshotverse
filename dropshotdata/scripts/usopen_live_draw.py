"""Load the REAL, live-updating US Open draw: bracket topology + completed results.

Why this exists
---------------
`simulate_us_open_2026.py` produces a *locked* pre-tournament forecast. It calls
`build_random_bracket()`, which invents a fresh seeding-consistent bracket for
every simulation, because the real draw was not wired in. That structurally
cannot absorb a result: it has no notion that Navone and Djokovic were ever
scheduled to meet. This module supplies the missing input.

Source
------
The tournament's own public draw feed, no auth (same family as the IBM
point-by-point feeds ticket 07b settled on):

    https://www.usopen.org/en_US/scores/feeds/2026/draws/{MS,WS}.json

It carries all 127 matches of a 128 draw AND updates `status`/`winner` live, so
one file is both the bracket and the results tape.

Verified facts this module relies on (checked 2026-08-31, not assumed)
---------------------------------------------------------------------
* `match_id` is `f"1{round}{n:02d}"` - R1 `1101..1164`, R2 `1201..1232`,
  R3 `1301..1316`, R4 `1401..1408`, QF `1501..1504`, SF `1601..1602`, F `1701`.
* Bracket topology: round `r` match `n` feeds round `r+1` match `(n+1)//2`.
  Verified empirically on 8 completed R1 matches by locating each winner in the
  next round - 8/8 agreed, including the Navone/Djokovic anchor (1148 -> 1224).
* The `slot` field in the older `usopen_*_entrants.json` scratch files is NOT
  the draw position - it is a seed-ordered list (2/128 slot-by-slot agreement
  with the feed). Do not use it for bracket order. This module reads bracket
  order from the feed only.
* Player identity: feed carries US Open tour ids (`atpz355`), the model uses
  Sackmann uids (`atp:100644`). These are joined on normalised name against the
  *bridged* archive (archive + tennismylife bridge, so post-freeze players
  resolve). Coverage 2026-08-31: men 128/128, women 127/128 (Gabriela Knutson
  is genuinely new and gets a synthetic uid).
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

FEED_URL = "https://www.usopen.org/en_US/scores/feeds/2026/draws/{code}.json"
EVENT_CODE = {"atp": "MS", "wta": "WS"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# round index (1-based) -> stage label used by the simulator
ROUND_STAGE = {1: "R128", 2: "R64", 3: "R32", 4: "R16", 5: "QF", 6: "SF", 7: "F"}
N_ROUNDS = 7


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def match_id(round_no: int, n: int, event: str = "1") -> str:
    """Feed's match id for round `round_no` (1-based), match `n` (1-based).

    The leading digit is the EVENT, not a constant: men's singles is `1`
    (1101..1164), women's singles is `2` (2101..2164). Always pass the prefix
    derived from the feed rather than assuming - getting this wrong raises
    loudly here, but a silently mismatched bracket would not."""
    return f"{event}{round_no}{n:02d}"


def parse_match_id(mid: str) -> tuple[int, int]:
    """-> (round_no, match_no). Event prefix is dropped."""
    return int(mid[1]), int(mid[2:])


def event_prefix(matches: list[dict]) -> str:
    """The leading event digit actually used by this feed."""
    prefixes = {m["match_id"][0] for m in matches if m.get("match_id")}
    if len(prefixes) != 1:
        raise ValueError(f"expected one event prefix in feed, got {sorted(prefixes)}")
    return prefixes.pop()


def quarters(bracket: list[str]) -> dict[str, str]:
    """uid -> quarter letter A/B/C/D by true draw position.

    A 128 draw splits into four 32-slot quarters, so quarter = slot // 32.
    Only meaningful because `bracket` is in REAL draw order (from the feed) -
    it would be noise against a randomised bracket. Players in the same
    quarter can meet no later than the semi-final, which is what makes this
    the useful lens on why a probability moved: a rival falling in your own
    quarter helps you far more than one falling across the draw."""
    n = len(bracket)
    per = n // 4
    return {uid: "ABCD"[min(i // per, 3)] for i, uid in enumerate(bracket)}


def fetch_draw(code: str, cache_path: str | Path | None = None,
               use_cache: bool = False) -> dict:
    """Download one draw feed. `use_cache` re-reads `cache_path` instead of
    hitting the network - use it for reproducible offline runs, never for a
    live refresh (the whole point of the feed is that it moves)."""
    if use_cache and cache_path and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(FEED_URL.format(code=code), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    if cache_path:
        Path(cache_path).write_text(json.dumps(data), encoding="utf-8")
    return data


def build_name_to_uid(df) -> dict[str, str]:
    """Normalised-name -> uid from a prepared+bridged archive frame.

    On a name collision the most RECENT appearance wins, which is the right tie
    break here: the draw is current players, and stale homonyms from decades ago
    would otherwise capture the name."""
    # Vectorised 2026-09-06 (ticket 40): the row loop normalised ~2.4M names
    # one character at a time and was 12% of a live run. Same semantics:
    # winner column first then loser column, rows in order, most recent date
    # wins, and on an equal date the FIRST occurrence wins (the old strict >).
    # groupby().idxmax() returns the first row holding the group maximum in
    # original order, which is exactly that rule. Verified identical by
    # tune_scratch/verify_perf_live_path.py.
    parts = []
    for col_u, col_n in (("winner_uid", "winner_name"), ("loser_uid", "loser_name")):
        sub = df[[col_u, col_n, "date"]].dropna()
        parts.append(pd.DataFrame({"uid": sub[col_u].values, "name": sub[col_n].values,
                                   "date": sub["date"].values}))
    long = pd.concat(parts, ignore_index=True)
    if not len(long):
        return {}
    uniq = pd.Series(long["name"].unique())
    long["key"] = long["name"].map(dict(zip(uniq, uniq.map(_norm))))
    best = long.loc[long.groupby("key", sort=False)["date"].idxmax()]
    return dict(zip(best["key"], best["uid"]))


def _player(team: dict) -> tuple[str, str] | None:
    """(full name, us-open id) for the singles player in a team slot."""
    if not team or not team.get("firstNameA"):
        return None
    return f"{team['firstNameA']} {team['lastNameA']}", team.get("idA") or ""


def load_live_draw(tour: str, name2uid: dict[str, str], *,
                   cache_path: str | Path | None = None,
                   use_cache: bool = False) -> dict:
    """The real bracket plus every result so far.

    Returns
    -------
    bracket : list[str]
        128 uids in true R1 slot order (R1 match 1101 team1, 1101 team2,
        1102 team1, ...). Index i faces index i^1 in round 1.
    results : dict[str, str]
        match_id -> winning uid, for matches the feed reports as decided.
    losers : set[str]
        Every uid that has lost a decided match, i.e. is out of the
        tournament. Read straight off the feed rather than inferred from a
        stage probability - a player knocked out in R2 still has
        P(reach R64) == 1.0, so probabilities cannot answer "are they out".
    names, seed_map : dict[str, str] / dict[str, float]
    unmatched : list[str]
        Feed players that got a synthetic uid because the archive has never
        seen them. Reported, never silently dropped.
    """
    code = EVENT_CODE[tour]
    data = fetch_draw(code, cache_path=cache_path, use_cache=use_cache)
    matches = data["matches"]
    by_id = {m["match_id"]: m for m in matches}
    draw_size = int(data["drawSize"])   # feed sends this as a string
    ev = event_prefix(matches)          # 1 = men's singles, 2 = women's

    names: dict[str, str] = {}
    seed_map: dict[str, float] = {}
    unmatched: list[str] = []

    def uid_for(team: dict) -> str | None:
        p = _player(team)
        if p is None:
            return None
        full, _usid = p
        uid = name2uid.get(_norm(full))
        if uid is None:
            uid = f"{tour}:NEW_{full.replace(' ', '_')}"
            if full not in unmatched:
                unmatched.append(full)
        names[uid] = full
        if team.get("seed") is not None:
            seed_map[uid] = float(team["seed"])
        return uid

    # --- bracket: R1 in match order, team1 then team2 -----------------------
    bracket: list[str] = []
    for n in range(1, (draw_size // 2) + 1):
        m = by_id.get(match_id(1, n, ev))
        if m is None:
            raise ValueError(f"{code}: missing R1 match {match_id(1, n, ev)} in feed")
        for side in ("team1", "team2"):
            uid = uid_for(m[side])
            if uid is None:
                raise ValueError(f"{code}: empty player slot in {m['match_id']} {side}")
            bracket.append(uid)

    # --- results: every decided match, any round ----------------------------
    # Both sides are resolved through the SAME uid_for path, so a player can
    # never end up neither a winner nor a loser because of a uid-resolution
    # quirk on one side only. `losers` is the authoritative elimination set:
    # who is out is a fact the feed reports, not something to infer from a
    # simulated probability being zero.
    results: dict[str, str] = {}
    losers: set[str] = set()
    for m in matches:
        w = m.get("winner")
        if w not in ("1", "2"):
            continue
        uid = uid_for(m["team1" if w == "1" else "team2"])
        lost = uid_for(m["team2" if w == "1" else "team1"])
        if uid is not None:
            results[m["match_id"]] = uid
        if lost is not None:
            losers.add(lost)

    return {"tour": tour, "code": code, "draw_size": draw_size,
            "bracket": bracket, "results": results, "losers": losers,
            "names": names, "seed_map": seed_map, "unmatched": unmatched,
            "event": ev, "raw": data}


def verify(draw: dict, verbose: bool = True) -> list[str]:
    """Structural checks. Returns a list of problems (empty == clean).

    These are anchors, not decoration: a silently wrong bracket produces
    perfectly plausible-looking probabilities, which is exactly the failure
    mode this project has hit repeatedly."""
    problems: list[str] = []
    n = draw["draw_size"]
    bracket = draw["bracket"]

    if len(bracket) != n:
        problems.append(f"bracket has {len(bracket)} slots, expected {n}")
    if len(set(bracket)) != len(bracket):
        dupes = [u for u in set(bracket) if bracket.count(u) > 1]
        problems.append(f"duplicate uids in bracket: {dupes[:5]}")

    # every winner must actually be one of the two players in that match
    by_id = {m["match_id"]: m for m in draw["raw"]["matches"]}
    for mid, uid in draw["results"].items():
        m = by_id[mid]
        sides = {draw["names"].get(uid)}
        actual = {(_player(m["team1"]) or ("", ""))[0], (_player(m["team2"]) or ("", ""))[0]}
        if not (sides & actual):
            problems.append(f"{mid}: winner {uid} not among that match's players {actual}")

    # topology: a round-r winner must appear in round r+1 match (n+1)//2
    for mid, uid in draw["results"].items():
        r, k = parse_match_id(mid)
        if r >= N_ROUNDS:
            continue
        nxt = by_id.get(match_id(r + 1, (k + 1) // 2, draw.get("event", "1")))
        if nxt is None:
            continue
        nxt_names = {(_player(nxt["team1"]) or ("", ""))[0], (_player(nxt["team2"]) or ("", ""))[0]}
        nm = draw["names"].get(uid)
        # the next match may not be populated yet - only flag a real contradiction
        if any(nxt_names) and nm and nm not in nxt_names:
            problems.append(f"{mid}: winner {nm} absent from next match "
                            f"{nxt['match_id']} {nxt_names}")

    if verbose:
        print(f"[{draw['code']}] bracket={len(bracket)} results={len(draw['results'])} "
              f"unmatched={len(draw['unmatched'])}")
        if draw["unmatched"]:
            print(f"  synthetic uids (never seen in archive): {draw['unmatched']}")
        print(f"  verify: {'CLEAN' if not problems else str(len(problems)) + ' PROBLEM(S)'}")
        for p in problems[:10]:
            print(f"    - {p}")
    return problems


if __name__ == "__main__":
    import hierarchical_model as hm
    import bridge_tennismylife as bridge

    print("loading bridged archive for the name->uid crosswalk...")
    df, _ = hm.prepare(bridge.load_matches_with_bridge(hm))
    n2u = build_name_to_uid(df)
    print(f"  {len(n2u):,} distinct names\n")

    for tour in ("atp", "wta"):
        d = load_live_draw(tour, n2u)
        verify(d)
        # named anchor: the tournament's biggest upset so far must round-trip
        for mid, uid in d["results"].items():
            m = {mm["match_id"]: mm for mm in d["raw"]["matches"]}[mid]
            losing = m["team2" if m["winner"] == "1" else "team1"]
            lp = _player(losing)
            if lp and lp[0] in ("Novak Djokovic",):
                print(f"  ANCHOR {mid}: {d['names'][uid]} beat {lp[0]} "
                      f"(seed {losing.get('seed')})")
        print()
