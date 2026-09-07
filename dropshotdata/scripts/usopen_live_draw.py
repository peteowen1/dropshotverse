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
  resolve). Coverage 2026-08-31 was reported as men 128/128, women 127/128,
  but FABLE-redteam-2026-09-07 F2 found two misses, neither genuinely new:
  the archive has "Bu Yunchaokete" (feed: "Yunchaokete Bu") and "Gabriela
  Andrea Knutson" (feed: "Gabriela Knutson"). `build_name_to_uid` now also
  indexes token-sorted and first+last keys, see `_alt_keys`.
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


def _norm_tokens(s: str | None) -> list[str]:
    """Per-word normalised tokens (same character rule as `_norm`)."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [t for t in re.split(r"[^a-z]+", s.lower()) if t]


def _alt_keys(s: str | None) -> list[str]:
    """Looser keys tried only when the exact `_norm` key misses.

    FABLE-redteam-2026-09-07 F2: the archive lists "Bu Yunchaokete" (surname
    first) and "Gabriela Andrea Knutson" (middle name) while the US Open feed
    sends "Yunchaokete Bu" and "Gabriela Knutson"; the letters-only `_norm`
    key cannot reconcile either, so both were minted as NEW_ and published at
    1500. Token-sorted key catches word-order flips; first+last catches middle
    names. Order matters: token-sorted first (stricter), then first+last."""
    toks = _norm_tokens(s)
    keys = []
    if len(toks) >= 2:
        keys.append("".join(sorted(toks)))
        if len(toks) > 2:
            keys.append(toks[0] + toks[-1])
    return keys


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
    out = dict(zip(best["key"], best["uid"]))
    # FABLE-redteam-2026-09-07 F2: secondary index (token-sorted, first+last)
    # under `_ALT`, built from the same most-recent-wins rows. Exact keys keep
    # priority because `lookup_uid` tries them first; among alt keys the most
    # recent appearance wins, same rule as the exact index. (A two-word name's
    # token-sorted key can equal its own exact key - keep it, that is how the
    # feed's flipped "Yunchaokete Bu" reaches the archive's "Bu Yunchaokete".)
    alt_rows = [(k, uid, date)
                for name, uid, date in zip(best["name"], best["uid"], best["date"])
                for k in _alt_keys(name)]
    if alt_rows:
        alt = pd.DataFrame(alt_rows, columns=["key", "uid", "date"])
        # An alt key that reaches more than one PLAYER is dropped, not resolved
        # most-recent-wins. Over the real archive 269 keys are ambiguous, e.g.
        # "ahmedomar" is both Ahmed Ben Omar and Omar Ahmed, and "adrianalopez"
        # is both Adriana Lopez and Adriana Coltorti Lopez. Picking one would
        # attribute a real player's whole rating history to a different real
        # player, silently and plausibly - strictly worse than the NEW_ mint
        # this fallback exists to avoid, because a NEW_ mint is loud and flat
        # 1500 while a wrong match looks right. Review of 05f9232, 2026-09-07.
        n_keys = alt["key"].nunique()
        uid_per_key = alt.groupby("key")["uid"].nunique()
        ambiguous = set(uid_per_key.index[uid_per_key > 1])
        if ambiguous:
            print(f"[build_name_to_uid] dropped {len(ambiguous)} of {n_keys} alt keys that reach "
                  f"more than one player (e.g. {sorted(ambiguous)[:3]})")
            alt = alt[~alt["key"].isin(ambiguous)]
        alt = alt.loc[alt.groupby("key", sort=False)["date"].idxmax()]
        out[_ALT] = dict(zip(alt["key"], alt["uid"]))
    else:
        out[_ALT] = {}
    return out


_ALT = "__alt_keys__"   # FABLE-redteam-2026-09-07 F2: sub-dict slot in name2uid


def lookup_uid(name2uid: dict, full: str) -> str | None:
    """Exact normalised key first, then the looser keys (see `_alt_keys`)."""
    uid = name2uid.get(_norm(full))
    if uid is not None:
        return uid
    alt = name2uid.get(_ALT) or {}
    for k in _alt_keys(full):
        uid = alt.get(k)
        if uid is not None:
            return uid
    return None


def _player(team: dict) -> tuple[str, str] | None:
    """(full name, us-open id) for the singles player in a team slot."""
    if not team or not team.get("firstNameA"):
        return None
    return f"{team['firstNameA']} {team['lastNameA']}", team.get("idA") or ""


def live_state(m: dict) -> dict | None:
    """The current (sets, games, server) of a match under way, or None.

    The feed carries this on every match and the repo has never read it: only
    `winner` was ever extracted, so an in-progress match looked exactly like an
    unplayed one to the simulator. Fields used, all verified against a
    completed R4 match on 2026-09-07:

      teamN.totalSetsWon  sets won so far
      teamN.serve         True for whoever serves the next point
      scores.sets         one entry per set STARTED, each [team1, team2] with
                          a `score` per side; the last entry is the set in
                          progress when there are more entries than sets won

    `p1_served_first` is derived, not given. Service alternates every game and
    carries across sets, so the player serving now also served the first game
    of the match exactly when an even number of games has been completed. A
    tiebreak counts as one game, which this parity handles correctly.

    Returns None for a match that has not started, both because there is
    nothing to condition on and because a fresh state is by construction
    identical to the pre-match probability - so the caller can treat "no live
    state" and "no change" as the same thing.
    """
    if m.get("winner") in ("1", "2"):
        return None
    t1, t2 = m.get("team1") or {}, m.get("team2") or {}
    if _player(t1) is None or _player(t2) is None:
        return None
    mid = m.get("match_id")
    sets1, sets2 = int(t1.get("totalSetsWon") or 0), int(t2.get("totalSetsWon") or 0)
    raw_sets = ((m.get("scores") or {}).get("sets")) or []
    games = [[int(side["score"]) for side in st]
             for st in raw_sets if isinstance(st, list) and len(st) == 2
             and all(isinstance(side, dict) and side.get("score") is not None for side in st)]

    # `scores.sets` must be the whole history: one entry per set decided, plus
    # at most one for a set in progress. Anything else and the game score
    # cannot be located, so this reports nothing rather than a number it
    # guessed. Defaulting a missing field to 0 here would publish "just started
    # the set" for a match that is 4-2 down it, which reads as an ordinary
    # projection and is wrong in the direction of the score it failed to see.
    if len(games) != len(raw_sets) or len(games) not in (sets1 + sets2, sets1 + sets2 + 1):
        print(f"[{mid}] WARNING: sets won {sets1}-{sets2} but scores.sets carries "
              f"{len(raw_sets)} entries of which {len(games)} are well formed; "
              f"cannot locate the game score, leaving this match pre-match")
        return None
    games1, games2 = games[-1] if len(games) > sets1 + sets2 else (0, 0)
    if sets1 + sets2 + games1 + games2 == 0:
        return None

    # Exactly one player serves. Missing or duplicated is a malformed payload,
    # not a default: `bool(t1["serve"])` would silently hand the serve to
    # player 2 whenever the field went missing on both.
    s1, s2 = bool(t1.get("serve")), bool(t2.get("serve"))
    if s1 == s2:
        print(f"[{mid}] WARNING: serve is {s1}/{s2} on the two sides, so who is "
              f"serving cannot be read; leaving this match pre-match")
        return None

    # Who served the match's first game. Service alternates every game and
    # carries across sets, so the player serving now also served first exactly
    # when an even number of games has been completed - EXCEPT inside a
    # tiebreak, where `serve` names a POINT server that rotates every one or
    # two points, so the parity flips from point to point and the answer is
    # wrong roughly half the time it is asked. None means "not known", and the
    # solver symmetrises over both orders rather than guessing.
    played = sum(sum(g) for g in games)
    in_tiebreak = games1 == 6 and games2 == 6
    return {"sets1": sets1, "sets2": sets2, "games1": games1, "games2": games2,
            "p1_serves": s1, "in_tiebreak": in_tiebreak,
            "p1_served_first": None if in_tiebreak else (s1 == (played % 2 == 0))}


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
    live : dict[str, dict]
        match_id -> the current score of a match UNDER WAY, as returned by
        `live_state`, plus the two players' uids. Empty when nothing is in
        progress. Decided matches are in `results`, never here.
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
        uid = lookup_uid(name2uid, full)   # FABLE-redteam-2026-09-07 F2
        if uid is None:
            uid = f"{tour}:NEW_{full.replace(' ', '_')}"
            if full not in unmatched:
                unmatched.append(full)
                # FABLE-redteam-2026-09-07 F2: say so every run, not only
                # under verify(verbose). A seeded NEW_ is almost certainly a
                # crosswalk miss, not a debutant.
                seeded = " SEEDED" if team.get("seed") is not None else ""
                print(f"[{code}] WARNING: no archive match for '{full}'{seeded}; "
                      f"minted {uid} (will be rated 1500 across the board)")
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
    live: dict[str, dict] = {}
    for m in matches:
        w = m.get("winner")
        if w not in ("1", "2"):
            st = live_state(m)
            if st is not None:
                u1, u2 = uid_for(m["team1"]), uid_for(m["team2"])
                if u1 is not None and u2 is not None:
                    live[m["match_id"]] = {**st, "uid1": u1, "uid2": u2}
            continue
        uid = uid_for(m["team1" if w == "1" else "team2"])
        lost = uid_for(m["team2" if w == "1" else "team1"])
        if uid is not None:
            results[m["match_id"]] = uid
        if lost is not None:
            losers.add(lost)

    return {"tour": tour, "code": code, "draw_size": draw_size,
            "bracket": bracket, "results": results, "losers": losers, "live": live,
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
