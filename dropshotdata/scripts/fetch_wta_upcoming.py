"""Fetch the WTA calendar and an upcoming event's entry list, ready to simulate.

WHY THIS IS WTA-ONLY
`simulate_tournament.py` already handles an unpublished draw - it generates
seeding-consistent random brackets - so a FIELD LIST is the only missing input
for forecasting an event before its draw comes out. Getting one is completely
asymmetric between tours:

  WTA  api.wtatennis.com is open and unauthenticated, and its own JS client
       documents the endpoints (ticket 32). Verified 2026-09-05: the calendar
       returns the real 2026 schedule, and /players returns a POPULATED entry
       list for events still two weeks away.
  ATP  api.atptour.com does not resolve and the site is server-rendered
       Sitecore. The US Open entrants came from usopen.org's own feeds, because
       Slams run infrastructure separate from the tour body - that route works
       for Slams and for nothing else. There is no equivalent for Beijing,
       Tokyo or Shanghai.

USE, AND THE LIMIT ON IT
Requests are spaced one second apart, carry an identifying User-Agent, and are
cached to disk so a re-run costs nothing. This is a fetch-on-demand tool, not a
poller. Ticket 32's own words: "do not let 'the endpoint is open' stand in for
'we are permitted to'." Pete has cleared building this; running it as a
sustained background poll for a public product is a separate decision that has
NOT been taken.

THE HARD PART IS THE CROSSWALK, AND IT IS THE SAME ONE AS ISSUE #3
WTA API player ids are not Sackmann ids (Sabalenka is 320760 to the API,
wta:214544 to us). Resolution is by normalised name, exactly as
bridge_tennismylife.py does it. So this script's most important output is not
the entry list - it is the COVERAGE line saying how many entrants resolved to a
player we can actually rate. An unresolved entrant is not a small gap: they
enter the simulation with no rating at all, and a forecast over a field that is
80% resolved is not 80% right, it is wrong in a way that still looks plausible.
Nothing here writes a simulatable file unless coverage clears --min-coverage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

import _common
DD = _common.DD
sys.path.insert(0, str(DD / "scripts"))

BASE = "https://api.wtatennis.com/tennis"
UA = ("dropshotverse/0.1 (non-commercial research; "
      "github.com/peteowen1/dropshotverse)")
CACHE = DD / "raw" / "wta_api"
REF = DD / "reference"
PAUSE = 1.0


def log(m: str = "") -> None:
    print(m, flush=True)


def norm(name: str) -> str:
    """Same normalisation bridge_tennismylife.py uses, so a player resolves
    identically whichever source names them."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(name))
                if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return " ".join(s.split())


def get(path: str, cache_key: str | None = None) -> dict | list:
    if cache_key:
        cf = CACHE / f"{cache_key}.json"
        if cf.exists():
            log(f"    (cached) {cache_key}")
            return json.loads(cf.read_text(encoding="utf-8"))
    req = urllib.request.Request(BASE + path, headers={
        "User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
    time.sleep(PAUSE)
    if cache_key:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{cache_key}.json").write_text(json.dumps(body), encoding="utf-8")
    return body


PAGE_SIZE = 100
MAX_PAGES = 20          # 2,000 tournaments; the tour plays nothing like that


def calendar(date_from: str, date_to: str) -> pd.DataFrame:
    """Every tournament in the window, following the endpoint's paging.

    This used to request `page=0&pageSize=100` and stop, which is one page. A
    WTA season plus the ITF events the endpoint returns runs well past 100, so
    a full-year window came back ending in early March and everything after it
    was invisible. Beijing 2026 starts on 30 September, so the event-forecast
    workflow failed with "no cached calendar row for group 1020" - after
    successfully fetching that same event's entry list, which does not go
    through here. Caught by a dry run on 2026-09-07, sixteen days before the
    forecast was due to be published.

    Paging stops on the first short or empty page, and the endpoint's own count
    is checked against what was actually collected, so stopping early on a
    short page in the middle of a sequence is caught rather than reported as a
    normal success. Each page is cached separately.

    That count is `pageInfo.numEntries`. The first version of this read
    `totalElements`, which the endpoint does not send - it returns exactly two
    top-level keys, `pageInfo` and `content` - so the guard was `None`, the
    comparison never ran, and the docstring claimed a protection that did not
    exist. A check that cannot fail is worse than no check, because it stops
    anyone looking.
    """
    rows = []
    total = None
    for page in range(MAX_PAGES):
        j = get(f"/tournaments/?from={date_from}&to={date_to}"
                f"&page={page}&pageSize={PAGE_SIZE}",
                cache_key=f"cal_{date_from}_{date_to}_p{page}")
        content = j.get("content", [])
        if total is None:
            total = (j.get("pageInfo") or {}).get("numEntries")
        for t in content:
            g = t.get("tournamentGroup") or {}
            rows.append(dict(
                group_id=g.get("id"), name=t.get("title"), start=t.get("startDate"),
                end=t.get("endDate"), level=t.get("level"), surface=t.get("surface"),
                draw=t.get("singlesDrawSize"), status=t.get("status"),
                year=t.get("year"), indoor=t.get("inOutdoor")))
        if len(content) < PAGE_SIZE:
            break
    else:
        log(f"  WARNING: stopped at the {MAX_PAGES}-page guard with a full last "
            f"page; the window may still be truncated")

    if total is None:
        raise SystemExit(
            f"calendar {date_from} to {date_to}: the response carries no "
            f"pageInfo.numEntries, so how much there was to collect cannot be "
            f"known. The endpoint's shape has changed; fix this rather than "
            f"trusting {len(rows)} rows to be all of them.")
    if len(rows) < total:
        raise SystemExit(
            f"calendar {date_from} to {date_to}: the endpoint reports "
            f"{total} tournaments and only {len(rows)} were collected. "
            f"Refusing to return a truncated calendar - a missing row here "
            f"reads downstream as 'that event does not exist'.")

    # A pre-2026-09-07 run cached the whole window under one key, holding only
    # the first page. simulate_wta_event globs this directory, so leaving that
    # file behind lets the original bug come back through a different reader.
    stale = CACHE / f"cal_{date_from}_{date_to}.json"
    if stale.exists():
        stale.unlink()
        log(f"  removed the pre-paging cache {stale.name}, which held one page")

    log(f"  {len(rows)} tournaments over {page + 1} page(s), "
        f"endpoint reports {total}")
    return pd.DataFrame(rows)


def field(group_id: int, year: int) -> pd.DataFrame:
    j = get(f"/tournaments/{group_id}/{year}/players",
            cache_key=f"players_{group_id}_{year}")
    rows = []
    for ev in (j.get("events") or []):
        # LS = ladies singles. Doubles entries share the endpoint and must not
        # be mixed into a singles field.
        if ev.get("eventTypeCode") != "LS":
            continue
        for ep in (ev.get("eventPlayers") or []):
            pl = (ep.get("players") or [{}])[0]
            rows.append(dict(
                wta_id=pl.get("id"), name=pl.get("fullName"),
                country=pl.get("countryCode") or "",
                seed=str(ep.get("seed") or "").strip(),
                entry_type=ep.get("entryType") or ""))
    return pd.DataFrame(rows)


def build_name_to_uid() -> tuple[dict[str, str], dict[str, str]]:
    """Normalised WTA name -> our uid, from the archive itself.

    Returns two maps. The second is keyed on the name with ALL SPACES REMOVED,
    used only as a fallback, and only where it is unambiguous.

    Why a second map: the archive writes Wang Xinyu as "Xin Yu Wang" while the
    API gives "XINYU WANG" - a space inside a romanised Chinese given name. That
    is a systematic risk on the China swing, not a one-off.

    Why NOT fuzzy matching, which is the obvious fix: the archive also contains
    "Xiyu Wang", a DIFFERENT player (Wang Xiyu vs Wang Xinyu). A fuzzy matcher
    would cheerfully conflate them and produce a confident forecast for the
    wrong person. Space-stripping keeps them distinct - "xinyuwang" vs
    "xiyuwang" - so this stays an exact match after normalisation rather than an
    approximate one. Any space-stripped key that maps to more than one player id
    is dropped rather than guessed.
    """
    m: dict[str, str] = {}
    squashed: dict[str, set[str]] = {}
    for year in range(2015, 2027):
        p = DD / "raw" / "github" / "wta" / f"wta_matches_{year}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False,
                        usecols=["winner_id", "winner_name", "loser_id", "loser_name"])
        for idc, namec in (("winner_id", "winner_name"), ("loser_id", "loser_name")):
            for i, n in zip(d[idc], d[namec]):
                if pd.notna(i) and pd.notna(n):
                    key = norm(n)
                    uid = f"wta:{int(i)}"
                    m.setdefault(key, uid)
                    squashed.setdefault(key.replace(" ", ""), set()).add(uid)
    unambiguous = {k: next(iter(v)) for k, v in squashed.items() if len(v) == 1}
    dropped = len(squashed) - len(unambiguous)
    if dropped:
        log(f"  {dropped} space-stripped keys map to >1 player and are dropped "
            f"rather than guessed")
    return m, unambiguous


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--to", dest="d_to",
                    default=str((pd.Timestamp.today() + pd.Timedelta(days=45)).date()))
    ap.add_argument("--min-draw", type=int, default=28)
    ap.add_argument("--fetch", type=int, default=None,
                    help="tournamentGroup id to pull the entry list for")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="refuse to write an entrants file below this share of "
                         "entrants resolved to a rateable player")
    args = ap.parse_args()

    log(f"calendar {args.d_from} -> {args.d_to}")
    cal = calendar(args.d_from, args.d_to)
    log(f"  {len(cal)} tournaments returned")
    sim = cal[(cal["draw"].fillna(0) >= args.min_draw)].sort_values("start")
    log(f"  {len(sim)} with a singles draw >= {args.min_draw}\n")
    log(sim[["group_id", "name", "start", "level", "surface", "draw", "status"]]
        .to_string(index=False, max_colwidth=38))

    if args.fetch is None:
        log("\n(pass --fetch <group_id> to pull an entry list)")
        return 0

    row = cal[cal["group_id"] == args.fetch]
    label = row["name"].iloc[0] if len(row) else str(args.fetch)
    log(f"\n=== entry list: {label} ({args.fetch}, {args.year}) ===")
    f = field(args.fetch, args.year)
    if f.empty:
        log("  no singles entrants returned - the field may not be published yet")
        return 1
    # Beijing 2026, live: 76 entryType "M" (main draw) + 42 "Q" (registered
    # for the SEPARATE qualifying tournament, not a main-draw slot at all -
    # none carry a seed, confirmed). The main-draw count (76) held steady
    # across two fetches while the Q count appeared fresh; the qualifying
    # field grows independently and must not inflate the 96-slot count this
    # simulates - it produced a downstream "118 entrants for 96 slots"
    # refusal that looked like a bug but was actually this filter missing.
    n_before = len(f)
    f = f[f["entry_type"] != "Q"].reset_index(drop=True)
    if len(f) != n_before:
        log(f"  excluded {n_before - len(f)} qualifying-tournament entrants "
            f"(entryType Q) - not a main-draw slot")
    log(f"  {len(f)} singles entrants, {(f['seed'] != '').sum()} seeded")

    log("\n  resolving to archive ids...")
    name_to_uid, squashed = build_name_to_uid()
    log(f"  name->uid map: {len(name_to_uid):,} WTA players")
    f["uid"] = f["name"].map(lambda n: name_to_uid.get(norm(n)))
    exact = int(f["uid"].notna().sum())
    # Fallback: same name with spaces removed, unambiguous only.
    miss = f["uid"].isna()
    if miss.any():
        f.loc[miss, "uid"] = f.loc[miss, "name"].map(
            lambda n: squashed.get(norm(n).replace(" ", "")))
        gained = int(f["uid"].notna().sum()) - exact
        if gained:
            log(f"  {exact} matched exactly, +{gained} via space-insensitive "
                f"fallback (e.g. archive 'Xin Yu Wang' vs API 'XINYU WANG')")
    resolved = f["uid"].notna()
    cov = float(resolved.mean())
    log(f"\n  COVERAGE: {int(resolved.sum())}/{len(f)} entrants resolved "
        f"({cov:.1%})")
    if (~resolved).any():
        log("  unresolved (would enter the simulation with no rating):")
        for n in f.loc[~resolved, "name"].head(15):
            log(f"    - {n}")

    if cov < args.min_coverage:
        log(f"\n  REFUSING to write: coverage {cov:.1%} is below "
            f"--min-coverage {args.min_coverage:.0%}. A forecast over a field "
            f"this incomplete is not partly right, it is wrong in a way that "
            f"still looks plausible.")
        return 1

    out = REF / f"wta_{args.fetch}_{args.year}_entrants.json"
    # An unresolved uid is pandas float NaN here, not None or "" - json.dumps
    # writes that as a bare, unquoted NaN token, and a reader that only checks
    # `uid or fallback` (NaN is truthy) keeps it, mixing float and str in a
    # sort and crashing. Write real JSON null instead.
    payload = [dict(section=0, slot=0, seed=r.seed, country=r.country,
                    name=r.name, uid=(r.uid if isinstance(r.uid, str) else None))
               for r in f.itertuples()]
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    log(f"\n  wrote {out} ({len(payload)} entrants)")
    log("  NOTE: `slot`/`section` are written as 0. They are NOT draw position "
        "- the US Open entrants file taught that lesson (2/128 agreement with "
        "the real bracket). The simulator generates its own seeding-consistent "
        "brackets; do not build a bracket from these fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
