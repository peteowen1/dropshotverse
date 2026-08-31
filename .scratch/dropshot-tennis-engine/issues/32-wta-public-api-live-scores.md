Type: finding
Status: confirmed open API; live in-play field population NOT yet observed

## Finding

**`api.wtatennis.com` is an open, unauthenticated, CORS-enabled REST API
carrying WTA live scoring.** Discovered 2026-08-31 by loading
`wtatennis.com/scores` in a real browser and reading its network traffic -
the same technique that found the IBM Slam feeds in ticket 07b.

**This overturns ticket 03's core WTA conclusion.** Ticket 03 recorded WTA
live as structurally unobtainable ("`liveScoringId` empty; rights are Stats
Perform's to 2030") and ticket 07b built Slam-only scope partly around that
finding. `liveScoringId` is in fact **populated** (`"1039"` Monterrey,
`"905"` US Open). The rights position is unchanged and real - what was wrong
is the claim that no public technical route exists.

No token, no key, no scraping of rendered HTML. It is WTA's own JSON API,
which their site's JS client calls directly.

## What it covers (verified 2026-08-31)

`GET /tennis/matches/global` returned 84 matches across three tournaments
simultaneously:

| tournament | EventID | level |
|---|---|---|
| US OPEN | 905 | Grand Slam |
| MONTERREY | 1039 | WTA 500 |
| PHILADELPHIA 125 | 1166 | WTA 125 |

**WTA 125 coverage is the significant part** - ticket 01 recorded that no
source anywhere covers WTA below tour level live ("TennisMyLife's manifest
has 49 ATP challenger files and zero WTA"). This one does.

## Per-match fields

Directly satisfies ticket 07b's `PointEvent` seam:

- `Serve` - which player is serving
- `PointA` / `PointB` - current game points
- `ScoreSet1A`..`ScoreSet5B`, `ScoreTbSet1`..`ScoreTbSet4` - sets + tiebreaks
- `MatchState`, `LastUpdated`, `MatchTimeStamp`, `MatchTimeTotal`
- `PlayerIDA`/`PlayerIDB` (WTA ids), country, seed, `Venue`, `RoundID`
- `GrandSlam` flag, `ResultString`, `ScoreString`, `Winner`
- `BinPacketBase64` - unexamined, may carry richer per-point detail

## MatchState enum (verbatim from WTA's own `bundle-es.js`)

```
UPCOMING:"U"  ON_COURT:"C"  WARMUP:"W"  PLAYING:"P"  SUSPENDED:"S"
TOILET_BREAK:"D"  MEDICAL_TIMEOUT:"M"  CHALLENGE_IN_PROG:"R"
CORRECTION_MODE:"E"  FINISHED:"F"  CANCELLED:"L"  RE_WARMUP:"w"
RESTARTED:"p"  TIME_ANNOUNCED:"T"  PLAYERS_ARRIVED:"g"
```

Client poll intervals in the same bundle: 10s / 15s / 60s.

## Wider endpoint surface

Read from `tournament-player-service.js` (their own API client), so these are
documented-by-source, not guessed:

```
/tennis/tournaments/?from=&to=&levels=&excludeLevels=
/tennis/tournaments/{groupId}/{year}
/tennis/tournaments/{groupId}/{year}/draw
/tennis/tournaments/{groupId}/{year}/oop          # order of play
/tennis/tournaments/{groupId}/{year}/matches
/tennis/tournaments/{groupId}/{year}/matches/{matchId}/{type}
/tennis/tournaments/{groupId}/{year}/players
/tennis/tournaments/{groupId}/{year}/standings
/tennis/tournaments/{groupId}/winners
/tennis/matches/global                            # cross-tournament scoreboard
/tennis/players/{id}/ranking | performance | detailed | matches | records
/tennis/players/{id}/headtohead/{id2}
/tennis/stats/{year}/{metric}
```

## NOT YET CONFIRMED - the one open question

At discovery time **no WTA match was in progress** (84 matches, all `F` or
`U`), so `Serve` / `PointA` / `PointB` were empty in every record. Their live
population is **inferred from the schema and the state enum, never observed.**

A verification poller (`scratchpad/verify_wta_live.py`) was left running
2026-08-31 ~11:15 UTC, polling every 3 minutes until it sees a live-state
match with those fields filled. It writes `wta_live_CONFIRMED.json` on
success and `wta_live_INCONCLUSIVE.json` on timeout - **the timeout case is
a real possible outcome, not a formality**: the fields may only populate in a
narrower state, or may not be served publicly at all.

Do not treat this as a working live WTA feed until that observation lands.

## Granularity caveat

This is a **current-state snapshot polled every 10-60s**, not a point-by-point
tape like the IBM Slam feeds. That is exactly ticket 07b's stated granularity
floor ("game-level, not point-level... roughly 80% of the value"), so it
qualifies - but it is a genuinely lesser artifact than what the Slams give,
and the live model must consume it through the same `PointEvent` seam rather
than assuming a per-point tape.

## Legal / ToS position - unresolved, Pete's call

Ticket 03 quoted WTA's ToS banning automated harvesting: *"'harvest' (or
collect) information from the WTA Sites using an automated software tool"*
and separately barring redistribution of site material. **That text has not
changed and this finding does not clear it.**

What has changed is only the technical picture. Using a site's own public
JSON API is the same category as the IBM Slam feeds already relied on
(ticket 07b), but this is WTA's own property and WTA is the tour whose live
rights are exclusively licensed to Stats Perform through 2030. The rights
holder in that arrangement actively litigates redistribution
(Sportradar v LSports, ticket 03).

**Recommendation: fine for private model input and research; get an explicit
decision from Pete before anything derived from it is published on the ITG
site.** Do not let "the endpoint is open" stand in for "we are permitted to
republish."

## ATP: checked the same way, still nothing

`atptour.com/en/scores/current` was loaded in the same browser session: 49
network requests, **zero data/API calls** - all ads, analytics, fonts, cookie
consent, partner logos. The `/-/media/` paths indicate Sitecore server-side
rendering, i.e. scores are baked into the HTML with no client API to call.
`api.atptour.com` and `api.atp.com` both fail to connect (no DNS/refused).

So the ATP-below-Slam gap ticket 07b identified is **unchanged**. Coverage
today: both draws at the Slams (IBM), WTA at every level (this ticket),
ATP tour events not at all.

## Also corrected while here

`glad94/infotennis` was described in my own earlier session summary as
extending live coverage to ATP tour events. **That was wrong** - it is
post-match only (Tournament Calendar, Results, Key Stats, Rally Analysis,
Stroke Analysis, Court Vision, "matches from Antwerp 2021 onwards"), exactly
as ticket 03 line 56 had already recorded. No live capability.
