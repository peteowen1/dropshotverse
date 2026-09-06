# What happens after the US Open

Written 2026-09-05. The live pipeline has a hard stop and nothing takes over.

## The deadline

`DropshotverseUSOpenLiveSim`'s trigger ends **2026-09-14T00:00:00**. As of
2026-09-05 the tournament is at **104 of 127 matches decided** in each singles
draw (~23 left, so roughly R16), which puts the final around **2026-09-13** —
about one day of margin, which is fine for the tournament and not fine for what
comes after it.

After that trigger expires:

- `live_predictions.parquet`, `pairwise.parquet`, `ratings.parquet` and
  `predictions.parquet` stop refreshing on R2.
- The ITG wall chart keeps rendering a **completed** tournament indefinitely,
  with no "this event is finished" state.
- **Nothing alarms.** A frozen file is still a valid file: ITG's
  `validate-tennis-sim.mjs` checks coverage and per-player gaps, all of which
  keep passing on stale data. `updated_at` simply stops moving and no check
  reads it.

## What the calendar actually holds

Derived from the 2025 archive rather than assumed — 2026's shape should match.

**ATP** (excluding Davis Cup ties, which are 4-player team events the bracket
simulator cannot represent at all):

| date | event | level | draw |
|---|---|---|---|
| Sep 12-14 | Davis Cup (many ties) | D | 4 |
| Sep 17 | Chengdu, Hangzhou | A | 28 |
| Sep 19 | Laver Cup | A | 12 (team/exhibition) |
| Sep 24 | Tokyo | A | 32 |
| Sep 25 | Beijing | A | 32 |
| **Oct 1** | **Shanghai Masters** | **M** | **96** |

**WTA:**

| date | event | level | draw |
|---|---|---|---|
| Sep 8 | Guadalajara | P | 28 |
| Sep 15 | Seoul | P | 28 |
| Sep 19 | BJK Cup Finals | D | 12 (team) |
| **Sep 24** | **Beijing** | **PM** | **96** |
| Oct 6 | Wuhan | PM | 56 |

So there is a **~10-day gap after the men's final where the only ATP tennis is
team competition the engine cannot simulate**, then two 250s, then the first
big simulatable event. WTA's gap is shorter and its first big event is Beijing
on Sep 24.

## Can we forecast them? Asymmetric, and this is the crux

`simulate_tournament.py` already handles an unpublished draw — it generates
seeding-consistent random brackets — so **a field list is the only missing
input**. Getting one differs completely by tour.

**WTA — solved, today.** `api.wtatennis.com` is open and unauthenticated, and
its own JS client documents exactly the endpoints needed (ticket 32, read from
`tournament-player-service.js`, not guessed):

    /tennis/tournaments/?from=&to=&levels=      calendar, so "what is next" is a query
    /tennis/tournaments/{groupId}/{year}/players   the field
    /tennis/tournaments/{groupId}/{year}/draw      the real draw once published

That is entry list, seeds and draw. Nothing else is needed.

**ATP — not solved.** `api.atptour.com` and `api.atp.com` do not resolve, and
atptour.com is server-rendered Sitecore with no client API (ticket 32). The US
Open entrants came from scraping **usopen.org's own feeds** — Slams run their
own infrastructure separate from the tour body, so that route works for Slams
and for nothing else. There is no equivalent for Chengdu, Tokyo or Shanghai.

Fallback worth considering, but label it honestly: simulate a **projected
field** from the top N by rating or ranking. That is a legitimate product for
the window before entry lists appear, but it is a different claim from "this is
the field" and must not be presented as a draw forecast.

## Before shipping any of it

Ticket 32 records the constraint in its own words: **do not let "the endpoint is
open" stand in for "we are permitted to."** Issue #2 has Pete's decision that
WTA data may go on the ITG site, but the terms-of-use question on sustained
polling of `api.wtatennis.com` for a public product is separate and open.

## Recommendation

1. **Give the finished tournament an explicit end state** rather than letting a
   live projection freeze. One flag in the published parquet
   (`event_status = complete`) that the ITG page can render as a result rather
   than a forecast. Cheapest item here and it removes the silent-staleness
   failure entirely.
2. **Add a freshness check that actually fails.** `updated_at` older than ~2
   hours during a live event should alarm; nothing currently reads it.
3. **Build the WTA calendar + field fetcher** against the three endpoints
   above. That delivers real upcoming-tournament forecasts for one tour and is
   the largest step available without new agreements.
4. **Decide what the page shows with no simulatable event running.** The
   pipeline already publishes `ratings.parquet` for both tours; a current-form
   ratings view is the natural gap-filler and needs no new data.
5. **Leave ATP tour events alone until #3 (canonical archive) is settled.** A
   projected-field forecast built on ratings that stop at 2026-08-24 compounds
   two approximations, and the crosswalk work fixes the more important one.
