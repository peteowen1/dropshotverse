Type: finding, parked
Status: endpoints confirmed to exist; access NOT established

## Context

Raised 2026-08-31. Pete noted pannaverse pulls football data from "the
analyst" and asked whether the same route serves tennis. It does not, yet -
but the investigation produced enough concrete detail to be worth recording
so nobody repeats it.

Companion to [ticket 30](30-data-source-evaluation.md) (base-archive
replacement) and [ticket 32](32-wta-public-api-live-scores.md) (the WTA API,
which is the actually-useful outcome of the same day's work).

## What pannaverse's Opta access actually is

Not a paid enterprise contract. `pannadata/scripts/opta/opta_scraper.py`
hits `https://api.performfeeds.com/soccerdata` with
`Referer`/`Origin: https://theanalyst.com`, a rotating pool of 12 browser
user-agents and randomised headers "to help prevent rate limiting"
(`_set_random_headers`, line 517). Auth is a hardcoded
`PROVIDER_ID = "1mjq6w6ezkxe611ykkj8rgz7f1"`, labelled in-code as an
"Opta/Stats Perform **public outlet token**", overridable via
`OPTA_OUTLET_ID` but falling back to that default so it works with no
credentials at all.

Confirmed live: loading `theanalyst.com/opta-football-match-centre` in a real
browser shows that exact token in flight, e.g.

```
soccerdata/match/{token}?tmcl={seasonId}&live=yes&_pgSz=380
    &mt.mDt=[2026-08-27T00:00:00Z TO 2026-09-04T00:00:00Z]
    &_rt=c&_fmt=jsonp&_clbk={cb}
soccerdata/matchlivewinprobability/{token}/{matchId}?_rt=c&_fmt=jsonp&_clbk={cb}
```

Note `_fmt=jsonp` (not json) - that is how the browser client dodges CORS.
`matchlivewinprobability` is a live win-probability endpoint, i.e. exactly the
shape dropshotverse would want if tennis were reachable.

pannaverse's own `docs/reviews/FABLE-REVIEW.md:150` records a 2026-06-15
security review flagging the hardcoded token fallback as an accepted-known
risk, **not** as licensed access.

## The tennis product exists

`https://api.performfeeds.com/tennisdata/...` is real - error responses carry
`xmlns="http://www.performgroup.com/schema/tennisdata"`.

Five endpoints respond (i.e. return a parameter error rather than
unknown-endpoint): **`match`, `tournament`, `rankings`, `players`,
`matchstats`**.

Confirmed NOT to exist on the tennis product: `tournamentcalendar`,
`tournamentcalendars`, `calendar`, `tournaments`, `competition`,
`competitions`, `seasons`, `season`, `schedule`, `tournamentschedule`,
`draw`, `drawsheet`, `fixtures`, `results`, `live`, `livescores`, `scores`,
`entrants`, `playerrankings`, `rankingtypes`, `headtohead`, `h2h`, `venues`,
`tour`, `competitionandseason`.

Observed error-code vocabulary:

| code | meaning |
|---|---|
| 10001 | unknown endpoint for this product |
| 10010 | forbidden / rate-limited (hit once) |
| 10200 | (seen on `players` with no params) |
| 10201 | unknown parameter name |
| 10202 | parameter recognised, value invalid (only seen on `rankings?type=`) |
| 10203 | missing required parameter |

## Why it is blocked

**Soccer's parameter vocabulary does not carry over.** `tmcl`, `fx`, `prsn`,
`ctst`, `tour`, `rankingType` all return 10201 (unknown parameter) on
`tennisdata` - verified with a correctly-shaped 25-character Opta ID, so this
is name rejection, not malformed-value rejection.

**There is no discovery endpoint.** Soccer enumerates seasons via
`tournamentcalendar`; tennis has no equivalent (see the exhaustive 404 list
above). So even with the right parameter name, a valid opaque 25-char
tournament ID would still be needed and there is no way to obtain one from
the API itself.

**No client app leaks tennis IDs.** theanalyst.com's page-sitemap lists every
tool page on the site - `opta-games`, `opta-quiz`, `opta-points-hub`,
`nfl-predictions`, `six-nations-stats`, and exactly one match centre:
**football**. There is no tennis match centre or tennis interactive page. The
tennis section is articles only, statically rendered, making zero API calls
(verified on `theanalyst.com/competition/tennis/articles` and on the US Open
2026 predictions article). So theanalyst.com has the `tennisdata` product
available and simply does not consume it.

**No public schema.** Stats Perform publish no endpoint/parameter
documentation; their answer is "contact sales" (already established in
ticket 15: no self-serve tier, unpublished pricing).

## Important caveat - entitlement was never tested

An earlier note in this session claimed the public outlet token "passes
entitlement" for tennis. **That was overstated and is retracted.** Every
tennis response obtained was a *schema* error (unknown endpoint / missing
param / unknown param), and schema validation runs before authentication and
entitlement checks. No request was ever well-formed enough to reach an
entitlement decision.

So the honest position is three unknowns, not one: no ID source, no parameter
schema, and **untested access**. The token may well have no tennis licence at
all.

## Why probing stopped

Parameter-name enumeration against a third-party API using a borrowed token
is fuzzing. It was stopped deliberately, for two reasons beyond the low odds:

1. **It risks pannaverse's working pipeline.** That token feeds 15 leagues of
   live football. One `403/10010` was already triggered. Getting it
   rate-limited or revoked breaks a running system to chase a speculative one.
2. **Tennis is the worst sport to test it on.** Stats Perform hold the
   exclusive WTA live rights to 2030 and are the party that litigates
   redistribution. The mechanism is identical to football's; the exposure is
   not.

## If picked up later

The only credible unlock is obtaining a valid tennis tournament/season ID
from outside the API - e.g. if Stats Perform ever ship a tennis match centre
or tennis widget on theanalyst.com, its network traffic would hand over both
the parameter names and a real ID in one go, exactly as the football match
centre does today. Re-check the page-sitemap for a tennis tool page before
spending any further effort.

Do not resume blind parameter enumeration.
