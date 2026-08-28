# 02 — Does livetennisapi.com actually deliver what it advertises?

Type: task
Status: closed — OUT OF SCOPE

## Closed 2026-08-27 — the question stopped mattering

This ticket's remaining work was to sign up for a live key and verify what the public OpenAPI spec could not: real coverage depth, data quality, rate limits and the exact terms behind signup.

**We no longer need any of it.** [Ticket 07b](07b-live-feed-decision.md) dropped the vendor entirely:
- Its **BASIC** tier was under consideration for historical data — the frozen archive ([ticket 13](13-snapshot-base-archive.md)) covers that, and [ticket 14](14-terms-model-outputs-clause.md)'s §3(c) makes accumulating a vendor feed into a training corpus questionable regardless.
- Its **ULTRA** tier ($99.99/mo) was the only route to live point-by-point — superseded by the tournaments' own free feeds ([ticket 16](16-capture-us-open-live.md)), which are richer and verified working.

**The work already done is kept, not wasted.** The `## Answer` below — established entirely from the public OpenAPI spec with no account — is what made the drop decision possible: it revealed that the advertised "$9.99 buys live point-by-point" claim was false (live is ULTRA), that the archive stops at 2022 with a separate 2023+ tape, and that only ~75% of post-2023 matches are point-complete. Without that read, the vendor would have looked like the obvious answer.

**Reopen only if** the Slam feeds close and a paid live source becomes necessary. Ticket 15 named Goalserve ($150/mo) as the broader-coverage fallback ahead of this vendor.

## Question

`livetennisapi.com` advertises a **$9.99/month Basic tier** claiming *both* point-by-point live match data *and* a historical archive back to **1968**. If those claims hold, one cheap subscription replaces the dead Sackmann historical foundation **and** supplies the live feed — collapsing the project's two hardest problems into one.

None of it is verified. The vendor is young and its upstream data rights are unclear (ATP/Challenger rights are exclusively Sportradar's 2024-2029; WTA's are Stats Perform's).

**Sign up for the free tier, hit the endpoints, and establish what is actually there.**

Specifically answer:
1. **Archive depth and breadth.** Does the historical archive really reach 1968? Which tours — ATP only, or WTA and Challenger/ITF too? Match-level only, or point-level history?
2. **Match statistics.** Does it return per-match serve/return box scores (aces, double faults, service points won, break points saved/faced)? This is the field that quietly rules out most vendors — `api-tennis.com` and SportsDataIO both return scores *without* statistics despite their marketing.
3. **Live granularity and latency.** Is live point-by-point real? How quickly do points appear? Is the WebSocket on the Ultra tier only?
4. **Coverage during a real event.** Are Challenger and ITF events actually covered, or main tour only?
5. **Rate limits** on the free and $9.99 tiers, and whether they permit polling multiple concurrent matches.
6. **Terms of service** — does it permit redistributing derived probabilities on a public non-commercial website?
7. **Client libraries.** `github.com/livetennisapi` ships MIT-licensed clients; check they are real and current.

Their claimed AI win-probability feature (Ultra, $99.99) is **not** the goal — we are building our own. It only matters as evidence they have genuine point-level data.

**Deliverable:** a verdict — does this replace the historical foundation, the live feed, both, or neither — with actual endpoint responses as evidence. If it fails, say which specific claim failed.

**Note:** signup needs Pete. Do not use his email address for a third-party signup without asking. Hand him a precise checklist if agent-driven signup isn't appropriate.

## Answer

**Established from the public OpenAPI spec (`github.com/livetennisapi/openapi`, v1.7.2, pushed 2026-08-23), the pricing page, the repo's README/CHANGELOG, and the terms page — NOT from a live account.** Line numbers below are into the downloaded spec file (`livetennis-openapi.yaml`, 3,718 lines). Tier gating per endpoint was already established before this pass and is not repeated except where new detail matters.

### 1. The 2023-today gap — there isn't one, but "coverage" and "completeness" are different claims

There is **no date-range hole**. The spec is explicit that history runs in two deliberately non-overlapping, continuous halves (lines 56-61):

> "the point-by-point tape (2023→now) covers January 2023 to now, match by match, point by point; the results archive (1968–2022) covers 1968 through 2022 ... The archive ends exactly where the tape begins, so no match is ever served from two datasets."

`/history/matches` (BASIC, line 599) confirms: "All tours, January 2023 → now." `/history/coverage` (BASIC) is a nightly-rebuilt rollup, not a marketing number — see Q4.

The real caveat is **completeness, not coverage**: the *existence* of a match record for 2023-now is universal, but the *point-by-point tape being whole* is not. The Historical Data API pricing page states "as of August 19, 2026, 75.0% of completed matches since 2023 are measured point-complete," and `/history/coverage`'s worked example (line 674-676) is blunt about how unevenly that 75% is spread: "51.1% of ITF singles matches are point-complete on the best basis against 3.5% of ITF doubles." So: no gap in *which* matches exist, but a real and tour-dependent gap in *how much of each match's point sequence* you actually get. `HistoryMatch.tape` (line 2601) and `HistoryTape.meta` (line 2341) both carry a `coverage` enum (`from_start | partial | reconstructed | reconstructed_partial | none`, defined line 2516-2539) precisely so you can filter for `points_complete=true` before backtesting rather than trust an average.

### 2. Serve statistics — obtainable, but at different tiers for different eras

- **1968-2022 archive (BASIC):** `ArchiveMatch.stats` (line 3413-3419) is a direct box score, winner/loser-keyed: `aces`, `double_faults`, `serve_points`, `first_in`, `first_won`, `second_won`, `serve_games`, `bp_saved`, `bp_faced` — exactly the svpt/1stIn/1stWon/2ndWon/SvGms/bpSaved/bpFaced fields the ticket asked about, just renamed. `null` for most pre-1991 matches (era didn't record it), never synthesised. `/history/archive/career` (`ArchiveCareer.serve`, line 3514-3534) sums the same fields per player career, with `matches_with_stats` stating coverage honestly.
- **2023-now (BASIC tape, `/history/matches/{matchId}`):** the tape itself (`HistoryTapeRow`, line 2265) is raw score states + the vendor's own `win_probability_p1`/`danger` — **no aces/DF/serve-split fields at all**. You *can* self-derive service/return points won, break points faced/saved/converted, and hold/break % from the raw point sequence yourself (that's exactly what the `MatchStatisticsSide` "derived" family does server-side, line 2699-2735) — but you cannot get aces, double faults, or the 1st/2nd serve split from BASIC at any tier for 2023+ matches. Those are "measured" (counted upstream, not derivable from points) and gated ULTRA (`/matches/{matchId}/statistics`, `MatchStatisticsMeasured`, line 2736-2885).
- **ULTRA in-play statistics** (`/matches/{matchId}/statistics`) is the only place a 2023+ match gets aces/DFs/serve-split/winners/UE as counted numbers, and even there coverage is tiered (line 2751-2767): Tier 1 (aces, DFs, points/break-points won) is present 98.6-100% of payloads on every tour including ITF; Tier 2 (1st/2nd serve split, BP saved, service games played) is present on ATP/WTA/Challenger but **absent on every ITF singles match**; Tier 3 (winners/UE by shot) historically covered ~24-47% of main-tour matches and **the upstream feed has delivered none of it since 2026-07-12** (measured 2026-08-17: 0 of 4,513 August payloads).
- **Bottom line for ticket 12's fitting strategy:** full aces/DF/serve-split box scores are available at BASIC only for the archive era (1968-2022, mostly 1991+); for 2023+ matches you need ULTRA for the same fields, and even then not for ITF (serve split/BP saved) or for winners/UE anywhere (feed stopped sending it). BASIC on its own gets you point counts and derived serve/return/break-point rates for 2023+ via your own tape reconstruction, not the counted stats.

### 3. Tour coverage, including WTA Challenger/ITF

Global coverage statement (lines 12-16): "ATP, WTA, Challenger, ITF and the junior Grand Slam draws." The `tour` filter enum everywhere is `[atp, wta, challenger, itf, juniors]` (line 1746). The 1968-2022 **archive is ATP/WTA only** (line 787-789: "The archive covers the ATP and WTA corpora only") — Challenger/ITF/juniors history only exists from 2023 forward via the tape, not in the archive.

On whether **WTA-side lower tiers** (Challenger has no women's equivalent structurally — its analogue is the ITF Women's World Tour and WTA 125s) are genuinely tracked, not just claimed:
- The ranking `system` enum includes `itf_jt` (juniors), `itf_mt` (ITF men's tour) and **`itf_wt` (ITF women's tour)** (line 2996) — a distinct system, not a shared bucket, meaning the vendor tracks ITF women's-tour rankings specifically.
- The UTR coverage numbers (lines 1152-1157) are measured, not marketing, and broken out per tour: "931 of 5,606 ITF players (16.6%), 197 of 1,903 Challenger (10.4%), 43 of 573 WTA (7.5%) and 15 of 525 ATP (2.9%)" of players active in the last 60 days — genuine per-player tracking across all four groups, women's lower tiers included.
- `/history/coverage`'s buckets (line 668-670) are `atp/wta/challenger/itf/juniors × singles/doubles`, each independently measured — so an empirical WTA/ITF completeness number exists and is queryable (see Q4), not asserted.
- Caveat: depth is real but shallow at the bottom. Per-point stream coverage is explicitly "never promised slate-wide; ITF and qualifying coverage in particular is partial" (line 345-346), and the ULTRA measured-stats Tier 2/3 fields are absent on ITF singles entirely (line 2757-2758, 2762-2763).

**This is the major finding the ticket flagged**: the vendor's schema shows genuine, separately-measured tracking of ITF and Challenger — both genders — not a main-tour-only product with marketing gloss. Depth is uneven (shallower on ITF than main tour) but the tracking is real, per the vendor's own internal completeness instrumentation, which is a much stronger signal than a claims page.

### 4. `/history/coverage`

Yes — exactly the measured completeness rollup you'd want to verify claims empirically (`CoverageRollup`, line 2540-2576). It is a **prebuilt, nightly-rebuilt artifact**, never computed at read time (`as_of`/`built_at` date it; `ledger_max_computed_at` is the newest underlying per-match measurement; a `503 coverage_unavailable` is returned honestly if the nightly build hasn't run, line 687-694). `method` (a string) states the full measurement rule in one paragraph so every number is self-defined. `buckets` is one `CoverageBucket` per `tour_draw` key (e.g. `atp_singles` ... `juniors_doubles`, plus `other` for team ties/mixed/exhibitions/unstated — counted, never dropped) with five numbers each (`CoverageBucket`, line 2577-2600): `completed` (matches in bucket), `any_tape` (≥1 row on any basis), `point_complete` (measured point-complete on the best basis — what `?points=complete` can serve), `complete_on_default_read` (point-complete without that flag — the gap between the two is where `?points=complete` is load-bearing), and `share` (`point_complete/completed`, null on an empty bucket). A bucket with zero completed matches is omitted rather than zero-filled. This genuinely lets you empirically audit coverage per tour × draw before committing to a backtest, rather than trusting the vendor's headline number.

### 5. Bulk packages (`/history/packages`, PRO)

`kind=tape` (default): monthly, JSONL = **one line per match** (a whole `HistoryTape` object, coverage meta included — not one line per point), CSV = flattened one row per point with no coverage columns (line 2452-2464). `kind=archive` (same PRO entitlement as tape): yearly exports of the 1968-2022 results archive, one `ArchiveMatch` row per line/row. `kind=rally` and `kind=elo`: yearly, **ULTRA** (or History Pro/Business), shot-by-shot rally corpus and the as-of Elo tape respectively. `kind=rankings`: ULTRA. Files carry `bytes`/`sha256` (over the exact downloaded bytes — gzip files are checksummed compressed, never re-inflated first) and yearly kinds are gzipped; the Elo package is explicitly "written in primary-key order, so a rebuild of the same data is byte-identical" (line 2472-2475) — a real reproducibility guarantee. Cadence: monthly for tape, yearly for the others, and "tape coverage is not a contiguous run of months and is still being extended backwards" (line 2449-2451) — treat the packages listing itself as the authoritative set of what exists, not an assumption of full backfill.

**Verdict: yes, this is a practical bulk-ingest path**, meaningfully better than paginated REST — pre-flattened NDJSON/CSV with checksums, one manifest call to discover what's built, no need to page through `/history/matches` + one tape call per match. It requires PRO on the live API ($29.99/mo) or the standalone Historical Data API Pro plan ($99/mo) — not available on BASIC.

### 6. Rate limits, quotas, `/usage`

Per the spec's own numbers (lines 25, 36, 42, 53-54): FREE 30/min · 100/day; BASIC 60/min · 1,000/day; PRO 300/min · 10,000/day; ULTRA 600/min · 500,000/day. `GET /usage` (FREE, any tier, line 1107-1123) reports `tier`, `base_tier` (differs from `tier` only under a temporary grant), `tier_expires_at`, `limits.per_minute`/`per_day`, `today.calls`/`errors`/`remaining_day`, and a 30-day daily history — and calls to `/usage` itself are quota-exempt. `429` responses (line 1811-1821) distinguish three cases by `error`/`scope`: per-minute (`rate_limited`, with `upgrade_url`/`tier`/`price` for the next tier up), per-day (`rate_limited` + `scope:"day"` + `resets_at`), and `abuse_throttled` (24-hour block for clients that keep hammering past the cap — "a well-behaved retry loop never sees" this). WebSocket: max 2 concurrent connections per key (line 113); the separate `/ws-token` push feed (Centrifugo protocol) exists specifically for higher fan-out without that ceiling.

### 7. Pricing (from the live pricing page and the Historical Data API page, fetched today)

**Live API:**
| Tier | Monthly | Annual |
|---|---|---|
| FREE | $0 | — |
| BASIC | $9.99/mo | $99/yr ($8.25/mo) |
| PRO | $29.99/mo | $299/yr ($24.92/mo) |
| ULTRA | $99.99/mo | $999/yr ($83.25/mo) |

**Standalone Historical Data API** (no live-API subscription needed, works with a FREE key per the spec, line 72-84):
| Tier | Price |
|---|---|
| Starter | $29/mo |
| Pro | $99/mo |
| Business | $299/mo |
| One-off 1-month pass | $49 |
| One-off 1-year pass | $399 |

The ticket's "$9.99/month Basic tier" claim is accurate and current as of today.

### 8. Terms — redistributing derived values

Fetched `livetennisapi.com/terms` today. It prohibits reselling/republishing **raw API responses**:

> "resell, redistribute, sublicense, or republish the raw API responses as a competing data feed or bulk dataset"

and separately forbids building a product that replicates the API's core function. It does **not** carve out a non-commercial exception, and does **not** explicitly address our specific case — publishing our own independently-computed win probabilities (a derived value, not a raw response) on a public non-commercial site. One clause is a yellow flag worth a careful full read rather than trusting a summarised fetch: the terms assert ownership over "the API, its data compilation, the arbitrated live feed, and **all model outputs**" to JSB Holdings — ambiguous whether "model outputs" means only *their* AI win-probability feature or could be stretched toward anything modelled from their data. This needs a full, careful human read of the actual terms document (not the AI-summarised fetch used here) before we commit to publishing our own probabilities derived from their feed.

### 9. Rally/charting provenance — this appears to be Match Charting Project data, re-served

Strong evidence this is **not** the vendor's own charting, but the public Match Charting Project (MCP) corpus:
- `/charting/matches/{chartingMatchId}` description (line 1462): "Every **Match Charting Project** stat family for one charted match, both players..."
- `/charting/players` description (lines 1414-1420): "The deepest serve/return profile we hold, **from the Match Charting Project**: serve placement (deuce/ad × wide/body/T), return depth and outcomes... COVERAGE IS CURATED — **11,646 charted matches** across both tours back to the 1960s, concentrated on the majors, NOT full-slate coverage" — that count and character (decades-deep but major-concentrated) matches the public MCP corpus's known scale and shape.
- `RallyPoint.raw` (line 3658-3662) preserves "the charter's shot string, verbatim... both serves joined by ';' when the first was a fault" — this is literally MCP's own shorthand notation being passed through, not a re-encoded proprietary format.

MCP is licensed **CC BY-NC-SA** (attribution required, non-commercial, share-alike) by its maintainer (Jeff Sackmann / contributors). **Neither the OpenAPI spec, the README, nor the terms page fetched today contains any attribution to the Match Charting Project, Jeff Sackmann, or a CC BY-NC-SA notice** — despite ULTRA being a $99.99/mo *paid, commercial* product built partly on this data. This is a real provenance/licensing exposure, but it is the vendor's exposure to resolve, not necessarily ours — it does mean that if we ever consume the rally/charting endpoints, we would be a step further removed from a CC BY-NC-SA chain whose attribution the vendor itself appears not to be honoring, which is worth flagging to Pete as a reason to be cautious about touching ULTRA's charting data at all, independent of whether we can afford it.

## Still needs a live key

- **Actual live latency**: how many seconds after a point is played does it land in `/matches/{matchId}/points` or the WebSocket — the spec documents behavior and shapes but not observed latency.
- **Real point-by-point fidelity during a live match**: whether `pbp_coverage: "point"` is actually assigned to a representative slate of matches you'd want to track, versus mostly `"game"` fallback — the spec says this varies and is partial for ITF/qualifying but gives no live numbers.
- **Whether `/history/coverage` and the Historical Data API's "75% point-complete since 2023" claim hold up against a real pull** — these are the vendor's own numbers; we haven't verified them against actual responses.
- **Client library currency**: whether `livetennisapi` (Python ≥1.4.0, JS ≥1.5.0) on PyPI/npm are real, current, and match the OpenAPI version — not checked in this pass (README confirms they exist and are MIT, but not that the published packages actually work).
- **The full, exact text of `/terms`** — the redistribution and "model outputs" ownership language above came from a summarized fetch; a load-bearing legal read needs the verbatim document, and ideally direct written confirmation from the vendor about whether publishing our own probability model derived from their raw feed is permitted on a public non-commercial site.
- **Whether signup/account creation triggers any KYC, card requirement, or ToS click-through that reveals additional restrictions** not present in the public terms page.
- **Real behavior of `/matches?status=live` for Challenger/ITF/WTA-lower-tier matches during an actual event** — the schema evidence (Q3) is strong but unverified against a live slate.
