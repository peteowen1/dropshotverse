Type: follow-up, parked
Status: not started

## Context

Raised 2026-08-29 while fixing ticket 29's stale-ratings gap (Sackmann's
tennis_atp/tennis_wta repos went 404 on 2026-08-27 - see
[[tennis-data-supply-collapse]] memory). Built a small tactical bridge
(`scripts/bridge_tennismylife.py`) using stats.tennismylife.org to gap-fill
the ~216 players/tour active since 2026-05-25 - that's shipped and working
(see ticket 29's "Decision" section). This ticket is the bigger, deliberate
question that bridge deferred: should the CANONICAL base archive itself
change, not just get patched forward.

## Two candidate sources found, neither a slam-dunk replacement on its own

**stats.tennismylife.org** (Tennismylife/TML-Database)
- Full historical depth: ATP back to 1968, WTA back to 1990 (their own
  README flags WTA reliability as "not yet 100% guaranteed" at that depth -
  a real caveat, not dismissed).
- Live, daily-updated, bulk CSV + API (`/api/data-files`).
- Already found ONE real data-corruption bug in the small slice touched
  today (a literal `"-35.9%"` string in a WTA `l_2ndWon` stat column,
  silently turning that whole column to object dtype - fixed in the bridge
  via `pd.to_numeric(errors="coerce")`, but this is exactly the kind of
  thing a full-archive swap needs to systematically screen for first).
- Different id scheme (alphanumeric, e.g. "H0BH") than Sackmann's numeric
  ids - full swap needs a FULL historical name crosswalk (tens of
  thousands of players across 50+ years), a much bigger and riskier
  undertaking than the ~216-player recent-window crosswalk already done.
- Mutable/live-updated data - same year-file's content can change later if
  TML corrects history. Cuts both ways: self-correcting, but not
  byte-reproducible the way Sackmann's static yearly files were.

**tennisdata.app/downloads/**
- Only 2021-2026 - NOT a base-archive candidate, coverage too short.
- Has match winner + first-set-winner ODDS (usually closing line) - this
  is the actually interesting part: a much lower-friction path than
  Betfair to extend ticket 28's "chase the market" match-level validation
  into 2026 (no account signup, just a CSV).
- Different schema (`home`/`away` + `winner_code`, not `winner`/`loser`) -
  needs its own translation layer and its own player-id crosswalk, same
  category of work as the TML bridge.
- Self-disclosed ~95%+ match coverage, updated multiple times daily for
  the current season.
- Site is behind Cloudflare-style bot protection - WebFetch/curl get a 403
  even with a normal user-agent; only loaded via actual browser automation
  (mcp__claude-in-chrome). Worth knowing before assuming a script can just
  curl it - real browser automation will be needed for a bulk pull.

## Decision needed (Pete's call, not made here)

1. Does the base archive move to TML at all, or does Sackmann-frozen-base +
   TML-bridge-forward stay the standing pattern indefinitely?
2. If a full TML swap is ever attempted: pull TML's full history and diff
   it against Sackmann's archive on the overlap years FIRST, to see how
   much it actually disagrees, before trusting it at 50-year scale - not
   just "it downloaded fine."
3. Is tennisdata.app worth pulling now for the 2026 match-odds extension of
   ticket 28's market-comparison work, independent of the base-archive
   question above?

## Not done here

No pull, no crosswalk, no diff run for either source beyond what's above -
explicitly parked per Pete: "Park as a follow up on data sources - raise a
GH issue about it and we'll fix later."
