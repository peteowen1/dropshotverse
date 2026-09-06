# Ticket 38: scoping the player-id crosswalk — it is not the bottleneck

Scoped 2026-09-06, before building anything. Three things were said to block on
it: issue #3's canonical archive, ATP entry lists, and seeding the in-match WP
model (ticket 37). Issue #3 calls the "50-year player-id crosswalk" **the big
piece**.

Measured, it is not. For any of the three.

## The numbers

Sackmann archive, all levels, 1877-2026:

| | |
|---|---|
| appearances | 3,549,746 |
| distinct players (all levels) | 65,622 |
| **distinct players, tour level only** | **15,515** |

Tour level is what matters - challenger/ITF/futures players never appear in an
entry list, a draw feed or a rating that reaches the site.

## Name ambiguity is 0.2%, and concentrated where nothing reads it

Of 15,515 tour-level players, **31 normalised names are ambiguous**. How they
separate:

| separable by | count |
|---|---|
| era (non-overlapping careers) | 17 |
| tour (one ATP, one WTA) | 3 |
| **genuinely overlapping** | **11** |

Only **8** collisions involve any player active since 2000, and only **4** have
both sides modern. Those four look like duplicate ids inside Sackmann's own
data rather than two people:

    martin landaluce         atp:211776 (2022)      vs atp:212021 (2023-2026)
    gunawan trismuwantara    atp:209870 (2020-21)   vs atp:211326 (2023-25)
    johanne christine svendsen  wta:241717 (2025)   vs wta:260672 (2022-24)
    tiantsoa sarah rakotomanga rajaonah  wta:239456 vs wta:266531 (both 2025)

Adjacent, non-overlapping year ranges under one name is the signature of a
re-issued id, not a namesake. That is a source data-quality question, and a
crosswalk cannot fix it - it would have to choose one of two ids for the same
person either way.

The reverse problem (one player, several spellings) affects **89 of 15,515**
(0.42%): "jan eric lundquist"/"jan erik lundquist", "billy higgins"/"william
higgins". Also small, also mostly historical.

## So what is actually blocking each of the three?

**1. Seeding the in-match WP model - effectively unblocked.** It needs the ~500
players in one tournament's draws, not 50 years. The WTA fetcher already
resolves 100% of entrants across four events (Beijing 76/76, Seoul 36/36,
Singapore 32/32, Guadalajara 33/33) with one space-insensitive fallback. The
missing input for ticket 37 is a pre-match serve estimate per player, which is a
MODELLING task, not an identity task.

**2. ATP entry lists - never was a crosswalk problem.** The blocker is that
`api.atptour.com` does not resolve and atptour.com is server-rendered Sitecore,
so there is no entry list to map ids TO. A perfect crosswalk changes nothing
here. This needs a source, or the projected-field fallback.

**3. Canonical archive, issue #3 - blocked on something else it also lists.**
That issue names four prerequisites: overlap-years diff, WTA-depth check,
corruption screening, and the crosswalk. The first three are questions about
whether TML's data MATCHES Sackmann's where they overlap and is deep enough to
replace it. Those are data-comparison questions and they are the real work. The
crosswalk is the smallest of the four.

## Recommendation

**Re-scope issue #3 rather than start the crosswalk.** The 50-year crosswalk as
described is roughly a day of careful work over 15,515 players with 11 real
ambiguities, not the multi-week piece the issue implies - and it should be done
*after* the depth/quality comparison, because if TML cannot match Sackmann's
history the crosswalk has nothing to map onto.

Order that follows from the measurement:

1. **Overlap-years diff.** Take the years both sources cover, join on
   name+date, and count matches present in one and missing in the other. Cheap,
   and it answers "is TML actually equivalent" before anything is built on it.
2. **WTA depth check.** Ticket 01 recorded TML's manifest as 49 ATP challenger
   files and **zero WTA**. If that still holds, TML cannot be canonical for WTA
   at all and the whole migration is ATP-only - which changes the decision
   completely and is one manifest fetch to confirm.
3. **Crosswalk**, mechanically, with era and tour as tie-breakers and the 11
   overlapping cases resolved by hand.

Doing 3 first is the trap: it is the concrete-feeling task, and it is the one
whose value depends entirely on the answers to 1 and 2.

## Reproduce

`scratchpad/scope_crosswalk.py`, `scratchpad/scope2.py` (throwaway; the numbers
above are the deliverable, not the scripts).
