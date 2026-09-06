# Ticket 39: TML's depth settles issue #3 — keep Sackmann as base

Probed 2026-09-06, following ticket 38's recommendation to answer the cheap
gating question before building a crosswalk. Took about twenty minutes of HEAD
requests. Issue #3 had been open since 2026-08-31 with a "50-year crosswalk"
queued in front of it.

## What TML actually carries

| tour | earliest | latest |
|---|---|---|
| ATP | **1968** (404 at 1965 and 1960) | 2026 |
| WTA | **1990** (404 for every year 1968-1989) | 2026 |

File sizes are comparable between tours where both exist (ATP 310-780 KB, WTA
272-590 KB per season), and 2020 is visibly smaller on both - the COVID season,
which is a good sign the data is real rather than padded.

ATP starting exactly at 1968 is the Open era boundary, so that is a deliberate
scope rather than an accident of hosting.

No manifest or index is served (`/manifest.json`, `/index.json`, `/files.json`,
`/list.json` all 404; `/` returns 308), so coverage has to be probed per file.

## Two corrections to the record

**Ticket 01's "TennisMyLife's manifest has 49 ATP challenger files and zero
WTA" is wrong**, and has been shaping scope since 2026-08-27. TML serves WTA for
every year from 1990. `bridge_tennismylife.py` has in fact been pulling
`2026_wta.csv` successfully all along and it carries real WTA rows - the claim
was contradicted by this project's own running code.

**But the real limitation is different from the one that was feared.** Not "no
WTA" - "no WTA before 1990".

## Which settles the migration decision

The Sackmann archive holds **61,840 tour-level WTA matches from 1968-1989**.
Moving to TML wholesale discards all of them: 22 seasons of women's tennis.

Whether that matters depends entirely on which product is asked about, and the
answer differs:

- **Current-form ratings** - nearly harmless. Elo converges; 1970s results carry
  almost no weight in a 2026 rating.
- **All-time GOAT list** - disqualifying. `map.md` names this as one of the
  engine's two ranking outputs. Losing 1968-1989 gives a women's all-time
  ranking that starts mid-Navratilova and excludes Margaret Court, Billie Jean
  King and most of Chris Evert. There is no way to present that honestly.

## Decision

**Keep the frozen Sackmann archive as the historical base and TML as the live
bridge** - which is what is already running, and is option A in issue #3. TML
cannot be canonical for WTA without losing a fifth of the women's record.

Consequences worth stating:

1. **The 50-year crosswalk is not needed** for this. The bridge maps only
   current players into an existing id space, which ticket 38 measured as
   effectively solved (31 ambiguous names in 15,515 tour-level players, 11
   genuinely overlapping, none of which a crosswalk could fix anyway).
2. **The Sackmann archive is now permanently frozen** (repos 404 since
   2026-08-27), so the historical base will never extend past 2026-06-02. Every
   future season arrives through the bridge. That is a real long-term
   dependency on one volunteer-run site and should be recorded as such.
3. **ATP-only migration remains available** if the historical depth question is
   ever ATP-specific - TML matches Sackmann's 1968 start there exactly. Not
   recommended: running two different base archives per tour would make every
   cross-tour comparison suspect.

Issue #3 can be closed as decided rather than left open pending a crosswalk.

## Reproduce

`scratchpad/probe_tml.py`, `scratchpad/probe_tml2.py` - HEAD requests only, one
per second.
