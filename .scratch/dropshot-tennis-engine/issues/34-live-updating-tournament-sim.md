Type: build
Status: shipped, running

## What was built (2026-08-31)

Live-updating US Open tournament projections. Ticket 29's simulator produced a
*locked* pre-tournament forecast that could not move; this makes it re-simulate
from the current state as results land, publishing to R2 every 30 minutes.

## Why the locked forecast could not absorb results

`simulate_us_open_2026.py` calls `build_random_bracket()`, which invents a fresh
*seeding-consistent but fictional* draw for every simulation, and starts all 128
players at R128. It averages over thousands of hypothetical draws. It therefore
has no notion that two specific players were ever scheduled to meet, so a result
has nowhere to land. This was a missing input, not a modelling limitation.

## The missing input: the tournament's own draw feed

    https://www.usopen.org/en_US/scores/feeds/2026/draws/{MS,WS}.json

No auth, same family as the IBM point-by-point feeds ticket 07b settled on. It
carries all 127 matches of a 128 draw AND updates `status`/`winner` live, so a
single file is both the bracket and the results tape.

## Code

- `scripts/usopen_live_draw.py` — loads the real bracket + results, maps feed
  players to Sackmann uids, and verifies the result structurally.
- `scripts/simulate_usopen_live.py` — treats decided matches as deterministic,
  samples only what remains. `--ignore-results` runs the real bracket with
  results cleared (diagnostic, see below). Writes JSON plus
  `blog/live_predictions.parquet` in the existing blog schema.
- `scripts/refresh_usopen_live.cmd` — driven by scheduled task
  `DropshotverseUSOpenLiveSim` (every 30 min, ends 2026-09-14). **Guards the
  upload behind the sim's exit code** so a failed run cannot re-publish stale
  data with a fresh `updated_at` that makes it look current.

## Three assumptions checked rather than trusted — two were WRONG

This is the part worth remembering.

1. **`match_id`'s leading digit is the EVENT, not a constant.** Men's singles is
   `1101..1164`, women's `2101..2164`. The code assumed `1`, which raised loudly
   on the women's draw. Had it been tolerant, it would have built a silently
   wrong bracket.
2. **The `slot` field in `usopen_*_entrants.json` is NOT draw position.** It is
   seed-ordered: **2/128** slot-by-slot agreement with the real feed. Using it
   would have produced a completely wrong bracket that still looked entirely
   plausible in the output. Caught only by cross-checking against the feed
   before use.
3. **Bracket topology** (round `r` match `n` -> round `r+1` match `(n+1)//2`)
   was verified empirically on 8 completed matches, 8/8 agreeing, including the
   Navone/Djokovic anchor `1148 -> 1224`. This one held.

Player identity: feed uses US Open tour ids (`atpz355`), model uses Sackmann
uids. Joined on normalised name against the *bridged* archive — men 128/128,
women 127/128 (Gabriela Knutson genuinely new, gets a synthetic uid, reported
not hidden).

## Verification

Eliminated players sit at exactly 0.000 from the round they lost
(Djokovic, Norrie, Krejcikova); their conquerors at 1.000 for the round reached
(Navone, Van Assche, Rakhimova). 20k-sim run: 256 rows, 17 ATP / 16 WTA
eliminated matching the decided-match counts, 32 players per quarter per tour.

## The comparison trap, and the fix

**`locked` vs `live` conflates two changes** — learning the real draw AND
absorbing results — so reading a causal story off that difference is wrong. It
produced two confidently wrong claims before being decomposed:

| claim made from the confounded comparison | what the decomposition showed |
|---|---|
| "Sabalenka +6.4 because rivals fell in her quarter" | +5.2 **draw**, +1.2 results — she drew a soft quarter, true before a ball was hit |
| "Tiafoe rose because Djokovic went out of his quarter" | +2.6 **draw**, +0.2 results |

`--ignore-results` exists to keep this separable. With only 17 of 127 matches
played, the draw dominates; the one genuine results-driven move was Alcaraz
+2.0.

## Deliberately not done

Ratings stay frozen at `DATE0` (tournament start). Only *bracket state* updates.
This matches ticket 29's measured finding that cold ratings beat hot ratings at
every damping level for tournament simulation — a player who wins advances but
gets no mid-event rating bump.

## Published

`inthegame-data/tennis/live_predictions.parquet` — schema adds `quarter`
(A/B/C/D by true draw position), `eliminated`, `matches_decided` to the existing
blog columns. ITG issue #581 asks for the wall-chart page; no Worker code needed
(browser reads R2 directly).
