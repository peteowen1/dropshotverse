# Pre-match forecasts for every upcoming ATP/WTA event, on ITG

Started 2026-09-15, Pete's ask: emulate what the US Open got (a simulator
page on ITG) for every upcoming tournament, both tours, continuously through
the season. **Scope note, agreed with Pete the same session**: this is the
PRE-MATCH bracket forecast only (title/round probabilities before or during
a tournament), not the live in-match win-probability work. That is Slam-only
infrastructure (`usopen.org`'s own feeds) and does not exist, and cannot
easily be built, for a regular tour stop - no equivalent feed exists for
Beijing, Tokyo, or any non-Slam event. Nothing here proposes building it.

## What already exists (as of 2026-09-15)

- `simulate_wta_event.py` - forecasts ONE WTA event from its WTA-API entry
  list. Works, verified end-to-end tonight against Beijing's partial field
  (D-59 dry run). Single-event, manually dispatched via
  `event-forecast.yml`, `workflow_dispatch` only.
- `fetch_wta_upcoming.py` - the WTA calendar and one event's entry list, from
  `api.wtatennis.com`. This is the ONLY calendar source that exists in this
  repo for either tour.
- Just fixed (2026-09-15): a real crash when an entrant's name doesn't
  resolve to an archive uid (pandas NaN serialized as a bare JSON token,
  read back as a float, crashing a mixed float/str sort). Matters a lot for
  this plan - generalizing to more, smaller events means more obscure
  players, which means more resolution misses, which means this exact path
  gets exercised far more often than it was for one WTA 1000.
- `upcoming_predictions.parquet`'s own schema already supports MULTIPLE
  events (keyed by `tournament_id`, "replacing that tournament's previous
  rows and keeping every other event") - the data side does not need a
  redesign to hold more than one event at a time.
- Nothing on the ITG blog reads `upcoming_predictions.parquet` at all.
  `inthegame-blog#660` (the US Open's own finished-event hero) is a
  DIFFERENT, still-OPEN piece of work - not a template already built for
  this.

## What does NOT exist and is the real work

### 1. An ATP calendar and field source

No ATP tournament API works (`api.atptour.com` doesn't resolve; the site is
server-rendered Sitecore). Tonight's research found a real path instead:

- **Calendar**: Wikipedia's `2026 ATP Tour` season page lists all 65
  tournaments (4 Slams, ATP Finals, 9 ATP 1000, 16 ATP 500, 29 ATP 250) with
  dates. This is a genuinely usable calendar source, unlike the tournament
  API. Not yet parsed into anything - needs the same care
  `fetch_wta_upcoming.py` gives the WTA calendar (cache, don't poll, page
  fully, refuse a truncated fetch).
- **Field per tournament**: harder, and Wikipedia alone is NOT enough.
  Checked the 2026 China Open (Beijing) article directly (raw HTML, not a
  summarised fetch) as the test case:
  - The ATP **seeds** table IS populated: 8 real names with REAL, CURRENT
    ATP ranks (Sinner #1, Zverev #2, Auger-Aliassime #5, Medvedev #6,
    Cobolli #7, de Minaur #9, Djokovic #12, Mensik #15).
  - The WTA seeds table is a bare unfilled template (`[[|]]` placeholders,
    no names) - Wikipedia does not always have this yet, this far out.
  - Neither tour has a full **entry list** table (all ~32-96 direct
    acceptances) on this particular article. A **Withdrawals** table exists
    and IS populated (real names, e.g. Hailey Baptiste, Katie Volynets for
    WTA) even where seeds aren't - a genuinely useful cross-check regardless
    of which tour.
  - So Wikipedia alone gives: a reliable calendar, SOMETIMES a seed list,
    and a withdrawals list. It does not reliably give a full field.

### 2. A field-CONSTRUCTION method for the gap Wikipedia leaves

Pete's proposal, tested tonight: use entry-rule + current-ranking
construction to fill in what neither the API nor Wikipedia has yet.

- The archive DOES carry an ATP rankings file
  (`raw/github/atp/atp_rankings_current.csv`) - but it is stale to
  2026-06-08, over 3 months behind, useless on its own for "who is
  currently good enough to get direct entry."
- **Tested using our own Elo rating (`elo_raw_ratings_full.parquet`,
  freshened daily via the tennismylife bridge) as a live proxy instead.**
  Result, real and important: naive Elo-rank order is NOT safe to use
  directly. The top 20 by raw Elo included Roger Federer (last match
  2021-06-28), Robin Soderling (last match 2011-07-11) and Rafael Nadal
  (last match 2024-11-19) - retired players whose rating simply hasn't
  decayed to irrelevance. **A recency filter (last match within some active
  window, on the order of 8-12 weeks - not yet tuned) is a hard requirement
  before this rating can stand in for a ranking.**
  - Even filtered to active players, an Elo-based list is a strength
    ordering, not an entry list: it cannot know a player skipped this
    specific event for a reason the data has no way to see (Carlos Alcaraz
    sits at #2 by our Elo and is genuinely absent from Beijing's real
    8-seed list - most likely a withdrawal or a deliberate skip, not a data
    error). **This is an irreducible gap, not a bug to fix** - a constructed
    field will sometimes include a player who isn't actually there.
  - The fix is combining sources, not trusting one: known Wikipedia seeds
    and withdrawals are ground truth and override the constructed list;
    Elo (recency-filtered) fills the remaining draw slots as a best-effort
    approximation; the result is tagged by provenance per player
    (`"official"` / `"wikipedia"` / `"constructed"`), the same way locked
    vs. provisional already works for the US Open and Beijing WTA.
- **Pete's other idea - "bring forward whoever played the last edition" -
  not yet tested.** Worth trying as a THIRD input to combine with the above
  (a returning champion or regular is a good prior even before rankings
  settle), particularly for smaller events where the field is thinner and
  more stable year to year. Not evaluated tonight; flagged as a real option
  for whoever picks this up.

### 3. A per-tournament entry-rules reference

Pete's ask: "check entry rules for ATP/WTA events and save a doc noting
each tournament's rules and who gets to enter." Not started. Shape it
should probably take: one row per tournament LEVEL (not per tournament -
entry rules are a property of Slam / 1000 / 500 / 250 and gender, not of
the specific stop), covering draw size, seed count (already have
`expected_seeds()` for WTA), wildcard count, qualifying spots, and the
direct-acceptance rank cutoff convention. This is what turns "top N of our
constructed ranking" into the RIGHT N for a specific event.

### 4. Generalizing the simulator past one hand-picked WTA event

`simulate_wta_event.py` is WTA-specific (`build_h2h_dict`, `build_player_table`
etc. are tour-parameterized already, underneath) but the field/entry-list
plumbing (`load_entrants`, `build_field`) is written for the API-shaped JSON
one script produces. Needs either a shared `simulate_tour_event.py` taking a
tour argument, or a parallel `simulate_atp_event.py` that reuses the shared
XGBoost/hier machinery but reads a CONSTRUCTED field (source #2 above)
instead of an API entry list. The antisymmetrised round-robin matrix code
(`simulate_tournament_xgb.py`) is already tour-agnostic and needs no change.

### 5. Automation across a whole season

Today, dispatch is manual and one event at a time
(`gh workflow run "Event forecast (WTA)" -f group_id=...`). "All future
tournaments" means something that:
- walks the (new) ATP calendar and the existing WTA calendar,
- decides per event whether it's close enough to forecast (a WTA 1000 gets
  dispatched ~1 week out per the existing convention; other levels need
  their own lead time, informed by item 3 above),
- runs the right simulator (WTA API-sourced, or ATP/under-populated-WTA
  constructed-field) and publishes,
- and RE-runs as the field firms up (constructed -> partially confirmed ->
  official), each time upgrading the `source` tag rather than leaving a
  stale provisional number live indefinitely.
This is genuinely new scheduling/orchestration, not an extension of the
existing manual `workflow_dispatch`.

### 6. The ITG blog page

Pete's inspiration is the US Open's `simulator.qmd` - a bracket simulation
view. Nothing reads `upcoming_predictions.parquet` on the blog today. A
generalized version needs:
- a way to pick or list which upcoming event you're looking at (the US Open
  page never needed this - there was only ever one event),
- the bracket/probability UI itself, reusable across events since the
  underlying data schema is already the same shape per tournament,
- honest treatment of provenance (item 2/4's `source` tag) surfaced to the
  reader - a constructed field should not look as certain as a locked one,
  the same design principle `simulate_wta_event.py`'s own
  `source="provisional"` already applies to unseeded/incomplete fields.
This is a `inthegame-blog` piece of work, a different repo, and per that
repo's own convention should probably be filed as its own scoped issue
(the way `inthegame-blog#660` was) rather than built ad hoc here.

## Suggested order for whoever picks this up

1. Parse the ATP calendar from Wikipedia (`2026 ATP Tour`) - cheap, no
   ambiguity, unblocks everything ATP.
2. Build and validate the recency-filtered Elo-based field constructor on
   Beijing ATP specifically, where Wikipedia's real 8 seeds are a ready-made
   answer key (get the top-8-by-construction to plausibly match, within
   reason, before trusting it past seed 8).
3. Write the entry-rules reference doc (item 3) - short, and needed before
   item 2's "top N" has a real N per tournament level.
4. `simulate_atp_event.py` (or generalize `simulate_wta_event.py`) once 1-3
   exist.
5. File the `inthegame-blog` issue for the multi-event sim page (item 6) -
   can happen in parallel with 1-4, since the schema it will read already
   exists.
6. Automation (item 5) last - only worth building once a handful of events
   have gone through the manual path and the lead-time/re-run conventions
   from item 3 are actually known to work.
