# Between events: a ratings-led tennis page that is actually current

Written 2026-09-06 (option A from the 2026-09-06 queue decision). Scope: what
the site shows when no event is live, using data that already publishes.
Option B, a forecast for the next event, is a separate plan.

## What exists today, and what is wrong with it

- **The page exists.** `inthegame-blog/tennis/ratings.qmd` reads
  `tennis/ratings.parquet` from R2 and renders Elo and BT for both tours,
  with a name search. Nothing new is needed to *show* ratings.
- **The data is stale and says it is fresh.** `ratings.parquet` was built
  once, on 2026-08-29 (300 rows: top 150 per tour by Elo among players
  active in the last 52 weeks; ratings as of the bridge's 2026-08-24). Nothing
  rebuilds it. The laptop task re-uploads it every 30 minutes alongside the
  live tables, so R2's `Last-Modified` is always minutes old, and the page's
  "Updated {date}" comes from that header. **The site currently labels
  eight-day-old ratings as updated today.** The parquet's own `updated_at`
  column carries the truth and nothing reads it.
- **The Elo input is frozen.** `elo_raw_ratings_full.parquet` (the live
  sim's Elo source too) stops at 2026-08-24; `regen_elo_raw_with_bridge.py`
  regenerates it with current TML rows but is run by hand. BT comes from the
  hierarchical snapshot at `as_of` plus the XGBoost pairwise model, so it
  moves with the bridge, but only when someone runs the build.
- **The index page does not know an event can end.** `live_predictions.parquet`
  carries `event_status` and `champion` (since 2026-09-06); nothing in
  `tennis/index.qmd`, `tennis-live-sim.js`, `wallchart.qmd` or
  `simulator.qmd` reads either. After the final the hero keeps rendering the
  finished bracket as if live, and the freshness alarm correctly goes quiet.
- `predictions.parquet` is the LOCKED pre-tournament forecast and must not
  be rebuilt by a refresh job; `build_blog_data.py` currently builds both
  tables in one call.

## The plan

### dropshotverse (about half a day, most of it the Elo timing)

1. **Split the build.** `build_blog_data.py --ratings-only`: builds
   `ratings.parquet` alone, leaves `predictions.parquet` untouched. Add a
   `ratings_as_of` column (the bridged archive's max match date) so the page
   can show the data date.
2. **Make the Elo input current inside the job.** The refresh must run
   `regen_elo_raw_with_bridge.py` first, or the Elo column stays at
   2026-08-24 forever. **Profile it before scheduling** (rule in
   `~/.claude/CLAUDE.md`): the old full replay was "~5 min, several GB";
   `tune_scratch/elo_compact.py` does the same replay bit-identically in 37 s
   and 409 MB, and is the obvious engine if the un-compact one is too slow
   for a runner. Do NOT run the regen on the laptop mid-event: it overwrites
   the file the live sim reads, and a TML correction to a pre-2026-08-30 row
   would move the frozen snapshot.
3. **Schedule it.** `.github/workflows/ratings-refresh.yml`, `ubuntu-latest`,
   daily at 06:00 UTC, same Release fetch as `live-projection.yml`, then
   regen Elo, build ratings, upload `ratings.parquet` only. GitHub's cron is
   unreliable at 30-minute granularity; a daily job that lands within a few
   hours is fine. Cost: a few Linux minutes a day.
4. **Alarm it.** Extend `check_blog_data_freshness.py --url` with a ratings
   check: warn (not fail) when `ratings.parquet`'s `updated_at` is older than
   three days. It already runs hourly.

### inthegame-blog (about two hours; the ITG session owns that repo)

5. **Tell the truth about the date.** `ratings.qmd`: "Ratings as of
   {ratings_as_of}" read from the parquet, not `dataUpdated()`'s
   `Last-Modified`. This is a one-line fix and worth doing before anything
   else, because it is wrong on the live site now.
6. **Let the index page end an event.** Read `event_status` and `champion`
   from the shared tennis sim. When both tours are `complete`: a result card
   (champions, final title odds they carried in) above a top-10 ratings
   strip per tour and a link to the ratings page. When no live rows exist at
   all (between events): ratings-led hero. The wall chart and simulator keep
   rendering the finished bracket under a "Final" label rather than "Live".
7. **Later, option B**: tournament-agnostic pages keyed by `tournament_id`,
   fed by a generic WTA event forecast (`fetch_wta_upcoming.py` field +
   `simulate_tournament.py`), first target Beijing (field seeds ~2026-09-23).

## Order

Step 5 first (a wrong label on a live page), then 1 to 4 on a branch with the
Elo timing measured before the schedule goes in, then 6. Step 2's profiling is
the one number this plan is missing; everything else is known.

## Anchors for the refreshed ratings (write down before looking)

- The world number 1 on each tour is top 3 by both Elo and BT.
- A player who has not played in 52 weeks is absent.
- `ratings_as_of` is within a week of today on every refresh, else the
  bridge broke and the alarm must say so.

## Status 2026-09-07

- **Step 5 done** on inthegame-blog dev by the ITG session: the page labels
  from the parquet ("Ratings as of {ratings_as_of}", else "Ratings built
  {updated_at}"). **Step 6 filed as inthegame-blog#660.**
- **Steps 1 to 4 shipped** (dropshotverse PR #20): `ratings-refresh.yml` daily
  at 06:00 UTC on Linux. Dry run: Elo regeneration 38 s, ratings build 114 s,
  3 min 50 s end to end; anchors PASS; Sinner/Alcaraz and Sabalenka/Gauff
  top two on their tours. First real run published 347 rows the same night.
- **Semantics worth knowing:** `ratings_as_of` is the start date of the
  latest tournament in the bridged data, and the Elo shown is the rating
  strictly BEFORE that date. An in-progress event is therefore excluded by
  construction (consistent with the simulator's ratings freeze) and lands the
  day the next tournament appears in the bridge. During a Slam the table
  does not move; that is the design, not a stuck job.
- **Staleness is a warning, not a gate:** the tour-level bridge brings
  nothing for five to seven weeks in the off-season.

