# 07a — Lock the historical data foundation

Type: grilling
Status: resolved
Blocked by: 01, 04

## Question

Split from the original ticket 07 on 2026-08-27, because the historical decision does not depend on the live one and was needlessly holding up the whole modelling path. Live is now ticket 07b.

Tickets 01 and 04 have reported. Decide:

1. **Base layer.** Confirm `Aneeshers/tennis-sackmann-archive` as the frozen historical source of truth, or name a better one. Establish exactly what it contains by loading it, not by trusting the ticket-01 summary — row counts by tour, level and decade, and which of the 49 columns are actually populated.
2. **Live-patch layers and how they stitch.** `stats.tennismylife.org` for current ATP Challenger/Qualifying; tennis-data.co.uk for fresh cross-tour main tour. The base is ID-keyed and the patches are not, so the join needs bespoke name resolution — decide whether that logic lives in `dropshotdata` at ingest (recommended: resolve once, store canonical IDs) or in `dropshot` at read time.
3. **Player identity.** Design the canonical player ID and the crosswalk. Ticket 04 found no published crosswalk exists anywhere; the working pattern is `carltoews/tennis` `merge_datasets.ipynb`, expecting 95-99% automatic match. Decide what happens to the unmatched tail — dropped, hand-patched, or quarantined.
4. **Odds join.** Confirm tennis-data.co.uk as the benchmark source, and note the year-dependent schema (only `B365W/L` and `PSW/L` survive recent seasons; nine other books are legacy-only). The loader must not assume a fixed column set.
5. **Storage shape.** Partitioned parquet in `dropshotdata`, published to a GitHub Release via piggyback using the pattern from `pannaverse/panna/R/piggyback.R` and `bouncerverse/bouncerdata/scripts/upload_to_release.R`. Decide the partitioning (by tour × year is the obvious candidate) and what the manifest records.
6. **Refresh cadence** for the live-patch layers, and what triggers it.

**Already decided, do not re-open:**
- **ATP Challenger stays in scope; futures/ITF is OUT** (revised 2026-08-27 on ticket 12's measured coverage). Challenger has 95.6%/99.5% serve-stat coverage in the 2010s/2020s — genuinely usable point counts. Futures has 4.6%. Ingest `atp_matches_qual_chall_*.csv`; do **not** ingest `atp_matches_futures_*.csv`.
- Challenger's justification is fixing the qualifier cold-start problem, *not* benchmarking — there are no free odds below ATP250. Any closing-line result is explicitly a main-tour number.
- ~~**The WTA cold-start problem has no solution in this archive.** It holds no women's lower-tier files at all.~~ **WRONG — corrected in the Answer below.** 48 `wta_matches_qual_itf_*.csv` files exist (425,139 matches, 1968-2018). The real limitation is that they are scoreline-only and stop in 2018.
- Archival policy is settled and being executed as ticket 13.

**Deliverable:** a named stack, a storage and partitioning spec, the player-identity design, and an explicit list of what we are choosing to live without. Complete enough that ticket 08 can be written against it.

## Answer — resolved 2026-08-27

All figures measured directly over the frozen archive. Scripts: `scratchpad/foundation_audit.py`, `scratchpad/serve_coverage.py`.

### CORRECTION FIRST — WTA lower-tier data does exist

**Two successive claims about WTA lower-tier data were wrong, both because they were measured against an incomplete download.** Recorded in full because the failure mode matters more than the fact.

1. First claim: "the archive holds no women's lower-tier data at all." Wrong — taken from a directory listing mid-download.
2. Second claim (this ticket's first draft): "48 files, 425,139 matches, stopping in 2018." Also wrong — the fetch script works alphabetically, so WTA's non-priority files were still arriving.

**Final, measured against the verified-complete archive** (473/473 files per mirror, all 465 CSVs byte-identical across GitHub and HuggingFace by sha256):

| | matches | serve stats | runs to |
|---|---|---|---|
| ATP tour | 199,389 | 79.8→93.9% since 1990s | 2026-06-01 |
| ATP qual/challenger | 240,471 | **95.6% (2010s), 99.5% (2020s)** | 2026 |
| WTA tour | 162,182 | 0% pre-2000, 85.6% (2010s), 95.2% (2020s) | 2026-05-25 |
| **WTA qual/ITF** | **624,475** (59 files) | 0% until 2010s, **7.2% (2010s), 23.7% (2020s)** | **2026** |

So WTA lower-tier data is **abundant and current, but overwhelmingly scoreline-only** — 23.7% serve-stat coverage at its best, against ATP Challenger's 99.5%. The substantive conclusion is unchanged: it feeds a match-outcome cold-start prior, not the point-level model. But it is neither absent nor stale.

**Also found, and ticket 01 missed it:** `atp_matches_amateur.csv` — 25,001 matches, **1877-07-09 to 1967-12-26**. ATP coverage in this archive starts at 1877, not 1968. Rows are structurally ragged (40 of 49 fields; no ranking system existed pre-1973), so read with `fill=TRUE`. Materially extends what the GOAT list could cover, though with no serve statistics whatsoever.

**Process lesson, now standing policy for this map: verify a download is complete before measuring anything from it.** Two wrong findings, reported confidently, from the same avoidable cause.

### 1. Base layer — CONFIRMED

The frozen archive at `dropshotdata/raw/{github,huggingface}/`. 533 files, 677 MB, two mirrors. ATP 1967-12-28→2026-06-01, WTA 1967-12-25→2026-05-25, zero null dates. 50 columns.

### 2. Player identity — the landmine, and the decision

**ATP and WTA player ids collide. 2,396 ids appear in both tours with different people behind them.** Verified examples: id `200003` is Jeremy Beale in ATP and **Justine Henin** in WTA; `200005` is Ugo Humbert and Jolene Watanabe.

A naive union of the two tours would silently merge them. Nothing in an aggregate metric would show it — the ratings would just be quietly wrong for thousands of players.

**Decision: the canonical key is `player_uid = "{tour}:{player_id}"`** — `atp:200003`, `wta:200003`. Namespacing happens **at ingest, once**, never at read time. The raw `player_id` is retained as a column but is never a join key on its own.

Encouragingly, ids are otherwise clean: only 34 ATP ids and 1 WTA id carry more than one name spelling, out of ~31,000 players. So within a tour, id→player is reliable and **name matching is only needed at source boundaries** (odds files, live feeds), not internally.

**Live-feed bonus:** the US Open feed (ticket 16) returns ids already in `atp*`/`wta*` form (`wta322191`, `atpm0ni`). Whether these map to archive ids is unverified and worth checking — it could remove most of the crosswalk work.

### 3. Ingest architecture

Name resolution and id namespacing live in **`dropshotdata` at ingest**. Resolve once, store canonical, so every consumer reads clean keys. `dropshot` never does fuzzy matching at read time.

Unmatched rows are **quarantined, not dropped** — written to a `_unmatched` partition with the reason. Silent dropping hides systematic bias (an unmatched tail is rarely random; it skews toward accented names and lower tiers).

### 4. Known data hazards, all measured

- **Non-standard scores: 3.15% ATP (13,836), 2.50% WTA (9,831)** — `RET`, `W/O`, `DEF`, `ABD`. Must be explicitly classified at ingest, never string-parsed as normal scorelines. Ticket 11 found the field-wide practice is discard-without-justification; we should at minimum flag them and decide deliberately.
- **Surface casing bug**: WTA has 31 rows of `'clay'` alongside 155,165 of `'Clay'`. Normalise.
- **Missing surface**: 2,990 ATP and 6,547 WTA rows.
- **`minutes` is barely usable**: 51.9% ATP, 6.6% WTA.
- **Ranks are incomplete**: `winner_rank` 89.9% ATP / 71.2% WTA — do not rely on rank as a model input without a missingness story.

### 5. What to ingest for v0

**In:** `atp_matches_*.csv`, `atp_matches_qual_chall_*.csv`, `wta_matches_*.csv`, `wta_matches_qual_itf_*.csv` (as a match-outcome-only cold-start layer, flagged as such).
**Out:** `atp_matches_futures_*.csv` (4.6% serve stats), all doubles files.

### 6. Storage

Partitioned parquet under `dropshotdata/`, partitioned by **`tour` / `level` / `year`** — level included because point-count availability differs so sharply by level that most reads will filter on it. Publish via piggyback to a GitHub Release using the pattern in `pannaverse/panna/R/piggyback.R` and `bouncerverse/bouncerdata/scripts/upload_to_release.R`; vendor `versebus.R` verbatim from `torpverse/torp/R/versebus.R`. Manifest records per-file sha256, row count, source mirror, and fetch timestamp.

**Note:** parquet work must run via `powershell.exe -Command 'Rscript "..."'` — `arrow` segfaults under Git Bash R.

### 7. Refresh cadence

The base is frozen and will not update — upstream is deleted. Freshness comes from the live-patch layers, so cadence is ticket 07b's problem, not this one. **Re-verify the frozen snapshot's checksums quarterly** so bit-rot is caught.

### What we are choosing to live without

1. Point-level data before 1991 on any tour — it was never recorded.
2. WTA point-level data before ~2010.
3. WTA lower-tier data after 2018, at any granularity.
4. Futures/ITF men's data entirely.
5. Reliable match duration, and complete rankings.
