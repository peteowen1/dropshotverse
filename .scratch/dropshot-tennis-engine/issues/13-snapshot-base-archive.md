# 13 — Snapshot and freeze the base historical archive

Type: task
Status: resolved

## Question

**The one time-sensitive item on this map.** Approved by Pete 2026-08-27.

Sackmann's `tennis_atp` and `tennis_wta` vanished without warning. The base layer this project depends on — `Aneeshers/tennis-sackmann-archive` — is one person's frozen copy of those now-deleted repos. It is a single point of failure that has already failed once upstream. If it goes, there may be no third copy.

**Pull it down, verify it, and freeze it under our own control.**

Steps:
1. **Fetch** the full archive from both the GitHub repo and its HuggingFace mirror. Record which mirror each file came from.
2. **Verify the two mirrors agree** — checksum every file across both. A mismatch is important information about whether either can be trusted.
3. **Sanity-check the contents against ticket 01's claims**, per the standing rule that every load gets checked: row counts, date ranges, unique player counts, and which of the 49 columns are actually populated — broken down by tour (ATP/WTA), level (tour/challenger/futures/qualifying) and decade. Print a summary. If ticket 01's description of the archive is wrong, this is where we find out.
4. **Record serve-statistic coverage** while the data is loaded — the fraction of matches with non-null `svpt`, `1stIn`, `1stWon`, `2ndWon`, `bpSaved`, by tour, level and decade. This is the exact number ticket 12 needs, so capture it here rather than loading everything twice.
5. **Write a manifest**: per-file sha256, byte size, source mirror, fetch timestamp, row count.
6. **Freeze it locally** under a `dropshotdata` working directory.

**Then stop.** Creating the GitHub repo and publishing the Release is outward-facing and needs Pete's explicit go-ahead — queue it with a recommendation rather than doing it. When it does happen, follow the pattern in `pannaverse/panna/R/piggyback.R` and `bouncerverse/bouncerdata/scripts/upload_to_release.R`, and vendor `versebus.R` verbatim from `torpverse/torp/R/versebus.R`.

**Note on tooling:** anything touching parquet must run via `powershell.exe -Command 'Rscript "path/to/script.R"'` — `arrow` segfaults under Git Bash R with no error message.

**Deliverable:** the frozen archive, the manifest, the printed sanity-check summary, and the serve-stat coverage table that ticket 12 needs.

## Answer

Resolved 2026-08-27. Archive frozen locally at `C:\dev\dropshotverse\dropshotdata\`. No git repo was created, nothing was pushed or published — local filesystem only, per the hard constraint. Scripts used: `dropshotdata/scripts/fetch_archive.py` (fetch + checksum both mirrors), `dropshotdata/scripts/sanity_check.R` (load + coverage), `dropshotdata/scripts/add_row_counts.py` (manifest row counts). Full outputs: `dropshotdata/manifest/`.

### 1–2. Fetch + mirror verification

Fetched all 473 files from GitHub (`Aneeshers/tennis-sackmann-archive`, commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`, pushed 2026-06-25) and all 473 files from the HuggingFace mirror (`https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive`), via plain HTTPS (`raw.githubusercontent.com` and the HF `resolve/main` endpoint — no `git clone`, no LFS pointer issues). Total ~797MB per mirror (~1.5GB on disk between both raw copies), well within a normal download — the "report size before downloading everything" trigger wasn't hit, so the full archive was pulled directly, ATP+WTA main-tour CSVs first as the priority layer regardless (120 files, 67MB) followed by everything else.

**Checksum result: all 465 CSV data files are byte-for-byte identical (sha256 match) across both mirrors — zero mismatches, zero one-sided files.** The two mirrors agree completely on data. The only differences are non-data packaging files: `README.md` content differs between the two (different sha256 — GitHub's repo README vs. HuggingFace's dataset card, expected and immaterial), `.gitattributes` exists only on the HF side (its LFS config) and `.gitignore` only on the GitHub side. This is strong corroboration for ticket 01's claim that this snapshot is a faithful, unmodified copy of Sackmann's data — two independently-hosted copies match exactly on every actual data file.

### 3. Sanity check — and a correction to an earlier mid-download claim

An earlier check-in during this ticket's execution, taken **while the download was still in progress** (not actually complete despite an interim report saying so), concluded that WTA had **no** qualifying/ITF files and only main-tour CSVs. **That was wrong, and it's important to say so plainly: it was an artifact of checking a partial download**, not a real property of the archive. My fetch script processes ATP's non-priority files (rankings, futures, qual_chall, doubles) before WTA's non-priority files alphabetically, so at that checkpoint WTA's `qual_itf` files genuinely hadn't been fetched yet. With the download actually complete (verified: 473/473 files both mirrors), **WTA does have 59 `wta_matches_qual_itf_YYYY.csv` files (1968–2026)**, matching ticket 01's description exactly. The `28 vs 27` slam_pointbypoint file-count discrepancy also raised at that checkpoint likewise resolves cleanly now: both mirrors have exactly 168 files each, identical filenames, identical checksums.

What **is** confirmed real: **WTA has zero doubles files** (ATP has 21, WTA has 0) — this part of ticket 01's "doubles (ATP only)" claim holds. So the real structural asymmetry between tours is doubles-only, not qualifying/ITF as the mid-download snapshot suggested.

**A genuine finding ticket 01 didn't mention:** the ATP directory contains `atp_matches_amateur.csv` — a 49-column-schema file covering the **pre-Open era, 1877-07-09 to 1967-12-26** (25,001 matches). Every row in this file is structurally ragged (only 40 of 49 fields populated per row — no ATP ranking system existed before 1973, and the file simply omits those trailing columns rather than leaving them blank) — `fread(..., fill=TRUE)` handles this correctly since the missing fields are legitimately absent, not corrupted. This means ATP's true coverage in this archive starts at **1877**, not 1968 as ticket 01 stated — ticket 01 slightly understated the archive's lower bound. There is no WTA equivalent (the WTA wasn't founded until 1973).

**Row counts, date ranges, unique players** (singles match-level files only: main + qual_chall/qual_itf + futures, excluding doubles and excluding the differently-schemed `slam_pointbypoint/*-matches*.csv` files, which are slam-specific point-by-point match metadata, not the 49-col Sackmann schema — an early version of this check accidentally globbed those in via a loose filename pattern and it visibly corrupted the results with implausible rows; fixed by scoping the glob to `atp/` and `wta/` only):

| tour | level | n_matches | min_date | max_date |
|---|---|---:|---|---|
| ATP | tour | 224,390 | 1877-07-09 | 2026-05-25 |
| ATP | qualifying | 82,656 | 1978-01-08 | 2026-06-01 |
| ATP | challenger | 157,815 | 1978-01-08 | 2026-06-01 |
| ATP | futures | 523,355 | 1990-12-29 | 2026-06-01 |
| WTA | tour | 162,182 | 1967-12-25 | 2026-05-25 |
| WTA | qualifying | 225,619 | 1968-04-22 | 2026-06-02 |
| WTA | challenger* | 398,856 | 1972-08-07 | 2026-06-02 |

\* WTA has no separate "Futures" tier the way ATP does. Its `qual_itf` file's non-qualifying rows span ITF Circuit tournaments across prize-money bands ($10K–$100K, tourney_level codes `10,15,20,25,35,40,50,60,75,80,100`) plus a generic legacy `C` code — this is WTA's closest analog to ATP's combined Challenger+Futures depth, folded into one "challenger" bucket here for comparability. Level classification is derived from `round` (any round starting with `Q` = qualifying, regardless of the parent event's level) crossed with `tourney_level` for non-qualifying rows — verified against the raw data with zero unclassified rows.

- **Total singles matches: 1,774,873** (ATP 988,216 / WTA 786,657).
- **Unique players:** ATP 35,061, WTA 30,561 (57,714 combined, by winner_id ∪ loser_id).
- **Column population:** confirmed all match files (other than the amateur-era file, by design) have the full 49-column Sackmann schema. Population is uneven: core identity/result columns (IDs, names, scores, dates) are ~99.98–100% populated; ranking columns 72–84%; player bio (height/hand) 46–99%; the serve-stat block (`w_/l_` prefixed columns) sits around 20.4% populated overall — see the coverage table below for the real breakdown by tour/level/decade.
- **Data-quality note for downstream use:** a naive `tourney_id + match_num` duplicate check flags 41,632 groups with >1 row, but this is mostly explained by `match_num` not being globally unique within a `tourney_id` (qualifying rounds and the main draw of the same event can reuse the same match_num under Sackmann's numbering) — restricting to `tourney_id + match_num + round + level` narrows this to 11,968 genuine collision groups. Worth deduping/investigating before any downstream join keyed on `tourney_id+match_num` alone; out of scope to chase further here.

**Verdict on ticket 01: mostly holds, with one correction (WTA does have qual/ITF, contra the mid-download check) and one addition (ATP goes back to 1877 via a bonus amateur-era file, not just 1968).**

### 4. Serve-statistic coverage table (the ticket-12-critical number)

Coverage = fraction of matches with a **complete** serve line — `svpt`, `1stIn`, `1stWon`, `2ndWon`, `bpSaved` all non-null for **both** players (winner and loser prefixes `w_`/`l_`). Real counts from the loaded data, not estimates.

**By tour:**

| tour | n_matches | n_with_serve | coverage |
|---|---:|---:|---:|
| ATP | 988,216 | 259,680 | **26.3%** |
| WTA | 786,657 | 101,989 | **13.0%** |

**By tour × level:**

| tour | level | n_matches | coverage |
|---|---|---:|---:|
| ATP | tour | 224,390 | 45.4% |
| ATP | qualifying | 82,656 | 71.7% |
| ATP | challenger | 157,815 | 47.0% |
| ATP | futures | 523,355 | 4.6% |
| WTA | tour | 162,182 | 29.5% |
| WTA | qualifying | 225,619 | 11.8% |
| WTA | challenger (ITF) | 398,856 | 6.9% |

**By tour × level × decade** (full table also written to `dropshotdata/manifest/serve_stat_coverage_by_tour_level_decade.csv`; zero rows before the 1990s across the board, so only decades with any non-zero coverage shown):

| tour | level | decade | n_matches | coverage |
|---|---|---|---:|---:|
| ATP | tour | 1990s | 37,150 | 79.9% |
| ATP | tour | 2000s | 32,335 | 88.3% |
| ATP | tour | 2010s | 29,397 | 92.5% |
| ATP | tour | 2020s | 17,567 | 93.9% |
| ATP | qualifying | 2010s | 32,799 | 91.2% |
| ATP | qualifying | 2020s | 29,460 | 99.6% |
| ATP | challenger | 2010s | 44,261 | 98.9% |
| ATP | challenger | 2020s | 30,568 | 99.5% |
| ATP | futures | 2020s | 93,574 | 26.0% |
| WTA | tour | 2000s | 28,790 | 30.1% |
| WTA | tour | 2010s | 27,868 | 85.6% |
| WTA | tour | 2020s | 16,056 | 95.2% |
| WTA | qualifying | 2010s | 49,193 | 26.1% |
| WTA | qualifying | 2020s | 81,979 | 16.7% |
| WTA | challenger (ITF) | 2010s | 144,062 | **0.7%** |
| WTA | challenger (ITF) | 2020s | 87,934 | 30.2% |

**Headline for ticket 12:**
1. **Coverage genuinely collapses before ~1990** exactly as ticket 12 anticipated — it is a hard zero for every tour/level combination in every decade before the 1990s (ATP) / 2000s (WTA tour). The Markov-likelihood inversion route is not optional for pre-modern history; it's required if that era is to be modeled at all.
2. **Modern-era ATP is in good shape at tour, qualifying, and challenger level** — all >90% by the 2020s. **ATP futures is the outlier: still only 26% covered even in the 2020s** — three-quarters of current-era ATP futures matches have no serve stats.
3. **WTA is structurally worse than ATP at every level except top tour.** Top-tour WTA reaches 95.2% by the 2020s, comparable to ATP. But **WTA's ITF-tier (challenger-equivalent) matches are almost entirely without serve stats — 0.7% in the 2010s, only 30.2% even in the 2020s** — versus ATP Challenger's 98.9%/99.5% in the same decades. This is not a freshness lag closing over time the way ATP's did; it looks like a standing reporting gap specific to WTA's lower tier.
4. **WTA qualifying coverage is anomalous: it drops from 26.1% (2010s) to 16.7% (2020s)** rather than rising — flagged here as a real, counter-intuitive pattern worth a closer look before ticket 12 finalizes an estimation strategy, not smoothed over.
5. Net effect: a fitting strategy that treats "ATP" and "WTA" symmetrically at the challenger/qualifying depth will be systematically starved of point-count data for WTA — ticket 12 should treat WTA's lower tiers as a scoreline-inversion-required regime even for recent years, while ATP's lower tiers (excluding futures) are largely fine using point counts directly from ~2010 onward.

### 5–6. Manifest and freeze

- `dropshotdata/manifest/manifest.json` / `manifest.csv`: per-file path, byte size (both mirrors), sha256 (both mirrors), `mirrors_agree` flag, row count, fetch timestamp. 474 entries (473 shared + the 2 mirror-specific packaging files).
- `dropshotdata/manifest/sanity_check_output.txt`: full console output of the sanity check.
- `dropshotdata/manifest/serve_stat_coverage_by_tour_level_decade.csv`, `row_counts_by_tour_level.csv`, `row_counts_by_tour_decade.csv`: the tables above as CSV.
- Raw frozen files: `dropshotdata/raw/github/` and `dropshotdata/raw/huggingface/` (both mirrors kept side by side, not deduplicated, so provenance per mirror is preserved).
- `dropshotdata/docs/README.md`: layout + provenance notes for anyone picking this up later.

### Recommendation (not done — needs Pete's go-ahead)

Publish this frozen snapshot as a `dropshotdata` GitHub repo + Release, following `pannaverse/panna/R/piggyback.R` and `bouncerverse/bouncerdata/scripts/upload_to_release.R` as the pattern, vendoring `versebus.R` verbatim from `torpverse/torp/R/versebus.R`. Given the single-point-of-failure risk this ticket exists to address, doing this soon is worth prioritizing — but it is outward-facing (new public repo + release) and per the hard constraint on this ticket, requires explicit sign-off before it happens.
