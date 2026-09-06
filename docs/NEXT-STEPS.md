# dropshotverse — Next Steps

> Living queue. Sessions update this when they finish work (rule in `C:\dev\CLAUDE.md`).
> Decision log: [DECISIONS.md](DECISIONS.md) · Vault map: [HOME.md](HOME.md)
> Created 2026-09-05 — this verse had no docs hub until then, and appeared in
> neither `C:\dev\ECOSYSTEM.md` nor `C:\dev\vault\NEXT-STEPS.md` (both fixed).
> Last updated: **2026-09-06**. Two big questions closed: the market chase is
> settled (no edge, measured three ways) and issue #3 is decided (D-11, keep
> Sackmann as base). WTA upcoming-tournament forecasting works end to end; the
> in-match win-probability model has its first calibrated result.
>
> **Read first if you are cold:** the market question is CLOSED. Fourteen
> rounds, three candidates taken to conclusion this session, all rejected, and
> a clean 2026 forward test confirming no exploitable edge. Do not reopen it
> without new data — see [issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md).

## Blocked on Pete

1. **~~The machine is out of memory~~ — cleared by 2026-09-06 13:20 (6.4 GB free, no reboot recorded). Kept for the measurement lessons.** Original text follows.
   **The machine is out of memory, and it is killing model runs.** Available
   0.3-1.6 GB of 31.5 GB; committed 70 GB against a 114.7 GB limit, with only
   ~10 GB resident — the box is thrashing. Three model runs were killed
   mid-flight on 2026-09-05. Largest consumer is a **single `WindowsTerminal`
   process holding 15.03 GB of paged-out scrollback**; 20 `claude` processes add
   8.3 GB.
   **CORRECTION — do NOT expect "Clear Buffer" to fix this.** That advice was
   given twice from this session and was wrong. `WindowsTerminal` holds
   15,396 MB **committed** but only **307 MB resident**; commit is reserved
   address space, most of it paged out at no cost in RAM. Clearing the
   scrollback frees pagefile, not physical memory — roughly 0.3 GB of relief,
   not 15 GB. The same correction applies to the 20 `claude` processes
   (8,620 MB committed, 2,439 MB resident). Caught by BOUNCER's session
   reconciling the parts against the whole, and verified here.

   **The real picture is that nothing explains it.** Installed 31.5 GB,
   available 0.05 GB, committed 70.6 GB, and the resident totals do not add up:
   all 560 processes together hold 9.5 GB resident, pools 3.2 GB, modified list
   0.84 GB — about **18 GB is unattributed to any process, pool or cache**.
   No user process accounts for the starvation, so no amount of closing things
   will reliably fix it. Unattributed memory at that scale is often
   driver-locked allocation, which a reboot clears; that is a hypothesis, not a
   finding, and a reboot is Pete's call and not available while he is on remote
   connect from a phone.

   **Measurement trap that produced the wrong picture:** rank by
   `WorkingSet64`, not the deprecated Int32 `WorkingSet`, and never sum the
   latter — `Memory Compression` reports a negative value that drags the total
   below zero and makes a large process look like it is hiding.

   **Largely superseded the same day — see the harness note below.** The
   "running low on memory" kills came from Claude Code's guard on *tracked*
   background tasks, not the Windows OOM killer. A detached `Start-Process`
   survives, so this stopped blocking dropshotverse before it was resolved.

   **Correction (2026-09-05):** the R jobs were misattributed here on first
   writing. `backtest_athletics.R` (PID 17712, 8.14 GB) was citiusverse's and
   that session had already killed it — it had stopped computing and started
   thrashing. `backtest_candidate_fed.R` is **auspol's**
   (`C:\dev\auspol\scripts\`), not citiusverse's; it was attributed by
   proximity in the process list, which was wrong.

2. **[PR #8](https://github.com/peteowen1/dropshotverse/pull/8) is open, `dev` -> `main`, reviewed, awaiting Pete's merge** (opened 2026-09-06 after a two-reviewer Sonnet pass; merging closes #6 and #7). Original blocker text follows.
   **30 commits on `dev`, no PR ever opened on this repo** (was 10 when first written). Same review-gate
   conflict as row 1 of the cross-verse dashboard: `~/.claude/CLAUDE.md` requires
   a reviewer-agent pass before any PR, and sessions have been running with
   agents disabled. `main` has never received anything since it was created
   2026-08-31. Includes the antisymmetry fix (#7), the 6-6 tiebreak fix, and
   pairwise publishing (#6).

3. ~~**#3 canonical base archive**~~ — **DECIDED and closed 2026-09-06.** Keep
   the frozen Sackmann archive as base, TML as live bridge. TML has ATP from
   1968 but WTA only from 1990, and the archive holds 61,840 tour-level WTA
   matches from 1968-1989 — losing them is harmless for current-form ratings
   and disqualifying for the all-time GOAT list. D-11 in
   [DECISIONS.md](DECISIONS.md).

4. **[#4](https://github.com/peteowen1/dropshotverse/issues/4) — move the blog
   refresh off the local scheduled task to GitHub Actions.** Needs Cloudflare
   repo secrets, which only Pete can create. Note `load_matches()` still globs
   local CSVs and needs a branch to read the release parquet first.

## Open queue

5. ~~**Finish the K-schedule grid**~~ / ~~**Decide round 13 downstream**~~ /
   ~~**The 2026 forward test**~~ — **all three CLOSED 2026-09-05**, left here
   by mistake on 09-06. The grid finished (commit `3869484`, it is a ridge);
   both Elo wins were tested downstream (`3cc4bda`: retirements die, K survives
   5/5); the 2026 forward test then killed the K change out of sample
   (`722066d`, +0.000029). Net: nothing from rounds 12-14 ships. Detail under
   "Recently landed" below.

8. **[#1](https://github.com/peteowen1/dropshotverse/issues/1) — hier
   tail-calibration bug — DOWNGRADED 2026-09-06 (D-12).** Tested directly: the
   shipped XGBoost does not inherit the tail (winners under 1%: 0.00% vs hier's
   0.71% on the same rows) and needs the feature (+0.0092 without it, 5/5
   folds). The standalone defect is real and unfixed but blocks nothing
   shipped. Only worth picking up as "would a fixed hier make a better
   feature", which is an upside experiment.

9. **[#5](https://github.com/peteowen1/dropshotverse/issues/5)** — publish
   match-pairing/fixture data so the blog can build a real bracket.
   **[#2](https://github.com/peteowen1/dropshotverse/issues/2)** — the WTA public
   live-scores API (`api.wtatennis.com`) overturns ticket 03's "WTA is walled".
   Its in-play fields have still never been observed populating.

10. **Forecasts for events after the US Open are not buildable yet.** The
    simulator handles an unknown draw (`simulate_tournament.py` generates
    seeding-consistent random brackets), so a field list is enough — but there
    is no generic entry-list source. The US Open entrants were scraped from
    `usopen.org` specifically.

## Live deadlines (absolute dates — you are reading this on a different day)

- **2026-09-13** — US Open men's final. Both scheduled tasks now run to
  **2026-09-16 12:00 local = 2026-09-15 22:00 New York**. Before 09-06 they
  ended at 2026-09-14T00:00 *local*, which is 2026-09-13 **10:00 in New York** —
  four hours before the men's final starts. Both the live projection and the
  point capture would have frozen mid-event with nothing alarming.
- **~2026-09-23** — re-run `fetch_wta_upcoming.py --fetch 1020` for the China
  Open (Beijing, starts 2026-09-30). Today the field is 76 of a 96 draw and
  **0 seeded**; qualifiers, late entries and seeding all land nearer the event.
  A forecast run now would be over a partial, unseeded field.
- **2026-09-16** — after the tasks expire, R2 stops refreshing. `event_status`
  now marks a finished event so the site can render a result rather than a
  frozen forecast, and `check_blog_data_freshness.py` fails on stale-while-live.
  **Nothing schedules that check yet** — it changes nothing on its own.

## Recently landed (2026-09-06)

- **The market question is CLOSED.** Rounds 12/13/14 all rejected; the 2026
  forward test (2,654 untouched matches, Jan-May) put the model at 0.59638
  against the market's 0.57168, and 38.0% right when it dissents (CI
  [32.3, 43.8]) — independently reproducing ticket 28 round 7's 44.6% on a
  different year, feature set and benchmark.
- **Round 14 is the cautionary tale.** The K-schedule change passed every
  tune-window check available, including the correct paired per-fold test
  (-0.001393, 5/5 folds, t=-5.70), and came back **+0.000029** out of sample.
  Fold consistency removes fold noise; it does nothing about having selected
  the point on those same folds.
- **WTA upcoming forecasting works** (`fetch_wta_upcoming.py`): real calendar,
  entry lists weeks ahead, 100% entrant resolution across four events.
- **In-match WP has its first calibrated result** (ticket 37): Mode A confirmed
  at scale (ECE 0.0471) over Mode B (0.0737), closing ticket 09's open question.
- **`usopen_2026_points.parquet`** — 79,917 points, 503 singles matches, three
  anchors passing.
- **Issue #3 decided and closed**; crosswalk scoped and found not to be the
  bottleneck for any of the three things said to block on it (ticket 38).

## Known and NOT chased

- **`check_blog_data_freshness.py` still runs from nothing (found by the
  2026-09-06 review).** It reads the local parquet the sim just wrote, so
  putting it in `refresh_usopen_live.cmd` would always pass. It needs its own
  schedule AND somewhere to report to; a scheduled task that exits 1 into a
  log nobody reads is not an alarm. The honest home is the ops health-check
  workflow reading the public R2 URL, which is #4 territory and needs Pete's
  Cloudflare secrets.
- **~~`DropshotverseUSOpenCapture` re-runs a full backfill every 5 minutes~~ —
  diagnosis was wrong, real defect FIXED 2026-09-06.** The backfill ran only on
  the `:04` UTC tick each hour, but took ~60 min (598 matches x 6 feeds x 0.7 s
  courtesy delay), and the task's `MultipleInstances=IgnoreNew` meant runs never
  overlapped: instead every 5-minute live sweep during that hour was dropped.
  Live point capture was blind roughly every other hour (log: backfills at 12,
  14, 16, 17, 19, 20, 22, 00, 02, 04 UTC). `backfill()` is now incremental via
  `reference/backfill_done.json` (seeded with 593 settled ids); a pass takes
  ~50 s. `capture_usopen.py backfill --full` forces the old behaviour. The
  intermittent exit 1 is still unexplained and still not chased.
- **The surface rating is not a self-correcting rating.**
  `elo_baseline.py:409` updates it with the blended `p_w`, which at `blend=0.0`
  comes entirely from OVERALL Elo — so it accumulates overall-Elo residuals
  bucketed by surface, with no feedback loop, drifting 2.8x wider (std 498 vs
  180). It never reaches Elo's own prediction, but `elo_surface_diff` **is** an
  XGBoost feature. Reproduced rather than changed; whether it should use `p_s`
  is untested and is a real open question.
- **Config D forward test** — would convert the +0.0247 upper bound into the
  shipped model's actual gap. Deliberately skipped: three measurements agree
  and a model right 38% of the time when it dissents has no edge worth sizing
  more precisely.

## Older (2026-09-05)

- **Round 12 — retirement as a pre-match fitness feature: REJECTED**, after five
  phases. Phases 1-3 all produced statistically strong, correctly-signed,
  anchor-passing results that were confounds (between-player, then within-player,
  then within-career). A placebo-event arm settled it: re-running the event study
  on random dates in the same players' careers reproduced almost the entire
  profile. Net of placebo the only real effect is **+0.0285 in the 30 days after
  retiring** — players do *better*, which is withdrawal selection.
- **Round 13 — scoring retirements in Elo: real, -0.00044**, axis decomposed.
- **Round 14 — K schedule: 14 of 27 points, provisionally null.** The Sackmann
  defaults had never been tuned in this project; a grep for `K_BASE`/`K_SHAPE`/
  `K_MIN` across every script returns only the two written that day.
- **`tune_scratch/elo_compact.py`** — memory-cheap Elo replay. 409 MB peak
  instead of several GB, ~37 s per arm instead of ~5 min, verified
  **bit-identical** to `elo_baseline.py` (max |delta| 0.000e+00 on 89,678 rows).
  The old screens were heavy because they held 1.23M rows as a DataFrame with
  string uids and keyed Python dicts on them. Every arm checkpoints to parquet,
  which is why the third OS kill cost one arm rather than the run.
