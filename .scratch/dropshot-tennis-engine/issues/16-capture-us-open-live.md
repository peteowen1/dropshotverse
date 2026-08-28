# 16 — Capture the US Open point stream while it is running

Type: task
Status: running

## Progress — capture live as of 2026-08-27

**Harness built and scheduled.** Script: `dropshotdata/scripts/capture_usopen.py`.

**Scheduled task `DropshotverseUSOpenCapture`** runs `pythonw.exe capture_usopen.py once` every **5 minutes**, indefinitely. Verified running: `LastTaskResult: 0`. Settings: starts on battery, doesn't stop on battery change, `StartWhenAvailable` so a missed window catches up after sleep, `MultipleInstances IgnoreNew` so a slow run never stacks, 20-minute execution cap.

To inspect or stop it:
```powershell
Get-ScheduledTaskInfo -TaskName DropshotverseUSOpenCapture
Unregister-ScheduledTask -TaskName DropshotverseUSOpenCapture -Confirm:$false
```

**Timing is far better than assumed.** `eventDays.json` shows Day 15 is **Sunday 13 September**, so the sample captured earlier was Fan Week mixed doubles and the main draw has not started. This captures the **entire** 2026 US Open, not the tail of it. Backfill accordingly finds almost nothing — the value is the next 17 days of polling.

**Design decisions worth keeping:**
- **Content-addressed writes** (`<feed>.<epoch>.<sha256-12>.json`). A tape grows point by point, so every distinct version is kept and a match can be replayed exactly as it unfolded. Repeated runs are idempotent, which is what makes a 5-minute scheduled task safe with no state to manage.
- **`once` mode, not a daemon.** The scheduler owns cadence; a crash costs one interval; reboots are a non-event.
- **Never transform on ingest.** These feeds are tournament-scoped and vanish between events. A parser bug must be fixable without a re-fetch we may not be able to make.
- **Day URLs come from the feed, never derived.** The path number is `tournDay` (which counts qualifying days), so Day 15 lives at `day_22.json`. Deriving it would silently fetch the wrong days.

**Five feeds captured per match:** `history` (the point tape), `complete`, `stats`, `distance` (Hawk-Eye distance-run, found in the catalogue and not in the original ticket), `insight`.

**Reference feeds captured:** `config_web` (the catalogue itself), `draws`, `players`, `competitors`, `seeds`, `event_days`, `schedule_days`, plus a per-day match index.

**State at handoff:** 127 files, 9 MB, 33 matches.

### Still to do on this ticket

- **Verify the Hawk-Eye fields on a men's or women's SINGLES match** once the main draw starts. `ServeWidth`, `ServeDepth`, `ReturnDepth`, `KickHeight` and `DistanceRun` were all empty in the doubles sample. If they populate for singles on tracked show courts, that is shot-level data for free.
- **Check whether the feed's `atp*`/`wta*` player ids map to the archive's ids** — could remove most of ticket 07a's crosswalk work.
- **Check Wimbledon's equivalent for past years** — same IBM architecture reportedly serves historical tournaments off-season, which would mean recoverable point data for past Slams.
- **Write a coverage log** of matches captured versus played.
- **Replace with a GitHub Action** once `dropshotdata` exists as a repo; the scheduled task is the version that works without one.

## Question

**Time-critical.** The US Open is in progress *now* (verified 2026-08-27: a live mixed-doubles semi-final returned 96 points of live data). The next Slam is the Australian Open in January. If we do not capture this fortnight's point streams, the opportunity is gone for five months.

Ticket 03's conclusion — that no free live tennis source is usable — was **wrong**, and this ticket exists because of the correction (see ticket 17).

## What was verified, 2026-08-27

No auth, no key, no token handshake. Only a browser `User-Agent`.

- **Feed catalogue:** `https://www.usopen.org/en_US/json/gen/config_web.json` — HTTP 200, 82 KB, lists every feed URL IBM's own frontend uses.
- **Live scores:** `https://www.usopen.org/en_US/scores/feeds/2026/matches/live/scores.json` — HTTP 200. Returns in-progress matches with `match_id`, court, round, `server`, `last_serve_speed`, set/game scores, and player IDs already in **`atp*`/`wta*` form** (e.g. `wta322191`, `atpm0ni`) — which may solve a chunk of the ticket-07a player-identity problem for free.
- **Point-by-point:** `https://www.usopen.org/en_US/scores/feeds/2026/slamtracker/history/<match_id>C.json` — HTTP 200, 160 KB for a match in its second set. `config_web.json` gives this a `rateSec: 1` refresh hint.

**57 fields per point.** Confirmed present and populated: `PointNumber`, `PointWinner`, `PointServer`, `ServeNumber`, `ServeIndicator`, `SetNo`, `GameNo`, `P1Score`/`P2Score`, `P1GamesWon`/`P2GamesWon`, `P1SetsWon`/`P2SetsWon`, `Ace`, `DoubleFault`, `Winner`, `WinnerShotType`, `UnforcedError`, `NetPoint`, `BreakPoint`, `BreakPointWon`, `BreakPointOpportunity`, `RallyCount`, `Rally20OrMore`, `Deuces5OrMore`, `Speed_KMH`/`Speed_MPH`, `ElapsedTime`, `EpochTimeStart`/`EpochTimeEnd`, `P1Momentum`/`P2Momentum`, and a plain-English `Sentence`.

**Empty in the sample:** `ServeWidth`, `ServeDepth`, `ReturnDepth`, `KickHeight`, and `P1DistanceRun`/`P2DistanceRun` (all zeros). These are the Hawk-Eye-derived fields. The sample was **doubles**, so they may populate for singles on tracked show courts — **verify against a singles match before claiming we have them.**

## The task

1. **Write a poller.** Read `live/scores.json` on a sensible interval (30-60s is ample — `rateSec: 1` is for a UI, and we are not one), diff for in-progress matches, and pull each match's `slamtracker/history/<id>C.json`. Be a good citizen: a real `User-Agent`, conditional requests where possible, back off on errors, never hammer.
2. **Store raw and immutable.** Write each fetch verbatim to `dropshotdata/raw/usopen/2026/<match_id>/<epoch>.json`. Never transform on ingest — the tape is the evidence.
3. **Capture the full draw**, not just show courts, and both singles and doubles. Log which matches were missed so coverage is known rather than assumed.
4. **Verify the Hawk-Eye fields on a singles match** and record the answer in this ticket. If `ServeWidth`/`ServeDepth`/`RallyCount` populate for singles, that is shot-level data for free and materially changes what the model can use.
5. **Snapshot `config_web.json` too** — it is the map of every other feed and may change between tournaments.
6. **Check `wimbledon.com`'s equivalent for past years** while building. The same IBM architecture reportedly serves historical tournaments off-season, which would mean recoverable point data for past Slams, not just this one.

## Constraints

- Read-only polling of a public JSON endpoint. **Do not** attempt to bypass any bot protection, and do not touch `atptour.com`, which sits behind Cloudflare.
- Do **not** republish raw feed content. Derived probabilities only — consistent with the source-separation decision on the map.
- This is capture only. No modelling, no transformation, no publishing.

**Deliverable:** the raw captured tapes, a coverage log of matches captured versus played, and a verdict on the Hawk-Eye fields for singles.
