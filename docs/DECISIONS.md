# dropshotverse — Decisions

> Work queue: [NEXT-STEPS.md](NEXT-STEPS.md) · Vault map: [HOME.md](HOME.md)
> Created 2026-09-05, back-filled from `.scratch/dropshot-tennis-engine/`.
> A row is the decision, the one fact that drove it, and a link to the evidence.

## Awaiting Pete

**D-4 — publish a "we match the market" claim?** Not yet, and this is a
recommendation rather than a question: the 0.00073 gap comes from a holdout
revealed a dozen times, and the 2026 forward test puts the honest gap at
0.0247 with the model right 38% of the time when it dissents
([issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md)).

## Made

**D-1 — canonical base archive.** Answered by D-11; issue #3 closed 2026-09-06.

**D-2 — the review gate.** Resolved 2026-09-06: PRs #8-#12 merged, each after
a Sonnet reviewer pass. `main` had received nothing since 2026-08-31.

**D-3 — machine memory.** Cleared 2026-09-06 without a reboot. The kills were
Claude Code's guard on tracked tasks, not the OS. Narrative in
[backlog/journal-2026-09-05.md](backlog/journal-2026-09-05.md).

**D-5 — 2026-08-27: the destination was revised on measured evidence, from
"beat the closing line" to "calibration parity with the market".** Ticket 10
measured the gap: surface-adjusted Elo trailed Pinnacle by 0.026 log-loss and,
where it disagreed with the market, was right only 40.5% of the time. Ingram's
published point-level model trailed FiveThirtyEight's Elo by 0.006, so a
point model of comparable quality would still have lost by a wide margin.

**D-6 — 2026-08-29: Pete reopened "beat the market" anyway,** explicitly, via
`/goal`. Eleven rounds of candidates followed; the gap closed from 0.0145 to
0.00073 on the holdout and WTA beat the market there (0.58779 vs 0.58852),
while ATP still trailed by 0.00214. Superseded by D-10 and the forward test.

**D-7 — 2026-08-31: the frozen Sackmann archive was published as a GitHub
Release.** Sackmann's repos went 404 on 2026-08-27 mid-project; the snapshot
existed as a single copy on one laptop until this.

**D-8 — 2026-08-31: WTA data may go on the ITG site.** Prompted by discovering
that `api.wtatennis.com` is open and unauthenticated, which overturned ticket
03's "WTA is walled" conclusion. ATP genuinely is walled (server-rendered
Sitecore, no client API); the asymmetry runs opposite to what the project
assumed for four days.

**D-9 — 2026-09-05: keep improving the model before running the 2026 forward
test.** Pete's call, a deliberate trade: more candidate rounds on a burnt
holdout with the unbiased measurement deferred. The forward test ran the next
day anyway once rounds 12-14 were exhausted.

**D-10 — 2026-09-05: the 2024-25 holdout is retired from candidate screening.**
Revealed at least a dozen times; 2026 was reserved as the final unbiased test
and has now been spent ([issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md)).

**D-11 — 2026-09-06: keep the frozen Sackmann archive as the historical base
and TML as the live bridge.** TML serves WTA only from 1990; the archive holds
61,840 tour-level WTA matches from 1968-1989, which the all-time GOAT list
needs. The crosswalk was never the gate. Detail:
[issues/39](../.scratch/dropshot-tennis-engine/issues/39-tml-depth-decides-issue-3.md),
[issues/38](../.scratch/dropshot-tennis-engine/issues/38-crosswalk-scoping.md).

**D-12 — 2026-09-06: issue #1's hier tail bug is not a live-pipeline problem;
`hier_logit_pwinner` stays.** On identical holdout rows the standalone model
gives the winner under 1% in 0.71% of matches, the shipped bag in 0.00%; and
dropping the feature costs +0.0092 log-loss on 5/5 paired folds. Tables on
[#1](https://github.com/peteowen1/dropshotverse/issues/1),
script `tune_scratch/screen_hier_contamination.py`.

**D-13 — 2026-09-06: the live projection publishes from GitHub Actions, on a
Windows runner, from two Releases.** Windows because every script hard-codes
`C:\dev\dropshotverse\...` and a junction resolves it; the derived inputs
became the `model-inputs-2026-09` Release; the R2 token reused is the laptop's
(Pete's call over minting an R2-only one). Workflow header comments and
[#4](https://github.com/peteowen1/dropshotverse/issues/4) carry the detail.

## Standing corrections

- **`slot` in the entrants files is not draw position** (2/128 agreement with
  the real draw). Building a bracket on it produces a completely wrong result
  that still looks plausible.
- **Ticket 01's "TennisMyLife's manifest has 49 ATP challenger files and zero
  WTA" is wrong**, and shaped scope from 2026-08-27 to 2026-09-06. TML serves
  WTA for every year from 1990; `bridge_tennismylife.py` had been pulling
  `2026_wta.csv` the whole time. The real limitation is no WTA *before* 1990.
- Two more (the RESUME log is stale from round 11; the archive does not end at
  Roland Garros) live in [HOME.md](HOME.md), where a cold session reads them.
