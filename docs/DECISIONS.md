# dropshotverse — Decisions

> Work queue: [NEXT-STEPS.md](NEXT-STEPS.md) · Vault map: [HOME.md](HOME.md)
> Created 2026-09-05, back-filled from `.scratch/dropshot-tennis-engine/`.

## Awaiting Pete

**D-1 — canonical base archive.** Stay on the frozen Sackmann snapshot plus the
TML bridge, or migrate to TML wholesale?
[#3](https://github.com/peteowen1/dropshotverse/issues/3), open since
2026-08-31. Required first either way: overlap-years diff, WTA-depth check,
corruption screening, and a **50-year player-id crosswalk** — the current bridge
resolves only ~216 recent players per tour. Everything downstream of
2026-08-24 depends on this.

**D-2 — the review gate.** 10 commits sit on `dev`; no PR has ever been opened
on this repo. `~/.claude/CLAUDE.md` requires a reviewer-agent pass before a PR
and sessions have been running with agents disabled. This is row 1 of the
cross-verse dashboard, not a dropshotverse-specific problem, but this verse is
now one of the repos it blocks.

**D-3 — machine memory.** Three model runs killed on 2026-09-05. The
non-destructive fix (clear the terminal scrollback buffer) is Pete's to press;
so is any decision about the citiusverse `backtest_athletics.R` job.

**D-4 — publish a "we match the market" claim?** Not yet, and this is a
recommendation rather than a question: the 0.00073 gap comes from a holdout
revealed a dozen times, and ticket 28 round 7's disagreement-accuracy finding
(44.6%, CI [41.5, 47.7]) has never been re-measured on the shipped config D.
Resolve both before anything goes on the site as a claim.

## Made

**D-5 — 2026-08-27: the destination was revised on measured evidence, from
"beat the closing line" to "calibration parity with the market".** Ticket 10
measured the gap: surface-adjusted Elo trailed Pinnacle by 0.026 log-loss and,
where it disagreed with the market, was right only 40.5% of the time. Ingram's
published point-level model trailed FiveThirtyEight's Elo by 0.006, so a
point model of comparable quality would still have lost by a wide margin.
Redrawn once, on evidence, at the moment the evidence arrived — rather than
quietly redefining success months later.

**D-6 — 2026-08-29: Pete reopened "beat the market" anyway,** explicitly, via
`/goal`. This reversed D-5's scope closure at his direct request. Eleven rounds
of candidates followed; the gap closed from 0.0145 to 0.00073 and WTA now beats
the market on the holdout (0.58779 vs 0.58852), while ATP still trails by
0.00214 after ten separate ATP-specific levers were tested and rejected.

**D-7 — 2026-08-31: the frozen Sackmann archive was published as a GitHub
Release.** Sackmann's repos went 404 on 2026-08-27 mid-project; the snapshot
existed as a single copy on one laptop until this.

**D-8 — 2026-08-31: WTA data may go on the ITG site.** Reverses the earlier
position; inthegame-blog#581 previously said the opposite and was corrected.
Prompted by discovering that `api.wtatennis.com` is open and unauthenticated,
which overturned ticket 03's "WTA is walled" conclusion — found by reading the
site's own network traffic. ATP genuinely is walled (server-rendered Sitecore,
no client API), so the asymmetry is real but runs opposite to what the project
assumed for four days.

**D-9 — 2026-09-05: keep improving the model before running the 2026 forward
test.** Pete's call, made after being shown that the 2024-25 holdout is
contaminated and that a clean 2026 population is sitting unused. Recorded
because it is a deliberate trade: more candidate rounds on a burnt holdout,
with the unbiased measurement deferred.

**D-10 — 2026-09-05: the 2024-25 holdout is retired from candidate screening.**
All of rounds 12/13/14 screened on the 2022-23 tune window and did not score the
holdout once. `RESUME-beat-elo.md`'s own round-11 stop-point flag is now
policy: 2026 is reserved as the final unbiased test.

## Standing corrections

- **`RESUME-beat-elo.md` is stale from round 11 onward.** It still quotes a
  0.0127 gap to market; later work closed that to 0.00073. Trust
  `reports/xgboost-shipped.md`.
- **The archive does not end at Roland Garros.** The CSV filenames stop at
  2026-06-02, but the TML bridge carries tour-level matches to 2026-08-24. A
  session reading only the file listing will conclude the ratings are three
  months stale. They are not.
- **`slot` in the entrants files is not draw position** (2/128 agreement with
  the real draw). Building a bracket on it produces a completely wrong result
  that still looks plausible.
