# dropshotverse — Next Steps

> Living queue. Sessions update this when they finish work (rule in `C:\dev\CLAUDE.md`).
> Decision log: [DECISIONS.md](DECISIONS.md) · Vault map: [HOME.md](HOME.md)
> Created 2026-09-05 — this verse had no docs hub until then, and appears in
> neither `C:\dev\ECOSYSTEM.md` nor `C:\dev\vault\NEXT-STEPS.md`.
> Last updated: 2026-09-05 — rounds 12/13/14 of the chase-the-market programme.

## Blocked on Pete

1. **The machine is out of memory, and it is killing model runs.** Available
   0.3-1.6 GB of 31.5 GB; committed 70 GB against a 114.7 GB limit, with only
   ~10 GB resident — the box is thrashing. Three model runs were killed
   mid-flight on 2026-09-05. Largest consumer is a **single `WindowsTerminal`
   process holding 15.03 GB of paged-out scrollback**; 20 `claude` processes add
   8.3 GB; four `Rscript` jobs (citiusverse `backtest_athletics.R`, started
   08:21) add 3-8 GB.
   **Non-destructive fix: Ctrl+Shift+P → "Clear Buffer"** (or Ctrl+Shift+K) in
   the terminal — frees the scrollback without ending the session. Killing the
   citius backtest is Pete's call; it may be a wanted run.

2. **10 commits on `dev`, no PR ever opened on this repo.** Same review-gate
   conflict as row 1 of the cross-verse dashboard: `~/.claude/CLAUDE.md` requires
   a reviewer-agent pass before any PR, and sessions have been running with
   agents disabled. `main` has never received anything since it was created
   2026-08-31. Includes the antisymmetry fix (#7), the 6-6 tiebreak fix, and
   pairwise publishing (#6).

3. **[#3](https://github.com/peteowen1/dropshotverse/issues/3) — canonical base
   archive: stay Sackmann-frozen + bridge, or move to TML?** The 50-year
   player-id crosswalk is the real work (the current bridge covers ~216 recent
   players per tour). Everything about data freshness past 2026-08-24 depends on
   this.

4. **[#4](https://github.com/peteowen1/dropshotverse/issues/4) — move the blog
   refresh off the local scheduled task to GitHub Actions.** Needs Cloudflare
   repo secrets, which only Pete can create. Note `load_matches()` still globs
   local CSVs and needs a branch to read the release parquet first.

## Open queue

5. **Finish the K-schedule grid — 13 of 27 points never ran** (killed for
   memory). Missing: all of `k_shape=0.5` (the faster-decay direction),
   `k_base=350 / k_shape=0.4`, and `k_base=250 / 0.4 / k_min=24`. 14 points are
   in and every one is worse than Sackmann's inherited `(250, 0.4, 16)`, but
   calling a null before testing `k_shape=0.5` would be reporting an untested
   region as evidence. ~8 minutes once memory allows.
   Detail: [issues/35](../.scratch/dropshot-tennis-engine/issues/35-fitness-and-retirement-scoring.md).

6. **Decide round 13 (scoring retirements in Elo) — needs the downstream test.**
   Settled on Elo: scoring RET rows at `ret_update=0.5` is worth **-0.00044**
   log-loss on the 2022-23 tune window, and the decomposition proved the rating
   delta carries it (counter-only is +0.000025, nothing). Unknown whether it
   survives into the shipped model: XGBoost consumes Elo as one feature of 19
   and has already absorbed most of its signal (ticket 28 round 8). Requires
   `elo_raw_ratings_full.parquet` regenerated under the new rule, features
   rebuilt, then a tune-window `xgb.cv`.
   **Independent of the market gap this is a correctness question** — 2.7% of
   tour results currently move nobody's rating in either direction, and those
   ratings are published to R2.

7. **The 2026 forward test — the one clean number nobody can call overfit.**
   The 2024-25 market holdout has been revealed at least a dozen times across
   tickets 27/28 and is no longer an unbiased estimate. 2026 has never been
   touched by any decision: `raw/odds/{atp,wta}/2026.xlsx` hold 1,976 and 1,921
   matches with Pinnacle `PSW`/`PSL` (Jan-Aug), and the archive+bridge covers
   the results. Also re-measure disagreement accuracy on config D — ticket 28
   round 7 found 44.6% (CI [41.5, 47.7]) at v2, and log-loss parity alongside
   sub-chance disagreement accuracy is a contradiction that should be resolved
   before any "we match the market" claim goes out.
   **Pete's call 2026-09-05: keep improving the model first.**

8. **[#1](https://github.com/peteowen1/dropshotverse/issues/1) — hier
   tail-calibration bug** makes `hierarchical_model.py` score worse than random
   in its own standalone predictions. Ticket 31 corrected the earlier diagnosis:
   a refit changes nothing and neither does XGBoost. Not established whether the
   defect contaminates `hier_logit_pwinner` as an XGBoost input or is absorbed
   by the downstream fit — never tested directly.

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

## Recently landed (2026-09-05)

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
