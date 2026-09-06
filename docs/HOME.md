# dropshotverse — Vault Home

Verse-level documentation for Pete's tennis analytics effort: pre-match win
probability, tournament draw simulation, and live in-match win probability for
ATP and WTA. Started 2026-08-27.

## The hubs

- [NEXT-STEPS.md](NEXT-STEPS.md) — living work queue.
- [DECISIONS.md](DECISIONS.md) — decision log: awaiting Pete, and made.
- [plans/](plans/) — design work; [backlog/](backlog/) — narrative rolled off the queue.
- The research record is in [`.scratch/dropshot-tennis-engine/`](../.scratch/dropshot-tennis-engine/), below.

## Git topology — a FOURTH shape, not in the C:\dev\CLAUDE.md table

> **dropshotverse is a SINGLE git repo.** `dropshotdata/` is an ordinary
> directory inside it, not a sub-repo and not a submodule. Commit everything
> from the verse root; there is no parent-pointer step and no `git -C` dance.

## Where the work actually is

| Path | What |
|---|---|
| `dropshotdata/scripts/` | the pipeline — Elo, hierarchical point model, XGBoost, simulators, publishers, capture |
| `dropshotdata/tune_scratch/` | ~100 experiment screens plus the derived inputs the pipeline reads. The research record; most screens are one-shot |
| `dropshotdata/raw/github/` | the frozen Sackmann archive (see below) |
| `dropshotdata/raw/odds/` | tennis-data.co.uk Pinnacle closing odds, 2018-2026, ATP + WTA |
| `dropshotdata/raw/usopen/` | point-by-point capture of the 2026 US Open |
| `dropshotdata/reference/` | committed inputs: US Open entrants lists |
| `dropshotdata/blog/` | what gets published to R2 for inthegame-blog |
| `.github/workflows/` | `live-projection.yml` (producer) and `check-freshness.yml` (alarm) |
| `.scratch/dropshot-tennis-engine/` | tickets, reports, the charter, session logs |

## The research record (`.scratch/dropshot-tennis-engine/`)

- **`reports/`** — one report per model (`elo-baseline.md`,
  `hierarchical-model.md`, `xgboost-shipped.md`). **Start here** for what the
  models are and what they score. The shipped report's 0.00073 gap is the
  burnt-holdout number; the honest gap is the 2026 forward test's 0.0247
  ([issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md)).
- **`issues/`** — 39 numbered tickets, the real detail.
- **`map.md`** — the charter from 2026-08-27, with a status banner.
- **`RESUME-beat-elo.md`** — the chase-the-market log. **Stale from round 11
  onward**; it still quotes a 0.0127 gap. Read its banner, not its prose.

## Data supply — read this before trusting a date range

Jeff Sackmann's GitHub repos **went 404 on 2026-08-27**, mid-project. What
exists is a frozen snapshot taken that day, published as the
`sackmann-archive` Release. Archive coverage ends **2026-06-02**.

A live bridge to `stats.tennismylife.org` (`scripts/bridge_tennismylife.py`)
downloads the current season's tour-level matches on every run, so ratings are
current to the last completed week, not stale at Roland Garros as the CSV
filenames suggest. TML carries `score` (retirements detectable) but not
`minutes`, and no WTA before 1990, which is why the archive stays the base
(D-11; [#3](https://github.com/peteowen1/dropshotverse/issues/3) closed).

## Releases (the data bus)

Two GitHub Releases carry everything not in git: `sackmann-archive` (the raw
CSVs) and `model-inputs-2026-09` (Elo replays and the shipped model). Asset
list and the fresh-clone commands: [`dropshotdata/docs/README.md`](../dropshotdata/docs/README.md).

## What is live

- **US Open 2026 projections** refresh to R2 every 30 minutes from the laptop
  task `DropshotverseUSOpenLiveSim`, with `live-projection.yml` on GitHub
  Actions as a second producer (D-13). Actions works end to end but GitHub's
  cron fired once in two hours on 2026-09-06, so the laptop stays primary
  through the final. See NEXT-STEPS "Awaiting Pete".
- **Freshness alarm**: `check-freshness.yml` reads the public R2 copy and
  fails a run, which mails Pete, if the file is stale while an event is live.
  It rides the same sporadic scheduler, so a quiet hour is not proof of health.
- **Point-by-point capture**: laptop task `DropshotverseUSOpenCapture`, every
  5 minutes to 2026-09-16 12:00 local.
- **The ITG wall chart** — [inthegame-blog#581](https://github.com/peteowen1/inthegame-blog/issues/581),
  shipped 2026-09-01, runs on dropshot's own pairwise match model.
