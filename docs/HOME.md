# dropshotverse — Vault Home

Verse-level documentation for Pete's tennis analytics effort: pre-match win
probability, tournament draw simulation, and live in-match win probability for
ATP and WTA. Started 2026-08-27.

## The three hubs

- [NEXT-STEPS.md](NEXT-STEPS.md) — living work queue. Sessions update this when they finish work.
- [DECISIONS.md](DECISIONS.md) — decision log: awaiting Pete, and made.
- The research record lives in [`.scratch/dropshot-tennis-engine/`](../.scratch/dropshot-tennis-engine/) — see below.

## Git topology — a FOURTH shape, not in the C:\dev\CLAUDE.md table

`C:\dev\CLAUDE.md` documents three verse topologies and calls the table "the
trap". dropshotverse is none of them:

> **dropshotverse is a SINGLE git repo.** `dropshotdata/` is an ordinary
> directory inside it, not a sub-repo and not a submodule. There is no
> `dropshotdata/.git`. Commit everything from the verse root; there is no
> parent-pointer step and no `git -C` dance.

Verified 2026-09-05 (`git -C dropshotverse/dropshotdata rev-parse --show-toplevel`
returns the verse root).

## Where the work actually is

| Path | What |
|---|---|
| `dropshotdata/scripts/` | the pipeline — Elo, hierarchical point model, XGBoost, simulators, publishers |
| `dropshotdata/tune_scratch/` | ~100 experiment screens. The research record; most are one-shot |
| `dropshotdata/raw/github/` | the frozen Sackmann archive (see below) |
| `dropshotdata/raw/odds/` | tennis-data.co.uk Pinnacle closing odds, 2018-2026, ATP + WTA |
| `dropshotdata/blog/` | what gets published to R2 for inthegame-blog |
| `.scratch/dropshot-tennis-engine/` | tickets, reports, the map, and the session log |

## The research record (`.scratch/dropshot-tennis-engine/`)

- **`map.md`** — the charter: destination, scope, and what was ruled out. Revised
  2026-08-27 on measured evidence.
- **`RESUME-beat-elo.md`** — the running log of the chase-the-market programme.
  **Stale from round 11 onward** — it still quotes a 0.0127 gap that later work
  closed to 0.00073. Trust `reports/xgboost-shipped.md` over it.
- **`issues/`** — 35 numbered tickets, the real detail.
- **`reports/`** — one report per model (`elo-baseline.md`,
  `hierarchical-model.md`, `xgboost-shipped.md`). **Start here** for what the
  models are and what they score.

## Data supply — read this before trusting a date range

Jeff Sackmann's GitHub repos **went 404 on 2026-08-27**, mid-project. What
exists is a frozen snapshot taken that day, published as a GitHub Release so it
is not a single copy on one laptop.

- Archive coverage ends **2026-06-02**.
- A live bridge to `stats.tennismylife.org` (`scripts/bridge_tennismylife.py`)
  carries tour-level 2026 matches from there to **2026-08-24** (1,309 rows).
  TML carries `score` (so retirements are detectable) but **not** `minutes`.
- So ratings are current to the week before the 2026 US Open, not stale at
  Roland Garros as a reading of the CSV filenames alone suggests.

Whether TML becomes canonical is [#3](https://github.com/peteowen1/dropshotverse/issues/3),
still open — it needs a 50-year player-id crosswalk.

## What is live

- **US Open 2026 projections**, refreshing to R2 every 30 minutes via the
  Windows scheduled task `DropshotverseUSOpenLiveSim` (to 2026-09-14).
  `DropshotverseUSOpenCapture` captures point-by-point (to 2026-09-13).
- **The ITG wall chart** — [inthegame-blog#581](https://github.com/peteowen1/inthegame-blog/issues/581),
  shipped 2026-09-01. The blog's tennis section runs on dropshot's own pairwise
  match model.
