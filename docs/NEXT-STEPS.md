# dropshotverse — Next Steps

> Living queue. Sessions update this when they finish work (rule in `C:\dev\CLAUDE.md`).
> Decision log: [DECISIONS.md](DECISIONS.md) · Vault map: [HOME.md](HOME.md) ·
> Rolled-off narrative: [backlog/](backlog/)
> Last updated: **2026-09-06 evening.**
>
> **Read first if you are cold:**
> - The market question is CLOSED (D-5, D-6, D-10). Fourteen rounds, a clean
>   2026 forward test, no exploitable edge. Do not reopen it without new data —
>   [issues/36](../.scratch/dropshot-tennis-engine/issues/36-forward-test-2026.md).
> - The live US Open projection publishes from **GitHub Actions**, not the
>   laptop (D-13). An hourly freshness alarm watches the public copy.
> - `main` is real: PRs #8-#12 merged 2026-09-06, each reviewed.

## Awaiting Pete

1. **Disable the laptop task `DropshotverseUSOpenLiveSim`** once one
   *scheduled* Actions run is green (the proof so far is manual run
   34012925126, every step green, 5 min 15 s). Both producers writing to R2
   in the meantime is harmless. Closes [#4](https://github.com/peteowen1/dropshotverse/issues/4).
2. **Do not publish a "we match the market" claim** (D-4, a recommendation):
   the 0.00073 gap is from a burnt holdout, and the 2026 forward test puts the
   honest gap at 0.0247 with 38% accuracy when dissenting.

## Open queue

1. **[#5](https://github.com/peteowen1/dropshotverse/issues/5)** — publish
   match-pairing/fixture data so the blog can build a real bracket. No blog
   consumer exists yet.
2. **Forecasts for events after the US Open: WTA solved, ATP not.**
   `fetch_wta_upcoming.py` gets real calendar and entry lists weeks ahead
   (100% entrant resolution across four events). ATP has no entry-list source
   outside the Slams' own feeds; a "projected field from top-N by rating" is
   the honest fallback and must be labelled as such.
   Plan: [plans/after-the-us-open.md](plans/after-the-us-open.md).
3. **Make the scripts' paths relative** so `live-projection.yml` can drop the
   `C:\dev\dropshotverse` junction and run on Linux. Every pipeline script
   hard-codes the absolute path. Do it between events, not during one.
4. **The surface rating is not a self-correcting rating.** `elo_baseline.py:409`
   updates it with the blended `p_w`, which at `blend=0.0` is entirely overall
   Elo, so it accumulates residuals bucketed by surface with no feedback loop
   (std 498 vs 180). `elo_surface_diff` **is** an XGBoost feature. Whether it
   should use `p_s` is untested.
5. **[#1](https://github.com/peteowen1/dropshotverse/issues/1) — hier tail
   bug, downgraded (D-12).** The shipped model does not inherit it and needs
   the feature (+0.0092 without). Only worth picking up as "would a fixed hier
   make a better feature": an upside experiment.
6. **[#2](https://github.com/peteowen1/dropshotverse/issues/2)** — the WTA
   public live-scores API. Its in-play fields have never been observed
   populating.
7. **`DropshotverseUSOpenCapture` exits 1 intermittently.** Cause unknown;
   597/597 matches captured, so not chased.

## Live deadlines (absolute dates)

- **2026-09-13** — US Open men's final. `live-projection.yml` has no expiry.
  The laptop tasks run to **2026-09-16 12:00 local** (2026-09-15 22:00 New
  York); the point capture still depends on the laptop.
- **~2026-09-23** — re-run `fetch_wta_upcoming.py --fetch 1020` for the China
  Open (Beijing, starts 2026-09-30). On 09-06 the field was 76 of 96 and
  0 seeded; a forecast now would be over a partial, unseeded field.
- **After the final** — `event_status = complete` lets the site render a
  result; the freshness alarm downgrades stale-while-complete to a note.

## Recently landed (2026-09-06)

- **Live projection runs from GitHub Actions** (D-13): `live-projection.yml`
  every 30 min from the `sackmann-archive` and `model-inputs-2026-09`
  Releases; `check-freshness.yml` hourly on the public R2 URL, no secrets.
- **Five PRs merged** (#8-#12). Review findings fixed before merge: the live
  sim and the point-table build printed anchor FAILs and published anyway;
  both now refuse the write. Per-column coverage gate on the point table.
- **Point capture was blind every other hour.** The hourly backfill took ~60
  min and the task ignores new instances, so live sweeps were dropped. Now
  incremental via `reference/backfill_done.json`, ~50 s a pass.
- **Issue #1 answered** (D-12): the hier tail bug does not reach the shipped
  model; dropping the feature costs +0.0092 on 5/5 folds.
- **Issue #3 decided and closed** (D-11): keep Sackmann as base, TML as bridge.
- **The market question closed**: the 2026 forward test (2,654 untouched
  matches) put the model at 0.59638 vs the market's 0.57168, 38.0% right when
  dissenting (CI [32.3, 43.8]). Round 14's K change was the cautionary tale:
  5/5 folds in tune, +0.000029 out of sample.
- **WTA upcoming forecasting works**; **in-match WP has its first calibrated
  result** (ticket 37, Mode A, ECE 0.0471); **`usopen_2026_points.parquet`**
  (80,690 points, all 27 columns populated).

Earlier detail: [backlog/journal-2026-09-05.md](backlog/journal-2026-09-05.md).

## Known and NOT chased

- **Config D forward test** — would turn the +0.0247 upper bound into the
  shipped model's exact gap. Skipped: three measurements agree, and a model
  right 38% of the time when it dissents has no edge worth sizing.
