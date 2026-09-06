# 40 — Where the live sim's time goes, and what in `scripts/` is dead weight

2026-09-06 evening. Two read-only reviews run while waiting on a GitHub token:
a bottleneck review of the live pipeline and a redundancy inventory of
`dropshotdata/scripts/`. The bottleneck review was then checked against a
cProfile of `simulate_usopen_live.py --sims 2000`, and the profile reordered
it: the biggest cost was in a file the reviewer had not opened.

## Profile (cProfile, 2,000 sims; pure-Python time roughly doubled by the profiler)

Total 569 s profiled. Un-profiled, the same run is about 4.5 min and the
20,000-sim production run about 4.5 min too, because the simulation itself is
~10% of the wall time. Cumulative seconds, project code only:

| rank | function | cum s | share | what it is |
|---|---|---:|---:|---|
| 1 | `bridge_tennismylife.build_name_to_id` (x2 tours) | 214 | 38% | `iterrows()` over 931K archive rows to build a name to id map, `norm`/`strip_accents` per row. Frozen data. |
| 2 | `simulate_tournament_updating.build_snapshots` | 99 | 17% | the hierarchical replay over 1.19M matches to snapshot state at `DATE0`, which is fixed for the whole tournament |
| 3 | `usopen_live_draw.build_name_to_uid` | 68 | 12% | `_norm` over 2.38M strings, generator joins per character |
| 4 | `xgb_bt_rating.build_pairwise_features_base` (x2) | 64 | 11% | 6.6M `Series.__getitem__` calls inside the 128x127 loop, plus `p_match` 29 s |
| 5 | `train_best_model.build_playstyle_features` | 40 | 7% | `groupby().transform(lambda)` EWM over ~1.5M long rows, recomputed every run on frozen data |
| 6 | `hierarchical_model.prepare` + `load_matches` | 36 | 6% | the prepared-archive pickle cache is bypassed because the bridged frame is prepared fresh |
| 7 | Monte Carlo (`run_tour` minus 4) | ~13 | 2% | at 2,000 sims; ~10x at 20,000 |

Ranks 1 to 5 are 85% of the run. Ranks 1, 3 and 5 are pure Python over data
that does not change between runs; rank 2 is a deterministic function of
`(tour, DATE0, archive)`; rank 4 is a data-structure choice.

Reviewer estimates versus measurement, for the record: the reviewer ranked the
replay first (right order of magnitude), put the pairwise loop at 2 to 4 s
(it is 64 s), and did not see rank 1 or rank 3 at all because
`bridge_tennismylife.py` and `usopen_live_draw.py` were outside its brief.
Reading finds defects; only the timer finds cost.

## Fixes, in order of seconds saved per fix

1. **Vectorise `build_name_to_id`**: `drop_duplicates` on (name, id) then
   `dict(zip(...))`, with the normaliser applied once per unique name. ~1 s.
   Nothing else changes. Also cacheable, but the vectorised form is fast
   enough not to need it.
2. **Cache `build_snapshots` output** per `(tour, tourney_id, DATE0, archive
   fingerprint)` as a pickle under `tune_scratch/`. Same-day reruns become a
   disk load. The fingerprint must include the bridged rows before `DATE0`.
3. **Vectorise `build_name_to_uid`**: normalise the unique names once with
   `str.normalize("NFKD")` and a vectorised strip, not a per-character
   generator per row.
4. **Pairwise loop on plain dicts**: pre-extract `ptab` to a dict of tuples;
   the loop then does no pandas indexing. Expect 64 s to a few seconds.
5. **Cache `build_playstyle_features`** to parquet, invalidated on archive
   mtime like `load_prepared()`; or replace the per-group lambda with
   `groupby().shift(1)` and a vectorised EWM.

Expected result: a live run under a minute locally; on the Windows runner the
job drops from ~5 min to ~2 min, with setup dominating. None of these change
any number the model produces; each one should be verified bit-identical on
`live_predictions.parquet` before it ships (the `verify_*_identity.py`
pattern in `tune_scratch/`).

Also seen, not on the live path: `capture_usopen.poll()` re-fetches
completed matches every cycle because `seen_final` is written but never
checked. `once()` (the mode the task runs) is unaffected.

## Result (2026-09-06, branch `perf/live-sim`)

All five shipped, each verified identical to the old code on the live inputs
(`tune_scratch/verify_perf1_name_to_id.py`, `verify_perf_live_path.py`):

| fix | before | after |
|---|---:|---:|
| 1 `build_name_to_id` (both tours) | 88 s | 7 s (the CSV reads) |
| 2 `build_snapshots` | 56 s | 2.5 s on a cache hit; keyed by tournament, date0, a hash of every row before date0 and a hash of the field (review finding: the field was missing at first) |
| 3 `build_name_to_uid` | 11-23 s | 2-3 s |
| 4 `build_pairwise_features_base` (both tours) | 17 s | 10 s (the rest is `hm.p_match`) |
| 5 `build_playstyle_features` | 13 s | 7 s |
| **full run, 20,000 sims** | **270 s (276 s re-measured)** | **127 s cold cache, 68 s steady state; both output parquets identical to the old run's, seeds being fixed** |

Not done: the prepare-cache bypass (rank 6, ~20 s) and `p_match` in the
pairwise loop. Both are next if the runner time matters; neither is worth it
for the laptop.

## Redundancy inventory of `dropshotdata/scripts/` (60 files)

Nothing dead was found inside the live files (`simulate_usopen_live.py`,
`xgb_bt_rating.py`, `train_best_model.py`, `hierarchical_model.py`). The
weight is around them:

| what | size | evidence |
|---|---|---|
| `hier_v18`..`v27` and `hier_tuned` `_final_state.pkl` + `_eval_predictions.parquet` pairs, `hier_fit.log`, `diagnose_joined.parquet` at `dropshotdata/` top level | ~56 MB | zero readers in `scripts/` or `tune_scratch/` (`hier_v25_eval_predictions.parquet`, `hier_final_state.pkl`, `elo_eval_predictions.parquet`, `hier_v24_eval_predictions.parquet` DO have readers and stay) |
| `train_xgb.py`, `train_xgb_v2..v10`, `train_xgb_widedata.py`, `train_xgb_split_by_tour.py`, `train_xgb_with_set_elo.py` | 13 files, 1,423 lines | superseded by `train_best_model.py` per its own header; no importers, no task or workflow |
| `tune_v2`, `tune_hyperparams`, `tune_mov_elo`, `tune_regime_b`, `tune_spread`, `tune_step2_single`, `tune_surface_decay_fix`, `tune_ticket21_{hparams,joint,rw_surf}`, `tune_ticket24_heavytail`, `tune_tier_weight`, `tune_uts_k`, `tune_wta_level_k` (+ the four they import) | 14 files, 2,040 lines | nothing outside the cluster imports in; ticket-era one-offs |
| `fetch_archive.py` | 201 lines | its source went 404 on 2026-08-27; job done, recorded in `manifest/` |

Duplicated helpers, most of which vanish with the removals above: `def log`
in 51 files, `def ll` in 16, `prep_X*` in 15, `ROUND_RANK` in 4,
`load_matches` in 3, `sha256` in 3, `parse_score_games` in 2 (one an admitted
"verbatim port"), and `DD = Path(r"C:\dev\dropshotverse\dropshotdata")` in 8.
Hard-coded `C:\dev\dropshotverse`: 137 occurrences in 69 files; that is the
size of the "make paths relative" job.

Recommended shape: move the `train_xgb_*` and `tune_*` families to
`dropshotdata/scripts/archive/` (git keeps them either way) rather than
deleting, delete the ~56 MB of unreferenced pickles and parquets (they are
not in git), and introduce one `_common.py` with `log`, `ll`, `ROUND_RANK`,
`DD` and `sha256` when the paths-relative change touches every file anyway.
Not checked: whether the duplicated `log`/`ll`/`prep_X` bodies have diverged;
whether `mc_parallel.py` (sole reader of `hier_v24_eval_predictions.parquet`)
is still run.

## Cull (2026-09-06, done)

- 32 scripts moved to `scripts/archive/` with a README: the 13 `train_xgb*`,
  the 14 orphaned `tune_*`, the four they import, and `fetch_archive.py`.
  `scripts/` went from 77 to 45 files. Three `tune_scratch` scripts that
  import `tune_ticket21` got a second `sys.path` entry.
- 26 unreferenced artefacts deleted from `dropshotdata/` (55 MB): the
  `hier_v18` to `v23b`, `v26`, `v27` prediction/state pairs, the `v24`, `v25`
  and `tuned` state pickles, `hier_fit.log`. Each was checked for a non-log
  reader immediately before deletion. Kept: everything a script reads
  (`hier_v26cand_*`, `hier_roundorder_*`, `hier_v24/v25_eval_predictions`,
  `hier_final_state`, `hier_tuned_eval_predictions`) and
  `diagnose_joined.parquet` (its writer still references it).
- The inventory's reference counts were low: it missed six `tune_scratch`
  importers and counted fit-log "wrote ..." lines as readers in the other
  direction. Grep before every move; the agent's table is a starting point.
- Not done here: `_common.py` for the duplicated helpers. It belongs with the
  paths-relative change, which touches every file anyway.

