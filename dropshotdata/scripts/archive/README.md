# scripts/archive — superseded research scripts

Moved here 2026-09-06 (ticket 40). Nothing live imports them; nothing in a
scheduled task or workflow runs them. They stay in git as the experimental
record the tickets cite, and they still run from here (they import each
other by name, so run them with this directory as the working directory or
on `sys.path`).

| family | superseded by |
|---|---|
| `train_xgb.py`, `train_xgb_v2` to `_v10`, `train_xgb_widedata`, `train_xgb_split_by_tour`, `train_xgb_with_set_elo` | `scripts/train_best_model.py`, which its header says reproduces this chain end to end |
| `tune_*` (ticket-era hierarchical-model tuning: tickets 17 to 26) | the frozen v25 config in `train_best_model.V25_HIER_PARAMS` |
| `fetch_archive.py` | its source went 404 on 2026-08-27; the freeze it produced is the `sackmann-archive` Release |

Three `tune_scratch` scripts still import `tune_ticket21` and carry a second
`sys.path` entry pointing here. Kept in `scripts/`: `train_xgb_granularity_variants.py`
(imported by `ci_variant_d_vs_a.py`).
