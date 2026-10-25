# Red-team of published numbers: dropshotverse (2026-10-25)

Reviewer: Claude Opus 5.5, read-only, 30-minute time box. Status: complete within the box. The effect sizes for F1 and F2 were not measured (that needs a rebuild, which was out of scope). Each finding says so.

Times are UTC unless marked AEST.

## What was read and fetched

- `CLAUDE.md`, `docs/NEXT-STEPS.md` (last updated 2026-09-15), `docs/DECISIONS.md` (D-1 to D-59), `docs/plans/validation-charter.md` (scorecard section), `docs/reviews/FABLE-redteam-2026-09-07.md`
- Scripts: `build_blog_data.py`, `regen_elo_raw_with_bridge.py`, `xgb_bt_rating.py` (lookups), `train_best_model.py` (train window), `upload_blog_data.py`, `usopen_live_draw.py` (name keys), `game_elo_raw_for_xgb.py` (header)
- Workflows: `ratings-refresh.yml`, `event-forecast.yml`, `live-projection.yml`
- Blog consumers: `inthegame-blog/tennis/upcoming.qmd`, `tennis-sim.js`, a grep of every `tennis/*.qmd` for market or accuracy claims (there are none)
- Public R2 copies, saved raw with headers in `C:\Users\peteo\AppData\Local\Temp\claude\C--dev\39fd69f0-45f0-4869-a182-ab55e2255873\scratchpad\dropshot_redteam_r2\` (and `...\scratchpad\r2\`, see the process note at the end): `ratings.parquet` (Last-Modified 2026-10-24 10:32), `tour-draws.json` (2026-10-25 05:19), `upcoming_predictions.parquet` (2026-09-15 07:48), `predictions.parquet` (2026-09-09 12:36), `live_predictions.parquet` and `pairwise.parquet` (2026-09-16 01:03), `upcoming_draw.json`, `upcoming_pairwise.parquet`, `draw.json`
- `gh release view model-inputs-2026-09` asset list, and the release's `us_open_2026_predictions.json`

## What the site publishes today

The only tennis numbers that still change are:

- the Ratings page: `ratings.parquet`, 342 rows, `ratings_as_of` 2026-10-17, rebuilt daily by `ratings-refresh.yml`
- the Draw Simulator for the tour events in play now (Osaka, Ningbo, Brussels, Almaty; Vienna and Basel next week): `tour-draws.json` gives the bracket and results, and the match probabilities are Bradley-Terry on `bt_rating`, `p = 1/(1+exp(-(bt_a - bt_b)))` (`tennis-sim.js:68`). None of these events has a pairwise matrix.

The US Open files are final and frozen. The Upcoming page holds one event, the Beijing WTA, and filters it out because it has started (`upcoming.qmd:100-104`).

## Findings, ranked by how much they would move a published number

### F1. Set-level and game-level Elo are frozen at 2026-06-02 in every rating and forecast published since. Game Elo is the model's largest input | still live | confidence: high on the fact, effect size not measured

**Claim.** Two inputs never get the tennismylife bridge rows: `set_elo_raw_ratings_full.parquet` and `game_elo_raw_ratings_full.parquet`. Both stop at 2026-06-02, the start of Roland Garros. No workflow regenerates them. `ratings-refresh.yml:65-70` and `event-forecast.yml:74` only run `regen_elo_raw_with_bridge.py`, which rewrites the overall and surface Elo file (`regen_elo_raw_with_bridge.py:28`). D-43 measured the shipped bag's mean |SHAP| share: game Elo 21%, surface Elo 19%, point model 13%, age 10%, overall Elo 3%. So the ratings labelled "as of 2026-10-17" carry overall and surface Elo that are current. The largest single feature, though, does not see Wimbledon, the North American hard-court swing, the US Open or the Asian swing. That is 4.5 months missing. The model then reads the disagreement between a fresh overall Elo and a stale game Elo as information.

**Evidence.**
- Local `tune_scratch/set_elo_raw_ratings_full.parquet` and `game_elo_raw_ratings_full.parquet`: max `date` 2026-06-02, 0 rows on or after 2026-08-30. Their file times are 2026-08-29 19:22 and 20:12 AEST.
- The release copies are the same size to the byte: `game_elo_raw_ratings_full.parquet 34906460` and `set_elo_raw_ratings_full.parquet 34900787` (`gh release view model-inputs-2026-09`, asset updatedAt 2026-09-09T14:09Z). The Actions runners download exactly these files.
- `game_elo_raw_for_xgb.py:28` loads the "prepared archive", meaning Sackmann only, with no bridge.
- Serve time: `xgb_bt_rating.py:297-300` looks up each player's last set and game Elo before `date0`. Line 339-340 falls back to 1500 when there is none.
- Of the 342 players in the published ratings table, 287 have tour matches in the overall-Elo file after the last game-Elo row. That count uses the local overall-Elo file, which only bridges to 2026-08-30, so it is a lower bound. Another 8 have no tour set or game Elo row at all and get the 1500 default: Andrea Guerrieri, Harry Wendelken, Sebastian Gorzny, Jack Kennedy, Daniel Merida, Reese Brantmeier, Thea Frodin and Kristina Liutova.
- Not found in any hub. A grep of `docs/*.md` and `docs/plans/*.md` for set or game Elo near stale, bridge, frozen or June returns nothing.

**Affects.** `bt_rating` on the Ratings page. Every Simulator probability for the current tour events. The Beijing provisional forecast. The US Open live projection and `pairwise.parquet` (dated 2026-08-30, already three months stale). The locked US Open forecast too, although "locked" means what was said, and it was said with these inputs.

**Suggested check.** Run `set_elo_raw_for_xgb.py` and `game_elo_raw_for_xgb.py` with the bridge rows appended, the same way `regen_elo_raw_with_bridge.py` does it. Rebuild `ratings.parquet --ratings-only` into scratch and diff `bt_rating` per player: report the mean |change| and the top 10 movers. Then score both versions on the 105 matches in F2 and on the Asian swing (Beijing, Tokyo, Wuhan, Shanghai results in the bridge). The fix is two more regen steps in `ratings-refresh.yml` and `event-forecast.yml`, plus a coverage assertion that the max date of every Elo input sits within 14 days of `ratings_as_of`.

### F2. The Simulator's live match odds are not measured, and on the 105 matches scored so far they are thin: ATP is no better than a coin | still live | confidence: low (n=105)

**Claim.** For every tour event now in the Simulator, the match probabilities are Bradley-Terry on `bt_rating`. That rating is the row mean of the model's logit matrix at one fixed context, Hard court and round of 32 (`tennis-sim.js:25-27, 85-89`). Nobody has scored these numbers as match probabilities. `tennis-sim.js` itself says the Bradley-Terry replay cannot reproduce the model's own odds (dropshotverse#6). The ratings are genuinely out of sample here: they include matches through 2026-10-17 and the events started 2026-10-19.

**Evidence.** `scratchpad\score_cur.py` scores every completed match in `tour-draws.json` (byes and unplayed matches dropped). It matches names by sorted tokens and gives an unrated player the median `bt_rating`, as the page does. Per-match rows are in `scratchpad\scored_current_events.csv`.

Log loss and Brier score are lower-is-better; a coin scores 0.693 log loss and 0.25 Brier. Accuracy is the share of matches where the favourite won, higher is better. "Elo" is the published `elo_overall` alone.

| slice | n | log loss (BT) | log loss (Elo) | Brier (BT) | Brier (Elo) | accuracy (BT) | accuracy (Elo) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 105 | 0.645 | 0.666 | 0.227 | 0.237 | 63.8% (67/105) | 61.9% (65/105) |
| both players rated | 88 | 0.637 | 0.671 | 0.223 | 0.239 | 65.9% (58/88) | 63.6% (56/88) |
| ATP | 49 | **0.700** | 0.704 | 0.252 | 0.253 | 59.2% (29/49) | 57.1% (28/49) |
| WTA | 56 | 0.597 | 0.633 | 0.206 | 0.222 | 67.9% (38/56) | 66.1% (37/56) |

Calibration by favourite strength, matches with both players rated (n=88). The columns are the favourite's mean predicted probability and how often the favourite actually won; a calibrated model has the two close.

| favourite p (BT) | n | mean predicted | favourite won |
|---|---:|---:|---:|
| 0.5-0.6 | 29 | 0.542 | 0.517 (15/29) |
| 0.6-0.7 | 32 | 0.641 | 0.812 (26/32) |
| 0.7-0.8 | 17 | 0.746 | 0.529 (9/17) |
| 0.8-0.9 | 6 | 0.830 | 0.667 (4/6) |
| 0.9-1.0 | 4 | 0.936 | 1.000 (4/4) |

Read with care. A 105-match standard error on log loss is roughly ±0.03, so none of this is significant. The ATP figure (0.700, n=49) is worse than a coin. The 0.7-0.8 bin (won 9 of 17 at a predicted 0.746) is the shape of over-confidence. Both are leads, not results. There is no bookmaker comparison: `raw/odds/` was not checked for October 2026 coverage, and D-15 keeps prices as a benchmark only.

**Suggested check.** Score the bridge's Asian swing (about 300+ tour matches, 2026-09-21 to 2026-10-17) the same way, with ratings taken as of each event's start date. That needs a backfilled ratings snapshot per week, because the daily file overwrites itself. Report log loss, Brier and calibration bins against Elo and against `raw/odds` where they join. If Bradley-Terry on `bt_rating` is over-confident, the cheapest fix is a single scale on `bt_a - bt_b`, fitted out of sample.

### F3. `bt_rating` is a hard-court, round-of-32 number applied to every event whatever the surface | still live, harmless this month | confidence: high on the code

`tennis-sim.js:25-27` documents that `bt_rating` uses a fixed Hard / R32 context. ATP entries in `tour-draws.json` carry `surface: None` and `level: None` (all four ATP events). This week's events are all hard courts (Brussels, Vienna and Basel are indoor hard), so nothing published is wrong now. It will be wrong from the clay and grass swings of 2027. The Simulator's fallback would give a clay specialist the same odds on clay as on hard. **Suggested check:** before February 2027, publish a per-surface `bt_rating`, or keep the fallback off clay and grass events.

### F4. The Upcoming page has nothing current: one provisional Beijing forecast from 2026-09-15, never re-run | still live (stale, not wrong) | confidence: high

`upcoming_predictions.parquet` on R2 holds 76 rows, all `tournament_id 2026-1020` (Beijing WTA), `source provisional`, `field_status "76/96 entrants, 0 seeds, 20 tbd"`, `ratings_as_of 2026-08-30`, `updated_at 2026-09-15T07:47Z`. The event started 2026-09-30. NEXT-STEPS says to "re-dispatch nearer 09-30 once it completes to move source from provisional to locked". The file shows that never happened. The page now hides the event (`upcoming.qmd:100-104`, dropshotverse#44), so a reader sees no forecast rather than a wrong one. The forecast was never scored against Beijing's result. That check is cheap: the champion's pre-event `p_title`, and match-level scoring from `upcoming_pairwise.parquet` against the bridge results.

### F5. Status of the 2026-09-07 Fable red-team findings

| Fable finding | status now | evidence |
|---|---|---|
| F1 laptop re-uploads a stale ratings file | **fixed** (the cause is gone) | The laptop tasks were disabled 2026-09-15. R2 `ratings.parquet` is the 2026-10-24 Actions build with `ratings_as_of`. `upload_blog_data.py:71` still globs every parquet in `blog/`, but a skip for files the run did not produce was added (`:81`). Not re-tested. |
| F2 real players minted as `NEW_` debutants | **fixed** | `usopen_live_draw.py:39, 81-82` now index token-sorted and first+last keys; the final `live_predictions.parquet` has 0 `NEW_` ids |
| F3 `ratings_as_of` label one tournament ahead | **fixed** | `build_blog_data.py:323-341` labels the last included match date (`included_through`) |
| F4 head-to-head counted at all levels in training, tour-only at serve | **fixed** (from a code comment, effect not re-measured) | `xgb_bt_rating.py:214` comment says the `level_file == "tour"` filter was removed to match `train_best_model.py:153` |
| F5 default age 24.0 published as measured | **mostly fixed, not confirmed** | 8 rows now publish age as NA. 6 sit at exactly 24.0: Ben Shelton and Qinwen Zheng were both born in October 2002, so 24.0 fits; Svrcina, Llamas Ruiz, Avanesyan and Bouzas Maneiro were not checked |

## Checked and fine

- **The locked US Open forecast really was made before the event, by a model that never saw 2026.** `predictions.parquet` (`source locked`) reshapes `tune_scratch/us_open_2026_predictions.json` (`build_blog_data.py:54, 235-258`). That JSON's file time is 2026-08-29 20:10 AEST (10:10 UTC), before the `DATE0` of 2026-08-30. The release copy is byte-identical (md5 `0e33535c53064e04e138d9df7bef2391` for both). The `updated_at` of 2026-09-09 is only the time it was reshaped. The bag trains on `TRAIN_YEARS = 1995..2023` (`train_best_model.py:79`), so no predicted match was in its training data. This is the check that failed on torp, and here it passes. Outcome: the men's champion Zverev had a locked title chance of 9.5% (ranked 3rd) and the women's champion Rybakina 18.5% (ranked 2nd). That is n=2 titles, not a score.
- **The locked forecast came from the model before the D-41/42/48/50 fixes.** That is correct for a locked forecast. It is mentioned so nobody "refreshes" it.
- **Live in-match win probability is not published.** `--live-score` is off and a CI guard fails any producer that passes it (D-21, D-22). Only the bracket projection was live.
- **The walk-forward scorecard** (`validation-charter.md:26-43`) asserts `train.date.max() < test.date.min()` per fold. It discloses that 2022-2025 are contaminated by model selection, and that only 2019-2021 support an absolute market claim (n=3 years). It also reconciles with the +0.0247 forward test. No tennis page quotes a market or accuracy claim: a grep of `tennis/*.qmd` for market, bookmaker, log loss, Brier, accuracy and calibration found nothing.
- **The current ratings are out of sample for the current events.** `ratings_as_of` is 2026-10-17 and the events started 2026-10-19. `build_blog_data.py` filters every lookup to `date < as_of`.
- **The final US Open files are consistent.** `live_predictions.parquet` says complete, 127/127 matches decided, champion Zverev / Rybakina, 256 rows.

## Not reached

- The size of F1. It needs a set and game Elo regen over the bridge, about 2 minutes each, which counts as running a pipeline.
- Whether the hierarchical point-model snapshot used by the daily ratings replays over the bridge rows. `build_blog_data.py:182` passes `df=bridged_df`, which suggests yes, but the snapshot code was not traced.
- Bookmaker coverage for October 2026 in `raw/odds/`.
- Scoring Beijing (F4) and the Asian swing (F2's suggested check).

## Process note

The session scratchpad is shared with at least one other agent in this workflow. Partway through, `scratchpad\r2\predictions.parquet` was overwritten at 19:46 AEST by another sport's file (columns `prediction_id, match_id, format, model_version, ...`), together with `player-ratings-v2.parquet`, `t20-teams.parquet` and others. Every tennis number above comes either from the first inspection (before the overwrite) or from a fresh download to `scratchpad\dropshot_redteam_r2\`. `ratings.parquet` and `tour-draws.json` were checked against the fresh copy and have identical md5s. Future fan-outs should give each agent its own scratch subfolder.
