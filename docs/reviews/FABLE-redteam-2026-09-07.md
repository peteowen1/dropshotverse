# Red-team of published numbers: dropshotverse (2026-09-07)

Reviewer: Claude Fable 5.1, medium effort, 20-minute time box. Status: complete (two confirmations left to a later session, listed at the end).

All times below are UTC unless marked local. The session ran 2026-09-06 14:15-14:35 UTC (2026-09-07 00:15 Sydney).

## What was read

- `C:/dev/dropshotverse/CLAUDE.md`, `docs/HOME.md`, `docs/NEXT-STEPS.md`, `docs/backlog/journal-2026-09-05.md`
- `dropshotdata/scripts/simulate_usopen_live.py` (358 lines, the live projection), `build_blog_data.py` (344, ratings and locked predictions), `upload_blog_data.py` (55), `refresh_usopen_live.cmd`, `usopen_live_draw.py` (first 200 lines), `xgb_bt_rating.py` (feature builders, lines 176-380), `build_xgb_features.py` (lines 100-230), `train_best_model.py` (lines 118-145)
- `.github/workflows/live-projection.yml`, `ratings-refresh.yml` (header and upload steps)
- Blog consumer: `C:/dev/inthegame-blog/tennis/index.qmd`, `tennis-live-sim.js` (lines 80-155), `ratings.qmd` (lines 55-85), `data-loader.js` (fetchParquet column handling)
- Data files opened: `dropshotdata/blog/live_predictions.parquet` (256 rows x 23), `pairwise.parquet` (227,584 x 8), `ratings.parquet` (300 x 7, dated 2026-08-29), `predictions.parquet` (256 x 17); the public R2 copies of `tennis/ratings.parquet` (347 x 8) and `tennis/live_predictions.parquet` (256 x 23); the live usopen.org draw feeds `MS.json` and `WS.json` (127 matches each); `raw/usopen/live_sim_task.log` tail
- `gh run list` for peteowen1/dropshotverse (last 11 runs), `gh release view model-inputs-2026-09` (asset updatedAt all 2026-09-06T04:37Z)
- Open issues on peteowen1/dropshotverse: #5 (fixture data), #2 (WTA live API), #1 (hier tail bug). `peteowen1/dropshotdata` does not exist as a repo (dropshotverse is a single repo).

## One row walked end to end

**The number:** Carlos Alcaraz, men's title probability 0.4185, in `blog/live_predictions.parquet` row 0 (`updated_at` 2026-09-06T14:05:56Z, 112 of 127 matches decided), the same file the blog's Tennis page reads through `tennis-live-sim.js:93`.

**How it is built (`simulate_usopen_live.py`):** the real bracket and results come from the usopen.org draw feed (`usopen_live_draw.py:56`). Ratings are frozen at `DATE0 = 2026-08-30` (line 49). For every ordered pair of the 128 players and every round, one XGBoost prediction is made (`build_round_pred_matrices`, line 123); those matrices are then used in 20,000 bracket playthroughs where decided matches are fixed (`simulate_from_state`, lines 64-97). Title probability is the share of playthroughs Alcaraz wins.

**The row the model consumes for one pairing (Alcaraz vs Zverev):** `pairwise.parquet` has p(Alcaraz beats Zverev) = 0.682 at R128, 0.697 at R32, 0.718 at the final. Across every ordered pair the spread over rounds has median 0.011 and max 0.098, so the `round` feature moves things a little, not wildly. Every value is inside [0,1]; every round matrix is antisymmetric to rounding (the log prints max |p_ij + p_ji - 1| = 0.0000).

**The upstream records for that pairing (`xgb_bt_rating.py:245-370`):** `ro` and `rs` are each player's last overall and hard-court Elo before 2026-08-30 from `elo_raw_ratings_full.parquet` (tour-level rows only, `date < date0`, line 248); set and game Elo the same way; the hierarchical point-model state from the snapshot cache; head-to-head wins before `date0` from tour-level rows (line 208); age from the Sackmann yearly CSVs aged forward linearly (line 221); `fatigue_diff` is fixed at 0 (line 360). The prediction is the mean of a 5-seed XGBoost bag.

**Does the number point the right way?** I scored the pairwise probabilities against the 223 US Open matches already completed in the feed (name join 223/223, every feed loser is flagged `eliminated`): men n=111, mean p given to the actual winner 0.661, accuracy 74.8%, log loss 0.608; women n=112, mean p 0.694, accuracy 79.5%, log loss 0.477. A coin is 0.693. Direction and calibration are fine at the aggregate. The walk found the model healthy; the defects below are at the edges (player identity, upload ordering, labels), not in the model's sign.

## Findings, ranked by embarrassment if true

### F1. The laptop re-uploads a 2026-08-29 ratings table every 30 minutes, overwriting the daily ratings refresh within half an hour | confidence: high | affects: `tennis/ratings.parquet` on R2, the blog's Ratings page (`tennis/ratings.qmd`)

**Claim.** `ratings-refresh.yml` (shipped 2026-09-06, "first real run published" in NEXT-STEPS item 2) writes a fresh 347-row `ratings.parquet` with a `ratings_as_of` column to R2 once a day. The laptop task then uploads the stale local copy over it on its next 30-minute cycle, so the site shows the fresh table for at most 30 minutes a day and the 08-29 table the rest of the time. The workflow header (`ratings-refresh.yml:3-5`) describes exactly this clobber as the problem it fixed, but only the producer was fixed, not the uploader.

**Evidence.**
- `dropshotdata/scripts/upload_blog_data.py:33` uploads `sorted(BLOG_DIR.glob("*.parquet"))`, every parquet in `blog/`; `refresh_usopen_live.cmd:24` calls it every run.
- Local `dropshotdata/blog/ratings.parquet`: mtime 2026-08-29 21:41 local, 15,799 bytes, 300 rows, 7 columns, no `ratings_as_of`, `updated_at` = 2026-08-29T11:38:17Z.
- `raw/usopen/live_sim_task.log`, run stamped `refresh ok 07/09/2026 0:06:15` local: `uploading inthegame-data/tennis/ratings.parquet (15,799 bytes)... OK` and `uploading inthegame-data/tennis/predictions.parquet (25,888 bytes)... OK`.
- Same minute: the "Ratings refresh" workflow run completed 2026-09-06T14:05:19Z, and R2 `tennis/ratings.parquet` at 14:17Z had `Last-Modified: 14:08:49 GMT`, 347 rows, `ratings_as_of` = 2026-08-30, `updated_at` 14:06:37Z. The Actions upload happened to land 2.5 minutes after the laptop's, which is the only reason the fresh file was on R2 when I looked.
- `live-projection.yml:109-112` already restricts the Actions producer to `live_predictions.parquet` and `pairwise.parquet` with the comment "Only what this run produced"; the laptop path never got the same restriction.

**Check to confirm.** After the next laptop cycle (any time from 2026-09-06 14:40Z while the laptop task runs):

```
curl -s -o r.parquet https://pub-ee4bf5b599a047f9ac2b9facc1587008.r2.dev/tennis/ratings.parquet
python -c "import pandas as pd; r=pd.read_parquet('r.parquet'); print(r.shape, list(r.columns), r['updated_at'].unique())"
```

Stale means: 300 rows, no `ratings_as_of`, `updated_at` 2026-08-29. Also `grep -c "uploading inthegame-data/tennis/ratings.parquet" C:/dev/dropshotverse/dropshotdata/raw/usopen/live_sim_task.log` (each laptop run adds one line).

**If true, fix sketch.** Give `upload_blog_data.py` an explicit file list and have `refresh_usopen_live.cmd` pass `live_predictions.parquet pairwise.parquet`, matching `live-projection.yml:111`; move the 08-29 `blog/ratings.parquet` and `blog/predictions.parquet` out of `blog/` on the laptop.

### F2. Two real players were published as unknown 1500-rated debutants because the name crosswalk missed them; one of them (Bu) was given a 0.9% chance in a first-round match he then won | confidence: high | affects: `pairwise.parquet`, `live_predictions.parquet`, the Draw and Simulator pages

**Claim.** `usopen_live_draw.py:191` mints a synthetic uid `{tour}:NEW_{name}` when a feed name is not found in the bridged archive. Two players got one: `atp:NEW_Yunchaokete_Bu` and `wta:NEW_Gabriela_Knutson`. A synthetic uid has no Elo, set/game Elo, hierarchical state, age or head-to-head, so `build_player_table` fills every rating with the 1500 default (`xgb_bt_rating.py:274-277`) and age 24.0 (`:224`). But both are established players with archive history: Yunchaokete Bu is an ATP top-100 player (the archive very likely lists him surname-first, "Bu Yunchaokete", which the letters-only normaliser cannot reconcile with the feed's "Yunchaokete Bu"), and the R2 ratings table itself carries `wta:213873 "Gabriela Andrea Knutson"` at Elo 2027.5, bt_rating -1.97. The docstring at `usopen_live_draw.py:37-38` says Knutson "is genuinely new"; the ratings table says she is not.

**Evidence.**
- `pairwise.parquet`, round R128: mean p(Bu beats opponent) over the other 127 men = 0.0217, max 0.253; p(Bu beats Jodar) = 0.00863. The feed shows Bu beat Jodar in round 1 (`draw_MS.json`, `status: Completed`). That 0.9% was the single lowest winner probability among all 223 completed matches.
- Knutson: mean p(beats opponent) at R128 = 0.0149, max 0.173, while her real row in R2 `ratings.parquet` has Elo 2027.5 (mid-table of 347, nothing like the bottom of the field).
- `live_predictions.parquet` rows 127 and 255 carry the `NEW_` ids; `ratings.parquet` has 0 `NEW_` ids, so the blog's bt_rating join for these two also falls to the field median (`build_blog_data.py:100-104` documents that fallback).

**Check to confirm.**

```
python - <<'X'
import pandas as pd, glob
for f in sorted(glob.glob("C:/dev/dropshotverse/dropshotdata/raw/github/atp/atp_matches_202[3-6].csv")):
    d = pd.read_csv(f, usecols=["winner_name","loser_name","winner_id","loser_id"])
    hit = d.winner_name.str.contains("Yunchaokete", na=False) | d.loser_name.str.contains("Yunchaokete", na=False)
    print(f[-8:], d[hit][["winner_name","winner_id","loser_name","loser_id"]].drop_duplicates().head(3).to_string())
X
grep -c "Gabriela Andrea Knutson" C:/dev/dropshotverse/dropshotdata/raw/github/wta/wta_matches_2026.csv
```

Any row with an id for Bu, or a non-zero count for Knutson, confirms the crosswalk missed a player who exists.

**Confirmed in-session (14:23Z):** `grep -l Yunchaokete raw/github/atp/atp_matches_202[4-6].csv` hits all three files, and the 2026 file spells him `Bu Yunchaokete` (surname first, as predicted); `grep -c "Gabriela Andrea Knutson" raw/github/wta/wta_matches_2026.csv` returns 2. Both players are in the archive under a name the letters-only normaliser cannot match to the feed's spelling, so both were published as 1500-rated unknowns.

**If true, fix sketch.** In `build_name_to_uid` also index the token-sorted name (so "Bu Yunchaokete" and "Yunchaokete Bu" collide) and a first+last form that drops middle names; make `verify()` refuse to simulate when a `NEW_` uid is minted for a seeded or ranked player; print every `NEW_` uid on every run so it is seen rather than absorbed.

### F3. `ratings_as_of` says "as of 2026-08-30" but every match dated 2026-08-30 is excluded from the table | confidence: medium | affects: R2 `tennis/ratings.parquet`, the Ratings page date label

**Claim.** `build_blog_data.py:301` sets `as_of = bridged_df["date"].max()` and then every lookup filters `date < as_of` (`build_active_pool` line 87, `build_player_table` at `xgb_bt_rating.py:248`, `_latest_rating_lookup` at `:237`, hier snapshot at `simulate_tournament_updating.py:197`). Sackmann and the bridge date every match of a tournament with the tournament's start date, so the strict `<` drops the whole most recent tournament (or week) from ratings that are labelled with that very date. The label claims more recency than the table has.

**Evidence.** R2 `ratings.parquet` `ratings_as_of` = 2026-08-30, which is `DATE0`, the US Open start date; every bridged row dated 2026-08-30 is outside the `<` filter. `ratings.qmd:71-82` renders `ratings_as_of` as "the last match date the ratings include".

**Check to confirm.** (Needs the archive and the bridge, about 1-2 minutes.)

```
python - <<'X'
import sys; sys.path.insert(0, "C:/dev/dropshotverse/dropshotdata/scripts")
import bridge_tennismylife as bridge, hierarchical_model as hm
df, _ = hm.prepare(bridge.load_matches_with_bridge(hm)); mx = df["date"].max()
print("as_of", mx.date(), "rows dated exactly as_of:", int((df["date"] == mx).sum()))
X
```

Any non-zero count confirms it.

**If true, fix sketch.** Either label `ratings_as_of` as `as_of - 1 day` or use `date <= as_of` in the ratings path only (the live sim's `date < DATE0` is deliberate and must stay).

### F4. Head-to-head is counted from all levels in training but tour-level only at serve time | confidence: medium (code), low (effect size) | affects: `pairwise.parquet` via the `h2h_diff` feature

**Claim.** `build_xgb_features.py:222` computes training `h2h` from `elo_raw_ratings_full.parquet` "unfiltered, all levels, for the fullest possible H2H history"; `xgb_bt_rating.py:208` builds the serve-time dictionary from `level_file == "tour"` rows only. The shipped model learned on a count that includes challenger and qualifying meetings, then is fed a smaller count. `n_prior_diff` has a similar acknowledged proxy (`xgb_bt_rating.py:35`). `fatigue_diff` fixed at 0 is fine for a Slam main draw (in a knockout both players have played the same number of event matches, so the diff is 0 in training too), so that one is not a skew.

**Check to confirm.** In `train_best_model.py` near line 190 confirm the `h2h` columns come from the all-levels table; then

```
python -c "import pandas as pd; e=pd.read_parquet('C:/dev/dropshotverse/dropshotdata/tune_scratch/elo_raw_ratings_full.parquet', columns=['level_file']); print(e.level_file.value_counts())"
```

If the non-tour share is material (over about 30% of rows), rebuild the US Open field's `h2h_diff` both ways and print the number of pairs where it differs.

**If true, fix sketch.** Build the serve-time dictionary from the same rows as training (drop the `level_file == "tour"` filter at `xgb_bt_rating.py:208`).

### F5. Twelve players in the published ratings table carry the hard-coded default age 24.0 | confidence: high that the value is the default, low that it moves a published probability | affects: R2 `tennis/ratings.parquet` `age` column, `age_diff` feature

**Claim.** `latest_age_at` returns 24.0 when the player has no age row before `date0` (`xgb_bt_rating.py:224`). The age table comes only from the frozen Sackmann CSVs (`load_age_table`, `:176-198`), not from the bridge, so any player whose only matches arrived through the bridge gets 24.0. R2 `ratings.parquet` has 12 rows at exactly 24.0: Andrea Guerrieri, Harry Wendelken, Arthur Cazaux, Toby Samuel, Sebastian Gorzny, Jack Kennedy, Daniel Merida, Diane Parry and four more. Cazaux and Parry are 23, so the default is close for them, but the column is published as a measured age.

**Check to confirm.**

```
python -c "import pandas as pd; r=pd.read_parquet('r.parquet'); print(r[r.age==24.0][['player_id','player_name','tour']].to_string())"
```

(`r.parquet` from the F1 download.) Then grep those ids in `raw/github/{tour}/*_matches_202*.csv` and look at `winner_age` / `loser_age`.

**If true, fix sketch.** Publish `age` as NA when defaulted (and log the count), or carry `age` from the bridge rows.

## Things checked that were fine

- Sign and calibration of the pairwise model on the 223 completed US Open matches: accuracy 74.8% (men) / 79.5% (women), log loss 0.608 / 0.477 vs coin 0.693; every feed loser is flagged `eliminated`; no name failed to join.
- `live_predictions.parquet`: title probabilities sum to 1.0000/1.0001 per tour, final to 2.000, semi to 4.000; no value outside [0,1]; `seed` NA for 75% of rows is the 32-of-128 seeding, not a defect.
- `pairwise.parquet`: 227,584 rows = 2 tours x 7 rounds x 128 x 127; per-round mean exactly 0.5; antisymmetric to rounding; round dependence is modest (median spread 0.011).
- The anchors in `simulate_usopen_live.py:140-176` gate the write (SystemExit on any failure) and the .cmd wrapper skips the upload on non-zero exit.
- The live copy on R2 is current: `updated_at` 2026-09-06T14:05:56Z, 112 decided, matching the laptop log.
- `model-inputs-2026-09` release assets all updated 2026-09-06T04:37Z, after the code they feed (PRs #15-#18 merged 2026-09-06), so no published-vs-deployed drift on the Actions path.
- The blog's `fetchParquet` tolerates a missing `ratings_as_of` column (warns "requested column(s) not in file", reads the rest), so F1 degrades the date label rather than breaking the page.
- Elo `p` uses only the overall rating (`blend=0.0`), so the surface rating's lack of feedback does not reach the Elo probability directly; it does reach XGBoost via `elo_surface_diff` (already queued, below).

## Already known, not re-derived

- Live projection cut from 276 s to 68 s on 2026-09-06 with identical output; GitHub Actions backup and freshness alarm shipped the same day.
- Surface Elo is not self-correcting (NEXT-STEPS item 6) and `elo_surface_diff` is an XGBoost feature.
- hier tail-calibration bug (issue #1, downgraded D-12).
- The 0.00073 market gap is from a burnt holdout; the honest 2026 forward-test gap is 0.0247 (D-4, issues/36).
- bt_rating imputed to the field median for draw players missing from `ratings.parquet` (documented in `build_blog_data.py:90-110`).

## Not reached in the time box

- Confirming F1 on R2 after the next laptop cycle (the check command above; a 10-second check any session can run). At 14:23Z R2 still held the fresh 347-row file; the laptop's next cycle was due about 14:35Z.
- F3's row count on the bridged frame (needs the archive loaded, about 2 minutes).
- Whether the women's R64 result (mean p 0.81, 30 of 32 right) is chance or a sign the R64 matrix is over-confident; n=32 is too small to say and the R128/R32 rounds look normal.
- The training-side `h2h`/`n_prior` definitions in `train_best_model.py` (F4 is from `build_xgb_features.py`; the shipped model is trained by `train_best_model.py` and may differ).
- `predictions.parquet` (the locked pre-tournament forecast) is also re-uploaded every 30 minutes with a fresh R2 Last-Modified; harmless because its contents do not change, but the F1 fix covers it.
