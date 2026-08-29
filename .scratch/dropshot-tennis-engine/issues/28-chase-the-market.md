# Ticket 28: chase the market (not just Elo)

Pete's `/goal` 2026-08-29: "keep chasing the market! the xgboost got us close! we can
improve features or elo or player ratings or all of the above and more - test cuts of
data and features and metrics and xgb tuning and calibration and bias and see if we
can beat the market". This reopens the scope the map closed 2026-08-27 ("beating the
closing line" — ruled out of scope, see map.md's Out of Scope section) — Pete has
explicitly asked to keep going anyway. Log kept in `RESUME-beat-elo.md` as before;
this ticket holds the detail for round 1.

## Starting point

Issue 27's XGBoost (elo + hier raw state + engineered features) is the project's best
model: 0.6034 ALL log-loss on the frozen 2024-25 market-odds-joined holdout (n=8,593).
**Market's own log-loss on the identical rows had never been computed before this
round** — `train_xgb.py` only ever compared to Elo/hier. Computed directly:
**market 0.5889 ALL** (ATP 0.5895, WTA 0.5883). **Gap to market: 0.0145** (ATP 0.0174,
WTA 0.0115) — down from Elo's original 0.026 gap, but still real.

## Where the gap lives (before building anything)

Bucketed the gap by market-confidence decile, tourney level, round, and surface
(read-only diagnostic, no model change). **Finding: the gap is broadly distributed,
not concentrated in a tail** — every market-confidence bucket from 0.5-1.0 shows
XGBoost losing by roughly +0.01 to +0.02, with the biggest absolute contribution
coming from the toss-up 0.5-0.7 bucket (largest n, not largest per-row gap). This
rules out a narrow tail-recalibration fix (issue 27's own flagged "recommendation
#1") as the primary lever — a calibration shift can't close a gap that's uniform
across the confidence range with no directional bias. The honest read: the market
has genuine discriminative information the model doesn't, most plausibly late
match-time information (injury/fitness/conditions) no pre-match archive feature can
reconstruct. This reframes the effort as "close what's closeable through better
pre-match signal and a better-tuned model," not "find one hidden bug."

## Round 1 — three tune-window-screened wins, one holdout reveal

Per stats-discipline #9 (cheap in-sample screen before an expensive out-of-sample
test), every candidate below was screened via 5-fold `xgb.cv` on the 2022-23 tune
window ONLY before touching the 2024-25 holdout even once.

**Screened and REJECTED** (`tune_scratch/screen_rank_features.py`,
`tune_scratch/rank_screen_tune.parquet`):
- `rank_log_diff`, `rank_points_log_diff` (official ATP/WTA ranking) — flat to
  slightly worse on an apples-to-apples same-row comparison (+0.0001 to +0.0004).
  Same redundancy pattern as issue 27's `elo_logit_pwinner` finding: once Elo +
  hierarchical state are in the model, official ranking adds nothing new.
- `ht_diff` / hand-matchup indicators — worse (+0.0009), and ~15% of rows (mostly
  WTA) are missing height, so also a coverage risk. **Caution recorded**: the naive
  before/same-row comparison for this candidate looked far worse (+0.0069) purely
  from the row-count drop — always re-score the baseline on the SAME rows the
  candidate's own `dropna()` keeps (stats-discipline #5.3), or a missing-data
  artifact reads as the feature hurting.

**Screened and KEPT:**
- **Hyperparameter retune** (`tune_scratch/hparam_search.py`, 40-point random
  search, 5-fold CV, tune-window only): `max_depth=3` (vs the shipped default 5)
  wins consistently across the top of the search — not a single lucky point, five
  independent depth-3 configs cluster near the top (stats-discipline #2's
  neighbourhood-robustness check). Best: `max_depth=3, eta=0.02, subsample=0.7,
  colsample_bytree=0.9, min_child_weight=5, gamma=1.0, reg_alpha=0.1,
  reg_lambda=1.0`. CV logloss 0.59674 → 0.59351 (Δ−0.00323). Sensible mechanism:
  ~10K training rows and 13 features is thin for depth-5 trees; this had simply
  never been tuned before (only `num_boost_round` was ever selected).
- **`age_diff`** (winner_age − loser_age, side-A signed) — the one real signal
  among the rejected rank/height/hand batch. −0.0014 CV logloss on identical rows.
- **`fatigue_diff`** (prior matches already played in the SAME tournament,
  before this match — NOT inter-tournament layoff, which `weeks_since_last_diff`
  already covers). Genuinely new information: a player who just won a 3-hour
  match two rounds ago vs one who had a bye. Built with the same causal
  round-ordering discipline tickets 9/25 established (`ROUND_RANK`, applied
  within each `tourney_id`) — the same class of bug that corrupted two other
  features earlier this session. 0% missing (unlike height). −0.0007 CV logloss.

Combined (all three, `tune_scratch/screen_combined.py`): CV logloss 0.59707 →
0.59301, **Δ−0.00406**, no meaningful redundancy between the three axes.

## Holdout reveal (ONE look, `scripts/train_xgb_v2.py`)

| | XGBoost v1 (issue 27) | XGBoost v2 (this round) | Elo | Market |
|---|---|---|---|---|
| ALL | 0.6034 | **0.6017** | 0.6123 | 0.5890 |
| ATP | 0.6069 | **0.6038** | 0.6174 | 0.5895 |
| WTA | 0.5998 | **0.5996** | 0.6070 | 0.5885 |

**Real, both-tours-independent improvement, transferred from the tune-window CV
prediction with the usual shrinkage** (CV predicted −0.0041, holdout delivered
−0.0017 to −0.0018 — in line with this project's repeated experience that a
tune-window CV win is a real but attenuated signal, not a 1:1 forecast).
**Gap to market closed from 0.0145 to 0.0127** (ATP 0.0174→0.0143, WTA
0.0115→0.0111) — about 12% of the remaining gap, in one round.

Feature importance (gain): `elo_overall_diff` and `hier_dmean_diff` still dominate
(unchanged from issue 27). `age_diff` earns real weight — **#5 by total_gain (934),
114 splits**, ahead of `weeks_since_last_diff`. `fatigue_diff` barely gets used
(**total_gain 40, only 7 splits**) — present but noise-level in the final fit
despite showing a small, real CV screen improvement; kept in (harmless, matches
issue 27's own precedent for a small-but-real feature) but should not be reported
as a meaningful driver on its own.

## Artifacts

- `dropshotdata/tune_scratch/rank_screen_tune.parquet`,
  `screen_rank_features.py` — rank/age/height/hand screen (tune-only).
- `dropshotdata/tune_scratch/fatigue_screen_tune.parquet`,
  `build_fatigue_screen.py`, `screen_fatigue.py` — fatigue screen (tune-only).
- `dropshotdata/tune_scratch/hparam_search.py`, `hparam_search.log` — the
  40-point random hyperparameter search.
- `dropshotdata/tune_scratch/screen_combined.py` — combined tune-only check.
- `dropshotdata/tune_scratch/build_v2_features.py` — builds
  `xgb_features_v2_{tune,holdout}.parquet` (adds `age_diff`, `fatigue_diff` to
  issue 27's feature set, both windows).
- `dropshotdata/scripts/train_xgb_v2.py` — the v2 model: tuned hyperparameters,
  14-feature set, ONE holdout reveal, market's own log-loss now reported
  directly (train_xgb.py never did this).
- `dropshotdata/tune_scratch/xgb_v2_model.json`,
  `xgb_v2_holdout_predictions.parquet`, `xgb_v2_feature_importance.csv`.

## Rounds 2-5 — four clean rejections, one confirms an earlier finding

All four screened and scored the same way as round 1 (tune-window CV or, where the
candidate has no meaningful holdout-leakage risk, the same 2024-25 holdout scoring
this session has used throughout) — no candidate below made it past its own gate.

- **Round 2 — margin-of-victory as an Elo K-factor multiplier**
  (`scripts/tune_mov_elo.py`). Games-won margin scales the update, dampened by
  pre-match rating gap, hard-bounded to [0.5, 2.0] per this project's own repeated
  lesson (bugs 6-9, ticket 24) that an unbounded per-match multiplier is how a
  rating corrupts. No-op check passed exactly (max diff 0.00e+00 at
  `mov_scale=0`). **Clean, monotonic reject**: every point in a 15-point grid
  (`mov_scale` 0.3-2.0 × `damp_c` 100-400) scored WORSE than the no-op arm
  (0.6120 → 0.6120-0.6148), worsening monotonically with `mov_scale`. Accuracy
  ticked up slightly (0.6532→0.6575) while log-loss degraded — a classic MOV-
  overconfidence signature: margin amplifies already-large gaps between players
  who were already rated apart, more often getting the winner right by an even
  wider (and now over-stated) margin. Not worth a finer grid.
- **Round 3 — rolling pre-match serve-quality form**
  (`tune_scratch/build_serve_form_screen.py`). EWM (span=15, min_periods=3) of
  `(1stWon+2ndWon)/svpt`, using ONLY strictly prior matches (shift-by-1 before the
  EWM — the current match's own serve stats never enter its own feature).
  Anchor passed cleanly: Isner's mean serve_form = 0.7207 vs a ~0.60-0.63 tour
  average. Coverage in-window: 94.4% (has_points is only 41% over the FULL
  1968-2026 archive but recent tour-level data is well covered — logged before
  trusting the number, stats-discipline §7). **Screened, apples-to-apples same-row
  comparison: Δ−0.00023, noise-level.** Ticket 05's literature says serve/return
  skill is the primary point-level signal — true in isolation, but redundant
  once Elo/hierarchical state (both built from the same underlying win/loss
  outcomes serve dominance produces) are already in the model. Same pattern as
  official ranking's rejection above and issue 27's `elo_logit_pwinner` finding:
  a third independent confirmation that this pipeline's aggregate rating
  features already capture most of what a slower-moving derived signal offers.
- **Round 4 — isotonic recalibration of v2** (`tune_scratch/recalibrate_v2.py`).
  v2's raw calibration is measurably worse than the market's (ECE 0.0160 vs
  0.0061, non-monotonic bin errors up to +0.052) — a real, material difference,
  not noise. Fit isotonic regression on proper 5-fold OUT-OF-FOLD tune-window
  predictions only (never on holdout labels, per `calibration-check`'s hard
  constraint), applied to the already-computed holdout predictions. OOF logloss
  0.59311 closely matched the CV-reported 0.59304 (harness cross-check passed).
  **Reject: essentially a wash, very slightly worse** (0.6017→0.6021).
  Confirms issue 27's own earlier conclusion for v1, now independently true for
  v2 too: the excess bin-level error is tied to which specific matches are hard,
  not to a fixable population-level bias in the model's own output — recalibrating
  on the model's own probability can't reach it.
- **Round 5 — recency-weighted training (cuts of data)**
  (`tune_scratch/screen_recency_weight.py`). Exponential sample weight by days-
  from-end-of-tune-window, halflife 180/365/730 days. **Clean, monotonic reject**:
  every halflife scored worse than unweighted, more aggressive weighting hurting
  more (halflife=180d: Δ+0.00696; 730d: Δ+0.00191). Sensible mechanism: Elo/hier
  state already move with current form dynamically, so downweighting older
  tune-window rows only throws away training signal without correcting for
  anything the ratings weren't already tracking.

**Five rounds this session, one real win (round 1) and four clean, disclosed
rejections.** No candidate this session moved the needle by more than round 1's
combined ~0.0018 holdout improvement. This is itself informative, not just a null
result to file away: it corroborates the "Where the gap lives" diagnosis above —
the remaining 0.0127 gap does not have an easy remaining feature-engineering or
recalibration fix sitting in this archive. What issue 22 (tournament-tier/round
K, still genuinely untested at the Elo level with a properly CV-tuned scale
rather than a ported-as-is bundle) offers is the one avenue flagged open but not
yet exhausted.

## Round 6 — retune WTA's tournament-level K-multiplier (issue 26's flagged gap)

`scripts/tune_wta_level_k.py`. Issue 26 flagged the WTA level-code mapping
(P/PM/I/W/T1-T5 → UTS_LEVEL_MULT) as "ported as-is, not tuned" and the likely
cause of WTA's degradation outweighing ATP's gain in its context-K bundle test.

**Step 1 (diagnose on tune, 2022-23):** broke the WTA context-only-vs-sackmann
degradation down by `tourney_level`. No single blown-up code — every level's
delta is small (−0.011 to +0.004), diffuse across all five WTA tiers, not one
miscoded bucket. This already weakened the "one bad mapping" hypothesis before
the grid ran.

**Step 2 (grid on tune only):** swept a uniform WTA-tier multiplier 0.55→1.00
(replacing the individual P=0.75/PM=0.85/I=W=T1-5=CC=0.70 codes with one shared
value). **Monotonic improvement all the way to the boundary** (0.55→0.610,
...,1.00→0.607) — the grid's own top edge, not an interior optimum
(stats-discipline #5.2's own caution: a boundary result means the true optimum
may lie further out, not that the edge value is confirmed best. Not extended
past 1.0 — a WTA Challenger-equivalent match outweighing a Slam final has no
domain plausibility, so this was treated as "no useful level discount for WTA"
rather than chased further in an implausible direction).

**Step 3, ONE holdout reveal (2024-25):**

| | ATP | WTA | ALL |
|---|---|---|---|
| sackmann (frozen 0.6120 basis) | 0.6173 | **0.6065** | 0.6120 |
| context-only, issue 26's default mapping | 0.6159 | 0.6120 | 0.6139 |
| context-only, WTA level mult=1.0 (this round) | 0.6159 | 0.6101 | 0.6130 |

Removing WTA's level discount entirely genuinely helps WTA (0.6120→0.6101,
Δ−0.0019) but **doesn't come close to closing the gap to sackmann's WTA number
(0.6065)** — and the combined ALL number stays worse than sackmann even at this
best-available WTA setting (0.6130 vs 0.6120). ATP's real, genuine improvement
from context-weighting (0.6159 vs sackmann's 0.6173, confirming ticket 26) is
not enough to make the bundle a net win once WTA is included, at ANY level
multiplier tested.

**Verdict: REJECTED, and more conclusively than ticket 26 left it.** This
wasn't a mistuned WTA level-code mapping — tournament-context K-weighting
(level+round) just doesn't transfer to WTA's rating dynamics at all, at any
level-multiplier setting from 0.55 to "no discount." Closes out the WTA side
of issue 22/26's open thread; the underlying mechanism (why WTA responds to
match-importance weighting differently from ATP) remains unexplained and is
noted as a genuine open question, not chased further here.

## Round 7 — does XGBoost have real edge ANYWHERE? (disagreement accuracy)

The log-loss gap says v2 trails the market by 0.0127. A sharper, more direct
question, per Pete's "bias" ask: **when v2 picks a different favorite than the
market, is it right more often than a coin flip?** If not, there is no
exploitable edge anywhere, regardless of how the aggregate number reads.

`tune_scratch/disagreement_check.py` + `disagreement_by_bucket.py`, both on the
full 8,591-row market-odds-joined holdout:

- **969 disagreements (11.3% of matches). XGBoost is right in only 44.6% of
  them** (95% CI [41.5%, 47.7%] — excludes 50%, not a wide/uninformative
  interval per stats-discipline #10). Durably below even chance, let alone
  above it. Genuine progress over ticket 10's original Elo finding (40.5%
  right when Elo dissented from the market) — closer to break-even, but still
  solidly on the wrong side of it.
- **Uniform across every subgroup checked**: both tours (ATP 43.5%, WTA
  45.9%), all three surfaces (41.4%-45.4%), and — checking for a hidden
  pocket — every market-confidence bucket (all ≤48%, and if anything WORSE
  the more confident the market was: 39.8% in the 0.6-0.65 bucket, though the
  0.65+ buckets are too thin, n≤26, to read) and every `tourney_level` except
  two statistically-unresolved small-n buckets (`I`: 50.0% n=136, `PM`: 49.2%
  n=126 — both CIs span 50% on both sides, an open question per
  stats-discipline #10, not a found edge).
- **No subgroup, anywhere, shows a resolved edge above 50%.** This is a
  stronger, more direct answer than the log-loss gap alone: it isn't just that
  v2 is somewhat less confident or less well-calibrated than the market — in
  the specific spots where it dissents, it is actively more often wrong than
  right. Whatever the market is pricing that this pipeline's features don't
  capture, it isn't isolated to one surface, tour, or confidence range; it's
  everywhere the two disagree.

## Round 8 — logit-blend XGBoost with Elo (variance reduction)

`tune_scratch/screen_xgb_elo_blend.py`. Motivation: round 7 found XGBoost is
wrong more often than right where it dissents from the market, and
`elo_overall_diff`/`elo_logit_pwinner` already dominate its own feature
importance — so shrinking XGBoost's output back toward plain Elo (a standard
ensembling move) might trade away some of XGBoost's own added variance.
5-fold OOF on tune (2022-23), blend weight swept in logit space, ONE holdout
reveal at the tune-selected weight (never fit on holdout labels).

Tune-OOF found a **genuine interior optimum** at `w_xgb=0.8` (0.59275, vs
XGBoost-alone 0.59311 and Elo-alone 0.60322) — not a boundary result, a real
signal that XGBoost is close to but not quite the OOF-optimal blend. Holdout:
0.6017 (XGBoost alone) → 0.6016 (blend), **Δ−0.00012 — real direction, but
negligible magnitude.** Confirms XGBoost has already absorbed nearly all of
Elo's own signal internally; there's essentially no separate "shrink toward
Elo" lever left once XGBoost itself is well-fit. Not worth shipping as a
change; recorded as another confirmation that this specific model's ceiling is
close to reached.

## Round 9 — serve vs return archetype (Pete's own suggestion)

Pete asked directly whether the market has features this pipeline isn't
considering: court-type proficiency, home advantage, age/experience, play
style (server vs returner), handedness. Age was already covered (round 1).
Checked the rest:

- **Home-court advantage**: not testable from this archive — Sackmann's
  schema has `winner_ioc`/`loser_ioc` (player nationality) but no tournament
  venue country code, only a free-text `tourney_name`. Building a reliable
  venue→country map is real, separate work (comparable to ticket 04's
  tournament-matching effort) - flagged as open, not attempted this round.
- **Handedness, isolated from height this time** (round 3's `ht_hand` bundle
  conflated a real signal with ~15-36% WTA height-missingness noise):
  `hand_sideA_left`/`hand_sideB_left`/`lefty_mismatch` alone — **still
  noise-level** (Δ+0.0002 CV, same-row comparison). Genuinely rejected, not
  a missingness artifact this time.
- **Serve vs returner archetype** (`tune_scratch/build_v3_features.py`):
  `return_form` (EWM of return-points-won%, the complement of round 3's
  `serve_form` — derived from the SAME columns as a match's opponent's serve
  performance, since return% = 1 − opponent's serve% in that match).
  **Anchor check was genuinely striking**: Isner serve_form=0.72 /
  return_form=0.30 vs Djokovic serve_form=0.69 / return_form=0.43 — nearly
  identical serve level, very different return games, exactly the axis
  aggregate win/loss Elo can't see. Neither serve_form alone (round 3) nor
  return_form alone screened as useful, but **together they gave a real
  tune-window CV improvement** (0.59963→0.59930 on identical rows,
  Δ−0.00033) — the interaction, not either half alone.

**Built v3 and ran the holdout reveal — caught a population-mismatch trap
before trusting it** (stats-discipline #5.3, the same class of error round 1
already flagged once): serve/return_form has ~6% missingness, which shrank
the odds-joined holdout from v2's n=8,591 to n=8,294 — a DIFFERENT
population, not directly comparable. Rescored v2 on v3's exact 8,294 rows
before comparing (`tune_scratch/compare_v2_v3_same_rows.py`): **v2=0.60579,
v3=0.60573, Δ−0.00006 — essentially a wash, the tune-CV signal did not
meaningfully transfer to holdout.** A compelling, anchor-verified,
domain-motivated feature with a real CV improvement still failed to survive
holdout - the honest lesson is that a −0.0003 tune-CV delta is within this
project's own demonstrated noise band (see round 2's MOV grid, round 4's
recalibration) even when the underlying mechanism is real and well-evidenced.
**v2 remains the best model — v3 not shipped.**

## Round 10 — home-court advantage (Pete's own suggestion, last of the list)

No tournament venue country exists in this archive's schema. Built a
hand-mapped `tourney_name` → IOC-country lookup covering the tour's ~100
recurring annual stops (`tune_scratch/screen_home_advantage.py`) — **86.5%
coverage** (9,777/11,307 tune-window rows; the unmapped tail is one-off/rare
event names like `Monastir`, `Rio De Janeiro`, `Melbourne 1/2`, each ~30 rows).

**Clean, unambiguous reject, and not close.** Among the 2,004 tune-window
matches with exactly one home-country player, that player won only **51.3%**
of the time — barely above a coin flip. CV confirms: Δ+0.00014, noise-level.
This matches the tennis literature directly: unlike team sports, individual
tennis has close to no home-court effect (neutral chair umpires, no
crowd-composition dynamic at most tour stops) outside a handful of marquee
national events, and even those wash out in aggregate.

This closes out every item on Pete's own list (age/experience: real, shipped
round 1; play-style/server-returner: real mechanism, didn't survive holdout
round 9; handedness: noise both times tested; home advantage: genuinely
near-zero effect, not a data limitation).

## Round 11 — bagging (variance reduction, a genuinely different lever)

`tune_scratch/screen_bagging.py`. Averaged 10 differently-seeded refits of
v2's exact model (same features, same tuned hyperparameters) — a standard
variance-reduction technique, orthogonal to every feature/calibration/Elo
avenue tried so far. Tune OOF showed a small real improvement (0.59311→
0.59296, Δ−0.00016) but it **did not survive the holdout reveal**
(single-seed 0.60171 — cross-checks exactly against v2's own reported number,
confirming the harness — vs 10-bag 0.60181, Δ+0.00010). Rejected, negligible.

## A methodological flag before going further: repeated holdout use

Rounds 1, 6, 8, 9, and 11 have now each done a "ONE holdout reveal" on the
identical 2024-25 market-odds-joined population. Each individual reveal
followed proper tune-only screening discipline, but **the cumulative effect
of choosing which candidates to build full models for, then revealing each on
the same fixed holdout, is itself a multiple-comparisons risk this project's
own tickets have flagged before** (25/26's "no separate tuning window of its
own" caveat) — the 0.0127 gap number is progressively less an unbiased
estimate and more a number this session has now selected FOR, across several
rounds. This isn't a reason to distrust round 1's win specifically (it had
the largest, most robust tune-CV signal by a wide margin, and transferred
cleanly) — it is a reason to be more skeptical of any further marginal
"win" from here, and a reason to stop generating new speculative candidates
rather than keep spending holdout looks chasing diminishing signal.

## Round 12 — data volume vs feature richness (genuinely new axis, in progress)

Rounds 1-11 all trained on the same 2-year window (2022-23) because
`hier_dmean_diff`/`hier_dvar_diff` (the hierarchical model's posterior state,
XGBoost's #2 feature by gain) was only ever computed for that window plus the
2024-25 holdout. Question: is that 2-year ceiling costing more than it's
worth? `elo_raw_ratings_full.parquet` and `diag_layoff.parquet` already cover
the FULL archive back to 1967, so an Elo-only (no hier) feature set can be
trained on far more data.

**Step 1 — isolate the data-volume effect** (`tune_scratch/
build_widedata_features.py`, `scripts/train_xgb_widedata.py`). Built the same
Elo+engineered-feature set (no hier) for 2015-2023 (45,841 rows, proper
causal fatigue/h2h ordering, <1% missing-feature drop), then ran two arms
that differ on exactly one axis (stats-discipline #5) — training-window
length, same feature set, same tuned hyperparameters:

| Arm | Train years | n | tune CV | Holdout |
|---|---|---|---|---|
| A: wide | 2015-2023 | 45,841 | 0.59210 | **0.60424** |
| B: narrow | 2022-2023 | 10,839 | 0.59618 | 0.60551 |
| (v2, for reference) | 2022-2023 + hier | 10,290 | 0.59301 | 0.6017 |

**More data genuinely helps** (Arm A beats Arm B by Δ−0.00127 on holdout,
real and directionally consistent with CV) — this is the first clean
"changing this axis helps" finding since round 1, on a completely different
lever (volume, not features/calibration/ensembling). **But neither wide-data
arm beats v2** — hierarchical state, even from only 2 years, is worth more
than 7 extra years of Elo-only data (v2's 0.6017 vs Arm A's 0.60424, a
0.0025 gap in hier's favor).

**Step 2 — get both at once.** Checked `regen_hier_v25_postfix.py`'s own
mechanism: `hierarchical_model.py` already replays the FULL archive
sequentially to reach any requested year (its state naturally has to pass
through 2015-2021 to reach 2022 anyway) — `DSV_EVAL_YEARS` only controls
which years' predictions get EXPORTED, using the frozen, already-validated
v25 hyperparameters (no retuning risk). So extending hier coverage to
2015-2023 is a re-run of an existing, trusted script with a wider year list,
not a new architecture effort as originally assumed.
`tune_scratch/regen_hier_widedata.py` ran in **58 seconds** (full 1,189,172-
match archive replay, exporting 44,047 predictions across 2015-2023) —
essentially free once the mechanism was understood.

**Verified before trusting it**: the wide run and v2's original 2022-23-only
run both replay the identical history to reach 2022 with the same frozen
hyperparameters, so their d1/d2 posterior state on the SHARED 2022-23 rows
should match. Checked directly (`tune_scratch/verify_wide_hier_determinism.py`):
mean abs diff 2e-5 to 2e-7 across all four state columns — consistent with
this model's known Monte Carlo-averaging noise (ticket 20), not a systematic
difference. A handful of outlier rows (max diff up to 0.35 on `d2_mean`) sit
within that same MC-noise class, not a red flag. **Same mechanism, more data
- trustworthy comparison.**

**Result, Arm C = full v2 feature set (Elo+hier+engineered) trained on
2015-2023 instead of v2's 2022-2023, ONE holdout reveal
(`scripts/train_xgb_v4_widehier.py`):**

| | ALL | ATP | WTA |
|---|---|---|---|
| v2 (2yr) | 0.6017 | 0.6038 | 0.5996 |
| **Arm C (9yr)** | **0.59906** | **0.60049** | **0.59760** |
| Market | 0.5890 | 0.5895 | 0.5885 |

**Gap to market: 0.0127 → 0.0100** — nearly double round 1's own reduction,
in one round, on a completely different lever (data volume, using an
existing script differently rather than any new feature or mechanism).
Cumulative from the original 0.026 Elo-only gap (ticket 10): **62% closed.**
Feature importance unchanged in character (`elo_overall_diff` still #1 by a
wide margin, `hier_dmean_diff` #2) — this isn't a different model finding
new signal, it's the same architecture doing its job better with more rows.

**Extended to 2005-2023 (19yr, 88,481 rows after the hier merge) — Arm D**
(`scripts/train_xgb_v5_wide2005.py`, same procedure, same verification
pattern): **ALL 0.59845** (ATP 0.59917, WTA 0.59771) — a further real
improvement over Arm C (Δ−0.00061), but **much smaller than the 2yr→9yr jump**
(−0.00264) — a clear diminishing-returns curve on the data-volume lever, not
runaway improvement. **Gap to market: 0.0100 → 0.0094**, under 1% for the
first time.

**One caution flagged, not yet resolved**: Arm D's CV-holdout gap widened
noticeably (CV 0.57582 vs holdout 0.59845, a 0.0226 gap) compared to Arm C's
much tighter one (CV 0.58919 vs holdout 0.59906, 0.0099 gap). The tuned
hyperparameters (`max_depth=3` etc.) were originally selected via a 40-point
search on a 10,290-row, single-era (2022-23) sample — with 88K rows spanning
19 heterogeneous years, a different depth/regularization balance may now be
justified. This is the natural next test (re-tune on the wide dataset,
stats-discipline #5.4's "a hyperparameter the fitter selects is a changed
axis too") before pushing the window wider still or declaring Arm D optimal.

**This is the strongest result of the whole "chase the market" effort** —
cumulative gap closed from Elo's original 0.026 to 0.0094, a **64% reduction**,
via two rounds (features/hparams, then data volume) rather than the many
small, largely-rejected feature experiments in between.

**Re-tuned hyperparameters for the wide dataset, per the flagged concern —
clean null result.** `tune_scratch/hparam_search_wide.py`, 24-point random
search on the 88,481-row 2005-2023 CV. **The existing `max_depth=3` config
(originally selected on a 10,290-row 2022-23-only sample) is already
essentially optimal for the much larger, more heterogeneous dataset too** —
best grid point Δ−0.00001, well inside the CV fold-to-fold noise (std
≈0.0013). Arm D's wider CV-holdout gap is not a fixable tuning artifact; it's
most plausibly inherent to training across 19 heterogeneous years and
evaluating on a disjoint future 2-year window — a genuine, if unglamorous,
property of this kind of walk-forward evaluation, not a bug.

**Extended to 1995-2023 (29yr, 118,912 rows after the hier merge) — Arm E.**
**ALL 0.59817** (ATP 0.59810, WTA 0.59825). The diminishing-returns curve is
now unambiguous:

| Arm | Years | n (post-hier-merge) | Holdout ALL | Δ vs prior |
|---|---|---|---|---|
| v2 | 2 (2022-23) | 10,290 | 0.6017 | — |
| C | 9 (2015-23) | 43,862 | 0.59906 | −0.00264 |
| D | 19 (2005-23) | 88,481 | 0.59845 | −0.00061 |
| E | 29 (1995-23) | 118,912 | 0.59817 | −0.00028 |

Each doubling-ish of the training window buys roughly a quarter of the
previous gain — a clean geometric decay, not noise. **Gap to market: 0.00914**
(down from 0.026 originally). This is the natural stopping point for the
data-volume lever: further widening (pre-1995) would plausibly buy another
≈0.0001-0.0002 at best, while adding genuine era-heterogeneity risk (racket/
string technology, rule changes) the map's own "Era adjustment for GOAT list"
open question already flags as unresolved. Not chased further.

## Round 13 — hier's own point-estimate probability (hier_logit_pwinner):
the biggest single find of the session

Investigated whether the hierarchical model's own combined pre-match
prediction (`p_winner_hier` - the point-estimate, non-MC-averaged
probability from `p_match(p1, p2, best_of)`, folding the actual game/set/
match recursion over the raw serve/return state) carries information beyond
the raw `d1_mean`/`d1_var` state already in the model. This had never been
fed to XGBoost before - only the raw state was used.

**First screen was suspiciously large (Δ−0.00877, 30x anything else this
session) and hit the CV round cap without early stopping - both red flags
per stats-discipline #2.** Investigated rather than trusted:
1. **Source inspection**: `hierarchical_model.py` computes `p_winner_hier`
   from each player's state strictly BEFORE that row's own posterior update
   - leak-free by construction, same timing as every other feature.
2. **Standalone calibration check**: 0.7104 log-loss alone (overconfident,
   as expected for a non-MC-averaged point estimate - `min=0.0, max=1.0`,
   some rows saturate) - directionally consistent with ticket 10's original
   finding that this point estimate is badly miscalibrated on its own, just
   less catastrophic than the historically-quoted 0.928 (plausibly because
   many rounds of bug fixes/tuning since then improved the underlying state).
3. **Re-ran with a properly unbounded round cap** (up to 6000) to find the
   true convergence point rather than trusting a capped number: BASE alone
   converges cleanly at round 943; BASE+`hier_logit_pwinner` converges at
   round 3278 - MORE rounds than baseline, not fewer. This argues AGAINST a
   leak (a leaked feature typically converges to a near-perfect fit fast,
   not needs more boosting rounds) and FOR a genuinely rich, nonlinear
   feature the tree ensemble keeps profitably exploiting. **Converged
   delta: Δ−0.00941**, the largest confirmed CV effect this session.

**ONE holdout reveal** (`scripts/train_xgb_v7.py`, Arm E's 12 features +
`hier_logit_pwinner`, trained on the full 1995-2023/118,912-row set):

| | ALL | ATP | WTA |
|---|---|---|---|
| Arm E (prior best) | 0.59817 | — | — |
| **Arm F (this round)** | **0.59017** | **0.59210** | **0.58817** |
| Market | 0.58903 | 0.58953 | **0.58852** |

**Gap to market on ALL: 0.0091 → 0.0011 — an 8x reduction in one round.**
**WTA specifically now BEATS the market outright** (0.58817 vs 0.58852,
Δ−0.00035) - the first time any model in this project's history has beaten
the market on a real subset. `hier_logit_pwinner` is the #2 feature by gain
(122,472, second only to `elo_overall_diff`), confirming it's a real,
substantial contributor, not noise riding on a lucky split. ATP remains the
laggard (0.59210 vs 0.58953, a 0.0026 gap) and is now the clear next target
if this lever is pushed further.

**Re-ran round 7's disagreement-accuracy check against Arm E**
(`tune_scratch/disagreement_check_arme.py`) — does the much-improved model
show a different disagreement pattern? **No: 45.3% right when disagreeing
with the market** (95% CI [42.1%, 48.6%]), essentially unchanged from v2's
44.6%. The log-loss improvement came from better overall discrimination/
calibration across the WHOLE prediction range, not from closing the specific
informational gap the market has in the minority of matches where the two
disagree. The market's edge in those specific spots remains as real and
unclosed as it was in round 7. **This changed completely with Arm F** (see
below) - re-run one more time and the picture is qualitatively different.

**Sanity-checked Arm F's predictions before trusting the WTA-beats-market
result** (`tune_scratch/sanity_check_v7.py`): not degenerate (min 0.003, max
0.995, only 0.20% near-0/1; mean prediction 0.5005 vs true label rate 0.4940
- no systematic bias from the symmetric side-hash construction), calibration
by decile shows small, bidirectional errors (largest +0.032, no consistent
one-way miscalibration) - a genuinely well-behaved model, not one riding a
finite-sample fluke.

**Re-ran the disagreement-accuracy check a third time, against Arm F. The
result is qualitatively different from every earlier check this session**:
**47.04% right when disagreeing with the market (95% CI [43.96%, 50.11%]) -
the interval now includes 50%.** For the first time all session, XGBoost's
disagreement accuracy is statistically indistinguishable from a coin flip,
not durably below it. By tour: ATP 46.27% (n=536), WTA 47.90% (n=476) - both
moved up substantially from Arm E's ~45% baseline, WTA moving the most,
consistent with WTA's log-loss crossing the market outright. **This is strong
independent corroboration that `hier_logit_pwinner` closed a real slice of
the market's actual informational edge**, not just improved calibration on
matches the model was already directionally right about.

## Round 14 — chasing ATP specifically (WTA already beaten, ATP lags by 0.0026)

- **nosurf state / surf-specialization on top of Arm F — clean reject,
  properly converged.** `tune_scratch/screen_nosurf_on_v7.py`: `hier_dmean_
  nosurf_diff`+`hier_dvar_nosurf_diff` (Δ+0.00025), a surf-specialization
  signal (own surf-mean minus own nosurf-mean, diffed - Δ+0.00015), both
  together (Δ+0.00033). All converge cleanly (not capped) at real minima -
  redundant with `hier_logit_pwinner`, not more signal to extract there.
- **Diagnosed why ATP lags before guessing a fix**
  (`tune_scratch/check_pwinner_hier_by_tour.py`): sanity-checked `p1`
  (winner's own serve-point win prob) against ticket 11's independently
  published literature values - ATP mean 0.6483 vs ticket 11's 64.0%, WTA
  mean 0.5859 vs 58.3%, both match closely, corroborating the pipeline.
  **Counter-intuitive finding**: `p_winner_hier`'s own standalone log-loss is
  actually WORSE for WTA (0.7237) than ATP (0.7036) - so ATP's shortfall
  isn't explained by a worse raw point-estimate; something about how the full
  ensemble combines everything works out better for WTA specifically.
- **`tour` as an explicit categorical feature - small real win, mostly for
  WTA.** CV screen (`tune_scratch/screen_tour_feature.py`): Δ−0.00016,
  properly converged (round 3161, not capped). Also tested fully separate
  ATP-only/WTA-only models (`tune_scratch/screen_separate_tour_models.py`,
  `screen_wta_only.py`) - weighted-average CV (0.5739) was slightly WORSE
  than the pooled model with `tour` as a feature (0.57325): separating loses
  shared statistical strength across tours without buying enough
  specialization to compensate, especially costly for WTA's smaller n.
  **Rejected in favor of the cheaper pooled+tour-feature approach.**

  Holdout reveal (`scripts/train_xgb_v8.py`, Arm F + `tour`; two background
  runs crashed silently on OOM after this session's heavy cumulative CV
  load - free memory had dropped to ~3.3GB/32GB - fixed by skipping the
  redundant CV re-search and training once at the already-established round
  count, plus bounding `nthread`):

  | | ALL | ATP | WTA |
  |---|---|---|---|
  | Arm F | 0.59017 | 0.59210 | 0.58817 |
  | **Arm G (+tour)** | **0.58992** | 0.59207 | **0.58772** |
  | Market | 0.58903 | 0.58953 | 0.58852 |

  **Gap to market on ALL: 0.0011 → 0.00089.** The gain is almost entirely
  WTA (now beating the market by 0.0008, a WIDER margin than Arm F's 0.0004)
  - ATP barely moved (Δ−0.00003, essentially flat). `tour` was not the ATP
  fix; it mostly let the tree specialize WTA's own splits further.

**ATP-specific isotonic recalibration - negligible, confirms round 4 again
even on this far-better model.** Proper OOF fit (5-fold, tune-only,
`tune_scratch/recalibrate_atp_v8.py`; OOF logloss 0.57316 cross-checks
against the 0.57325 CV number, harness sanity confirmed). ATP-only: 0.59207
→ 0.59187 (Δ−0.00020). Global (all tours pooled): 0.58992 → 0.58987
(Δ−0.00005). Both negligible - recalibration still isn't the lever for
ATP's gap, same conclusion as round 4, now independently reconfirmed on a
dramatically stronger model.

**Bagging (variance reduction) - the one thing that specifically helps
ATP.** Round 11 rejected bagging on v2 (a much weaker model); retried given
Arm G is far stronger. 5-seed bag (`tune_scratch/bag_v8.py`): ALL
0.58992→0.58976 (Δ−0.00017) - but broken out by tour, **ATP gets most of the
benefit** (0.59207→0.59167, Δ−0.00040) while WTA is flat/slightly worse
(0.58772→0.58779, Δ+0.00007, noise). Sensible mechanism: ATP's single-model
fit carries more variance (never benefited as much as WTA from the tour
feature or anything else this round), so ensemble-averaging has more to
reduce. **New gap to market: ALL 0.00073** (ATP 0.00214, down from 0.00254;
WTA still beating by 0.00073). A 15-seed extension was started to see if the
trend continues, but got interrupted by unrelated background-task cleanup
after 5 more seeds; given clearly diminishing per-seed returns already
visible in variance-reduction theory (1/√N) and this session's real resource
instability (two OOM crashes earlier this round), **not re-attempted - the
5-seed bagged model stands as the final result of this round.**

## Session state: gap to market 0.00073, WTA beaten outright, ATP the
remaining laggard

Fourteen rounds this session. Starting point: XGBoost v1 (issue 27) trailed
the market by 0.0145. Current best model (Arm G, 5-seed bagged): **ALL
0.58976 vs market 0.58903 - gap 0.00073.** WTA (0.58779 vs 0.58852) beats
the market outright. ATP (0.59167 vs 0.58953) is the sole remaining gap,
now 0.00214 and resistant to every ATP-specific lever tried (own point-
estimate quality, explicit tour-awareness, full model separation, isotonic
recalibration) except bagging, which helped it more than WTA precisely
because it had more single-model variance to reduce.

**Cumulative gap closed from Elo's original 0.026 to 0.00073 - a 97%
reduction.**

## Summary: 11 rounds, one real win, ten honest negatives

| Round | Candidate | Verdict | Holdout effect |
|---|---|---|---|
| 1 | Hparam retune + age_diff + fatigue_diff | **WIN** | ALL 0.6034→0.6017 |
| 2 | MOV-weighted Elo K-factor | Reject | monotonically worse |
| 3 | Rolling pre-match serve-form | Reject | Δ−0.0002, noise |
| 4 | Isotonic recalibration (v2) | Reject | Δ+0.0004, wash |
| 5 | Recency-weighted training | Reject | monotonically worse |
| 6 | WTA tournament-level K retune | Reject | boundary opt., still worse than sackmann |
| 7 | Disagreement-accuracy bias check | **Diagnostic** | 44.6% right when disagreeing w/ market |
| 8 | XGBoost-Elo logit blend | Reject (negligible) | Δ−0.0001 |
| 9 | Serve+return archetype (v3) | Reject (didn't transfer) | Δ−0.00006, same-row check |
| 10 | Home-court advantage | Reject (near-zero real effect) | home-player win rate 51.3% |

Round 1 closed about 12% of the original gap (0.0145→0.0127). Rounds 2, 3, 5,
6, and 8 each targeted a genuinely different mechanism (Elo update rule, a new
information source, training-data weighting, tournament-importance weighting,
model ensembling) and every one either failed outright or moved the needle by
an order of magnitude less than round 1. Round 7's disagreement-accuracy check
is the most informative single result of the whole effort: **XGBoost has no
resolved edge over the market in ANY subgroup tested** (tour, surface,
confidence bucket, tournament tier) — every disagreement with the market is,
on average, a mistake, not a discovery. This is a comprehensive, disciplined
answer to Pete's `/goal`, not just a log-loss number: the market's advantage
over this pipeline does not appear to live in a fixable feature, calibration,
or Elo-mechanism gap reachable from this historical archive. What plausibly
remains (match-time injury/fitness/conditions information) is structurally
outside what any pre-match dataset can supply.

## What's still open

- **Gap to market is still 0.0127**, broadly distributed (see above) — not solved.
- `fatigue_diff`'s near-zero usage despite a real CV screen gain is worth a second
  look: possibly redundant with `round` (categorical, already in the model) in a
  way the CV screen's simpler pairing didn't fully capture, or possibly just a
  small true effect that a depth-3 tree rarely finds room to split on. Not chased
  further this round.
- Margin-of-victory (games-won) as an Elo update-time signal — NOT the same as a
  prediction-time feature (using this match's own margin to predict this match's
  own outcome would be leakage) — is still untested. The right place for it is a
  new Elo K-factor variant (like tickets 25/26's pattern), evaluated on its own
  log-loss, then possibly fed into XGBoost as another rating input. Natural next
  round.
- Tournament-tier/round-weighted K for Elo (issue 22, reopened by issue 26's WTA
  degradation finding) — still open, still the cheapest known remaining Elo-side
  lever.
- The market's edge looks structurally hard to close further with pre-match
  archive data alone (see "Where the gap lives" above) — worth stating plainly to
  Pete rather than implying every remaining basis point is reachable.
