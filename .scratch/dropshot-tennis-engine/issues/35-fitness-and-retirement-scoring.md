# Ticket 35: fitness signal and retirement scoring (rounds 12-13)

Opened 2026-09-05. Pete's instruction: keep improving the model before running
the clean 2026 forward test. Continues ticket 28's chase-the-market programme,
which had stopped at round 11 with a self-imposed stop-point flag on
multiple-comparison grounds.

**Both rounds screened on the 2022-23 TUNE window only. The 2024-25 market
holdout was not scored once.** That holdout has been revealed at least a dozen
times across tickets 27/28 and `RESUME-beat-elo.md` already flags it as no
longer a clean estimate. The 2026 season (Jan-Aug, Pinnacle odds present,
never touched by any decision) is being preserved as the real test.

---

## Round 12: a player's own recent retirement as a pre-match feature — REJECTED

### The idea

Ticket 28 diagnosed the residual gap to the market as "most plausibly late
match-time information (injury/fitness/conditions) no pre-match archive feature
can reconstruct." That diagnosis is half wrong in an interesting way. A
retirement IS in the archive. `elo_baseline.prepare()` drops every RET/W-O row
(line 161) because the OUTCOME is uninformative about skill — but *the fact
that a named player's body failed in a named week* is a different quantity,
genuinely pre-match for every later match, and nothing in the pipeline reads it.

### What was measured, in five phases

`tune_scratch/screen_fitness_signal.py`, `_phase2.py`, `_phase3.py`,
`_phase4.py`, `_phase5_placebo.py`.

**Phase 1 — prevalence and a first look.** Anchor A1 (RET/W-O prevalence
2-5% of tour matches) PASSED at 2.70% (RET 1.92%, W/O+DEF 0.77%, n=361,571
tour-level rows). Measuring win rate against *Elo's own expectation* rather
than against 50% (so the test cannot rediscover "injured players are often also
weaker players"), the deficit showed a clean monotone decay:

| days since own breakdown | n | actual | Elo-expected | deficit |
|---|---|---|---|---|
| 0-30 | 20,753 | 0.5479 | 0.5748 | **+0.0269** |
| 30-90 | 39,219 | 0.5664 | 0.5819 | +0.0155 |
| 90-365 | 106,783 | 0.5785 | 0.5853 | +0.0068 |
| 365+ | 136,646 | 0.5473 | 0.5528 | +0.0055 |

z = +7.79 on the 0-30d cell. Textbook dose-response. Effect sizing
(stats-discipline #3): 16.14% of matches have exactly one side inside 90 days,
mean deficit +0.0212, crude log-loss headroom **~0.00015** against a 0.00073
gap to market — small but around a fifth of what is left.

**Phase 2 — fitness trait or Elo bookkeeping bug? ANCHOR FAILED.** The
non-zero floor at 365+ days does not fit a fitness story. Competing story: the
dropped RET rows mean a retiring player never takes the rating debit. Anchor B1
predicted the never-broke-down control group would be unbiased. It was biased by
**-0.0181** — they win *more* than Elo predicts, opposite sign. Per
stats-discipline #1 rule 2 an anchor failure indicts the method, and it did:
"never retired" is largely "has not played many matches", and Elo underrates
players whose ratings have not converged from the 1500 start. Every
between-group contrast in phases 1-2 was measuring career stage alongside
fitness. B2's apparent "deficit rises with career breakdown count" is the same
confound wearing a different hat — 5+ career retirements means a long career,
which means an ageing player, and Elo lags decline.

**Phase 3 — within-player design. All anchors passed, still wrong.** Comparing
each player to himself (unit = player-match, residual = won minus Elo's own p)
differences out skill, age, career stage, injury-proneness and tour at a
stroke. Effect **-0.0366** (95% bootstrap CI [-0.0449, -0.0272]), same sign in
both tours (ATP -0.0325, WTA -0.0424). C1, C2, C3 all PASS. Elo's overall mean
residual came out at +0.00000, a clean calibration sanity check.

And yet the dose-response was **flat**: -0.0192 (0-14d), -0.0342 (14-30d),
-0.0335 (30-90d), -0.0298 (90-365d) — weakest in the first fortnight and
undiminished a year later. Acute injury does not behave like that. Cause: the
control cell was ">=365d since a breakdown OR never broke down", and a player's
never-broke-down observations are all early-career. Elo lags a player's own
improvement upward and their decline downward, so the contrast was partly
early-career vs late-career — a confound that survives within-player
differencing because it is within-CAREER.

**Phase 4 — event study with time-local controls.** Baseline window
[-120,-60) days before the retirement; career trajectory is near-constant
across four months. All three anchors FAILED:

| window | events | obs | effect vs baseline |
|---|---|---|---|
| [-120,-60) | 6,370 | 56,990 | +0.0000 (by construction) |
| [-60,-30) | 5,077 | 23,068 | -0.0668 |
| [-30,0) | 5,446 | 24,723 | -0.0682 |
| **[0,30)** | 5,475 | 23,189 | **+0.0341** |
| [30,90) | 5,218 | 38,360 | -0.0801 |
| [90,180) | 5,617 | 60,201 | -0.0621 |

D1 FAIL (post-event effect is positive), D2 FAIL (the "effect" fully predates
the retirement), D3 FAIL. "Player does better right after retiring and worse
everywhere else" is not a story about injury.

**Phase 5 — placebo events, and the actual explanation.** Re-ran phase 4's
identical machinery on random (player, date) pairs drawn from the same players'
own careers, where no retirement occurred. The placebo **reproduced almost the
whole profile**: -0.0682 / -0.0801 / +0.0055 / -0.0555 / -0.0521 against the
real -0.0668 / -0.0682 / +0.0341 / -0.0801 / -0.0621. The negative wings are
manufactured by the `>= 3 baseline observations` eligibility rule, which selects
on the player being actively in form during the baseline window. Net of placebo
the retirement effect is:

    [-60,-30) +0.0013   [-30,0) +0.0119   [0,30) +0.0285
    [30,90)  -0.0245    [90,180) -0.0100

The only sizeable net effect is **positive** in the 30 days after retiring —
consistent with withdrawal selection: a player who retires hurt skips the events
they cannot play, W/O rows are excluded from scoring entirely, and the matches
that do appear are the ones they were fit enough to contest.

### Verdict

**Rejected.** There is no usable acute post-retirement fitness penalty in this
archive. Phases 1-3 all produced statistically strong, anchor-passing,
correctly-signed results that were confounds; only the placebo settled it.

**The reusable lesson: a placebo-event arm is the diagnostic form of
stats-discipline #5's limiting case.** Run the identical machinery on dates
where nothing happened; whatever structure survives is the method's, not the
sport's. Three successive confound layers (between-player, within-player,
within-career) each passed the previous round's checks.

---

## Round 13: should the Elo engine SCORE retirements? — SURVIVES the screen

### The idea

Separate question from round 12, raised by phase 2 and left undecided there.
`prepare()` drops 9,747 tour-level RET/W-O matches (2.70%; 37,345 or 3.04%
across all levels including challenger/ITF). The drop is symmetric, so the
player who *won* that match is never credited either. Every one of those results
is discarded wholesale rather than discounted.

Single axis: `ret_k_mult`, the multiplier on the K-factor for such a row's
update. Everything else held at the shipped arm (blend=0.0, decay=0.3,
Sackmann decaying K, causal round ordering).

### Limiting case verified exactly

`tune_scratch/verify_ret_harness_identity.py`. Predicted in advance that
`ret_k_mult=0.0` must reproduce `elo_baseline.py` to float precision. It does:
**max |delta| = 0.000e+00, mean |delta| = 0.000e+00, 100.00000% of 89,678
joined rows within 1e-9.** Bit-identical, so the screen's deltas are
attributable to the retirement update alone and nothing else.

Worth recording that the first version of the screen *claimed* this check and
did not run it — it printed the published parquet's schema and moved on. The
number was withheld until the check actually executed.

### Result (2022-23 tune window, n=10,904 tour-level completed matches)

`tune_scratch/screen_ret_in_elo.py`, then `screen_ret_vs_wo_split.py` splitting
the bundled category into its two mechanisms.

Delta vs today's engine (ret=0, wo=0), log-loss:

| ret_mult \ wo_mult | 0.00 | 0.25 | 0.50 |
|---|---|---|---|
| 0.00 | +0.000000 | -0.000069 | -0.000109 |
| 0.25 | -0.000272 | -0.000333 | -0.000367 |
| 0.50 | -0.000432 | -0.000486 | -0.000514 |
| 0.75 | -0.000499 | -0.000546 | **-0.000568** |

**RET carries the result, W/O is minor.** With the other axis at zero: the RET
axis alone reaches -0.000499, the W/O axis alone only -0.000109. That matches
the mechanism — a retirement is a real, partially-played match with genuine
evidence in it; a walkover is a result where the winner never hit a ball.

Baseline log-loss 0.599955 -> 0.599386 at the best cell.

### Status: not yet shippable

Three things stand between this and a decision.

1. ~~**`ret_k_mult` is a BUNDLED axis.**~~ **RESOLVED 2026-09-05 — the flaw was
   real, the consequence was not.** In `run()` / `run2()` the match counters
   `n_all` / `n_surf` incremented inside the same `if mult > 0.0` guard that
   applied the rating delta, so switching the multiplier on changed two things:
   whether a retirement moves the ratings, and whether it advances every later
   match's K (Sackmann's K decays with matches played). Decomposed on the
   compact harness (`tune_scratch/screen_k_decomp.py`, part A, W/O held at 0):

   | arm | ret_update | ret_counts | log-loss | delta vs A0 |
   |---|---|---|---|---|
   | A0 baseline (today's engine) | 0.0 | no | 0.599955 | — |
   | A1 counter only | 0.0 | yes | 0.599980 | **+0.000025** |
   | A2 rating delta only | 0.5 | no | 0.599518 | **-0.000437** |
   | A3 both (the old bundled arm) | 0.5 | yes | 0.599522 | -0.000432 |

   **The rating delta carries all of it; the counter contributes nothing** (and
   is fractionally harmful on its own). I had written that the counter "could
   plausibly carry the whole result" — it does not. The retirement story
   stands, and round 13's number is attributable to scoring the match.

   Lesson kept anyway: the bundling was a genuine #5 violation and was only
   caught by re-reading my own guard clause after the grid had run. It happened
   to cost nothing. Next time it might not.

2. **Boundary flag (stats-discipline #5.2).** The best cell sits at
   `ret_mult=0.75`, the top of the grid — possibly not an interior optimum.
   Extension to 1.0/1.25/1.5 was run; see the result appended below.
3. **This is a -0.00057 improvement to ELO, not to the shipped model.** XGBoost
   consumes Elo as one feature among 19 and ticket 28 round 8 already showed it
   has absorbed nearly all of Elo's own signal (an XGBoost-Elo blend was worth
   Δ−0.0001). The downstream effect will attenuate, possibly to nothing.
   Deciding this needs `elo_raw_ratings_full.parquet` regenerated under the new
   rule, features rebuilt, and a tune-window `xgb.cv` — not another Elo-only
   number.

**Independent of the market gap, this is a correctness question about the
shipped ratings**, which feed `ratings.parquet` on R2 and the published US Open
forecast. 2.7% of tour results currently move nobody's rating in either
direction.

---

## Round 14 (opened, not yet run): the Sackmann K schedule has never been tuned

Found while writing up round 13's bundled-axis flaw. `elo_baseline.py` line 29:

    K_BASE, K_SHAPE, K_MIN = 250.0, 0.4, 16.0     # Sackmann-style decaying K

A grep across the whole project for `k_factor` / `K_BASE` / `K_SHAPE` / `K_MIN`
returns **only** `screen_ret_in_elo.py` and `screen_ret_vs_wo_split.py` — the
two scripts written today. Nothing in `tune_scratch/`'s 100-odd screens, nothing
in tickets 17/25/26, touches these three numbers.

What HAS been done to the K-factor, for the avoidance of re-running it:

- **ticket 26** replaced the whole schedule with tennis-crystal-ball's UTS
  context stack (level x round x best-of x kFunction) — rejected, and rejected
  again for WTA specifically in ticket 28 round 6.
- **ticket 28 round 2** multiplied K by a margin-of-victory term — rejected
  monotonically across a 15-point grid.
- **ticket 25** added the layoff/inactivity decay, which is a separate axis
  applied to the RATING, not to K.

So every previous K experiment either replaced the schedule wholesale or
multiplied it by something new. The three parameters that define the shipped
schedule's own shape were inherited from Sackmann and never gridded.

Two reasons this is now the highest-prior candidate:

1. **Round 13's result may already be this finding in disguise.** Because the
   match counters increment inside the same guard as the rating update, the
   -0.00057 could be entirely the K schedule advancing faster, which is a
   statement about `K_SHAPE`, not about retirements. The decomposition that
   settles round 13 is the same experiment as the first slice of round 14.
2. It is single-family, cheap (one Elo replay per grid point, ~5 min), has a
   clean limiting case (K_BASE=250, K_SHAPE=0.4, K_MIN=16 must reproduce the
   current engine bit-identically — the harness is already verified to do so),
   and unlike a new feature it improves the RATINGS themselves, which are a
   published product on R2 and feed the tournament simulator.

Screen on 2022-23. Do not touch 2024-25.

### Result: 14 of 27 grid points, PROVISIONALLY a null — NOT yet complete

`tune_scratch/screen_k_decomp.py` part B. The run was killed by the OS for
memory at 18 of 28 arms (machine-level pressure, not this process — see below);
the per-arm parquet checkpoint preserved everything completed.

Limiting case PASSED: (250.0, 0.4, 16.0) reproduced the baseline to
0.599955 exactly.

| k_base | k_shape | k_min | log-loss | delta |
|---|---|---|---|---|
| **250** | **0.4** | **16** | **0.599955** | **— (shipped)** |
| 250 | 0.4 | 8 | 0.599973 | +0.000018 |
| 150 | 0.3 | 24 | 0.600244 | +0.000289 |
| 150 | 0.3 | 8 / 16 | 0.600333 | +0.000379 |
| 150 | 0.4 | 16 | 0.600808 | +0.000853 |
| 150 | 0.4 | 24 | 0.601066 | +0.001111 |
| 150 | 0.4 | 8 | 0.601580 | +0.001625 |
| 250 | 0.3 | any | 0.605447 | +0.005492 |
| 350 | 0.3 | any | 0.613598 | +0.013644 |

**Every point tried is worse than Sackmann's inherited defaults.** The nearest
rival (k_min=8) is +0.000018, indistinguishable.

Two things worth noting rather than glossing:

- **`K_MIN` is inert wherever `K_SHAPE=0.3`** — identical to six decimals across
  8/16/24 at both k_base=250 and 350. With K = 250/(n+5)^0.3, a player at
  n=1000 still has K≈31, above every floor tested, so the floor never binds.
  It only bites at k_shape=0.4 with a small k_base.
- **13 points were never run**: `k_shape=0.5` entirely (9 points, the
  faster-decay direction), `k_base=350 / k_shape=0.4` (3), and
  `k_base=250 / k_shape=0.4 / k_min=24` (1). Calling this a null before
  `k_shape=0.5` is tested would be reporting an untested region as evidence.

Provisional read: the K schedule is not a lever, and Sackmann's defaults sit at
or very near the optimum of this grid. Finish `k_shape=0.5` before recording
that as settled.

### The provisional null was WRONG, and the untested region held the win

`k_shape=0.5` was the one region with no evidence, and refusing to call the null
was the right decision. Full 27-point grid (`screen_k_remaining.py`):
**`k_base=350, k_shape=0.5, k_min=16` at -0.000616**, best per shape being
0.3 → +0.000289, 0.4 → 0.000000, **0.5 → -0.000616**. Monotone into the
direction that had not been tested.

Extended past that boundary (`screen_k_extend.py`, 48 arms, k_base to 700,
k_shape to 0.7, k_min 12/16/20/28). The script's own boundary check still says
`ON EDGE for k_base, k_shape` — I extended upward and the optimum moved back
to the new *lower* edge.

**The reason is a ridge, and it is the actual finding.** The top cells:

| arm | delta |
|---|---|
| kb=350 ks=0.5 kmin=16 | -0.000616 |
| kb=550 ks=0.6 kmin=16 | -0.000586 |
| kb=350 ks=0.5 kmin=20 | -0.000557 |
| kb=550 ks=0.6 kmin=20 | -0.000543 |

`k_base` and `k_shape` trade off: a higher starting K with faster decay traces
nearly the same K(n) curve, so the search slides along a valley floor and stops
wherever the grid stops. **Do not ship a specific triple as though it were
identified.** The transferable claim is the direction: *the shipped K decays too
slowly, and the faster-decay ridge is worth ~0.0006 on Elo.* `k_min` is the one
axis that genuinely resolved — 16-20 wins, 12 and 28 are worse on both sides,
and the shipped 16 is already on it.

---

## Downstream: does either survive into the model?

`screen_xgb_downstream.py` then `screen_xgb_paired_folds.py`. Both arms use the
same rows, folds and hyperparameters; only the four Elo-derived columns
(`elo_overall_diff`, `elo_surface_diff`, `elo_logit_pwinner`, `n_prior_diff`)
are recomputed per arm, so a difference is attributable to the Elo change alone.

**Limiting case:** rebuilding those columns at the shipped configuration
reproduces `xgb_features_tune.parquet` at max |delta| **0.000e+00 on all four**.

### Paired per-fold result — the unpaired test would have got this wrong

The unpaired comparison put the K change at -0.0011 against a fold-to-fold
spread of 0.0133, i.e. 8% of noise, which reads as nothing. But the arms share
folds, and baseline log-loss ranges 0.574 to 0.609 across those folds — a 0.035
swing that is entirely about which matches land in which fold and is **common to
both arms**. Pairing removes it. `xgb.cv` returns only across-fold means and
stds, never per-fold values, so it cannot do this test at all; the folds had to
be rebuilt by hand.

`num_boost_round` pinned at 217 for every arm. The unpaired run let each arm
select its own stopping point (217 vs 250), which is a second changed axis
(stats-discipline #5 rule 4).

| candidate | mean delta | folds improved | t |
|---|---|---|---|
| round 13 retirements @0.5 | -0.000315 | **3/5** | -0.68 |
| **round 14 K ridge** | **-0.001393** | **5/5** | **-5.70** |
| both | -0.001454 | 5/5 | -3.36 |

**Round 13 REJECTED downstream.** Real on Elo, cleanly decomposed, and it does
not reach the model: 3/5 folds with two going the wrong way is noise.

**Round 14 SURVIVES.** All five folds improve, -0.00074 to -0.00207. It is the
only candidate across fourteen rounds of this programme to move every fold in
the same direction.

### What is NOT established

- **The magnitude.** The K point was selected on this same tune window, so
  -0.0014 is optimistically biased. 5/5 supports the direction; the size needs
  the 2026 forward test.
- **The parameters.** Ship "K should decay faster", not `(350, 0.5, 16)` — see
  the ridge above.
- **Transfer to config D.** This ran on the v1 19-feature set. Config D carries
  set- and game-level Elo that would absorb some of the same signal, so this is
  an upper bound, not an estimate.

---

## Found on the way: the surface rating is not a self-correcting rating

`elo_baseline.py:409` updates the surface rating with the **blended** `p_w`:

    surf[(w, s)] = rs_w + ksw * (1 - p_w)

At the shipped `blend=0.0`, `p_w` is computed entirely from OVERALL Elo. So the
surface rating is updated by a residual that is not its own: it accumulates
overall-Elo surprises bucketed by surface, with no feedback loop, and drifts
**2.8x wider** than a self-correcting rating (file std 498 vs 180 for a
`p_s`-updated version; range -2508..+1918 vs -761..+883).

This never touches Elo's own prediction, because blend=0 keeps the surface
rating out of it. But `elo_surface_diff` **is** an XGBoost feature. So the model
consumes a quantity that looks like a surface rating and is not one.

Found only because the downstream limiting case demanded exact equality: three
of four columns matched at 0.000e+00 while `elo_surface_diff` was off by up to
2,166 rating points. A tolerance-based check would have hidden it.

**Also a correction to an earlier claim in this ticket.** The compact harness
was described as "bit-identical to `elo_baseline.py`". That verification
compared `p_winner`, and at blend=0.0 `p_winner` does not depend on the surface
rating at all — so the surface state was never exercised by it. The claim was
true of what it tested and stated too broadly. Rounds 13/14 are unaffected
(they score Elo log-loss, which at blend=0 is pure overall Elo; overall matched
exactly on 10,290 rows and `p_winner` on 89,678, and the two ratings are
separate arrays with separate counters).

Open question worth its own round: should the surface rating be updated with
`p_s`? It would make `elo_surface_diff` an actual surface rating. Untested.

---

## Harness note: `tune_scratch/elo_compact.py`

Rounds 12-14 were killed by the OS for memory three times. The cause was not the
arithmetic: the screens held 1.23M rows as a pandas DataFrame with STRING uids
("atp:104925") and keyed Python dicts on them. Millions of small string objects
plus dict overhead is where the gigabytes went.

`elo_compact.py` factorises players to int32 once, reduces the CSVs one file at
a time (never concatenating DataFrames), caches to a 4.8 MB `.npz`, and indexes
flat numpy arrays:

| | old harness | compact |
|---|---|---|
| peak working set | several GB | **409 MB** |
| per arm | ~5 min | **~37 s** |
| rows / players | 1,226,517 | 1,226,517 / 44,913 |

`verify_compact_identity.py`: max |delta| **0.000e+00**, 100.00000% of 89,678
joined rows against `elo_eval_predictions.parquet`. The `np.lexsort((round_rank,
day))` had to reproduce pandas' two-stage stable mergesort exactly; any slip
would have diverged the ratings within a few thousand rows.

Every arm checkpoints to parquet on completion. That is why the third kill cost
one arm rather than the run — worth keeping as the default shape for any long
screen on this machine.
