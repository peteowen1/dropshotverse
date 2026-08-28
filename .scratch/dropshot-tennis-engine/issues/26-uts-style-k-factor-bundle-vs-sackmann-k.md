# 26 — UTS-style (tennis-crystal-ball) K-factor: bundle-vs-bundle test against our Sackmann-style K

Type: task
Status: closed

## Question

Raised 2026-08-28 by Pete, from a screenshot of UTS's (Ultimate Tennis
Statistics / tennis-crystal-ball) K-factor methodology. Ticket 25 fixed
`elo_baseline.py`'s causal-ordering bug and added inactivity/layoff decay,
moving Elo's holdout log-loss from 0.6152 to 0.6120 - but explicitly scoped
out tournament-tier/round-weighted K as a separate, larger change. This
ticket is that follow-up: implement UTS's whole K-factor philosophy
(tournament context + current-rating-based K) as a coherent ALTERNATIVE to
our current matches-played-decay K, and honestly compare the two bundles -
not an isolated single-factor ablation, a whole-philosophy swap, per Pete's
actual ask ("test how UTS elo compares to ours").

Reference: `EloCalculator.java`, github.com/mcekovic/tennis-crystal-ball,
commit `fcda1e607deb289369a0ce37d0f8c756b97910a6` - fetched and read
directly (not just the screenshot summary, which turned out to describe
only the LEVEL switch; the reference's ROUND switch, its `default: 0.65`,
and its `DELTA_RATING_CAP=200` are pieces the screenshot didn't fully
capture).

## Answer

### What was built

`uts_context_k()` / `uts_kfunction()` (new, `elo_baseline.py`) implement the
reference's full three-part multiplicative stack:

1. `kFactor(level, round, bestOf, outcome) = 32 * level_mult * round_mult *
   bestof_mult * wo_mult` - context only, shared by both players.
2. `kFunction(rating) = 1 + 18 / (1 + 2^((rating-1500)/63))` - PER PLAYER,
   evaluated on that player's OWN current rating.
3. `newRating = rating ± clip(kFactor(...) * (1-p_win) * kFunction(rating), 200)`
   - the reference's own `DELTA_RATING_CAP=200` is genuinely load-bearing:
     `kFunction` alone reaches ~10x near START, so `32 * kFunction(1500) =
     320` for a full-weight (Slam, Bo5) match - well above 200.

`run_elo()` gained a new `k_mode` parameter (`'sackmann'` default, or
`'uts'`), REPLACING `k_factor(n)`'s matches-played decay entirely when
`k_mode='uts'` - a different K-philosophy, not a combination with it, per
the task's instruction. `decay` (layoff/inactivity pull toward START) is
kept as a separate, orthogonal axis in both modes.

**Level-code mapping, a real judgment call, documented plainly.** This
archive's `tourney_level` vocabulary is NOT tennis-crystal-ball's own
normalized scheme (checked via `df['tourney_level'].value_counts()`, not
assumed - codes present: `G,F,M,D,O,A,W,P,PM,I,T1-T5,C`, numeric ITF
prize-money tiers `10..100`, `CC,E,J`, two `+H` hybrid codes). Mapped:
`G=1.00, F=0.90, M=0.85, PM=0.85 (WTA Premier Mandatory ~ Masters-equivalent),
O=0.80, A=0.75, P=0.75 (WTA Premier ~ ATP-500-equivalent), D=0.70, I=0.70,
W=0.70 (WTA's post-2021 simplified single-tier code), T1-T5=0.70, CC=0.70`.
Everything else (Challenger `C`, all numeric ITF tiers, exhibitions,
juniors, the `+H` codes) falls to the reference's OWN `default: 0.65` - not
a new choice invented here. **Round switch quirk, kept faithfully**: the
reference has no `default` case in its round switch, so every qualifying
round (`Q1-Q5, ER`) falls through UNMULTIPLIED - i.e. gets the SAME full
weight as a Final. Confirmed by reading the Java source directly, not
assumed; genuinely counter-intuitive, kept as-is rather than "fixed."
**Walkover multiplier** is wired for fidelity but structurally never fires
on this archive - `prepare()` already drops every RET/W/O/DEF/ABD row.

### Sanity checks, done before trusting any comparison number

1. **Correctness gate on the refactor itself**: `run_elo(df, blend=0.0,
   decay=0.3)` with the new (default) `k_mode='sackmann'` reproduces the
   already-committed **0.6120** holdout log-loss exactly - the refactor
   that added `k_mode` did not silently change the pre-existing Sackmann
   path.
2. **`uts_kfunction()` matches the reference's own documented checkpoints
   exactly**: 10.000/5.495/1.640/1.073/1.008 at ratings 1500/1600/1800/2000/
   2200 (task brief's ~10x/~5.5x/~1.64x/~1.07x/~1.008x, confirmed to 3dp).
3. **Rating-range sanity check on REAL archive ratings, not just the
   formula in isolation** (`tune_uts_k.py` STEP 1): 44,585 rated players
   from the Sackmann-mode fit, ratings range 950-2734 (p1=1191, p50=1446,
   p99=2090) around `START=1500` - so the scale itself isn't badly
   mismatched. But **90.7% of these ratings still sit in kFunction's
   "provisional" zone (kFunction>3.0)**, i.e. still moving fast rather than
   stabilized. This is a real, structural difference from whatever
   population UTS's own reference implementation was tuned against: this
   archive is 89% lower-tier (qual/challenger/ITF) matches, so most rated
   players never accumulate enough tour-level matches to converge the way
   a top-pro-focused Elo would. Flagged here as the likely mechanism behind
   the result below, not asserted without evidence - see the ablation.
4. **Anchor checks pass on every UTS arm tested** (Sinner/Alcaraz,
   Sabalenka/Swiatek top-5 active, no <10-match player in top 20).

Code committed (`45336a8`) after these four checks passed, before trusting
the log-loss comparison, per the task's git-discipline instruction.

### The headline bundle-vs-bundle comparison

Same exact 2024-25 market-odds-joined holdout (n=8,598) as every prior
round. `blend=0.0` held fixed throughout (unrelated axis, untouched).

| Config | ALL logloss | ATP | WTA |
|---|---|---|---|
| **Current best** (Sackmann K, ordering-fixed, decay=0.3) | **0.6120** | 0.6173 | 0.6065 |
| UTS K, decay=0.3 (apples-to-apples bundle) | 0.6127 | 0.6156 | 0.6096 |
| UTS K, decay=0.0 (sensitivity check) | 0.6139 | 0.6164 | 0.6113 |
| Hierarchical (v25+round9, cited, not rerun) | 0.6275 | - | - |

**UTS's K-philosophy bundle is worse than ours on this holdout, by a real
but small margin: +0.0007 log-loss (decay=0.3, the fair apples-to-apples
comparison), or +0.0019 with decay off.** Not a wash in the "no detectable
difference" sense - it's a small, consistent, anchor-clean loss.

**A genuine, opposite-signed tour split hides inside that net number.**
UTS's K *improves* ATP (0.6156 vs 0.6173, Δ−0.0017) but *degrades* WTA
(0.6096 vs 0.6065, Δ+0.0031) - the WTA degradation is larger in magnitude
and outweighs ATP's gain in the combined number. This mirrors a pattern the
hierarchical-model tuning effort has hit repeatedly this session (WTA
behaving differently from ATP under changes to evidence-weighting
mechanisms) - not chased further here, flagged as a possible shared root
cause worth someone's attention across both efforts.

**Decay sensitivity confirms the two mechanisms are complementary, not
redundant or conflicting.** UTS-K with `decay=0.0` (0.6139) is worse across
every cut than UTS-K with `decay=0.3` (0.6127) - our existing
layoff/inactivity mechanism (time-since-last-match) is doing real,
additive work alongside UTS's kFunction (current-rating-level), not
fighting it or double-counting the same signal. `decay=0.3` is confirmed
the fairer, and also simply the better, choice for any UTS-K arm.

### Secondary diagnostic (per the task's "if time permits" note): which piece of the bundle is doing the work?

Ablated the bundle into its two components, `decay=0.3` fixed throughout
(diagnostic only - not scored as a real candidate, not anchor-checked):

| Variant | ALL | ATP | WTA |
|---|---|---|---|
| Context only (level/round/best-of, kFunction fixed=1.0) | 0.6131 | 0.6159 | 0.6102 |
| kFunction only (rating-based, context fixed=32) | 0.6143 | 0.6186 | 0.6099 |

**The tournament-context multiplier (level/round/best-of) is almost the
entire story, in both directions.** Context-only's ATP number (0.6159) is
nearly identical to the full bundle's (0.6156) and clearly better than
Sackmann's (0.6173) - it alone explains almost all of ATP's improvement.
kFunction-only's ATP number (0.6186) is actually WORSE than Sackmann's
baseline (0.6173) - the rating-based piece is a net negative for ATP on its
own. Both pieces contribute similarly to WTA's degradation. **Read together
with the Step 1 sanity check**: kFunction's design premise (fast K for
still-separating ratings, stable K once a player's true skill is resolved)
assumes a population that mostly reaches the "stable" regime - this archive
mostly doesn't (90.7% of ratings still provisional), so kFunction spends
most of its time amplifying noise from thin-history players rather than
doing its intended stabilizing job. The context multiplier's improvement is
a more mundane, more explicable effect: it downweights lower-tier/early-
round matches relative to Sackmann's flat matches-played K (which does not
care about tournament importance at all), and that appears to help ATP
specifically.

## Honest read: is UTS's K-philosophy better, worse, or a wash?

**Worse, on net, by a small but real margin (+0.0007 log-loss on the fair
comparison) - not a wash, not a win.** The mechanism is reasonably well
understood, not just "the number went the wrong way": UTS's rating-based
kFunction is built for a population that converges to stable, separated
ratings, and this archive's real composition (dominated by lower-tier/ITF
churn) doesn't give it that population, so it mostly adds noise instead of
the stabilization it was designed for. The tournament-context piece alone
is closer to a genuine partial win (helps ATP meaningfully, roughly matches
the full bundle's ATP number) but is outweighed in the combined metric by
a WTA-specific degradation whose root cause wasn't chased further this
round (flagged as a candidate shared-root-cause with the hierarchical
model's own repeated WTA-specific findings).

**Overfitting-risk caveat, same as ticket 25's**: this is evaluated
directly on 2024-25 with no separate tuning window of its own (a
pre-existing property of this baseline, not new here) - but this ticket
never tuned anything on this window (no grid search, no parameter
selection - the reference's own constants were ported as-is, untouched),
so the usual "did we just fit this window's noise" risk that applies to a
grid search does not apply here in the same way. The result is a single,
un-tuned bundle comparison, which is a materially lower-risk claim than a
selected-from-a-grid one.

## This does NOT change the "gap to Elo" number

Elo's frozen comparison-point number remains **0.6120** (Sackmann K,
ordering-fixed, decay=0.3) - UTS's K-philosophy was tested and REJECTED as
an improvement, so `elo_eval_predictions.parquet` (the canonical artifact)
is untouched by this ticket. The hierarchical model's gap to beat stays
**≈0.0155** (0.6275 − 0.6120), same as ticket 25 left it.

## What's still open

- The tournament-tier/round context piece alone (without kFunction) came
  close to a genuine ATP-specific win but was outweighed by a WTA
  degradation whose cause wasn't traced. A future ticket could isolate
  whether the WTA degradation is level-mapping-specific (the WTA tier
  codes W/P/PM/I/T1-T5 mapped here are a real judgment call, not
  code-verified against a WTA-specific reference) versus a genuine
  tour-level effect.
- Margin-of-victory (games-won) signal - still explicitly out of scope,
  same as ticket 25 left it.

## Code

`dropshotdata/scripts/elo_baseline.py` (`uts_context_k`, `uts_kfunction`,
`UTS_LEVEL_MULT`, `UTS_ROUND_MULT`, `run_elo(k_mode=...)`),
`dropshotdata/scripts/tune_uts_k.py` (the full bundle test + ablation,
reusable). Commits: `45336a8` (implementation + sanity checks), and this
writeup (docs-only, see `RESUME-beat-elo.md`'s newest section).
