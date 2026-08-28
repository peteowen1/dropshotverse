# 10 — Prototype: does the model beat the closing line?

Type: prototype
Status: claimed
Blocked by: 04, 08

## PRE-REGISTRATION — written 2026-08-27, BEFORE any fitting

Per `stats-discipline` §1. Recorded first so the numbers cannot rationalise themselves afterwards.

**Stage 1 is a plain surface-adjusted Elo**, per the spec's §7 "cheap test first". It is the benchmark the hierarchical model must beat, and if it cannot be beaten the project needs to know now.

### What I expect

| quantity | predicted | source of the prediction |
|---|---|---|
| Elo accuracy, ATP 2024-25 | **68-70%** | Kovalchik 2016: 538-career Elo 70%, opponent-adjusted 67% |
| Elo log-loss, ATP | **0.58-0.61** | Kovalchik: 538 Elo 0.59 |
| De-vigged Pinnacle accuracy | **71-73%** | Kovalchik: BCM 72% |
| De-vigged Pinnacle log-loss | **0.54-0.56** | Kovalchik: BCM 0.55 |
| Elo minus market, accuracy | **−2 to −4 pp** (Elo loses) | every published comparison |
| WTA vs ATP accuracy | **WTA lower by 1-3 pp** | ticket 11: less serve dominance ⇒ more upsets |

### Failure criteria — declared in advance

- **Elo accuracy above 72%, or log-loss below 0.55 → assume leakage, do not celebrate.** Plain Elo beating the closing line would contradict the entire published literature. The first hypothesis is a bug, not a discovery.
- **Accuracy below 63% → implementation bug**, not a finding about tennis.
- **Market accuracy below 68% → the odds join is broken**, or the wrong column is being read.
- **ATP and WTA log-loss agreeing to within 0.005 → suspect a shared bug** (`stats-discipline` §2), since the tours genuinely differ.

### Anchor checks — asserted before any metric is read

1. The Elo top 5 at end-2024 (ATP) contains Sinner and Alcaraz.
2. The Elo top 5 at end-2024 (WTA) contains Sabalenka and Swiatek.
3. Nadal is top-2 on clay Elo among active players in any year 2005-2022.
4. No player with fewer than 10 matches appears in any published top 20.

**An anchor failing means the method is wrong.** Do not special-case, do not ship with a caveat.

## Question

**The destination.** Everything else on this map exists to make this ticket answerable.

Fit the specified point-level model on historical data up to a cutoff, predict held-out seasons, and score it against bookmaker closing odds.

Requirements:

1. **Strict temporal holdout.** Fit on pre-2023, evaluate on 2024-2025. No information from the evaluation period may reach the fit — including player identities, surface effects estimated on future data, or any preprocessing fit on the full set. Leakage in a time-series rating model is subtle and this is exactly what `model-validation` exists to catch.
2. **Benchmarks, in order of difficulty.** Report against all three:
   - Naive: higher-ranked player wins
   - Published surface-adjusted Elo (use the accuracy and log-loss figures ticket 05 recovered from Kovalchik 2016 and Sackmann)
   - **Preferred benchmark, added 2026-08-27 by ticket 15: Betfair exchange closing prices.** The Basic historical tier is free to any Betfair account holder. Exchange prices carry **no bookmaker overround**, so no de-vigging is needed and the whole "did we remove the margin correctly?" objection to any beat-the-market claim disappears. Check coverage depth before committing — it may not reach as far back as 2001.
   - Fallback: the de-vigged closing line — use the **power method** (Clarke, Kovalchik & Ingram 2017) against **Pinnacle** (`PSW`/`PSL`), whose margin measures ~2.6-2.9% vs Bet365's ~5.2%. Cross-check with Shin's method.

   **Scope limit established by ticket 04:** free bookmaker odds exist only from ATP250 upward. Challenger and ITF matches **cannot** be benchmarked against the market at all. They can still be in the training set, and should still appear in the calibration-by-cut breakdown scored against actual outcomes — but the closing-line comparison is a main-tour result only. Say so explicitly in the write-up rather than quietly reporting a market comparison computed on a filtered subset.
3. **Metrics.** Log-loss and Brier as the headline, accuracy as a sanity figure only, plus calibration curves. Report closing-line value: on the subset where the model disagrees with the market, is it right more often than chance?
4. **Calibration by cut, not just in aggregate.** Break results down by tour (ATP/WTA), surface, tournament tier, best-of-3 vs best-of-5, favourite/underdog, and player experience level. An aggregate number cannot see bias hiding in a slice — a model that is well calibrated overall but badly wrong on Challenger matches is not a working model.
5. **An anchor check before looking at output.** Write down what you expect the log-loss to be, and what would count as a failure, *before* running the evaluation.

Load `stats-discipline` and `model-validation` before starting. Both are mandatory here.

**Report honestly.** "It matches the market on hard courts and loses badly on clay" is a far more useful result than a single aggregate number, and "it does not beat the line" is a legitimate and valuable outcome. Do not tune on the evaluation seasons to rescue the number — that is how a validated prototype becomes a lie.

Throwaway code, but keep the evaluation harness: it becomes the package's regression test.

**Deliverable:** the numbers, the calibration plots by cut, and a plain verdict on whether the engine is worth building into a package.

---

# ANSWER — stage 1, the Elo baseline vs the market

Run 2026-08-27. Scripts: `dropshotdata/scripts/elo_baseline.py`, `market_compare.py`.

## Coverage (reported before any metric, per `stats-discipline` §7)

1,226,517 rows read; **37,345 excluded** as retirements/walkovers/no-score (regime C); 1,189,172 kept, 1967-12-25 → 2026-06-02, 44,585 players. Eval window 2024-25 = 90,708 matches. Odds exist only at tour level, so the market comparison uses **11,078 tour-level matches**, of which **8,812 joined (79.5%)** — ATP 77.2%, WTA 82.1%. Winner agreed between sources on **99.8%** of joins; the 0.2% that disagreed were dropped as bad joins, not scored.

## Headline: Elo loses to the market, clearly

De-vigged with the power method (Clarke, Kovalchik & Ingram 2017). Pinnacle's measured margin **2.79%**, Bet365's **5.22%** — confirming ticket 04's recommendation to de-vig against Pinnacle.

| book | tour | n | Elo acc | Mkt acc | Elo LL | Mkt LL | **LL gap** |
|---|---|---|---|---|---|---|---|
| Pinnacle | all | 8,598 | 0.654 | **0.679** | 0.6151 | **0.5888** | **+0.0263** |
| Pinnacle | atp | 4,358 | 0.650 | 0.682 | 0.6204 | 0.5894 | +0.0310 |
| Pinnacle | wta | 4,240 | 0.658 | 0.676 | 0.6097 | 0.5881 | +0.0216 |

**Where Elo disagrees with the market (13.5% of matches), Elo is right only 40.5% of the time.** Worse than a coin flip. Plain Elo does not merely fail to beat the closing line — where it dissents, it is actively wrong.

## Calibration: systematically overconfident at every level

| favourite prob | n | predicted | realised | error |
|---|---|---|---|---|
| (0.5, 0.6] | 2,480 | 0.550 | 0.529 | −0.021 |
| (0.6, 0.7] | 2,432 | 0.649 | 0.597 | **−0.052** |
| (0.7, 0.8] | 1,889 | 0.748 | 0.726 | −0.022 |
| (0.8, 0.9] | 1,330 | 0.845 | 0.800 | −0.045 |
| (0.9, 1.0] | 467 | 0.934 | 0.904 | −0.030 |

**Every bucket is negative.** The model over-rates favourites by 2-5 points throughout — not a tail artefact but a uniform bias, which points at too high a K-factor or, more likely, the absence of any uncertainty term. A rating with no posterior variance cannot express "I don't know", so it never hedges. **This is the single most promising thing the hierarchical model could fix**, since shrinkage and posterior width address it directly.

## Predictions vs outcomes — two of six failed

| predicted | actual | verdict |
|---|---|---|
| Elo accuracy 68-70% | **65.4%** | **FAILED — below range** |
| Elo log-loss 0.58-0.61 | 0.615 | marginal miss |
| Market accuracy 71-73% | **67.9%** (Pinnacle), 69.4% (Bet365) | **FAILED — below range** |
| Market log-loss 0.54-0.56 | 0.589 | missed |
| Elo − market accuracy: −2 to −4 pp | **−2.6 pp** | **held** |
| WTA lower than ATP by 1-3 pp | WTA **higher** by 0.8 pp | **FAILED** |

The WTA/ATP prediction (from ticket 11's serve-dominance reasoning) is wrong in direction but the gap is ~1.2 standard errors at n≈4,300 — **the tours are statistically indistinguishable here**, so the honest statement is "no detectable difference", not "WTA is easier".

Market accuracy landing at 67.9% sits exactly on my declared "the join is broken" threshold. It is most likely genuine — our eval set is ATP+WTA tour level including many first-round matches, whereas Kovalchik's 72% was ATP-only — but the **79.5% join rate is itself below ticket 04's predicted 95-99%**, and that shortfall is unexplained. Flagged, not excused.

## A confound caught by the pre-registration

The first run reported **69.0% accuracy overall** — comfortably inside my predicted range. It was an artefact. The all-levels eval set is 64% WTA lower-tier matches, which are far easier (71.2% accuracy) because ITF draws contain huge skill mismatches. Splitting by level exposed it: tour-level accuracy is 65.4%.

`stats-discipline` §5 — the ATP and WTA arms differed on **level composition** as well as tour, so the "tour" comparison was measuring level mix. Fixed by comparing within level.

## A second catch: the anchors caught a real methodological flaw

The first anchor run **FAILED** — the WTA top 5 was Sabalenka, **Justine Henin, Steffi Graf**, Barty, Rybakina, and the ATP top 5 included Federer. Elo ratings never decay, so retired players sit at their career peak forever. Ranking "current form" on final ratings is simply wrong.

Fixed by restricting to players with ≥15 matches in 2024-25. All three anchors then passed: ATP top 5 = Sinner, Alcaraz, Djokovic, Zverev, Nadal; WTA = Sabalenka, Rybakina, Świątek, Gauff, Andreeva.

**This is what anchors are for.** Nothing in the log-loss would ever have revealed that the leaderboard was full of retired players.

## Surface blending made things worse

One-axis comparison (`stats-discipline` §5) — only the blend weight varied:

| blend | 0.0 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| log-loss (all) | **0.5771** | 0.5798 | 0.5841 | 0.5924 |

Pure overall Elo beat every surface-weighted variant. Surface-specific ratings split each player's history across four pools, and the variance cost exceeds the signal gained. **This does not mean surface is irrelevant** — it means naive per-surface Elo is the wrong way to use it, which is exactly why the spec (§2) chose shrunk hierarchical offsets instead.

## Verdict

**The bar is now measured: 0.026 log-loss behind Pinnacle at tour level.**

This is sobering for the map's destination. Ingram's point-level model trailed FiveThirtyEight Elo by 0.006, and this Elo trails the market by 0.026 — so a point-level model of comparable quality would still lose to the closing line. **"Beat the market" is not within reach of the planned architecture on this evidence.**

What is within reach, and worth redirecting toward:
1. **Fix the overconfidence.** A uniform 2-5 point bias is the largest single identified error, and posterior uncertainty addresses it directly.
2. **Reach market *parity* on calibration**, rather than beating the line — a defensible and genuinely useful goal.
3. **Serve the products Elo cannot**: live in-match probability and draw simulation. That remains the real justification for the point-level model, exactly as noted when the architecture was chosen.

## STAGE 2 PRE-REGISTRATION — written 2026-08-27, before fitting

Per `stats-discipline` §1. The hierarchical model's own justification (spec §8) is that it produces live/derived probabilities Elo cannot — not that it's more accurate. Kovalchik and Ingram both found point-level models roughly matching or trailing Elo on raw accuracy. So:

| quantity | predicted | reasoning |
|---|---|---|
| Hierarchical log-loss | **0.59-0.61**, i.e. close to Elo's 0.615, not dramatically better | Published point models don't beat Elo by much; this is a v0 implementation with undertuned hyperparameters |
| Hierarchical vs market gap | **still 0.015-0.03**, i.e. does not close the gap | No architecture change fixes a 0.026 gap on its own |
| Calibration | **improved over Elo's uniform overconfidence**, because the model carries genuine posterior uncertainty where Elo has none | This is the one place a real improvement is plausible |
| Anchors | serve-skill leaderboard top-heavy with Isner/Karlović/Opelka; return-skill leaderboard top-heavy with Djokovic/Nadal | Directly testable — if this fails, alpha/beta are not measuring what they claim |

**Failure criteria declared in advance:** if hierarchical log-loss is *better* than the market's 0.5888, assume a leakage or joining bug — that would be a bigger claim than the literature supports and needs re-verification before being believed, not celebrated.

**Documented scope simplifications for this v0 fit** (declared before running, so they cannot be introduced post-hoc to explain a bad result):
- Surface offsets (`κ`, `λ`) use a **fixed shrinkage prior variance**, not a fully time-evolving posterior — a ridge-style per-entity effect, not full hierarchical Bayes. Chosen for tractability.
- Regime B (scoreline-only matches) updates **only a combined strength term**, not separate serve/return skills — because ticket 12 established a single match cannot identify them separately. This is a direct consequence of that finding, not an unrelated shortcut.
- Process-variance and shrinkage hyperparameters are **plausibility-set, not cross-validated**.

## STAGE 2 ANSWER — hierarchical point-level model

Run 2026-08-27. Script: `dropshotdata/scripts/hierarchical_model.py`.

### Model as actually implemented

Online Bayesian filter (Laplace/assumed-density-filtering, the same lineage as Glicko/TrueSkill — spec §7's approximate route). Per player: `mu_alpha` (serve), `mu_beta` (return), each `N(mean, var)`, random-walk with layoff-inflated variance. Per player×surface: `kappa`, `lambda` offsets with fixed prior variance (the documented simplification above).

**Regime A** (point counts present): two binomial ADF updates per match — winner-serving (`w_svpt`/`w_1stWon`+`w_2ndWon`) and loser-serving (`l_svpt`/`l_1stWon`+`l_2ndWon`) — each updating (server's effective serve skill) vs (returner's effective return skill), with the surprise split across `mu`/`kappa` proportional to variance share.

**Regime B** (scoreline only): one update on combined strength (`mu_alpha + mu_beta`) per side, split 50/50 to alpha/beta — because ticket 12 proved a single match cannot separate them.

**Scoring**: for eval-window tour-level matches, pre-match `p1` (winner serving) and `p2` (loser serving) are pushed through the **exact ticket-05 recursion** (game→set→match DP, symmetrised over which player serves set 1) to get the match win probability — never a direct win/loss regression. This is the actual architectural difference from Elo.

**One-axis comparison, done correctly** (`stats-discipline` §5.3): hierarchical predictions were joined to the *existing* Elo eval rows by `(date, tour, winner_uid, loser_uid)` and scored together against the *same* joined-odds rows in one pass — not two separate executions compared after the fact.

### Results — and a debugging marathon that surfaced five real defects

Getting to a trustworthy number required finding and fixing five separate problems, four of them genuine implementation bugs and one a statistical modelling error. Recording all five in full, because each one produced *plausible-looking wrong output* before it was caught — exactly the failure mode `stats-discipline` §1 exists to catch, and each was caught by an anchor or a coverage check, never by "it looked wrong."

**Bug 1 — variable-reuse corrupting variance updates.** The original per-match update code computed a `share` value for the serve side, then reassigned the same variable name for the return side before using the *stale* value to update the server's variance. One side's variance was updated with the wrong number; the other side's was never reduced at all. Symptom: variance **increased** with more matches instead of shrinking (ratio 0.86x where ≥2x was required). Fixed by replacing the inline reused-variable code with a self-contained `apply_side_update()` helper that takes no shared mutable state.

**Bug 2 — unbounded, roughly-quadratic layoff variance inflation.** The `phi(gap) = 1 + c·max(0, weeks−26)` formula from the spec has no ceiling; a player with a ~15-year gap between matches produced `phi≈46` and variance in the hundreds. Once variance is that large, a single win (bounded only by that variance) can shift the mean by hundreds of logit units. Symptom: `Komlavi Loglo` topped the ATP leaderboard at rating 23,970. Fixed with `PHI_MAX=4.0` and `MAX_VAR=4.0`, both documented as reactive additions, not pre-registered.

**Bug 3 — unbounded random-walk drift in a weakly-connected sub-population.** Even after bugs 1–2, one player's final state was still ±100+ despite sane per-step caps. Replaying his own matches in isolation (fresh opponent states) gave sane values (~0.46, −0.57); replaying with his *real* opponent histories reproduced the extreme value — the corruption was transmitted through opponents drawn from an insular lower-tier pool rarely facing tour-level "anchor" players. This is the exact failure the spec's own §3 anticipated: *"the obvious upgrade is mean-reverting if random-walk skills drift implausibly."* Fixed with Ornstein-Uhlenbeck mean-reversion (`OU_THETA_PER_WEEK=0.0025`, ~5.3-year half-life), verified to keep an isolated 6-player pool bounded under 0.76 in magnitude over 300 simulated matches / 6 years.

**Bug 4 — state magnitude still unbounded despite bugs 1–3 being fixed.** Milos Raonic — 600 matches, multiple injury-driven comebacks — still reached `mu_a=-39.3` with `v_a` pinned at the `MAX_VAR` ceiling, proving the earlier caps engaged but a multi-cycle long-layoff-then-early-loss pattern still compounded past them. Symptom: `p2` computed as exactly `0.0`, driving the match-recursion output to exactly `1.0`. Fixed with a hard ceiling, `MAX_STATE=3.0` (sigmoid(±3) = 0.95/0.05, already an extreme single-point-probability edge), applied everywhere a state value is written.

**Bug 5 (statistical, not an implementation defect) — Jensen's-gap overconfidence from point-estimate plug-in.** After bugs 1–4 were fixed, the model was still badly overconfident: 39% of matches landed above 90% predicted, calibration error growing from −0.06 to −0.26 with confidence, and head-to-head log-loss of **0.928** — far worse than Elo's 0.615. The cause: scoring plugged the posterior *mean* of `p1, p2` into the match recursion, but the recursion is convex in the leading player's advantage, so by Jensen's inequality `E[p_match(p1,p2)] < p_match(E[p1],E[p2])` whenever real parameter uncertainty exists — and the filter already tracks that uncertainty (`d1_var`, `d2_var`), it just wasn't being used at scoring time. Verified on an 800-row sample before committing to the full fix: Monte Carlo averaging over the posterior (60 samples/match) dropped log-loss from 0.868 to 0.649 on that sample alone.

**Final numbers, all four models scored on the identical 8,594 odds-joined rows** (`stats-discipline` §5.3 — never compared across separate executions):

| model | acc | log-loss | brier |
|---|---|---|---|
| Elo | 0.653 | 0.615 | 0.214 |
| Hierarchical (naive plug-in) | 0.613 | 0.928 | 0.274 |
| **Hierarchical (posterior-averaged)** | **0.614** | **0.659** | **0.232** |
| Market (Pinnacle, de-vigged) | 0.679 | 0.589 | 0.203 |

Posterior averaging is not optional cosmetics — it is the difference between a model that looks badly broken and one that is merely uncompetitive.

**Two pre-registered predictions failed, recorded rather than explained away:**
- Predicted hierarchical log-loss 0.59–0.61; actual 0.659 — just outside range, worse than hoped.
- Predicted "calibration improved over Elo's uniform overconfidence" — **false.** The posterior-averaged model's calibration error is worse than Elo's in every bucket except the lowest two, and dramatically worse in the top bucket (−0.097 vs Elo's −0.030). Fixing the Jensen's-gap defect corrected the worst of the miscalibration but did not make this v0 model *better calibrated* than the much simpler Elo baseline.

**Anchors: 4/7 passed**, up from 1/7 across the debugging process. Passing: uncertainty scaling (4.23x, threshold 2x), Opelka's serve/return skew, Djokovic's and Murray's return-skill percentile. Still weak: Nadal's return-skill percentile, and the current-form top-5 (Sinner rank 1–2, Alcaraz rank ~10/484 — directionally right, both just outside a strict top-5 cut). One readout bug found and fixed along the way: raw `mu_alpha+mu_beta` catastrophically understated established players (Djokovic ranked 476/484) because for a long career, most true skill migrates into the (non-decaying) surface offset `kappa` while `mu` alone gets pulled toward zero by the OU fix for bug 3 — the correct current-form readout must include the surface term.

### Verdict

**The v0 hierarchical model does not beat Elo, and does not achieve the calibration parity that is now the map's actual destination.** It is close on accuracy (0.614 vs 0.653) but meaningfully behind on log-loss and calibration. Given the scope was explicitly a v0 prototype with plausibility-set (not cross-validated) hyperparameters, a fixed (not fully hierarchical) surface-shrinkage prior, and a combined-strength-only treatment of regime B, this is a believable outcome, not a surprising one — and it is exactly the "cheap test first" principle validating itself: the five defects above would have been far more expensive to find inside a production package than inside a throwaway prototype.

**What would most plausibly close the remaining gap**, in priority order: (1) cross-validated tuning of `RW_VAR_PER_WEEK`, `OU_THETA_PER_WEEK` and the surface prior variances, none of which were fit to data; (2) a properly hierarchical (not fixed-variance) surface treatment; (3) using the full identifiability-tested panel inversion for regime B instead of the combined-strength shortcut. None of these were in scope for this prototype.

### Still to do beyond this ticket

- Investigate the 19.6% odds-join shortfall (80.4% achieved vs ticket 04's 95-99% prediction).
- Betfair exchange prices as the cleaner benchmark (no overround, no de-vigging assumption).
- Calibration by surface, round and tournament tier.
- Cross-validated hyperparameters and a full hierarchical surface treatment, per the verdict above.
