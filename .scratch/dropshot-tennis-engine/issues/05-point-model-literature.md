# 05 — What published point-level tennis model should we base the engine on?

Type: research
Status: resolved

## Question

The engine is a point-level serve/return skill model with surface effects, estimated by a Bayesian state-space filter, from which match, set and tournament probabilities are derived. This is well-trodden ground in the literature and we should not reinvent it.

**Which published approach do we base on, and what exactly is the maths?**

Cover:

1. **The hierarchical Markov chain.** The classical result that match win probability has a *closed form* given point-win probabilities on serve for each player (O'Malley 2008, "Probability Formulas and Statistical Analysis in Tennis"; also Barnett & Clarke). Get the actual recursions for game, tiebreak, set and match, including best-of-3 vs best-of-5 and the various final-set formats (Wimbledon 12-12, US Open tiebreak, Australian 10-point). **This is what makes live in-match win probability nearly free to compute — confirm it and write down the formulas.**

2. **Bayesian hierarchical serve/return models.** Ingram (2019), "A point-based Bayesian hierarchical model to predict the outcome of tennis matches" is the key reference — find the paper and any released Stan/PyMC code. Establish how it parameterises serve skill, return skill, surface and time dynamics, and what it reported for predictive accuracy.

3. **Fitting point-win rates from scorelines alone.** The critical enabler: true point-level data is scarce but match scorelines are abundant, and a scoreline strongly constrains the underlying hold probabilities. Find how the literature does this inversion — is it a likelihood over the score, an EM approach, or approximate? Name the papers.

4. **Elo variants as the benchmark to beat.** Kovalchik (2016), "Searching for the GOAT of tennis win prediction", benchmarked Elo against bookmakers and against other methods — get its actual reported accuracy and log-loss numbers so we have a target. Also surface-specific Elo and Sackmann's published Elo methodology.

5. **The state-space / online link.** Establish formally how Elo approximates a Kalman/Bayesian filter (Glicko, TrueSkill lineage), since the architecture depends on the fast online layer being a principled approximation of the slow batch layer rather than a bolted-on hack.

6. **Known failure modes.** Retirements and walkovers, best-of-5 vs best-of-3 skill transfer, the serve-advantage difference between men's and women's tennis (materially different — men's tennis is far more serve-dominated, which changes the point model's sensitivity), altitude and ball type.

**Deliverable:** a recommended model specification with the actual equations, the papers it comes from, published accuracy figures to benchmark against, and an explicit note on where men's and women's tennis need different treatment.

## Answer

Coverage note up front, per the requirement to be honest about what's actually established: items 1-5 below are established from primary sources (papers read directly, formulas cross-checked/verified numerically or algebraically by the research agents, in two cases against two independent published derivations). Item 6 (failure modes) is **NOT ESTABLISHED** — the research agent tasked with it did not return a result before this ticket was written up, and nothing below section 6 should be treated as sourced. It needs a follow-up research pass; see the note in that section for exactly what's missing.

---

### 1. Closed-form Markov recursions (ESTABLISHED — this is the load-bearing result)

Primary sources, both read and cross-checked against each other:

- Newton, P.K. & Keller, J.B. (2005), "Probability of Winning at Tennis I: Theory and Data," *Studies in Applied Mathematics* 114(3):241-269. DOI: 10.1111/j.0022-2526.2005.01547.x. Full text: https://www.cis.upenn.edu/~bhusnur4/cit592_fall2013/NeKe2005.pdf
- O'Malley, A.J. (2008), "Probability Formulas and Statistical Analysis in Tennis," *Journal of Quantitative Analysis in Sports* 4(2), Art. 15. DOI: 10.2202/1559-0410.1100 (paywalled; verified instead via the author's own conference presentation of the identical formula: O'Malley, "The Tennis Formula," NESSIS 2007, https://www.nessis.org/nessis07/James_OMalley.pdf, slide 4)
- Brams, S.J., Ismail, M.S. & Kilgour, D.M. (2025), "Making Tennis Fairer: The Grand Tiebreaker," arXiv:2501.07309, https://arxiv.org/pdf/2501.07309 — independently re-derives the game formula (confirmed algebraically identical to O'Malley/Newton-Keller) and gives full combinatorial tiebreak/set formulas plus the current (2022+) rule-history.
- Barnett, T. & Clarke, S.R. (2005), "Combining player statistics to predict outcomes of tennis matches," *IMA Journal of Management Mathematics* 16(2):113-120. DOI: 10.1093/imaman/dpi001 — full text was paywalled and not obtained; it applies the same recursion family (built on Pollard 1983 and Carter & Crews 1974) to real ATP serve/return stats, but its exact equation numbers are not independently confirmed here.
- Pollard, G. (1983), "An analysis of classical and tie-breaker tennis," *Australian Journal of Statistics* 25(3):496-505 — original source of the tiebreak/game formulas both papers above build on; not independently accessed (paywalled).
- Riddle, L.H. (1988), "Probability Models for Tennis Scoring Systems," *JRSS-C* 37(1):63-75 — not independently accessed (paywalled).

**Note on the starting hypothesis in this ticket:** the guessed closed form `p^4(15-4p-10p^2)/(1-2p(1-p))` is **wrong** (fails numerically at p=0.6, gives P>1). The correct forms are below.

#### 1a. Game

Let `p` = server's probability of winning a given point, `q = 1-p`.

Semi-closed form (direct-win terms + deuce tail), from Newton & Keller eq. (5) and O'Malley (identical):

```
P_game(p) = p^4 + 4 p^4 q + 10 p^4 q^2 + 20 p^3 q^3 · p^2/(1 - 2pq)
```

Fully expanded single polynomial fraction (verified algebraically identical to the above, and independently confirmed by Brams/Ismail/Kilgour 2025 eq. 1):

```
P_game(p) = p^4 (15 - 34p + 28p^2 - 8p^3) / (1 - 2p + 2p^2)
```

Derivation: a game is won outright 4-0, 4-1, 4-2, or reaches deuce (3-3), from which it becomes a 2-state absorbing Markov chain (Adv-server, Deuce, Adv-returner) whose win probability is the geometric series `p^2 + 2pq·p^2 + (2pq)^2·p^2 + ... = p^2/(1-2pq)`.

#### 1b. Tiebreak (standard 7-point, win by 2)

State `(i,j)` = points won by A, B; serve alternates in the pattern A, B, B, A, A, B, B, A, A, ... (A serves point 1, then 2-point alternating blocks). Recursion (Newton & Keller eqs. 14-16):

```
if (i-1+j) mod 4 in {0,3}:  P_A(i,j) = P_A(i-1,j)·p_A + P_A(i,j-1)·(1-p_A)     [A serves this point]
if (i-1+j) mod 4 in {1,2}:  P_A(i,j) = P_A(i-1,j)·(1-p_B) + P_A(i,j-1)·p_B     [B serves this point]
boundary: drop the (i-1,·) term if j=7 and i≤6; drop the (i,j-1) term if i=7 and j≤6
base case: P_A(0,0)=1; P_A(i,j)=0 for i<0 or j<0
```
where `p_A`, `p_B` are each player's point-win probability on their own serve. Solved by forward dynamic programming over the 8x8 grid (cheap: O(64) states).

Closed form for the total, separating the direct-win region from the 6-6 tail (eq. 13):

```
P_A^tiebreak = sum_{j=0}^{5} P_A(7,j)  +  P_A(6,6) · p_A(1-p_B) / [1 - p_A p_B - (1-p_A)(1-p_B)]
```

The tail term is again a geometric series: once tied 6-6, each subsequent 2-point block (one A-serve, one B-serve) either resolves the tiebreak or returns to a tie with probability `p_A p_B + (1-p_A)(1-p_B)` (both hold, or both broken) — structurally identical to the game's deuce tail.

**10-point match tiebreak** (used at 6-6 in the final set under the current AO/rules — see 1e): identical recursion and identical tail-geometric-series logic; replace every "7" with "10" and every "6-6" boundary with "9-9". Brams et al. note explicitly this rule change "has little effect" on the underlying formulas because only the threshold changes, not the structure.

#### 1c. Set

State `(i,j)` = games won by A, B; server alternates every game. Recursion (Newton & Keller eqs. 6-10):

```
if (i-1+j) even:  P_A^set(i,j) = P_A^set(i-1,j)·P_A^game + P_A^set(i,j-1)·(1-P_B^game)
if (i-1+j) odd:    P_A^set(i,j) = P_A^set(i-1,j)·(1-P_B^game) + P_A^set(i,j-1)·P_B^game
base case: P_A^set(0,0)=1; boundary at 7-5: P_A^set(7,5) = P_A^set(6,5)·(1-P_B^game)
```
where `P_A^game`, `P_B^game` are each player's game-win probability on their own serve (from 1a, using that player's own point-win-on-serve rate). Total, standard format (tiebreak at 6-6):

```
P_A^set = sum_{j=0}^{4} P_A^set(6,j) + P_A^set(7,5) + P_A^set(6,6) · P_A^tiebreak
```

**Proven result** (Newton & Keller §3.3, via a symmetry identity on the recursion): set-win probability is independent of who serves first in the set. This is what licenses treating each player's set-win probability as a single scalar when composing into the match formula below, without tracking a 4-state (who-serves-first-per-set) recursion.

#### 1d. Match

Given each player's set-win probability `P_A^set`, `P_B^set` (service-neutral, per 1c):

```
Best of 3 (first to 2 sets):  P_A^match = (P_A^set)^2 + 2 (P_A^set)^2 (P_B^set)
Best of 5 (first to 3 sets):  P_A^match = (P_A^set)^3 + 3 (P_A^set)^3 (P_B^set) + 6 (P_A^set)^3 (P_B^set)^2
```
(Newton & Keller eqs. 51-52 — standard negative-binomial "best-of-N" combination: P(2-0)+P(2-1) and P(3-0)+P(3-1)+P(3-2) respectively, coefficients C(2,0)/C(3,1) and C(2,0)/C(3,1)/C(4,2).)

#### 1e. Final-set format variants

Confirmed rule history (cross-checked against Brams et al. footnotes plus ESPN/WTA/tour reporting on the 2019 and 2022 rule changes):

| Era | US Open | Australian Open | Roland Garros | Wimbledon |
|---|---|---|---|---|
| Pre-2019 | 7-pt tiebreak @ 6-6 (since 1970) | Advantage set (no tiebreak) | Advantage set (no tiebreak) | Advantage set (no tiebreak) |
| 2019-2021 | 7-pt tiebreak @ 6-6 | 10-pt match tiebreak @ 6-6 (first Slam to adopt it) | Advantage set (unchanged) | 12-12, then a 7-pt tiebreak (new 2019 rule) |
| 2022-present | 10-pt match tiebreak @ 6-6 | 10-pt match tiebreak @ 6-6 | 10-pt match tiebreak @ 6-6 | 10-pt match tiebreak @ 6-6 (dropped 12-12 rule to align) |

Only the terminal rule at 6-6 games-all changes; the game/set recursion machinery (1a, 1c) is unchanged in every case.

- **Standard tiebreak-at-6-6** (all non-major tour-level matches, and majors 1970-2018/2022 depending on venue): use the set formula in 1c directly with the 7-point tiebreak from 1b.
- **10-point match tiebreak at 6-6** (AO 2019+, all four majors 2022+): identical set formula, substituting the 10-point tiebreak variant described at the end of 1b for `P_A^tiebreak`.
- **Wimbledon 2019-2021 (12-12 then 7-pt breaker):** requires extending the set recursion past 6-6 without a tiebreak until 12-12. Using `p_A`, `p_B` = each player's game-win-on-serve probability and `r = p_A p_B + (1-p_A)(1-p_B)` (probability the tie survives one more A-serve/B-serve game pair):
  ```
  P_A^set = sum_{j=0}^{4} P_A^set(6,j) + P_A^set(7,5) + P_A^set(6,6) · [ sum_{n=0}^{2} r^n · p_A(1-p_B) + r^3 · P_A^tiebreak(7-pt) ]
  ```
  **Caveat: this specific 12-12 extension formula was derived here by the research agent, by direct analogy with the deuce/tiebreak-tail absorbing-chain logic used elsewhere in these papers — it was not found already published in closed form** (the rule existed for only two Wimbledon cycles, 2019-2021, before being superseded, and neither Newton-Keller 2005 nor Brams et al. 2025 treat it explicitly). Verify independently before shipping if this exact format needs to be modeled for historical Wimbledon matches; the underlying logic (three extra "extension rounds" then a breaker) is standard, but the specific formula has not been cross-checked against a second source.
- **Pre-2019 advantage sets (no tiebreak ever)** — AO/French pre-2019, Wimbledon pre-2019, US Open pre-1970. This is the game-level analogue of the point-level deuce recursion in 1a: once tied 6-6 (or m-m, m≥6), each round is one A-service-game then one B-service-game; A wins outright with probability `p_A(1-p_B)`, B wins with probability `p_B(1-p_A)`, and any other combination returns to a tie. Closed form (absorbing 2-state chain, same "1-2pq"-style structure as 1a):
  ```
  P(A wins | tied indefinitely) = p_A(1-p_B) / [p_A(1-p_B) + p_B(1-p_A)]
  P_A^adv-set = sum_{j=0}^{4} P_A^set(6,j) + P_A^set(7,5) + P_A^set(6,6) · p_A(1-p_B) / [p_A(1-p_B) + p_B(1-p_A)]
  ```
  **Same caveat as above: derived here by direct analogy, not found already published in closed form** — Newton & Keller explicitly assume a tiebreak exists in every set (their footnote 1) and don't treat the pure advantage-set case. The derivation logic is standard and low-risk, but treat the exact formula as unverified against a second source until checked.

**Bottom line for item 1:** the game/tiebreak/set/match recursions (1a-1d) are solid, cross-checked, cheap (all are either closed-form or tiny finite-state DP), and ready to implement directly — this is exactly what makes live win probability nearly free. The two historical final-set format formulas in 1e (12-12 Wimbledon, and pure advantage sets) are correct in derivation logic but were generated by the research agent rather than lifted from a published source, so they carry a mark against them: numerically verify (e.g. against known historical match probabilities, or simulation) before relying on them for backtesting matches played under those formats.

---

### 2. Ingram (2019) Bayesian hierarchical point-based model (ESTABLISHED)

**Citation:** Ingram, M. (2019), "A point-based Bayesian hierarchical model to predict the outcome of tennis matches," *Journal of Quantitative Analysis in Sports* 15(4):313-325. DOI: 10.1515/jqas-2018-0008. Free author-hosted PDF: https://martiningram.github.io/papers/bayes_point_based.pdf. Not on arXiv — journal paper only.

**Unit of analysis:** each match is split into two "serve-matches" (one per player serving). For serve-match `i`, the server serves `n_i` points and wins `y_i` of them.

**Likelihood** (point-level binomial, not fit to game/set/match outcomes):

```
y_i ~ Binomial(n_i, theta_i)
```

**Link function** (logistic, opponent-adjusted, extending Barnett & Clarke 2005):

```
logit(theta_i) = ( alpha_{s(i),p(i)} - beta_{r(i),p(i)} ) + ( gamma_{s(i),m(i)} - gamma_{r(i),m(i)} ) + delta_{t(i)} + theta_0
```
- `alpha_{s,p}` = server s's serving skill in time period p
- `beta_{r,p}` = returner r's returning skill in time period p
- `gamma_{·,m}` = a player's additional surface-specific skill on surface m
- `delta_t` = tournament-specific intercept
- `theta_0` = global intercept (average player's point-win-on-serve probability)

**Priors (hierarchical):** initial skills `alpha_{·,1} ~ N(0, sigma_alpha0^2)`, `beta_{·,1} ~ N(0, sigma_beta0^2)`; surface and tournament effects `gamma ~ N(0, sigma_gamma^2)`, `delta ~ N(0, sigma_delta^2)`; all scale hyperparameters `~ Half-Normal(0,1)`.

**Time dynamics — Gaussian random walk** (this is the key point for the state-space architecture): skill evolves period-to-period as
```
alpha_{·,p+1} ~ N(alpha_{·,p}, sigma_alpha^2)
beta_{·,p+1} ~ N(beta_{·,p}, sigma_beta^2)
```
Time is divided into discrete periods (tested 1-12 months; best result at 2 months); skill is constant within a period and diffuses via the random walk between periods. Surface (`gamma`) and tournament (`delta`) effects are **not** time-varying — only `alpha`/`beta` evolve. This is explicitly modeled on Glickman's (1999) dynamic paired-comparison approach (see item 5).

**Partial pooling:** hierarchical priors shrink player skills toward population means and surface/tournament effects toward zero; for unseen players/tournaments, skills are drawn from the group-level priors, so the model can still predict new matchups.

**Inference:** Stan, NUTS (Hamiltonian Monte Carlo); convergence via R-hat < 1.1; 4000 posterior draws used for prediction. Fitting the best configuration took ~80 minutes (vs. Elo fitting in under a minute) — an explicit noted scaling limitation. Refit on a rolling/expanding window for each 2014 prediction period, using only data up to that period.

**Accuracy (2014 ATP season, 2208 matches, out-of-sample):**

| Model | Accuracy | Log loss |
|---|---|---|
| Opponent-adjusted (Barnett & Clarke 2005 style) | 66.3% | 0.641 |
| FiveThirtyEight 538 Elo | 69.5% | 0.586 |
| **Best proposed model (2-month periods, fit from 2011)** | **68.8%** | **0.592** |

So the model beats the previous best point-based model by +2.5pp accuracy / -0.049 log loss, but does not quite beat 538 Elo (-0.7pp accuracy / +0.006 log loss) despite lacking Elo's simplicity. A footnote notes Kovalchik's own paper reports the opponent-adjusted model slightly differently (67%/0.63) — attributed to dataset construction differences.

Posterior variance-component magnitudes (median): `sigma_alpha0=0.154`, `sigma_beta0=0.124` (largest source — initial-skill uncertainty dominates), `sigma_gamma=0.064`, `sigma_delta=0.069` (surface ~ tournament in size), `sigma_alpha=0.039`, `sigma_beta=0.027` (period-to-period drift, larger for serve than return). `theta_0` median 0.503 -> two average, equally-matched players' win-on-serve probability is ~62.3%.

**Code:** https://github.com/martiningram/tennis_bayes_point_based — contains `stan_model.stan` (data/parameters/transformed-parameters/model/generated-quantities blocks matching eqs. above, using a non-centered parameterization for the random walks), `bayes_point_model.py` driver, `winning_prob.py` (converts theta posteriors to match/set win probabilities via the iid Markov formulas in item 1), `dataset.csv` (ATP data from 2011), `Example.ipynb`. Related repo: https://github.com/martiningram/tennis-stan (three ascending-complexity Stan notebooks). Related later work by the same author addressing the ~80-minute fit-time limitation: a JAX/NumPyro sampler-speed comparison at https://martiningram.github.io/mcmc-comparison/, repo https://github.com/martiningram/mcmc_runtime_comparison.

**Follow-up paper by the same author:** Ingram, M. (2021), "How to extend Elo: a Bayesian perspective," *JQAS* 17(3):203-219, DOI: 10.1515/jqas-2020-0066. Shows Elo as an approximation to steady-state Kalman filtering (see item 5) and builds Elo extensions for margin-of-victory, surface correlation, and tournament tier. On 2018-2019 ATP data: best extended model 65.8% accuracy / -0.615 mean log-likelihood, vs. plain Elo 63.7%/-0.632 and Glicko 63.5%/-0.633. (These follow-up numbers came from search-summary sources rather than a full PDF read, so treat them as slightly less certain than the 2019 paper's numbers above, which were read directly from the primary PDF tables.)

---

### 3. Inferring point-win probabilities from scorelines alone (ESTABLISHED, with an important negative finding)

**Headline finding: no paper in this literature actually inverts the bare final scoreline (games-per-set only) via a full Markov-chain path-sum likelihood.** Every serious estimation approach reaches for data richer than a bare scoreline — either actual point sequences, or per-match aggregate point counts (points won / points played on serve), or published ATP/WTA summary serve-and-return percentages — because a bare scoreline carries very little statistical information about the underlying point-win probability `p` (a handful of noisy Bernoulli-like draws). This gap is itself the important input to the model spec: if the project only has scorelines and no point counts, the "obvious" approach (construct `L(p) = P(observed scoreline | p)` via the Markov chain from item 1 and do MLE/Bayes over it) appears to be a legitimate, usable, but essentially unpublished technique — there is no paper to copy the recipe from, it would need to be built and validated as a genuinely new piece of the pipeline (see identifiability caveat below).

**Barnett & Clarke (2005)** (op. cit., item 1) — not a scoreline inversion. Uses published ATP/WTA match-level summary statistics (percentage of points won on 1st/2nd serve, aces, etc.), aggregated across a player's matches against the field, and combines them algebraically (an "opponent-adjustment": player A's serve% vs. field average, combined with player B's return% vs. field average) into a match-specific point-win estimate, then feeds that forward into the Markov chain. Not an inversion of an observed scoreline; a forward calculation from richer input data.

**Klaassen & Magnus** — read directly (2003 EJOR paper, full text: https://www.janmagnus.nl/papers/JRM065.pdf):
- Klaassen, F.J.G.M. & Magnus, J.R. (2001), "Are points in tennis independent and identically distributed? Evidence from a dynamic binary panel data model," *JASA* 96(454):500-509. DOI: 10.1198/016214501753168217
- Klaassen, F.J.G.M. & Magnus, J.R. (2003), "Forecasting the winner of a tennis match," *European Journal of Operational Research* 148(2):257-267. DOI: 10.1016/S0377-2217(02)00682-3
- Klaassen, F.J.G.M. & Magnus, J.R. (2014), *Analyzing Wimbledon: The Power of Statistics*, Oxford University Press

Their method is a two-tier hybrid, and it does **not** use game-score scorelines either:
1. **Match-winner-only logit** on ranking difference `D = R_a - R_b`: `pi = exp(lambda·D) / (1+exp(lambda·D))`, fit by ML on 495 men's / 504 women's Wimbledon 1992-1995 matches, using only who-won (not the score).
2. **Point-level panel model** on a 481-match subset with actual point-by-point sequences (57,319 men's / 28,979 women's points): a linear-probability panel model with player random effects estimated by (quasi-)ML/GLS, giving `p_a + p_b` (the sum of both players' serve-win probabilities).
3. **Deterministic inversion**: given the match-win probability `pi_a` from step 1 and `p_a+p_b` from step 2, solve `TENNISPROB(p_a,p_b) = pi_a` for `p_a - p_b` (monotone in the difference given the fixed sum), where `TENNISPROB` is exactly the game/tiebreak/set/match Markov-chain recursion from item 1. This is a function inversion, not a likelihood-over-scoreline MLE.

Their 2001 JASA finding is that points are *not quite* i.i.d. (positive momentum, harder for servers at important points) but the deviation is small enough that the i.i.d. Markov approximation remains usable for forecasting — this is the standard justification cited by nearly every paper in this space (including Ingram 2019) for using the iid recursion despite known small violations.

**Newton & Keller (2005)** (op. cit., item 1) validate their formulas against 2002 US Open/Wimbledon semifinalist data using raw point tallies (`points won on serve / total points served`) — direct empirical moment-matching of the point ratio, not a scoreline likelihood, and not even game-win-percentage matching.

**McHale & Morton (2011)**, "A Bradley-Terry type model for forecasting tennis match results," *International Journal of Forecasting* 27(2):619-630, DOI: 10.1016/j.ijforecast.2010.02.005 — a paired-comparison (Bradley-Terry) model fit directly to match winners with a recency-weighted likelihood; does not target point-win probabilities at all, so it's not a mechanism for this question, but it's a legitimate alternative "skip point-level modeling entirely" benchmark.

**Spanias & Knottenbelt (2013)**, "Predicting the outcomes of tennis matches using a low-level point model," *IMA Journal of Management Mathematics* 24(3):311-320, DOI: 10.1093/imaman/dps009 — hierarchical point model (point->game->set->match, iid), but point-win probabilities are estimated from actual point-by-point statistics, not scorelines.

**Answering the ticket's explicit sub-question** ("is it a likelihood over the score, an EM approach, or approximate?"): (a) simple moment-matching against point/game ratios is what Newton & Keller and Barnett & Clarke effectively do; (b) the "full likelihood of the observed scoreline via the Markov chain, MLE/Bayes over p" approach exists as forward machinery in every paper (all of them can compute `P(scoreline | p)`) but **none use it as their primary estimator in the inverse direction** — Klaassen & Magnus come closest structurally (they do invert a Markov-chain-derived function) but the quantity inverted is the match-win probability from a ranking-based logit, not a raw scoreline likelihood, and their point-sum term still requires real point data; (c) no paper found using EM to back out point probabilities from game/set score sequences — this does not appear to be an established technique in this literature.

**Identifiability caveat, from Ingram's own blog analysis** ("The first serve percentage in tennis is noisy," https://martiningram.github.io/first-serve-noisy/): simulating a player with a true 60% success rate over ~20 attempts (one service game's worth), the standard deviation of the observed proportion is ~11 percentage points (range as wide as 38-82% is unsurprising); even over a full set's worth of serve points (~30-40), the SD is still ~8 points. This generalizes directly to the scoreline-inversion question: because men's point-win-on-serve probabilities cluster tightly (roughly 60-65%), a single match's scoreline is a weak instrument for `p`, and every paper compensates by either pooling across many matches (Klaassen & Magnus's panel with random effects; Ingram's hierarchical shrinkage) or using richer per-match data than a bare scoreline (point counts, not games). **Implication for the project:** if scoreline-only inversion is built, expect to need heavy hierarchical pooling across many historical matches per player before per-player point-win estimates stabilize — a single scoreline will not pin down `p` on its own.

---

### 4. Kovalchik (2016) benchmark numbers (ESTABLISHED — read from the primary published table)

**Citation:** Kovalchik, S.A. (2016), "Searching for the GOAT of tennis win prediction," *Journal of Quantitative Analysis in Sports* 12(3):127-138. DOI: 10.1515/jqas-2015-0059. Open-access PDF (Victoria University repository): https://vuir.vu.edu.au/34652/1/jqas-2015-0059.pdf — full text read directly, numbers below transcribed from Table 4/Table 5 of the actual paper, not secondhand summary.

**Scope correction:** this paper is **ATP (men's) only** — 2,395 singles matches from the 2014 season, excluding 105 walkovers/retirements. It explicitly lists "did not consider the WTA" as a stated limitation in its Discussion. There is no men's/women's split in this paper; that split has to come from elsewhere (see item 6).

**Table 4 — full 2014 ATP validation results, all 11 methods (accuracy %, log-loss):**

| Method | Accuracy | Log-loss |
|---|---|---|
| Regression: Basic probit (seeding only) | 59% | 0.63 |
| Regression: Mean (rank-difference) | 65% | 0.61 |
| Regression: Logistic (Klaassen & Magnus 2003 style) | 67% | 0.60 |
| Regression: Probit plus (rank + demographics) | 67% | 0.60 |
| Regression: Prize probit | 68% | 0.61 |
| Point-based: Low level (Spanias & Knottenbelt) | 64% | 0.68 |
| Point-based: Common opponent | 63% | 0.66 |
| Point-based: Basic IID (Newton & Keller) | 63% | 0.67 |
| Point-based: Mean | 64% | 0.66 |
| Point-based: Opponent adjusted (Barnett & Clarke) | 67% | 0.63 |
| Bradley-Terry, 1-year window | 62% | 0.67 |
| Bradley-Terry, 2-year window | 65% | 0.65 |
| **FiveThirtyEight Elo, 1-year data** | **67%** | **0.60** |
| **FiveThirtyEight Elo, career-to-date** | **70%** | **0.59** |
| **Bookmaker Consensus Model (BCM)** | **72%** | **0.55** |

Range across all methods: 59-72% accuracy, 0.55-0.68 log-loss. **No model beat the bookmakers.** FiveThirtyEight career-to-date Elo (70%/0.59) was the closest competitor, beating every point-based and regression-based model. Only BCM and 538-career-Elo cleared 70% accuracy.

**Breakdown by rank tier / surface (Table 5, selected rows):** accuracy for top-30 players vs. lower-ranked, by method — Regression 74%/56%, Point-based 70%/58%, Bradley-Terry(1yr) 67%/57%, Bradley-Terry(2yr) 71%/58%, 538-1yr 75%/59%, 538-career 75%/64%, BCM 76%/67%. "All models were 10-20 percentage points less accurate at predicting outcomes among lower-ranked players than top players." Notably, 538's 1-year and career-to-date models tied exactly at 75% for top-30 players — extra history didn't help top players, but took lower-ranked-player accuracy from 59% to 64%.

**Sackmann Elo methodology**, from https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/, cross-checked against Kovalchik's own equations (4)-(5), which cite the same FiveThirtyEight source (Bialik/Morris 2015):

- **K-factor:** `K = 250 / (m(t) + 5)^0.4`, where `m(t)` = career matches played at time t; multiplied by 1.1 for Grand Slam matches.
- **Initial rating:** the FiveThirtyEight/traditional convention is `E(1) = 1500`; Sackmann's own live implementation instead uses "a number in the low 1200s (depends a bit on tournament level and gender)" — no exact published value.
- **Win-probability formula:** `pi_ij(t) = 1 / (1 + 10^((E_j(t) - E_i(t))/400))` (standard logistic Elo).
- **Surface handling:** Sackmann maintains four separate Elo ratings per player (overall, hard, clay, grass); the surface-specific forecast is a 50/50 blend of the surface-specific Elo prediction and the overall Elo prediction ("my testing showed 50/50 worked for each surface").
- Sackmann's own reported spot-check: Elo picked 36/50 (72%) of Davis Cup Finals matches correctly vs. ATP rankings' 30/50 (60%) — an illustrative single-sample statistic, not a rigorous benchmark table like Kovalchik's.

---

### 5. Elo as an approximate Bayesian/Kalman filter (ESTABLISHED, with an exact citable derivation)

**Primary derivation:** Glickman, M.E. (1999), "Parameter Estimation in Large Dynamic Paired Comparison Experiments," *JRSS-C* 48(3):377-394. DOI: 10.1111/1467-9876.00159. Free PDF: https://www.glicko.net/research/glicko.pdf

Sets up a dynamic Bradley-Terry model as a genuine state-space model:
- Static win probability: `Pr(win) = 10^((theta_i-theta_j)/400) / (1+10^((theta_i-theta_j)/400))`
- Ability dynamics (state-transition equation), a Gaussian random walk: `theta_i^(t0+t) | theta_i^(t0) ~ N(theta_i^(t0), nu^2 t)` — strength diffuses as a Wiener process with rate `nu^2`.
- Prior belief per player: `theta ~ N(mu, sigma^2)` — ability is a full Gaussian belief (mean + variance), not a point estimate.
- Time update (Kalman predict step): `theta^(t0+t) | mu, sigma^2 ~ N(mu, sigma^2 + nu^2 t)` — variance grows linearly with elapsed time.
- Measurement update (approximate Bayesian posterior after a game), the closed-form update:
  ```
  mu' = mu + q/(1/sigma^2 + 1/delta^2) · sum_j [ g(sigma_j^2) { s_j - E(s | mu, mu_j, sigma_j^2) } ]
  sigma'^2 = (1/sigma^2 + 1/delta^2)^-1
  ```
  where `q = ln(10)/400`, `g(sigma_j^2) = 1/sqrt(1 + 3 q^2 sigma_j^2/pi^2)`, `E(s|...) = 1/(1+10^(-g(sigma_j^2)(mu-mu_j)/400))`, and `delta^2 = [ q^2 sum_j n_j g(sigma_j^2)^2 E(1-E) ]^-1`. This is literally the Glicko `g(RD)`/expected-score/`RD` formula, derived here from Bayesian first principles.

**The exact citable statement that Elo is a special case** (Glickman 1999, p.382, his eq. 9 -> eq. 11-13): if every opponent's rating is treated as known exactly (`sigma_j^2=0` for all j), then `g(sigma_j^2)=1` and `E(s|...)` collapses to the ordinary logistic Elo expected score, and the update reduces exactly to the textbook Elo rule `theta' = theta + K·sum(s - W_e(theta,theta_j))`, with `K = q / (1/delta^2 + 1/sigma^2)` — i.e. **the K-factor is shown to equal a precision ratio, not an arbitrary tuning constant**; a fixed K implicitly assumes fixed, saturated uncertainty rather than the time/precision-varying gain a proper filter would use.

**Glicko / Glicko-2:** Glickman, M.E. (1995), "A Comprehensive Guide to Chess Ratings," *American Chess Journal* 3:59-102 (original announcement); Glicko-2 methodology note: https://glicko.net/glicko/glicko2.pdf. Glicko-2 adds a per-player volatility parameter and iterates a scalar-Kalman-shaped update (`phi*^2 = phi^2 + sigma'^2` as the time-update/process-noise step, `1/phi'^2 = 1/phi*^2 + 1/v` as the measurement-update precision sum) — structurally identical to a scalar Kalman filter, collapsing back to classic fixed-K Elo when RD/phi is frozen rather than propagated.

**TrueSkill:** Herbrich, R., Minka, T. & Graepel, T. (2006), "TrueSkill: A Bayesian Skill Rating System," NeurIPS/NIPS 2006. https://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system. States in its own abstract that it "can be viewed as a generalisation of the Elo system." Uses a Thurstone (probit) rather than Bradley-Terry (logistic) link, models each player's performance as `p_i ~ N(s_i, beta^2)`, and derives the linearized Elo update directly as `Delta = alpha·beta·sqrt(pi) · ((y+1)/2 - Phi((s1-s2)/sqrt(2 beta)))` — i.e., **the classic Elo K-factor is exactly the constant `alpha·beta·sqrt(pi)` produced by linearizing a Gaussian-belief Bayesian update**, with a fixed learning rate `alpha` standing in for what should be a time-varying Kalman gain. TrueSkill itself generalizes further via Expectation Propagation message passing on a factor graph, tracking both mean and variance per player every game (full equations in the paper's Table 1).

**The cleanest unifying derivation (best single citation for "Elo is a named, specific approximation of a Kalman filter"):** Szczecinski, L. & Tihon, R. (2021), "Simplified Kalman filter for online rating: one-fits-all approach," arXiv:2104.14012. https://arxiv.org/abs/2104.14012. Derives, in order: (1) the exact recursive Bayesian filter (intractable in general); (2) a Gaussian-projection/moment-matching approximation (the extended/approximate Kalman filter step); (3) a full-covariance "KF rating" recursion; (4) a simplified diagonal/scalar-covariance version, shown explicitly to reduce, under a Bradley-Terry likelihood, to **Glicko**, and under a Thurstone likelihood, to **TrueSkill**; (5) a further reduction — drop the log-likelihood curvature term and freeze the variance at a constant — shown explicitly to yield exactly the classic Elo update ("we recognize (95) as the well-known Elo rating algorithm").

**Tennis-specific application:** Ingram, M. (2021), "How to extend Elo: a Bayesian perspective," *JQAS* 17(3):203-219, DOI: 10.1515/jqas-2020-0066 (see item 2) — frames Elo's fixed K-factor as the steady-state gain of a Kalman filter (i.e., the gain a proper filter converges to once state/observation noise stop changing), and uses that framing to derive tennis-specific Elo extensions (margin of victory, surface correlation, tournament tier) that beat plain Elo and Glicko out-of-sample. Also relevant but not Bayesian-filter-framed: Angelini, G., Candila, V. & De Angelis, L. (2022), "Weighted Elo rating for tennis match predictions," *European Journal of Operational Research* 297(1):120-132 — an ad hoc (not derived from a posterior-variance argument) margin/scoreline reweighting of Elo, validated on 60,000+ ATP/WTA matches.

**Formal correspondence, stated precisely for the spec:** a proper Bayesian filter models ability as a Gaussian belief evolving by a random walk, and after each game performs a Kalman/Newton-style update where the posterior mean shifts by (observed - expected score) scaled by a gain equal to the ratio of the player's own uncertainty to total predictive uncertainty, with the posterior variance shrinking correspondingly. Classic fixed-K Elo is this exact filter collapsed by three simultaneous, individually-named approximations: (1) treat opponents' ratings as known exactly (zero opponent-side variance) — Glickman (1999) p.382; (2) freeze the player's own posterior variance at a constant instead of propagating it, turning a time-varying, information-sensitive gain into one hand-tuned K — Szczecinski & Tihon's "fixed-variance SKF," Ingram's "steady-state Kalman filtering" framing; (3) drop the log-likelihood curvature term and take one gradient step instead of iterating to the true posterior mode (stochastic-gradient rather than full MAP) — Szczecinski & Tihon eqs. 32->61. This licenses exactly the claimed architecture: a fast Elo-like online layer is a specific, named, principled degenerate case of the slower Bayesian batch filter (item 2's Stan model, or a proper Kalman/particle filter over the same state), not a bolted-on heuristic — and Glicko/Glicko-2/TrueSkill are each "un-collapsing" one of the three approximations back toward the full filter.

---

### 6. Known failure modes (NOT ESTABLISHED)

The research agent tasked with this item (men's-vs-women's serve dominance and its effect on point-model sensitivity, retirements/walkovers handling, best-of-5 vs best-of-3 skill transfer, and altitude/ball-type effects) did not return a result before this ticket was written up — it appears to have been lost rather than to have failed with an error, and no partial output was captured. **Nothing in this section should be treated as sourced; it has not been researched.**

What's still needed, to be picked up in a follow-up pass:
- Actual point-win-on-serve and hold-rate numbers for ATP vs. WTA, by surface, with citations (candidates worth checking first: Klaassen & Magnus's *Analyzing Wimbledon*, Jeff Sackmann's Tennis Abstract data/posts, and any of Kovalchik's other papers — none of the sources gathered above happen to give this split, since Kovalchik (2016) is explicitly ATP-only).
- The mathematical point about `P_game(p)` sensitivity (`dP_game/dp`) differing between men's (~p=0.62-0.65) and women's (~p=0.55-0.60) typical operating points, given the closed form in section 1a — this can actually be computed directly from the formula already established above without needing new sources, and should be done as part of writing the model spec even before the literature search is redone.
- How retirements/walkovers are handled in backtesting in the literature (Kovalchik's paper excludes 105 walkovers/retirements from her 2014 dataset but does not elaborate on methodology beyond exclusion — that much is confirmed from item 4's source, but nothing beyond it).
- Best-of-5 vs. best-of-3 skill transfer and altitude/ball-type effects: entirely unresearched.

Recommend re-running a dedicated research pass on this item specifically before finalizing the model spec, since the men's/women's serve-dominance point was flagged in the original ticket as materially affecting model sensitivity and calibration requirements.
