# 08 — Specify the rating engine

Type: grilling
Status: resolved
Blocked by: 05, 07a, 11, 12

## Question

Write the actual model specification. Charting settled the *shape* — point-level serve/return skill, surface as a covariate, Bayesian state-space estimation, Elo as the fast online approximation and benchmark. This ticket settles the *detail*.

Decide and write down:

1. **Parameterisation.** What exactly are the latent quantities per player — a serve skill and a return skill, on what scale, and how do they combine into P(server wins point)? Logit-additive, or something else?
2. **Surface.** Per-player surface offsets, or the citius-style shared-shock × per-player-sensitivity structure (`citiusverse/citius/R/ability.R`) where only the interaction reorders rankings? The latter is far more parsimonious and is prior art in this ecosystem.
3. **Time dynamics.** The process model for how skill evolves — random walk, mean-reverting (Ornstein-Uhlenbeck), or something with explicit form/streak structure. Include the story for injury layoffs.
4. **The slow/fast relationship.** Precisely how the fast online update approximates the batch posterior, and how often the fast layer is re-seeded from the slow one. If these two can drift apart, define what the "official" rating is at any instant.
5. **Sparse players.** How Challenger/ITF players with a handful of matches are shrunk, and toward what — a tour-level prior, an age curve, or their own tier's mean.
6. **Men's vs women's.** Whether these are two independent fits or one model with tour as a covariate. Serve dominance differs materially between tours, so at minimum the surface and serve parameters cannot be shared naively.
7. **Fitting strategy.** Full MCMC, variational, or a Kalman-style filter with conjugate updates. This is a practical constraint: millions of matches across four tours back to 1968 will not fit under naive Stan MCMC in reasonable time.
8. **What the model must export** to satisfy the ITG Worker integration spec from ticket 06.

Load `stats-discipline` before this session. The output of this project is rankings, which is precisely what that skill exists to guard.

**Deliverable:** a written model spec complete enough that ticket 10's prototype can be built from it without further design decisions.

---

# ANSWER — the engine spec

Written 2026-08-27. `stats-discipline` loaded; §1 (anchors), §5 (one-axis arms), §6 (cheap test first) and §7 (coverage logging) are built into §9–10 below rather than left as advice.

## 1. Observation model

For a point served by player *i* against returner *j*, on surface *s*, at time *t*:

```
logit p_ij,s,t  =  α_i(t) + κ_i,s  −  β_j(t) − λ_j,s  +  δ_tour
```

- `α_i(t)` — serve skill, `β_j(t)` — return skill. Both latent, both evolving.
- `κ_i,s`, `λ_j,s` — per-player surface offsets (§2).
- `δ_tour` — a tour-level intercept, ATP vs WTA.

`p` is the probability the **server** wins the point. Everything else — game, set, match, tournament — is *derived* from `p` through the closed-form recursions in ticket 05, never modelled separately. That single fact is what makes one engine serve pre-match odds, draw simulation and live in-match probability.

**Normalisation (required, per ticket 12).** Only `α_i − β_j` enters any match, so adding a constant to every serve skill and subtracting it from every return skill leaves the likelihood unchanged. **Fix `mean(β) = 0` within each tour.** Without this the sampler wanders a flat ridge and nothing converges. Same anchor Bradley-Terry and Elo require.

**First and second serve split — where the data allows it.** Ticket 11 found the ATP/WTA gap is driven almost entirely by *first*-serve outcomes; second-serve differences were significant only at the Australian Open. So in the point-count regime model three quantities per player (first-serve-in rate, first-serve-won, second-serve-won) from the `1stIn`/`1stWon`/`2ndWon` columns, and collapse to a single blended `p` in the scoreline-only regime. **Do not carry the split into regimes that cannot identify it.**

## 2. Surface

**Decision: hierarchical per-player offsets, not free parameters.**

```
κ_i,s = σ_κ,s · z_i,s        z_i,s ~ Normal(0, 1)
```

`σ_κ,s` is fitted per surface, so the data decides how much surface specialisation exists at all, and individual players are shrunk toward zero. A player with 40 clay matches gets a near-zero offset; Nadal does not.

This is the citius pattern from `citiusverse/citius/R/ability.R` — a shared effect times a per-entity sensitivity, where only the interaction reorders rankings. It is prior art in this ecosystem and far more parsimonious than free per-player-per-surface skills, which would quadruple the parameter count for players who never had the matches to estimate them.

Surfaces: Hard, Clay, Grass, Carpet. **Normalise casing first** — ticket 07a found 31 WTA rows of `'clay'` against 155,165 of `'Clay'`. Missing surface (2,990 ATP / 6,547 WTA rows) is its own level, not an imputation.

## 3. Time dynamics

Gaussian random walk on both skills, in weeks:

```
α_i(t+Δ) = α_i(t) + ε,     ε ~ Normal(0, σ²_α · Δ · φ_i(Δ))
```

`φ_i(Δ)` inflates process variance across a gap. A player absent nine months should return with a *wider* posterior, not the same one they left with — that is the honest statement, and it is what lets the model react quickly to a genuine post-injury decline instead of insisting for six months that the player is still elite.

Recommended form: `φ(Δ) = 1 + c·max(0, Δ − 6 weeks)`, with `c` fitted. Start simple; a mean-reverting (Ornstein-Uhlenbeck) process is the obvious upgrade if random-walk skills drift implausibly for long-retired players. **Size that effect before building it** (§3 of `stats-discipline`) — it may be worth nothing.

## 4. The three data regimes

This is the spine of the whole design. Ticket 12 measured coverage; each regime gets the likelihood it can actually support.

| Regime | Coverage | Likelihood |
|---|---|---|
| **A — point counts** | ATP tour 1991+, WTA tour 2010+, ATP Challenger 2010+ | Binomial on serve points won. Published, low risk. |
| **B — scoreline only** | Everything pre-1991; WTA lower tier; futures | Markov likelihood over the observed set scores (ticket 05 §1a–1e). |
| **C — unusable** | Retirements, walkovers, missing scores (3.15% ATP / 2.50% WTA) | Excluded from the likelihood, **counted and reported**, never silently dropped. |

**Regime B is viable but only in the panel.** Ticket 12 established this by computation: a single 6-4 6-4 puts the MLE at the parameter boundary, but 150 sets recovers truth to within 0.01. So B contributes real information for players with substantial match counts and almost none for a player with twenty matches — which the posterior will say honestly if the priors are right.

Regime membership is a **column on every row**, not a preprocessing filter. Every diagnostic must be sliceable by it.

## 5. Sparse players and cold start

Hierarchical priors do the work:

```
α_i ~ Normal(μ_level(i), σ²_level)
```

where `level` is the player's entry tier (tour / challenger / ITF). A debutant inherits their tier's mean with that tier's variance — so an unknown qualifier starts at a plausible Challenger-level skill with wide bands, not at a global average or an arbitrary 1500.

This is the whole justification for ingesting ATP Challenger (ticket 07a): it means a rising player already has a posterior before their first tour match. **The WTA equivalent is weaker** — its lower tier is 7.2%/23.7% serve-stat coverage, so those players enter through regime B with wider bands. State that limitation in the output; do not paper over it.

## 6. One model or two? — one, with tour as a grouping factor

Per ticket 11's recommendation, and the reasoning is quantitative:

- The tours genuinely differ: 64.0% vs 58.3% service points won, hold rates 81.2% vs 70.0%.
- But `dP_match/dp` differs by only **~5%** between their operating points (5.221 vs 4.981). The sensitivity gap is at the point→game step (26% steeper for WTA), not game→match.

So: **separate `δ_tour`, separate variance components `σ_α`, `σ_β`, `σ_κ` per tour; shared recursion, shared surface structure, shared time dynamics.** One fit, grouped — not two fits, and not naive pooling.

Ratings are **never comparable across tours.** The model has no cross-tour matches to identify a common scale, so any "is Sabalenka better than Alcaraz" number would be an artefact of the normalisation choice. Rank within tour; say so on the page.

## 7. Fitting strategy

Full MCMC over ~1.7M matches and ~58,000 players is not feasible. Staged approach:

1. **Filter pass.** Forward-filter/backward-smooth with a Gaussian approximation to the binomial likelihood (Laplace at each step). Linear in matches, gives the full posterior path per player, and is the *same* machinery as the online fast layer — which is the point.
2. **Validate on a subset with real MCMC.** Fit Stan on one modern ATP season and confirm the filter's posteriors match. If they diverge, the approximation is wrong and everything downstream is suspect. Ingram's released code (`github.com/martiningram/tennis_bayes_point_based`) is the reference implementation.
3. **Regime B needs numerical integration** — the Markov set-score likelihood has no conjugate form. Precompute `P(set score | p1, p2)` on a grid once (the `identifiability.py` machinery already does this) and interpolate.

**Cheap test first (§6 of `stats-discipline`).** Before any of this, fit a plain surface-adjusted Elo and score it. It is an hour's work, it is the benchmark ticket 10 must beat, and if the elaborate model cannot clear it the project needs to know that on day one rather than month three. Remember Ingram's own point-level model scored **68.8% / 0.592** against FiveThirtyEight Elo's **69.5% / 0.586** — the sophisticated model *lost*.

## 8. The two-speed layer and the ITG export

**Slow layer:** the batch filter above. Refit nightly. Source of truth for rankings.

**Fast layer:** a single Kalman update step per point — the same filter, one observation at a time, seeded from the slow layer's posterior. Closed form, no ML runtime.

**Re-seed the fast layer from the slow layer at every match start.** The fast layer never persists across matches; that keeps "what is this player's official rating" unambiguous — it is always the slow layer's.

**Export contract** (ticket 06): a flat JSON blob to R2, per player `{α, β, κ_hard, κ_clay, κ_grass, λ_hard, λ_clay, λ_grass, sd_α, sd_β}` — ~10 floats, under 1 KB per match once filtered to the two competitors. The Worker implements the ticket 05 recursions directly in JavaScript, roughly 50 lines of arithmetic. Copy the failure contract from `inthegame-blog/worker/src/live-wp-calibration.js`: **never throw, fall back to the pre-match probability on any load failure.**

## 9. Anchor checks — write these BEFORE looking at any output

Per `stats-discipline` §1, encoded as assertions in the fitting script, not as intentions. **An anchor failing means the method is wrong — do not special-case it.**

**Current form:**
1. The current world #1 is top-5 in current-form rating on their best surface.
2. A qualifier does not outrank a top-10 seed in the same draw.

**Skill decomposition** (this one tests that α and β are actually measuring what their names claim):
3. Known giant servers (Isner, Karlović, Opelka in the men's data) are top-5% on `α` and below median on `β`.
4. Known elite returners (Djokovic, Nadal) are top-5% on `β`.

**Surface:**
5. Nadal is #1 or #2 on clay among all men, all-time.
6. Federer is top-3 on grass.

**GOAT list:**
7. Djokovic, Federer and Nadal are all top-5 men.
8. Serena Williams, Graf and Navratilova are all top-5 women.

**Uncertainty:**
9. A player with fewer than 20 matches has a posterior SD at least twice that of a player with more than 500.
10. Pre-1991 players have wider bands than modern ones — if they do not, regime B is leaking information it does not have.

Report the anchor results **alongside** every ranking table, so a reader knows they were tested rather than assumed.

## 10. Coverage logging — mandatory, including in scratch scripts

Per `stats-discipline` §7. Every fit prints, before any metric:
- n matches, by tour × level × regime × decade
- date range actually included
- every exclusion with its reason and count (regime C, missing surface, unmatched players)
- n players, and the distribution of matches per player

A silently narrower sample changes the answer without changing how the number reads. This has cost real time in torp and citius; it costs one `print()` here.

## 11. Design traps specific to this build

**Cross-tour id collision.** Ticket 07a: 2,396 ids appear in both tours with different people. `player_uid = "{tour}:{id}"` is namespaced at ingest — but **assert it in the fitting script too**, because a rating engine that silently merges Justine Henin with Jeremy Beale produces perfectly plausible-looking output.

**The Elo comparison must differ on ONE axis** (`stats-discipline` §5). A naive "point model vs Elo" comparison changes the likelihood, the time dynamics, the surface treatment and the data regime simultaneously, and would measure nothing. Fix the training window, the match set and the surface treatment; vary only the likelihood. State the limiting case first: with per-match updates and no shrinkage, the hierarchical model should approach Elo — if it does not, something else differs and that contradiction is the finding.

**Suspicious agreement** (§2). If the ATP and WTA fits return near-identical variance components despite genuinely different tours, suspect a shared bug before celebrating a shared truth.

**Retirement handling is a decision, not a default.** Ticket 11 found the field-wide practice is discard-without-justification. Discarding is not neutral — a retirement is informative about a player's state. For v0: exclude from the likelihood, flag the player-week as elevated process variance, and record the count.
