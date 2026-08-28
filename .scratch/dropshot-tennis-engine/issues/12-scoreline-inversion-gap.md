# 12 — How do we actually estimate point-win rates, given nobody has published bare-scoreline inversion?

Type: research
Status: resolved
Blocked by: 13

## Question

Ticket 05 returned an important negative finding: **no published paper inverts bare scorelines through the full Markov likelihood.** Everyone works from point counts or panel data instead (Klaassen & Magnus, Barnett & Clarke).

This matters because charting assumed the opposite. The map's Notes claim the point-level model can be fit on match scorelines for the bulk of history, with true point data as a calibration anchor. That assumption is now **unproven**, and it is load-bearing for the whole project. Settle it.

**The likely reprieve, and the first thing to check.** The Sackmann 49-column schema recovered in ticket 01 carries per-match serve statistics: `svpt`, `1stIn`, `1stWon`, `2ndWon`, `SvGms`, `bpSaved`, `bpFaced`, `ace`, `df` for both players. Those are **point counts**, not bare scorelines — exactly the input Klaassen & Magnus use. If serve stats are present on a large enough share of matches, the inversion problem largely evaporates and we are on well-trodden ground.

So establish, in order:

1. **Coverage of serve statistics** in the `Aneeshers/tennis-sackmann-archive` base layer. What fraction of matches have non-null `svpt` / `1stWon` / `2ndWon`, broken down by tour (ATP/WTA), level (tour/challenger/futures/qualifying), and decade? This is a data question — actually load the files and count, don't estimate. Expect coverage to collapse going back in time and at lower levels.
2. **What to do for matches without serve stats.** Options, in rough order of ambition: drop them; treat the scoreline as a censored observation and use the Markov likelihood directly (the unpublished route); or impute from a player's contemporaneous matches that do have stats. Assess each. If the Markov-likelihood route is genuinely unpublished, note that it is a research contribution, not a known quantity — and price the risk accordingly.
3. **Does the live layer have serve stats at all?** Ticket 01 flagged the freshest cross-tour source has a gutted 13-column schema. If live matches arrive without serve statistics, the production path differs from the training path — which is a calibration hazard worth naming now.
4. **Identifiability.** Given a scoreline alone, are both players' serve-point probabilities even separately identifiable, or only some combination of them? Work this through — it may be the reason nobody has published the method.

**Deliverable:** a verdict on whether the project's core fitting assumption holds, with real coverage numbers rather than guesses, and a recommended estimation strategy per data regime. If the answer is that serve stats cover most of the modern era and the deep past has to be handled differently, say so plainly — that is a perfectly good outcome, but it must be a decision rather than an oversight.

## Answer — item 1 (serve-stat coverage), measured 2026-08-27

Computed directly over the frozen archive (224 files, `dropshotdata/raw/github/`) — real counts, not estimates. Script: `scratchpad/serve_coverage.py`.

**Headline: 307,508 of 1,151,796 matches (26.7%) carry serve counts. But the distribution is what matters, and it is deeply uneven.**

Percentage of matches with non-null `w_svpt`, by decade:

| tour / level | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|
| **ATP tour** | 0.0 | 0.0 | **79.8** | **88.3** | **92.5** | **93.9** |
| **WTA tour** | 0.0 | 0.0 | **0.0** | 30.1 | **85.6** | **95.2** |
| **ATP qual/chall** | 0.0 | 0.0 | 0.0 | 0.0 | **95.6** | **99.5** |
| **ATP futures** | — | — | 0.0 | 0.0 | 0.0 | 26.0 |
| ATP doubles | — | — | — | 0.0 | 0.0 | 0.0 |

Where `w_svpt` is present, **all ten serve fields are present** (`with_all` matches `with_svpt` to within 6 rows across 1.15M). The data is clean — there is no partial-field problem to engineer around.

### Verdict: the assumption holds where it matters least, and fails where it matters most

1. **Modern era, main tour — the point-count route works.** ATP tour is 87.5% covered since 1990 and WTA 47.7% (85.6% in the 2010s, 95.2% in the 2020s). This is exactly the Klaassen & Magnus input, on well-trodden ground. **No unpublished inversion needed for current-form ratings.**
2. **Pre-1991 has literally zero serve statistics on any tour.** Not sparse — absent. The era was not recorded. So **the all-time GOAT list, which spans 1968 onward, is precisely the product that needs scoreline-only inference** — the route ticket 05 established nobody has published. The risk landed on the second ranking product, not the first.
3. **WTA's usable point-count history starts ~2010, ATP's ~1991.** A 19-year asymmetry. Any cross-tour or cross-era comparison must handle this explicitly or it will silently compare a well-estimated ATP rating against a poorly-estimated WTA one.
4. **Futures/ITF is effectively bare** — 4.6% overall, only 26% even in the 2020s. ATP Challenger, by contrast, is excellent from 2010 (95.6%, then 99.5%). **This weakens the case for ingesting futures specifically**, while strongly supporting Challenger.

### Recommended estimation strategy per regime

- **1991+ ATP tour, 2010+ WTA tour, 2010+ ATP Challenger:** fit on point counts. Standard, published, low risk.
- **2000s WTA and 2020s futures (partial coverage):** fit on point counts where present; treat absence as missing-at-random only after checking it is not systematic by tournament tier.
- **Pre-1991 everything, and futures generally:** scoreline-only. Either accept an Elo-style match-outcome rating for this regime, or treat the Markov-likelihood inversion as a genuine research contribution and price the risk. **Do not assume it will work.**

## Answer — item 4 (identifiability), computed 2026-08-27

**The question:** ticket 05 found nobody has published scoreline inversion through the Markov likelihood. Before treating that as an opportunity, establish whether it is instead a **dead end** — a ridged likelihood that cannot identify `(p1, p2)` in principle.

**Method:** implemented `P(final set score | p1, p2)` exactly — the closed-form game probability from ticket 05 §1a, then a DP over game states honouring serve alternation, with 7-6 lumped since a bare scoreline reports it that way regardless of tiebreak margin. Then examined the log-likelihood surface over a 0.01 grid. Script: `scratchpad/identifiability.py`. Truth used: `p1=0.64, p2=0.58` (the ATP/WTA operating points from ticket 11).

**Result: it is a sample-size problem, not an impossibility.**

| observed | MLE found | truth | 95%-ish width on p1 | plausible grid cells |
|---|---|---|---|---|
| one set, 6-4 | (0.42, 0.35) | (0.64, 0.58) | **0.50** | 1,421 / 2,601 |
| one match, 6-4 6-4 | (0.42, 0.35) | (0.64, 0.58) | 0.32 | 511 / 2,601 |
| 10 sets | (0.63, 0.59) | (0.64, 0.58) | 0.23 | 288 / 2,601 |
| 40 sets | (0.58, 0.53) | (0.64, 0.58) | 0.14 | 100 / 2,601 |
| **150 sets** | **(0.65, 0.58)** | (0.64, 0.58) | **0.06** | 22 / 2,601 |

**Per match, inversion is hopeless.** A single 6-4 6-4 puts the MLE at the grid boundary — far from truth — because a wide range of `(p1, p2)` produce that scoreline about equally well. Anyone attempting this one match at a time would conclude it does not work, **which may be exactly why it is unpublished.**

**Across a panel it works, and works well.** By ~10 sets the MLE is close; by 150 sets it recovers truth to within 0.01 with a ±0.03 interval. Identification comes from **repetition**, not from any single scoreline.

**The near-ridge is real but not fatal.** Shifting `p1` up and `p2` down by 0.02 slightly *improved* the likelihood (+0.08) — so a local ridge exists — but ±0.10 was clearly worse (−1.32). Serve alternation makes the set score depend asymmetrically on the two players, which is what breaks the ridge. **A global normalisation is still required** (only `serve_i − return_j` enters any single match, so adding a constant to every serve skill and subtracting it from every return skill is unidentified across the whole panel). Fix mean return skill at zero, exactly as Bradley-Terry and Elo need an anchor.

**Caveats, stated plainly.** Data was simulated from the same model used to fit, so this measures identifiability, **not** robustness to misspecification. The repetition was against one fixed opponent; real panels have a *network* of shared opponents, which should identify better, not worse. The grid is coarse at 0.01.

### Verdict for the model spec

- **Modern era (ATP 1991+, WTA 2010+, ATP Challenger 2010+):** fit on point counts. Published, low risk.
- **Pre-1991 and other scoreline-only regimes:** scoreline inversion is **viable in the panel** — but only for players with substantial match counts. A player with 500 career matches will be well identified; one with 20 will not. This is precisely the situation a Bayesian model handles honestly, by returning a wide posterior rather than a confident wrong number.
- **The GOAT list is therefore feasible**, with the honest caveat that pre-1991 players get materially wider uncertainty bands than modern ones — and those bands must be *shown*, not hidden behind a point estimate.

### Still open

Items 2 (imputation strategy for partially-covered regimes) and 3 (whether a live feed carries serve stats — ticket 16 will answer this from the US Open capture).
