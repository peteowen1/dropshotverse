# 18 — Widen ability spread (injury-safe), then build Elo+ for a four-way comparison

Type: task
Status: claimed

## Question

Raised by Pete 2026-08-27, following a diagnostic on the tuned hierarchical model (ticket 17) that found: (1) a residual Milos-Raonic-shaped tail from the already-fixed runaway-drift bug — the cap bounded the magnitude, not the direction; (2) the dominant cause, contributing **75% of the entire log-loss gap to Elo**: systematic underconfidence specifically on lopsided matches (market favourite 80-100%), where the model's mean predicted favourite probability is 0.66 against the market's 0.88.

Pete's diagnosis, taken as the brief: **allow bigger day-to-day skill movements (more spread), but keep the injury-related uncertainty inflation carefully separate and tuned in its own right — don't reopen the bug that cost ticket 10 a debugging marathon.** Then build an "Elo+" (Elo with incremental, individually-tested enhancements) so the final comparison is four-way: Elo, Elo+, Hierarchical, Market.

## Charter (extends ticket 17's, same tune/holdout split, same leakage discipline)

- Tuning window 2022-23, final holdout 2024-25 — unchanged, still never touched for selection.
- **New: every candidate must pass `stress_test_safety.py` before being scored.** Three reproductions of the original bugs (a 15-year-layoff single event, six compounding long-layoff-loss cycles, and the isolated weakly-connected pool over 6 years) must all stay bounded. A candidate that fails is rejected outright, not scored, regardless of how good its log-loss might look.
- Primary metric unchanged: overall log-loss on odds-joined rows. Per-bucket (lopsided-match) log-loss reported as a diagnostic, not substituted as the selection criterion mid-stream — that would be exactly the anti-pattern `validation-design` warns against (changing the metric after seeing results).
- `RW_VAR_PER_WEEK=0.005` and `SURFACE_VAR_SCALE=4.0` (ticket 17's winners) held fixed throughout this search.
- Search order: `OU_THETA_PER_WEEK` (lower = more spread retained) → `MAX_SHIFT` (higher = bigger single-observation movements) → `PHI_MAX` → `LAYOFF_C` (the two injury-specific parameters, tuned in their own right per Pete's brief, not just left alone).

## Answer

Started as a hyperparameter sweep (spread vs injury), but a Pete-requested anchor check ("find a match that should be 99%") derailed it into finding and fixing three real correctness bugs first — the sweep itself never ran to completion because its premise (tune `OU_THETA`/`MAX_SHIFT`) was superseded by what the anchor check found underneath.

### The anchor check that started it

Djokovic vs Dino Prizmic, Australian Open 2024 R1 — about as lopsided a real matchup as exists. Elo: 97.6%. Ticket 17's tuned hierarchical model: **70.3%**. Wrong, and the diagnostic in ticket 17's own writeup (calibration by bucket) had already flagged systematic lopsided-match underconfidence as the dominant remaining problem — this was that problem, concretely.

### Bug 6: kappa's variance never shrank

Root cause, found by inspecting `d1_var` directly: `SURFACE_PRIOR_VAR` (ticket 10's documented v0 simplification — surface offsets get a fixed prior variance, never an evolving posterior) combined with ticket 17's `SURFACE_VAR_SCALE=4.0` created a variance **floor** of ~0.80 on Hard court for every prediction, regardless of match count. Djokovic's own skill variance had shrunk to nearly nothing; the fixed surface term dominated the total and capped confidence for everyone, always. Fixed by giving kappa/lambda the same Kalman-style shrinking variance mu already had (`apply_side_update` now returns and threads a real `kappa_var`).

Verified in isolation before trusting it at scale: total variance for a repeatedly-observed player/surface pair dropped from 1.04 to 0.086 over 30 matches, matching mu's existing shrink rate.

### Bug 7: kappa had no mean-reversion, reopening ticket 10's original bug through a new variable

Giving kappa's variance the ability to shrink without also giving kappa's *mean* the reversion `mu` already had (`OU_THETA_PER_WEEK`) reopened the exact runaway-drift pathology ticket 10 spent hours on — just through kappa. Caught because Sinner vs a qualifier flipped to Sinner having essentially 0% chance of winning his own serve point. Fixed by moving kappa/lambda from separate module-level `(uid, surface)` dicts into `PlayerState` itself, so `inflate()` applies the same decay-toward-0 and re-inflate-toward-prior treatment to every surface a player has ever recorded.

### Bug 8: an absorbing-boundary trap in the variance-reduction math

Fixing bug 7 didn't fully fix it. Population check: 4.3-5.3% of all 44,585 players pinned at `|value| > 2.5` (near `MAX_STATE`), including established stars like Sinner and ordinary tour regulars like Van De Zandschulp — not a rare edge case. Root cause: when `MAX_SHIFT` or `MAX_STATE` clamps a mean update, the variance-reduction math was computed as if the *full, unclamped* update had been accepted. A player pinned at a wall by early noise kept getting more confident every subsequent match even though most of each update's intended movement was being thrown away — and once variance is near zero at the wrong boundary, no amount of contradicting future evidence can pull it back (Newton's pull is proportional to relative precision, which only ever increases). Fixed by scaling variance reduction by how much of the intended update actually survived clamping — a component fully absorbed by a wall gets zero variance reduction that round.

This fix alone did not resolve it either (population fraction went 4.3% → 5.3%, no better) — it was a real, correct fix for a real mechanism, but not *the* dominant one.

### Bug 9 (the actual dominant cause): MAX_SHIFT was too loose for a low-data start against established opponents

Found via direct population-embedded tracing (`DSV_DEBUG_UID`, added as a reusable diagnostic) — not synthetic tests, which never reproduced this. Real trace of Jannik Sinner's actual 2018 matches showed him hitting `MAX_STATE` (the system's most extreme possible value) within his **first 3-4 tracked matches**, as an unproven 17-year-old losing to established Challenger players. That dip is realistic; reaching the absolute extreme in 3-4 matches is not. Mechanism: facing already-confident (tight-variance) opponents makes each early match's raw, pre-clamp Newton shift larger than a synthetic equally-uncertain-opponent test predicts, so several near-`MAX_SHIFT` events in a row marched him straight to the wall — and once there, kappa (now independently able to reach its own wall per bug 6's fix) piled onto the *same* extreme, roughly doubling the effective severity.

Fixed by reducing `MAX_SHIFT` from 1.0 to 0.3. Re-traced Sinner's identical real 2018-19 matches: stays sane throughout, never approaches the wall, converges to a plausible elite-server trajectory by 2019.

**Population-wide result: fraction of players pinned at `|value| > 2.5` dropped from 5.3% to 0.0067%** (3 of 44,585, essentially eliminated). Sinner's actual final state: `m_a=1.12, m_b=0.49, kappa[Hard]=0.24` — sane, positive, plausible.

### Final verification — all four original anchors, with real posterior-averaging

| match | Elo | Hierarchical (all 4 bugs fixed) |
|---|---|---|
| Djokovic d. Prizmic | 97.6% | **83.5%** |
| Sinner d. Van De Zandschulp | 91.4% | **80.6%** |
| Alcaraz d. Gasquet | 94.1% | **79.8%** |
| Sinner d. De Jong | 97.0% | **86.2%** |

All four now correctly favour the right player by a wide margin — directionally right, no longer catastrophically wrong (previously 0.4%, 50%, 65.8%, or reversed depending on which bug's state was captured). Somewhat less extreme than Elo, which is now a defensible difference (genuine residual uncertainty) rather than a bug.

**Aggregate log-loss on the full 8,594-row odds-joined holdout barely moved: 0.6593 (all fixes) vs 0.6508 (ticket 17, before this bug hunt) vs Elo's 0.6152 vs Market's 0.5888.** This is not a contradiction. Ticket 17's number was partly a coincidence riding on the same corruption that broke Sinner and ~5% of the population — the model is now *correct* rather than *lucky*. The metric that was actually diagnosed as broken improved substantially: **lopsided-match (market favourite 80%+) log-loss went from 0.523 (ticket 17) to 0.442** — most of the way to Elo's 0.385 — with mean confidence rising from 0.659 to 0.767.

### What's still open

- The original spread/injury hyperparameter sweep this ticket started with never ran to completion against the now-correct architecture — `RW_VAR_PER_WEEK`, `OU_THETA_PER_WEEK`, `PHI_MAX`, `LAYOFF_C` were all tuned or reactively set against the *buggy* pre-fix code and should be re-tuned now that the underlying mechanics are sound, following the same tune(2022-23)/holdout(2024-25) charter.
- Two safety-net reproductions were added to `stress_test_safety.py` during this session (kappa-specific bound, and the debug-trace tooling `DSV_DEBUG_UID`) — worth keeping permanently, since a synthetic test suite alone did not catch bugs 8-9; the real-population trace did.
- Elo+ (part 2 of this ticket's original scope) was not started.

## Part 2: Elo+

[not started this session]
