# 19 — Re-optimize hyperparameters targeting lopsided-match confidence specifically

Type: task
Status: claimed

## Question

Ticket 18's optimization (`tune_v2.py`) selected on **overall** log-loss, which is dominated by the ~80% of matches that are close-to-moderate — it improved overall log-loss (0.659→0.644) but left the lopsided bucket (market favourite 80%+, ~20% of matches) essentially unmoved (0.442→0.443), the exact thing Pete asked about. Re-run the search with a metric that actually weights the lopsided bucket, per his explicit request.

## Charter (extends ticket 17/18's — same tune/holdout split, same corruption gate)

- **Primary metric, locked before any candidate runs:** a single weighted log-loss over all rows, not two separate scores summed ad hoc. Each row gets weight 3.0 if it falls in the lopsided bucket (market-implied favourite probability > 0.80) and weight 1.0 otherwise. `score = sum(weight_i * -log(p_i)) / sum(weight_i)`. This keeps every row in one coherent metric — rows outside the lopsided bucket still count, just less — rather than discarding the rest of the calibration surface.
- **Why 3x, not something else:** the lopsided bucket is ~19.7% of rows; a 3x weight raises its share of the total score to ~42%, a meaningful shift without letting it dominate entirely. This is a judgement call, stated plainly rather than hidden — a genuine sensitivity analysis over the weight itself would be a legitimate follow-up, not done here given session time.
- Tuning window 2022-23, final holdout 2024-25 — unchanged.
- Corruption gate unchanged: reject any candidate above 0.02% of players pinned at `|value| > 2.5` (ticket 18's near-zero baseline × 3, after confirming the residual flagged cases at that baseline are genuine all-time greats — Margaret Court, Chris Evert, Steffi Graf, Monica Seles — not bugs).
- **Warm-started** from ticket 18's overall-log-loss optimum (`RW_VAR_PER_WEEK=0.008, SURFACE_VAR_SCALE=2.0, OU_THETA_PER_WEEK=0.0010, MAX_SHIFT=0.2`) rather than the original ticket-10 defaults, since that is a reasonable starting point and the search explores from there under the new objective.
- **Explicit expectation, stated before running:** this search should plausibly select a *larger* `MAX_SHIFT` than ticket 18's 0.2, trading some close-match calibration for lopsided-match confidence — that trade is the whole point of changing the metric. If the search reselects the *same* config, that is itself informative (it would mean 0.2 was already a genuine joint optimum, not an artefact of the old metric) and should be reported as such, not treated as a failed run.
- **Final reveal reports BOTH** the new weighted metric and the plain overall log-loss on 2024-25, so a reader can see exactly what was traded for what.

## Answer

Search ran to completion (`tune_v3_lopsided.py`). Selected config: `RW_VAR_PER_WEEK=0.005, SURFACE_VAR_SCALE=1.0, OU_THETA_PER_WEEK=0.0005, MAX_SHIFT=0.2, PHI_MAX=4.0, LAYOFF_C=0.06`. Tuning-window (2022-23) weighted log-loss 0.6045→0.5974.

**MAX_SHIFT did not move from ticket 18's 0.2** — the charter's stated expectation was that a larger MAX_SHIFT might win under this metric; instead 0.3 scored worse (weighted=0.617) and 0.4/0.5/0.6 were all rejected outright by the corruption gate. Per the charter, this is itself the finding: 0.2 is a genuine joint optimum against both metrics, not an artefact of the old one. What actually moved the needle was slower mean-reversion (`OU_THETA_PER_WEEK` 0.0010→0.0005) and less surface-variance inflation (`SURFACE_VAR_SCALE` 2.0→1.0) — both act to make the model trust an established rating more once it has one, which is a more targeted lever for lopsided-match confidence than "allow bigger single-match jumps."

### Final holdout reveal (2024-25, n=8,594 odds-joined rows)

| | Elo | Hierarchical v19 (ticket 18) | **Hierarchical v20 (this ticket)** | Market |
|---|---|---|---|---|
| Overall acc | 0.6534 | 0.6271 | 0.6310 | 0.6792 |
| Overall log-loss | 0.6152 | 0.6443 | **0.6467** | 0.5888 |
| Lopsided (mkt_fav>0.80) log-loss | 0.3853 | 0.4427 | **0.4244** | 0.3645 |
| Lopsided mean confidence | 0.839 | 0.750 | **0.779** | 0.875 |

The trade the charter anticipated happened exactly as described: overall log-loss got very slightly worse (0.6443→0.6467, third decimal place) while lopsided-bucket log-loss improved meaningfully (0.4427→0.4244) and mean confidence on the market's biggest favourites rose from 0.750 to 0.779 — about a third of the remaining gap to Elo's 0.839 closed. This directly answers Pete's original ask ("still needs more confidence in these top end players") — real, measured progress, achieved by pointing the optimizer at the actual metric he cares about rather than overall log-loss.

**Still true after this fix:** the hierarchical model remains behind both Elo and the market on every headline number, overall and lopsided. This ticket closes the "we were optimizing the wrong thing" gap; it does not close the "still underperforms Elo" gap from ticket 10 — that remains open, and per the architecture-gap discussion (tournament tier, margin-of-victory / full-scoreline inversion, round weighting) is more likely to need new features than further hyperparameter search.

`hier_v20_eval_predictions.parquet` / `hier_v20_final_state.pkl` are the current canonical model outputs, superseding v19's.
