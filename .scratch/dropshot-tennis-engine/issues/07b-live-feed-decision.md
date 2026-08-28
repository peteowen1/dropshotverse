# 07b — Pick the live feed and set the budget

Type: grilling
Status: resolved
Blocked by: 02, 03, 14

## Question

Split from the original ticket 07 on 2026-08-27. Historical is ticket 07a.

Ticket 03 established that **no free live source is defensible for a public site** — Sofascore and Flashscore both ban the extraction required, ATP and WTA sold exclusive live rights, and every open-source scraper is dead, stale or post-match-only. So the budget question is forced, not optional.

Decide, once ticket 02 reports on what livetennisapi.com actually delivers:

1. **The vendor**, and the monthly ceiling. Known candidates: livetennisapi.com (~$10-30/mo, unverified young vendor), Goalserve ($150/mo, established, 5-second point-by-point, covers Challenger/ITF, 30-day trial), API-Tennis ($40-120/mo but confirmed to return scores *without* serve statistics).
2. **The source seam.** Design the abstraction so the feed is swappable without touching the model — this survived from the original charting decision even though the free half of it did not. Define the interface: what the model needs per point, independent of who supplies it.
3. **Granularity floor.** What is the minimum the live model can work with — full point-by-point, or is game-score-only enough to be worth shipping? This matters because it sets how cheap a vendor can be. Note the ticket-12 hazard: if live matches arrive *without* serve statistics while the model was trained *with* them, training and production paths differ and calibration suffers.
4. **Coverage decision.** Ticket 01 found no source anywhere covers WTA Challenger/ITF live. Decide which events actually get a live model — likely slams and 1000-level both tours, ATP Challenger where available, nothing below.
5. **Failure behaviour.** Copy the contract from `inthegame-blog/worker/src/live-wp-calibration.js`: never throw, fall back cleanly. Decide what the page shows when the feed dies mid-match.
6. **Whether live ships at all in v0**, or waits until the ratings are proven.

**Deliverable:** a named vendor with a monthly figure, the source-seam interface, the covered-events list, and the degradation behaviour.

---

# ANSWER — resolved 2026-08-27

**The premise of this ticket was wrong.** It was written believing ticket 03's finding that no free live source existed, which made a paid feed forced. Ticket 03 was corrected: it had searched rendered pages and scraper repos, never the backend APIs the tournament sites' own JavaScript calls — which is where the equivalent source lives in every other verse Pete runs.

## 1. The decision: $0/month, direct from the tournaments

**No vendor. No subscription.** Live data comes from the tournaments' own public feeds.

| source | granularity | auth | status |
|---|---|---|---|
| **US Open** (IBM) | **true point-by-point** — 57 fields incl. serve speed, rally count, ace/DF/winner/UE, break-point state | **none** (User-Agent only) | **verified live 2026-08-27**, capturing now |
| **Wimbledon** (IBM) | same architecture | none | responds off-season; point granularity architecturally identical, not re-verified live |
| **Australian Open** (Infosys) | match level confirmed; court-vision exists per config | none | open, unverified at point level |
| **Roland Garros** (Infosys) | schedule confirmed; same court-vision pattern | none | open, unverified at point level |
| ATP tour events (Infosys) | match/rally/stroke | `Referer` header + **AES-CBC response**, key derived from `lastModified`; publicly cracked in `glad94/infotennis` | works, but `atptour.com` sits behind Cloudflare |
| WTA live | — | — | **walled.** `liveScoringId` empty; rights are Stats Perform's to 2030 |

This is richer than livetennisapi's **$99.99/month ULTRA** tier, which was the only paid option offering live points.

## 2. Covered events for v0: the four Slams

**Slams only.** Not a compromise — the right scope:

- They are where the data is cleanest, free, and verified.
- They are where an audience actually exists for a live win-probability page.
- Best-of-five men's matches are where an in-match model is most interesting, because comebacks are possible and the probability curve actually moves.
- ATP tour events are reachable but fragile (Cloudflare + AES); add them only once the Slam version is proven.
- **WTA below Slam level is not reachable at all.** Slam-only scope means men's and women's coverage is *symmetric* — which no other scope choice achieves. That is a genuine argument for this scope rather than against it.

## 3. Budget: $0, and livetennisapi is dropped entirely

Its BASIC tier was under consideration for *historical* data; the frozen archive (ticket 13) already covers that, and ticket 14's §3(c) makes accumulating a vendor feed into a training corpus questionable anyway. Its ULTRA tier was the only route to live points; the Slam feeds supersede it.

**Revisit only if** the Slam feeds close, or ATP/WTA tour-level live coverage becomes a requirement. Ticket 15 established Goalserve ($150/mo) as the fallback with the broadest coverage.

## 4. The source seam

The model needs remarkably little. Define the interface at the minimum, so the widest range of sources can satisfy it:

```
PointEvent {
  match_id, seq            # ordering
  server                   # which player served
  point_winner             # who won it
  sets_won[2], games_won[2], points[2]   # score state AFTER the point
  is_tiebreak
}
```

Everything else — serve speed, rally count, shot type — is **enrichment, never required**. The closed-form recursion needs only the score state and who is serving.

Consequence: a feed carrying only *game* scores can still drive the display, updating at game boundaries instead of per point. That is roughly 80% of the value and it means a much weaker source is still usable. **The granularity floor is game-level, not point-level.**

Adapters live in `dropshotdata`; `dropshot` and the Worker see only `PointEvent`.

## 5. The training/production mismatch hazard — resolved

Ticket 12 warned that if live matches arrive without serve statistics while the model was trained with them, calibration suffers. **The US Open feed carries `Ace`, `DoubleFault`, `Speed_KMH` and `RallyCount` per point**, so the Slam production path can match the training path.

But the live model does not need them: it consumes only pre-match skill estimates plus the score state. The mismatch hazard is therefore **structurally absent** in the chosen design, not merely mitigated.

## 6. Failure behaviour

Copy the contract from `inthegame-blog/worker/src/live-wp-calibration.js` verbatim:

- **Never throw.** Any feed failure falls back to the last known good state.
- Degradation ladder: live point probability → last computed probability with a staleness indicator → pre-match probability → hide the component. **Never a blank or a spinner**, which is what a viewer sees precisely when they came to watch.
- Show a timestamp whenever the displayed state is not current.

## 7. Does live ship in v0?

**Yes — but after the rating engine clears its calibration bar (ticket 10 stage 2), not before.** A live win-probability display driven by a badly calibrated model is worse than none: it is confidently wrong in public, in real time.

Sequencing is comfortable: the US Open capture runs to 13 September, so by the time the engine is ready there will be a complete Slam of point data to test the live path against offline, without waiting for another tournament.
