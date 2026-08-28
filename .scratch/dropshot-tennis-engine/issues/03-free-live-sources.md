# 03 — What free live tennis data exists, and can it legally sit behind a public site?

Type: research
Status: resolved — **BUT ITS CONCLUSION IS WRONG, see correction below**

## CORRECTION, 2026-08-27

**This ticket's headline finding — "no free live tennis source is defensible for a public site" — is false, and the map's live-data decisions were briefly built on it.**

The investigation examined rendered pages, known scraper repos and vendor ToS. It did **not** look for the undocumented backend APIs those sites' own JavaScript calls — which is exactly where the equivalent sources live in every other verse Pete runs (`api.afl.com.au/cfs/` for AFL, TheAnalyst's `/soccerdata/` for football).

Looking there found the opposite answer. **The US Open's IBM feeds are open, unauthenticated, and were verified live today** returning 96 points × 57 fields from an in-progress match, including serve speed, rally count, ace/double-fault/winner/unforced-error flags and break-point state. Wimbledon runs the same architecture. Details in ticket 16; the full endpoint survey is in ticket 17.

**What survives from this ticket:** the analysis of Sofascore and Flashscore (genuinely ToS-prohibited), the confirmation that ATP and WTA sold exclusive live rights, and the assessment that the open-source scrapers are dead or post-match-only. Those findings stand. The error was concluding *absence* from a search that never looked in the right place.

**Lesson worth keeping:** a negative result is only as good as the search behind it. This one was stated with more confidence than its method justified.

---


## Question

The fallback if paid feeds are rejected or fail: **what free or scraped live point-by-point tennis source is actually usable for a public-facing win-probability display, and what is the honest legal and operational risk?**

Investigate:
- **Sofascore** and **Flashscore** — both carry live point-by-point. Neither has an official public API. Map the undocumented JSON/WebSocket endpoints, how stable they've been, and whether they're rate-limited or bot-protected (Cloudflare, TLS fingerprinting).
- **Maintained open-source scrapers.** Check `glad94/infotennis`, `panaC/tennis-dataset`, `n63li/Tennis-API` and anything fresher — verify last commit date and whether they still work, don't just list them.
- **Official sites**: atptour.com, wtatennis.com live scores, and the Infosys MatchBeats slam feeds (which fed Sackmann's now-dead point-by-point repos). Are these scrapeable and at what granularity?
- **Anything genuinely free and sanctioned** — federation feeds, ITF, university/research access programmes.

Then the part that actually matters: **terms of service and legal exposure.** Be blunt. The output here is a *public non-commercial website* showing derived win probabilities, not a private notebook. Distinguish clearly between:
- scraping for private analysis,
- publishing *derived* numbers computed from scraped data,
- republishing the underlying scores themselves.

Also assess **operational fragility**: how often do these endpoints break, and what does the failure look like on a live page mid-match?

**Deliverable:** a ranked shortlist of free live options with granularity, legal posture and fragility for each, plus a clear statement of whether any is defensible for a public site. If the honest answer is "none are", say so plainly — that finding sends the data-foundation decision toward a paid feed.

## Answer

*Researched 2026-08-27. ToS quotes pulled from live pages where fetchable; GitHub maintenance verified via the GitHub REST API (`pushed_at`), not READMEs.*

### Bottom line

**No free live tennis source is defensible for a public, non-commercial site.** Every option that actually has point-by-point live granularity (Sofascore, Flashscore) explicitly prohibits the automated extraction that getting the data requires, and the two tours whose own sites carry live data (ATP, WTA) have sold **exclusive** commercial rights to that exact product — Sportradar for ATP (2024–2029) and Stats Perform for WTA — to a market where the rights holder actively litigates unauthorized redistribution (see Sportradar v LSports below). The open-source scrapers that exist are either dead, stale, or don't scrape live data at all. This is a genuine finding, not a gap in the search: it pushes ticket 07 toward a paid feed, or toward accepting that the free tier of inthegame.blog either has no live win-probability feature or runs on borrowed time against a ToS violation that can be cut off mid-match with no warning.

### Ranked shortlist

| Rank | Option | Live point-by-point? | Maintenance / fragility | Legal posture for a public site |
|---|---|---|---|---|
| 1 (least bad, still not defensible) | **Sofascore undocumented JSON API** (`api.sofascore.com`) | Yes — point-level, low latency, via `/event/{id}/point-by-point` style endpoints reverse-engineered by community wrappers | Endpoints have been stable in shape for years (multiple wrapper packages track it), but sit behind Cloudflare; no official support, breaks without notice | **Prohibited.** ToS: *"You must not use data mining, robots or similar gathering or extraction methods... you must not use any automated means to use the Platform, such as the use of robots or scripts, scraping, crawling, simulation or automated browsing techniques."* Their own help-desk FAQ confirms there is no public API and that data-source agreements prevent them sharing endpoints. |
| 2 | **Flashscore point tracker** | Yes — live point tracker exists for tennis | Heavy Cloudflare + JA3/TLS-fingerprint bot protection (`robots.txt` blocks CCBot, Diffbot, Meta/AI crawlers outright), frequent structural changes | **Prohibited.** ToS (livesport.eu, governs Flashscore properties): *"Without prior authorisation in writing from the Provider, Visitors are not authorised to copy, modify, tamper with, distribute, transmit, display, reproduce, transfer, upload, download or otherwise use or alter any of the content of the Website."* No non-commercial carve-out found anywhere in the document. |
| 3 | **ATP Tour (atptour.com) / Infosys MatchStats widget** | **No** for anything genuinely live at point level for a third party — the live in-match feed is Sportradar's exclusive product; the public site exposes post-match aggregate stats via the Infosys widget | Site was fully revamped end of 2023, breaking every scraper built against the old markup (see `infotennis` history below) | robots.txt is comparatively permissive (`Content-Signal: search=yes, ai-train=no`, generic `Allow: /`), but that governs crawling for indexing, not redistribution. ToS page returns 403 to automated fetch (itself telling). Real constraint isn't the ToS text, it's that **ATP sold exclusive global data/streaming rights to Sportradar through 2029** — republishing point-level ATP data competes directly with a live product the tour is contractually committed to protect. |
| 4 | **WTA Tour (wtatennis.com)** | No live point-by-point exposed publicly; WTA's live data is Stats Perform's exclusive product | N/A | ToS explicitly bans both the harvesting and the redistribution: *"'harvest' (or collect) information from the WTA Sites using an automated software tool or manually on a mass basis"* and separately *"You may not modify, copy, reproduce, republish, upload, post, transmit, translate, sell, create derivative works of, exploit, or distribute... any material from the WTA Sites unless explicitly authorized."* |
| 5 | **`glad94/infotennis` (GitHub)** | **No** — scrapes **post-match** stats only (Key Stats, Rally/Stroke Analysis, Court Vision) from ATP's Infosys widget; explicitly not live/point-by-point | Actively maintained relative to the field: last push **2026-01-06** (`pushed_at` via API), commit history shows ongoing fixes for site-structure breakage (e.g. "Updated elem search for player flags after source html change," "Fix processing_set_stats() to handle new 2025 data columns"). Author states scraping the ATP calendar broke entirely after the site's 2023 revamp and had to be rewritten. Author: *"I have not tested this project beyond my own local machine."* | Scrapes atptour.com against the same ToS/rights-holder issues as #3. Even if it worked live, it doesn't have the granularity this project needs. |
| 6 | **`n63li/Tennis-API` (GitHub)** | Claims ATP/WTA stats scraping | **Stale.** Last push **2023-07-05**, before ATP's late-2023 site revamp that broke `infotennis`'s equivalent scrapers. High likelihood it is currently broken and unmaintained (13+ months with zero commits through a known breaking change). | Same underlying-site issues as #3/#4. |
| 7 | **`panaC/tennis-dataset` (GitHub)** | Claimed live-score + odds + stats scraper | **Dead.** Created 2018, last push **2018-11-10** — almost 8 years stale. Do not use. | N/A — moot, it doesn't run. |
| 8 | **`Tennis-API/tennis-api` (GitHub)** | Looked promising (recent push 2026-05-23) | Turned out to be an **HTML-only landing/docs repo** (2 stars, no description, `language: HTML`) for the commercial tennis-api.com product — not an open-source scraper at all. | N/A, it's marketing copy for a paid API. |
| 9 | **Sanctioned/free official channel (ITF, Grand Slam developer program, university research access)** | **None found.** No ITF public API, no Wimbledon/AUS Open/Roland Garros/US Open developer or researcher program for live or point-by-point data. The one adjacent fact: Wimbledon has publicized *using* Bright Data's web-scraping infrastructure internally for its own analytics — the opposite direction from a public free-access program. | — | No option exists to evaluate. |

### Legal analysis (the part that has to be blunt)

The question asked me to separate three distinct acts. They carry genuinely different risk, and conflating them is how projects talk themselves into publishing something indefensible.

**1. Scraping for private analysis (not published).** Lowest real-world risk. This is a breach of contract under every ToS reviewed (Sofascore's "no data mining/robots," Flashscore's "no copy/reproduce/distribute," WTA's "no harvest"), but enforcement against a single non-monetizing individual scraping for personal research is rare, and *Bright Data v. X Corp* shows courts are skeptical of ToS-alone breach-of-contract claims against scrapers with no accountholder relationship to the site. The operational risk is still real: Cloudflare/TLS-fingerprint blocking can and does kill access without warning, IP-ban, no recourse.

**2. Publishing *derived* numbers (a win probability your own model computed from scraped inputs).** Meaningfully higher risk than #1, not because the *published number* itself is obviously unlawful — a probability computed by an independent model is a new work, and raw sports facts are not copyrightable (*NBA v. Motorola*, 2d Cir. 1997, holding real-time NBA scores taken off broadcasts were not "hot news" misappropriation) — but because:
   - The site's *operation* depends on a continuous, ongoing ToS violation as its input pipeline. That's qualitatively different from a one-off private lookup: it's a public product whose uptime is contingent on Sofascore/Flashscore not noticing or caring, and Cloudflare updates in 2026 have specifically tightened bot classification (see fragility section below).
   - *Morris Communications v. PGA Tour* (M.D. Fla. 2000, aff'd 11th Cir.) establishes that a tour's *compilation* of real-time scores, produced via its own infrastructure, is property the tour can lawfully restrict third-party redistribution of — the court explicitly distinguished this from raw, independently-observed facts. ATP/WTA's live point-by-point feeds are exactly this kind of tour-produced compilation, now under exclusive third-party license.
   - *Sportradar v. LSports* (Israel, Be'er Sheva District Court, ongoing) is the live, current-generation version of this fight: Sportradar is suing a competitor for allegedly copying its proprietary live sports feed, using deliberately seeded errors as proof of copying. This is precisely the shape of claim a rights holder or its exclusive licensee could bring against a scraper feeding a public probability widget, if the site got enough traffic to be worth their attention.
   - **Practical read:** publishing a derived number is more defensible than republishing scores, but it is not "safe" — it is "less likely to draw a cease-and-desist, still built on a foundation that can be cut off or contested at any time," and a non-commercial posture does not appear anywhere as an exception in any ToS reviewed.

**3. Republishing the underlying scores/points themselves.** **Indefensible.** Every ToS reviewed bans this outright (Sofascore's data-mining clause, Flashscore's reproduction/distribution clause, WTA's explicit "may not... republish... any material"), and doing so directly reproduces the specific commercial product — live, compiled, point-level tennis data — that ATP and WTA have sold exclusive rights to through 2029. This is the fact pattern *Morris v. PGA Tour* says tours can lawfully lock down, and the fact pattern Sportradar is currently litigating against a competitor over. Do not do this regardless of commercial/non-commercial framing; "non-commercial" is not a defense that appears in any of these agreements or cases.

### Operational fragility (what failure looks like mid-match)

- **Cloudflare bot management** sits in front of both Sofascore and Flashscore. 2026 brought tighter defaults (Cloudflare's "Precursor" enterprise bot product, continuous behavioral scoring of mouse/keyboard/focus signals) and Flashscore's `robots.txt` already hard-blocks known crawler user-agents (CCBot, Diffbot, Meta bots, Bytespider, etc.) — evidence of an actively defended perimeter, not a passive one.
- **TLS/JA3 fingerprinting**: a plain Python `requests`/`httpx` client has a different handshake fingerprint than a real browser and gets challenged or blocked outright; getting past this requires rotating proxies and browser-matched TLS stacks, which is an ongoing arms race, not a one-time integration.
- **Silent breakage on site redesigns**: ATP's site revamp at the end of the 2023 season broke every scraper built against the old markup (documented directly in `infotennis`'s own commit history and README). There is no deprecation notice — it just stops working.
- **What this looks like on a live page**: the failure mode is not a clean error, it's a stale or blank win-probability tile sitting next to a live scoreboard during the exact moment (break point, tiebreak) a visitor is looking at the page — the worst possible UX for the feature's entire reason for existing. There is no graceful degradation available from these sources; a paid feed's SLA is what buys the ability to fail gracefully instead.

### What this means for ticket 07

Ticket 07 (data foundation decision) should treat "free live source" as **not an available option** for the public-facing win-probability display, full stop — not a cost/quality tradeoff to weigh, but a category that doesn't clear the legal bar regardless of budget. The only free-tier-compatible fallback that survives this analysis is showing no live win probability (or a heavily delayed/non-real-time one built from data obtained through a channel that does grant redistribution rights), while a paid feed is evaluated for the real live feature. Ticket 02's evaluation of `livetennisapi.com`'s $9.99 tier is the more productive next step than any option surfaced here.
