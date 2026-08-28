# 14 — Can we publish our own ratings derived from a vendor's feed?

Type: task
Status: resolved

## ANSWER — read verbatim 2026-08-27

Fetched `https://livetennisapi.com/terms` (HTTP 200, 17,961 bytes) and extracted the full text rather than a summary. Provider is **JSB Holdings LLC**, a Delaware LLC. Twelve numbered sections — **except there is no §5; the document jumps from 4 to 6.** A drafting error, not a hidden clause, but worth noting in a document being relied on.

### The "model outputs" fear was unfounded

§4 verbatim:

> **4. Intellectual Property** — The API, its data compilation, the arbitrated live feed, and all model outputs, win-probabilities, and analysis are the property of JSB Holdings and are protected by applicable intellectual-property laws. These Terms grant you a license to use the responses in your application; they do not transfer ownership of the underlying data, models, or compilation.

**The same phrase is defined by usage in §2:**

> Model outputs, live model fields, market prices, and the WebSocket feed are available only on the tiers where they are listed.

There, "model outputs" is unambiguously **their own tier-gated product feature** — the ULTRA AI win-probability. §4 lists it beside "the API, its data compilation, the arbitrated live feed", all things *they* produce. Read in context it asserts ownership of **their** models' outputs, not of anything a licensee computes.

§4's second sentence confirms the reading: it distinguishes a licence to use responses from ownership of "the underlying data, models, or compilation" — the ordinary licence-not-assignment boilerplate. It says nothing about derived works the licensee creates.

And §1 grants affirmatively: *"use its responses within your own applications and services."*

**Verdict: publishing our own independently-computed win probabilities is not caught by §4.** The earlier alarm came from reading one phrase without the section that defines it.

### The clause that does bite is §3(c)

> **3. Acceptable Use** — You will not: (a) resell, redistribute, sublicense, or republish the raw API responses as a competing data feed or bulk dataset; (b) use the API to build a product whose primary purpose is to replicate or substitute for the API itself; **(c) scrape, cache, or store the data beyond what is reasonably necessary to operate your application;** (d) … unlawful purpose …; or (e) misrepresent the source of the data or imply endorsement.

**§3(c) constrains accumulation, which is exactly what a training corpus is.**

- Using live scores to drive a live display: squarely within §1. Fine.
- Accumulating their feed into a permanent historical training set: **arguably beyond "reasonably necessary to operate your application".**

That distinction lands precisely on the source-separation architecture already chosen — fit on CC-licensed Sackmann data, use any vendor feed only for live scores driving a computed display that is never republished as data. **That decision was right, and now for an evidenced reason rather than an instinct.**

§3(a) is not a problem (we would never republish raw responses). §3(e) means **attribution is required** wherever their data is visibly used.

### Three clauses that reinforce the durability concern

- **§9 Suspension and Termination** — *"We may suspend or terminate your access, with or without notice, for any violation of these Terms, for abuse or excessive load, or as required to protect the service."* No notice required.
- **§7 Availability** — *"a best-effort, single-region service"*, coverage *"may be reduced, delayed, or interrupted at any time"*, endpoints and fields may be removed.
- **§10 Changes** — they may update the Terms unilaterally; continued use is acceptance. So today's favourable §4 reading is not durable.
- **§8** caps total liability at fees paid in the preceding three months — **$30 on BASIC.**

Governing law is Delaware (§11). Contact for questions: `hello@livetennisapi.com` (§12).

### Decision

1. **Unblock 07b.** The ownership clause does not prevent publishing our own ratings. This is no longer a blocker.
2. **Keep source separation as architecture, not caution.** §3(c) makes it the correct reading of the licence, not merely a prudent hedge: train on the CC-licensed archive, confine any vendor feed to live scores.
3. **If a vendor feed is ever used for training, ask first.** §12 gives an address. A one-line email reply converts §3(c)'s ambiguity into a record, and costs nothing.
4. **Attribute wherever vendor data is visibly used** (§3(e)).
5. **Do not treat any vendor as a foundation.** §7, §9 and §10 together mean best-effort service, no-notice termination, and unilaterally revisable terms. Confirms ticket 15's durability finding from a second direction.

**Note:** the US Open feeds (ticket 16) are a different question entirely — those are the tournament's own public feeds, governed by usopen.org's terms, not these. This ticket says nothing about them.

## Question

**This gates the whole project, not just the live feature.** Raised 2026-08-27 by ticket 02.

livetennisapi.com's terms assert ownership, to JSB Holdings, over "the API, its data compilation, the arbitrated live feed, and **all model outputs**."

That phrase is ambiguous in a way that matters enormously here. If "model outputs" means only *their* AI win-probability product, we are fine. If it stretches to anything modelled from their data, then our ratings, our GOAT list and our live win probability — the entire published output of dropshotverse — are contested, because all of them would be models fit on their feed.

The terms separately prohibit reselling or republishing "the raw API responses as a competing data feed or bulk dataset," and forbid building a product that replicates the API's core function. Neither obviously covers publishing an independently-computed probability. Neither obviously excludes it either. There is no non-commercial carve-out anywhere.

**The finding above came from a summarised fetch, not a verbatim read.** For something this load-bearing that is not good enough.

Do this:
1. **Read the actual terms document at `livetennisapi.com/terms` in full, verbatim.** Not a summary, not a fetch digest — the real text. Quote the relevant clauses exactly.
2. **Establish whether "model outputs" is defined anywhere** in the document, or left undefined.
3. **Check whether signup adds a click-through agreement** with terms beyond the public page.
4. **Recommend whether to seek written confirmation from the vendor** that publishing our own probability model, derived from their feed, on a public non-commercial site is permitted. An email reply saying yes is cheap and converts an ambiguity into a record.

**The architectural escape hatch, worth costing even if the terms turn out fine.** The historical data has an alternative provenance: the Sackmann-derived archive being frozen in ticket 13 is CC BY-NC-SA — attribution required, non-commercial, no ownership claim over derived models. So the published ratings and GOAT list could be fit *entirely* on that CC-licensed base, with the vendor feed used only for live scores driving an in-match display that is computed but never republished as a dataset. That separation dodges the ambiguity cleanly. Assess whether it costs anything in model quality.

**Deliverable:** the verbatim clauses, a plain verdict on whether we can publish derived ratings, and a recommendation on the source-separation strategy.
