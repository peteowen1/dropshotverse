# 15 — Is there a more institutionally durable tennis data source?

Type: research
Status: resolved

## Answer

Researched 2026-08-27. **Verdict: no institutional tennis data source with a hobbyist tier exists, and the reason is structural.** One genuinely valuable addition was found — Betfair — but it answers a different question.

### A. Official and institutional programmes — nothing available

- **Tennis Australia's Game Insight Group** is real and active (joint with Victoria University, publishes peer-reviewed work on AO Hawk-Eye data), but has **no public data release and no repeatable academic-application channel**. Access runs through personal research collaboration. Not disprovable from outside, but nothing to apply to.
- **ATP** has no public API. Its digital platform is built by **Infosys** under a partnership extended to 2028 — fan-facing only.
- **WTA** data rights are **exclusively Stats Perform's, extended to 2030**.
- **All four Slams** (Wimbledon/US Open with IBM, Roland Garros/AO with Infosys) expose consumer dashboards only.
- **Hawk-Eye** academic use exists but every instance is a one-off grant tied to a specific study and ethics approval — Wimbledon 2016-18 ball-impact data, an ATP "Hawk-Eye Base" set 2005-2009, AO tracking. None carry redistribution rights, none are open programmes.

**Every official channel either exposes nothing or funnels through the Sportradar/Stats Perform exclusives already known.**

### B. Commercial vendors — verified pricing

| Vendor | Self-serve? | Verified price |
|---|---|---|
| Sportradar Tennis | Trial only | Free trial ~1,000 calls; production **$10k+/mo**, sales-gated |
| Stats Perform / Opta | No | Unpublished, contact sales |
| Genius Sports | No | Unpublished, enterprise only |
| Goalserve | Yes | **$150/mo**, $1,200/yr — point-by-point every 5s, all tours |
| SportsDataIO | Yes | Free trial (scrambled) → ~$99-149/mo → custom |
| Enetpulse | Effectively no | Unpublished, est. $500-1,000+/mo |

Goalserve and SportsDataIO are **resellers, not rightsholders** — the same provenance ambiguity as livetennisapi, just older businesses. Everything larger is enterprise-gated with unpublished pricing, which is *worse* accessibility rather than better trust.

### C. The one real find — Betfair

**Betfair Historical Data's Basic tier is genuinely free** to any registered account holder (verified via Betfair's own quant/automation hub). Advanced/Pro tiers add tick-level data with the full order book. The live Exchange API has a free delayed/dev key; a live app key is a **£499 one-off**.

On durability this is in a different class from everything else here: a 20+ year old regulated exchange, publicly owned via Flutter Entertainment. **But it is odds and market data, not match or point data** — a complement, not a substitute. Its real value is as a **benchmark upgrade**: exchange prices carry no bookmaker overround, so they sidestep the de-vigging step entirely. See ticket 04.

### D. Academic and open-source — thin

- UCI's "Tennis Major Tournament Match Statistics" is real but tiny (174 rows, 2013 Slams only) with documented naming/missing-data problems. Teaching toy.
- No large maintained point-by-point dataset found on Zenodo or Harvard Dataverse. PapersWithCode **not directly checked** — flagged unverified.
- **`Tennismylife/TML-Database` bills itself as "live updated" but its last commit was 2026-01-27 — over 7 months stale.** This does *not* contradict ticket 01, which cited the separate `stats.tennismylife.org` `/api/data-files` endpoint rather than the GitHub repo, but the repo itself must not be trusted as fresh.
- **`Aneeshers/tennis-sackmann-archive`** — 7 stars, single maintainer, created June 2026, relicensed CC BY-NC-SA 4.0. **Its right to relicense Sackmann's original data is legally unclear.** A single point of failure, which is exactly why ticket 13 froze our own copy.
- **livetennisapi launched a "free for academic research" tier** with its own Zenodo DOI (10.5281/zenodo.22048731) five days before this research. The GitHub org was created 2026-07-18 — about six weeks old. The Zenodo snapshot carries an independent DOI and CC-BY-NC-4.0 licence so *that snapshot* survives the company folding, and its terms look narrower than BASIC's (no output-ownership language). **But it is the same young vendor, just a new lane — it does not resolve the durability concern.**

### Verdict

**Nothing beats the livetennisapi + frozen-Sackmann-archive combination for this project.** Every bigger name is either enterprise-sales-gated with unpublished pricing, or a different small reseller carrying identical provenance questions. The genuine addition is **Betfair's free historical data as the probability benchmark** — durable, free, and something that will plausibly still exist in three years.

**Still unverified:** GIG's internal access policy (absence of a *public* programme confirmed, absence of any policy not); PapersWithCode; Enetpulse and Genius Sports pricing beyond secondary sources; whether AELTC or ATP hold undisclosed academic MOUs.

## Question

Raised by Pete 2026-08-27, after reading ticket 02: *"not sure i love this one — any public APIs like the Analyst API source we have for panna or Champion Data API we have for torp?"*

The concern is **institutional durability**, not price. His other verses run on established feeds. livetennisapi.com is a young single-operator vendor whose terms claim ownership over "all model outputs" (ticket 14). Betting the foundation of a new verse on it, months after Sackmann's repos vanished, means depending on something that could disappear the same way.

**Find whether anything with better institutional character exists.**

1. **Official and institutional programmes.** Tennis Australia's Game Insight Group is a real analytics research unit — do they publish data or run academic access? Likewise ITF, ATP, WTA, and the Grand Slam bodies with their tech partners (IBM at Wimbledon and the US Open, Infosys at Roland Garros and the Australian Open). Hawk-Eye/Sony research access. Any federation or university open-data release.
2. **Established vendors with a self-serve tier** a hobbyist can actually buy without an enterprise sales call. Verify current pricing pages: Sportradar, Stats Perform, Genius Sports, Goalserve, SportsDataIO, Enetpulse, Opta. **Betfair's historical exchange data is free to account holders and worth checking specifically** — exchange prices have no bookmaker overround, which makes them a *better* benchmark than the bookmaker odds ticket 04 settled on.
3. **Academic and open-science datasets.** Zenodo, Figshare, Harvard Dataverse, OpenML. Tennis is well studied; replication datasets attached to published papers may carry point-level or Hawk-Eye tracking data with durable DOIs — which is exactly the institutional durability being asked for.
4. **Other credible open projects** actively maintained in 2026 that publish data rather than analysis code.

**Judge on durability as much as coverage** — will it still exist in three years? That is the actual question. A slightly worse dataset from an institution that will outlive the project beats a better one from a vendor that might not.

**Deliverable:** a ranked shortlist of at most 4, each with granularity, coverage, access model, real price, licence, and a verified-live-today confirmation. Then an explicit verdict: does anything genuinely beat the livetennisapi + frozen-Sackmann-archive combination, or is that actually the best available?

## Comparison baseline

What any alternative has to beat:
- **Frozen Sackmann archive** (ticket 13) — free, CC BY-NC-SA, ATP+WTA+challenger+futures+qualifying, deep history, slam point-by-point. Weakness: frozen at ~2026-05-25, no live path, depends on one person's snapshot of a deleted repo.
- **livetennisapi BASIC $9.99/mo** — 1968-2022 archive plus 2023-now point tape, all tours including WTA ITF. Weaknesses: young vendor, ~75% point-completeness, "all model outputs" ownership clause.
- **tennis-data.co.uk** (ticket 04) — free odds, ATP 2001+/WTA 2007+, main tour only.
