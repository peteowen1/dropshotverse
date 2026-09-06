"""Ticket 10 stage 2: the hierarchical point-level serve/return model from ticket 08.

Architecture (spec section numbers refer to issues/08-rating-engine-spec.md):

  logit p_ij,s = alpha_i + kappa_i,s - beta_j - lambda_j,s      (S1)

Everything else (game/set/match/tournament probability) is DERIVED from p via
the closed-form recursion (ticket 05), never modelled directly. That is the
whole point of this architecture over Elo, and it is what gets scored below.

Fitting is an online Bayesian filter (Laplace / assumed-density-filtering,
Glicko/TrueSkill lineage) rather than full MCMC - the "filter pass" the spec
calls for as the tractable route over ~1.9M matches (S7).

DOCUMENTED SIMPLIFICATIONS (declared in the ticket 10 pre-registration BEFORE
this was run, so they cannot be introduced after the fact to explain a result):
  - kappa/lambda (surface offsets) use a FIXED shrinkage prior variance, not a
    fully time-evolving posterior. A ridge-style per-entity effect, not full
    hierarchical Bayes. Chosen for tractability in a v0 prototype.
  - Regime B (scoreline-only matches) updates only a COMBINED strength term
    (alpha+beta), split 50/50, never separate serve/return skills - because
    ticket 12 proved a single match cannot identify them separately. This is a
    direct consequence of that finding, not an unrelated shortcut.
  - Process-variance and shrinkage hyperparameters are plausibility-set, not
    cross-validated.

stats-discipline applied throughout:
  section 1 - anchors, asserted before the headline numbers are trusted
  section 5 - the Elo comparison varies ONLY the fitting method; same data,
              same eval window, same odds join, scored in one pass on
              identical rows (5.3)
  section 7 - coverage logged before any metric
"""
from __future__ import annotations

import math
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _common
ARCHIVE = _common.RAW / "github"
ELO_PARQUET = _common.DD / "elo_eval_predictions.parquet"
# Ticket 17: eval years are now env-overridable so the SAME script produces
# predictions for the tuning window (2022-2023) or the untouched final holdout
# (2024-2025) without editing the file - the discipline that keeps the two
# separate is procedural (which window's numbers get looked at), not code.
EVAL_YEARS = tuple(int(y) for y in os.environ.get("DSV_EVAL_YEARS", "2024,2025").split(","))
DEBUG_UID = os.environ.get("DSV_DEBUG_UID", "")   # e.g. "atp:206173" - traces one
                                                   # player's real state through
                                                   # every regime-A match, in true
                                                   # population context (not an
                                                   # isolated replay, which never
                                                   # reproduces the failure this
                                                   # ticket is chasing)
SURFACES = ("Hard", "Clay", "Grass", "Carpet", "Unknown")

# Bug fix (verified 2026-08-28, not tied to any ticket number): the archive's
# tourney_date is a TOURNAMENT-level date (its start date), not a per-match
# date - 26% of player-appearances share a date with another match by the
# same player (up to 23 matches on one date). Sorting by date alone (with a
# stable mergesort) means the order WITHIN a date is whatever order
# load_matches() happened to concatenate rows in - and load_matches() reads
# ALL main-draw yearly files for a tour first, THEN ALL qual_chall yearly
# files, THEN ALL qual_itf yearly files (see the `pats` loop), so for every
# single tournament, every year, both tours, qualifying rounds land AFTER the
# entire main draw in file order - even though qualifying happens BEFORE the
# main draw starts in real life. A player who just won Q1-Q3 was, until this
# fix, evaluated in their R128 match using a state that hadn't yet
# incorporated those real qualifying wins. Confirmed systemic (not a one-off)
# via the load_matches() code path itself, not just a spot check of one
# tournament.
#
# Q1 < Q2 < Q3 < Q4 < Q5 < ER < R128 < R64 < R32 < R16 < QF < RR < SF < F/BR
# RR (round-robin, season-ending Finals events) is a group stage, ranked
# before the SF/F knockout bracket it feeds. BR (bronze/3rd-place playoff) is
# ranked with F - both happen after SF and are effectively concurrent.
ROUND_RANK: dict[str, int] = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4,
    "ER": 5,
    "R128": 6, "R64": 7, "R32": 8, "R16": 9,
    "QF": 10, "RR": 11, "SF": 12,
    "F": 13, "BR": 13,
}


def round_rank(series: pd.Series) -> pd.Series:
    """Map a `round` column to a within-date sort rank. `'f'` is normalized to
    `'F'` first (a single-row lowercase/uppercase inconsistency confirmed to
    be the same round, not a distinct one - a WTA ITF-qualifying final).
    Any round value not in ROUND_RANK falls back to a neutral rank (NaN,
    sorted last within its date via na_position='last') rather than crashing
    or guessing a position - this only affects rows whose round label isn't
    in the vocabulary this fix was built against, and is logged so a new
    label doesn't silently misorder matches."""
    normalized = series.astype(str).str.strip()
    normalized = normalized.where(normalized != "f", "F")
    ranks = normalized.map(ROUND_RANK)
    unknown = ranks.isna()
    if unknown.any():
        bad_vals = sorted(normalized[unknown].unique())
        warnings.warn(
            f"prepare(): {int(unknown.sum())} row(s) have round value(s) "
            f"{bad_vals} not in ROUND_RANK - falling back to date-only order "
            "for these rows (sorted last within their date).",
            stacklevel=2,
        )
    return ranks

# ---- hyperparameters. Originally plausibility-set (see docstring); ticket 17
# cross-validates the three flagged in ticket 10's verdict via env override, so
# a sweep can subprocess this script per candidate with no shared state.
INIT_VAR = float(os.environ.get("DSV_INIT_VAR", "1.2"))   # initial belief variance for a brand-new player's mu
# Ticket 20 ablation only: force the Kalman-gain split so ~all of every update
# lands on mu (surface offset gets none), to test whether the model can
# degenerate to single-scalar-Elo-like behaviour. Unset in normal runs.
_FORCE_SHARE_MU = os.environ.get("DSV_FORCE_SHARE_MU")
FORCE_SHARE_MU = float(_FORCE_SHARE_MU) if _FORCE_SHARE_MU is not None else None
# Ticket 20 hypothesis 2: regime B (scoreline-only, 72.5% of matches) treats a
# whole match win as a single n=1,y=1 Bernoulli trial in adf_update - a real
# match win is stronger evidence of a skill gap than one coin flip. This scales
# both y and n together (a "fractional/tempered likelihood" - see Grunwald &
# van Ommen), which raises the Hessian's curvature and produces a bigger,
# faster-shrinking update without changing the implied win probability sigmoid(d).
REGIME_B_N = float(os.environ.get("DSV_REGIME_B_N", "1.0"))
# Ticket 24 (scoped-down heavy-tail proxy, see apply_side_update's docstring):
# reinflate variance in proportion to how much of an update's intended shift
# was clipped away by MAX_SHIFT, instead of merely protecting variance from
# shrinking (bug 8's fix). Off (0.0) until scanned through the corruption gate.
CLIP_REINFLATE_K = float(os.environ.get("DSV_CLIP_REINFLATE_K", "0.0"))
# Ticket 24: heavy-tailed (scale-mixture-of-normals) process-noise treatment
# for the Gaussian PRIOR term inside adf_update's/adf_update_full_dp's Newton
# step - the actual generalization this ticket asked for, not another
# clamp-detection proxy like CLIP_REINFLATE_K above. A literal Student-t
# random walk (mixture/lambda variables carried and convolved at every one of
# ~1.9M per-match time steps) has no closed-form recursion, so per issue 24's
# own spec ("approach a": a local Gaussian approximation with an
# adjusted/inflated variance per update) this treats each individual ADF
# update's local Gaussian prior (m_d, v_d, fed by the ordinary Gaussian
# random walk in inflate()) as an approximate Student-t via its scale-mixture
# representation, solved by iteratively reweighting the prior's EFFECTIVE
# variance inside the SAME Newton loop already in adf_update - no new outer
# loop, no new state carried between matches, no new per-player fields.
# See _heavy_tail_var's docstring for the exact formula and the deliberate
# floor-at-1.0 deviation from a literal Student-t. Default 0 = exact
# pre-existing Gaussian behaviour (verified bit-identical, not just
# mathematically equal - v_eff is never reassigned from v_d when this is off).
HEAVY_TAILED_NOISE = os.environ.get("DSV_HEAVY_TAILED_NOISE", "0") == "1"
STUDENT_T_NU = float(os.environ.get("DSV_STUDENT_T_NU", "4.0"))  # tuned only on 2022-23
# Ticket 22: tournament-tier weighting. tourney_level/round were loaded but
# never used. ticket 20's REGIME_B_N scan proved that RAISING evidence weight
# for any match subset reopens the absorbing-boundary corruption trap (no
# value above baseline passed the gate) - so this only ever DOWN-weights
# lower-tier matches (multiplier in (0,1]), never exceeding the
# already-validated baseline evidence for any match. Elite = Slam/Masters
# 1000/Tour Finals/Premier Mandatory/Olympics (~9.9% of the archive).
ELITE_LEVELS = frozenset({"G", "M", "F", "PM", "O"})
TIER_WEIGHT = float(os.environ.get("DSV_TIER_WEIGHT", "1.0"))  # multiplier for NON-elite matches
# Issue 20 round 4: the locked project decision (dropshotverse-tennis-effort
# memory) treats "WTA qual/ITF" as a match-outcome COLD-START PRIOR ONLY,
# distinct from ATP Challenger's full inclusion - but that distinction was
# never implemented in code. TIER_WEIGHT above is symmetric across tours (it
# keys only on tourney_level, so it discounts ATP Challenger exactly as much
# as WTA ITF, which is presumably why ticket 22's TIER_WEIGHT scan didn't
# transfer to the pooled holdout - it traded away real ATP Challenger signal
# to buy a WTA fix). This is a genuinely tour-conditioned lever: only
# level_file=="lower" WTA rows (qual/ITF/$25k-$100k circuit) are discounted;
# ATP's lower tier (Challenger) is untouched. Multiplier in (0,1], default
# 1.0 = no-op (identical to current behaviour) until scanned through the
# corruption gate. Motivation: WTA's lower-tier archive volume is ~3.8x its
# main-tour volume (605k vs 158k rows) vs ATP's ~1.2x (232k vs 194k), and
# only 27.9% of WTA lower-tier rows (2022-25) carry point-level data vs
# 93-99.8% for every other tour/level combination - so WTA main-tour player
# states are disproportionately built from high-volume, low-instrumentation,
# weak-field evidence relative to every other population in the archive.
WTA_LOWER_WEIGHT = float(os.environ.get("DSV_WTA_LOWER_WEIGHT", "1.0"))
# Default cross-validated in ticket 17 (coordinate search, selected on the
# 2022-23 window only, verified on the untouched 2024-25 holdout): 0.010 -> 0.005.
RW_VAR_PER_WEEK = float(os.environ.get("DSV_RW_VAR_PER_WEEK", "0.005"))
LAYOFF_WEEKS = 26.0      # gap after which variance inflation accelerates
LAYOFF_C = float(os.environ.get("DSV_LAYOFF_C", "0.06"))
# Discovered empirically 2026-08-27: the raw phi(gap) formula is UNBOUNDED and
# scales roughly quadratically with gap length (inc ~ weeks * c * weeks for
# large weeks). A player with a ~15-year gap between two tour matches produced
# phi~46 and variance in the hundreds, which then let a single regime-B win
# (bounded only by that variance) shift their mean by hundreds of logit units -
# the actual cause of the runaway leaderboard values, not a re-run of the
# earlier share-reuse bug. A real player's uncertainty after any layoff,
# however long, should saturate rather than grow without bound. Both caps
# below are the fix: NOT pre-registered, added in direct response to this
# failure, and reported as such rather than silently folded into "the design".
PHI_MAX = float(os.environ.get("DSV_PHI_MAX", "4.0"))
# Ticket 20 diagnostic-agent finding (round 3, 2026-08-28): calibration by
# layoff length, controlled for round to rule out a round confound, showed
# real (non-noise) OVERconfidence specifically in the 1-12 week gap regime
# (error -0.04 to -0.16), well below the 26-week threshold LAYOFF_WEEKS/phi
# was designed for (injury-scale layoffs). A normal between-tournament gap
# currently adds almost no variance (RW_VAR_PER_WEEK*weeks*phi(~1) for a
# typical 2-week gap is ~0.01, negligible against typical total variance of
# 0.3-0.6) - the same weak-disagreement-with-Elo matches that come back wrong
# more often than chance cluster in exactly this window. Independent of the
# long-layoff phi mechanism (which stays untouched - it targets a different,
# already-validated failure mode): a flat step bump once ANY real gap has
# occurred. Off (0.0) until scanned.
SHORT_LAYOFF_BUMP = float(os.environ.get("DSV_SHORT_LAYOFF_BUMP", "0.0"))
SHORT_LAYOFF_MIN_WEEKS = float(os.environ.get("DSV_SHORT_LAYOFF_MIN_WEEKS", "1.0"))
# Round-3-continuation (2026-08-28), attempt 15 in issues/20: the unbounded
# version above (any weeks>=SHORT_LAYOFF_MIN_WEEKS, forever) was already tried
# and rejected - it conflates the diagnostic's actual 1-12wk finding with
# multi-year layoffs already handled (and validated) by the separate
# LAYOFF_C/phi mechanism, double-counting variance for those. This upper
# bound scopes the bump to the window the diagnostic evidence actually
# supports. Default is effectively unbounded (huge) so leaving it unset
# reproduces the old (rejected, still-0.0-by-default) behaviour exactly.
SHORT_LAYOFF_MAX_WEEKS = float(os.environ.get("DSV_SHORT_LAYOFF_MAX_WEEKS", "1e9"))
MAX_VAR = float(os.environ.get("DSV_MAX_VAR", "4.0"))     # ceiling on any single mu_alpha/mu_beta variance
# Changed 2026-08-27 (ticket 18), same session as the kappa-shrinkage fix:
# 1.0 -> 0.3. Real-population trace (DSV_DEBUG_UID) showed Jannik Sinner
# hitting MAX_STATE within his first 3-4 tracked matches in 2018 as an
# unproven 17-year-old losing to established Challenger players - a legitimate
# early dip that should NOT reach the system's most extreme possible value in
# a handful of matches. Facing already-confident opponents makes each early
# match's raw (pre-clamp) shift larger than a synthetic equally-uncertain-
# opponent test would predict, so several near-cap events in a row marched him
# straight to the wall. At 0.3 his real early trajectory stays sane throughout.
MAX_SHIFT = float(os.environ.get("DSV_MAX_SHIFT", "0.2"))  # ceiling on |shift| from any single observation
                             # (ticket 19 re-tune, was stale at 0.3 - every fit
                             # script overrides this explicitly, so the stale
                             # default never actually affected a reported result)

# Capping per-event shift and per-player variance was NOT enough: a debug
# replay of one affected player's OWN matches in isolation gave sane values
# (~0.46, -0.57), while the full fit gave -41.5/+145.5 for the same player.
# The corruption is transmitted THROUGH opponents drawn from a weakly-connected
# lower-tier sub-population that rarely faces the tour-level "anchor" players -
# a known pathology of Bradley-Terry/Elo-style systems on sparse graphs. Small
# per-step caps only slow an unbounded random walk; they do not stop it from
# drifting arbitrarily far given years of matches within an insular pool.
# Spec section 3 anticipated exactly this and named the fix: "the obvious
# upgrade is a mean-reverting (Ornstein-Uhlenbeck) process if random-walk
# skills drift implausibly [long]." Implementing that now, in direct response
# to this failure - a third documented deviation from the pure-random-walk
# design, not planned in advance.
# Default cross-validated in ticket 17: 0.0025 -> 0.0010 (~13-year half-life).
# The original 0.0025 was reactive - added to stop a specific runaway-drift
# bug, not tuned; once the actual bugs behind that failure were fixed, the
# data wanted much slower reversion than the safety-net value.
# Re-tuned in ticket 19 (lopsided-weighted metric, post bugs-6-9 fix): 0.0010
# -> 0.0005 (~26-year half-life). Slower reversion trusts an established
# rating more, which is what improved lopsided-match confidence.
OU_THETA_PER_WEEK = float(os.environ.get("DSV_OU_THETA_PER_WEEK", "0.0005"))
                             # original rationale: ~5.3-year half-life: slow relative to a season,
                             # strong enough to bound decade-scale drift

# Fourth and final safety net, found via Milos Raonic: mu_a=-39.3 after 600
# matches and several injury-layoff comebacks, despite v_a sitting at the
# MAX_VAR ceiling (proving the earlier caps engaged - the mean still drifted
# to an implausible magnitude across MULTIPLE long-layoff-then-early-loss
# cycles, a compounding path none of the per-step caps individually bounds).
# sigmoid(+-3.0) = 0.95/0.05 is already an extreme single-point-probability
# edge for real tennis; no player's baseline serve/return skill should exceed
# it. A hard ceiling on state magnitude, same safety-net philosophy as
# MAX_VAR, is the direct fix - not a re-derivation of the compounding path.
MAX_STATE = float(os.environ.get("DSV_MAX_STATE", "3.0"))
# Default cross-validated in ticket 17: 1.0x -> 4.0x (looser surface shrinkage
# than originally guessed - genuine specialists need more room than assumed).
# Re-tuned in ticket 18 (post bugs-6-9 fix) to 2.0x, then ticket 19 (lopsided-
# weighted metric) to 1.0x - once kappa carried real shrinking variance
# (bug 6) the looser 4.0x/2.0x scales were inflating surface uncertainty more
# than the data supported, which was suppressing top-end confidence.
_SURF_SCALE = float(os.environ.get("DSV_SURFACE_VAR_SCALE", "1.0"))
SURFACE_PRIOR_VAR = {k: v * _SURF_SCALE for k, v in
                     {"Hard": 0.10, "Clay": 0.14, "Grass": 0.22,
                      "Carpet": 0.20, "Unknown": 0.10}.items()}
MEASUREMENT_FLOOR_N = 1  # binomial n=0 guard


def log(m: str) -> None:
    print(m, flush=True)


_SET_SCORE_RE = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")


def parse_score_games(score: str) -> tuple[float, float]:
    """Ticket 21: sum games won per side across all sets, e.g. "6-4 3-6
    7-6(4)" -> (16.0, 16.0). Tiebreak point counts in parens are discarded -
    only the games column is used. Returns (nan, nan) for anything that
    isn't a plain set-by-set score (match-tiebreak deciders written as
    "[10-8]", or any other irregular format) so the caller falls back to the
    old win/loss-only evidence rather than guess at a malformed token."""
    total_w = total_l = 0
    tokens = str(score).split()
    if not tokens:
        return math.nan, math.nan
    for tok in tokens:
        m = _SET_SCORE_RE.match(tok)
        if not m:
            return math.nan, math.nan
        total_w += int(m.group(1))
        total_l += int(m.group(2))
    if total_w + total_l <= 0:
        return math.nan, math.nan
    return float(total_w), float(total_l)


_LEGAL_SET_SCORES = frozenset(
    {(a, b) for a in range(0, 8) for b in range(0, 8)
     if ((a >= 6 or b >= 6) and abs(a - b) >= 2 and max(a, b) <= 7)
     or (a, b) in {(7, 6), (6, 7)}}
)


def parse_score_sets(score: str) -> list[tuple[int, int]] | None:
    """Ticket 21 full-DP: per-set (winner_games, loser_games) tuples, order
    preserved (unlike parse_score_games's lumped sum) - e.g. "3-6 7-6(4) 6-2"
    -> [(3,6),(7,6),(6,2)]. Winner's games are always token-group(1), the same
    archive convention parse_score_games already relies on (the match
    winner's games are listed first in each set token, regardless of who won
    that particular set).

    Returns None (never a partial list) if ANY set fails to parse OR isn't a
    legal short-form set score representable by p_set_full's DP - old-rules
    undecided long final sets (e.g. pre-tiebreak-era deciders like 70-68), or
    a malformed/irregular token. The caller falls back to the existing
    games-won-count approximation for the WHOLE match in that case, rather
    than mixing two different likelihood models within one match."""
    tokens = str(score).split()
    if not tokens:
        return None
    sets: list[tuple[int, int]] = []
    for tok in tokens:
        m = _SET_SCORE_RE.match(tok)
        if not m:
            return None
        ga, gb = int(m.group(1)), int(m.group(2))
        if (ga, gb) not in _LEGAL_SET_SCORES:
            return None
        sets.append((ga, gb))
    return sets or None


def sigmoid(x: float) -> float:
    if x > 35:
        return 1.0
    if x < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------- ticket-05 recursion
# Array-based (not dict-based) for speed - this runs twice per eval match.

def p_game(p: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1 - p
    base = p**4 + 4 * p**4 * q + 10 * p**4 * q**2
    deuce = 20 * p**3 * q**3 * (p * p) / (1 - 2 * p * q)
    return base + deuce


def p_tiebreak(p_a_serve: float, p_b_serve: float) -> float:
    """P(A wins a 7-point tiebreak), A serves point 1, then alternating pairs.

    Newton & Keller eqs. 13-16 (ticket 05 section 1b): forward DP to the
    absorbing scores, then an EXACT geometric tail from 6-6.

    Previously the tail was approximated by running the DP to a 24-point cap
    and splitting whatever mass was still tied 50/50, on the reasoning that
    "both players' serve probabilities roughly cancel over a long alternating
    tail". They do not: the tail favours the stronger server, and the coin flip
    handed the weaker player half of it. That cost up to 8.0e-4 on the tiebreak
    and, because a set at 6-6 resolves through here, fed straight into p_set.
    """
    # A serves point t (1-based) iff (t-1) mod 4 in {0, 3}: A,B,B,A,A,B,B,...
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    win_a = 0.0
    p66 = 0.0
    for t in range(1, 14):                  # 12 points reach 7-5 or 6-6; 13 is slack
        a_serves = (t - 1) % 4 in (0, 3)
        pa = p_a_serve if a_serves else (1 - p_b_serve)
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), pr in states.items():
            for (na, nb), w in (((a + 1, b), pa), ((a, b + 1), 1 - pa)):
                prob = pr * w
                if prob == 0:
                    continue
                if na == 7 and nb <= 5:
                    win_a += prob
                elif nb == 7 and na <= 5:
                    pass                    # B wins outright
                elif na == 6 and nb == 6:
                    p66 += prob             # hand to the geometric tail below
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    if states:      # unreachable: every path absorbs by point 12
        raise AssertionError(f"p_tiebreak left unresolved states: {states}")

    # From 6-6 each 2-point block (A serves, then B) either ends it or returns
    # to a tie with probability r. A takes the block outright with pa*(1-pb).
    r = p_a_serve * p_b_serve + (1 - p_a_serve) * (1 - p_b_serve)
    if 1 - r <= 0.0:
        # Degenerate: both hold every point (1,1) or both lose every point
        # (0,0), so no block ever resolves and the tail is undefined. Reachable
        # because sigmoid() clamps to exactly 0.0/1.0 beyond +/-35. Split it.
        return win_a + p66 * 0.5
    return win_a + p66 * (p_a_serve * (1 - p_b_serve) / (1 - r))


def p_set_full(p_a_serve: float, p_b_serve: float, a_serves_first: bool
              ) -> dict[tuple[int, int], float]:
    """Game-by-game DP returning the complete terminal-score distribution
    {(games_a, games_b): probability}. p_set() is a thin wrapper (sum over
    a > b); regime B uses the full distribution to score a SPECIFIC observed
    set score (e.g. exactly 6-4), not just win/loss.

    6-6 resolution (Newton & Keller eq. 10, ticket 05 section 1c):

        P_A^set = sum_j P(6,j) + P(7,5) + P(6,6) * P_A^tiebreak

    At 6-6 a TIEBREAK is played. There is no 13th game. The DP used to play
    one anyway and then multiply by the tiebreak probability, so the mass
    reaching (7,6) was P(6,6)*[pa*tb + (1-pa)*(1-tb)] instead of P(6,6)*tb -
    and the second term credited A with B's tiebreak share. Two consequences,
    both measured against the closed form:

      * it compressed the favourite's set probability by up to 5.7pp, and
      * the 13th game's server depends on `a_serves_first`, so p_set varied by
        up to 5.1pp with serve order. Newton & Keller section 3.3 PROVE set-win
        probability is independent of who serves first; p_match's
        q = 0.5*(q1+q2) symmetrisation was averaging away a violation rather
        than a genuine asymmetry.

    The previous docstring described this as a "7-6 quirk, preserved as-is"
    for the ticket 21 refactor. It was a defect, and stress_test_safety.py
    certified it by comparing the DP to a verbatim copy of itself."""
    outcomes: dict[tuple[int, int], float] = {}
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    tb = p_tiebreak(p_a_serve, p_b_serve)   # hoisted: was recomputed per state
    for _ in range(13):
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), pr in states.items():
            game_idx = a + b
            a_serves_game = (game_idx % 2 == 0) == a_serves_first
            pa = p_game(p_a_serve) if a_serves_game else 1 - p_game(p_b_serve)
            for winner, w in ((0, pa), (1, 1 - pa)):
                na, nb = (a + 1, b) if winner == 0 else (a, b + 1)
                prob = pr * w
                if prob == 0:
                    continue
                if (na >= 6 or nb >= 6) and abs(na - nb) >= 2 and max(na, nb) <= 7:
                    outcomes[(na, nb)] = outcomes.get((na, nb), 0.0) + prob
                elif na == 6 and nb == 6:
                    outcomes[(7, 6)] = outcomes.get((7, 6), 0.0) + prob * tb
                    outcomes[(6, 7)] = outcomes.get((6, 7), 0.0) + prob * (1 - tb)
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    leftover = sum(states.values())   # negligible leftover mass (deep-deuce tail)
    if leftover > 0:
        outcomes[(7, 6)] = outcomes.get((7, 6), 0.0) + leftover * 0.5
        outcomes[(6, 7)] = outcomes.get((6, 7), 0.0) + leftover * 0.5
    return outcomes


def p_set(p_a_serve: float, p_b_serve: float, a_serves_first: bool) -> float:
    """P(A wins the set). Exact DP over games; 7-6 resolved via p_tiebreak.
    Now a thin wrapper over p_set_full (ticket 21 refactor) - see that
    docstring for the bit-identical-behaviour guarantee."""
    return sum(pr for (a, b), pr in
              p_set_full(p_a_serve, p_b_serve, a_serves_first).items() if a > b)


def p_match(p_a_serve: float, p_b_serve: float, best_of: int) -> float:
    """P(A wins the match). Symmetrised over who serves set 1; sets treated iid
    given the point probabilities (standard tennis-model assumption, ticket 05)."""
    q1 = p_set(p_a_serve, p_b_serve, a_serves_first=True)
    q2 = p_set(p_a_serve, p_b_serve, a_serves_first=False)
    q = 0.5 * (q1 + q2)
    need = 2 if best_of == 3 else 3
    # DP over sets won
    states = {(0, 0): 1.0}
    final = 0.0
    for _ in range(need * 2 - 1):
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), pr in states.items():
            for winner, w in ((0, q), (1, 1 - q)):
                na, nb = (a + 1, b) if winner == 0 else (a, b + 1)
                prob = pr * w
                if na == need:
                    final += prob
                elif nb == need:
                    pass
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    return final


# ------------------------------------------------------------- ADF/Laplace update

def apply_side_update(mu_mean: float, mu_var: float, kappa_mean: float,
                      kappa_var: float, old_effective: float,
                      new_effective_mean: float, new_effective_var: float
                      ) -> tuple[float, float, float, float]:
    """Split a shift/variance-reduction in an 'effective' composite term
    (mu + kappa) back onto mu AND kappa, both with real evolving posterior
    variance. Self-contained, no shared mutable state across calls - the
    original bug this replaced was exactly that: a `share` variable computed
    for one side and silently reused for another.

    CHANGED 2026-08-27 (ticket 18), in direct response to a Pete-requested
    anchor check: Djokovic vs a qualifier (AO 2024 R1) scored 70.3% - Elo said
    97.6%. Root cause found by inspecting the variance directly: kappa
    previously had a FIXED prior variance that NEVER shrank (ticket 10's
    documented v0 simplification), so ticket 17's SURFACE_VAR_SCALE=4.0 - which
    improved aggregate tuning-window log-loss - created a variance FLOOR of
    ~2x0.40=0.80 on Hard court for every prediction, regardless of match count.
    Djokovic's own mu variance had shrunk to nearly nothing; the fixed surface
    term dominated the total and never let genuinely lopsided predictions
    happen. Kappa now gets the SAME Kalman-style shrinkage mu already had.

    CHANGED AGAIN, same session: the fix above reopened the exact class of bug
    it was meant to close, through a THIRD mechanism - an absorbing-boundary
    trap. When MAX_SHIFT or MAX_STATE clamps a mean update, the variance
    reduction below was computed as if the FULL, unclamped update had been
    accepted - so a player pinned at a wall by early noise kept getting MORE
    confident every subsequent match, even though the wall meant most of each
    update's intended movement was being thrown away. Once variance is tiny
    and the mean is stuck at the wrong boundary, no amount of contradicting
    future evidence can ever pull it back (Newton's pull is proportional to
    relative precision, which only ever increases). Found via a Pete-requested
    anchor check on Jannik Sinner (his OWN eventual state, not an opponent's,
    reached mu_a=-3.0 despite 490 matches and a career log that is clean and
    sane in isolation) and confirmed at population scale: 2,591 of 44,585
    players pinned at |value| > 2.5, concentrated among the mean-clamp-prone.
    Fix: variance reduction is now scaled by how much of the INTENDED update
    actually survived clamping - a component whose mean update was fully
    absorbed by a wall gets NO variance reduction that round, since no real
    information reached the point estimate.
    """
    v_e = mu_var + kappa_var
    share_mu = mu_var / v_e if v_e > 0 else 0.5
    if FORCE_SHARE_MU is not None:
        share_mu = FORCE_SHARE_MU
    raw_shift = new_effective_mean - old_effective
    shift = max(-MAX_SHIFT, min(MAX_SHIFT, raw_shift))
    shift_survival = abs(shift / raw_shift) if raw_shift != 0 else 1.0

    intended_mu_mean = mu_mean + share_mu * shift
    intended_kappa_mean = kappa_mean + (1 - share_mu) * shift
    new_mu_mean = max(-MAX_STATE, min(MAX_STATE, intended_mu_mean))
    new_kappa_mean = max(-MAX_STATE, min(MAX_STATE, intended_kappa_mean))

    mu_delta_intended = share_mu * shift
    kappa_delta_intended = (1 - share_mu) * shift
    mu_survival = abs((new_mu_mean - mu_mean) / mu_delta_intended) if mu_delta_intended != 0 else 1.0
    kappa_survival = abs((new_kappa_mean - kappa_mean) / kappa_delta_intended) if kappa_delta_intended != 0 else 1.0

    reduction = max(0.0, 1 - new_effective_var / v_e) if v_e > 0 else 0.0
    new_mu_var = mu_var * (1 - reduction * share_mu * shift_survival * mu_survival)
    new_kappa_var = kappa_var * (1 - reduction * (1 - share_mu) * shift_survival * kappa_survival)
    # Ticket 24 (scoped-down): bug 8's shift_survival fix above only stops
    # variance from SHRINKING on a clamped (surprising) update - it never lets
    # variance GROW to reflect that a big surprise was only partially
    # absorbed. A true heavy-tailed process-noise model would do that
    # automatically; this is the cheap proxy - reinflate proportional to how
    # much of the intended move got clipped away, so a wrong lock-in caused by
    # a clamp can still be corrected by later evidence instead of freezing.
    # Off by default (K=0) so existing behaviour is unchanged until scanned.
    clip_frac_mu = 1.0 - shift_survival * mu_survival
    clip_frac_kappa = 1.0 - shift_survival * kappa_survival
    new_mu_var = new_mu_var * (1 + CLIP_REINFLATE_K * clip_frac_mu)
    new_kappa_var = new_kappa_var * (1 + CLIP_REINFLATE_K * clip_frac_kappa)
    new_mu_var = min(MAX_VAR, new_mu_var)
    new_kappa_var = min(MAX_VAR, new_kappa_var)
    return new_mu_mean, new_mu_var, new_kappa_mean, new_kappa_var


def _heavy_tail_var(d: float, m_d: float, v_d: float, nu: float) -> float:
    """Ticket 24: effective (possibly inflated) prior variance for a Newton
    step's prior term, approximating a Student-t(nu) prior on d via a
    floored scale-mixture-of-normals weight.

    Standard Student-t-as-scale-mixture EM/IRLS result: d | lambda ~
    N(m_d, v_d/lambda), lambda ~ Gamma(nu/2, nu/2) => posterior
    E[lambda | d] = (nu+1) / (nu + r^2), r^2 = (d-m_d)^2 / v_d - so the
    effective conditional variance at the current Newton iterate is
    v_d / E[lambda|d] = v_d * (nu + r^2) / (nu + 1).

    DEVIATION FROM A LITERAL STUDENT-T, BY DESIGN: a genuine Student-t with
    scale^2=v_d is actually slightly TIGHTER than v_d exactly at the mode
    (ratio nu/(nu+1) < 1 at r=0) - accepting that would silently retune
    every ordinary, unsurprising update's effective variance, entangling
    this ticket's fix with every other hyperparameter that already assumes
    v_d IS the working variance (RW_VAR_PER_WEEK, PHI_MAX, etc., all tuned
    under that assumption). Flooring the ratio at 1.0 means v_d is NEVER
    shrunk below its ordinary value, only ever inflated, and only once r^2
    exceeds 1 (nu+r^2 > nu+1). A typical/unsurprising update (r^2<=1) is
    therefore byte-for-byte unaffected by nu - the whole mechanism is
    isolated to genuinely large surprises, the exact failure mode ticket 24
    targets (what currently hits MAX_SHIFT), without silently reopening any
    already-tuned hyperparameter's calibration."""
    if v_d <= 0:
        return v_d
    r2 = (d - m_d) * (d - m_d) / v_d
    ratio = (nu + r2) / (nu + 1.0)
    return v_d * ratio if ratio > 1.0 else v_d


def adf_update(m_a: float, v_a: float, m_b: float, v_b: float,
              y: float, n: float) -> tuple[float, float, float, float]:
    """Binomial Laplace update on d = a-b, y successes of n trials, p=sigmoid(d).
    Returns updated (m_a, v_a, m_b, v_b), splitting shift & variance reduction
    proportional to each side's prior variance share (Kalman-gain-like).

    Ticket 24 (DSV_HEAVY_TAILED_NOISE): when enabled, the Gaussian prior
    term's variance v_d is replaced each Newton iteration by
    _heavy_tail_var(d, m_d, v_d, STUDENT_T_NU), a floored scale-mixture
    approximation to a Student-t(nu) prior - recomputed against the CURRENT
    Newton iterate `d`, so this is a genuine iteratively-reweighted (IRLS)
    scheme converging toward the Student-t-prior MAP, not a single static
    adjustment. When off (default), v_eff is never reassigned from v_d, so
    this is bit-identical to the pre-ticket-24 Gaussian-only update."""
    if n <= 0:
        return m_a, v_a, m_b, v_b
    m_d, v_d = m_a - m_b, v_a + v_b
    d = m_d
    v_eff = v_d
    for _ in range(6):
        p = sigmoid(d)
        if HEAVY_TAILED_NOISE:
            v_eff = _heavy_tail_var(d, m_d, v_d, STUDENT_T_NU)
        grad = (y - n * p) - (d - m_d) / v_eff
        hess = -(n * p * (1 - p) + 1.0 / v_eff)
        if hess == 0:
            break
        d -= grad / hess
    p = sigmoid(d)
    if HEAVY_TAILED_NOISE:
        v_eff = _heavy_tail_var(d, m_d, v_d, STUDENT_T_NU)
    var_post = 1.0 / (n * p * (1 - p) + 1.0 / v_eff)
    shift = d - m_d
    share_a = v_a / (v_a + v_b)
    share_b = 1.0 - share_a
    # reduction stays relative to the REAL prior variance v_d (not v_eff) -
    # v_eff is only a temporary Newton-step device for how much the mean is
    # allowed to move under a heavy tail; "how much did this observation
    # actually shrink our real uncertainty" is still measured against the
    # real prior, same as the pre-ticket-24 code.
    reduction = max(0.0, 1.0 - var_post / v_d)
    return (m_a + share_a * shift, v_a * (1 - reduction * share_a),
            m_b - share_b * shift, v_b * (1 - reduction * share_b))


# --------------------------------------------------- ticket 21: full-DP regime-B
# Off by default (DSV_REGIME_B_FULL_DP=0) - the fallback games-won-count path
# above (adf_update on parse_score_games's summed counts) stays the exact
# behaviour v25 was selected on. This block replaces it with the ACTUAL
# log-likelihood of the observed set-by-set scoreline, computed from p_set's
# own game-by-game DP, when enabled.
REGIME_B_FULL_DP = os.environ.get("DSV_REGIME_B_FULL_DP", "0") == "1"

# A fixed, pre-registered grid, not an unbounded cache: a fit walks through
# ~1.9M matches and the combined-strength gap d ranges continuously as player
# states evolve, so an LRU keyed on raw d values would grow without a real
# bound over a long run. Snapping to a fixed 0.01-wide grid over the full
# reachable range of d (cw,cl each bounded by +-2*MAX_STATE, so d in
# +-4*MAX_STATE with margin) gives O(1) lookups and a fixed, predictable
# memory footprint regardless of run length.
FULL_DP_GRID_STEP = 0.01
FULL_DP_GRID_MIN = -12.5
FULL_DP_GRID_MAX = 12.5
FULL_DP_H = 0.05   # finite-difference step for the Newton solve (5 grid
                    # steps - comfortably resolved after grid-snapping, so
                    # d-h/d/d+h never alias to the same grid point)
_SCORE_PROB_FLOOR = 1e-6   # ticket 12: a single extreme set (e.g. 6-0) at a
                            # moderate d can have near-zero DP mass - floor it
                            # so log-likelihood/Newton curvature stay finite.
                            # This is the exact per-match grid-boundary
                            # failure mode ticket 12 warned about, guarded
                            # against here instead of rediscovered as a NaN.

_full_dp_grid_cache: dict[float, dict[tuple[int, int], float]] = {}


def _snap_to_grid(d: float) -> float:
    d = max(FULL_DP_GRID_MIN, min(FULL_DP_GRID_MAX, d))
    return round(round(d / FULL_DP_GRID_STEP) * FULL_DP_GRID_STEP, 6)


def set_outcome_dist_for_d(d: float) -> dict[tuple[int, int], float]:
    """Ticket 21: the per-set outcome distribution for combined-strength gap
    d, under the symmetric split ticket 12 established is the only
    identifiable option from a single match - alpha=beta=c/2 for each
    player, so a player's own serve-point-win probability is
    sigmoid(their combined edge / 2): p_a_serve=sigmoid(d/2),
    p_b_serve=sigmoid(-d/2)=1-p_a_serve (NOT p_a_serve==p_b_serve - that
    degenerate reading would make p_match(p,p,...) ~= 0.5 for every d,
    destroying all signal; the "symmetric" assumption is between a player's
    OWN serve and return skill, not between the two players' serve rates).
    At d=0 this gives p_a_serve=p_b_serve=0.5, the same coin-flip-point
    default regime A already has for two untested players with m_a=m_b=0.

    Symmetrized over which side serves the first game of the set (real
    who-served-set-1 data isn't in the archive), matching p_match's own
    q1/q2 averaging. Snapped to the fixed grid and cached (see module-level
    comment) rather than computed fresh per call."""
    d_snap = _snap_to_grid(d)
    cached = _full_dp_grid_cache.get(d_snap)
    if cached is not None:
        return cached
    pa = sigmoid(d_snap / 2.0)
    pb = sigmoid(-d_snap / 2.0)
    o1 = p_set_full(pa, pb, True)
    o2 = p_set_full(pa, pb, False)
    keys = set(o1) | set(o2)
    dist = {k: 0.5 * (o1.get(k, 0.0) + o2.get(k, 0.0)) for k in keys}
    _full_dp_grid_cache[d_snap] = dist
    return dist


def p_specific_set_score(d: float, ga: int, gb: int) -> float:
    """P(this exact set score | d), floored per _SCORE_PROB_FLOOR."""
    return max(set_outcome_dist_for_d(d).get((ga, gb), 0.0), _SCORE_PROB_FLOOR)


def match_log_likelihood(d: float, set_scores: list[tuple[int, int]]) -> float:
    """log P(the full observed set-by-set scoreline | d) - product over sets
    (independence given d is the same standard tennis-model assumption
    p_match already makes for whole sets, ticket 05)."""
    return sum(math.log(p_specific_set_score(d, ga, gb)) for ga, gb in set_scores)


def adf_update_full_dp(m_a: float, v_a: float, m_b: float, v_b: float,
                       set_scores: list[tuple[int, int]], weight: float = 1.0
                       ) -> tuple[float, float, float, float]:
    """Ticket 21: regime-B update using the ACTUAL log-likelihood of the
    observed set-by-set scoreline (via set_outcome_dist_for_d / p_set's own
    DP), replacing adf_update's Bernoulli games-won-count approximation.

    adf_update's closed-form Newton step is Bernoulli-conjugate-specific -
    its grad/hess have a closed form only because the binomial
    log-likelihood does. This richer likelihood has no closed form, so
    grad/hess of the LIKELIHOOD term are estimated by central finite
    differences and then combined with the Gaussian PRIOR term exactly the
    way adf_update does (same Laplace/Newton structure, same final
    prior-variance-share split into m_a/m_b) - a numerical generalisation of
    the same update, not a different mechanism.

    `weight` reuses the existing tier_scale evidence-strength dial
    (REGIME_B_N * TIER_WEIGHT * WTA_LOWER_WEIGHT) as a temperature on the
    log-likelihood (LL -> weight*LL) - the same role REGIME_B_N already
    plays by scaling y and n together in the old approximation, so the
    already-tuned dials stay meaningful under this mode instead of needing a
    brand new redundant hyperparameter.

    Ticket 24 (DSV_HEAVY_TAILED_NOISE): same floored scale-mixture prior
    treatment as adf_update - see _heavy_tail_var's docstring. Applies to
    the same Gaussian prior term (-(d-m_d)/v_d, -1/v_d) this function's
    Newton step already carries; off by default, bit-identical when off."""
    m_d, v_d = m_a - m_b, v_a + v_b
    if v_d <= 0 or not set_scores or weight <= 0:
        return m_a, v_a, m_b, v_b
    d = m_d
    h = FULL_DP_H
    v_eff = v_d
    hess = -1.0 / v_eff
    for _ in range(4):
        if HEAVY_TAILED_NOISE:
            v_eff = _heavy_tail_var(d, m_d, v_d, STUDENT_T_NU)
        ll_p = weight * match_log_likelihood(d + h, set_scores)
        ll_0 = weight * match_log_likelihood(d, set_scores)
        ll_m = weight * match_log_likelihood(d - h, set_scores)
        grad_ll = (ll_p - ll_m) / (2 * h)
        hess_ll = (ll_p - 2 * ll_0 + ll_m) / (h * h)
        grad = grad_ll - (d - m_d) / v_eff
        hess = hess_ll - 1.0 / v_eff
        if hess == 0 or not math.isfinite(hess) or not math.isfinite(grad):
            break
        step = grad / hess
        if not math.isfinite(step):
            break
        d -= step
        d = max(FULL_DP_GRID_MIN, min(FULL_DP_GRID_MAX, d))
    # Final curvature at the converged d, for the posterior (Laplace) variance.
    if HEAVY_TAILED_NOISE:
        v_eff = _heavy_tail_var(d, m_d, v_d, STUDENT_T_NU)
    ll_p = weight * match_log_likelihood(d + h, set_scores)
    ll_0 = weight * match_log_likelihood(d, set_scores)
    ll_m = weight * match_log_likelihood(d - h, set_scores)
    hess_ll = (ll_p - 2 * ll_0 + ll_m) / (h * h)
    hess = hess_ll - 1.0 / v_eff
    if math.isfinite(hess) and hess < 0:
        var_post = -1.0 / hess
    else:
        # pathological/flat curvature (should not happen given the floor on
        # p_specific_set_score, but guarded rather than trusted): don't
        # shrink variance at all this round, same fail-safe spirit as
        # adf_update's n<=0 guard.
        var_post = v_d
    var_post = max(var_post, 1e-6)
    shift = d - m_d
    share_a = v_a / (v_a + v_b)
    share_b = 1.0 - share_a
    reduction = max(0.0, 1.0 - var_post / v_d)
    return (m_a + share_a * shift, v_a * (1 - reduction * share_a),
            m_b - share_b * shift, v_b * (1 - reduction * share_b))


# ------------------------------------------------------------------- data prep

def load_matches() -> pd.DataFrame:
    frames = []
    cols = ["tourney_date", "tourney_level", "surface", "score", "best_of", "round",
            "winner_id", "loser_id", "winner_name", "loser_name",
            "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    for tour in ("atp", "wta"):
        pats = [f"{tour}_matches_[0-9]*.csv",
                f"{tour}_matches_qual_chall_[0-9]*.csv",
                f"{tour}_matches_qual_itf_[0-9]*.csv"]
        for pat in pats:
            for p in sorted((ARCHIVE / tour).glob(pat)):
                try:
                    df = pd.read_csv(p, low_memory=False)
                except Exception:                              # noqa: BLE001
                    continue
                if not len(df):
                    continue
                for c in cols:
                    if c not in df.columns:
                        df[c] = np.nan
                df["tour"] = tour
                df["level_file"] = "tour" if "qual" not in p.name else "lower"
                frames.append(df[cols + ["tour", "level_file"]])
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    dropped: dict[str, int] = {}
    n0 = len(df)

    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    bad = df["date"].isna()
    dropped["unparseable date"] = int(bad.sum())
    df = df[~bad]

    df["surface"] = df["surface"].astype(str).str.strip().str.title()
    df.loc[~df["surface"].isin(SURFACES), "surface"] = "Unknown"

    for side in ("winner", "loser"):
        df[f"{side}_uid"] = df["tour"] + ":" + df[f"{side}_id"].astype("Int64").astype(str)
    bad = df["winner_uid"].str.contains("<NA>") | df["loser_uid"].str.contains("<NA>")
    dropped["missing player id"] = int(bad.sum())
    df = df[~bad]

    sc = df["score"].astype(str)
    bad = sc.str.contains(r"RET|W/O|DEF|ABD|Walkover|Def\.", case=False, na=False) | (sc.str.strip() == "")
    dropped["retirement/walkover/no score"] = int(bad.sum())
    df = df[~bad]

    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    df.loc[~df["best_of"].isin([3, 5]), "best_of"] = 3

    df["w_won"] = df["w_1stWon"].fillna(0) + df["w_2ndWon"].fillna(0)
    df["l_won"] = df["l_1stWon"].fillna(0) + df["l_2ndWon"].fillna(0)
    df["has_points"] = df["w_svpt"].notna() & (df["w_svpt"] > 0) & \
                       df["l_svpt"].notna() & (df["l_svpt"] > 0)

    # Ticket 21: real games-per-set counts for regime B (see run()'s regime-B
    # branch) - parsed once here, not per-fit-iteration, since score never changes.
    parsed_games = df["score"].astype(str).map(parse_score_games)
    df["total_games_w"] = [p[0] for p in parsed_games]
    df["total_games_l"] = [p[1] for p in parsed_games]

    # Bug fix (see ROUND_RANK docstring above): tourney_date is a
    # tournament-level date, so sorting by date alone leaves same-date match
    # order at the mercy of raw file-concatenation order, which systematically
    # puts qualifying rounds after the whole main draw. Sort by (date,
    # round_rank) instead - matches with the identical date AND round rank
    # (the genuinely-simultaneous case, e.g. two different R32 matches on the
    # same day) keep their original file-order relative sequence via the
    # stable mergesort, same as before.
    df["_round_rank"] = round_rank(df["round"])
    df = df.sort_values(["date", "_round_rank"], kind="mergesort", na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_round_rank"])
    dropped["_total_in"] = n0
    dropped["_kept"] = len(df)

    # Speed fix: pandas 3.0 defaults string columns to a PyArrow-backed `str`
    # dtype. itertuples() (used once per match in run()'s hot loop, ~1.19M
    # times) iterates every column including these via Arrow's __iter__,
    # profiled at 26s of a ~65s fit just for that - 4x slower than plain
    # object dtype for the exact same values. Cast once here, not per fit.
    for col in df.columns:
        if df[col].dtype == "str":
            df[col] = df[col].astype(object)
    return df, dropped


PREPARED_CACHE_PATH = _common.SCRATCH / "prepared_matches_cache.pkl"


def load_prepared(use_cache: bool = True) -> tuple[pd.DataFrame, dict[str, int]]:
    """Speed fix: load_matches()+prepare() cost ~20s (12.9s CSV glob-read +
    7.6s parsing) and NEVER changes across hyperparameter candidates within a
    scan - re-paying it on every subprocess in a multi-candidate sweep is pure
    waste. Cache the prepared DataFrame, invalidated whenever any archive CSV
    is newer than the cache file (so a real data refresh still gets picked up)
    OR whenever this script itself is newer than the cache (so an edit to
    prepare()'s logic - e.g. the round-order sort fix - can't silently keep
    serving a cache built under the old code; caught the hard way: the
    archive-mtime check alone does NOT invalidate on a prepare()-only code
    change, since no CSV touches disk when only this file changes)."""
    if use_cache and PREPARED_CACHE_PATH.exists():
        archive_mtime = max((f.stat().st_mtime for f in ARCHIVE.rglob("*.csv")), default=0.0)
        script_mtime = Path(__file__).stat().st_mtime
        cache_mtime = PREPARED_CACHE_PATH.stat().st_mtime
        if cache_mtime > archive_mtime and cache_mtime > script_mtime:
            with open(PREPARED_CACHE_PATH, "rb") as fh:
                return pickle.load(fh)   # nosec: our own cache, written moments/days ago by this same script
    raw = load_matches()
    df, dropped = prepare(raw)
    if use_cache:
        PREPARED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREPARED_CACHE_PATH, "wb") as fh:
            pickle.dump((df, dropped), fh)
    return df, dropped


class PlayerState:
    """ticket 18: kappa/lambda moved IN from separate uid+surface-keyed dicts
    at the run() level, so inflate() can apply the same OU-reversion and
    variance-inflation to them that mu already gets. This was not optional
    once kappa's variance was allowed to shrink (see apply_side_update's
    docstring): a mean with a shrinking variance and NO reversion is exactly
    the failure mode ticket 10 spent hours diagnosing for mu, and giving
    kappa's variance the shrink without also giving its mean the reversion
    reopened that same class of bug through a different variable - caught by
    an anchor check (Sinner given essentially 0% to win his own serve point
    against a qualifier) rather than shipped silently.
    """
    __slots__ = ("m_a", "v_a", "m_b", "v_b", "kappa", "kappa_var",
                "lam", "lam_var", "last_seen", "n_matches")

    def __init__(self) -> None:
        self.m_a, self.v_a = 0.0, INIT_VAR
        self.m_b, self.v_b = 0.0, INIT_VAR
        self.kappa: dict[str, float] = {}
        self.kappa_var: dict[str, float] = {}
        self.lam: dict[str, float] = {}
        self.lam_var: dict[str, float] = {}
        self.last_seen: pd.Timestamp | None = None
        self.n_matches = 0

    def get_kappa(self, surface: str) -> tuple[float, float]:
        return self.kappa.get(surface, 0.0), self.kappa_var.get(surface, SURFACE_PRIOR_VAR[surface])

    def get_lam(self, surface: str) -> tuple[float, float]:
        return self.lam.get(surface, 0.0), self.lam_var.get(surface, SURFACE_PRIOR_VAR[surface])


def inflate(state: PlayerState, date: pd.Timestamp) -> None:
    if state.last_seen is None:
        state.last_seen = date
        return
    weeks = max(0.0, (date - state.last_seen).days / 7.0)
    if weeks <= 0:
        return
    phi = min(PHI_MAX, 1.0 + LAYOFF_C * max(0.0, weeks - LAYOFF_WEEKS))
    inc = RW_VAR_PER_WEEK * weeks * phi
    # Ticket 20 diagnostic-agent finding (round 3): calibration by layoff
    # length showed real overconfidence in the 1-12 week gap regime. Tried the
    # obvious fix - a flat variance bump for ANY player at ANY match once
    # weeks>=SHORT_LAYOFF_MIN_WEEKS, applied globally across the whole 50-year
    # archive - and it made the tuning-window metric monotonically WORSE as
    # the bump increased (0.6194->0.6245 from 0.0->0.06). The calibration
    # finding itself is real; this GLOBAL, UNCONDITIONAL implementation of the
    # fix is wrong - most short gaps in the archive are ordinary tour
    # scheduling, not the specific effect the diagnostic found. Left wired
    # (default 0.0, no-op) rather than removed, since a future attempt may
    # want to condition this on something more specific than "any gap".
    if SHORT_LAYOFF_MIN_WEEKS <= weeks <= SHORT_LAYOFF_MAX_WEEKS:
        inc += SHORT_LAYOFF_BUMP
    state.v_a = min(MAX_VAR, state.v_a + inc)
    state.v_b = min(MAX_VAR, state.v_b + inc)
    # Ornstein-Uhlenbeck mean-reversion toward 0 (the population average) - see
    # OU_THETA_PER_WEEK docstring. Decay factor, not a fixed decrement, so it
    # scales correctly regardless of how far the mean has drifted.
    decay = math.exp(-OU_THETA_PER_WEEK * weeks)
    state.m_a *= decay
    state.m_b *= decay
    # Same treatment for every surface-specific offset this player has ever
    # accumulated - variance re-inflates (capped at that surface's own prior,
    # never MAX_VAR: a surface offset should never become MORE uncertain than
    # the population baseline for that surface) and the mean decays toward 0.
    for surf in list(state.kappa_var):
        state.kappa_var[surf] = min(SURFACE_PRIOR_VAR[surf], state.kappa_var[surf] + inc)
        state.kappa[surf] = state.kappa.get(surf, 0.0) * decay
    for surf in list(state.lam_var):
        state.lam_var[surf] = min(SURFACE_PRIOR_VAR[surf], state.lam_var[surf] + inc)
        state.lam[surf] = state.lam.get(surf, 0.0) * decay
    state.last_seen = date


def run(df: pd.DataFrame) -> pd.DataFrame:
    players: dict[str, PlayerState] = {}

    def get(uid: str) -> PlayerState:
        s = players.get(uid)
        if s is None:
            s = PlayerState()
            players[uid] = s
        return s

    eval_rows = []
    n = len(df)
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False)):
        w, l = row.winner_uid, row.loser_uid
        s = row.surface
        date = row.date
        sv = get(w)
        sl = get(l)
        inflate(sv, date)
        inflate(sl, date)

        # ticket 18: kappa/lambda now live on PlayerState itself, so inflate()
        # above already applied the same reversion/re-inflation to them as mu -
        # required once their variance could shrink (see PlayerState docstring).
        k_w, vk_w = sv.get_kappa(s)
        k_l, vk_l = sl.get_kappa(s)
        lm_w, vlm_w = sv.get_lam(s)
        lm_l, vlm_l = sl.get_lam(s)

        is_eval = (date.year in EVAL_YEARS)
        if is_eval and row.has_points and row.level_file == "tour":
            e_w = sv.m_a + k_w                       # winner effective serve
            f_l = sl.m_b + lm_l                      # loser effective return
            e_l = sl.m_a + k_l
            f_w = sv.m_b + lm_w
            p1 = sigmoid(e_w - f_l)                  # P(winner wins own serve pt)
            p2 = sigmoid(e_l - f_w)                  # P(loser wins own serve pt)
            p_win_match = p_match(p1, p2, int(row.best_of))
            # saved for the Jensen's-gap check: does plugging in POINT
            # ESTIMATES of p1,p2 into the convex match-recursion systematically
            # overstate confidence relative to averaging over the posterior?
            # ticket 18: use the ACTUAL current (shrunk) surface variance, not
            # the fixed prior - this was the bug (Djokovic vs a qualifier
            # scored 70.3% because this line used to always add back the full,
            # never-shrinking SURFACE_PRIOR_VAR regardless of match history).
            v_e1 = sv.v_a + vk_w
            v_f1 = sl.v_b + vlm_l
            v_e2 = sl.v_a + vk_l
            v_f2 = sv.v_b + vlm_w
            eval_rows.append({
                "date": date, "tour": row.tour, "surface": s, "best_of": row.best_of,
                "winner_uid": w, "loser_uid": l,
                "winner_name": row.winner_name, "loser_name": row.loser_name,
                "p_winner_hier": p_win_match, "p1": p1, "p2": p2,
                "d1_mean": e_w - f_l, "d1_var": v_e1 + v_f1,
                "d2_mean": e_l - f_w, "d2_var": v_e2 + v_f2,
                # Ticket 21 diagnostic: same eval row without kappa/lambda
                # (surface-specific offset), mu-only, to test whether the
                # surface term is net signal or net noise on the margin (the
                # weak-disagreement-with-Elo bucket that came back WORSE than
                # a coin flip - see issue 20).
                "d1_mean_nosurf": sv.m_a - sl.m_b, "d1_var_nosurf": sv.v_a + sl.v_b,
                "d2_mean_nosurf": sl.m_a - sv.m_b, "d2_var_nosurf": sl.v_a + sv.v_b,
            })

        if row.has_points:
            # winner serving: (winner effective serve) vs (loser effective return)
            e_w = sv.m_a + k_w
            f_l = sl.m_b + lm_l
            v_e = sv.v_a + vk_w
            v_f = sl.v_b + vlm_l
            n1, y1 = float(row.w_svpt), float(row.w_won)
            new_ew, new_ve, new_fl, new_vf = adf_update(e_w, v_e, f_l, v_f, y1, n1)
            sv.m_a, sv.v_a, k_w, vk_w = apply_side_update(sv.m_a, sv.v_a, k_w, vk_w,
                                                          e_w, new_ew, new_ve)
            sl.m_b, sl.v_b, lm_l, vlm_l = apply_side_update(sl.m_b, sl.v_b, lm_l, vlm_l,
                                                            f_l, new_fl, new_vf)

            # loser serving: (loser effective serve) vs (winner effective return)
            e_l = sl.m_a + k_l
            f_w = sv.m_b + lm_w
            v_e2 = sl.v_a + vk_l
            v_f2 = sv.v_b + vlm_w
            n2, y2 = float(row.l_svpt), float(row.l_won)
            new_el, new_ve2, new_fw, new_vf2 = adf_update(e_l, v_e2, f_w, v_f2, y2, n2)
            sl.m_a, sl.v_a, k_l, vk_l = apply_side_update(sl.m_a, sl.v_a, k_l, vk_l,
                                                          e_l, new_el, new_ve2)
            sv.m_b, sv.v_b, lm_w, vlm_w = apply_side_update(sv.m_b, sv.v_b, lm_w, vlm_w,
                                                            f_w, new_fw, new_vf2)

            sv.kappa[s], sv.kappa_var[s] = k_w, vk_w
            sl.kappa[s], sl.kappa_var[s] = k_l, vk_l
            sv.lam[s], sv.lam_var[s] = lm_w, vlm_w
            sl.lam[s], sl.lam_var[s] = lm_l, vlm_l

            if DEBUG_UID and DEBUG_UID in (w, l):
                who = sv if w == DEBUG_UID else sl
                other = row.loser_name if w == DEBUG_UID else row.winner_name
                kh, vkh = who.get_kappa(s)
                log(f"  [DEBUG {DEBUG_UID}] {date.date()} vs {other} ({'W' if w==DEBUG_UID else 'L'}, {s}): "
                    f"m_a={who.m_a:+.3f} v_a={who.v_a:.4f} m_b={who.m_b:+.3f} v_b={who.v_b:.4f} "
                    f"kappa[{s}]={kh:+.3f} v={vkh:.4f}")
        else:
            # regime B: combined-strength update only (ticket 12: a single match
            # cannot identify serve/return separately), split 50/50 to alpha/beta
            cw = sv.m_a + sv.m_b
            cl = sl.m_a + sl.m_b
            vw = sv.v_a + sv.v_b
            vl = sl.v_a + sl.v_b
            tier_scale = REGIME_B_N if row.tourney_level in ELITE_LEVELS else REGIME_B_N * TIER_WEIGHT
            # Issue 20 round 4: tour-conditioned discount, WTA lower-tier
            # (qual/ITF) regime-B evidence only - see WTA_LOWER_WEIGHT
            # docstring above. Scoped to regime B only (same precedent as
            # TIER_WEIGHT itself), no-op by default.
            if row.tour == "wta" and row.level_file == "lower":
                tier_scale *= WTA_LOWER_WEIGHT
            # Ticket 21: the old fake y=n=tier_scale trial told the filter the
            # winner won "the whole match" with total certainty (n=n successes
            # out of n) - exactly the lie that made attempt #6 (ticket 20,
            # DSV_REGIME_B_N > 1.0) blow the corruption gate the moment it was
            # scaled up: more weight on a maximally-overconfident data point
            # only makes things worse. The score column (loaded, never used
            # here) carries the REAL split. Using actual games-won-per-set as
            # y/n treats each game as an iid Bernoulli(sigmoid(d)) trial, which
            # ignores serve-alternation correlation within a set (a real but
            # unmodelled overdispersion) - so tier_scale still applies as an
            # evidence-strength dial and must be re-scanned through the
            # corruption gate at its new meaning, not assumed safe at 1.0 just
            # because the old fake-evidence version was.
            # Ticket 21 full-DP: try the richer set-by-set-scoreline
            # likelihood first (only when enabled - see REGIME_B_FULL_DP
            # docstring). Falls back to the games-won-count approximation
            # below for any match parse_score_sets rejects (long-form
            # undecided deciders, malformed tokens) so a single archive
            # irregularity never breaks the whole fit, and identically to
            # the pre-existing behaviour whenever the flag is off.
            full_dp_sets = parse_score_sets(row.score) if REGIME_B_FULL_DP else None
            if full_dp_sets is not None:
                new_cw, new_vw, new_cl, new_vl = adf_update_full_dp(
                    cw, vw, cl, vl, full_dp_sets, weight=tier_scale)
            elif not math.isnan(row.total_games_w):
                y_games = row.total_games_w * tier_scale
                n_games = (row.total_games_w + row.total_games_l) * tier_scale
                new_cw, new_vw, new_cl, new_vl = adf_update(cw, vw, cl, vl, y_games, n_games)
            else:
                y_games = n_games = tier_scale   # unparseable score: old fallback
                new_cw, new_vw, new_cl, new_vl = adf_update(cw, vw, cl, vl, y_games, n_games)
            # Ticket 21 retested bug 8's shift-survival fix here (reverted in
            # ticket 20) now that y/n are real evidence, not the old fake
            # all-or-nothing trial - result was numerically IDENTICAL to no
            # fix at all (0.5903 both ways on the tuning window): real evidence
            # almost never hits MAX_SHIFT, so shift-survival scaling is a no-op
            # at this evidence strength. Not reapplied - dead weight, not a bug.
            dshift = max(-MAX_SHIFT, min(MAX_SHIFT, new_cw - cw))
            lshift = max(-MAX_SHIFT, min(MAX_SHIFT, new_cl - cl))
            sv.m_a = max(-MAX_STATE, min(MAX_STATE, sv.m_a + 0.5 * dshift))
            sv.m_b = max(-MAX_STATE, min(MAX_STATE, sv.m_b + 0.5 * dshift))
            sl.m_a = max(-MAX_STATE, min(MAX_STATE, sl.m_a + 0.5 * lshift))
            sl.m_b = max(-MAX_STATE, min(MAX_STATE, sl.m_b + 0.5 * lshift))
            redw = max(0.0, 1 - new_vw / vw) if vw > 0 else 0.0
            redl = max(0.0, 1 - new_vl / vl) if vl > 0 else 0.0
            sv.v_a = min(MAX_VAR, sv.v_a * (1 - 0.5 * redw))
            sv.v_b = min(MAX_VAR, sv.v_b * (1 - 0.5 * redw))
            sl.v_a = min(MAX_VAR, sl.v_a * (1 - 0.5 * redl))
            sl.v_b = min(MAX_VAR, sl.v_b * (1 - 0.5 * redl))

        sv.n_matches += 1
        sl.n_matches += 1

        if (i + 1) % 200_000 == 0:
            log(f"  ...{i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")

    log(f"fit complete: {n:,} matches in {time.time()-t0:.0f}s, "
        f"{len(eval_rows):,} eval predictions")
    return pd.DataFrame(eval_rows), players


if __name__ == "__main__":
    log("loading archive...")
    df, dropped = load_prepared(use_cache=os.environ.get("DSV_USE_CACHE", "1") != "0")

    log("\n=== COVERAGE ===")
    log(f"  rows read              : {dropped.pop('_total_in'):,}")
    kept = dropped.pop("_kept")
    for reason, n in dropped.items():
        log(f"  excluded, {reason:<28}: {n:,}")
    log(f"  kept                   : {kept:,}")
    regime_a = int(df["has_points"].sum())
    log(f"  regime A (point counts): {regime_a:,} ({100*regime_a/len(df):.1f}%)")
    log(f"  regime B (scoreline)   : {len(df)-regime_a:,}")
    regime_b_mask = ~df["has_points"]
    regime_b_parsed = int(df.loc[regime_b_mask, "total_games_w"].notna().sum())
    regime_b_total = int(regime_b_mask.sum())
    log(f"  regime B games-parsed  : {regime_b_parsed:,}/{regime_b_total:,} "
        f"({100*regime_b_parsed/max(regime_b_total,1):.1f}%) - ticket 21")
    if REGIME_B_FULL_DP:
        full_dp_ok = int(df.loc[regime_b_mask, "score"].astype(str)
                        .map(lambda s: parse_score_sets(s) is not None).sum())
        log(f"  regime B full-DP-eligible: {full_dp_ok:,}/{regime_b_total:,} "
            f"({100*full_dp_ok/max(regime_b_total,1):.1f}%) - ticket 21 full-DP "
            f"(ineligible matches fall back to games-won-count, unchanged)")

    preds, players = run(df)
    # Ticket 17: output filenames are env-overridable so sweep candidates never
    # clobber ticket 10's already-reported final artifacts (the un-suffixed
    # defaults). Every sweep run writes to its own named file.
    out_preds = os.environ.get("DSV_OUT_PREDS",
                               str(_common.DD / "hier_eval_predictions.parquet"))
    out_state = os.environ.get("DSV_OUT_STATE",
                               str(_common.DD / "hier_final_state.pkl"))
    preds.to_parquet(out_preds)
    log(f"wrote {out_preds}")

    # ---- persist final states for anchor checks / next script
    # ticket 18: kappa/lambda now flattened back to the old (uid, surface)->value
    # shape so hier_anchors.py's existing readout code keeps working unchanged.
    kappa_flat = {(uid, surf): val for uid, v in players.items() for surf, val in v.kappa.items()}
    lam_flat = {(uid, surf): val for uid, v in players.items() for surf, val in v.lam.items()}
    with open(out_state, "wb") as f:
        pickle.dump({"players": {k: (v.m_a, v.v_a, v.m_b, v.v_b, v.n_matches)
                                 for k, v in players.items()},
                    "kappa": kappa_flat, "lam": lam_flat}, f)
    log(f"wrote {out_state}")
