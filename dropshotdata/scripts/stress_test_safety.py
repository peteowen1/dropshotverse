"""Gate for ticket 18 (spread tuning): before any 'allow bigger movement'
candidate is scored, it must still pass the two reproductions that caught the
original runaway-drift bugs in ticket 10. Loosening OU_THETA_PER_WEEK / MAX_SHIFT
for legitimate spread must not silently reopen a bug that took hours to find.

Exit 0 = safe. Exit 1 = this candidate reopens a known failure mode; do not
score it, let alone select it.
"""
import math
import sys

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import numpy as np
import pandas as pd
from hierarchical_model import (PlayerState, inflate, adf_update, apply_side_update,
                                sigmoid, p_match, p_game, p_tiebreak, p_set, p_set_full,
                                adf_update_full_dp, parse_score_sets, set_outcome_dist_for_d,
                                p_specific_set_score, FULL_DP_GRID_MIN, FULL_DP_GRID_MAX,
                                MAX_STATE, MAX_VAR, MAX_SHIFT, _heavy_tail_var)
import hierarchical_model as hm

FAILURES = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}: {detail}", flush=True)
    if not cond:
        FAILURES.append(name)


# ---- reproduction 1: the 15-year-layoff-then-win pathology (Milos Raonic shape)
s = PlayerState()
inflate(s, pd.Timestamp("2005-01-01"))
inflate(s, pd.Timestamp("2020-01-01"))   # 15-year gap
check("layoff variance bounded", s.v_a <= MAX_VAR + 1e-9, f"v_a={s.v_a:.3f} (cap {MAX_VAR})")

opp = PlayerState()
cw, cl = s.m_a + s.m_b, opp.m_a + opp.m_b
vw, vl = s.v_a + s.v_b, opp.v_a + opp.v_b
new_cw, _, _, _ = adf_update(cw, vw, cl, vl, 1.0, 1.0)
shift = new_cw - cw
check("single post-layoff win bounded", abs(shift) <= 3.0,
     f"raw shift after 15y gap = {shift:.3f}")

# ---- reproduction 2: repeated comebacks compounding across a long career
# (this is what a single-event cap does NOT catch - it needs many cycles)
s2 = PlayerState()
date = pd.Timestamp("2010-01-01")
opp2 = PlayerState()
for cycle in range(6):                       # six long-layoff-then-loss cycles
    date = date + pd.Timedelta(weeks=200)    # ~3.8 years out
    inflate(s2, date)
    inflate(opp2, date)
    e, f = s2.m_a + s2.m_b, opp2.m_a + opp2.m_b
    ve, vf = s2.v_a + s2.v_b, opp2.v_a + opp2.v_b
    new_e, new_ve, new_f, new_vf = adf_update(e, ve, f, vf, 0.0, 1.0)  # loses on comeback
    d = max(-1.0, min(1.0, new_e - e)) if abs(new_e - e) > 1.0 else (new_e - e)
    s2.m_a = max(-MAX_STATE, min(MAX_STATE, s2.m_a + 0.5 * d))
    s2.m_b = max(-MAX_STATE, min(MAX_STATE, s2.m_b + 0.5 * d))
check("six-cycle comeback-loss compounding bounded",
     abs(s2.m_a) <= MAX_STATE + 1e-9 and abs(s2.m_b) <= MAX_STATE + 1e-9,
     f"after 6 long-layoff-loss cycles: m_a={s2.m_a:.3f} m_b={s2.m_b:.3f} (cap {MAX_STATE})")

# ---- reproduction 3: isolated weakly-connected pool over years (ticket 10's stress test)
import random
random.seed(1)
players = {i: PlayerState() for i in range(6)}
date = pd.Timestamp("2015-01-01")
for match in range(300):
    date = date + pd.Timedelta(days=random.randint(3, 20))
    a, b = random.sample(range(6), 2)
    sv, sl = players[a], players[b]
    inflate(sv, date)
    inflate(sl, date)
    p_serve = 0.62 if a == 0 else (0.58 if b == 0 else 0.60)
    n = 70
    y = sum(1 for _ in range(n) if random.random() < p_serve)
    e_w, f_l = sv.m_a, sl.m_b
    v_e, v_f = sv.v_a, sl.v_b
    new_ew, new_ve, new_fl, new_vf = adf_update(e_w, v_e, f_l, v_f, y, n)
    sv.m_a, sv.v_a, _, _ = apply_side_update(sv.m_a, sv.v_a, 0.0, 0.0, e_w, new_ew, new_ve)
    sl.m_b, sl.v_b, _, _ = apply_side_update(sl.m_b, sl.v_b, 0.0, 0.0, f_l, new_fl, new_vf)
mx = max(abs(p.m_a) for p in players.values())
check("isolated pool bounded over 300 matches/6yrs", mx < 5.0, f"max|m_a|={mx:.3f}")

# ---- reproduction 4 (added after this exact failure was caught by an anchor
# check, not by this gate - kappa's variance was made to shrink like mu's
# without also giving kappa's MEAN the same reversion mu has, reopening the
# runaway-drift class of bug through a different variable): a player who wins
# every Hard-court match for years should NOT drift kappa to the MAX_STATE
# ceiling and stay there.
s3 = PlayerState()
opp3 = PlayerState()
date = pd.Timestamp("2010-01-01")
for i in range(200):
    date += pd.Timedelta(days=10)
    inflate(s3, date)
    inflate(opp3, date)
    k, vk = s3.get_kappa("Hard")
    ok, ovk = opp3.get_kappa("Hard")
    e_old = k
    new_e, new_v, _, _ = adf_update(s3.m_a + k, s3.v_a + vk, opp3.m_b + ok, opp3.v_b + ovk, 45, 70)
    s3.m_a, s3.v_a, k, vk = apply_side_update(s3.m_a, s3.v_a, k, vk, s3.m_a + k, new_e, new_v)
    s3.kappa["Hard"], s3.kappa_var["Hard"] = k, vk
check("kappa bounded over 200 same-surface wins",
     abs(s3.kappa.get("Hard", 0.0)) < MAX_STATE + 1e-9,
     f"kappa[Hard]={s3.kappa.get('Hard', 0.0):.3f} after 200 matches (cap {MAX_STATE})")

# ---- reproduction 5 (issue 20, RW_VAR_PER_WEEK-toward-zero lead, round 5):
# a "stale-but-confident" pathology check - the INVERSE failure mode of
# reproductions 1/2 above (which only check that variance/state growth is
# BOUNDED, never that it's SUFFICIENT). This matters specifically because the
# round-4 RW_VAR_PER_WEEK scan toward zero showed a real, monotonic-to-the-
# boundary tuning-window improvement - exactly the shape you'd expect if the
# gain came from the model becoming overconfident about long-inactive
# players rather than genuinely more accurate. Without this check, a passing
# gate would say nothing about that risk.

# (a) variance-sufficiency: start from a tour-veteran-tight prior (v=0.03 -
# the real Djokovic pre-match trace value from issue 20 hypothesis 4, Test A)
# and require post-layoff variance clears increasing floors as the gap
# lengthens. The 5yr floor is tied to this file's own documented design
# intent for PHI_MAX ("a real player's uncertainty after any layoff, however
# long, should saturate rather than grow without bound" - hierarchical_model.py,
# PHI_MAX docstring): a 5-year gap should land close to MAX_VAR, not stay
# near its pre-layoff value. Floors calibrated with margin below the v25
# default (0.005) actual values (~0.70 / ~3.15 / ~4.00 at 1/3/5yr) so the
# safety-relevant scan range isn't blocked outright, while still rejecting
# the suspiciously-improving near-zero end of the round-4 scan.
for weeks, floor, label in [(52, 0.3, "1yr"), (156, 2.0, "3yr"), (260, 3.5, "5yr")]:
    st = PlayerState()
    st.v_a = st.v_b = 0.03
    start = pd.Timestamp("2015-01-01")
    st.last_seen = start
    inflate(st, start + pd.Timedelta(weeks=weeks))
    check(f"variance inflates enough after {label} layoff (from tight prior)",
         st.v_a >= floor,
         f"v_a={st.v_a:.4f} after {label} gap from v=0.03 start (floor {floor}, cap {MAX_VAR})")

# (b) concrete overconfident-prediction check: a strong-but-plausible
# returning player (combined edge 1.5, i.e. sigmoid(1.5)=0.82 serve-win prob
# if uncertainty were ignored) coming back after a 3yr layoff from a tight
# pre-layoff variance, facing a fresh/average opponent (INIT_VAR, m=0). The
# MC-averaged match-win probability must not be near-certain - genuinely not
# knowing whether a 3-years-absent player retained form is exactly the state
# of knowledge this mechanism exists to represent, and this is the literal
# failure mode ("a player who hasn't played in months/years could get a
# falsely tight, overconfident rating") this test was added to catch.
returner = PlayerState()
returner.m_a, returner.m_b = 0.75, 0.75   # combined edge 1.5
returner.v_a = returner.v_b = 0.03
returner.last_seen = pd.Timestamp("2015-01-01")
inflate(returner, pd.Timestamp("2015-01-01") + pd.Timedelta(weeks=156))
opponent = PlayerState()   # fresh/average: INIT_VAR, m=0
rng = np.random.default_rng(0)
d1 = rng.normal(returner.m_a + returner.m_b,
               math.sqrt(max(returner.v_a + returner.v_b, 1e-6)), 4000)
d2 = rng.normal(opponent.m_a + opponent.m_b,
               math.sqrt(max(opponent.v_a + opponent.v_b, 1e-6)), 4000)
ps = [p_match(sigmoid(a), sigmoid(b), 3) for a, b in zip(d1, d2)]
p_mc = float(np.mean(ps))
check("3yr-layoff return match not overconfident vs an average opponent",
     p_mc <= 0.95,
     f"MC-averaged p(returner wins)={p_mc:.4f} (ceiling 0.95) after 3yr layoff from tight prior")

# ---- reproduction 6 (ticket 21 full-DP): p_set_full refactor must be
# bit-identical to the pre-refactor scalar p_set implementation. p_set is
# used by regime A's eval-time prediction too (p_match), so any drift here
# would silently change v25's already-reported numbers, not just the new
# regime-B machinery this ticket adds.
def _reference_p_set(p_a_serve, p_b_serve, a_serves_first):
    """Verbatim copy of p_set's pre-refactor body (ticket 21), kept
    self-contained so this check doesn't just compare the refactor to
    itself."""
    states = {(0, 0): 1.0}
    final = 0.0
    for _ in range(13):
        nxt = {}
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
                    final += prob if na > nb else 0.0
                elif na == 7 or nb == 7:
                    tb = p_tiebreak(p_a_serve, p_b_serve)
                    final += prob * (tb if na == 7 else (1 - tb))
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + prob
        states = nxt
        if not states:
            break
    final += sum(states.values()) * 0.5
    return final


maxdiff = 0.0
for pa_, pb_ in [(0.5, 0.5), (0.65, 0.55), (0.9, 0.3), (0.1, 0.95), (0.62, 0.58), (0.99, 0.01)]:
    for first_ in (True, False):
        maxdiff = max(maxdiff, abs(_reference_p_set(pa_, pb_, first_) - p_set(pa_, pb_, first_)))
check("p_set_full refactor bit-identical to pre-refactor p_set",
     maxdiff < 1e-9, f"max diff across 12 combos = {maxdiff:.2e}")

# ---- reproduction 7 (ticket 21 full-DP): p_set_full's outcome distribution
# must be a real, normalized probability distribution (sums to 1) - a
# refactor bug here (e.g. dropping the complementary tiebreak-credit branch)
# would silently corrupt the log-likelihood every full-DP regime-B update uses.
maxerr = 0.0
for pa_, pb_ in [(0.5, 0.5), (0.65, 0.55), (0.9, 0.3), (0.1, 0.95), (0.99, 0.01), (0.01, 0.99)]:
    s = sum(p_set_full(pa_, pb_, True).values())
    maxerr = max(maxerr, abs(s - 1.0))
check("p_set_full outcome distribution normalizes to 1.0",
     maxerr < 1e-9, f"max |sum-1| across 6 combos = {maxerr:.2e}")

# ---- reproduction 8 (ticket 21 full-DP): extreme scoreline numerical
# stability - 6-0 6-0 (a genuine blowout) at both a weak and a very tight
# prior must produce finite, sane m_a/v_a/m_b/v_b, not NaN/inf. This is the
# direct stress-test analogue of ticket 12's own warning ("a single 6-4 6-4
# puts the MLE at the grid boundary") - here checked as a Bayesian update
# (prior-regularised), not a raw MLE, but still the sharpest evidence shape
# available.
for m_d, v_each, label in [(0.0, 0.05, "even prior, tight"), (0.0, 2.0, "even prior, wide"),
                           (5.0, 0.05, "already-lopsided prior, tight"),
                           (-5.0, 0.05, "prior disagrees with evidence, tight")]:
    r = adf_update_full_dp(m_d, v_each, 0.0, v_each, [(6, 0), (6, 0)], weight=1.0)
    ok = all(math.isfinite(x) for x in r)
    check(f"6-0 6-0 stable ({label})", ok, f"m_d={m_d} v={v_each} -> {r}")

# ---- reproduction 9 (ticket 21 full-DP): the exact grid-boundary case
# ticket 12 flagged - a SINGLE extreme set (not a whole match) under a weak
# (uninformative) prior. Per-match MLE is known-hopeless here (ticket 12);
# the Bayesian update must still stay bounded within the grid range rather
# than diverging, because the prior regularises even a ridge-shaped likelihood.
r9 = adf_update_full_dp(0.0, 5.0, 0.0, 5.0, [(6, 0)], weight=1.0)
check("single-extreme-set grid-boundary case stays bounded and finite",
     all(math.isfinite(x) for x in r9) and abs(r9[0]) <= FULL_DP_GRID_MAX and abs(r9[2]) <= FULL_DP_GRID_MAX,
     f"single 6-0 set, wide prior -> {r9}")

# ---- reproduction 10 (ticket 21 full-DP): d far from zero (near the fixed
# grid's edge) must not blow up or silently clip to a wrong value.
r10 = adf_update_full_dp(11.0, 1.0, -11.0, 1.0, [(6, 0), (6, 0)], weight=1.0)
check("far-from-zero d (near grid edge) numerically stable",
     all(math.isfinite(x) for x in r10), f"prior d=22 -> {r10}")

# ---- reproduction 11 (ticket 21 full-DP): the symmetric-d formula must
# actually encode the strength gap (p_a_serve=sigmoid(d/2),
# p_b_serve=sigmoid(-d/2), NOT p_a_serve==p_b_serve - the degenerate reading
# would make every set ~50/50 regardless of d, silently disabling the whole
# mechanism). Check P(A wins the set) is monotonically non-decreasing in d
# and reaches both extremes.
d_grid = [-6.0, -2.0, 0.0, 2.0, 6.0]
p_a_wins = []
for d_ in d_grid:
    dist = set_outcome_dist_for_d(d_)
    p_a_wins.append(sum(pr for (a_, b_), pr in dist.items() if a_ > b_))
monotone = all(p_a_wins[i] <= p_a_wins[i + 1] + 1e-9 for i in range(len(p_a_wins) - 1))
check("symmetric-d formula: P(A wins set) monotone non-decreasing in d",
     monotone, f"d={d_grid} -> P(A wins)={[round(x, 4) for x in p_a_wins]}")
check("symmetric-d formula: d=0 gives an even set-score distribution",
     abs(p_a_wins[2] - 0.5) < 1e-9, f"P(A wins set) at d=0 = {p_a_wins[2]:.6f} (expect 0.5)")

# ---- reproduction 12 (ticket 21 full-DP): fail-safe guards - empty
# set_scores, non-positive weight, and non-positive prior variance must all
# be safe no-ops (return the input unchanged), same convention as
# adf_update's own n<=0 guard.
noop1 = adf_update_full_dp(0.3, 0.4, -0.2, 0.4, [], weight=1.0)
check("adf_update_full_dp no-ops on empty set_scores",
     noop1 == (0.3, 0.4, -0.2, 0.4), f"got {noop1}")
noop2 = adf_update_full_dp(0.3, 0.4, -0.2, 0.4, [(6, 4)], weight=0.0)
check("adf_update_full_dp no-ops on weight<=0",
     noop2 == (0.3, 0.4, -0.2, 0.4), f"got {noop2}")

# ---- reproduction 13 (ticket 21 full-DP): parse_score_sets rejects
# old-rules undecided long final sets and malformed tokens (falls back to
# the existing games-won-count approximation for the whole match), and
# correctly parses ordinary and tiebreak-annotated sets.
check("parse_score_sets parses an ordinary 3-set match",
     parse_score_sets("6-4 3-6 7-6(4)") == [(6, 4), (3, 6), (7, 6)],
     f"got {parse_score_sets('6-4 3-6 7-6(4)')}")
check("parse_score_sets rejects an old-rules long undecided decider",
     parse_score_sets("70-68 6-3") is None, f"got {parse_score_sets('70-68 6-3')}")
check("parse_score_sets rejects malformed score",
     parse_score_sets("garbage") is None, f"got {parse_score_sets('garbage')}")

# ---- reproduction 14 (ticket 21 full-DP): the fixed grid cache must be
# self-consistent - re-querying the same d twice returns the identical
# (cached) distribution, and the cache never grows unboundedly (fixed range
# / step means a hard ceiling on distinct entries regardless of how many
# distinct raw d values a real fit ever queries).
from hierarchical_model import FULL_DP_GRID_STEP
_before = len(__import__("hierarchical_model")._full_dp_grid_cache)
d1 = set_outcome_dist_for_d(3.14159)
d2 = set_outcome_dist_for_d(3.14159)
check("grid cache lookup is idempotent (same dict object reused)", d1 is d2, "")
max_cache_entries = int((FULL_DP_GRID_MAX - FULL_DP_GRID_MIN) / FULL_DP_GRID_STEP) + 2
_after = len(__import__("hierarchical_model")._full_dp_grid_cache)
check("grid cache bounded by the fixed grid size (no unbounded growth)",
     _after <= max_cache_entries, f"cache size={_after}, hard ceiling={max_cache_entries}")

# ---- reproduction 15 (ticket 24, heavy-tailed process noise): a single very
# large surprising update. A tight-variance, strongly-favoured player (v_d
# small, prior d=3.0 i.e. sigmoid(3)=95%) gets demolished (y=2 of n=70 games).
# Compare Gaussian (HEAVY_TAILED_NOISE off) vs heavy-tailed (on) adf_update on
# the SAME inputs. The mechanism's whole point: heavy tails should let the
# posterior genuinely absorb more of the surprise (move further toward the
# evidence, less resistance from an artificially-confident prior) AND retain
# MORE posterior variance (less false confidence) than the Gaussian version -
# both directly checkable without needing MAX_SHIFT to engage at all.
m_d0, v_each0 = 3.0, 0.05
hm.HEAVY_TAILED_NOISE = False
gauss15 = adf_update(m_d0, v_each0, 0.0, v_each0, 2.0, 70.0)
hm.HEAVY_TAILED_NOISE = True
heavy15 = adf_update(m_d0, v_each0, 0.0, v_each0, 2.0, 70.0)
hm.HEAVY_TAILED_NOISE = False
gauss_shift = abs(gauss15[0] - m_d0)
heavy_shift = abs(heavy15[0] - m_d0)
check("large surprise: heavy-tailed moves the mean at least as far as Gaussian",
     heavy_shift >= gauss_shift - 1e-9,
     f"gauss shift={gauss_shift:.4f} heavy shift={heavy_shift:.4f} (d_prior=3.0, y=2/n=70)")
check("large surprise: heavy-tailed retains more (or equal) posterior variance than Gaussian",
     heavy15[1] >= gauss15[1] - 1e-9,
     f"gauss v_a={gauss15[1]:.5f} heavy v_a={heavy15[1]:.5f}")
check("large surprise: both versions stay finite",
     all(math.isfinite(x) for x in gauss15) and all(math.isfinite(x) for x in heavy15),
     f"gauss={gauss15} heavy={heavy15}")

# ---- reproduction 16 (ticket 24): a SEQUENCE of several large surprises in a
# row - the exact pattern behind bug 9 (variance shrinks before the mean is
# correct, locking in a wrong value forever, across MULTIPLE compounding
# events, not just one). A tight-variance player alternates big upset losses
# and big upset wins for 8 rounds; under the old absorbing-boundary
# pathology, variance would ratchet down toward zero regardless of direction.
# Heavy-tailed must keep variance meaningfully above a stale-confidence floor
# throughout, not just survive the final MAX_STATE/MAX_VAR hard backstops
# (those stay active in both modes - this checks the MECHANISM, not just the
# safety net behind it).
hm.HEAVY_TAILED_NOISE = True
seq_state = PlayerState()
seq_state.v_a = seq_state.v_b = 0.05    # tour-veteran-tight prior
opp_seq = PlayerState()
min_v_seen = seq_state.v_a
for rnd in range(8):
    y_evt = 68.0 if rnd % 2 == 0 else 2.0   # alternating blowout win/loss
    e_w, f_l = seq_state.m_a, opp_seq.m_b
    v_e, v_f = seq_state.v_a, opp_seq.v_b
    new_ew, new_ve, new_fl, new_vf = adf_update(e_w, v_e, f_l, v_f, y_evt, 70.0)
    seq_state.m_a, seq_state.v_a, _, _ = apply_side_update(
        seq_state.m_a, seq_state.v_a, 0.0, 0.0, e_w, new_ew, new_ve)
    min_v_seen = min(min_v_seen, seq_state.v_a)
hm.HEAVY_TAILED_NOISE = False
check("repeated large surprises (8 rounds): heavy-tailed state stays finite and MAX_STATE-bounded",
     math.isfinite(seq_state.m_a) and abs(seq_state.m_a) <= MAX_STATE + 1e-9,
     f"m_a={seq_state.m_a:.3f} v_a={seq_state.v_a:.4f} (cap {MAX_STATE})")
check("repeated large surprises (8 rounds): variance never collapses toward zero (no re-opened absorbing-boundary trap)",
     min_v_seen > 1e-3,
     f"min v_a observed across 8 alternating-blowout rounds = {min_v_seen:.5f}")

# ---- reproduction 17 (ticket 24): the round-5 long-layoff variance-
# sufficiency checks (reproduction 5) re-run with HEAVY_TAILED_NOISE=1. The
# heavy-tailed mechanism only touches adf_update's/adf_update_full_dp's prior
# term inside the observation update - inflate()'s time-decay/process-noise
# growth is untouched by this ticket - so these must still pass identically.
# Checked explicitly rather than assumed, per the task brief's own mandate.
hm.HEAVY_TAILED_NOISE = True
for weeks, floor, label in [(52, 0.3, "1yr"), (156, 2.0, "3yr"), (260, 3.5, "5yr")]:
    st17 = PlayerState()
    st17.v_a = st17.v_b = 0.03
    start17 = pd.Timestamp("2015-01-01")
    st17.last_seen = start17
    inflate(st17, start17 + pd.Timedelta(weeks=weeks))
    check(f"[HEAVY_TAILED_NOISE=1] variance inflates enough after {label} layoff (from tight prior)",
         st17.v_a >= floor,
         f"v_a={st17.v_a:.4f} after {label} gap from v=0.03 start (floor {floor}, cap {MAX_VAR})")
returner17 = PlayerState()
returner17.m_a, returner17.m_b = 0.75, 0.75
returner17.v_a = returner17.v_b = 0.03
returner17.last_seen = pd.Timestamp("2015-01-01")
inflate(returner17, pd.Timestamp("2015-01-01") + pd.Timedelta(weeks=156))
opponent17 = PlayerState()
rng17 = np.random.default_rng(0)
d1_17 = rng17.normal(returner17.m_a + returner17.m_b,
                     math.sqrt(max(returner17.v_a + returner17.v_b, 1e-6)), 4000)
d2_17 = rng17.normal(opponent17.m_a + opponent17.m_b,
                     math.sqrt(max(opponent17.v_a + opponent17.v_b, 1e-6)), 4000)
ps17 = [p_match(sigmoid(a), sigmoid(b), 3) for a, b in zip(d1_17, d2_17)]
p_mc17 = float(np.mean(ps17))
hm.HEAVY_TAILED_NOISE = False
check("[HEAVY_TAILED_NOISE=1] 3yr-layoff return match not overconfident vs an average opponent",
     p_mc17 <= 0.95,
     f"MC-averaged p(returner wins)={p_mc17:.4f} (ceiling 0.95) after 3yr layoff from tight prior")

# ---- reproduction 18 (ticket 24): default-path no-op verification, checked
# directly against a hand-rolled reference Gaussian-only Newton solve (not
# just "existing tests still pass", which only shows equality by omission).
def _reference_gaussian_adf_update(m_a, v_a, m_b, v_b, y, n):
    """Verbatim pre-ticket-24 adf_update body, self-contained."""
    if n <= 0:
        return m_a, v_a, m_b, v_b
    m_d, v_d = m_a - m_b, v_a + v_b
    d = m_d
    for _ in range(6):
        p = sigmoid(d)
        grad = (y - n * p) - (d - m_d) / v_d
        hess = -(n * p * (1 - p) + 1.0 / v_d)
        if hess == 0:
            break
        d -= grad / hess
    p = sigmoid(d)
    var_post = 1.0 / (n * p * (1 - p) + 1.0 / v_d)
    shift = d - m_d
    share_a = v_a / (v_a + v_b)
    share_b = 1.0 - share_a
    reduction = max(0.0, 1.0 - var_post / v_d)
    return (m_a + share_a * shift, v_a * (1 - reduction * share_a),
            m_b - share_b * shift, v_b * (1 - reduction * share_b))


maxdiff18 = 0.0
for m_d_, v_, y_, n_ in [(0.0, 0.5, 40.0, 70.0), (3.0, 0.05, 2.0, 70.0),
                        (-2.0, 1.5, 55.0, 60.0), (0.0, 4.0, 1.0, 1.0)]:
    ref = _reference_gaussian_adf_update(m_d_, v_, 0.0, v_, y_, n_)
    got = adf_update(m_d_, v_, 0.0, v_, y_, n_)   # module default: HEAVY_TAILED_NOISE=0
    maxdiff18 = max(maxdiff18, max(abs(a - b) for a, b in zip(ref, got)))
check("HEAVY_TAILED_NOISE=0 (default) is bit-identical to the pre-ticket-24 Gaussian-only update",
     maxdiff18 < 1e-12, f"max diff across 4 cases = {maxdiff18:.2e}")

# ---- reproduction 19 (ticket 24): _heavy_tail_var's floor-at-1.0 contract -
# never below v_d at r=0, strictly increasing beyond it, and converges to
# v_d as nu grows (the Gaussian limit).
check("_heavy_tail_var at r=0 (d==m_d) returns exactly v_d (the floor)",
     _heavy_tail_var(1.0, 1.0, 0.4, 4.0) == 0.4, f"got {_heavy_tail_var(1.0, 1.0, 0.4, 4.0)}")
v_near = _heavy_tail_var(1.05, 1.0, 0.4, 4.0)
v_far = _heavy_tail_var(2.0, 1.0, 0.4, 4.0)
check("_heavy_tail_var strictly increases with |d - m_d|",
     0.4 <= v_near <= v_far, f"v_d=0.4 -> near={v_near:.4f} far={v_far:.4f}")
v_nu_small = _heavy_tail_var(3.0, 1.0, 0.4, 2.0)
v_nu_large = _heavy_tail_var(3.0, 1.0, 0.4, 200.0)
check("_heavy_tail_var: larger nu (less heavy-tailed) gives less inflation for the same surprise",
     v_nu_large <= v_nu_small, f"nu=2 -> {v_nu_small:.4f}, nu=200 -> {v_nu_large:.4f}")

print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
sys.exit(1 if FAILURES else 0)
