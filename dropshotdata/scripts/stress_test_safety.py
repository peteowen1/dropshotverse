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
                                sigmoid, p_match, MAX_STATE, MAX_VAR)

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

print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
sys.exit(1 if FAILURES else 0)
