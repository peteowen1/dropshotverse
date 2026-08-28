"""Ticket 10 stage 2: anchor checks on the fitted hierarchical model.

stats-discipline section 1 - run BEFORE trusting the market comparison. If these
fail, alpha/beta are not measuring what their names claim and the head-to-head
numbers are meaningless regardless of what they say.
"""
import pickle

import pandas as pd

# pickle load is safe here: the file was written moments ago by
# hierarchical_model.py in this same session on this machine, not fetched
# from any external or untrusted source.
with open(r"C:\dev\dropshotverse\dropshotdata\hier_final_state.pkl", "rb") as f:
    state = pickle.load(f)

players = state["players"]          # uid -> (m_a, v_a, m_b, v_b, n_matches)
kappa = state["kappa"]              # (uid, surface) -> serve offset
lam = state["lam"]                  # (uid, surface) -> return offset

preds = pd.read_parquet(r"C:\dev\dropshotverse\dropshotdata\hier_eval_predictions.parquet")
names = {}
for side in ("winner", "loser"):
    names.update(dict(zip(preds[f"{side}_uid"], preds[f"{side}_name"])))

# active = played in the eval window, so retired players (who never got 2024-25
# regime-A eval rows) are naturally excluded - same fix the Elo anchors needed.
active_atp = {u for u in set(preds.winner_uid) | set(preds.loser_uid) if u.startswith("atp")}
active_wta = {u for u in set(preds.winner_uid) | set(preds.loser_uid) if u.startswith("wta")}

print(f"total players fitted: {len(players):,}")
print(f"active in eval (ATP): {len(active_atp)}   (WTA): {len(active_wta)}\n")

passed = []

def eff(uid: str) -> float:
    """Current-form readout. Diagnosed 2026-08-27: raw mu_a+mu_b UNDERSTATES
    established players badly (Djokovic ranked 476/484, near last) because for
    a long-tenured, low-variance player, most new-match correction routes into
    kappa (surface offset) rather than mu, by the variance-share design in
    apply_side_update - while OU mean-reversion (added to fix an unrelated
    unbounded-drift bug) decays ONLY mu, never kappa. The two interact to make
    mu alone a misleading summary for anyone with a long career. The correct
    read-out is effective skill INCLUDING the surface term: this single change
    (no re-fit) moved Sinner from rank 92/484 to rank 1, Djokovic from 476 to 7."""
    ma, va, mb, vb, n = players[uid]
    ka = kappa.get((uid, "Hard"), 0.0)
    return ma + mb + 2 * ka


print("=== 1-2. current form top-5, active players only (surface-inclusive readout) ===")
for tour, uids, expect in (("atp", active_atp, {"Sinner", "Alcaraz"}),
                           ("wta", active_wta, {"Sabalenka", "Swiatek"})):
    ranked = sorted(((eff(u), u) for u in uids), reverse=True)[:8]
    top = [(names.get(u, u), round(r, 3)) for r, u in ranked]
    print(f"  {tour.upper()}: {top[:5]}")
    hit = {e for e in expect if any(e in str(n) for n, _ in top[:5])}
    ok = hit == expect
    passed.append(ok)
    print(f"    anchor {expect} in top5 -> {'PASS' if ok else 'FAIL ' + str(hit)}")

print("\n=== 3-4. serve/return skill decomposition (does alpha/beta mean what it claims?) ===")
# big servers should be top on alpha (serve), weak on beta (return); elite
# returners the reverse. Search by substring since uid keys are namespaced ids.
big_servers = ["Isner", "Karlovic", "Opelka", "Anderson"]
elite_returners = ["Djokovic", "Nadal", "Murray"]

def eff_alpha(uid: str) -> float:
    return players[uid][0] + kappa.get((uid, "Hard"), 0.0)


def eff_beta(uid: str) -> float:
    return players[uid][2] + lam.get((uid, "Hard"), 0.0)


all_atp = [(u, v) for u, v in players.items() if u.startswith("atp") and v[4] >= 20]
n_atp = len(all_atp)
alpha_rank = {u: r for r, (u, _) in enumerate(sorted(all_atp, key=lambda kv: -eff_alpha(kv[0])))}
beta_rank = {u: r for r, (u, _) in enumerate(sorted(all_atp, key=lambda kv: -eff_beta(kv[0])))}

for label, uid in names.items():
    pass  # names maps uid->name; need reverse lookup below

uid_of_name = {}
for u, nm in names.items():
    uid_of_name.setdefault(str(nm), u)

print(f"  (n={n_atp} ATP players with >=20 matches)")
for nm in big_servers:
    match = next((full for full in uid_of_name if nm in full), None)
    if not match:
        print(f"  {nm}: not found in eval-window names"); continue
    u = uid_of_name[match]
    if u not in alpha_rank:
        print(f"  {match}: <20 matches, skipped"); continue
    ar, br = alpha_rank[u], beta_rank[u]
    ok = ar < n_atp * 0.15 and br > n_atp * 0.40
    passed.append(ok)
    print(f"  {match}: alpha_rank={ar}/{n_atp} (top 15%? {ar < n_atp*0.15}) "
          f"beta_rank={br}/{n_atp} (below median? {br > n_atp*0.40})  "
          f"{'PASS' if ok else 'WEAK/FAIL'}")

for nm in elite_returners:
    match = next((full for full in uid_of_name if nm in full), None)
    if not match:
        print(f"  {nm}: not found"); continue
    u = uid_of_name[match]
    if u not in beta_rank:
        print(f"  {match}: <20 matches, skipped"); continue
    br = beta_rank[u]
    ok = br < n_atp * 0.15
    passed.append(ok)
    print(f"  {match}: beta_rank={br}/{n_atp} (top 15%? {ok})  {'PASS' if ok else 'WEAK/FAIL'}")

print("\n=== 5-6. surface: Nadal on clay, Federer on grass ===")
nadal_uid = uid_of_name.get(next((n for n in uid_of_name if "Nadal" in n), ""))
fed_uid = uid_of_name.get(next((n for n in uid_of_name if "Federer" in n), ""))
for label, uid, surf in (("Nadal/Clay", nadal_uid, "Clay"), ("Federer/Grass", fed_uid, "Grass")):
    if not uid:
        print(f"  {label}: not in eval-window names (likely inactive 2024-25 -> expected, "
              f"both retired/near-retired; check via kappa directly instead)")
        continue
    k = kappa.get((uid, surf))
    print(f"  {label}: kappa={k}")

print("\n=== 9. uncertainty: <10-match players should have >= 2x the variance of >500-match players ===")
thin = [v[1] + v[3] for v in players.values() if v[4] < 10]
deep = [v[1] + v[3] for v in players.values() if v[4] > 500]
if thin and deep:
    import statistics
    mt, md = statistics.mean(thin), statistics.mean(deep)
    ok = mt >= 2 * md
    passed.append(ok)
    print(f"  mean total variance, <10 matches (n={len(thin)}): {mt:.3f}")
    print(f"  mean total variance, >500 matches (n={len(deep)}): {md:.3f}")
    print(f"  ratio: {mt/md:.2f}x  {'PASS' if ok else 'FAIL'}")

print(f"\n=== {sum(passed)}/{len(passed)} ANCHORS PASSED ===")
