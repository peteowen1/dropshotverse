"""Ticket 26 (see issues/): UTS-style (tennis-crystal-ball) K-factor,
implemented and tested as ONE bundled alternative K-philosophy against
elo_baseline.py's current best (Sackmann-style matches-played K, ordering
fix + decay=0.3), on the exact same 2024-25 market-odds-joined holdout used
throughout this session.

Bundle, not an ablation: this tests "UTS's whole K stack" (tournament
context: level/round/best-of, + current-rating kFunction, + the reference's
own DELTA_RATING_CAP=200) versus our whole "matches-played decay" K
philosophy, per Pete's actual ask ("test how UTS elo compares to ours") -
not an isolated single-factor test. `blend` (surface mixing) is held fixed
at 0.0 throughout (the pre-existing blend grid's winner) and untouched.

`decay` (layoff/inactivity pull toward START) is a SEPARATE, orthogonal
mechanism from UTS's own K-function - UTS's rating-based kFunction addresses
"how fast should THIS rating level move", not "how long has this player been
out". Tested BOTH ways for honesty:
  - decay=0.3 (kept, matching current best) - the fairer single bundle test,
    since it isolates the K-mechanism change to the K-mechanism alone.
  - decay=0.0 (off) - a sensitivity check, since UTS's own kFunction being
    ~10x near START could plausibly interact with a returning player's
    rating being pulled back toward START by decay (both mechanisms push in
    a similar direction for a fresh-off-layoff player), and it's worth
    knowing whether that interaction matters here.

Step 0 (correctness gate, done FIRST): re-derive the k_mode='sackmann'
(default) arm through run_elo() post-refactor and confirm it reproduces the
already-committed 0.6120 logloss number - the run_elo() refactor that added
k_mode must not have silently changed the pre-existing Sackmann-K path.
"""
import math

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")   # archived 2026-09-06: elo_baseline / train_best_model stayed in scripts/
from elo_baseline import (
    load_matches, prepare, run_elo, score, EVAL_YEARS, START,
    uts_kfunction,
)
from tune_layoff_decay import market_join, BLEND

DECAY_BEST = 0.3   # current best config's decay (RESUME-beat-elo.md)


def market_logloss(fitted: pd.DataFrame, ev: pd.Series) -> dict:
    preds = fitted[ev].copy()
    j = market_join(preds)
    pr = np.clip(j["p_elo"].values, 1e-9, 1 - 1e-9)
    row = {"n": len(j), "acc": float((pr > 0.5).mean()),
           "logloss": float(-np.log(pr).mean()), "brier": float(((1 - pr) ** 2).mean())}
    for tour in ("atp", "wta"):
        sub = j[j["tour"] == tour]
        pr_t = np.clip(sub["p_elo"].values, 1e-9, 1 - 1e-9)
        row[f"{tour}_n"] = len(sub)
        row[f"{tour}_logloss"] = float(-np.log(pr_t).mean())
    return row


def anchor_check(label: str, fitted: pd.DataFrame, state: dict) -> bool:
    ov, n_all = state["overall"], state["n_all"]
    names: dict[str, str] = {}
    for side in ("winner", "loser"):
        names.update(dict(zip(fitted[f"{side}_uid"], fitted[f"{side}_name"])))
    recent = fitted[fitted["date"].dt.year >= 2024]
    active = pd.concat([recent["winner_uid"], recent["loser_uid"]]).value_counts()
    active = set(active[active >= 15].index)
    print(f"\n=== ANCHOR CHECKS ({label}) ===")
    passed = []
    for tour, expect in (("atp", {"Sinner", "Alcaraz"}), ("wta", {"Sabalenka", "Swiatek"})):
        ranked = sorted(((r, u) for u, r in ov.items() if u.startswith(tour) and u in active),
                        reverse=True)[:10]
        top = [(names.get(u, u), round(r)) for r, u in ranked]
        print(f"  {tour.upper()} top 5 active: {top[:5]}")
        hit = {e for e in expect if any(e in str(n) for n, _ in top[:5])}
        ok = hit == expect
        passed.append(ok)
        print(f"    anchor: {expect} in top 5? -> {hit or 'NONE'} {'PASS' if ok else 'FAIL'}")
    thin = [names.get(u, u) for u, r in sorted(ov.items(), key=lambda kv: -kv[1])[:20]
            if n_all.get(u, 0) < 10]
    passed.append(not thin)
    print(f"  anchor: no <10-match player in top 20 -> {'PASS' if not thin else 'FAIL ' + str(thin)}")
    ok_all = all(passed)
    print(f"  ANCHORS ({label}): {sum(passed)}/{len(passed)} passed -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    print("loading + preparing archive...", flush=True)
    raw = load_matches()
    df, dropped = prepare(raw)
    ev = df["date"].dt.year.isin(EVAL_YEARS)

    print(f"\n=== STEP 0: correctness gate - refactored run_elo(), k_mode='sackmann' default ===")
    fit_sack, state_sack = run_elo(df, BLEND, DECAY_BEST)  # k_mode default = 'sackmann'
    row_sack = market_logloss(fit_sack, ev)
    print(f"  sackmann (blend={BLEND}, decay={DECAY_BEST}): n={row_sack['n']} "
          f"logloss={row_sack['logloss']:.4f} (expect ~0.6120, per RESUME-beat-elo.md)")
    if abs(row_sack["logloss"] - 0.6120) > 0.001:
        print("  WARNING: refactor may have changed sackmann-mode behavior - investigate before trusting UTS numbers")
    else:
        print("  OK: matches the previously-committed number - refactor did not change the sackmann path")

    print(f"\n=== STEP 1: sanity-check uts_kfunction() range on REAL archive ratings ===")
    ratings = np.array(list(state_sack["overall"].values()))
    print(f"  overall ratings from the sackmann fit: n={len(ratings)} "
          f"min={ratings.min():.1f} p1={np.percentile(ratings,1):.1f} "
          f"p50={np.percentile(ratings,50):.1f} p99={np.percentile(ratings,99):.1f} "
          f"max={ratings.max():.1f} (START={START})")
    kf = np.array([uts_kfunction(r) for r in ratings])
    print(f"  uts_kfunction() over these ratings: min={kf.min():.3f} p1={np.percentile(kf,1):.3f} "
          f"p50={np.percentile(kf,50):.3f} p99={np.percentile(kf,99):.3f} max={kf.max():.3f}")
    # reference's own documented checkpoints, verified against our formula directly
    checkpoints = [1500, 1600, 1800, 2000, 2200]
    print("  reference checkpoints (expect ~10x/5.5x/1.64x/1.07x/1.008x):")
    for r in checkpoints:
        print(f"    rating={r}: kFunction={uts_kfunction(r):.3f}")
    frac_high = float((kf > 3.0).mean())
    frac_extreme = float((ratings < 1000).mean() + (ratings > 2500).mean())
    print(f"  fraction of real ratings with kFunction>3.0 (still-provisional zone): {frac_high:.4f}")
    print(f"  fraction of real ratings outside [1000,2500] (far outside UTS's own typical range): {frac_extreme:.4f}")

    print(f"\n=== STEP 2: UTS-K variant, decay={DECAY_BEST} (primary, apples-to-apples bundle test) ===")
    fit_uts_d3, state_uts_d3 = run_elo(df, BLEND, DECAY_BEST, k_mode="uts")
    row_uts_d3 = market_logloss(fit_uts_d3, ev)
    print(f"  UTS-K (blend={BLEND}, decay={DECAY_BEST}): n={row_uts_d3['n']} "
          f"logloss={row_uts_d3['logloss']:.4f} atp={row_uts_d3['atp_logloss']:.4f} "
          f"wta={row_uts_d3['wta_logloss']:.4f}")
    anchors_uts_d3 = anchor_check(f"UTS-K, decay={DECAY_BEST}", fit_uts_d3, state_uts_d3)

    print(f"\n=== STEP 3: UTS-K variant, decay=0.0 (sensitivity check) ===")
    fit_uts_d0, state_uts_d0 = run_elo(df, BLEND, 0.0, k_mode="uts")
    row_uts_d0 = market_logloss(fit_uts_d0, ev)
    print(f"  UTS-K (blend={BLEND}, decay=0.0): n={row_uts_d0['n']} "
          f"logloss={row_uts_d0['logloss']:.4f} atp={row_uts_d0['atp_logloss']:.4f} "
          f"wta={row_uts_d0['wta_logloss']:.4f}")
    anchors_uts_d0 = anchor_check("UTS-K, decay=0.0", fit_uts_d0, state_uts_d0)

    print(f"\n=== STEP 4 (diagnostic, secondary): which piece of the UTS bundle drives the ATP/WTA split? ===")
    print("  (context-multiplier stack alone, kFunction fixed=1.0; and kFunction alone, context fixed=32)")
    fit_ctx, _ = run_elo(df, BLEND, DECAY_BEST, k_mode="uts_context_only")
    row_ctx = market_logloss(fit_ctx, ev)
    print(f"  uts_context_only (decay={DECAY_BEST}): logloss={row_ctx['logloss']:.4f} "
          f"atp={row_ctx['atp_logloss']:.4f} wta={row_ctx['wta_logloss']:.4f}")
    fit_kf, _ = run_elo(df, BLEND, DECAY_BEST, k_mode="uts_kfunc_only")
    row_kf = market_logloss(fit_kf, ev)
    print(f"  uts_kfunc_only   (decay={DECAY_BEST}): logloss={row_kf['logloss']:.4f} "
          f"atp={row_kf['atp_logloss']:.4f} wta={row_kf['wta_logloss']:.4f}")
    print("  (diagnostic only - not scored as a real candidate arm, not anchor-checked)")

    print(f"\n=== FINAL COMPARISON TABLE (2024-25 market-odds-joined holdout) ===")
    hier_logloss = 0.6275   # from RESUME-beat-elo.md round 10 - not rerun, cited for context
    table = pd.DataFrame([
        {"model": "Elo (current best: Sackmann K, ordering-fixed, decay=0.3)", "logloss": row_sack["logloss"],
         "atp": row_sack["atp_logloss"], "wta": row_sack["wta_logloss"]},
        {"model": f"Elo (UTS K, decay={DECAY_BEST})", "logloss": row_uts_d3["logloss"],
         "atp": row_uts_d3["atp_logloss"], "wta": row_uts_d3["wta_logloss"]},
        {"model": "Elo (UTS K, decay=0.0)", "logloss": row_uts_d0["logloss"],
         "atp": row_uts_d0["atp_logloss"], "wta": row_uts_d0["wta_logloss"]},
        {"model": "Hierarchical (v25+round9, NOT rerun this task - cited for context)",
         "logloss": hier_logloss, "atp": float("nan"), "wta": float("nan")},
    ])
    print(table.to_string(index=False))
    table.to_csv(r"C:\dev\dropshotverse\dropshotdata\tune_scratch\uts_k_comparison.csv", index=False)
    print("\nwrote tune_scratch/uts_k_comparison.csv")

    print(f"\nanchors: UTS decay={DECAY_BEST} -> {'PASS' if anchors_uts_d3 else 'FAIL'}; "
          f"UTS decay=0.0 -> {'PASS' if anchors_uts_d0 else 'FAIL'}")
