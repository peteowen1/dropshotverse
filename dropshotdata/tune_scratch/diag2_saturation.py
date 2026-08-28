"""Diagnostic (read-only): why did DSV_SHORT_LAYOFF_BUMP backfire globally?
Hypothesis: inflate() fires on almost EVERY match (weeks>=1 is the common
case, not a rare event), and the bump is additive with only a ceiling
(MAX_VAR) as a backstop, applied cumulatively across a player's ENTIRE
career and to BOTH v_a and v_b every time. So even a small per-event bump
compounds over hundreds of matches into most active players' variance being
pushed to (or near) MAX_VAR, well before they ever reach the specific
1-12-week-gap eval matches the calibration finding was diagnosed on -
destroying exactly the "known players have earned low uncertainty"
information the model needs to be confidently right on lopsided matches
elsewhere, while barely touching the narrow slice it was meant to fix.

Uses only tune_scratch/diag_layoff.parquet (full-archive weeks-since-last,
already computed, no refit needed).
"""
import numpy as np
import pandas as pd

LAYOFF = r"C:\dev\dropshotverse\dropshotdata\tune_scratch\diag_layoff.parquet"
MAX_VAR = 4.0
RW_VAR_PER_WEEK = 0.005

if __name__ == "__main__":
    lay = pd.read_parquet(LAYOFF)
    print(f"archive rows: {len(lay):,}")

    # stack winner-side and loser-side observations into one long per-(uid,date)
    # table: every time ANY player takes the court, is their gap-since-last
    # >=1 week (bump-eligible) or not?
    w = lay[["date", "winner_uid", "weeks_since_last_w", "n_prior_w"]].rename(
        columns={"winner_uid": "uid", "weeks_since_last_w": "weeks", "n_prior_w": "n_prior"})
    l = lay[["date", "loser_uid", "weeks_since_last_l", "n_prior_l"]].rename(
        columns={"loser_uid": "uid", "weeks_since_last_l": "weeks", "n_prior_l": "n_prior"})
    long = pd.concat([w, l], ignore_index=True)
    long = long.dropna(subset=["weeks"])  # first match for a player has no prior gap
    print(f"\nlong (per-player-appearance) rows with a defined gap: {len(long):,}")

    print("\n=== how common is each gap bucket, across EVERY match in the 50-year archive ===")
    bucket = pd.cut(long["weeks"], [-0.01, 1/7, 1, 4, 12, 26, 1e6],
                     labels=["<1day(same-day/RR)", "1day-1wk", "1-4wk", "4-12wk", "12-26wk", ">26wk(LAYOFF_WEEKS)"])
    print(bucket.value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.1%}"))
    frac_bump_eligible = (long["weeks"] >= 1).mean()
    print(f"\nfraction of ALL player-appearances with weeks>=1 (SHORT_LAYOFF_BUMP triggers): {frac_bump_eligible:.1%}")
    print("-> this is NOT a rare '1-12wk gap' event; it fires on the vast majority of a player's")
    print("   matches, since back-to-back same-week matches (same tournament, weeks<1) are the")
    print("   minority case, not the norm, across a 50-year multi-tour archive.")

    print("\n=== cumulative bump-eligible event count by the time a player reaches n_prior matches ===")
    # per player, cumulative count of weeks>=1 events, indexed by their match number
    long = long.sort_values(["uid", "date"])
    long["bump_event"] = (long["weeks"] >= 1).astype(int)
    long["cum_bump_events"] = long.groupby("uid")["bump_event"].cumsum()
    for n_target in [10, 25, 50, 100, 200, 400]:
        sub = long[long["n_prior"] == n_target]
        if len(sub) == 0:
            continue
        print(f"  at match #{n_target:4d} (n_prior={n_target}): "
              f"median cum bump-events so far = {sub['cum_bump_events'].median():.0f}, "
              f"p25={sub['cum_bump_events'].quantile(.25):.0f}, p75={sub['cum_bump_events'].quantile(.75):.0f}  (n={len(sub)})")

    print("\n=== back-of-envelope: additive variance from bump alone (ignoring ADF shrinkage), vs MAX_VAR ceiling ===")
    print(f"MAX_VAR={MAX_VAR}, baseline RW_VAR_PER_WEEK contribution per week ~{RW_VAR_PER_WEEK} (much smaller,")
    print("and only accrues per elapsed week, not per event) -- compare to a per-EVENT flat SHORT_LAYOFF_BUMP:")
    for bump in [0.005, 0.01, 0.02, 0.04, 0.06]:
        for n_target in [50, 100, 200]:
            med_events = long.loc[long["n_prior"] == n_target, "cum_bump_events"].median()
            if np.isnan(med_events):
                continue
            add = bump * med_events
            print(f"  BUMP={bump:.3f}  by match #{n_target:4d}: median cum bump-events={med_events:.0f} "
                  f"-> additive var (ignoring shrinkage) = {add:.2f}  "
                  f"{'>= MAX_VAR (SATURATES)' if add >= MAX_VAR else ''}")

    print("\n=== how many DISTINCT players ever reach n_prior=100+ (i.e. actually accumulate this) ===")
    max_n = long.groupby("uid")["n_prior"].max()
    for thresh in [10, 50, 100, 200]:
        print(f"  players with >= {thresh} matches: {(max_n >= thresh).sum():,} / {max_n.shape[0]:,} total players")
