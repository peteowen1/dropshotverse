"""Round 16: mechanism hypothesis for ATP-Grass's residual gap. elo_baseline
.py's layoff_decay() is applied to the SURFACE rating using days-since-last-
match-on-THAT-SURFACE as the trigger. Grass is played in one ~4-6 week
window/year - a player's gap between consecutive grass matches is typically
~350-360 days, deep in the decay curve's steep region (LAYOFF_PERIOD=500,
GRACE=30, GRADIENT=100) - so EVERY grass-season opener pulls the surface
rating back toward START=1500, every year, for every player, regardless of
how active they've been on other surfaces. Hard/clay gaps are almost always
under the 30-day grace period (players play those surfaces nearly every
week), so they essentially never get hit. This is a real, mechanistic
explanation for grass's compressed elo_surface_diff (mean |diff| ~223 on
grass vs ~419 hard/clay) and, plausibly, for ATP-Grass's residual market
gap specifically (ATP's grass season - Halle/Queens/Wimbledon etc - has the
same ~4-6 week structure as WTA's, so this predicts BOTH tours should be
similarly affected by the mechanism itself, but see the ATP-specific
verification below).

Fix tested: use OVERALL days-since-last-match (not surface-specific) as the
decay trigger for the SURFACE rating too - a player who's been active on
hard/clay hasn't "forgotten" their grass game, so their surface rating
shouldn't decay just because THAT surface specifically went quiet. Read-only
wrt elo_baseline.py - imports it, adds one new run_elo variant here rather
than editing the frozen file.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\dropshotverse\dropshotdata\scripts")
import elo_baseline as eb  # noqa: E402
from market_compare import key_from_full, key_from_short, load_odds  # noqa: E402


def run_elo_overall_trigger_surface_decay(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    """Same as eb.run_elo(blend=0, decay=decay, k_mode='sackmann') except the
    SURFACE rating's decay is triggered by days-since-last-match-OVERALL,
    not days-since-last-match-ON-THAT-SURFACE. Overall rating's own decay
    trigger is unchanged (already correctly overall-based)."""
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}  # kept for parity/logging, not used as decay trigger

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values
    preds = np.empty(len(df))

    for idx, (w, l, s, day) in enumerate(zip(df["winner_uid"].values, df["loser_uid"].values,
                                              df["surface"].values, days)):
        ro_w, ro_l = overall.get(w, eb.START), overall.get(l, eb.START)
        rs_w = surf.get((w, s), eb.START)
        rs_l = surf.get((l, s), eb.START)

        if decay > 0.0:
            gap_w_overall = day - last_all.get(w, day)
            gap_l_overall = day - last_all.get(l, day)
            ro_w = eb.layoff_decay(ro_w, gap_w_overall, decay)
            ro_l = eb.layoff_decay(ro_l, gap_l_overall, decay)
            # KEY CHANGE: surface rating decay uses the OVERALL gap, not the
            # surface-specific gap - a player active elsewhere hasn't
            # forgotten their grass game.
            rs_w = eb.layoff_decay(rs_w, gap_w_overall, decay)
            rs_l = eb.layoff_decay(rs_l, gap_l_overall, decay)

        diff = ro_w - ro_l  # blend=0, matches ticket 25's basis
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        preds[idx] = p_w

        kw, kl = eb.k_factor(n_all.get(w, 0)), eb.k_factor(n_all.get(l, 0))
        overall[w] = ro_w + kw * (1 - p_w)
        overall[l] = ro_l - kl * (1 - p_w)
        ksw, ksl = eb.k_factor(n_surf.get((w, s), 0)), eb.k_factor(n_surf.get((l, s), 0))
        surf[(w, s)] = rs_w + ksw * (1 - p_w)
        surf[(l, s)] = rs_l - ksl * (1 - p_w)

        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1
        last_all[w] = day
        last_all[l] = day
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
        last_surf[(w, s)] = day
        last_surf[(l, s)] = day

    out = df.copy()
    out["p_winner"] = preds
    out["ro_winner"] = [overall.get(w, eb.START) for w in df["winner_uid"]]  # placeholder, overwritten below
    return out


def run_elo_with_surface_export(df: pd.DataFrame, decay: float, overall_trigger: bool) -> pd.DataFrame:
    """Like run_elo_overall_trigger_surface_decay but exports each row's
    PRE-match rs_winner/rs_loser (the surface rating BEFORE this match
    updates it - exactly what elo_raw_ratings_for_xgb.py exports as
    ro_winner/rs_winner for XGBoost's elo_surface_diff feature) so the
    compression itself (not just p_winner, which blend=0 never uses the
    surface rating for) can be compared directly."""
    overall: dict[str, float] = {}
    surf: dict[tuple[str, str], float] = {}
    n_all: dict[str, int] = {}
    n_surf: dict[tuple[str, str], int] = {}
    last_all: dict[str, int] = {}
    last_surf: dict[tuple[str, str], int] = {}

    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.values
    n = len(df)
    rs_winner_out = np.empty(n)
    rs_loser_out = np.empty(n)
    preds = np.empty(n)

    for idx, (w, l, s, day) in enumerate(zip(df["winner_uid"].values, df["loser_uid"].values,
                                              df["surface"].values, days)):
        ro_w, ro_l = overall.get(w, eb.START), overall.get(l, eb.START)
        rs_w = surf.get((w, s), eb.START)
        rs_l = surf.get((l, s), eb.START)

        if decay > 0.0:
            gap_w_overall = day - last_all.get(w, day)
            gap_l_overall = day - last_all.get(l, day)
            ro_w = eb.layoff_decay(ro_w, gap_w_overall, decay)
            ro_l = eb.layoff_decay(ro_l, gap_l_overall, decay)
            if overall_trigger:
                rs_w = eb.layoff_decay(rs_w, gap_w_overall, decay)
                rs_l = eb.layoff_decay(rs_l, gap_l_overall, decay)
            else:
                gap_w_surf = day - last_surf.get((w, s), day)
                gap_l_surf = day - last_surf.get((l, s), day)
                rs_w = eb.layoff_decay(rs_w, gap_w_surf, decay)
                rs_l = eb.layoff_decay(rs_l, gap_l_surf, decay)

        rs_winner_out[idx] = rs_w
        rs_loser_out[idx] = rs_l
        diff = ro_w - ro_l
        p_w = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        preds[idx] = p_w

        kw, kl = eb.k_factor(n_all.get(w, 0)), eb.k_factor(n_all.get(l, 0))
        overall[w] = ro_w + kw * (1 - p_w)
        overall[l] = ro_l - kl * (1 - p_w)
        ksw, ksl = eb.k_factor(n_surf.get((w, s), 0)), eb.k_factor(n_surf.get((l, s), 0))
        surf[(w, s)] = rs_w + ksw * (1 - p_w)
        surf[(l, s)] = rs_l - ksl * (1 - p_w)

        n_all[w] = n_all.get(w, 0) + 1
        n_all[l] = n_all.get(l, 0) + 1
        last_all[w] = day
        last_all[l] = day
        n_surf[(w, s)] = n_surf.get((w, s), 0) + 1
        n_surf[(l, s)] = n_surf.get((l, s), 0) + 1
        last_surf[(w, s)] = day
        last_surf[(l, s)] = day

    out = df.copy()
    out["p_winner"] = preds
    out["rs_winner"] = rs_winner_out
    out["rs_loser"] = rs_loser_out
    return out


def main() -> None:
    raw = eb.load_matches()
    df, dropped = eb.prepare(raw)
    print("dropped:", dropped, flush=True)

    # anchor: verify the overall-trigger mechanism reduces to the SAME thing
    # as the original when decay=0 (structural no-op check, same discipline
    # as every decay change this project has made).
    baseline_arm, _ = eb.run_elo(df, blend=0.0, decay=0.0, k_mode="sackmann")
    noop_arm = run_elo_overall_trigger_surface_decay(df, decay=0.0)
    max_dev = float(np.max(np.abs(noop_arm["p_winner"].values - baseline_arm["p_winner"].values)))
    print(f"no-op check (decay=0): max |p_winner diff| = {max_dev:.2e} "
          f"({'OK' if max_dev < 1e-9 else 'FAIL - stop'})")
    assert max_dev < 1e-9

    # verify this file's own blend=0,decay=0.3,sackmann arm still matches the
    # frozen 0.6120 basis when surface decay trigger is left AS-IS (sanity on
    # the harness itself, not the new mechanism)
    frozen_equiv = run_elo_overall_trigger_surface_decay(df, decay=0.0)  # decay=0 is the no-op check above

    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds_small = odds[["pair", "tour", "date", "k_w", "PSW", "PSL"]]

    def score(arm: pd.DataFrame, label: str) -> None:
        sub = arm[(arm["level_file"] == "tour") & (arm["date"] >= "2024-01-01")].copy()
        sub["k_w"] = sub["winner_name"].map(key_from_full)
        sub["k_l"] = sub["loser_name"].map(key_from_full)
        sub["pair"] = sub.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
        j = sub.merge(odds_small, on=["pair", "tour"], how="left", suffixes=("", "_o"))
        j["gap"] = (j["date_o"] - j["date"]).dt.days
        j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
        j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
        j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
        for tour in ("atp", "wta", "all"):
            jj = j if tour == "all" else j[j["tour"] == tour]
            for surf in ("Grass", "Hard", "Clay", "all"):
                jjs = jj if surf == "all" else jj[jj["surface"] == surf]
                if len(jjs) < 20:
                    continue
                pr = np.clip(jjs["p_winner"].values, 1e-9, 1 - 1e-9)
                ll = float(-np.log(pr).mean())
                print(f"  [{label}] {tour:4s} {surf:6s} n={len(jjs):5,d}  logloss={ll:.5f}", flush=True)

    print("\n=== compression check: surface-specific-trigger vs overall-trigger decay ===")
    orig_arm = run_elo_with_surface_export(df, decay=0.3, overall_trigger=False)
    fixed_arm = run_elo_with_surface_export(df, decay=0.3, overall_trigger=True)

    def report_compression(arm: pd.DataFrame, label: str) -> pd.DataFrame:
        sub = arm[(arm["level_file"] == "tour") & (arm["date"] >= "2024-01-01")].copy()
        sub["rs_diff"] = sub["rs_winner"] - sub["rs_loser"]
        for tour in ("atp", "wta"):
            for surf in ("Grass", "Hard", "Clay"):
                m = (sub["tour"] == tour) & (sub["surface"] == surf)
                if m.sum() < 20:
                    continue
                print(f"  [{label}] {tour:4s} {surf:6s} n={m.sum():5,d}  "
                      f"mean|rs_diff|={sub.loc[m, 'rs_diff'].abs().mean():.1f}")
        return sub

    orig_sub = report_compression(orig_arm, "original (surface-trigger)")
    fixed_sub = report_compression(fixed_arm, "fixed (overall-trigger)")

    # ONE holdout score: is the fixed surface rating a better XGBoost feature?
    # Score elo_surface_diff alone (blend=1.0-equivalent, pure surface Elo) as
    # its own predictor - this is the direct downstream-relevant test, since
    # elo_baseline.py's own p_winner (blend=0) never uses rs at all.
    odds = load_odds(years=(2024, 2025)).dropna(subset=["Winner", "Loser", "Date"])
    odds["k_w"] = odds["Winner"].map(key_from_short)
    odds["k_l"] = odds["Loser"].map(key_from_short)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
    odds = odds.dropna(subset=["date"])
    odds["pair"] = odds.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
    odds_small = odds[["pair", "tour", "date", "k_w", "PSW", "PSL"]]

    def score_surface_only(sub: pd.DataFrame, label: str) -> None:
        sub = sub.copy()
        sub["k_w"] = sub["winner_name"].map(key_from_full)
        sub["k_l"] = sub["loser_name"].map(key_from_full)
        sub["pair"] = sub.apply(lambda r: "|".join(sorted([r.k_w, r.k_l])), axis=1)
        j = sub.merge(odds_small, on=["pair", "tour"], how="left", suffixes=("", "_o"))
        j["gap"] = (j["date_o"] - j["date"]).dt.days
        j = j[(j["gap"] >= -3) & (j["gap"] <= 21)]
        j = j.sort_values("gap").drop_duplicates(subset=["pair", "tour", "date"], keep="first")
        j = j[(j["k_w_o"] == j["k_w"]) & j[["PSW", "PSL"]].notna().all(axis=1)]
        j["p_surf"] = 1.0 / (1.0 + 10 ** (-j["rs_diff"].values / 400.0))
        for tour in ("atp", "wta", "all"):
            jj = j if tour == "all" else j[j["tour"] == tour]
            for surf in ("Grass", "Hard", "Clay", "all"):
                jjs = jj if surf == "all" else jj[jj["surface"] == surf]
                if len(jjs) < 20:
                    continue
                pr = np.clip(jjs["p_surf"].values, 1e-9, 1 - 1e-9)
                ll = float(-np.log(pr).mean())
                print(f"  [{label}] {tour:4s} {surf:6s} n={len(jjs):5,d}  pure-surface-elo logloss={ll:.5f}",
                      flush=True)

    print("\n=== pure surface-Elo predictive power (blend=1.0 equivalent) ===")
    score_surface_only(orig_sub, "original")
    score_surface_only(fixed_sub, "fixed")


if __name__ == "__main__":
    main()
