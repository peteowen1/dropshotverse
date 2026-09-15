"""D-42 follow-up: real examples, before any dispersion-ratio arithmetic.
Per Pete's model-building rule - design on real rows first, fit only once
they read right.

For each main-draw US Open match 2022-2026 with a uniquely-resolved player on
BOTH sides, prints: who played, hier's PRE-tournament predicted serve-hold
probability for each (from their frozen state as of that year's date0, via
the exact formula simulate_tournament_updating.predict_p1_p2 uses), and the
REAL serve-hold rate each player achieved in that specific match (from the
point tape - a hard number, not a model output). No aggregation, no
dispersion ratio yet - just: state before, what hier expected, what actually
happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _common
import hierarchical_model as hm
import simulate_tournament_updating as stu
from resolve_usopen_player_names import build_archive_index, resolve_one

DD = _common.DD
TOURNEY_ID = {"atp": "560", "wta": "560"}   # US Open code, both tours
YEARS = [2022, 2023, 2024, 2025]


def log(m: str = "") -> None:
    print(m, flush=True)


def load_field_and_date(tour: str, year: int) -> tuple[pd.Timestamp, str]:
    raw = pd.read_csv(DD / f"raw/github/{tour}/{tour}_matches_{year}.csv", low_memory=False)
    tmatch = raw[raw["tourney_id"] == f"{year}-{TOURNEY_ID[tour]}"]
    if tmatch.empty:
        raise SystemExit(f"no {tour} US Open ({TOURNEY_ID[tour]}) rows in {year} archive")
    date0 = pd.to_datetime(tmatch["tourney_date"].iloc[0], format="%Y%m%d")
    surface = str(tmatch["surface"].iloc[0]).strip().title()
    return date0, surface


def real_serve_hold_rates(points: pd.DataFrame, match_id: str) -> tuple[float, float, int, int]:
    g = points[points["match_id"] == match_id]
    s1 = g[g["PointServer"] == "1"]
    s2 = g[g["PointServer"] == "2"]
    r1 = float((s1["PointWinner"] == "1").mean()) if len(s1) else float("nan")
    r2 = float((s2["PointWinner"] == "2").mean()) if len(s2) else float("nan")
    return r1, r2, len(s1), len(s2)


def main() -> None:
    names = pd.read_parquet(_common.SCRATCH / "usopen_player_names_by_match.parquet")
    names = names[names["p1_name"].notna()].copy()
    names["mid"] = names["match_id"].astype(str)
    names = names[names["mid"].str.len() == 4]   # main draw only (D-56)

    df, _ = hm.prepare(hm.load_matches())
    rows_out = []

    for year in YEARS:
        points = pd.read_parquet(DD / f"usopen_{year}_points.parquet",
                                 columns=["match_id", "PointServer", "PointWinner"])
        for tour_ev, tour in (("1", "atp"), ("2", "wta")):
            date0, surface = load_field_and_date(tour, year)
            ids_by_key, _ = build_archive_index(tour)
            sub = names[(names["year"] == year) & (names["mid"].str[0] == tour_ev)]

            resolved_uids: dict[str, str] = {}
            for _, r in sub.iterrows():
                for nm in (r["p1_name"], r["p2_name"]):
                    if nm not in resolved_uids:
                        outcome, uid = resolve_one(nm, ids_by_key)
                        resolved_uids[nm] = f"{tour}:{uid}" if outcome == "unique" else None

            field = sorted({u for u in resolved_uids.values() if u})
            snaps = stu.build_snapshots({f"{tour}:{year}usopen": {"date0": date0, "field_uids": field}},
                                        df=df, cache_dir=_common.SCRATCH / "snapshot_cache")
            snap = snaps.get(f"{tour}:{year}usopen", {})

            for _, r in sub.iterrows():
                u1, u2 = resolved_uids.get(r["p1_name"]), resolved_uids.get(r["p2_name"])
                if not u1 or not u2 or u1 not in snap or u2 not in snap:
                    continue
                pred1, pred2 = stu.predict_p1_p2(snap[u1], snap[u2], surface)
                act1, act2, n1, n2 = real_serve_hold_rates(points, r["mid"])
                if n1 < 20 or n2 < 20:
                    continue
                rows_out.append({"year": year, "tour": tour, "match_id": r["mid"],
                                 "p1": r["p1_name"], "p2": r["p2_name"],
                                 "hier_pred_p1": pred1, "actual_p1": act1, "n1": n1,
                                 "hier_pred_p2": pred2, "actual_p2": act2, "n2": n2})

    out = pd.DataFrame(rows_out)
    out_path = _common.SCRATCH / "hier_vs_actual_serve_examples.parquet"
    out.to_parquet(out_path, index=False)
    log(f"{len(out):,} scoreable player-match rows (from {out['match_id'].nunique():,} matches) -> {out_path}")

    long = pd.concat([
        out.rename(columns={"p1": "player", "hier_pred_p1": "hier_pred", "actual_p1": "actual", "n1": "n_serve_pts"})[
            ["year", "tour", "match_id", "player", "hier_pred", "actual", "n_serve_pts"]],
        out.rename(columns={"p2": "player", "hier_pred_p2": "hier_pred", "actual_p2": "actual", "n2": "n_serve_pts"})[
            ["year", "tour", "match_id", "player", "hier_pred", "actual", "n_serve_pts"]],
    ], ignore_index=True)
    long["gap"] = long["actual"] - long["hier_pred"]

    log(f"\nPer-player-match: hier's pre-tournament predicted serve-hold prob vs the "
        f"REAL rate in that one match (n_serve_pts = real service points behind it).")
    log("Positive gap = player held serve MORE than hier expected.\n")

    log("== 10 players hier over-predicted the most (real hold rate well BELOW prediction) ==")
    log(long.nsmallest(10, "gap")[["year", "player", "hier_pred", "actual", "gap", "n_serve_pts"]]
        .to_string(index=False, formatters={"hier_pred": "{:.1%}".format, "actual": "{:.1%}".format,
                                             "gap": "{:+.1%}".format}))

    log("\n== 10 players hier under-predicted the most (real hold rate well ABOVE prediction) ==")
    log(long.nlargest(10, "gap")[["year", "player", "hier_pred", "actual", "gap", "n_serve_pts"]]
        .to_string(index=False, formatters={"hier_pred": "{:.1%}".format, "actual": "{:.1%}".format,
                                             "gap": "{:+.1%}".format}))

    log(f"\n== spread comparison (the actual D-42 question) ==")
    log(f"  hier_pred  std={long['hier_pred'].std():.4f}  range=[{long['hier_pred'].min():.1%}, {long['hier_pred'].max():.1%}]")
    log(f"  actual     std={long['actual'].std():.4f}  range=[{long['actual'].min():.1%}, {long['actual'].max():.1%}]")
    log(f"  dispersion ratio (hier std / actual std) = {long['hier_pred'].std() / long['actual'].std():.3f}")
    log(f"  n = {len(long):,} player-match rows, {out['year'].nunique()} years")


if __name__ == "__main__":
    main()
