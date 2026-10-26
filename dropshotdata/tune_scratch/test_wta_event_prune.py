"""simulate_wta_event must drop started events from every published table.

A started event is not forecast: no model work runs and only the pruning
happens, which may empty the files. Each write drops events
whose start_date is before --as-of from the merged parquet, and the draw JSON
and pairwise table keep only the events that survive in that parquet.
Needs xgboost (the script imports the model stack), so CI does not run this;
the pure pruning rules are tested in test_elo_freshness.py, which CI does run.
Offline: the model layer is stubbed; the table reads and writes are real, in a
temp directory. Every run has a local table file first, so no read falls back
to the public copy.
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dropshotdata" / "scripts"))

import simulate_wta_event as s  # noqa: E402

AS_OF = "2026-10-25"
UIDS = ["wta:A", "wta:B", "wta:C", "wta:D"]
OUTS = ("upcoming_predictions.parquet", "upcoming_draw.json", "upcoming_pairwise.parquet")


def stubs(start_date, calls):
    """Stand-ins for the model layer. calls["model"] counts how often the
    trained model is requested, i.e. how much model work ran."""
    def train():
        calls["model"] += 1
        return None, None, None

    n = len(UIDS)
    hist = pd.DataFrame({"date": [pd.Timestamp("2026-10-01")], "winner_uid": [UIDS[0]], "winner_name": ["A"],
                         "loser_uid": [UIDS[1]], "loser_name": ["B"]})
    return mock.patch.multiple(
        s,
        calendar_row=lambda group, year: {
            "startDate": start_date, "title": "Test Open", "surface": "Hard", "singlesDrawSize": 4,
            "tournamentGroup": {"name": "Test", "level": "WTA 250"}},
        load_entrants=lambda group, year, refetch: [],
        build_field=lambda entrants, tour: {"field_uids": UIDS, "names": {u: u[-1] for u in UIDS},
                                            "seed_map": {}, "synthetic": []},
        train_bagged_model=train,
        load_age_table=lambda: None,
        bridge=SimpleNamespace(load_matches_with_bridge=lambda hm: None),
        hm=SimpleNamespace(prepare=lambda _: (hist, None)),
        stu=SimpleNamespace(build_snapshots=lambda *a, **k: {}),
        build_h2h_dict=lambda uids: {},
        build_player_table=lambda *a, **k: {},
        build_round_pred_matrices=lambda uids, *a, **k: {st: np.full((n, n), 0.5) for st in s.STAGE_ORDER[:-1]},
        random_bracket_with_byes=lambda uids, seed_map, n_slots, n_byes, rng: list(uids),
        simulate_bracket=lambda bracket, idx, matrices, weakest, rng: {bracket[0]: "W"},
        flatten_pairwise=lambda matrices, uids, tour: pd.DataFrame(
            {"player_a": [UIDS[0]], "player_b": [UIDS[1]], "p": [0.5]}),
        build_estimated_draw=lambda bracket, names, seed_map, n_slots: [{"player": u} for u in bracket],
    )


def run(d, start, *extra, group=1050):
    """Run main() on a stubbed model, reading and writing files in d.
    Returns (SystemExit code or None, calls)."""
    calls = {"model": 0}
    argv = ["simulate_wta_event.py", "--group", str(group), "--year", "2026", "--sims", "10",
            "--out", str(d / OUTS[0]), "--out-draw", str(d / OUTS[1]), "--out-pairwise", str(d / OUTS[2]),
            *extra]
    with stubs(start, calls), mock.patch.object(sys, "argv", argv):
        try:
            s.main()
        except SystemExit as e:
            return e.code, calls
    return None, calls


def write_prev(d):
    """Published tables holding a started event (1020), one that started on
    as-of itself (1040), two upcoming (1030, 1040), and a draw entry with no
    parquet rows at all (0999)."""
    rows = []
    for tid, start in [("2026-1010", "2026-09-02"), ("2026-1020", "2026-09-30"),
                       ("2026-1030", "2026-11-07"), ("2026-1040", AS_OF)]:
        rows += [{"tournament_id": tid, "start_date": start, "player_id": u, "p_title": 0.25} for u in UIDS]
    pd.DataFrame(rows).to_parquet(d / OUTS[0], index=False)
    tourn = [{"tournament_id": t, "name": t} for t in ("2026-1010", "2026-1020", "2026-1030", "2026-1040", "2026-0999")]
    (d / OUTS[1]).write_text(json.dumps({"tournaments": tourn}), encoding="utf-8")
    pd.DataFrame([{"tournament_id": t, "player_a": UIDS[0], "player_b": UIDS[1], "p": 0.5}
                  for t in ("2026-1010", "2026-1020", "2026-1030", "2026-1040")]).to_parquet(d / OUTS[2], index=False)


def write_empty(d):
    """Published tables that exist but hold no rows."""
    pd.DataFrame(columns=["tournament_id", "start_date", "player_id", "p_title"]).to_parquet(d / OUTS[0], index=False)
    (d / OUTS[1]).write_text(json.dumps({"tournaments": []}), encoding="utf-8")
    pd.DataFrame(columns=["tournament_id", "player_a", "player_b", "p"]).to_parquet(d / OUTS[2], index=False)


def test_started_event_is_not_forecast_but_tables_are_pruned(d):
    """The China Open case: group 1020 started 2026-09-30 and is in the
    published tables. Asking for it does no model work and drops it."""
    write_prev(d)
    code, calls = run(d, "2026-09-30", "--allow-partial", "--as-of", AS_OF, group=1020)
    assert code is None, code
    assert calls["model"] == 0, calls
    ids = ["2026-1030", "2026-1040"]
    assert sorted(pd.read_parquet(d / OUTS[0])["tournament_id"].unique()) == ids
    assert [t["tournament_id"] for t in json.loads((d / OUTS[1]).read_text(encoding="utf-8"))["tournaments"]] == ids
    assert sorted(pd.read_parquet(d / OUTS[2])["tournament_id"].unique()) == ids


def test_started_event_can_empty_the_tables(d):
    """Only started events published: all three files end empty, with schema."""
    rows = [{"tournament_id": "2026-1020", "start_date": "2026-09-30", "player_id": u, "p_title": 0.25} for u in UIDS]
    cols = list(rows[0])
    pd.DataFrame(rows).to_parquet(d / OUTS[0], index=False)
    (d / OUTS[1]).write_text(json.dumps({"tournaments": [{"tournament_id": "2026-1020"}]}), encoding="utf-8")
    pd.DataFrame([{"tournament_id": "2026-1020", "player_a": "x", "player_b": "y", "p": 0.5}]).to_parquet(
        d / OUTS[2], index=False)
    code, calls = run(d, "2026-09-30", "--as-of", AS_OF, group=1020)
    assert code is None and calls["model"] == 0, (code, calls)
    table = pd.read_parquet(d / OUTS[0])
    assert len(table) == 0 and list(table.columns) == cols, (len(table), list(table.columns))
    assert json.loads((d / OUTS[1]).read_text(encoding="utf-8")) == {"tournaments": []}
    pw = pd.read_parquet(d / OUTS[2])
    assert len(pw) == 0 and list(pw.columns) == ["tournament_id", "player_a", "player_b", "p"], list(pw.columns)


def test_started_event_with_no_tables_writes_nothing(d):
    with mock.patch.object(s, "read_existing_parquet", lambda local, url: None), \
            mock.patch.object(s, "read_existing_draw", lambda local, url: None):
        code, calls = run(d, "2026-09-30", "--as-of", AS_OF, group=1020)
    assert code is None and calls["model"] == 0, (code, calls)
    assert [n for n in OUTS if (d / n).exists()] == []


def test_prunes_started_events_from_all_three_files(d):
    write_prev(d)
    code, calls = run(d, "2026-11-02", "--allow-partial", "--as-of", AS_OF)
    assert code is None, code
    table = pd.read_parquet(d / OUTS[0])
    assert sorted(table["tournament_id"].unique()) == ["2026-1030", "2026-1040", "2026-1050"], \
        sorted(table["tournament_id"].unique())
    assert len(table[table["tournament_id"] == "2026-1030"]) == len(UIDS)
    assert table.loc[table["tournament_id"] == "2026-1050", "start_date"].unique().tolist() == ["2026-11-02"]
    draw = json.loads((d / OUTS[1]).read_text(encoding="utf-8"))
    assert [t["tournament_id"] for t in draw["tournaments"]] == ["2026-1030", "2026-1040", "2026-1050"], \
        [t["tournament_id"] for t in draw["tournaments"]]
    pw = pd.read_parquet(d / OUTS[2])
    assert sorted(pw["tournament_id"].unique()) == ["2026-1030", "2026-1040", "2026-1050"], \
        sorted(pw["tournament_id"].unique())


def test_empty_published_table(d):
    write_empty(d)
    code, _ = run(d, "2026-11-02", "--allow-partial", "--as-of", AS_OF)
    assert code is None, code
    assert sorted(pd.read_parquet(d / OUTS[0])["tournament_id"].unique()) == ["2026-1050"]
    assert [t["tournament_id"] for t in json.loads((d / OUTS[1]).read_text(encoding="utf-8"))["tournaments"]] == ["2026-1050"]
    assert sorted(pd.read_parquet(d / OUTS[2])["tournament_id"].unique()) == ["2026-1050"]


def test_default_as_of_is_today_utc(d):
    write_empty(d)
    today = datetime.now(timezone.utc).date()
    code, calls = run(d, (today - timedelta(days=1)).isoformat(), "--allow-partial")
    assert code is None and calls["model"] == 0, (code, calls)
    assert len(pd.read_parquet(d / OUTS[0])) == 0
    code, _ = run(d, (today + timedelta(days=30)).isoformat(), "--allow-partial")
    assert code is None, code
    assert sorted(pd.read_parquet(d / OUTS[0])["tournament_id"].unique()) == ["2026-1050"]


TESTS = [test_started_event_is_not_forecast_but_tables_are_pruned, test_started_event_can_empty_the_tables,
         test_started_event_with_no_tables_writes_nothing,
         test_prunes_started_events_from_all_three_files, test_empty_published_table,
         test_default_as_of_is_today_utc]

if __name__ == "__main__":
    for t in TESTS:
        with tempfile.TemporaryDirectory() as tmp:
            t(Path(tmp))
        print(f"{t.__name__}: ok")
    print("\nALL WTA EVENT PRUNE TESTS PASSED")
