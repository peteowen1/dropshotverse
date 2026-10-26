"""Pruning of the published upcoming-event tables (issue #44).

Kept apart from simulate_wta_event.py, which imports xgboost and the whole
model stack, so CI can test it with pandas and pyarrow alone.

A table holds only events that have not started. Dates are calendar dates:
an offset-bearing startDate keeps the date as written and loses the offset, so
a tz-aware value can never be compared with the tz-naive --as-of.
"""
from __future__ import annotations

import pandas as pd


def calendar_date(value) -> pd.Timestamp:
    """The calendar date of value as a tz-naive midnight Timestamp (NaT if it
    does not parse). "2026-09-30T00:00:00+10:00" is 2026-09-30, not the UTC day."""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if ts is pd.NaT:
        return pd.NaT
    return ts.tz_localize(None).normalize()


def prune_started(table: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Keep the rows of events that have not started (start_date >= as_of).

    An event starting on as_of itself is kept. An unparseable start_date is
    dropped, since nothing shows the event is still upcoming. Every event may
    go: the result can be empty and still has the input's columns. Returns the
    kept rows and the sorted ids of the events that were dropped.
    """
    as_of = calendar_date(as_of)
    if pd.isna(as_of):
        # NaT compares False with everything, which would drop every event.
        raise ValueError(f"prune_started: as_of {as_of!r} is not a date")
    dates = table["start_date"].map(calendar_date)
    if len(table) and dates.isna().all():
        # A renamed or nulled start_date would otherwise empty the live file
        # and pass for "every event had started".
        raise ValueError("prune_started: no row has a parseable start_date")
    keep = (dates >= as_of).to_numpy(dtype=bool)
    dropped = sorted(table.loc[~keep, "tournament_id"].unique())
    return table.loc[keep].copy(), dropped
