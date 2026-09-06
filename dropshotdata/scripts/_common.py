"""Shared constants for dropshotdata/scripts (ticket 40).

DD resolves from this file's location, so the scripts run from any checkout:
the laptop at C:/dev/dropshotverse, a git worktree, or a Linux Actions
runner. Before 2026-09-06 every script hard-coded the laptop path and the
Actions workflow needed a junction to fake it.
"""
from __future__ import annotations

import os
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# DROPSHOT_DATA_DIR lets a worktree or test run the scripts against another
# checkout's data (a worktree has the code but none of the gitignored data).
DD = Path(os.environ["DROPSHOT_DATA_DIR"]) if os.environ.get("DROPSHOT_DATA_DIR") else SCRIPTS.parent
SCRATCH = DD / "tune_scratch"
BLOG = DD / "blog"
RAW = DD / "raw"


def log(m: str = "") -> None:
    print(m, flush=True)
