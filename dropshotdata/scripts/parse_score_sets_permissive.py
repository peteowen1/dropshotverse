"""Recovers the ~2.3% set-score coverage gap found while building set-level
Elo (issue 29) - hierarchical_model.py's parse_score_sets() is deliberately
strict (only returns scores representable by p_set_full's DP, which has no
way to model a super-tiebreak-decided set or an unbounded advantage set), a
real requirement for ITS use case (ticket 21's full-DP set-likelihood).

Set-level Elo doesn't need DP-representability - it only needs "who won
more games in each set", a much weaker requirement. This is a separate,
purpose-built parser for that, NOT a fix to the frozen file (their
constraints are genuinely different, not the same requirement done wrong).

Handles the two formats found in the ~27,307-row gap (see issue 29):
  - final-set super-tiebreak in brackets, e.g. "6-2 4-6 [10-5]"
  - long-form no-tiebreak deciding sets, e.g. "7-5 2-6 4-6 7-6(1) 11-9"
plus everything the strict parser already handles (standard sets, regular
tiebreaks "7-6(4)").
"""
from __future__ import annotations

import re

_SET_TOKEN_RE = re.compile(r"^\[?(\d+)-(\d+)\]?(?:\(\d+\))?$")


def parse_score_sets_permissive(score: str) -> list[tuple[int, int]] | None:
    """Returns [(winner_games, loser_games), ...] per set, or None if score
    is unparseable/empty. No legality restriction on the (ga, gb) values -
    any positive-integer pair is accepted, covering standard sets, regular
    tiebreaks, bracketed super-tiebreaks, and long-form deciders alike."""
    tokens = str(score).split()
    if not tokens:
        return None
    sets: list[tuple[int, int]] = []
    for tok in tokens:
        m = _SET_TOKEN_RE.match(tok)
        if not m:
            return None
        ga, gb = int(m.group(1)), int(m.group(2))
        if ga == gb:
            return None   # a set can't end tied - malformed token, don't guess
        sets.append((ga, gb))
    return sets or None


if __name__ == "__main__":
    tests = ["6-4 6-4", "6-2 4-6 [10-5]", "7-5 2-6 4-6 7-6(1) 11-9", "7-6(5) 3-6 10-5",
            "UNK", "6-1 0-0", ""]
    for t in tests:
        print(f"{t!r:35s} -> {parse_score_sets_permissive(t)}")
