"""Moves transcribed from screen recordings of the iOS game (Nov 2026).

Each case: board before the move, the piece placed, and the board and points the
real game showed afterwards. 7 = M. Coordinates are (row, col), row 0 at the top.
These pin down the engine's default ("ios") scoring rules.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import pytest  # noqa: E402
import merged_fast  # noqa: E402

CASES = {
    # Three 6's make an M on the placed cell: 6*3.
    "six_to_M": (
        [[0, 0, 3, 6, 6], [5, 4, 0, 0, 0], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
        [6], [(1, 4)], 18,
        [[0, 0, 3, 0, 0], [5, 4, 0, 0, 7], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
    ),
    # Four 2's merge at once: 2*4, landing on the placed cell.
    "four_tile_group": (
        [[0, 3, 0, 0, 6], [0, 5, 0, 0, 2], [0, 0, 0, 0, 2], [0, 5, 0, 2, 0], [0, 0, 0, 0, 0]],
        [2], [(2, 3)], 8,
        [[0, 3, 0, 0, 6], [0, 5, 0, 0, 0], [0, 0, 0, 3, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 0]],
    ),
    # Domino 3/6: only the 3 merges.
    "domino_one_half": (
        [[6, 0, 3, 3, 7], [5, 0, 5, 1, 7], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
        [3, 6], [(0, 1), (1, 1)], 9,
        [[6, 4, 0, 0, 7], [5, 6, 5, 1, 7], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
    ),
    # Domino 4/2, both halves merge in separate groups: 6 + 12, both counted
    # (the score ticked 60 -> 66 -> 78). mergedMain gave only 12.
    "domino_both_halves": (
        [[0, 3, 0, 0, 6], [2, 0, 4, 4, 2], [0, 0, 2, 2, 0], [0, 0, 4, 0, 1], [0, 0, 4, 0, 0]],
        [4, 2], [(1, 1), (2, 1)], 18,
        [[0, 3, 0, 0, 6], [2, 5, 0, 0, 2], [0, 3, 0, 0, 0], [0, 0, 4, 0, 1], [0, 0, 4, 0, 0]],
    ),
    # Partner chain: 2's -> 3, which joins the domino's 3 -> 4 on the 3's cell -> 5.
    # (6 + 9 + 12) * (1*2*3) = 162 (score 78 -> 240).
    "partner_chain": (
        [[0, 3, 0, 0, 6], [2, 5, 0, 0, 2], [2, 3, 0, 0, 0], [0, 0, 4, 0, 1], [0, 0, 4, 0, 0]],
        [2, 3], [(3, 0), (3, 1)], 162,
        [[0, 3, 0, 0, 6], [0, 5, 0, 0, 2], [0, 0, 0, 0, 0], [0, 5, 0, 0, 1], [0, 0, 0, 0, 0]],
    ),
    # Three M's explode around the placed M: 7*3 + the 4, 5 and 1 destroyed = 31
    # (score 356 -> 387). mergedMain gave 21.
    "M_explosion": (
        [[6, 4, 0, 0, 7], [5, 6, 5, 1, 7], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
        [4, 7], [(0, 2), (0, 3)], 31,
        [[6, 4, 0, 0, 0], [5, 6, 0, 0, 0], [0, 6, 1, 0, 4], [5, 0, 4, 0, 0], [2, 2, 1, 1, 0]],
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_ios_recording(name):
    before, vals, coords, points, after = CASES[name]
    board, _, pts = merged_fast.apply(before, set(range(1, 8)), vals, coords)
    assert (board, pts) == (after, points)


@pytest.mark.parametrize("name", sorted(CASES))
def test_domino_orientation_does_not_matter(name):
    before, vals, coords, points, after = CASES[name]
    if len(vals) == 2:
        board, _, pts = merged_fast.apply(before, set(range(1, 8)), vals[::-1], coords[::-1])
        assert (board, pts) == (after, points)
