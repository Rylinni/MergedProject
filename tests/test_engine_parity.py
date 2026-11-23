"""The C++ engine must reproduce mergedMain.py exactly.

Run: python -m pytest tests/test_engine_parity.py
Set PARITY_CASES to change how many random positions are checked.
"""
import collections
import copy
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import mergedMain  # noqa: E402
import merged_fast  # noqa: E402

CASES = int(os.environ.get("PARITY_CASES", "20000"))


def py_apply(board, unlocks, move):
    g = mergedMain.mergeGame(board=copy.deepcopy(board), score=0, unlocks=list(unlocks))
    g.playMove(move)
    return g.board, set(g.unlocks), g.score


def py_legal(board, piece):
    g = mergedMain.mergeGame(board=copy.deepcopy(board), score=0, unlocks=[1, 2])
    return {tuple(tuple(c) for c in coords) for _, coords in g.findLegalMoves(piece)}


def check(board, unlocks, piece):
    moves = py_legal(board, piece)
    assert moves == merged_fast.legal_moves(board, piece), (board, piece)
    for coords in random.sample(sorted(moves), min(4, len(moves))):
        want = py_apply(board, unlocks, (list(piece), list(coords)))
        got = merged_fast.apply(board, unlocks, list(piece), list(coords))
        assert got == want, (board, unlocks, piece, coords, got, want)


def test_unit_cases_from_test_merge():
    # Same boards as test_merge.py, with the placed cells emptied first.
    cases = [
        ([[1, 1, 1, 0, 0]] + [[0] * 5] * 4, ([1], [(0, 0)]), 3),
        ([[0, 1, 1, 1, 0], [0, 2, 2, 2, 0]] + [[0] * 5] * 3, ([1, 2], [(0, 3), (1, 3)]), 6),
        ([[0, 2, 2, 0, 0], [3, 3, 0, 4, 4], [5, 5, 0, 6, 6], [0, 7, 7, 0, 0], [0] * 5],
         ([2, 4], [(1, 2), (2, 2)]), 61200),
    ]
    for board, (vals, coords), score in cases:
        for (x, y) in coords:
            board[x][y] = 0
        _, _, pts = merged_fast.apply(board, {1, 2}, vals, coords)
        assert pts == score
        assert py_apply(board, [1, 2], (vals, coords))[2] == score


def test_random_play_positions():
    rng = random.Random(1234)
    random.seed(1234)
    n = 0
    while n < CASES // 2:
        g = mergedMain.mergeGame()
        while g.findLegalMoves():
            check(g.board, set(g.unlocks), list(g.nextPiece))
            assert merged_fast.room_for_double(g.board) == g.roomForDouble()
            n += 1
            g.playMove(rng.choice(g.findLegalMoves()))


def test_synthetic_positions():
    # Dense random boards with high values hit chains, partner merges and explosions.
    rng = random.Random(99)
    random.seed(99)
    for _ in range(CASES // 2):
        fill = rng.choice([0.4, 0.6, 0.75, 0.9])
        top = rng.randint(2, 7)
        board = [[rng.randint(1, top) if rng.random() < fill else 0 for _ in range(5)] for _ in range(5)]
        if not any(0 in row for row in board):
            board[rng.randrange(5)][rng.randrange(5)] = 0
        unlocks = {1, 2} | set(rng.sample(range(3, 8), rng.randint(0, 5)))
        if rng.random() < 0.5:
            piece = [rng.randint(1, 7)]
        else:
            a = rng.randint(1, 7)
            piece = [a, rng.randint(1, 7) if rng.random() < 0.15 else rng.choice([v for v in range(1, 8) if v != a])]
        check(board, unlocks, piece)
        assert merged_fast.room_for_double(board) == mergedMain.mergeGame(board=copy.deepcopy(board)).roomForDouble()


def test_spawn_distribution():
    board = [[0, 3, 0, 0, 0], [0, 1, 1, 0, 0], [0] * 5, [2, 0, 0, 0, 0], [0] * 5]
    unlocks = {1, 2, 3, 4, 5}
    trials = 40000
    random.seed(7)
    py = collections.Counter()
    for _ in range(trials):
        g = mergedMain.mergeGame(board=copy.deepcopy(board), score=0, unlocks=list(unlocks))
        py[tuple(sorted(g.nextPiece))] += 1
    cpp = merged_fast.spawn_counts(board, unlocks, trials, seed=7)
    exact = merged_fast.piece_distribution(board, unlocks)
    assert abs(sum(exact.values()) - 1) < 1e-12
    assert set(py) == set(cpp) == set(exact)
    for k, p in exact.items():
        sd = (p * (1 - p) / trials) ** 0.5
        assert abs(py[k] / trials - p) < 5 * sd, (k, py[k] / trials, p)
        assert abs(cpp[k] / trials - p) < 5 * sd, (k, cpp[k] / trials, p)
