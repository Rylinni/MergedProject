"""ctypes wrapper around engine/build/libmerged.so (build with `make -C engine`).

Boards are 5x5 lists of ints (as in mergedMain); unlocks are sets of values.
"""
import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "build", "libmerged.so")


def _load():
    if not os.path.exists(_LIB_PATH):
        subprocess.run(["make", "-C", _HERE, "build/libmerged.so"], check=True)
    lib = ctypes.CDLL(_LIB_PATH)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.mg_apply.restype = ctypes.c_longlong
    lib.mg_apply.argtypes = [u8p, ctypes.c_uint8, ctypes.c_int, u8p, u8p, u8p, u8p, ctypes.c_int]
    lib.mg_legal.restype = ctypes.c_int
    lib.mg_legal.argtypes = [u8p, ctypes.c_int, u8p, u8p]
    lib.mg_room_for_double.restype = ctypes.c_int
    lib.mg_room_for_double.argtypes = [u8p, ctypes.c_int]
    lib.mg_spawn_counts.restype = None
    lib.mg_spawn_counts.argtypes = [u8p, ctypes.c_uint8, ctypes.c_ulonglong, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    lib.mg_piece_distribution.restype = None
    lib.mg_piece_distribution.argtypes = [u8p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_double)]
    return lib


_lib = _load()


def _cells(board):
    return (ctypes.c_uint8 * 25)(*[v for row in board for v in row])


def _unlock_mask(unlocks):
    m = 0
    for v in unlocks:
        m |= 1 << v
    return m


def _unlock_set(mask):
    return {v for v in range(8) if mask >> v & 1}


def apply(board, unlocks, values, coords, rules="ios"):
    """Place piece `values` at (row, col) `coords`; return (board, unlocks, points).

    rules="ios" uses the scoring seen in the real game; "legacy" matches mergedMain.py.
    """
    assert rules in ("ios", "legacy")
    n = len(values)
    out = (ctypes.c_uint8 * 25)()
    unl = ctypes.c_uint8()
    pts = _lib.mg_apply(_cells(board), _unlock_mask(unlocks), n, (ctypes.c_uint8 * 2)(*values, *([0] * (2 - n))),
                        (ctypes.c_uint8 * 2)(*[x * 5 + y for x, y in coords], *([0] * (2 - n))),
                        out, ctypes.byref(unl), int(rules == "legacy"))
    return [list(out[r * 5:(r + 1) * 5]) for r in range(5)], _unlock_set(unl.value), pts


def legal_moves(board, values):
    """Set of placements as tuples of (row, col), first entry gets values[0]."""
    n = len(values)
    out = (ctypes.c_uint8 * 200)()
    k = _lib.mg_legal(_cells(board), n, (ctypes.c_uint8 * 2)(*values, *([0] * (2 - n))), out)
    res = set()
    for i in range(k):
        a, b = out[2 * i], out[2 * i + 1]
        res.add(((a // 5, a % 5),) if b == 255 else ((a // 5, a % 5), (b // 5, b % 5)))
    return res


def room_for_double(board, interior_only=True):
    return bool(_lib.mg_room_for_double(_cells(board), int(interior_only)))


def _decode(arr):
    out = {}
    for k, v in enumerate(arr):
        if v:
            a, b = divmod(k, 8)
            out[(a,) if b == 0 else (a, b)] = v
    return out


def spawn_counts(board, unlocks, n, seed=1):
    counts = (ctypes.c_int * 64)()
    _lib.mg_spawn_counts(_cells(board), _unlock_mask(unlocks), seed, n, counts)
    return _decode(counts)


def piece_distribution(board, unlocks):
    probs = (ctypes.c_double * 64)()
    _lib.mg_piece_distribution(_cells(board), _unlock_mask(unlocks), probs)
    return _decode(probs)
