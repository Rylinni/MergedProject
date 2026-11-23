// Fast Merged! engine. A bit-exact port of mergedMain.py (verified by
// tests/test_engine_parity.py), with the uncertain parts of the iOS rules
// exposed as Rules so they can be changed once confirmed against the real game.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace merged {

constexpr int N = 5;
constexpr int CELLS = N * N;
constexpr int M = 7;        // the "M" block
constexpr int EXPLODE = 8;  // three M's merged -> 3x3 explosion
constexpr uint8_t NONE = 255;
constexpr int MAX_MOVES = 100;  // 25 singles or <= 80 ordered domino placements

struct Tables {
    uint32_t nb[CELLS];   // orthogonal neighbours
    uint32_t box[CELLS];  // 3x3 explosion area, clipped to the board
    uint32_t interior;    // rows/cols 1..3 (legacy roomForDouble scan area)
    Tables() : interior(0) {
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++) {
                int i = x * N + y;
                nb[i] = box[i] = 0;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
                        box[i] |= 1u << (nx * N + ny);
                        if ((dx == 0) != (dy == 0)) nb[i] |= 1u << (nx * N + ny);
                    }
                if (x >= 1 && x <= 3 && y >= 1 && y <= 3) interior |= 1u << i;
            }
    }
};
inline const Tables T;

// Rules that differ between mergedMain.py and the iOS game, or that are still
// unconfirmed. Defaults follow what screen recordings of the iOS game showed
// (see tests/test_ios_observations.py); Rules::legacy() reproduces mergedMain.py.
struct Rules {
    // Scoring, confirmed from recordings:
    bool bank_smaller_half = true;       // domino whose halves merge separately: both score
                                         // (mergedMain discarded the smaller half's points)
    bool explosion_scores_cleared = true;  // tiles destroyed by an M explosion score their value
    // Spawning (M spawning and ~50% dominoes are consistent with recordings; the rest unconfirmed):
    double p_double = 0.5;               // chance the next piece is a domino (if there is room)
    bool double_room_interior_only = true;  // mergedMain quirk: only interior cells are checked for domino room
    bool double_distinct = true;         // mergedMain: domino halves always differ
    int max_spawn = M;                   // highest value that can spawn (once unlocked); 6 = M never spawns
    double weight[EXPLODE] = {0, 1, 1, 1, 1, 1, 1, 1};  // relative spawn weight per value

    static Rules legacy() {
        Rules r;
        r.bank_smaller_half = false;
        r.explosion_scores_cleared = false;
        return r;
    }
};

struct Piece {
    uint8_t n = 1;
    uint8_t v[2] = {1, 0};
};
struct Move {
    uint8_t a = NONE, b = NONE;  // v[0] goes to a, v[1] to b (b == NONE for singles)
};

struct Board {
    uint8_t c[CELLS];
    uint8_t unlocked;  // bit v set => v has been reached (mergedMain "unlocks")
    void clear() {
        std::memset(c, 0, sizeof c);
        unlocked = (1 << 1) | (1 << 2);
    }
    uint32_t mask_eq(int v) const {
        uint32_t m = 0;
        for (int i = 0; i < CELLS; i++) m |= uint32_t(c[i] == v) << i;
        return m;
    }
    uint32_t empty() const { return mask_eq(0); }
};

inline int popcount(uint32_t x) { return __builtin_popcount(x); }

// Connected same-valued group containing `start` (start always included),
// never stepping onto `exclude` (-1 = no exclusion). Mirrors getIdenticals.
inline uint32_t flood(const Board& b, int value, int start, int exclude) {
    uint32_t same = b.mask_eq(value);
    if (exclude >= 0) same &= ~(1u << exclude);
    uint32_t comp = 1u << start, frontier = comp;
    while (frontier) {
        uint32_t nxt = 0;
        for (uint32_t f = frontier; f; f &= f - 1) nxt |= T.nb[__builtin_ctz(f)];
        nxt &= same & ~comp;
        comp |= nxt;
        frontier = nxt;
    }
    return comp;
}

struct Resolver {
    Board& b;
    const Rules& R;
    int64_t withheld = 0, mult = 1;
    int merges = 0;
    Resolver(Board& bd, const Rules& r) : b(bd), R(r) {}

    void upgrade(int p) {
        b.c[p]++;
        merges++;
        mult *= merges;
        if (b.c[p] == EXPLODE) {
            for (uint32_t m = T.box[p]; m; m &= m - 1) {
                int i = __builtin_ctz(m);
                if (R.explosion_scores_cleared && i != p) withheld += b.c[i];
                b.c[i] = 0;
            }
        } else {
            b.unlocked |= 1 << b.c[p];
        }
    }
    // Mirrors evalPiece. pcell < 0 means no partner.
    bool eval_piece(int value, int coor, int pcell, int pval, bool first) {
        uint32_t comp = flood(b, value, coor, pcell);
        if (pcell >= 0 && popcount(comp) < 3) comp = flood(b, value, coor, -1);
        if (popcount(comp) >= 3 && !(pcell >= 0 && first && pval == value)) {
            for (uint32_t m = comp; m; m &= m - 1) {
                int i = __builtin_ctz(m);
                withheld += b.c[i];
                if (i != coor) b.c[i] = 0;
            }
            upgrade(coor);
            return true;
        }
        return false;
    }
};

// Place `p` with `mv` on `b` (cells must be empty) and resolve merges.
// Returns points scored. Mirrors playMove + evalBoard.
inline int64_t apply(Board& b, const Piece& p, Move mv, const Rules& R) {
    Resolver r(b, R);
    int64_t banked = 0;
    if (p.n == 1) {
        b.c[mv.a] = p.v[0];
        int v = p.v[0];
        while (r.eval_piece(v, mv.a, -1, 0, false)) v++;
    } else {
        b.c[mv.a] = p.v[0];
        b.c[mv.b] = p.v[1];
        int sv, sc, bv, bc;
        if (p.v[0] < p.v[1]) { sv = p.v[0]; sc = mv.a; bv = p.v[1]; bc = mv.b; }
        else                 { sv = p.v[1]; sc = mv.b; bv = p.v[0]; bc = mv.a; }
        bool bigger_can_eval = popcount(flood(b, bv, bc, -1)) > 2;
        int v = sv;
        bool up = r.eval_piece(v, sc, -1, 0, false);
        while (up) { v++; up = r.eval_piece(v, sc, bc, bv, true); }
        if (bigger_can_eval) {
            if (R.bank_smaller_half) banked = r.mult * r.withheld;
            r.withheld = 0; r.mult = 1; r.merges = 0;
        }
        if (b.c[bc] == bv) {
            v = bv;
            up = r.eval_piece(v, bc, sc, sv, false);
            while (up) { v++; up = r.eval_piece(v, bc, sc, sv, false); }
        }
    }
    return banked + r.mult * r.withheld;
}

inline bool room_for_double(const Board& b, const Rules& R) {
    uint32_t e = b.empty();
    uint32_t cand = R.double_room_interior_only ? (e & T.interior) : e;
    for (; cand; cand &= cand - 1)
        if (T.nb[__builtin_ctz(cand)] & e) return true;
    return false;
}

inline int legal_moves(const Board& b, const Piece& p, Move* out) {
    uint32_t e = b.empty();
    int n = 0;
    for (uint32_t m = e; m; m &= m - 1) {
        int a = __builtin_ctz(m);
        if (p.n == 1) { out[n++] = {uint8_t(a), NONE}; continue; }
        for (uint32_t q = T.nb[a] & e; q; q &= q - 1) out[n++] = {uint8_t(a), uint8_t(__builtin_ctz(q))};
    }
    return n;
}

struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed = 1) { reseed(seed); }
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    void reseed(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform() { return (next() >> 11) * 0x1.0p-53; }
    int below(int n) { return int(uniform() * n); }
};

// Values that may currently spawn, as a bitmask.
inline uint32_t spawnable(const Board& b, const Rules& R) {
    uint32_t m = 0;
    for (int v = 1; v <= std::min(R.max_spawn, M); v++)
        if ((b.unlocked >> v & 1) && R.weight[v] > 0) m |= 1u << v;
    return m;
}

inline int weighted_pick(uint32_t allowed, const Rules& R, Rng& rng) {
    double tot = 0;
    for (uint32_t m = allowed; m; m &= m - 1) tot += R.weight[__builtin_ctz(m)];
    double r = rng.uniform() * tot;
    int last = 0;
    for (uint32_t m = allowed; m; m &= m - 1) {
        last = __builtin_ctz(m);
        r -= R.weight[last];
        if (r < 0) return last;
    }
    return last;
}

inline Piece spawn(const Board& b, const Rules& R, Rng& rng) {
    uint32_t allowed = spawnable(b, R);
    Piece p;
    p.n = 1;
    p.v[0] = uint8_t(weighted_pick(allowed, R, rng));
    p.v[1] = 0;
    if (rng.uniform() < R.p_double && room_for_double(b, R)) {
        uint32_t rest = R.double_distinct ? (allowed & ~(1u << p.v[0])) : allowed;
        if (rest) { p.n = 2; p.v[1] = uint8_t(weighted_pick(rest, R, rng)); }
    }
    return p;
}

// Exact distribution of the next piece given the board after a move.
// Dominoes are returned once per unordered pair (both orientations are legal moves anyway).
struct PieceProb { Piece p; double prob; };
inline int piece_distribution(const Board& b, const Rules& R, PieceProb* out) {
    uint32_t allowed = spawnable(b, R);
    double W = 0;
    for (uint32_t m = allowed; m; m &= m - 1) W += R.weight[__builtin_ctz(m)];
    bool room = room_for_double(b, R);
    double pd = room ? R.p_double : 0.0;
    double pair[EXPLODE][EXPLODE] = {};
    int n = 0;
    for (uint32_t m = allowed; m; m &= m - 1) {
        int a = __builtin_ctz(m);
        double pa = R.weight[a] / W;
        uint32_t rest = R.double_distinct ? (allowed & ~(1u << a)) : allowed;
        double Wr = 0;
        for (uint32_t q = rest; q; q &= q - 1) Wr += R.weight[__builtin_ctz(q)];
        double single = pa * (1 - pd);
        if (!rest) single = pa;  // cannot form a domino: stays single
        else
            for (uint32_t q = rest; q; q &= q - 1) {
                int c = __builtin_ctz(q);
                pair[std::min(a, c)][std::max(a, c)] += pd * pa * R.weight[c] / Wr;
            }
        Piece p; p.n = 1; p.v[0] = uint8_t(a); p.v[1] = 0;
        out[n++] = {p, single};
    }
    for (int a = 1; a < EXPLODE; a++)
        for (int c = a; c < EXPLODE; c++)
            if (pair[a][c] > 0) {
                Piece p; p.n = 2; p.v[0] = uint8_t(a); p.v[1] = uint8_t(c);
                out[n++] = {p, pair[a][c]};
            }
    return n;
}

struct Game {
    Board b;
    Piece next;
    int64_t score = 0;
    int moves = 0;
    Rules R;
    Rng rng;
    explicit Game(const Rules& rules = Rules(), uint64_t seed = 1) : R(rules), rng(seed) { reset(seed); }
    void reset(uint64_t seed) {
        rng.reseed(seed);
        b.clear();
        score = 0;
        moves = 0;
        next = spawn(b, R, rng);
    }
    int legal(Move* out) const { return legal_moves(b, next, out); }
    int64_t play(Move mv) {
        int64_t pts = apply(b, next, mv, R);
        score += pts;
        moves++;
        next = spawn(b, R, rng);
        return pts;
    }
};

}  // namespace merged
