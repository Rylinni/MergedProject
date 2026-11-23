// The family of boards that allow the largest possible move (70,560 points).
//
// Derivation (see models/README-maxmerge.md): a move scores k! x (tiles merged +
// explosion debris). k <= 6, only with a 2+4 domino: the 2 on S climbs 2->3->4
// using S's three other neighbours, hands off to the 4 on F, which climbs 5->6->M->
// explosion using F's three other neighbours. Groups between moves hold at most 2
// equal tiles, so every merge takes the landing tile + one pair (3 tiles), except
// the hand-off: F + S + a pair of 4s = 4 tiles. Debris can only sit on F's two far
// diagonals: at most an M and a 6, with the 5 opposite S. Maximum: (85 + 13) x 720.
//
// A max board therefore has, for some adjacent S, F in the inner 3x3:
//   S, F empty; around S one tile of a 2 pair, a 3 pair and a 4 pair (any order);
//   F's far neighbour 5, its two side neighbours 6 and M; the far diagonal next to
//   the 6 holds an M, the one next to the M holds a 6; each of the six neighbours
//   has its pair partner on a cell that is not one of these.
#pragma once
#include <array>
#include <vector>

#include "merged.hpp"

namespace merged {

struct MaxLayout {
    uint8_t S, F;
    uint8_t cell[8];   // required cells: 3 around S, F's far/6-side/M-side, two far diagonals
    uint8_t value[8];  // required values
    uint32_t core;     // S, F and the 8 cells (partners must lie outside)
};

inline std::vector<MaxLayout> make_max_layouts() {
    std::vector<MaxLayout> out;
    auto inner = [](int r, int c) { return r >= 1 && r <= 3 && c >= 1 && c <= 3; };
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int sr = 1; sr <= 3; sr++)
        for (int sc = 1; sc <= 3; sc++)
            for (int d = 0; d < 4; d++) {
                int fr = sr + dr[d], fc = sc + dc[d];
                if (!inner(fr, fc)) continue;
                // perpendicular direction
                int pr = dc[d], pc = dr[d];
                int S = sr * N + sc, F = fr * N + fc;
                int sOpp = (sr - dr[d]) * N + (sc - dc[d]);
                int sL = (sr + pr) * N + (sc + pc), sR = (sr - pr) * N + (sc - pc);
                int far = (fr + dr[d]) * N + (fc + dc[d]);
                int fL = (fr + pr) * N + (fc + pc), fR = (fr - pr) * N + (fc - pc);
                int dL = (fr + dr[d] + pr) * N + (fc + dc[d] + pc), dR = (fr + dr[d] - pr) * N + (fc + dc[d] - pc);
                const int sN[3] = {sOpp, sL, sR};
                const int perms[6][3] = {{2, 3, 4}, {2, 4, 3}, {3, 2, 4}, {3, 4, 2}, {4, 2, 3}, {4, 3, 2}};
                for (auto& pm : perms)
                    for (int side = 0; side < 2; side++) {
                        MaxLayout m;
                        m.S = uint8_t(S); m.F = uint8_t(F);
                        int six = side ? fL : fR, em = side ? fR : fL;          // F's 6 side and M side
                        int dSix = side ? dL : dR, dEm = side ? dR : dL;        // diagonals next to them
                        const int cells[8] = {sN[0], sN[1], sN[2], far, six, em, dSix, dEm};
                        const int vals[8] = {pm[0], pm[1], pm[2], 5, 6, M, M, 6};
                        m.core = (1u << S) | (1u << F);
                        for (int k = 0; k < 8; k++) { m.cell[k] = uint8_t(cells[k]); m.value[k] = uint8_t(vals[k]); m.core |= 1u << cells[k]; }
                        out.push_back(m);
                    }
            }
    return out;
}

inline const std::vector<MaxLayout>& max_layouts() {
    static const std::vector<MaxLayout> L = make_max_layouts();
    return L;
}

// Progress towards any max board, in [0, 1]. Per layout: the 8 required tiles score
// their value (higher tiles are harder to make), each of the six pair-tiles with its
// partner outside the core scores +2, and a wrong tile on S, F or a required cell
// costs as much as the value it blocks. The best layout counts.
inline double max_progress(const Board& b) {
    static const double kMax = (2 + 3 + 4 + 5 + 6 + 7 + 7 + 6) + 6 * 2;
    double best = 0;
    for (const MaxLayout& m : max_layouts()) {
        double s = 0;
        if (b.c[m.S]) s -= 8;
        if (b.c[m.F]) s -= 8;
        for (int k = 0; k < 8; k++) {
            int v = b.c[m.cell[k]], want = m.value[k];
            if (v == want) {
                s += want;
                if (k < 6) {
                    uint32_t partner = T.nb[m.cell[k]] & ~m.core;
                    for (uint32_t q = partner; q; q &= q - 1)
                        if (b.c[__builtin_ctz(q)] == want) { s += 2; break; }
                }
            } else if (v) s -= want;
        }
        if (s > best) best = s;
    }
    return best / kMax;
}

// True if some placement of a 2+4 domino scores the maximum now.
inline bool max_ready(const Board& b, const Rules& R) {
    Piece p; p.n = 2; p.v[0] = 2; p.v[1] = 4;
    Move ms[MAX_MOVES];
    int n = legal_moves(b, p, ms);
    for (int i = 0; i < n; i++) {
        Board nb = b;
        if (apply(nb, p, ms[i], R) >= 70560) return true;
    }
    return false;
}

}  // namespace merged
