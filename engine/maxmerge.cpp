// Search for the largest single-move score possible under the iOS rules.
//
// Simulated annealing over boards. A board is allowed only if it could exist
// between moves: no connected group of 3+ equal tiles (it would already have
// merged). Its value is the best points over every piece the game can deal
// (singles 1..M, dominoes of two different values) and every placement.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "merged.hpp"

using namespace merged;

static bool stable(const Board& b) {
    uint32_t seen = 0;
    for (int i = 0; i < CELLS; i++) {
        if (!b.c[i] || (seen >> i & 1)) continue;
        uint32_t g = flood(b, b.c[i], i, -1);
        if (popcount(g) >= 3) return false;
        seen |= g;
    }
    return true;
}

struct Best { int64_t pts = -1; Piece p; Move m; };

static Best best_move(const Board& b, const Rules& R) {
    Best best;
    Move ms[MAX_MOVES];
    for (int v0 = 1; v0 <= M; v0++)
        for (int v1 = 0; v1 <= M; v1++) {
            if (v1 == v0) continue;
            Piece p;
            p.n = v1 ? 2 : 1;
            p.v[0] = uint8_t(v0);
            p.v[1] = uint8_t(v1);
            int n = legal_moves(b, p, ms);
            for (int i = 0; i < n; i++) {
                Board nb = b;
                int64_t pts = apply(nb, p, ms[i], R);
                if (pts > best.pts) { best.pts = pts; best.p = p; best.m = ms[i]; }
            }
        }
    return best;
}

static void print_board(const Board& b, const Best& bm) {
    for (int x = 0; x < N; x++) {
        std::printf("   ");
        for (int y = 0; y < N; y++) {
            int i = x * N + y;
            char ch = b.c[i] == 0 ? '.' : b.c[i] == M ? 'M' : char('0' + b.c[i]);
            bool here = i == bm.m.a || i == bm.m.b;
            std::printf(here ? "[%c]" : " %c ", here ? (i == bm.m.a ? (bm.p.v[0] == M ? 'M' : char('0' + bm.p.v[0]))
                                                               : (bm.p.v[1] == M ? 'M' : char('0' + bm.p.v[1]))) : ch);
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    long steps = argc > 1 ? std::atol(argv[1]) : 2000000;
    int restarts = argc > 2 ? std::atoi(argv[2]) : 8;
    uint64_t seed = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;
    Rules R;
    Rng rng(seed);
    Board global{};
    Best gbest;
    for (int r = 0; r < restarts; r++) {
        Board b;
        b.clear();
        b.unlocked = 0xFE;
        Best cur = best_move(b, R);
        Board bestb = b;
        Best bb = cur;
        for (long s = 0; s < steps; s++) {
            double T = 2000.0 * std::pow(0.5 / 2000.0, double(s) / double(steps));
            Board nb = b;
            int k = 1 + rng.below(2);
            for (int j = 0; j < k; j++) nb.c[rng.below(CELLS)] = uint8_t(rng.below(M + 1));
            if (!stable(nb)) continue;
            Best nbm = best_move(nb, R);
            if (nbm.pts < 0) continue;
            double d = double(nbm.pts - cur.pts);
            if (d >= 0 || rng.uniform() < std::exp(d / T)) {
                b = nb;
                cur = nbm;
                if (cur.pts > bb.pts) { bb = cur; bestb = b; }
            }
        }
        std::printf("restart %d: best %lld\n", r, (long long)bb.pts);
        std::fflush(stdout);
        if (bb.pts > gbest.pts) { gbest = bb; global = bestb; }
    }
    Board after = global;
    apply(after, gbest.p, gbest.m, R);
    std::printf("\nLargest single move found: %lld points\n", (long long)gbest.pts);
    std::printf("Board before (piece placed in [ ]):\n");
    print_board(global, gbest);
    if (gbest.p.n == 2) std::printf("Piece: domino %d + %d\n", gbest.p.v[0], gbest.p.v[1]);
    else std::printf("Piece: single %d\n", gbest.p.v[0]);
    Best none;
    none.m = Move{};
    std::printf("Board after:\n");
    print_board(after, none);
}
