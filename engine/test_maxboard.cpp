// Checks the max-board characterisation in maxboard.hpp against the engine.
#include <cstdio>
#include "maxboard.hpp"
using namespace merged;

static bool stable(const Board& b) {
    for (int i = 0; i < CELLS; i++) if (b.c[i] && popcount(flood(b, b.c[i], i, -1)) >= 3) return false;
    return true;
}

int main() {
    Rules R;
    // 1) The 70,560 board found by annealing must have full progress and be ready.
    const char* rows[5] = {"M35.5", ".65M3", "MM.66", "33.22", "52441"};
    Board b; b.clear();
    for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) { char ch = rows[r][c]; b.c[r * 5 + c] = ch == '.' ? 0 : ch == 'M' ? 7 : ch - '0'; }
    std::printf("annealed board: progress %.3f, ready %d\n", max_progress(b), int(max_ready(b, R)));

    // 2) Random boards built from the description must all allow 70,560.
    Rng rng(42);
    const auto& L = max_layouts();
    long built = 0, ready = 0, full = 0;
    for (int trial = 0; built < 20000 && trial < 2000000; trial++) {
        const MaxLayout& m = L[rng.below(int(L.size()))];
        Board x; x.clear();
        for (int k = 0; k < 8; k++) x.c[m.cell[k]] = m.value[k];
        bool ok = true;
        for (int k = 0; k < 6 && ok; k++) {  // a partner next to each of the six pair tiles
            uint32_t opts = T.nb[m.cell[k]] & ~m.core & x.empty();
            if (!opts) { ok = false; break; }
            int n = popcount(opts), pick = rng.below(n);
            for (uint32_t q = opts; q; q &= q - 1) if (pick-- == 0) { x.c[__builtin_ctz(q)] = m.value[k]; break; }
        }
        if (!ok) continue;
        double fill = rng.uniform();
        for (int i = 0; i < CELLS; i++)
            if (!x.c[i] && !(m.core >> i & 1) && rng.uniform() < fill) x.c[i] = uint8_t(1 + rng.below(7));
        if (!stable(x)) continue;
        built++;
        if (max_ready(x, R)) ready++;
        if (max_progress(x) > 0.9999) full++;
    }
    std::printf("random boards built from the description: %ld, ready for 70,560: %ld, progress 1.0: %ld\n", built, ready, full);
}
