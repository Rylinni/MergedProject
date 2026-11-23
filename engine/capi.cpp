// C ABI for Python (ctypes). See merged_fast.py.
#include "merged.hpp"

using namespace merged;

static Piece make_piece(int n, const uint8_t* vals) {
    Piece p;
    p.n = uint8_t(n);
    p.v[0] = vals[0];
    p.v[1] = n == 2 ? vals[1] : 0;
    return p;
}

extern "C" {

// Place a piece (n = 1 or 2 values at cell indices pos[]) and resolve merges.
// Returns points scored; writes the new board and unlocked bitmask.
// legacy != 0 uses mergedMain.py scoring, otherwise the iOS-observed rules.
long long mg_apply(const uint8_t* cells, uint8_t unlocked, int n, const uint8_t* vals, const uint8_t* pos,
                   uint8_t* cells_out, uint8_t* unlocked_out, int legacy) {
    Board b;
    for (int i = 0; i < CELLS; i++) b.c[i] = cells[i];
    b.unlocked = unlocked;
    Move mv{pos[0], n == 2 ? pos[1] : uint8_t(0)};  // b unused for singles
    long long pts = apply(b, make_piece(n, vals), mv, legacy ? Rules::legacy() : Rules());
    for (int i = 0; i < CELLS; i++) cells_out[i] = b.c[i];
    *unlocked_out = b.unlocked;
    return pts;
}

// Legal placements: writes pairs (a, b) into out (b = 255 for singles), returns count.
int mg_legal(const uint8_t* cells, int n, const uint8_t* vals, uint8_t* out) {
    Board b;
    for (int i = 0; i < CELLS; i++) b.c[i] = cells[i];
    b.unlocked = 0;
    Move ms[MAX_MOVES];
    int k = legal_moves(b, make_piece(n, vals), ms);
    for (int i = 0; i < k; i++) { out[2 * i] = ms[i].a; out[2 * i + 1] = ms[i].b; }
    return k;
}

int mg_room_for_double(const uint8_t* cells, int interior_only) {
    Board b;
    for (int i = 0; i < CELLS; i++) b.c[i] = cells[i];
    Rules R;
    R.double_room_interior_only = interior_only != 0;
    return room_for_double(b, R);
}

// Sample `count` pieces with default rules; counts[v0 * 8 + v1] (v0 <= v1; v1 = 0 for singles).
void mg_spawn_counts(const uint8_t* cells, uint8_t unlocked, unsigned long long seed, int count, int* counts) {
    Board b;
    for (int i = 0; i < CELLS; i++) b.c[i] = cells[i];
    b.unlocked = unlocked;
    Rules R;
    Rng rng(seed);
    for (int i = 0; i < count; i++) {
        Piece p = spawn(b, R, rng);
        int a = p.v[0], c = p.n == 2 ? p.v[1] : 0;
        if (c && c < a) std::swap(a, c);
        counts[a * 8 + c]++;
    }
}

// Exact next-piece distribution with default rules; same key layout as mg_spawn_counts.
void mg_piece_distribution(const uint8_t* cells, uint8_t unlocked, double* probs) {
    Board b;
    for (int i = 0; i < CELLS; i++) b.c[i] = cells[i];
    b.unlocked = unlocked;
    PieceProb pp[64];
    int n = piece_distribution(b, Rules(), pp);
    for (int i = 0; i < n; i++) probs[pp[i].p.v[0] * 8 + (pp[i].p.n == 2 ? pp[i].p.v[1] : 0)] += pp[i].prob;
}
}
