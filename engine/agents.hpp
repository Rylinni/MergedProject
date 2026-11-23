// Baseline agents for the fast engine.
#pragma once
#include <cmath>
#include <limits>

#include "merged.hpp"

namespace merged {

// Hand-written estimate of future points from a position (Phase 1 baseline;
// to be replaced by a learned n-tuple value function).
struct Heuristic {
    double w_empty = 10.0;  // per empty cell
    double w_pair = 3.0;    // per adjacent equal pair, times value
    double operator()(const Board& b) const {
        uint32_t e = b.empty();
        if (!e) return 0.0;  // board full => game over (domino room is guaranteed otherwise)
        double h = w_empty * popcount(e);
        for (int i = 0; i < CELLS; i++) {
            int v = b.c[i];
            if (!v) continue;
            // count right/down neighbours only so each pair is seen once
            if (i % N < N - 1 && b.c[i + 1] == v) h += w_pair * v;
            if (i + N < CELLS && b.c[i + N] == v) h += w_pair * v;
        }
        return h;
    }
};

struct Agent {
    virtual ~Agent() = default;
    virtual Move choose(const Game& g, Rng& rng) = 0;
};

struct RandomAgent : Agent {
    Move choose(const Game& g, Rng& rng) override {
        Move ms[MAX_MOVES];
        int n = g.legal(ms);
        return ms[rng.below(n)];
    }
};

// Expectimax over the real next-piece distribution. depth = number of piece
// placements searched (1 = greedy on points + heuristic).
struct ExpectimaxAgent : Agent {
    int depth;
    Heuristic h;
    Rules R;
    explicit ExpectimaxAgent(int d, const Rules& r, Heuristic hh = Heuristic()) : depth(d), h(hh), R(r) {}

    // Expected future points from an afterstate, searching `d` more placements.
    double after_value(const Board& b, int d) const {
        if (!b.empty()) return 0.0;
        if (d == 0) return h(b);
        PieceProb pp[64];
        int np = piece_distribution(b, R, pp);
        double ev = 0;
        Move ms[MAX_MOVES];
        for (int k = 0; k < np; k++) {
            int n = legal_moves(b, pp[k].p, ms);
            double best = 0;  // no legal move => game over, no more points
            for (int i = 0; i < n; i++) {
                Board nb = b;
                double v = double(apply(nb, pp[k].p, ms[i])) + after_value(nb, d - 1);
                if (i == 0 || v > best) best = v;
            }
            ev += pp[k].prob * best;
        }
        return ev;
    }

    Move choose(const Game& g, Rng& rng) override {
        Move ms[MAX_MOVES];
        int n = g.legal(ms);
        double best = -std::numeric_limits<double>::infinity();
        Move bm = ms[0];
        int ties = 0;
        for (int i = 0; i < n; i++) {
            Board nb = g.b;
            double v = double(apply(nb, g.next, ms[i])) + after_value(nb, depth - 1);
            if (v > best + 1e-9) { best = v; bm = ms[i]; ties = 1; }
            else if (v > best - 1e-9 && rng.below(++ties) == 0) bm = ms[i];  // random tie-break
        }
        return bm;
    }
};

}  // namespace merged
