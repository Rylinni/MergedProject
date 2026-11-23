// Baseline agents for the fast engine.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include "merged.hpp"
#include "ntuple.hpp"

namespace merged {

// Estimate of future points from an afterstate (board after a move resolved).
struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double operator()(const Board& b) const = 0;
};

// Hand-written baseline.
struct Heuristic : Evaluator {
    double w_empty = 10.0;  // per empty cell
    double w_pair = 3.0;    // per adjacent equal pair, times value
    double operator()(const Board& b) const override {
        uint32_t e = b.empty();
        if (!e) return 0.0;  // board full => game over (a domino only spawns if it fits)
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

// Learned n-tuple value function.
struct NTupleEval : Evaluator {
    const NTuple& net;
    explicit NTupleEval(const NTuple& n) : net(n) {}
    double operator()(const Board& b) const override { return b.empty() ? net.value(b) : 0.0; }
};

// Risk-seeking evaluator for top scores: V + lambda * std of future points,
// with std from a second network trained on E[(G / SCALE)^2].
struct RiskEval : Evaluator {
    static constexpr double SCALE = 1000.0;
    const NTuple& net;
    const NTuple& moments;
    double lambda;
    RiskEval(const NTuple& v, const NTuple& m, double l) : net(v), moments(m), lambda(l) {}
    double std_dev(const Board& b) const {
        double v = net.value(b) / SCALE, m2 = moments.value(b);
        return std::sqrt(std::max(0.0, m2 - v * v)) * SCALE;
    }
    double operator()(const Board& b) const override {
        if (!b.empty()) return 0.0;
        return net.value(b) + (lambda != 0 ? lambda * std_dev(b) : 0.0);
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
// placements searched (1 = greedy on points + evaluator).
struct ExpectimaxAgent : Agent {
    int depth;
    const Evaluator& h;
    Rules R;
    bool max_objective = false;  // "largest merge" mode: value = biggest single move to come
    ExpectimaxAgent(int d, const Rules& r, const Evaluator& e, bool max_obj = false)
        : depth(d), h(e), R(r), max_objective(max_obj) {}
    double combine(double pts, double future) const { return max_objective ? std::max(pts, future) : pts + future; }

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
                double pts = double(apply(nb, pp[k].p, ms[i], R));  // must run before after_value(nb)
                double v = combine(pts, after_value(nb, d - 1));
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
            double pts = double(apply(nb, g.next, ms[i], R));  // must run before after_value(nb)
            double v = combine(pts, after_value(nb, depth - 1));
            if (v > best + 1e-9) { best = v; bm = ms[i]; ties = 1; }
            else if (v > best - 1e-9 && rng.below(++ties) == 0) bm = ms[i];  // random tie-break
        }
        return bm;
    }
};

}  // namespace merged
