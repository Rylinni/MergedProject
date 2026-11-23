// TD(0) afterstate training of the n-tuple value network by self-play.
//
// The agent plays greedily w.r.t. points + V(afterstate). After each move the
// previous afterstate is moved towards (points of this move + V(new afterstate));
// the last afterstate of a game is moved towards 0. Threads share one network
// and update it without locks (hogwild), which is standard for n-tuple TD.
//
// Top-score mode (--moments-out): a second network learns E[G^2] of the future
// return G (in units of SCALE points), so std(G) = sqrt(E[G^2] - V^2). With
// --risk L the policy maximises points + V + L * std, i.e. it seeks variance;
// both networks then learn the value of that risk-seeking policy.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "agents.hpp"

using namespace merged;

static void usage() {
    std::fprintf(stderr,
        "usage: merged_train --out FILE [--in FILE] [--games N] [--threads T] [--alpha A]\n"
        "                    [--alpha-end A] [--report N] [--seed S] [--legacy]\n"
        "                    [--moments-in FILE] [--moments-out FILE] [--risk L]\n"
        "  alpha is the total step size (spread over all lookups); it decays\n"
        "  geometrically from --alpha to --alpha-end over the run.\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    std::string out, in, mom_in, mom_out;
    double risk = 0;
    long games = 100000, report = 5000;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    double alpha = 0.1, alpha_end = -1;
    uint64_t seed = 12345;
    Rules R;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--out") out = next();
        else if (a == "--in") in = next();
        else if (a == "--games") games = std::atol(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--alpha") alpha = std::atof(next());
        else if (a == "--alpha-end") alpha_end = std::atof(next());
        else if (a == "--report") report = std::atol(next());
        else if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--legacy") R = Rules::legacy();
        else if (a == "--moments-in") mom_in = next();
        else if (a == "--moments-out") mom_out = next();
        else if (a == "--risk") risk = std::atof(next());
        else usage();
    }
    if (out.empty()) usage();
    if (alpha_end < 0) alpha_end = alpha;
    if (risk != 0 && mom_out.empty()) { std::fprintf(stderr, "--risk needs --moments-out\n"); return 2; }

    NTuple net;
    if (!in.empty() && !net.load(in)) { std::fprintf(stderr, "cannot load %s\n", in.c_str()); return 1; }
    NTuple mom;
    const bool moments = !mom_out.empty();
    if (!mom_in.empty() && !mom.load(mom_in)) { std::fprintf(stderr, "cannot load %s\n", mom_in.c_str()); return 1; }
    RiskEval risk_eval(net, mom, risk);
    std::fprintf(stderr, "n-tuple: %zu tuples, %zu weights (%.1f MB), %d lookups/eval\n", net.tuples.size(),
                 net.num_weights(), net.num_weights() * 4 / 1e6, net.lookups());

    std::atomic<long> next_game{0};
    std::mutex mu;
    double sum_score = 0, sum_moves = 0, best = 0, total_moves = 0;
    long done = 0;
    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Move ms[MAX_MOVES];
        for (long gi; (gi = next_game++) < games;) {
            double frac = double(gi) / double(games);
            float step = float(alpha * std::pow(alpha_end / alpha, frac) / net.lookups());
            Game g(R, seed + uint64_t(gi));
            Board prev{};
            bool have_prev = false;
            for (;;) {
                int n = g.legal(ms);
                if (n == 0) break;
                double bv = -1e300;
                int bi = 0;
                Board bafter{};
                int64_t bpts = 0;
                for (int i = 0; i < n; i++) {
                    Board nb = g.b;
                    int64_t pts = apply(nb, g.next, ms[i], R);
                    double v = double(pts) + (risk != 0 ? risk_eval(nb) : (nb.empty() ? net.value(nb) : 0.0));
                    if (v > bv) { bv = v; bi = i; bafter = nb; bpts = pts; }
                }
                if (have_prev) {
                    double va = bafter.empty() ? net.value(bafter) : 0.0;
                    double target = double(bpts) + va;
                    net.update(prev, float(step * (target - net.value(prev))));
                    if (moments) {
                        double r = double(bpts) / RiskEval::SCALE, vs = va / RiskEval::SCALE;
                        double m2 = r * r + 2 * r * vs + (bafter.empty() ? mom.value(bafter) : 0.0);
                        mom.update(prev, float(step * (m2 - mom.value(prev))));
                    }
                }
                g.play(ms[bi]);
                prev = g.b;
                have_prev = true;
            }
            if (have_prev) {
                net.update(prev, float(step * (0.0 - net.value(prev))));
                if (moments) mom.update(prev, float(step * (0.0 - mom.value(prev))));
            }

            std::lock_guard<std::mutex> lk(mu);
            sum_score += double(g.score);
            sum_moves += g.moves;
            total_moves += g.moves;
            best = std::max(best, double(g.score));
            if (++done % report == 0) {
                double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("games=%ld  mean=%.0f  max=%.0f  moves/game=%.0f  alpha=%.4f  %.0fs  (%.0f moves/s)\n",
                            done, sum_score / report, best, sum_moves / report, step * net.lookups(), secs,
                            total_moves / secs);
                std::fflush(stdout);
                sum_score = sum_moves = best = 0;
                net.save(out);
                if (moments) mom.save(mom_out);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    if (!net.save(out)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    if (moments && !mom.save(mom_out)) { std::fprintf(stderr, "cannot write %s\n", mom_out.c_str()); return 1; }
    std::fprintf(stderr, "saved %s\n", out.c_str());
}
