// Evaluation harness: plays N seeded games with an agent and reports score
// statistics. Game i always uses seed (seed + i), so runs are reproducible.
// Different agents see correlated (not identical) piece sequences, because the
// domino-room check depends on the board.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "agents.hpp"

using namespace merged;

static void usage() {
    std::fprintf(stderr,
        "usage: merged_eval [--agent random|greedy|expectimax] [--depth D] [--games N]\n"
        "                   [--seed S] [--threads T] [--csv FILE]\n"
        "                   [--w-empty X] [--w-pair X]\n"
        "  rules: [--p-double P] [--double-anywhere] [--double-same-ok]\n"
        "         [--max-spawn V] [--weights w1,w2,...,w7]\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    std::string agent = "greedy", csv;
    int depth = 1, games = 1000, threads = std::max(1u, std::thread::hardware_concurrency());
    uint64_t seed = 1;
    Rules R;
    Heuristic H;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--agent") agent = next();
        else if (a == "--depth") depth = std::atoi(next());
        else if (a == "--games") games = std::atoi(next());
        else if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--csv") csv = next();
        else if (a == "--w-empty") H.w_empty = std::atof(next());
        else if (a == "--w-pair") H.w_pair = std::atof(next());
        else if (a == "--p-double") R.p_double = std::atof(next());
        else if (a == "--double-anywhere") R.double_room_interior_only = false;
        else if (a == "--double-same-ok") R.double_distinct = false;
        else if (a == "--max-spawn") R.max_spawn = std::atoi(next());
        else if (a == "--weights") {
            std::string w = next();
            int v = 1;
            for (size_t p = 0; v < EXPLODE && p <= w.size(); v++) {
                size_t q = w.find(',', p);
                R.weight[v] = std::atof(w.substr(p, q - p).c_str());
                if (q == std::string::npos) { v++; break; }
                p = q + 1;
            }
        } else usage();
    }
    if (agent == "greedy") { agent = "expectimax"; depth = 1; }

    std::vector<int64_t> scores(games);
    std::vector<int> lengths(games), top(games);
    std::atomic<int> next_game{0};
    auto t0 = std::chrono::steady_clock::now();
    auto worker = [&]() {
        std::unique_ptr<Agent> ag;
        if (agent == "random") ag.reset(new RandomAgent());
        else if (agent == "expectimax") ag.reset(new ExpectimaxAgent(depth, R, H));
        else usage();
        Move ms[MAX_MOVES];
        for (int i; (i = next_game++) < games;) {
            Game g(R, seed + uint64_t(i));
            Rng arng(~(seed + uint64_t(i)));
            while (g.legal(ms) > 0) g.play(ag->choose(g, arng));
            scores[i] = g.score; lengths[i] = g.moves;
            top[i] = 31 - __builtin_clz(g.b.unlocked);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<int64_t> s = scores;
    std::sort(s.begin(), s.end());
    double mean = 0, var = 0, len = 0;
    for (auto x : s) mean += double(x);
    mean /= games;
    for (auto x : s) var += (x - mean) * (x - mean);
    var /= std::max(1, games - 1);
    for (int l : lengths) len += l;
    auto pct = [&](double q) { return s[std::min<size_t>(s.size() - 1, size_t(q * s.size()))]; };
    int reach[EXPLODE + 1] = {};
    for (int t : top) reach[t]++;

    std::printf("agent=%s depth=%d games=%d seed=%llu\n", agent.c_str(), depth, games, (unsigned long long)seed);
    std::printf("mean=%.1f  sem=%.1f  median=%lld  p90=%lld  p99=%lld  max=%lld\n", mean,
                std::sqrt(var / games), (long long)pct(0.5), (long long)pct(0.9), (long long)pct(0.99),
                (long long)s.back());
    std::printf("moves/game=%.1f  highest unlocked:", len / games);
    for (int v = 2; v <= M; v++) std::printf(" %d:%.1f%%", v, 100.0 * reach[v] / games);
    std::printf("\ntime=%.1fs  (%.2f games/s, %d threads)\n", secs, games / secs, threads);

    if (!csv.empty()) {
        FILE* f = std::fopen(csv.c_str(), "w");
        std::fprintf(f, "game,seed,score,moves,highest_unlocked\n");
        for (int i = 0; i < games; i++)
            std::fprintf(f, "%d,%llu,%lld,%d,%d\n", i, (unsigned long long)(seed + i), (long long)scores[i],
                         lengths[i], top[i]);
        std::fclose(f);
    }
}
