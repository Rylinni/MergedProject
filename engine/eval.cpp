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
        "usage: merged_eval [--agent random|greedy|expectimax|ntuple] [--depth D] [--games N]\n"
        "                   [--weights FILE]   (n-tuple weights for --agent ntuple)\n"
        "                   [--moments FILE --risk L]  (top-score mode: maximise V + L*std)\n"
        "                   [--objective sum|max]      (max: weights trained for the largest merge)\n"
        "                   [--record-by score|merge]  (which game --record keeps)\n"
        "                   [--seed S] [--threads T] [--csv FILE] [--record FILE]\n"
        "  --record writes the best game's full move list as JSON\n"
        "                   [--w-empty X] [--w-pair X]\n"
        "  rules: [--legacy-scoring] [--p-double P] [--double-anywhere] [--double-same-ok]\n"
        "         [--double-room interior|adjacent|two-empty] [--max-spawn V] [--spawn-weights w1,w2,...,w7]\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    std::string agent = "greedy", csv, weights, moments, record;
    double risk = 0;
    bool max_obj = false, record_by_merge = false;
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
        else if (a == "--record") record = next();
        else if (a == "--weights") weights = next();
        else if (a == "--moments") moments = next();
        else if (a == "--risk") risk = std::atof(next());
        else if (a == "--objective") { std::string o = next(); if (o == "max") max_obj = true; else if (o != "sum") usage(); }
        else if (a == "--record-by") { std::string o = next(); if (o == "merge") record_by_merge = true; else if (o != "score") usage(); }
        else if (a == "--w-empty") H.w_empty = std::atof(next());
        else if (a == "--w-pair") H.w_pair = std::atof(next());
        else if (a == "--legacy-scoring") { R.bank_smaller_half = false; R.explosion_scores_cleared = false; }
        else if (a == "--legacy") R = Rules::legacy();
        else if (a == "--p-double") R.p_double = std::atof(next());
        else if (a == "--double-anywhere") R.double_room = Rules::DoubleRoom::AdjacentPair;
        else if (a == "--double-room") {
            std::string m = next();
            if (m == "interior") R.double_room = Rules::DoubleRoom::InteriorAdjacent;
            else if (m == "adjacent") R.double_room = Rules::DoubleRoom::AdjacentPair;
            else if (m == "two-empty") R.double_room = Rules::DoubleRoom::TwoEmpty;
            else usage();
        }
        else if (a == "--double-same-ok") R.double_distinct = false;
        else if (a == "--max-spawn") R.max_spawn = std::atoi(next());
        else if (a == "--spawn-weights") {
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
    NTuple net, mom;
    std::unique_ptr<Evaluator> ev(new Heuristic(H));
    if (agent == "ntuple") {
        if (weights.empty() || !net.load(weights)) { std::fprintf(stderr, "cannot load --weights %s\n", weights.c_str()); return 1; }
        ev.reset(new NTupleEval(net));
        if (!moments.empty()) {
            if (!mom.load(moments)) { std::fprintf(stderr, "cannot load --moments %s\n", moments.c_str()); return 1; }
            ev.reset(new RiskEval(net, mom, risk));
        }
    }

    std::vector<int64_t> scores(games), bigs(games, 0);
    // Per-game move log (only with --record): piece values, cells, points.
    struct Rec { uint8_t v0, v1, a, b; int64_t pts; };
    std::vector<std::vector<Rec>> logs(record.empty() ? 0 : games);
    std::vector<int> lengths(games), top(games);
    std::atomic<int> next_game{0};
    auto t0 = std::chrono::steady_clock::now();
    auto worker = [&]() {
        std::unique_ptr<Agent> ag;
        if (agent == "random") ag.reset(new RandomAgent());
        else if (agent == "expectimax" || agent == "ntuple") ag.reset(new ExpectimaxAgent(depth, R, *ev, max_obj));
        else usage();
        Move ms[MAX_MOVES];
        for (int i; (i = next_game++) < games;) {
            Game g(R, seed + uint64_t(i));
            Rng arng(~(seed + uint64_t(i)));
            while (g.legal(ms) > 0) {
                Move mv = ag->choose(g, arng);
                Piece p = g.next;
                int64_t pts = g.play(mv);
                bigs[i] = std::max(bigs[i], pts);
                if (!record.empty()) logs[i].push_back({p.v[0], uint8_t(p.n == 2 ? p.v[1] : 0), mv.a, mv.b, pts});
            }
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
    int best_i = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
    int big_i = int(std::max_element(bigs.begin(), bigs.end()) - bigs.begin());
    {
        double mb = 0;
        std::vector<int64_t> sb = bigs;
        std::sort(sb.begin(), sb.end());
        for (auto x : bigs) mb += double(x);
        std::printf("biggest single move per game: mean=%.0f  median=%lld  max=%lld (seed %llu)\n", mb / games,
                    (long long)sb[sb.size() / 2], (long long)sb.back(), (unsigned long long)(seed + big_i));
    }
    if (record_by_merge) best_i = big_i;
    std::printf("best game: seed=%llu score=%lld moves=%d  (rerun with --seed %llu --games 1 --threads 1)\n",
                (unsigned long long)(seed + best_i), (long long)scores[best_i], lengths[best_i],
                (unsigned long long)(seed + best_i));
    std::printf("moves/game=%.1f  highest unlocked:", len / games);
    for (int v = 2; v <= M; v++) std::printf(" %d:%.1f%%", v, 100.0 * reach[v] / games);
    std::printf("\ntime=%.1fs  (%.2f games/s, %d threads)\n", secs, games / secs, threads);

    if (!record.empty()) {
        FILE* f = std::fopen(record.c_str(), "w");
        std::fprintf(f, "{\"agent\":\"%s\",\"depth\":%d,\"seed\":%llu,\"score\":%lld,\"moves\":[", agent.c_str(),
                     depth, (unsigned long long)(seed + best_i), (long long)scores[best_i]);
        for (size_t k = 0; k < logs[best_i].size(); k++) {
            auto& r = logs[best_i][k];
            std::fprintf(f, "%s[%d,%d,%d,%d,%lld]", k ? "," : "", r.v0, r.v1, r.a, r.b == NONE ? -1 : r.b,
                         (long long)r.pts);
        }
        std::fprintf(f, "]}\n");
        std::fclose(f);
    }
    if (!csv.empty()) {
        FILE* f = std::fopen(csv.c_str(), "w");
        std::fprintf(f, "game,seed,score,moves,highest_unlocked\n");
        for (int i = 0; i < games; i++)
            std::fprintf(f, "%d,%llu,%lld,%d,%d\n", i, (unsigned long long)(seed + i), (long long)scores[i],
                         lengths[i], top[i]);
        std::fclose(f);
    }
}
