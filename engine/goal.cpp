// "Reach the biggest merge in as few turns as possible, resetting whenever you like."
//
// An episode runs until one move makes a chain of at least --chain steps (6 = the
// x720 domino chain ending in an M explosion). Game overs and voluntary resets start
// a new game inside the same episode, and every move counts as a turn.
//
// Objective: maximise E[gamma^T], T = turns until the goal. For gamma close to 1 this
// is the same as minimising expected turns, but the values stay in [0, 1], which an
// n-tuple network learns far better than raw turn counts.
//   value of a move that reaches the goal  = 1
//   value of any other move                = gamma * V(afterstate)
//   value of resetting (or of a game over) = V(empty board) = V_fresh
// The network learns V by TD(0). Policy: best move, or reset if V_fresh beats it.
//
//   merged_goal train --out W.bin [--in W.bin] [--init-from W.bin --init-value X] ...
//   merged_goal eval  --policy goal --weights W.bin ...          (resets allowed)
//   merged_goal eval  --policy convex:15000 --weights W.bin ...  (baseline, never resets)
#include <algorithm>
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

struct Opts {
    std::string mode, out, in, init_from, weights, policy = "goal", csv;
    int chain = 6, threads = 4, depth = 1;
    long episodes = 1000, report = 1000, max_turns = 500000;
    double gamma = 0.9995, alpha = 0.02, alpha_end = -1, init_value = 0.3, margin = 0.0;
    bool resets = true;
    uint64_t seed = 1;
};

static void usage() {
    std::fprintf(stderr,
        "usage: merged_goal train --out FILE [--in FILE | --init-from FILE --init-value X]\n"
        "                         [--chain L] [--gamma G] [--episodes N] [--alpha A] [--alpha-end A]\n"
        "                         [--no-resets] [--threads T] [--seed S] [--report N]\n"
        "       merged_goal eval  --weights FILE [--policy goal|convex:K|sum] [--chain L] [--gamma G]\n"
        "                         [--episodes N] [--no-resets] [--threads T] [--seed S] [--csv FILE]\n");
    std::exit(2);
}

static const Board& empty_board() {
    static Board b = [] { Board x; x.clear(); return x; }();
    return b;
}

struct Choice { int idx = -1; double q = -1; bool goal = false; int chain = 0; int64_t pts = 0; };

// Best move under the goal objective. V of a full board is V_fresh (game over -> new game).
static Choice goal_choice(const NTuple& net, const Game& g, const Move* ms, int n, int chain_goal, double gamma,
                          double v_fresh, Rng& rng) {
    Choice best;
    int ties = 0;
    for (int i = 0; i < n; i++) {
        Board nb = g.b;
        int ch = 0;
        int64_t pts = apply(nb, g.next, ms[i], g.R, &ch);
        bool goal = ch >= chain_goal;
        double q = goal ? 1.0 : gamma * (nb.empty() ? net.value(nb) : v_fresh);
        if (q > best.q + 1e-12) { best = {i, q, goal, ch, pts}; ties = 1; }
        else if (q > best.q - 1e-12 && rng.below(++ties) == 0) best = {i, q, goal, ch, pts};
    }
    return best;
}

struct EpisodeStats {
    long turns = 0, games = 1, resets = 0, gameovers = 0;
    int64_t goal_points = 0;
    long first_turn[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // turn when a chain of >= k steps first happened
    bool capped = false;
};

int main(int argc, char** argv) {
    if (argc < 2) usage();
    Opts o;
    o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--out") o.out = next();
        else if (a == "--in") o.in = next();
        else if (a == "--init-from") o.init_from = next();
        else if (a == "--init-value") o.init_value = std::atof(next());
        else if (a == "--weights") o.weights = next();
        else if (a == "--policy") o.policy = next();
        else if (a == "--chain") o.chain = std::atoi(next());
        else if (a == "--gamma") o.gamma = std::atof(next());
        else if (a == "--episodes") o.episodes = std::atol(next());
        else if (a == "--report") o.report = std::atol(next());
        else if (a == "--max-turns") o.max_turns = std::atol(next());
        else if (a == "--alpha") o.alpha = std::atof(next());
        else if (a == "--alpha-end") o.alpha_end = std::atof(next());
        else if (a == "--margin") o.margin = std::atof(next());
        else if (a == "--no-resets") o.resets = false;
        else if (a == "--threads") o.threads = std::atoi(next());
        else if (a == "--depth") o.depth = std::atoi(next());
        else if (a == "--seed") o.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--csv") o.csv = next();
        else usage();
    }
    const bool train = o.mode == "train";
    if (!train && o.mode != "eval") usage();
    if (o.alpha_end < 0) o.alpha_end = o.alpha;

    NTuple net;
    if (train) {
        if (o.out.empty()) usage();
        if (!o.in.empty()) { if (!net.load(o.in)) { std::fprintf(stderr, "cannot load %s\n", o.in.c_str()); return 1; } }
        else if (!o.init_from.empty()) {
            // Start from another model's weights, rescaled so V(empty board) = init_value. Scaling a
            // linear network keeps its move ranking, so play starts out like that model.
            if (!net.load(o.init_from)) { std::fprintf(stderr, "cannot load %s\n", o.init_from.c_str()); return 1; }
            double v0 = net.value(empty_board());
            double k = o.init_value / v0;
            for (auto& t : net.w) for (auto& x : t) x = float(x * k);
            for (auto& x : net.w_empty) x = float(x * k);
            std::fprintf(stderr, "initialised from %s: V(empty) %.1f scaled by %.3g -> %.3f\n", o.init_from.c_str(), v0, k,
                         net.value(empty_board()));
        }
    } else if (o.weights.empty() || !net.load(o.weights)) { std::fprintf(stderr, "cannot load --weights\n"); return 1; }

    // Baseline policies reuse the ordinary expectimax agent and never reset.
    double convex_k = 0;
    bool goal_policy = o.policy == "goal";
    if (!goal_policy) {
        if (o.policy.rfind("convex:", 0) == 0) convex_k = std::atof(o.policy.c_str() + 7);
        else if (o.policy != "sum") usage();
    }
    NTupleEval ntev(net);
    Rules R;

    std::atomic<long> next_ep{0};
    std::mutex mu;
    std::vector<EpisodeStats> all(train ? 0 : o.episodes);
    double rep_turns = 0, rep_resets = 0, rep_games = 0;
    long rep_n = 0, done = 0, total_turns = 0;
    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Move ms[MAX_MOVES];
        ExpectimaxAgent base(o.depth, R, ntev, false, convex_k);
        for (long ep; (ep = next_ep++) < o.episodes;) {
            double frac = double(ep) / double(o.episodes);
            float step = float(o.alpha * std::pow(o.alpha_end / o.alpha, frac) / net.lookups());
            Rng rng(o.seed * 1000003ULL + uint64_t(ep));
            uint64_t game_seed = (o.seed + 7) * 1000003ULL + uint64_t(ep) * 4096;
            Game g(R, game_seed++);
            EpisodeStats st;
            Board prev = empty_board();  // afterstate the next decision is reached from
            long in_game = 0;
            auto new_game = [&]() { g.reset(game_seed++); st.games++; prev = empty_board(); in_game = 0; };
            for (;;) {
                if (st.turns >= o.max_turns) { st.capped = true; break; }
                int n = g.legal(ms);
                double v_fresh = net.value(empty_board());
                if (n == 0) {  // game over: forced reset
                    if (train) net.update(prev, float(step * (v_fresh - net.value(prev))));
                    st.gameovers++;
                    new_game();
                    continue;
                }
                Choice c;
                if (goal_policy) c = goal_choice(net, g, ms, n, o.chain, o.gamma, v_fresh, rng);
                else {
                    Move mv = base.choose(g, rng);
                    for (int i = 0; i < n; i++) if (ms[i].a == mv.a && ms[i].b == mv.b) c.idx = i;
                    Board nb = g.b;
                    c.pts = apply(nb, g.next, ms[c.idx], R, &c.chain);
                    c.goal = c.chain >= o.chain;
                }
                if (goal_policy && o.resets && in_game > 0 && v_fresh > c.q + o.margin) {
                    if (train) net.update(prev, float(step * (v_fresh - net.value(prev))));
                    st.resets++;
                    new_game();
                    continue;
                }
                if (train) net.update(prev, float(step * (c.q - net.value(prev))));
                g.play(ms[c.idx]);
                st.turns++;
                in_game++;
                for (int k = 1; k <= std::min(c.chain, 7); k++) if (st.first_turn[k] < 0) st.first_turn[k] = st.turns;
                if (c.goal) { st.goal_points = c.pts; break; }
                prev = g.b;
            }
            std::lock_guard<std::mutex> lk(mu);
            total_turns += st.turns;
            if (!train) all[ep] = st;
            rep_turns += double(st.turns); rep_resets += double(st.resets); rep_games += double(st.games); rep_n++;
            if (++done % o.report == 0 || done == o.episodes) {
                double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("episodes=%ld  turns/episode=%.0f  games/episode=%.1f  resets/episode=%.1f  V(empty)=%.4f  alpha=%.4f  %.0fs (%.0f turns/s)\n",
                            done, rep_turns / rep_n, rep_games / rep_n, rep_resets / rep_n, net.value(empty_board()),
                            step * net.lookups(), secs, total_turns / secs);
                std::fflush(stdout);
                rep_turns = rep_resets = rep_games = 0; rep_n = 0;
                if (train) net.save(o.out);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < o.threads; t++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    if (train) { net.save(o.out); return 0; }

    // ---- evaluation report ----
    std::vector<long> turns;
    long capped = 0, resets = 0, gameovers = 0, games = 0, ge65k = 0, max70560 = 0;
    for (auto& s : all) {
        if (s.capped) capped++;
        else turns.push_back(s.turns);
        resets += s.resets; gameovers += s.gameovers; games += s.games;
        if (s.goal_points >= 65000) ge65k++;
        if (s.goal_points >= 70560) max70560++;
    }
    std::sort(turns.begin(), turns.end());
    double mean = 0;
    for (long t : turns) mean += double(t);
    mean /= std::max<size_t>(1, turns.size());
    auto pct = [&](double q) { return turns.empty() ? 0L : turns[std::min(turns.size() - 1, size_t(q * turns.size()))]; };
    double sd = 0;
    for (long t : turns) sd += (t - mean) * (t - mean);
    sd = std::sqrt(sd / std::max<size_t>(1, turns.size() - 1));
    std::printf("policy=%s goal=%d-step chain resets=%s episodes=%ld\n", o.policy.c_str(), o.chain,
                goal_policy && o.resets ? "allowed" : "off", o.episodes);
    std::printf("turns to goal: mean=%.0f (s.e. %.0f)  median=%ld  p90=%ld  max=%ld   (%ld episodes hit the %ld-turn cap)\n",
                mean, sd / std::sqrt(std::max<size_t>(1, turns.size())), pct(0.5), pct(0.9), turns.empty() ? 0L : turns.back(),
                capped, o.max_turns);
    std::printf("per episode: games=%.1f  voluntary resets=%.1f  game overs=%.1f\n", double(games) / o.episodes,
                double(resets) / o.episodes, double(gameovers) / o.episodes);
    std::printf("median turns to first chain of k+ steps:");
    for (int k = 3; k <= o.chain; k++) {
        std::vector<long> f;
        for (auto& s : all) if (s.first_turn[k] >= 0) f.push_back(s.first_turn[k]);
        std::sort(f.begin(), f.end());
        std::printf("  %d: %ld", k, f.empty() ? -1L : f[f.size() / 2]);
    }
    std::printf("\ngoal move points: >=65,000 in %.1f%% of episodes, the 70,560 maximum in %.1f%%\n",
                100.0 * ge65k / o.episodes, 100.0 * max70560 / o.episodes);
    if (!o.csv.empty()) {
        FILE* f = std::fopen(o.csv.c_str(), "w");
        std::fprintf(f, "episode,turns,games,resets,gameovers,goal_points,capped\n");
        for (size_t i = 0; i < all.size(); i++)
            std::fprintf(f, "%zu,%ld,%ld,%ld,%ld,%lld,%d\n", i, all[i].turns, all[i].games, all[i].resets, all[i].gameovers,
                         (long long)all[i].goal_points, int(all[i].capped));
        std::fclose(f);
    }
}
