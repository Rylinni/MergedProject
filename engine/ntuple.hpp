// N-tuple value network: V(afterstate) = expected future points.
// Each base tuple is a fixed set of cells whose joint contents index a weight
// table; it is applied to all 8 rotations/reflections of the board (weight
// sharing), plus a small table on the number of empty cells.
#pragma once
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "merged.hpp"

namespace merged {

inline std::array<std::array<uint8_t, CELLS>, 8> make_symmetries() {
    std::array<std::array<uint8_t, CELLS>, 8> s{};
    for (int k = 0; k < 8; k++)
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++) {
                int a = x, b = y;
                for (int r = 0; r < (k & 3); r++) { int t = a; a = b; b = N - 1 - t; }
                if (k >= 4) b = N - 1 - b;
                s[k][x * N + y] = uint8_t(a * N + b);
            }
    return s;
}

struct NTuple {
    static constexpr int SYMS = 8;
    std::vector<std::vector<uint8_t>> tuples;            // base tuples (cell lists)
    std::vector<std::vector<float>> w;                   // weights per base tuple
    std::vector<std::vector<uint8_t>> cells;             // [t][s*len + k] = cell
    float w_empty[CELLS + 1] = {};

    NTuple() {
        auto rect = [](int r, int c, int h, int wd) {
            std::vector<uint8_t> t;
            for (int i = 0; i < h; i++)
                for (int j = 0; j < wd; j++) t.push_back(uint8_t((r + i) * N + c + j));
            return t;
        };
        // 2x3 blocks (up to symmetry these cover every 2x3 window) and full rows.
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 2; c++) tuples.push_back(rect(r, c, 2, 3));
        for (int r = 0; r < 3; r++) tuples.push_back(rect(r, 0, 1, 5));
        auto sym = make_symmetries();
        for (auto& t : tuples) {
            w.emplace_back(size_t(1) << (3 * t.size()), 0.0f);
            std::vector<uint8_t> cs;
            for (int s = 0; s < SYMS; s++)
                for (uint8_t c : t) cs.push_back(sym[s][c]);
            cells.push_back(cs);
        }
    }

    size_t num_weights() const {
        size_t n = CELLS + 1;
        for (auto& v : w) n += v.size();
        return n;
    }

    template <class F>
    void for_each_index(const Board& b, F&& f) {
        for (size_t t = 0; t < tuples.size(); t++) {
            const size_t len = tuples[t].size();
            const uint8_t* cs = cells[t].data();
            for (int s = 0; s < SYMS; s++, cs += len) {
                uint32_t idx = 0;
                for (size_t k = 0; k < len; k++) idx |= uint32_t(b.c[cs[k]]) << (3 * k);
                f(w[t][idx]);
            }
        }
        f(w_empty[popcount(b.empty())]);
    }

    double value(const Board& b) const {
        double v = 0;
        const_cast<NTuple*>(this)->for_each_index(b, [&](float& x) { v += x; });
        return v;
    }
    void update(const Board& b, float step) {
        for_each_index(b, [&](float& x) { x += step; });
    }
    int lookups() const { return int(tuples.size()) * SYMS + 1; }

    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        uint32_t hdr[2] = {0x4d4e5431u /* "MNT1" */, uint32_t(tuples.size())};
        std::fwrite(hdr, sizeof hdr, 1, f);
        for (auto& v : w) std::fwrite(v.data(), sizeof(float), v.size(), f);
        std::fwrite(w_empty, sizeof w_empty, 1, f);
        return std::fclose(f) == 0;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t hdr[2];
        bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x4d4e5431u && hdr[1] == tuples.size();
        for (auto& v : w) ok = ok && std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
        ok = ok && std::fread(w_empty, sizeof w_empty, 1, f) == 1;
        std::fclose(f);
        return ok;
    }
};

}  // namespace merged
