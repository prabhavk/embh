#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdio>
#include <tuple>
#include <map>

namespace emtr {  

    // Alias that matches your EMTR_results element type
    using Row = std::tuple<
        std::string, // method ("Parsimony" | "Dirichlet" | "HSS")
        std::string, // root
        int,         // repetition (1..N)
        int,         // iter (max iterations taken by EM)
        double,      // ll_initial
        double,      // ecd_ll_first
        double,      // ecd_ll_final
        double       // ll_final
    >;

    inline void debug_counts(const std::vector<Row>& v, const char* where) {
        std::unordered_map<std::string,size_t> by;
        for (const auto& t : v) by[std::get<0>(t)]++;
        std::printf("[COUNT]\t%s size=%zu :: ", where, v.size());
        bool first=true;
        for (auto& kv : by) { if(!first) std::printf(", "); std::printf("%s:%zu", kv.first.c_str(), kv.second); first=false; }
        if (first) std::printf("(empty)");
        std::printf("\n"); std::fflush(stdout);
    }

    // Convenience: push one row into your vector (keeps call sites tidy)
    inline void push_result(
        std::vector<Row>& results,
        const std::string& method,
        const std::string& root,
        int repetition,
        int iter,
        double ll_initial,
        double ecd_ll_first,
        double ecd_ll_final,
        double ll_final)
    {
        results.emplace_back(method, root, repetition, iter,
                             ll_initial, ecd_ll_first, ecd_ll_final, ll_final);
    }

    // ---------- 4x4 helpers ----------
    using Md = std::array<std::array<double,4>,4>;

    inline std::mt19937_64& rng() {
        static thread_local std::mt19937_64 g{0xC0FFEEULL};
        return g;
    }

    inline bool starts_with(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
    }

    inline std::pair<int,int> ord_pair(int a, int b) {
        return (a < b) ? std::make_pair(a,b) : std::make_pair(b,a);
    }

    inline std::vector<std::string> split_ws(const std::string& s) {
        std::istringstream iss(s);
        std::vector<std::string> out;
        for (std::string tok; iss >> tok;) out.push_back(tok);
        return out;
    }

    inline Md MT(const Md& P) {
        Md Pt{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Pt[j][i] = P[i][j];
        return Pt;
    }

    inline Md MM(const Md& A, const Md& B) {
        Md R{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    R[i][j] += A[i][k] * B[k][j];
        return R;
    }  
}
