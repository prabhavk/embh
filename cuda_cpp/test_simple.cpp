/*
 * Simple 3-taxon test to compare with C implementation
 * Tree structure: ((A,B),C)
 *       root (h0)
 *        /  \
 *      h1    C
 *     /  \
 *    A    B
 */

#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <map>
using namespace std;

using Md = array<array<double,4>,4>;

// Compute F81 transition matrix
Md compute_f81_matrix(double t, const array<double,4>& pi) {
    Md P{};
    double beta = 0.0;
    for (int i = 0; i < 4; i++) {
        beta += pi[i] * pi[i];
    }
    beta = 1.0 / (1.0 - beta);

    double exp_bt = exp(-beta * t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i][j] = exp_bt + pi[j] * (1.0 - exp_bt);
            } else {
                P[i][j] = pi[j] * (1.0 - exp_bt);
            }
        }
    }
    return P;
}

int main() {
    cout << "C++ Simple 3-Taxon Log-Likelihood Test" << endl;
    cout << "========================================" << endl << endl;

    // Base frequencies (uniform)
    array<double,4> pi = {0.25, 0.25, 0.25, 0.25};
    cout << "Base frequencies: A=0.25, C=0.25, G=0.25, T=0.25" << endl;

    // Branch lengths
    double t_A_h1 = 0.1;
    double t_B_h1 = 0.1;
    double t_h1_h0 = 0.05;
    double t_C_h0 = 0.15;

    cout << "Branch lengths: A-h1=0.10, B-h1=0.10, h1-h0=0.05, C-h0=0.15" << endl << endl;

    // Compute transition matrices
    Md P_A_h1 = compute_f81_matrix(t_A_h1, pi);
    Md P_B_h1 = compute_f81_matrix(t_B_h1, pi);
    Md P_h1_h0 = compute_f81_matrix(t_h1_h0, pi);
    Md P_C_h0 = compute_f81_matrix(t_C_h0, pi);

    // Patterns (same as C test)
    vector<array<int,3>> patterns = {
        {0, 0, 0},  // AAA
        {0, 0, 0},  // AAA (duplicate)
        {0, 0, 1},  // AAC
        {0, 1, 0},  // ACA
        {1, 1, 1},  // CCC
        {1, 1, 1},  // CCC
        {1, 1, 1},  // CCC
        {2, 2, 2},  // GGG
        {2, 2, 2},  // GGG
        {3, 3, 3},  // TTT
    };

    cout << "Number of site patterns: " << patterns.size() << endl;
    cout << "Computing log-likelihood..." << endl;

    double total_log_likelihood = 0.0;
    int cache_hits = 0;
    int cache_misses = 0;

    // Cache for memoization: signature -> (message, log_scale)
    map<array<int,1>, pair<array<double,4>, double>> cache_A;
    map<array<int,1>, pair<array<double,4>, double>> cache_B;
    map<array<int,1>, pair<array<double,4>, double>> cache_C;

    for (size_t site = 0; site < patterns.size(); site++) {
        int obs_A = patterns[site][0];
        int obs_B = patterns[site][1];
        int obs_C = patterns[site][2];

        array<double,4> msg_A, msg_B, msg_C;
        double log_scale_A = 0, log_scale_B = 0, log_scale_C = 0;

        // Check cache for A
        array<int,1> sig_A = {obs_A};
        if (cache_A.find(sig_A) != cache_A.end()) {
            msg_A = cache_A[sig_A].first;
            log_scale_A = cache_A[sig_A].second;
            cache_hits++;
        } else {
            // Compute message from A to h1
            // msg_A[h1] = sum_A P[h1][A] * I(obs_A)
            //           = P[h1][obs_A]
            double max_val = 0;
            for (int h1 = 0; h1 < 4; h1++) {
                msg_A[h1] = P_A_h1[h1][obs_A];
                max_val = max(max_val, msg_A[h1]);
            }
            for (int h1 = 0; h1 < 4; h1++) msg_A[h1] /= max_val;
            log_scale_A = log(max_val);
            cache_A[sig_A] = {msg_A, log_scale_A};
            cache_misses++;
        }

        // Check cache for B
        array<int,1> sig_B = {obs_B};
        if (cache_B.find(sig_B) != cache_B.end()) {
            msg_B = cache_B[sig_B].first;
            log_scale_B = cache_B[sig_B].second;
            cache_hits++;
        } else {
            double max_val = 0;
            for (int h1 = 0; h1 < 4; h1++) {
                msg_B[h1] = P_B_h1[h1][obs_B];
                max_val = max(max_val, msg_B[h1]);
            }
            for (int h1 = 0; h1 < 4; h1++) msg_B[h1] /= max_val;
            log_scale_B = log(max_val);
            cache_B[sig_B] = {msg_B, log_scale_B};
            cache_misses++;
        }

        // Check cache for C
        array<int,1> sig_C = {obs_C};
        if (cache_C.find(sig_C) != cache_C.end()) {
            msg_C = cache_C[sig_C].first;
            log_scale_C = cache_C[sig_C].second;
            cache_hits++;
        } else {
            double max_val = 0;
            for (int h0 = 0; h0 < 4; h0++) {
                msg_C[h0] = P_C_h0[h0][obs_C];
                max_val = max(max_val, msg_C[h0]);
            }
            for (int h0 = 0; h0 < 4; h0++) msg_C[h0] /= max_val;
            log_scale_C = log(max_val);
            cache_C[sig_C] = {msg_C, log_scale_C};
            cache_misses++;
        }

        // Compute likelihood at root h0
        // P(A,B,C) = sum_h0 P(h0) * [sum_h1 P(h1|h0) * msg_A[h1] * msg_B[h1]] * msg_C[h0]
        double site_likelihood = 0.0;
        for (int h0 = 0; h0 < 4; h0++) {
            double sum_h1 = 0.0;
            for (int h1 = 0; h1 < 4; h1++) {
                sum_h1 += P_h1_h0[h0][h1] * msg_A[h1] * msg_B[h1];
            }
            site_likelihood += pi[h0] * sum_h1 * msg_C[h0];
        }

        double total_log_scale = log_scale_A + log_scale_B + log_scale_C;
        double site_log_lik = total_log_scale + log(site_likelihood);
        total_log_likelihood += site_log_lik;

        cout << "  Site " << site << ": pattern (" << obs_A << "," << obs_B << "," << obs_C
             << "), log-lik = " << site_log_lik << endl;
    }

    cout << endl << "========================================" << endl;
    cout << "RESULTS:" << endl;
    cout << "  Total log-likelihood: " << total_log_likelihood << endl;
    cout << "  Cache hits: " << cache_hits << endl;
    cout << "  Cache misses: " << cache_misses << endl;
    cout << "  Hit rate: " << (100.0 * cache_hits / (cache_hits + cache_misses)) << "%" << endl;
    cout << "========================================" << endl;

    return 0;
}
