// Test leaf-only computation to validate grouped approach
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <set>
#include <array>
#include <iomanip>

using namespace std;

double jc_prob(int parent_state, int child_state, double t) {
    double exp_term = exp(-4.0 * t / 3.0);
    if (parent_state == child_state) {
        return 0.25 + 0.75 * exp_term;
    } else {
        return 0.25 - 0.25 * exp_term;
    }
}

int main() {
    // Simple test: single leaf edge, single pattern
    // Edge: h_0 -> Nag with branch length 0.013721
    // Pattern: base = 2 (G)

    double t = 0.013721;

    // Compute message: P(parent_state | observed_base=G)
    array<double, 4> message;
    for (int p = 0; p < 4; p++) {
        message[p] = jc_prob(p, 2, t);  // P(child=2 | parent=p)
    }

    cout << "Raw message for observed G:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "  message[" << i << "] = " << message[i] << endl;
    }

    // Scale
    double max_val = *max_element(message.begin(), message.end());
    cout << "Max value: " << max_val << endl;
    cout << "Log scale: " << log(max_val) << endl;

    for (int i = 0; i < 4; i++) {
        message[i] /= max_val;
    }

    cout << "Scaled message:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "  message[" << i << "] = " << message[i] << endl;
    }

    // Root probability (uniform)
    array<double, 4> root_prob = {0.25, 0.25, 0.25, 0.25};

    // For a single-edge tree, the log-likelihood is:
    // log(sum_p root_prob[p] * message[p]) + log_scale
    double marginal = 0.0;
    for (int i = 0; i < 4; i++) {
        marginal += root_prob[i] * message[i];
    }

    cout << "Marginal: " << marginal << endl;
    cout << "Log-likelihood: " << log(max_val) + log(marginal) << endl;

    return 0;
}
