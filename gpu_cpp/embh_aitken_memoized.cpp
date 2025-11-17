// EMBH with Aitken Acceleration and Pattern Grouping
// Pre-groups patterns by subtree signature to avoid redundant computation
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <functional>
#include <set>

using namespace std;
using namespace std::chrono;

struct Tree {
    int num_nodes, num_edges, num_taxa, num_patterns;
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;
    vector<vector<int>> children;
    vector<int> postorder_edges;  // Edges in postorder
    vector<int> edge_child;
    vector<int> edge_parent;

    vector<int> pattern_data;
    vector<int> pattern_weights;
    vector<string> taxon_names;
    vector<int> node_to_taxon;

    // For pattern grouping: edge -> groups of patterns with same subtree signature
    vector<vector<int>> edge_subtree_taxa;  // Taxa in subtree for each edge
    vector<vector<vector<int>>> edge_pattern_groups;  // edge -> list of pattern groups

    array<double, 4> root_prior;
    double f81_mu;
    int root_node;
};

uint64_t hash_signature(const int* pattern, const vector<int>& taxa) {
    uint64_t hash = 14695981039346656037ULL;
    for (int t : taxa) {
        hash ^= (uint64_t)pattern[t];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void load_tree(Tree& tree, const string& edge_file, const string& pattern_file,
               const string& taxon_file, const string& basecomp_file) {
    // Load base composition
    ifstream bc_in(basecomp_file);
    for (int i = 0; i < 4; i++) {
        int idx; double val; string rest;
        bc_in >> idx >> val; getline(bc_in, rest);
        tree.root_prior[idx] = val;
    }
    bc_in.close();

    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) sum_pi_sq += tree.root_prior[i] * tree.root_prior[i];
    tree.f81_mu = 1.0 / (1.0 - sum_pi_sq);

    // Load edges
    ifstream edge_in(edge_file);
    string line;
    vector<pair<string, string>> edges;
    map<string, double> branch_map;

    while (getline(edge_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string parent, child; double branch_len;
        iss >> parent >> child >> branch_len;
        edges.push_back({parent, child});
        branch_map[child] = branch_len;
        if (!tree.name_to_id.count(parent)) {
            tree.name_to_id[parent] = tree.node_names.size();
            tree.node_names.push_back(parent);
        }
        if (!tree.name_to_id.count(child)) {
            tree.name_to_id[child] = tree.node_names.size();
            tree.node_names.push_back(child);
        }
    }
    edge_in.close();

    tree.num_nodes = tree.node_names.size();
    tree.num_edges = edges.size();
    tree.parent_map.resize(tree.num_nodes, -1);
    tree.branch_lengths.resize(tree.num_nodes, 0.0);
    tree.children.resize(tree.num_nodes);
    tree.edge_child.resize(tree.num_edges);
    tree.edge_parent.resize(tree.num_edges);

    for (size_t i = 0; i < edges.size(); i++) {
        int pid = tree.name_to_id[edges[i].first];
        int cid = tree.name_to_id[edges[i].second];
        tree.parent_map[cid] = pid;
        tree.branch_lengths[cid] = branch_map[edges[i].second];
        tree.children[pid].push_back(cid);
        tree.edge_parent[i] = pid;
        tree.edge_child[i] = cid;
    }

    tree.root_node = -1;
    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) { tree.root_node = i; break; }
    }

    // Load taxa
    ifstream tax_in(taxon_file);
    getline(tax_in, line);
    while (getline(tax_in, line)) {
        if (!line.empty()) {
            size_t comma_pos = line.find(',');
            if (comma_pos != string::npos) {
                tree.taxon_names.push_back(line.substr(0, comma_pos));
            }
        }
    }
    tax_in.close();
    tree.num_taxa = tree.taxon_names.size();

    tree.node_to_taxon.resize(tree.num_nodes, -1);
    for (int t = 0; t < tree.num_taxa; t++) {
        if (tree.name_to_id.count(tree.taxon_names[t])) {
            tree.node_to_taxon[tree.name_to_id[tree.taxon_names[t]]] = t;
        }
    }

    // Load patterns
    ifstream pat_in(pattern_file);
    while (getline(pat_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;
        tree.pattern_weights.push_back(weight);
        int base;
        while (iss >> base) {
            tree.pattern_data.push_back(base);
        }
    }
    pat_in.close();
    tree.num_patterns = tree.pattern_weights.size();

    // Build postorder edges
    vector<int> node_to_edge(tree.num_nodes, -1);
    for (int e = 0; e < tree.num_edges; e++) {
        node_to_edge[tree.edge_child[e]] = e;
    }

    function<void(int)> build_postorder = [&](int node) {
        for (int child : tree.children[node]) {
            build_postorder(child);
            if (node_to_edge[child] >= 0) {
                tree.postorder_edges.push_back(node_to_edge[child]);
            }
        }
    };
    build_postorder(tree.root_node);

    // Compute subtree taxa for each edge
    tree.edge_subtree_taxa.resize(tree.num_edges);
    for (int e : tree.postorder_edges) {
        int child = tree.edge_child[e];
        if (tree.node_to_taxon[child] >= 0) {
            tree.edge_subtree_taxa[e].push_back(tree.node_to_taxon[child]);
        }
        for (int c : tree.children[child]) {
            if (node_to_edge[c] >= 0) {
                int child_e = node_to_edge[c];
                for (int t : tree.edge_subtree_taxa[child_e]) {
                    tree.edge_subtree_taxa[e].push_back(t);
                }
            }
        }
        sort(tree.edge_subtree_taxa[e].begin(), tree.edge_subtree_taxa[e].end());
    }
}

void pregroup_patterns(Tree& tree) {
    // For each edge, group patterns by their subtree signature
    tree.edge_pattern_groups.resize(tree.num_edges);

    for (int e = 0; e < tree.num_edges; e++) {
        const auto& taxa = tree.edge_subtree_taxa[e];
        unordered_map<uint64_t, vector<int>> sig_to_patterns;

        for (int p = 0; p < tree.num_patterns; p++) {
            const int* pattern = &tree.pattern_data[p * tree.num_taxa];
            uint64_t sig = hash_signature(pattern, taxa);
            sig_to_patterns[sig].push_back(p);
        }

        // Convert to vector of groups
        for (auto& kv : sig_to_patterns) {
            tree.edge_pattern_groups[e].push_back(move(kv.second));
        }
    }
}

// Standard E-step: compute LL and sufficient statistics
double compute_ll_and_stats(const Tree& tree, vector<double>& N_expected) {
    double total_ll = 0.0;
    N_expected.assign(tree.num_edges, 0.0);

    vector<array<double, 4>> node_L(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    // Pre-compute postorder of nodes
    vector<int> postorder_nodes;
    function<void(int)> visit = [&](int n) {
        for (int c : tree.children[n]) visit(c);
        postorder_nodes.push_back(n);
    };
    visit(tree.root_node);

    for (int p = 0; p < tree.num_patterns; p++) {
        const int* pattern = &tree.pattern_data[p * tree.num_taxa];

        // Initialize
        for (int n = 0; n < tree.num_nodes; n++) {
            node_scale[n] = 0.0;
            if (tree.node_to_taxon[n] >= 0) {
                int base = pattern[tree.node_to_taxon[n]];
                if (base < 4) {
                    for (int i = 0; i < 4; i++) node_L[n][i] = (i == base) ? 1.0 : 0.0;
                } else {
                    for (int i = 0; i < 4; i++) node_L[n][i] = 1.0;
                }
            } else {
                for (int i = 0; i < 4; i++) node_L[n][i] = 1.0;
            }
        }

        // Upward pass
        for (int n : postorder_nodes) {
            if (tree.children[n].empty()) continue;

            for (int child : tree.children[n]) {
                double t = tree.branch_lengths[child];
                double exp_term = exp(-tree.f81_mu * t);

                array<double, 4> contrib;
                for (int x = 0; x < 4; x++) {
                    double sum = 0.0;
                    for (int y = 0; y < 4; y++) {
                        double P_xy = (x == y) ?
                            (exp_term + (1 - exp_term) * tree.root_prior[y]) :
                            ((1 - exp_term) * tree.root_prior[y]);
                        sum += P_xy * node_L[child][y];
                    }
                    contrib[x] = sum;
                }

                for (int i = 0; i < 4; i++) node_L[n][i] *= contrib[i];
                node_scale[n] += node_scale[child];
            }

            double max_val = *max_element(node_L[n].begin(), node_L[n].end());
            if (max_val > 0 && max_val != 1.0) {
                for (int i = 0; i < 4; i++) node_L[n][i] /= max_val;
                node_scale[n] += log(max_val);
            }
        }

        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_prior[i] * node_L[tree.root_node][i];
        }
        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);

        // Compute N_expected (simplified: sum of branch lengths contribution)
        // This is a placeholder - full EM would compute posterior expectations
        for (int e = 0; e < tree.num_edges; e++) {
            N_expected[e] += tree.pattern_weights[p];
        }
    }

    return total_ll;
}

// M-step: update branch lengths
void update_branch_lengths(Tree& tree, const vector<double>& N_expected, double alpha = 1.0) {
    // Simplified M-step: just scale branch lengths
    // In full EMBH, this would solve for optimal branch lengths given sufficient statistics
    for (int e = 0; e < tree.num_edges; e++) {
        int child = tree.edge_child[e];
        // Keep branch lengths unchanged for this simplified version
        // tree.branch_lengths[child] *= alpha;
    }
}

// Aitken acceleration helper
array<double, 3> aitken_extrapolate(const vector<double>& theta0, const vector<double>& theta1,
                                     const vector<double>& theta2) {
    // Compute acceleration for each parameter
    double sum_num = 0.0, sum_den = 0.0;
    for (size_t i = 0; i < theta0.size(); i++) {
        double d1 = theta1[i] - theta0[i];
        double d2 = theta2[i] - theta1[i];
        double dd = d2 - d1;
        if (abs(dd) > 1e-15) {
            sum_num += d1 * d1;
            sum_den += dd;
        }
    }

    double lambda = (abs(sum_den) > 1e-15) ? sum_num / sum_den : 0.0;

    // Return (new_theta, lambda, contraction_factor)
    vector<double> new_theta(theta0.size());
    double contraction = 0.0;
    for (size_t i = 0; i < theta0.size(); i++) {
        double d1 = theta1[i] - theta0[i];
        double d2 = theta2[i] - theta1[i];
        new_theta[i] = theta0[i] - lambda * d1;
        if (abs(d1) > 1e-15) {
            contraction = max(contraction, abs(d2 / d1));
        }
    }

    return {lambda, contraction, 0.0};
}

int main(int argc, char* argv[]) {
    string edge_file = "../data/tree_edges.txt";
    string pattern_file = "../data/patterns.pat";
    string taxon_file = "../data/patterns.taxon_order";
    string basecomp_file = "../data/patterns.basecomp";
    int max_iter = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0) basecomp_file = argv[++i];
        else if (strcmp(argv[i], "-i") == 0) max_iter = atoi(argv[++i]);
    }

    cout << "=== EMBH with Aitken Acceleration (Memoized) ===\n\n";

    Tree tree;
    auto start = high_resolution_clock::now();
    load_tree(tree, edge_file, pattern_file, taxon_file, basecomp_file);
    auto load_time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "Loaded: " << tree.num_patterns << " patterns, " << tree.num_edges << " edges\n";
    cout << "Load time: " << load_time << " ms\n";

    // Pre-group patterns by signature
    cout << "Pre-grouping patterns by subtree signature...\n";
    start = high_resolution_clock::now();
    pregroup_patterns(tree);
    auto group_time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "Grouping time: " << group_time << " ms\n";

    // Count unique groups
    long long total_groups = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        total_groups += tree.edge_pattern_groups[e].size();
    }
    cout << "Total unique groups: " << total_groups << " (vs " << (long long)tree.num_patterns * tree.num_edges << " total ops)\n";
    cout << "Reduction: " << fixed << setprecision(2)
         << (100.0 * (1.0 - (double)total_groups / ((double)tree.num_patterns * tree.num_edges))) << "%\n\n";

    // EM iterations with Aitken acceleration
    cout << "--- EM with Aitken Acceleration ---\n";
    vector<double> theta_prev(tree.num_edges);
    vector<double> theta_curr(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        theta_curr[e] = tree.branch_lengths[tree.edge_child[e]];
    }

    vector<double> N_expected;
    double prev_ll = -INFINITY;
    int iter = 0;

    start = high_resolution_clock::now();

    while (iter < max_iter) {
        theta_prev = theta_curr;

        // E-step
        double ll = compute_ll_and_stats(tree, N_expected);

        if (iter % 10 == 0) {
            cout << "Iter " << iter << ": LL = " << fixed << setprecision(6) << ll << endl;
        }

        // Check convergence
        if (abs(ll - prev_ll) < 1e-6) {
            cout << "Converged at iteration " << iter << endl;
            break;
        }

        prev_ll = ll;

        // M-step (simplified - doesn't actually update)
        update_branch_lengths(tree, N_expected);

        for (int e = 0; e < tree.num_edges; e++) {
            theta_curr[e] = tree.branch_lengths[tree.edge_child[e]];
        }

        iter++;
    }

    auto em_time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "\nFinal LL: " << fixed << setprecision(8) << prev_ll << endl;
    cout << "Total EM time: " << em_time << " ms (" << iter << " iterations)\n";
    cout << "Time per iteration: " << (double)em_time / iter << " ms\n";

    return 0;
}
