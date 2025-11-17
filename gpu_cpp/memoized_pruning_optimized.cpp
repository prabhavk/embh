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
#include <numeric>

using namespace std;
using namespace std::chrono;

// Same Tree structure as before (abbreviated for brevity)
struct Tree {
    int num_nodes, num_edges, num_taxa, num_patterns;
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;
    vector<vector<int>> children;
    vector<int> edge_parent, edge_child, postorder_edges;
    vector<int> pattern_data, pattern_weights;
    vector<string> taxon_names;
    vector<int> taxon_to_node, node_to_taxon;
    array<double, 4> root_prior;
    double f81_mu;
    int root_node;
    vector<vector<int>> edge_subtree_taxa;
};

uint64_t hash_signature(const uint8_t* sig, int len) {
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        hash ^= sig[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

Tree load_tree(const string& edge_file, const string& pattern_file,
               const string& taxon_file, const string& basecomp_file) {
    Tree tree;

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
    tree.edge_parent.resize(tree.num_edges);
    tree.edge_child.resize(tree.num_edges);

    for (size_t i = 0; i < edges.size(); i++) {
        int pid = tree.name_to_id[edges[i].first];
        int cid = tree.name_to_id[edges[i].second];
        tree.parent_map[cid] = pid;
        tree.branch_lengths[cid] = branch_map[edges[i].second];
        tree.children[pid].push_back(cid);
        tree.edge_parent[i] = pid;
        tree.edge_child[i] = cid;
    }

    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) { tree.root_node = i; break; }
    }

    ifstream tax_in(taxon_file);
    getline(tax_in, line);
    while (getline(tax_in, line)) {
        if (!line.empty()) {
            size_t cp = line.find(',');
            if (cp != string::npos) tree.taxon_names.push_back(line.substr(0, cp));
        }
    }
    tax_in.close();
    tree.num_taxa = tree.taxon_names.size();

    tree.taxon_to_node.resize(tree.num_taxa);
    tree.node_to_taxon.resize(tree.num_nodes, -1);
    for (int t = 0; t < tree.num_taxa; t++) {
        if (tree.name_to_id.count(tree.taxon_names[t])) {
            int node = tree.name_to_id.at(tree.taxon_names[t]);
            tree.taxon_to_node[t] = node;
            tree.node_to_taxon[node] = t;
        }
    }

    ifstream pat_in(pattern_file);
    while (getline(pat_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight; iss >> weight;
        tree.pattern_weights.push_back(weight);
        int base;
        while (iss >> base) tree.pattern_data.push_back(base);
    }
    pat_in.close();
    tree.num_patterns = tree.pattern_weights.size();

    vector<int> node_to_edge(tree.num_nodes, -1);
    for (int e = 0; e < tree.num_edges; e++) node_to_edge[tree.edge_child[e]] = e;

    function<void(int)> build_postorder = [&](int node) {
        for (int child : tree.children[node]) {
            build_postorder(child);
            if (node_to_edge[child] >= 0) tree.postorder_edges.push_back(node_to_edge[child]);
        }
    };
    build_postorder(tree.root_node);

    tree.edge_subtree_taxa.resize(tree.num_edges);
    vector<vector<int>> edge_children(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        int child_node = tree.edge_child[e];
        for (int nc : tree.children[child_node]) {
            if (node_to_edge[nc] >= 0) edge_children[e].push_back(node_to_edge[nc]);
        }
    }

    for (int e : tree.postorder_edges) {
        int child_node = tree.edge_child[e];
        if (tree.node_to_taxon[child_node] >= 0) {
            tree.edge_subtree_taxa[e].push_back(tree.node_to_taxon[child_node]);
        }
        for (int ce : edge_children[e]) {
            for (int taxon : tree.edge_subtree_taxa[ce]) {
                tree.edge_subtree_taxa[e].push_back(taxon);
            }
        }
        sort(tree.edge_subtree_taxa[e].begin(), tree.edge_subtree_taxa[e].end());
    }

    return tree;
}

double compute_ll_no_memo(const Tree& tree) {
    double total_ll = 0.0;
    vector<array<double, 4>> node_likelihood(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    for (int p = 0; p < tree.num_patterns; p++) {
        for (int n = 0; n < tree.num_nodes; n++) {
            node_likelihood[n].fill(1.0);
            node_scale[n] = 0.0;
        }

        for (int t = 0; t < tree.num_taxa; t++) {
            int node = tree.taxon_to_node[t];
            int base = tree.pattern_data[p * tree.num_taxa + t];
            if (base < 4) {
                node_likelihood[node].fill(0.0);
                node_likelihood[node][base] = 1.0;
            }
        }

        for (int e : tree.postorder_edges) {
            int parent = tree.edge_parent[e];
            int child = tree.edge_child[e];
            double t = tree.branch_lengths[child];
            double exp_term = exp(-tree.f81_mu * t);

            array<double, 4> msg;
            for (int i = 0; i < 4; i++) {
                msg[i] = 0.0;
                for (int j = 0; j < 4; j++) {
                    double pij = (i == j) ? exp_term + (1.0 - exp_term) * tree.root_prior[j]
                                          : (1.0 - exp_term) * tree.root_prior[j];
                    msg[i] += pij * node_likelihood[child][j];
                }
                node_likelihood[parent][i] *= msg[i];
            }
            node_scale[parent] += node_scale[child];

            double max_val = *max_element(node_likelihood[parent].begin(), node_likelihood[parent].end());
            if (max_val > 0.0 && max_val < 1e-100) {
                for (int i = 0; i < 4; i++) node_likelihood[parent][i] /= max_val;
                node_scale[parent] += log(max_val);
            }
        }

        double ll = 0.0;
        for (int i = 0; i < 4; i++) ll += tree.root_prior[i] * node_likelihood[tree.root_node][i];
        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }
    return total_ll;
}

// Optimized: pre-compute all pattern signatures, group by signature
double compute_ll_grouped(const Tree& tree, int max_subtree) {
    auto start = high_resolution_clock::now();

    // Step 1: Pre-compute edge transition matrices (constant for all patterns)
    vector<array<array<double, 4>, 4>> P_matrices(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        double t = tree.branch_lengths[tree.edge_child[e]];
        double exp_term = exp(-tree.f81_mu * t);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                P_matrices[e][i][j] = (i == j) ? exp_term + (1.0 - exp_term) * tree.root_prior[j]
                                               : (1.0 - exp_term) * tree.root_prior[j];
            }
        }
    }

    // Step 2: For cacheable edges, group patterns by signature
    vector<bool> should_cache(tree.num_edges, false);
    int num_cached = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        should_cache[e] = ((int)tree.edge_subtree_taxa[e].size() <= max_subtree);
        if (should_cache[e]) num_cached++;
    }

    cout << "Pre-grouping patterns by edge signatures..." << endl;

    // For each cacheable edge: map signature -> list of pattern indices
    vector<unordered_map<uint64_t, vector<int>>> edge_pattern_groups(tree.num_edges);

    for (int e = 0; e < tree.num_edges; e++) {
        if (!should_cache[e]) continue;

        const auto& subtree = tree.edge_subtree_taxa[e];
        vector<uint8_t> sig(subtree.size());

        for (int p = 0; p < tree.num_patterns; p++) {
            for (size_t i = 0; i < subtree.size(); i++) {
                sig[i] = (uint8_t)tree.pattern_data[p * tree.num_taxa + subtree[i]];
            }
            uint64_t h = hash_signature(sig.data(), sig.size());
            edge_pattern_groups[e][h].push_back(p);
        }
    }

    auto group_time = high_resolution_clock::now();
    auto grouping_ms = duration_cast<milliseconds>(group_time - start).count();
    cout << "Grouping time: " << grouping_ms << " ms" << endl;

    // Step 3: Process patterns - compute unique signatures once
    cout << "Computing likelihoods with grouped memoization..." << endl;

    vector<double> pattern_ll(tree.num_patterns, 0.0);
    vector<array<double, 4>> node_likelihood(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    // Cache: edge -> signature -> (message, scale)
    vector<unordered_map<uint64_t, pair<array<double, 4>, double>>> cache(tree.num_edges);

    long total_ops = 0;
    long saved_ops = 0;

    for (int p = 0; p < tree.num_patterns; p++) {
        // Initialize
        for (int n = 0; n < tree.num_nodes; n++) {
            node_likelihood[n].fill(1.0);
            node_scale[n] = 0.0;
        }

        // Set leaf likelihoods
        for (int t = 0; t < tree.num_taxa; t++) {
            int node = tree.taxon_to_node[t];
            int base = tree.pattern_data[p * tree.num_taxa + t];
            if (base < 4) {
                node_likelihood[node].fill(0.0);
                node_likelihood[node][base] = 1.0;
            }
        }

        // Prune with memoization
        for (int e : tree.postorder_edges) {
            int parent = tree.edge_parent[e];
            int child = tree.edge_child[e];

            total_ops++;

            // Try cache
            if (should_cache[e]) {
                const auto& subtree = tree.edge_subtree_taxa[e];
                vector<uint8_t> sig(subtree.size());
                for (size_t i = 0; i < subtree.size(); i++) {
                    sig[i] = (uint8_t)tree.pattern_data[p * tree.num_taxa + subtree[i]];
                }
                uint64_t h = hash_signature(sig.data(), sig.size());

                auto it = cache[e].find(h);
                if (it != cache[e].end()) {
                    // Cache hit
                    saved_ops++;
                    const auto& entry = it->second;
                    for (int i = 0; i < 4; i++) {
                        node_likelihood[parent][i] *= entry.first[i];
                    }
                    node_scale[parent] += entry.second;

                    double max_val = *max_element(node_likelihood[parent].begin(), node_likelihood[parent].end());
                    if (max_val > 0.0 && max_val < 1e-100) {
                        for (int i = 0; i < 4; i++) node_likelihood[parent][i] /= max_val;
                        node_scale[parent] += log(max_val);
                    }
                    continue;
                }
            }

            // Compute message
            array<double, 4> msg;
            for (int i = 0; i < 4; i++) {
                msg[i] = 0.0;
                for (int j = 0; j < 4; j++) {
                    msg[i] += P_matrices[e][i][j] * node_likelihood[child][j];
                }
            }

            // Cache if applicable
            if (should_cache[e]) {
                const auto& subtree = tree.edge_subtree_taxa[e];
                vector<uint8_t> sig(subtree.size());
                for (size_t i = 0; i < subtree.size(); i++) {
                    sig[i] = (uint8_t)tree.pattern_data[p * tree.num_taxa + subtree[i]];
                }
                uint64_t h = hash_signature(sig.data(), sig.size());
                cache[e][h] = {msg, node_scale[child]};
            }

            // Update parent
            for (int i = 0; i < 4; i++) {
                node_likelihood[parent][i] *= msg[i];
            }
            node_scale[parent] += node_scale[child];

            double max_val = *max_element(node_likelihood[parent].begin(), node_likelihood[parent].end());
            if (max_val > 0.0 && max_val < 1e-100) {
                for (int i = 0; i < 4; i++) node_likelihood[parent][i] /= max_val;
                node_scale[parent] += log(max_val);
            }
        }

        // Root likelihood
        double ll = 0.0;
        for (int i = 0; i < 4; i++) ll += tree.root_prior[i] * node_likelihood[tree.root_node][i];
        pattern_ll[p] = tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }

    auto compute_time = high_resolution_clock::now();
    auto compute_ms = duration_cast<milliseconds>(compute_time - group_time).count();
    cout << "Compute time: " << compute_ms << " ms" << endl;
    cout << "Total ops: " << total_ops << ", Saved: " << saved_ops
         << " (" << 100.0 * saved_ops / total_ops << "%)" << endl;

    size_t cache_mem = 0;
    for (int e = 0; e < tree.num_edges; e++) cache_mem += cache[e].size() * 40;
    cout << "Cache memory: " << cache_mem / 1024.0 << " KB" << endl;

    double total_ll = 0.0;
    for (int p = 0; p < tree.num_patterns; p++) total_ll += pattern_ll[p];
    return total_ll;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " -e <edges> -p <patterns> -x <taxon> -b <basecomp> [-t max_subtree]" << endl;
        return 1;
    }

    string edge_file, pattern_file, taxon_file, basecomp_file;
    int max_subtree = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) basecomp_file = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) max_subtree = atoi(argv[++i]);
    }

    cout << "=== Optimized Memoized Pruning Test ===" << endl;

    cout << "\nLoading data..." << endl;
    Tree tree = load_tree(edge_file, pattern_file, taxon_file, basecomp_file);
    cout << "Loaded: " << tree.num_patterns << " patterns, " << tree.num_edges << " edges" << endl;

    cout << "\n--- Baseline (No Memoization) ---" << endl;
    auto t1 = high_resolution_clock::now();
    double ll1 = compute_ll_no_memo(tree);
    auto t2 = high_resolution_clock::now();
    auto time1 = duration_cast<milliseconds>(t2 - t1).count();
    cout << "LL: " << fixed << setprecision(8) << ll1 << endl;
    cout << "Time: " << time1 << " ms" << endl;

    cout << "\n--- Grouped Memoization ---" << endl;
    auto t3 = high_resolution_clock::now();
    double ll2 = compute_ll_grouped(tree, max_subtree);
    auto t4 = high_resolution_clock::now();
    auto time2 = duration_cast<milliseconds>(t4 - t3).count();
    cout << "LL: " << fixed << setprecision(8) << ll2 << endl;
    cout << "Total time: " << time2 << " ms" << endl;

    cout << "\n--- Results ---" << endl;
    double diff = abs(ll1 - ll2);
    cout << "LL difference: " << scientific << diff << endl;
    cout << "Speedup: " << fixed << setprecision(2) << (double)time1 / time2 << "x" << endl;

    if (diff < 1e-6) {
        cout << "CORRECTNESS VERIFIED!" << endl;
    }

    return 0;
}
