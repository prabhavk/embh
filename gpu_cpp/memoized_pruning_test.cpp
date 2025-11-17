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

// Tree structure
struct Tree {
    int num_nodes;
    int num_edges;
    int num_taxa;
    int num_patterns;

    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;
    vector<vector<int>> children;

    vector<int> edge_parent;
    vector<int> edge_child;
    vector<int> postorder_edges;

    vector<int> pattern_data;
    vector<int> pattern_weights;
    vector<string> taxon_names;
    vector<int> taxon_to_node;
    vector<int> node_to_taxon;

    array<double, 4> root_prior;
    double f81_mu;
    int root_node;

    // Subtree information for memoization
    vector<vector<int>> edge_subtree_taxa;  // taxa in subtree for each edge
    vector<bool> edge_should_cache;         // whether to cache this edge
};

// Cache entry: conditional likelihood and scale factor
struct CacheEntry {
    array<double, 4> likelihood;  // P(data below | parent state)
    double log_scale;
};

// Hash function for signature (FNV-1a 64-bit)
uint64_t hash_signature(const vector<uint8_t>& sig) {
    uint64_t hash = 14695981039346656037ULL;
    for (uint8_t val : sig) {
        hash ^= val;
        hash *= 1099511628211ULL;
    }
    return hash;
}

Tree load_tree(const string& edge_file, const string& pattern_file,
               const string& taxon_file, const string& basecomp_file) {
    Tree tree;

    // Load base composition
    ifstream bc_in(basecomp_file);
    for (int i = 0; i < 4; i++) {
        int idx;
        double val;
        string rest;
        bc_in >> idx >> val;
        getline(bc_in, rest);
        tree.root_prior[idx] = val;
    }
    bc_in.close();

    // Compute F81 mu
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += tree.root_prior[i] * tree.root_prior[i];
    }
    tree.f81_mu = 1.0 / (1.0 - sum_pi_sq);

    // Load edges
    ifstream edge_in(edge_file);
    string line;
    vector<pair<string, string>> edges;
    map<string, double> branch_map;

    while (getline(edge_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string parent, child;
        double branch_len;
        iss >> parent >> child >> branch_len;
        edges.push_back({parent, child});
        branch_map[child] = branch_len;

        if (!tree.name_to_id.count(parent)) {
            int id = tree.node_names.size();
            tree.name_to_id[parent] = id;
            tree.node_names.push_back(parent);
        }
        if (!tree.name_to_id.count(child)) {
            int id = tree.node_names.size();
            tree.name_to_id[child] = id;
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
        int parent_id = tree.name_to_id[edges[i].first];
        int child_id = tree.name_to_id[edges[i].second];
        tree.parent_map[child_id] = parent_id;
        tree.branch_lengths[child_id] = branch_map[edges[i].second];
        tree.children[parent_id].push_back(child_id);
        tree.edge_parent[i] = parent_id;
        tree.edge_child[i] = child_id;
    }

    // Find root
    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) {
            tree.root_node = i;
            break;
        }
    }

    // Load taxon order
    ifstream tax_in(taxon_file);
    getline(tax_in, line);  // Skip header
    while (getline(tax_in, line)) {
        if (!line.empty()) {
            size_t comma_pos = line.find(',');
            if (comma_pos != string::npos) {
                string taxon_name = line.substr(0, comma_pos);
                tree.taxon_names.push_back(taxon_name);
            }
        }
    }
    tax_in.close();
    tree.num_taxa = tree.taxon_names.size();

    // Build taxon to node mapping
    tree.taxon_to_node.resize(tree.num_taxa);
    tree.node_to_taxon.resize(tree.num_nodes, -1);
    for (int t = 0; t < tree.num_taxa; t++) {
        if (tree.name_to_id.count(tree.taxon_names[t])) {
            int node = tree.name_to_id.at(tree.taxon_names[t]);
            tree.taxon_to_node[t] = node;
            tree.node_to_taxon[node] = t;
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

    // Build post-order traversal of edges
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
    vector<vector<int>> edge_children(tree.num_edges);

    for (int e = 0; e < tree.num_edges; e++) {
        int child_node = tree.edge_child[e];
        for (int node_child : tree.children[child_node]) {
            if (node_to_edge[node_child] >= 0) {
                edge_children[e].push_back(node_to_edge[node_child]);
            }
        }
    }

    for (int e : tree.postorder_edges) {
        int child_node = tree.edge_child[e];

        // Check if child_node is a taxon
        if (tree.node_to_taxon[child_node] >= 0) {
            tree.edge_subtree_taxa[e].push_back(tree.node_to_taxon[child_node]);
        }

        // Add taxa from child edges
        for (int child_e : edge_children[e]) {
            for (int taxon : tree.edge_subtree_taxa[child_e]) {
                tree.edge_subtree_taxa[e].push_back(taxon);
            }
        }

        // Sort for consistent signature
        sort(tree.edge_subtree_taxa[e].begin(), tree.edge_subtree_taxa[e].end());
    }

    // Determine which edges to cache (>99% reuse rate heuristic: subtree size <= 10)
    tree.edge_should_cache.resize(tree.num_edges, false);
    for (int e = 0; e < tree.num_edges; e++) {
        // Cache edges with small subtrees (high reuse rate)
        tree.edge_should_cache[e] = (tree.edge_subtree_taxa[e].size() <= 10);
    }

    return tree;
}

// Standard Felsenstein pruning (no memoization)
double compute_ll_no_memo(const Tree& tree) {
    double total_ll = 0.0;

    vector<array<double, 4>> node_likelihood(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

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

        // Post-order traversal
        for (int e : tree.postorder_edges) {
            int parent = tree.edge_parent[e];
            int child = tree.edge_child[e];
            double t = tree.branch_lengths[child];

            // Compute transition matrix
            double exp_term = exp(-tree.f81_mu * t);
            array<array<double, 4>, 4> P;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j) {
                        P[i][j] = exp_term + (1.0 - exp_term) * tree.root_prior[j];
                    } else {
                        P[i][j] = (1.0 - exp_term) * tree.root_prior[j];
                    }
                }
            }

            // Compute message from child to parent
            array<double, 4> msg;
            for (int i = 0; i < 4; i++) {
                msg[i] = 0.0;
                for (int j = 0; j < 4; j++) {
                    msg[i] += P[i][j] * node_likelihood[child][j];
                }
            }

            // Multiply into parent
            for (int i = 0; i < 4; i++) {
                node_likelihood[parent][i] *= msg[i];
            }
            node_scale[parent] += node_scale[child];

            // Scale if needed
            double max_val = *max_element(node_likelihood[parent].begin(), node_likelihood[parent].end());
            if (max_val > 0.0 && max_val < 1e-100) {
                for (int i = 0; i < 4; i++) {
                    node_likelihood[parent][i] /= max_val;
                }
                node_scale[parent] += log(max_val);
            }
        }

        // Compute likelihood at root
        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_prior[i] * node_likelihood[tree.root_node][i];
        }

        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }

    return total_ll;
}

// Memoized Felsenstein pruning
double compute_ll_memoized(const Tree& tree, vector<long>& stats) {
    double total_ll = 0.0;

    // Cache: edge -> signature -> (likelihood, scale)
    vector<unordered_map<uint64_t, CacheEntry>> cache(tree.num_edges);

    vector<array<double, 4>> node_likelihood(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    long cache_hits = 0;
    long cache_misses = 0;
    long cache_skips = 0;

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

        // Post-order traversal with memoization
        for (int e : tree.postorder_edges) {
            int parent = tree.edge_parent[e];
            int child = tree.edge_child[e];

            bool use_cache = tree.edge_should_cache[e];
            uint64_t sig_hash = 0;

            if (use_cache) {
                // Compute signature hash
                vector<uint8_t> sig;
                sig.reserve(tree.edge_subtree_taxa[e].size());
                for (int taxon : tree.edge_subtree_taxa[e]) {
                    sig.push_back((uint8_t)tree.pattern_data[p * tree.num_taxa + taxon]);
                }
                sig_hash = hash_signature(sig);

                // Cache lookup
                auto it = cache[e].find(sig_hash);
                if (it != cache[e].end()) {
                    // Cache hit - use cached message
                    cache_hits++;
                    const CacheEntry& entry = it->second;
                    for (int i = 0; i < 4; i++) {
                        node_likelihood[parent][i] *= entry.likelihood[i];
                    }
                    node_scale[parent] += entry.log_scale;

                    // Scale parent if needed
                    double max_val = *max_element(node_likelihood[parent].begin(),
                                                   node_likelihood[parent].end());
                    if (max_val > 0.0 && max_val < 1e-100) {
                        for (int i = 0; i < 4; i++) {
                            node_likelihood[parent][i] /= max_val;
                        }
                        node_scale[parent] += log(max_val);
                    }
                    continue;
                }
                cache_misses++;
            } else {
                cache_skips++;
            }

            // Cache miss or not caching - compute normally
            double t = tree.branch_lengths[child];

            // Compute transition matrix
            double exp_term = exp(-tree.f81_mu * t);
            array<array<double, 4>, 4> P;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j) {
                        P[i][j] = exp_term + (1.0 - exp_term) * tree.root_prior[j];
                    } else {
                        P[i][j] = (1.0 - exp_term) * tree.root_prior[j];
                    }
                }
            }

            // Compute message from child to parent
            array<double, 4> msg;
            double child_scale = node_scale[child];
            for (int i = 0; i < 4; i++) {
                msg[i] = 0.0;
                for (int j = 0; j < 4; j++) {
                    msg[i] += P[i][j] * node_likelihood[child][j];
                }
            }

            // Store in cache if caching this edge
            if (use_cache) {
                CacheEntry entry;
                entry.likelihood = msg;
                entry.log_scale = child_scale;
                cache[e][sig_hash] = entry;
            }

            // Multiply into parent
            for (int i = 0; i < 4; i++) {
                node_likelihood[parent][i] *= msg[i];
            }
            node_scale[parent] += child_scale;

            // Scale if needed
            double max_val = *max_element(node_likelihood[parent].begin(),
                                           node_likelihood[parent].end());
            if (max_val > 0.0 && max_val < 1e-100) {
                for (int i = 0; i < 4; i++) {
                    node_likelihood[parent][i] /= max_val;
                }
                node_scale[parent] += log(max_val);
            }
        }

        // Compute likelihood at root
        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_prior[i] * node_likelihood[tree.root_node][i];
        }

        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }

    stats[0] = cache_hits;
    stats[1] = cache_misses;
    stats[2] = cache_skips;

    // Count cache memory usage
    size_t cache_memory = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        cache_memory += cache[e].size() * sizeof(CacheEntry);
    }
    stats[3] = cache_memory;

    return total_ll;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " -e <edges> -p <patterns> -x <taxon_order> -b <basecomp> [-t max_subtree_size]" << endl;
        cerr << "\nTests memoized Felsenstein pruning algorithm:" << endl;
        cerr << "  1. Validates correctness (memoized LL == non-memoized LL)" << endl;
        cerr << "  2. Measures cache hit rates" << endl;
        cerr << "  3. Compares performance" << endl;
        return 1;
    }

    string edge_file, pattern_file, taxon_file, basecomp_file;
    int max_subtree_size = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) basecomp_file = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) max_subtree_size = atoi(argv[++i]);
    }

    cout << "=== Memoized Felsenstein Pruning Test ===" << endl;

    // Load data
    cout << "\nLoading tree and patterns..." << endl;
    Tree tree = load_tree(edge_file, pattern_file, taxon_file, basecomp_file);

    // Update cache decision based on threshold
    int edges_cached = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        tree.edge_should_cache[e] = ((int)tree.edge_subtree_taxa[e].size() <= max_subtree_size);
        if (tree.edge_should_cache[e]) edges_cached++;
    }

    cout << "Loaded: " << tree.num_nodes << " nodes, " << tree.num_edges << " edges, "
         << tree.num_taxa << " taxa, " << tree.num_patterns << " patterns" << endl;
    cout << "Edges to cache (subtree size <= " << max_subtree_size << "): "
         << edges_cached << " / " << tree.num_edges << endl;

    // Test 1: Non-memoized version
    cout << "\n--- Non-Memoized Pruning ---" << endl;
    auto start1 = high_resolution_clock::now();
    double ll_no_memo = compute_ll_no_memo(tree);
    auto end1 = high_resolution_clock::now();
    auto time1 = duration_cast<milliseconds>(end1 - start1);
    cout << "Log-likelihood: " << fixed << setprecision(8) << ll_no_memo << endl;
    cout << "Time: " << time1.count() << " ms" << endl;

    // Test 2: Memoized version
    cout << "\n--- Memoized Pruning ---" << endl;
    vector<long> stats(4);
    auto start2 = high_resolution_clock::now();
    double ll_memo = compute_ll_memoized(tree, stats);
    auto end2 = high_resolution_clock::now();
    auto time2 = duration_cast<milliseconds>(end2 - start2);
    cout << "Log-likelihood: " << fixed << setprecision(8) << ll_memo << endl;
    cout << "Time: " << time2.count() << " ms" << endl;

    // Statistics
    long cache_hits = stats[0];
    long cache_misses = stats[1];
    long cache_skips = stats[2];
    size_t cache_memory = stats[3];

    cout << "\n--- Cache Statistics ---" << endl;
    cout << "Cache hits: " << cache_hits << endl;
    cout << "Cache misses: " << cache_misses << endl;
    cout << "Cache skips (non-cached edges): " << cache_skips << endl;
    double hit_rate = 100.0 * cache_hits / (cache_hits + cache_misses);
    cout << "Hit rate: " << fixed << setprecision(2) << hit_rate << "%" << endl;
    cout << "Cache memory: " << cache_memory / 1024.0 << " KB" << endl;

    // Validation
    cout << "\n--- Validation ---" << endl;
    double diff = abs(ll_no_memo - ll_memo);
    cout << "LL difference: " << scientific << diff << endl;
    if (diff < 1e-8) {
        cout << "CORRECTNESS VERIFIED: Memoized LL matches non-memoized!" << endl;
    } else {
        cout << "WARNING: Results differ!" << endl;
    }

    // Performance comparison
    cout << "\n--- Performance ---" << endl;
    double speedup = (double)time1.count() / time2.count();
    cout << "Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;

    if (speedup > 1.0) {
        cout << "Memoization provides " << (speedup - 1) * 100 << "% improvement" << endl;
    } else {
        cout << "Memoization overhead exceeds benefit (consider larger subtree threshold)" << endl;
    }

    return 0;
}
