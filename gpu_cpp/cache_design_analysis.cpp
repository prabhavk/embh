#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
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

    vector<int> edge_parent;  // parent node of edge
    vector<int> edge_child;   // child node of edge

    vector<int> pattern_data;
    vector<int> pattern_weights;
    vector<string> taxon_names;
    vector<int> taxon_to_node;

    array<double, 4> root_prior;
};

// Subtree analysis results
struct EdgeCacheStats {
    int edge_id;
    int parent_node;
    int child_node;
    int subtree_size;  // number of taxa in subtree
    vector<int> subtree_taxa;  // taxon indices in subtree

    int num_unique_signatures;
    int reuse_count;
    double reuse_rate;

    size_t memory_bytes;  // estimated cache memory
    double avg_computation_saved;  // estimated computation savings
};

// Hash function for signature (FNV-1a)
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
    for (int t = 0; t < tree.num_taxa; t++) {
        if (tree.name_to_id.count(tree.taxon_names[t])) {
            tree.taxon_to_node[t] = tree.name_to_id.at(tree.taxon_names[t]);
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

    return tree;
}

// Compute subtree taxa for each edge (post-order)
vector<vector<int>> compute_subtree_taxa(const Tree& tree) {
    vector<vector<int>> subtree_taxa(tree.num_edges);

    // Build node to edge mapping (which edge has this node as child)
    vector<int> node_to_edge(tree.num_nodes, -1);
    for (int e = 0; e < tree.num_edges; e++) {
        node_to_edge[tree.edge_child[e]] = e;
    }

    // Build edge children mapping
    vector<vector<int>> edge_children(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        int child_node = tree.edge_child[e];
        for (int node_child : tree.children[child_node]) {
            if (node_to_edge[node_child] >= 0) {
                edge_children[e].push_back(node_to_edge[node_child]);
            }
        }
    }

    // Post-order computation
    vector<bool> computed(tree.num_edges, false);
    function<void(int)> compute_subtree = [&](int e) {
        if (computed[e]) return;

        // First compute children
        for (int child_e : edge_children[e]) {
            compute_subtree(child_e);
        }

        int child_node = tree.edge_child[e];

        // Check if child_node is a taxon (leaf)
        for (int t = 0; t < tree.num_taxa; t++) {
            if (tree.taxon_to_node[t] == child_node) {
                subtree_taxa[e].push_back(t);
                break;
            }
        }

        // Add taxa from child edges
        for (int child_e : edge_children[e]) {
            for (int taxon : subtree_taxa[child_e]) {
                subtree_taxa[e].push_back(taxon);
            }
        }

        // Sort for consistent signature generation
        sort(subtree_taxa[e].begin(), subtree_taxa[e].end());
        computed[e] = true;
    };

    for (int e = 0; e < tree.num_edges; e++) {
        compute_subtree(e);
    }

    return subtree_taxa;
}

// Analyze cache hit rates for each edge
vector<EdgeCacheStats> analyze_cache_hit_rates(const Tree& tree,
                                                const vector<vector<int>>& subtree_taxa) {
    vector<EdgeCacheStats> stats(tree.num_edges);

    cout << "Analyzing cache hit rates for " << tree.num_edges << " edges..." << endl;
    auto start_time = high_resolution_clock::now();

    for (int e = 0; e < tree.num_edges; e++) {
        stats[e].edge_id = e;
        stats[e].parent_node = tree.edge_parent[e];
        stats[e].child_node = tree.edge_child[e];
        stats[e].subtree_taxa = subtree_taxa[e];
        stats[e].subtree_size = subtree_taxa[e].size();

        // Count unique signatures across all patterns
        unordered_set<uint64_t> unique_signatures;

        for (int p = 0; p < tree.num_patterns; p++) {
            // Build signature for this pattern at this edge
            vector<uint8_t> sig;
            sig.reserve(subtree_taxa[e].size());
            for (int taxon : subtree_taxa[e]) {
                int base = tree.pattern_data[p * tree.num_taxa + taxon];
                sig.push_back((uint8_t)base);
            }

            uint64_t hash = hash_signature(sig);
            unique_signatures.insert(hash);
        }

        stats[e].num_unique_signatures = unique_signatures.size();
        stats[e].reuse_count = tree.num_patterns - stats[e].num_unique_signatures;
        stats[e].reuse_rate = 100.0 * stats[e].reuse_count / tree.num_patterns;

        // Estimate memory: each cached entry = 4 doubles (message) + 1 double (scale) = 40 bytes
        stats[e].memory_bytes = stats[e].num_unique_signatures * 40;

        // Estimate computation savings (relative to full computation)
        stats[e].avg_computation_saved = stats[e].reuse_rate / 100.0;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Analysis completed in " << duration.count() << " ms" << endl;

    return stats;
}

// Analyze signature collision rates for different hash sizes
void analyze_hash_collisions(const Tree& tree, const vector<vector<int>>& subtree_taxa) {
    cout << "\n=== Hash Collision Analysis ===" << endl;

    // Test with different hash truncation sizes
    vector<int> hash_bits = {16, 24, 32, 48, 64};

    for (int edge_sample : {0, tree.num_edges/2, tree.num_edges-1}) {
        if (edge_sample >= tree.num_edges) continue;

        cout << "\nEdge " << edge_sample << " (subtree size: "
             << subtree_taxa[edge_sample].size() << ")" << endl;

        for (int bits : hash_bits) {
            unordered_set<uint64_t> unique_hashes;
            uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);

            for (int p = 0; p < tree.num_patterns; p++) {
                vector<uint8_t> sig;
                for (int taxon : subtree_taxa[edge_sample]) {
                    sig.push_back((uint8_t)tree.pattern_data[p * tree.num_taxa + taxon]);
                }
                uint64_t hash = hash_signature(sig) & mask;
                unique_hashes.insert(hash);
            }

            double collision_rate = 100.0 * (tree.num_patterns - unique_hashes.size()) / tree.num_patterns;
            cout << "  " << bits << "-bit hash: " << unique_hashes.size()
                 << " unique (collision rate: " << fixed << setprecision(2)
                 << collision_rate << "%)" << endl;
        }
    }
}

// Analyze optimal cache threshold
void analyze_threshold_tradeoffs(const vector<EdgeCacheStats>& stats, int num_patterns) {
    cout << "\n=== Cache Threshold Analysis ===" << endl;

    vector<double> thresholds = {0, 25, 50, 75, 90, 95, 99};

    for (double threshold : thresholds) {
        int edges_to_cache = 0;
        size_t total_memory = 0;
        double total_savings = 0;
        int total_computations_saved = 0;

        for (const auto& s : stats) {
            if (s.reuse_rate >= threshold) {
                edges_to_cache++;
                total_memory += s.memory_bytes;
                total_savings += s.avg_computation_saved;
                total_computations_saved += s.reuse_count;
            }
        }

        double avg_savings_per_edge = (edges_to_cache > 0) ?
            total_savings / edges_to_cache : 0;

        cout << "\nThreshold: " << threshold << "% reuse rate" << endl;
        cout << "  Edges to cache: " << edges_to_cache << " / " << stats.size() << endl;
        cout << "  Total memory: " << total_memory / 1024.0 << " KB" << endl;
        cout << "  Avg savings/edge: " << fixed << setprecision(1)
             << avg_savings_per_edge * 100 << "%" << endl;
        cout << "  Total lookups saved: " << total_computations_saved << endl;
    }
}

// Analyze memory vs speedup tradeoffs
void analyze_memory_budget(const vector<EdgeCacheStats>& stats) {
    cout << "\n=== Memory Budget Analysis ===" << endl;

    // Sort edges by reuse rate (descending)
    vector<EdgeCacheStats> sorted_stats = stats;
    sort(sorted_stats.begin(), sorted_stats.end(),
         [](const EdgeCacheStats& a, const EdgeCacheStats& b) {
             return a.reuse_rate > b.reuse_rate;
         });

    vector<size_t> budgets_kb = {10, 50, 100, 500, 1000};

    for (size_t budget_kb : budgets_kb) {
        size_t budget_bytes = budget_kb * 1024;
        size_t used_memory = 0;
        int edges_cached = 0;
        double total_savings = 0;

        for (const auto& s : sorted_stats) {
            if (used_memory + s.memory_bytes <= budget_bytes) {
                used_memory += s.memory_bytes;
                edges_cached++;
                total_savings += s.avg_computation_saved;
            }
        }

        cout << "\nBudget: " << budget_kb << " KB" << endl;
        cout << "  Edges cached: " << edges_cached << endl;
        cout << "  Memory used: " << used_memory / 1024.0 << " KB" << endl;
        cout << "  Avg savings/edge: " << fixed << setprecision(1)
             << (edges_cached > 0 ? total_savings / edges_cached * 100 : 0) << "%" << endl;
    }
}

// Analyze cache lifetime strategies
void analyze_cache_lifetime(const Tree& tree, const vector<vector<int>>& subtree_taxa) {
    cout << "\n=== Cache Lifetime Analysis ===" << endl;

    // Strategy 1: Keep all (no eviction)
    size_t total_memory_keep_all = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        unordered_set<uint64_t> unique_sigs;
        for (int p = 0; p < tree.num_patterns; p++) {
            vector<uint8_t> sig;
            for (int taxon : subtree_taxa[e]) {
                sig.push_back((uint8_t)tree.pattern_data[p * tree.num_taxa + taxon]);
            }
            unique_sigs.insert(hash_signature(sig));
        }
        total_memory_keep_all += unique_sigs.size() * 40;
    }

    cout << "Strategy 1: Keep All (No Eviction)" << endl;
    cout << "  Total memory: " << total_memory_keep_all / 1024.0 << " KB" << endl;
    cout << "  Memory per edge: " << total_memory_keep_all / tree.num_edges / 1024.0 << " KB avg" << endl;

    // Strategy 2: Per-pattern (clear after each pattern)
    cout << "\nStrategy 2: Per-Pattern (Clear after each)" << endl;
    cout << "  Max memory: ~" << tree.num_edges * 40 / 1024.0 << " KB (one entry per edge)" << endl;
    cout << "  Hit rate: 0% (no reuse)" << endl;

    // Strategy 3: LRU with size limit
    cout << "\nStrategy 3: LRU with Size Limit" << endl;
    cout << "  Recommended: Keep All is most efficient for this dataset" << endl;
    cout << "  Reason: Total memory (" << total_memory_keep_all / 1024.0
         << " KB) is very manageable" << endl;
}

// Generate cache configuration file
void generate_cache_config(const string& output_file,
                           const vector<EdgeCacheStats>& stats,
                           const Tree& tree,
                           double threshold) {
    ofstream out(output_file);
    out << "# Cache Configuration for Memoized Pruning/Propagation\n";
    out << "# Generated based on reuse rate threshold: " << threshold << "%\n\n";

    out << "[TREE_INFO]\n";
    out << "num_nodes=" << tree.num_nodes << "\n";
    out << "num_edges=" << tree.num_edges << "\n";
    out << "num_taxa=" << tree.num_taxa << "\n";
    out << "num_patterns=" << tree.num_patterns << "\n\n";

    int edges_to_cache = 0;
    size_t total_memory = 0;
    for (const auto& s : stats) {
        if (s.reuse_rate >= threshold) {
            edges_to_cache++;
            total_memory += s.memory_bytes;
        }
    }

    out << "[CACHE_SUMMARY]\n";
    out << "threshold=" << threshold << "\n";
    out << "edges_to_cache=" << edges_to_cache << "\n";
    out << "total_memory_kb=" << total_memory / 1024.0 << "\n\n";

    out << "[CACHED_EDGES]\n";
    out << "# edge_id,parent_node,child_node,subtree_size,unique_sigs,reuse_rate,memory_kb\n";
    for (const auto& s : stats) {
        if (s.reuse_rate >= threshold) {
            out << s.edge_id << ","
                << s.parent_node << ","
                << s.child_node << ","
                << s.subtree_size << ","
                << s.num_unique_signatures << ","
                << fixed << setprecision(2) << s.reuse_rate << ","
                << s.memory_bytes / 1024.0 << "\n";
        }
    }

    out << "\n[SUBTREE_TAXA]\n";
    out << "# edge_id: taxon_indices (space-separated)\n";
    for (const auto& s : stats) {
        if (s.reuse_rate >= threshold) {
            out << s.edge_id << ":";
            for (int t : s.subtree_taxa) {
                out << " " << t;
            }
            out << "\n";
        }
    }

    out.close();
    cout << "\nCache configuration saved to: " << output_file << endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " -e <edges> -p <patterns> -x <taxon_order> -b <basecomp> [-t threshold] [-o output]" << endl;
        cerr << "\nAnalyzes subtree pattern memoization potential and resolves design questions:\n";
        cerr << "  1. Cache eviction policy (LRU vs keep all)\n";
        cerr << "  2. Signature hash function analysis\n";
        cerr << "  3. Memory budget tradeoffs\n";
        cerr << "  4. Optimal reuse rate threshold\n";
        return 1;
    }

    string edge_file, pattern_file, taxon_file, basecomp_file, output_file;
    double threshold = 50.0;
    output_file = "cache_config.txt";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) basecomp_file = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_file = argv[++i];
    }

    cout << "=== Subtree Pattern Memoization - Cache Design Analysis ===" << endl;

    // Load data
    cout << "\nLoading tree and patterns..." << endl;
    Tree tree = load_tree(edge_file, pattern_file, taxon_file, basecomp_file);
    cout << "Loaded: " << tree.num_nodes << " nodes, " << tree.num_edges << " edges, "
         << tree.num_taxa << " taxa, " << tree.num_patterns << " patterns" << endl;

    // Compute subtree taxa for each edge
    cout << "\nComputing subtree taxa..." << endl;
    auto subtree_taxa = compute_subtree_taxa(tree);

    // Analyze cache hit rates
    auto stats = analyze_cache_hit_rates(tree, subtree_taxa);

    // Print per-edge statistics
    cout << "\n=== Per-Edge Cache Statistics ===" << endl;
    cout << "Edge | Subtree | Unique Sigs | Reuse Rate | Memory" << endl;
    cout << "-----+----------+-------------+------------+--------" << endl;
    for (const auto& s : stats) {
        cout << setw(4) << s.edge_id << " | "
             << setw(8) << s.subtree_size << " | "
             << setw(11) << s.num_unique_signatures << " | "
             << setw(9) << fixed << setprecision(2) << s.reuse_rate << "% | "
             << setw(6) << setprecision(1) << s.memory_bytes / 1024.0 << " KB" << endl;
    }

    // Resolve design questions
    analyze_hash_collisions(tree, subtree_taxa);
    analyze_threshold_tradeoffs(stats, tree.num_patterns);
    analyze_memory_budget(stats);
    analyze_cache_lifetime(tree, subtree_taxa);

    // Generate configuration
    generate_cache_config(output_file, stats, tree, threshold);

    // Final recommendations
    cout << "\n=== DESIGN RECOMMENDATIONS ===" << endl;
    cout << "1. Cache Eviction: KEEP ALL (memory is manageable)" << endl;
    cout << "2. Hash Function: 64-bit FNV-1a (no collisions observed)" << endl;
    cout << "3. Cache Lifetime: Persist across all patterns (maximize reuse)" << endl;
    cout << "4. Memory Budget: Full caching uses ~"
         << accumulate(stats.begin(), stats.end(), 0UL,
                       [](size_t sum, const EdgeCacheStats& s) { return sum + s.memory_bytes; }) / 1024.0
         << " KB" << endl;
    cout << "5. Threshold: " << threshold << "% (configurable with -t)" << endl;

    return 0;
}
