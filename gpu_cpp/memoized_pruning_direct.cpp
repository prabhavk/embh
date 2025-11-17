// Direct-indexing memoized pruning (no hash overhead)
// For small subtrees, use direct array lookup instead of hash tables
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <functional>

using namespace std;
using namespace std::chrono;

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

void load_tree(Tree& tree, const string& edge_file, const string& pattern_file,
               const string& taxon_file, const string& basecomp_file) {

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

    tree.root_node = -1;
    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) { tree.root_node = i; break; }
    }

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

    tree.taxon_to_node.resize(tree.num_taxa);
    tree.node_to_taxon.resize(tree.num_nodes, -1);
    for (int i = 0; i < tree.num_taxa; i++) {
        if (tree.name_to_id.count(tree.taxon_names[i])) {
            tree.taxon_to_node[i] = tree.name_to_id[tree.taxon_names[i]];
            tree.node_to_taxon[tree.taxon_to_node[i]] = i;
        }
    }

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

    vector<bool> visited(tree.num_nodes, false);
    function<void(int)> visit = [&](int node) {
        visited[node] = true;
        for (int child : tree.children[node]) {
            if (!visited[child]) visit(child);
        }
        for (size_t i = 0; i < edges.size(); i++) {
            if (tree.edge_child[i] == node) {
                tree.postorder_edges.push_back(i);
                break;
            }
        }
    };
    visit(tree.root_node);

    tree.edge_subtree_taxa.resize(tree.num_edges);
    // Subtree computation will be done separately
}

void compute_subtree_taxa(Tree& tree) {
    // Build node_to_edge mapping
    vector<int> node_to_edge(tree.num_nodes, -1);
    for (int e = 0; e < tree.num_edges; e++) {
        node_to_edge[tree.edge_child[e]] = e;
    }

    // Build edge_children
    vector<vector<int>> edge_children(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        int child_node = tree.edge_child[e];
        for (int node_child : tree.children[child_node]) {
            if (node_to_edge[node_child] >= 0) {
                edge_children[e].push_back(node_to_edge[node_child]);
            }
        }
    }

    // Process in post-order
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

        // Sort
        sort(tree.edge_subtree_taxa[e].begin(), tree.edge_subtree_taxa[e].end());
    }
}

struct CacheEntry {
    array<double, 4> likelihood;
    double log_scale;
};

// Baseline: no memoization
double compute_ll_baseline(Tree& tree,
                          vector<array<array<double,4>,4>>& P_matrices) {
    double total_ll = 0.0;
    vector<array<double, 4>> likelihood(tree.num_nodes);
    vector<double> log_scale(tree.num_nodes);

    for (int p = 0; p < tree.num_patterns; p++) {
        for (int n = 0; n < tree.num_nodes; n++) log_scale[n] = 0.0;

        for (int n = 0; n < tree.num_nodes; n++) {
            if (tree.node_to_taxon[n] >= 0) {
                int base = tree.pattern_data[p * tree.num_taxa + tree.node_to_taxon[n]];
                for (int i = 0; i < 4; i++) likelihood[n][i] = (i == base) ? 1.0 : 0.0;
            } else {
                // Internal nodes start with likelihood 1.0
                for (int i = 0; i < 4; i++) likelihood[n][i] = 1.0;
            }
        }

        for (int e : tree.postorder_edges) {
            int child = tree.edge_child[e];
            int parent = tree.edge_parent[e];

            if (tree.children[child].empty()) continue;

            for (int i = 0; i < 4; i++) {
                double sum = 0.0;
                for (int j = 0; j < 4; j++) {
                    sum += P_matrices[e][i][j] * likelihood[child][j];
                }
                likelihood[child][i] = sum;
            }

            double max_val = *max_element(likelihood[child].begin(), likelihood[child].end());
            if (max_val > 0 && max_val != 1.0) {
                for (int i = 0; i < 4; i++) likelihood[child][i] /= max_val;
                log_scale[child] += log(max_val);
            }

            if (tree.node_to_taxon[parent] < 0) {
                for (int i = 0; i < 4; i++) likelihood[parent][i] *= likelihood[child][i];
                log_scale[parent] += log_scale[child];
            }
        }

        int root = tree.root_node;
        double root_ll = 0.0;
        for (int i = 0; i < 4; i++) {
            root_ll += tree.root_prior[i] * likelihood[root][i];
        }
        total_ll += tree.pattern_weights[p] * (log(root_ll) + log_scale[root]);
    }

    return total_ll;
}

// Compute direct index from pattern bases for edge's subtree taxa
inline int compute_direct_index(const int* pattern_bases, const vector<int>& taxa) {
    int idx = 0;
    for (int t : taxa) {
        idx = (idx << 2) | pattern_bases[t];
    }
    return idx;
}

// Memoized with direct indexing (no hash)
double compute_ll_direct_memo(Tree& tree,
                             vector<array<array<double,4>,4>>& P_matrices,
                             int max_subtree_size) {
    double total_ll = 0.0;
    vector<array<double, 4>> likelihood(tree.num_nodes);
    vector<double> log_scale(tree.num_nodes);

    // Identify which edges to cache based on subtree size
    vector<bool> cache_edge(tree.num_edges, false);
    vector<int> cache_size(tree.num_edges, 0);

    for (int e = 0; e < tree.num_edges; e++) {
        int st_size = tree.edge_subtree_taxa[e].size();
        if (st_size > 0 && st_size <= max_subtree_size) {
            cache_edge[e] = true;
            cache_size[e] = 1 << (2 * st_size); // 4^st_size
        }
    }

    // Allocate direct-indexed cache for each edge
    vector<vector<CacheEntry>> cache(tree.num_edges);
    vector<vector<bool>> cache_valid(tree.num_edges);

    size_t total_cache_entries = 0;
    for (int e = 0; e < tree.num_edges; e++) {
        if (cache_edge[e]) {
            cache[e].resize(cache_size[e]);
            cache_valid[e].resize(cache_size[e], false);
            total_cache_entries += cache_size[e];
        }
    }

    long long total_ops = 0, cache_hits = 0;

    for (int p = 0; p < tree.num_patterns; p++) {
        for (int n = 0; n < tree.num_nodes; n++) log_scale[n] = 0.0;

        const int* pattern_bases = &tree.pattern_data[p * tree.num_taxa];

        for (int n = 0; n < tree.num_nodes; n++) {
            if (tree.node_to_taxon[n] >= 0) {
                int base = pattern_bases[tree.node_to_taxon[n]];
                for (int i = 0; i < 4; i++) likelihood[n][i] = (i == base) ? 1.0 : 0.0;
            } else {
                // Internal nodes start with likelihood 1.0
                for (int i = 0; i < 4; i++) likelihood[n][i] = 1.0;
            }
        }

        for (int e : tree.postorder_edges) {
            int child = tree.edge_child[e];
            int parent = tree.edge_parent[e];

            if (tree.children[child].empty()) continue;

            total_ops++;

            bool used_cache = false;
            if (cache_edge[e]) {
                int idx = compute_direct_index(pattern_bases, tree.edge_subtree_taxa[e]);
                if (cache_valid[e][idx]) {
                    likelihood[child] = cache[e][idx].likelihood;
                    log_scale[child] = cache[e][idx].log_scale;
                    cache_hits++;
                    used_cache = true;
                }
            }

            if (!used_cache) {
                for (int i = 0; i < 4; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < 4; j++) {
                        sum += P_matrices[e][i][j] * likelihood[child][j];
                    }
                    likelihood[child][i] = sum;
                }

                double max_val = *max_element(likelihood[child].begin(), likelihood[child].end());
                if (max_val > 0 && max_val != 1.0) {
                    for (int i = 0; i < 4; i++) likelihood[child][i] /= max_val;
                    log_scale[child] += log(max_val);
                }

                if (cache_edge[e]) {
                    int idx = compute_direct_index(pattern_bases, tree.edge_subtree_taxa[e]);
                    cache[e][idx].likelihood = likelihood[child];
                    cache[e][idx].log_scale = log_scale[child];
                    cache_valid[e][idx] = true;
                }
            }

            if (tree.node_to_taxon[parent] < 0) {
                for (int i = 0; i < 4; i++) likelihood[parent][i] *= likelihood[child][i];
                log_scale[parent] += log_scale[child];
            }
        }

        int root = tree.root_node;
        double root_ll = 0.0;
        for (int i = 0; i < 4; i++) {
            root_ll += tree.root_prior[i] * likelihood[root][i];
        }
        total_ll += tree.pattern_weights[p] * (log(root_ll) + log_scale[root]);
    }

    cout << "Cache entries: " << total_cache_entries << " ("
         << (total_cache_entries * sizeof(CacheEntry) / 1024.0) << " KB)\n";
    cout << "Total ops: " << total_ops << ", Hits: " << cache_hits
         << " (" << (100.0 * cache_hits / total_ops) << "%)\n";

    return total_ll;
}

int main(int argc, char* argv[]) {
    string edge_file = "../data/tree_edges.txt";
    string pattern_file = "../data/patterns.pat";
    string taxon_file = "../data/patterns.taxon_order";
    string basecomp_file = "../data/patterns.basecomp";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0) basecomp_file = argv[++i];
    }

    cout << "=== Direct-Index Memoized Pruning Test ===\n\n";
    cout.flush();

    Tree tree;
    load_tree(tree, edge_file, pattern_file, taxon_file, basecomp_file);
    cout << "Loaded: " << tree.num_patterns << " patterns, "
         << tree.num_edges << " edges, " << tree.num_taxa << " taxa\n";
    cout.flush();

    compute_subtree_taxa(tree);
    cout << "\n";

    // Pre-compute transition matrices
    vector<array<array<double,4>,4>> P_matrices(tree.num_edges);
    for (int e = 0; e < tree.num_edges; e++) {
        double t = tree.branch_lengths[tree.edge_child[e]];
        double exp_term = exp(-tree.f81_mu * t);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    P_matrices[e][i][j] = exp_term + (1 - exp_term) * tree.root_prior[j];
                } else {
                    P_matrices[e][i][j] = (1 - exp_term) * tree.root_prior[j];
                }
            }
        }
    }

    cout << "--- Baseline ---\n";
    auto start = high_resolution_clock::now();
    double baseline_ll = compute_ll_baseline(tree, P_matrices);
    auto end = high_resolution_clock::now();
    auto baseline_time = duration_cast<milliseconds>(end - start).count();
    cout << "LL: " << fixed << setprecision(8) << baseline_ll << "\n";
    cout << "Time: " << baseline_time << " ms\n\n";

    // Test different max subtree sizes
    for (int max_size : {4, 5, 6, 7}) {
        cout << "--- Direct Memo (max subtree = " << max_size << ") ---\n";
        start = high_resolution_clock::now();
        double memo_ll = compute_ll_direct_memo(tree, P_matrices, max_size);
        end = high_resolution_clock::now();
        auto memo_time = duration_cast<milliseconds>(end - start).count();
        cout << "LL: " << fixed << setprecision(8) << memo_ll << "\n";
        cout << "Time: " << memo_time << " ms\n";
        cout << "Speedup: " << fixed << setprecision(2) << (double)baseline_time / memo_time << "x\n";
        cout << "LL diff: " << scientific << setprecision(8) << abs(baseline_ll - memo_ll) << "\n\n";
    }

    return 0;
}
