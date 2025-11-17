// Memoized Belief Propagation for Clique Tree
// Caches upward and downward messages based on subtree pattern signatures
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

struct CliqueTree {
    int num_nodes, num_cliques, num_taxa, num_patterns;
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;
    vector<vector<int>> children;

    // Clique structure: clique c represents edge (x_c, y_c) where x_c is parent
    vector<int> clique_x_node;  // Parent node of edge
    vector<int> clique_y_node;  // Child node of edge
    vector<int> clique_parent;  // Parent clique (-1 for root cliques)
    vector<vector<int>> clique_children;
    vector<int> postorder_cliques;

    vector<int> pattern_data;
    vector<int> pattern_weights;
    vector<string> taxon_names;
    vector<int> node_to_taxon;  // -1 if not a taxon

    // For memoization: subtree taxa for each clique
    vector<vector<int>> clique_subtree_taxa;
    vector<vector<int>> clique_complement_taxa;

    array<double, 4> root_prior;
    double f81_mu;
    int root_node;
};

// FNV-1a hash for signatures
uint64_t hash_signature(const vector<uint8_t>& sig) {
    uint64_t hash = 14695981039346656037ULL;
    for (uint8_t val : sig) {
        hash ^= val;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void load_tree(CliqueTree& tree, const string& edge_file, const string& pattern_file,
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

    // Load tree edges
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
    tree.num_cliques = edges.size();
    tree.parent_map.resize(tree.num_nodes, -1);
    tree.branch_lengths.resize(tree.num_nodes, 0.0);
    tree.children.resize(tree.num_nodes);
    tree.clique_x_node.resize(tree.num_cliques);
    tree.clique_y_node.resize(tree.num_cliques);

    for (size_t i = 0; i < edges.size(); i++) {
        int pid = tree.name_to_id[edges[i].first];
        int cid = tree.name_to_id[edges[i].second];
        tree.parent_map[cid] = pid;
        tree.branch_lengths[cid] = branch_map[edges[i].second];
        tree.children[pid].push_back(cid);
        tree.clique_x_node[i] = pid;
        tree.clique_y_node[i] = cid;
    }

    tree.root_node = -1;
    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) { tree.root_node = i; break; }
    }

    // Load taxa
    ifstream tax_in(taxon_file);
    getline(tax_in, line);  // Skip header
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
            int node = tree.name_to_id[tree.taxon_names[t]];
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

    // Build clique tree structure
    // Parent clique of clique c is the clique whose y_node == x_node of c
    tree.clique_parent.resize(tree.num_cliques, -1);
    tree.clique_children.resize(tree.num_cliques);

    map<int, int> node_to_clique;  // Maps y_node -> clique index
    for (int c = 0; c < tree.num_cliques; c++) {
        node_to_clique[tree.clique_y_node[c]] = c;
    }

    for (int c = 0; c < tree.num_cliques; c++) {
        int x = tree.clique_x_node[c];
        if (node_to_clique.count(x)) {
            int parent_c = node_to_clique[x];
            tree.clique_parent[c] = parent_c;
            tree.clique_children[parent_c].push_back(c);
        }
    }

    // Build postorder traversal of cliques
    function<void(int)> build_postorder = [&](int c) {
        for (int child_c : tree.clique_children[c]) {
            build_postorder(child_c);
        }
        tree.postorder_cliques.push_back(c);
    };

    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_parent[c] == -1) {
            build_postorder(c);
        }
    }

    // Compute subtree taxa for each clique (leaves in y's subtree)
    tree.clique_subtree_taxa.resize(tree.num_cliques);
    for (int c : tree.postorder_cliques) {
        int y = tree.clique_y_node[c];

        // If y is a taxon, add it
        if (tree.node_to_taxon[y] >= 0) {
            tree.clique_subtree_taxa[c].push_back(tree.node_to_taxon[y]);
        }

        // Add taxa from child cliques
        for (int child_c : tree.clique_children[c]) {
            for (int t : tree.clique_subtree_taxa[child_c]) {
                tree.clique_subtree_taxa[c].push_back(t);
            }
        }

        sort(tree.clique_subtree_taxa[c].begin(), tree.clique_subtree_taxa[c].end());
    }

    // Compute complement taxa (all taxa NOT in subtree)
    set<int> all_taxa;
    for (int t = 0; t < tree.num_taxa; t++) all_taxa.insert(t);

    tree.clique_complement_taxa.resize(tree.num_cliques);
    for (int c = 0; c < tree.num_cliques; c++) {
        set<int> subtree_set(tree.clique_subtree_taxa[c].begin(),
                              tree.clique_subtree_taxa[c].end());
        for (int t : all_taxa) {
            if (!subtree_set.count(t)) {
                tree.clique_complement_taxa[c].push_back(t);
            }
        }
    }
}

struct MessageCacheEntry {
    array<double, 4> message;  // Message to parent (upward) or from parent (downward)
    double log_scale;
};

// Baseline: No memoization - Standard Felsenstein pruning on nodes
double compute_ll_baseline(const CliqueTree& tree) {
    double total_ll = 0.0;

    // Node likelihoods: L[n][state] = P(subtree data | n = state)
    vector<array<double, 4>> node_L(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    // Pre-compute postorder of nodes
    vector<int> postorder_nodes;
    vector<bool> visited(tree.num_nodes, false);
    function<void(int)> visit = [&](int n) {
        visited[n] = true;
        for (int c : tree.children[n]) {
            if (!visited[c]) visit(c);
        }
        postorder_nodes.push_back(n);
    };
    visit(tree.root_node);

    for (int p = 0; p < tree.num_patterns; p++) {
        const int* pattern = &tree.pattern_data[p * tree.num_taxa];

        // Reset
        for (int n = 0; n < tree.num_nodes; n++) {
            node_scale[n] = 0.0;
            for (int i = 0; i < 4; i++) node_L[n][i] = 1.0;
        }

        // Initialize leaves
        for (int n = 0; n < tree.num_nodes; n++) {
            if (tree.node_to_taxon[n] >= 0) {
                int base = pattern[tree.node_to_taxon[n]];
                if (base < 4) {
                    // Standard base (A=0, C=1, G=2, T=3)
                    for (int i = 0; i < 4; i++) {
                        node_L[n][i] = (i == base) ? 1.0 : 0.0;
                    }
                } else {
                    // Gap or unknown (base >= 4): all states equally likely
                    for (int i = 0; i < 4; i++) {
                        node_L[n][i] = 1.0;
                    }
                }
            }
        }

        // Upward pass
        for (int n : postorder_nodes) {
            if (tree.children[n].empty()) continue;  // Leaf

            // n is internal: multiply likelihoods from children
            for (int child : tree.children[n]) {
                double t = tree.branch_lengths[child];
                double exp_term = exp(-tree.f81_mu * t);

                // Compute sum_y P[x][y] * L_child[y] for each x
                array<double, 4> child_contrib;
                for (int xstate = 0; xstate < 4; xstate++) {
                    double sum = 0.0;
                    for (int ystate = 0; ystate < 4; ystate++) {
                        double P_xy = (xstate == ystate) ?
                            (exp_term + (1 - exp_term) * tree.root_prior[ystate]) :
                            ((1 - exp_term) * tree.root_prior[ystate]);
                        sum += P_xy * node_L[child][ystate];
                    }
                    child_contrib[xstate] = sum;
                }

                // Multiply into parent's likelihood
                for (int i = 0; i < 4; i++) {
                    node_L[n][i] *= child_contrib[i];
                }
                node_scale[n] += node_scale[child];
            }

            // Scale to prevent underflow - scale whenever max < 1
            double max_val = *max_element(node_L[n].begin(), node_L[n].end());
            if (max_val > 0 && max_val != 1.0) {
                for (int i = 0; i < 4; i++) node_L[n][i] /= max_val;
                node_scale[n] += log(max_val);
            }
        }

        // Root likelihood
        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_prior[i] * node_L[tree.root_node][i];
        }

        if (p == 0) {
            cerr << "P0: root_L = [" << node_L[tree.root_node][0] << ", "
                 << node_L[tree.root_node][1] << ", " << node_L[tree.root_node][2] << ", "
                 << node_L[tree.root_node][3] << "], scale=" << node_scale[tree.root_node]
                 << ", ll=" << ll << endl;
            cerr << "root_prior = [" << tree.root_prior[0] << ", " << tree.root_prior[1]
                 << ", " << tree.root_prior[2] << ", " << tree.root_prior[3] << "]" << endl;
        }

        if (ll <= 0 || isnan(ll) || isinf(ll)) {
            cerr << "P" << p << " has ll=" << ll << endl;
            return -INFINITY;
        }

        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }

    return total_ll;
}

// Memoized version with hash table caching on edges
// Cache the upward message for each edge (contribution to parent likelihood)
double compute_ll_memoized(const CliqueTree& tree, long long& cache_hits, long long& cache_misses) {
    double total_ll = 0.0;
    cache_hits = 0;
    cache_misses = 0;

    // Cache for each edge (clique): hash -> (message, scale)
    // message[i] = sum_y P[i][y] * L_child[y]
    vector<unordered_map<uint64_t, MessageCacheEntry>> edge_cache(tree.num_cliques);

    vector<array<double, 4>> node_L(tree.num_nodes);
    vector<double> node_scale(tree.num_nodes);

    // Pre-compute postorder of nodes
    vector<int> postorder_nodes;
    vector<bool> visited(tree.num_nodes, false);
    function<void(int)> visit_node = [&](int n) {
        visited[n] = true;
        for (int c : tree.children[n]) {
            if (!visited[c]) visit_node(c);
        }
        postorder_nodes.push_back(n);
    };
    visit_node(tree.root_node);

    // Map from child node to edge index
    map<int, int> child_to_edge;
    for (int c = 0; c < tree.num_cliques; c++) {
        child_to_edge[tree.clique_y_node[c]] = c;
    }

    for (int p = 0; p < tree.num_patterns; p++) {
        const int* pattern = &tree.pattern_data[p * tree.num_taxa];

        // Reset
        for (int n = 0; n < tree.num_nodes; n++) {
            node_scale[n] = 0.0;
            for (int i = 0; i < 4; i++) node_L[n][i] = 1.0;
        }

        // Initialize leaves
        for (int n = 0; n < tree.num_nodes; n++) {
            if (tree.node_to_taxon[n] >= 0) {
                int base = pattern[tree.node_to_taxon[n]];
                if (base < 4) {
                    for (int i = 0; i < 4; i++) {
                        node_L[n][i] = (i == base) ? 1.0 : 0.0;
                    }
                } else {
                    // Gap or unknown
                    for (int i = 0; i < 4; i++) {
                        node_L[n][i] = 1.0;
                    }
                }
            }
        }

        // Upward pass with memoization on edges
        for (int n : postorder_nodes) {
            if (tree.children[n].empty()) continue;  // Leaf

            // n is internal: process each child edge
            for (int child : tree.children[n]) {
                int edge_idx = child_to_edge[child];

                // Compute signature for this edge's subtree
                vector<uint8_t> sig;
                sig.reserve(tree.clique_subtree_taxa[edge_idx].size());
                for (int t : tree.clique_subtree_taxa[edge_idx]) {
                    sig.push_back((uint8_t)pattern[t]);
                }
                uint64_t sig_hash = hash_signature(sig);

                array<double, 4> child_contrib;
                double child_scale_contrib;

                // Check cache
                auto it = edge_cache[edge_idx].find(sig_hash);
                if (it != edge_cache[edge_idx].end()) {
                    // Cache hit
                    cache_hits++;
                    child_contrib = it->second.message;
                    child_scale_contrib = it->second.log_scale;
                } else {
                    // Cache miss - compute contribution
                    cache_misses++;

                    double t = tree.branch_lengths[child];
                    double exp_term = exp(-tree.f81_mu * t);

                    for (int xstate = 0; xstate < 4; xstate++) {
                        double sum = 0.0;
                        for (int ystate = 0; ystate < 4; ystate++) {
                            double P_xy = (xstate == ystate) ?
                                (exp_term + (1 - exp_term) * tree.root_prior[ystate]) :
                                ((1 - exp_term) * tree.root_prior[ystate]);
                            sum += P_xy * node_L[child][ystate];
                        }
                        child_contrib[xstate] = sum;
                    }
                    child_scale_contrib = node_scale[child];

                    // Cache result
                    edge_cache[edge_idx][sig_hash] = {child_contrib, child_scale_contrib};
                }

                // Multiply into parent's likelihood
                for (int i = 0; i < 4; i++) {
                    node_L[n][i] *= child_contrib[i];
                }
                node_scale[n] += child_scale_contrib;
            }

            // Scale to prevent underflow
            double max_val = *max_element(node_L[n].begin(), node_L[n].end());
            if (max_val > 0 && max_val < 1e-100) {
                for (int i = 0; i < 4; i++) node_L[n][i] /= max_val;
                node_scale[n] += log(max_val);
            }
        }

        // Root likelihood
        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_prior[i] * node_L[tree.root_node][i];
        }

        total_ll += tree.pattern_weights[p] * (log(ll) + node_scale[tree.root_node]);
    }

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

    cout << "=== Memoized Belief Propagation Test ===\n\n";

    CliqueTree tree;
    load_tree(tree, edge_file, pattern_file, taxon_file, basecomp_file);
    cout << "Loaded: " << tree.num_patterns << " patterns, "
         << tree.num_cliques << " cliques, " << tree.num_taxa << " taxa\n";

    // Show subtree size distribution
    map<int, int> st_count;
    for (int c = 0; c < tree.num_cliques; c++) {
        st_count[tree.clique_subtree_taxa[c].size()]++;
    }
    cout << "Subtree sizes: ";
    for (auto& kv : st_count) {
        cout << kv.first << ":" << kv.second << " ";
    }
    cout << "\n";

    // Debug: check tree structure
    int num_leaves = 0;
    for (int n = 0; n < tree.num_nodes; n++) {
        if (tree.children[n].empty()) num_leaves++;
    }
    cout << "Root: " << tree.root_node << ", Leaves: " << num_leaves << "\n\n";

    // Baseline
    cout << "--- Baseline (No Memoization) ---\n";
    auto start = high_resolution_clock::now();
    double baseline_ll = compute_ll_baseline(tree);
    auto end = high_resolution_clock::now();
    auto baseline_time = duration_cast<milliseconds>(end - start).count();
    cout << "LL: " << fixed << setprecision(8) << baseline_ll << "\n";
    cout << "Time: " << baseline_time << " ms\n\n";

    // Memoized
    cout << "--- Memoized ---\n";
    long long hits, misses;
    start = high_resolution_clock::now();
    double memo_ll = compute_ll_memoized(tree, hits, misses);
    end = high_resolution_clock::now();
    auto memo_time = duration_cast<milliseconds>(end - start).count();
    cout << "LL: " << fixed << setprecision(8) << memo_ll << "\n";
    cout << "Time: " << memo_time << " ms\n";
    cout << "Cache hits: " << hits << ", misses: " << misses
         << " (" << (100.0 * hits / (hits + misses)) << "% hit rate)\n";
    cout << "Speedup: " << fixed << setprecision(2) << (double)baseline_time / memo_time << "x\n";
    cout << "LL difference: " << scientific << setprecision(8) << abs(baseline_ll - memo_ll) << "\n";

    if (abs(baseline_ll - memo_ll) < 1e-8) {
        cout << "\nCORRECTNESS VERIFIED!\n";
    } else {
        cout << "\nWARNING: Results don't match!\n";
    }

    return 0;
}
