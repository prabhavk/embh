/**
 * GPU-Accelerated EMBH using Clique Tree Data Structure
 *
 * Key differences from edge-based version (embh_aitken_gpu.cu):
 * - Uses clique tree structure where each clique = (parent_node, child_node)
 * - Each clique has a 4x4 belief matrix (joint distribution)
 * - Message passing is between cliques sharing separator variables
 * - After calibration, clique beliefs directly give expected counts for E-step
 *
 * Advantage: More principled graphical model framework, easier to extend
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <array>
#include <string>
#include <algorithm>
#include <functional>
#include <tuple>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <queue>

using namespace std;

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
             << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
}

// GPU constants
#define MAX_CLIQUES 256
#define MAX_CHILDREN 10
#define MAX_NODES 512

// GPU tree structure (stored in constant memory)
struct GPUCliqueTree {
    int num_cliques;
    int num_nodes;
    int num_leaves;
    int root_node;

    // Clique data
    int clique_x_node[MAX_CLIQUES];
    int clique_y_node[MAX_CLIQUES];
    int clique_parent[MAX_CLIQUES];
    int clique_children[MAX_CLIQUES][MAX_CHILDREN];
    int clique_num_children[MAX_CLIQUES];

    // Traversal orders
    int postorder_edges[MAX_CLIQUES][2];
    int num_postorder_edges;
    int preorder_edges[MAX_CLIQUES][2];
    int num_preorder_edges;

    // Model parameters
    int leaf_taxon_index[MAX_NODES];
    double root_prior[4];
};

__constant__ GPUCliqueTree d_tree;

// Atomic add for double (for older GPUs without native support)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use built-in atomicAdd for double on Pascal and newer
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// GPU kernel: Compute F81 transition matrix
__device__ void compute_f81_transition(double branch_length, double f81_mu, const double* root_prior, double* P) {
    double t = branch_length * f81_mu;
    double exp_term = exp(-t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i * 4 + j] = exp_term + root_prior[j] * (1.0 - exp_term);
            } else {
                P[i * 4 + j] = root_prior[j] * (1.0 - exp_term);
            }
        }
    }
}

/**
 * GPU Kernel: Full belief propagation and expected count computation
 *
 * Each thread processes one pattern:
 * 1. Initialize clique potentials
 * 2. Upward pass (leaves to root)
 * 3. Downward pass (root to leaves)
 * 4. Compute calibrated beliefs (= expected counts for this pattern)
 * 5. Accumulate weighted counts
 */
__global__ void clique_tree_estep_kernel(
    const int* patterns,           // [num_patterns * num_leaves]
    const int* weights,            // [num_patterns]
    const double* branch_lengths,  // [num_nodes]
    double f81_mu,
    int num_patterns,
    double* expected_counts,       // [num_cliques * 16] output (accumulated)
    double* pattern_lls            // [num_patterns] output
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    // Local storage
    double init_pot[MAX_CLIQUES][16];
    double msg_from_parent[MAX_CLIQUES][4];
    double msg_from_child[MAX_CLIQUES][MAX_CHILDREN][4];
    double scale_from_parent[MAX_CLIQUES];
    double scale_from_child[MAX_CLIQUES][MAX_CHILDREN];

    int num_cliques = d_tree.num_cliques;
    int num_leaves = d_tree.num_leaves;
    const int* pattern = &patterns[pid * num_leaves];

    // 1. Initialize clique potentials
    for (int c = 0; c < num_cliques; c++) {
        int y_node = d_tree.clique_y_node[c];
        double P[16];
        compute_f81_transition(branch_lengths[y_node], f81_mu, d_tree.root_prior, P);

        bool y_is_leaf = (y_node < num_leaves);
        int leaf_state = y_is_leaf ? pattern[y_node] : -1;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (y_is_leaf && leaf_state < 4) {
                    init_pot[c][i * 4 + j] = (j == leaf_state) ? P[i * 4 + j] : 0.0;
                } else {
                    init_pot[c][i * 4 + j] = P[i * 4 + j];
                }
            }
        }

        scale_from_parent[c] = 0.0;
        for (int ch = 0; ch < MAX_CHILDREN; ch++) {
            scale_from_child[c][ch] = 0.0;
            for (int d = 0; d < 4; d++) {
                msg_from_child[c][ch][d] = 1.0;
            }
        }
        for (int d = 0; d < 4; d++) {
            msg_from_parent[c][d] = 1.0;
        }
    }

    // 2. Upward pass: leaves to root
    for (int e = 0; e < d_tree.num_postorder_edges; e++) {
        int parent_clique = d_tree.postorder_edges[e][0];
        int child_clique = d_tree.postorder_edges[e][1];

        double factor[16];
        for (int i = 0; i < 16; i++) factor[i] = init_pot[child_clique][i];

        double logScalingFactor = 0.0;

        // Multiply by messages from child's children
        for (int ch = 0; ch < d_tree.clique_num_children[child_clique]; ch++) {
            int gc = d_tree.clique_children[child_clique][ch];
            int child_y = d_tree.clique_y_node[child_clique];
            int gc_x = d_tree.clique_x_node[gc];
            int gc_y = d_tree.clique_y_node[gc];

            bool gc_shares_child_y = (child_y == gc_x || child_y == gc_y);

            if (gc_shares_child_y) {
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        factor[x * 4 + y] *= msg_from_child[child_clique][ch][y];
            } else {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        factor[x * 4 + y] *= msg_from_child[child_clique][ch][x];
            }

            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            for (int i = 0; i < 16; i++) factor[i] /= maxv;
            logScalingFactor += log(maxv) + scale_from_child[child_clique][ch];
        }

        // Marginalize
        double message[4] = {0.0, 0.0, 0.0, 0.0};
        int child_y = d_tree.clique_y_node[child_clique];
        int parent_x = d_tree.clique_x_node[parent_clique];
        int parent_y = d_tree.clique_y_node[parent_clique];
        bool child_y_shared = (child_y == parent_x || child_y == parent_y);

        if (child_y_shared) {
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    message[y] += factor[x * 4 + y];
        } else {
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++)
                    message[x] += factor[x * 4 + y];
        }

        double maxv = 0.0;
        for (int d = 0; d < 4; d++) if (message[d] > maxv) maxv = message[d];
        for (int d = 0; d < 4; d++) message[d] /= maxv;
        logScalingFactor += log(maxv);

        // Store in parent
        int child_idx = -1;
        for (int ch = 0; ch < d_tree.clique_num_children[parent_clique]; ch++) {
            if (d_tree.clique_children[parent_clique][ch] == child_clique) {
                child_idx = ch;
                break;
            }
        }

        if (child_idx >= 0) {
            for (int d = 0; d < 4; d++)
                msg_from_child[parent_clique][child_idx][d] = message[d];
            scale_from_child[parent_clique][child_idx] = logScalingFactor;
        }
    }

    // 3. Downward pass: root to leaves
    for (int e = 0; e < d_tree.num_preorder_edges; e++) {
        int parent_clique = d_tree.preorder_edges[e][0];
        int child_clique = d_tree.preorder_edges[e][1];

        double factor[16];
        for (int i = 0; i < 16; i++) factor[i] = init_pot[parent_clique][i];

        double logScalingFactor = 0.0;

        // Multiply by parent's parent message
        if (d_tree.clique_parent[parent_clique] >= 0) {
            int gp = d_tree.clique_parent[parent_clique];
            int parent_y = d_tree.clique_y_node[parent_clique];
            int gp_x = d_tree.clique_x_node[gp];
            int gp_y = d_tree.clique_y_node[gp];
            bool gp_shares_parent_y = (parent_y == gp_x || parent_y == gp_y);

            if (gp_shares_parent_y) {
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        factor[x * 4 + y] *= msg_from_parent[parent_clique][y];
            } else {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        factor[x * 4 + y] *= msg_from_parent[parent_clique][x];
            }

            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            for (int i = 0; i < 16; i++) factor[i] /= maxv;
            logScalingFactor += log(maxv) + scale_from_parent[parent_clique];
        }

        // Multiply by sibling messages
        for (int ch = 0; ch < d_tree.clique_num_children[parent_clique]; ch++) {
            int sibling = d_tree.clique_children[parent_clique][ch];
            if (sibling == child_clique) continue;

            int parent_y = d_tree.clique_y_node[parent_clique];
            int sib_x = d_tree.clique_x_node[sibling];
            int sib_y = d_tree.clique_y_node[sibling];
            bool sib_shares_parent_y = (parent_y == sib_x || parent_y == sib_y);

            if (sib_shares_parent_y) {
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        factor[x * 4 + y] *= msg_from_child[parent_clique][ch][y];
            } else {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        factor[x * 4 + y] *= msg_from_child[parent_clique][ch][x];
            }

            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            for (int i = 0; i < 16; i++) factor[i] /= maxv;
            logScalingFactor += log(maxv) + scale_from_child[parent_clique][ch];
        }

        // Marginalize
        double message[4] = {0.0, 0.0, 0.0, 0.0};
        int parent_y = d_tree.clique_y_node[parent_clique];
        int child_x = d_tree.clique_x_node[child_clique];
        int child_y = d_tree.clique_y_node[child_clique];
        bool parent_y_shared = (parent_y == child_x || parent_y == child_y);

        if (parent_y_shared) {
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    message[y] += factor[x * 4 + y];
        } else {
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++)
                    message[x] += factor[x * 4 + y];
        }

        double maxv = 0.0;
        for (int d = 0; d < 4; d++) if (message[d] > maxv) maxv = message[d];
        for (int d = 0; d < 4; d++) message[d] /= maxv;
        logScalingFactor += log(maxv);

        for (int d = 0; d < 4; d++)
            msg_from_parent[child_clique][d] = message[d];
        scale_from_parent[child_clique] = logScalingFactor;
    }

    // 4. Compute calibrated beliefs and accumulate expected counts
    double clique_scales[MAX_CLIQUES];
    int weight = weights[pid];

    for (int c = 0; c < num_cliques; c++) {
        double belief[16];
        for (int i = 0; i < 16; i++) belief[i] = init_pot[c][i];

        double logScale = 0.0;

        // Multiply by parent message
        if (d_tree.clique_parent[c] >= 0) {
            int parent = d_tree.clique_parent[c];
            int c_y = d_tree.clique_y_node[c];
            int p_x = d_tree.clique_x_node[parent];
            int p_y = d_tree.clique_y_node[parent];
            bool parent_shares_c_y = (c_y == p_x || c_y == p_y);

            if (parent_shares_c_y) {
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        belief[x * 4 + y] *= msg_from_parent[c][y];
            } else {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        belief[x * 4 + y] *= msg_from_parent[c][x];
            }

            double sum = 0.0;
            for (int i = 0; i < 16; i++) sum += belief[i];
            for (int i = 0; i < 16; i++) belief[i] /= sum;
            logScale += log(sum) + scale_from_parent[c];
        }

        // Multiply by children messages
        for (int ch = 0; ch < d_tree.clique_num_children[c]; ch++) {
            int child = d_tree.clique_children[c][ch];
            int c_y = d_tree.clique_y_node[c];
            int ch_x = d_tree.clique_x_node[child];
            int ch_y = d_tree.clique_y_node[child];
            bool child_shares_c_y = (c_y == ch_x || c_y == ch_y);

            if (child_shares_c_y) {
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        belief[x * 4 + y] *= msg_from_child[c][ch][y];
            } else {
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                        belief[x * 4 + y] *= msg_from_child[c][ch][x];
            }

            double sum = 0.0;
            for (int i = 0; i < 16; i++) sum += belief[i];
            for (int i = 0; i < 16; i++) belief[i] /= sum;
            logScale += log(sum) + scale_from_child[c][ch];
        }

        // Final normalization
        double sum = 0.0;
        for (int i = 0; i < 16; i++) sum += belief[i];
        for (int i = 0; i < 16; i++) belief[i] /= sum;
        logScale += log(sum);

        // Accumulate expected counts (belief is P(X, Y | data))
        for (int i = 0; i < 16; i++) {
            atomicAdd(&expected_counts[c * 16 + i], belief[i] * weight);
        }

        // Store this clique's scale
        clique_scales[c] = logScale;
    }

    // 5. Compute log-likelihood at a clique that contains the root node
    // Find which clique contains the root node as its X variable
    int root_clique = -1;
    for (int c = 0; c < d_tree.num_cliques; c++) {
        if (d_tree.clique_x_node[c] == d_tree.root_node) {
            root_clique = c;
            break;
        }
    }

    // If root is not found as X, use clique 0 (this shouldn't happen for proper rooted tree)
    if (root_clique < 0) root_clique = 0;

    double belief[16];
    for (int i = 0; i < 16; i++) belief[i] = init_pot[root_clique][i];

    if (d_tree.clique_parent[root_clique] >= 0) {
        // Has parent - multiply by parent message
        for (int x = 0; x < 4; x++)
            for (int y = 0; y < 4; y++)
                belief[x * 4 + y] *= msg_from_parent[root_clique][x];
    }

    for (int ch = 0; ch < d_tree.clique_num_children[root_clique]; ch++) {
        int c_y = d_tree.clique_y_node[root_clique];
        int child = d_tree.clique_children[root_clique][ch];
        int ch_x = d_tree.clique_x_node[child];
        int ch_y = d_tree.clique_y_node[child];
        bool child_shares_c_y = (c_y == ch_x || c_y == ch_y);

        if (child_shares_c_y) {
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++)
                    belief[x * 4 + y] *= msg_from_child[root_clique][ch][y];
        } else {
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    belief[x * 4 + y] *= msg_from_child[root_clique][ch][x];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < 16; i++) sum += belief[i];
    for (int i = 0; i < 16; i++) belief[i] /= sum;

    // Get marginal over root variable (which is X of root_clique)
    double marginal_root[4] = {0.0, 0.0, 0.0, 0.0};
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
            marginal_root[x] += belief[x * 4 + y];

    double site_ll = 0.0;
    for (int d = 0; d < 4; d++) {
        site_ll += d_tree.root_prior[d] * marginal_root[d];
    }

    pattern_lls[pid] = (clique_scales[root_clique] + log(site_ll)) * weight;
}

// CPU-side tree structure
struct CliqueTreeEM {
    int num_nodes;
    int num_leaves;
    int root_node;
    string root_name;  // Name of root node

    // Node properties
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;

    // Clique tree structure
    int num_cliques;
    vector<int> clique_x_node;
    vector<int> clique_y_node;
    vector<int> clique_parent;
    vector<vector<int>> clique_children;

    // Traversal orders
    vector<pair<int, int>> postorder_edges;
    vector<pair<int, int>> preorder_edges;

    // Pattern data
    int num_taxa;
    int num_patterns;
    vector<int> pattern_data;  // flattened [num_patterns * num_leaves]
    vector<int> pattern_weights;
    vector<string> taxon_names;

    // F81 model parameters
    array<double, 4> root_prior;
    double f81_mu;
};

void compute_f81_mu(CliqueTreeEM& tree) {
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += tree.root_prior[i] * tree.root_prior[i];
    }
    tree.f81_mu = 1.0 / (1.0 - sum_pi_sq);
}

CliqueTreeEM load_clique_tree(const string& edge_file, const string& pattern_file,
                               const string& taxon_file, const string& basecomp_file,
                               const string& specified_root = "") {
    CliqueTreeEM tree;

    // Load taxon order
    ifstream tin(taxon_file);
    string line;
    getline(tin, line);  // header
    map<string, int> taxon_to_pos;
    while (getline(tin, line)) {
        size_t c = line.find(',');
        if (c != string::npos) {
            string taxon = line.substr(0, c);
            int pos = stoi(line.substr(c + 1));
            tree.taxon_names.push_back(taxon);
            taxon_to_pos[taxon] = pos;
        }
    }
    tin.close();
    tree.num_taxa = tree.taxon_names.size();

    // Load edges as undirected graph (adjacency list with lengths)
    map<string, vector<pair<string, double>>> adj_list;
    set<string> all_nodes;

    ifstream ein(edge_file);
    while (getline(ein, line)) {
        istringstream iss(line);
        string p, ch;
        double len;
        iss >> p >> ch >> len;
        // Store both directions (undirected tree)
        adj_list[p].push_back({ch, len});
        adj_list[ch].push_back({p, len});
        all_nodes.insert(p);
        all_nodes.insert(ch);
    }
    ein.close();

    // Identify leaves and internal nodes
    vector<string> leaf_names;
    vector<string> internal_names;

    for (const auto& name : all_nodes) {
        if (name.substr(0, 2) == "h_") {
            internal_names.push_back(name);
        } else {
            leaf_names.push_back(name);
        }
    }

    tree.num_leaves = leaf_names.size();
    tree.num_nodes = all_nodes.size();

    // Assign node IDs based on taxon order
    leaf_names.clear();
    leaf_names.resize(tree.num_leaves);
    for (const auto& p : taxon_to_pos) {
        if (p.second < tree.num_leaves) {
            leaf_names[p.second] = p.first;
            tree.name_to_id[p.first] = p.second;
        }
    }

    sort(internal_names.begin(), internal_names.end());
    int next_id = tree.num_leaves;
    for (const auto& name : internal_names) {
        tree.name_to_id[name] = next_id++;
    }

    // Find root - either specified or auto-detect (node with most neighbors for internal, or degree 1)
    string root_name;
    if (!specified_root.empty()) {
        if (all_nodes.find(specified_root) == all_nodes.end()) {
            cerr << "Error: Specified root '" << specified_root << "' not found in tree!" << endl;
            exit(1);
        }
        root_name = specified_root;
        cout << "Using specified root: " << root_name << endl;
    } else {
        // Find the natural root (the one with no parent in original file direction)
        // This is the node that appears only as a parent, never as a child
        set<string> appears_as_parent, appears_as_child;
        ifstream ein2(edge_file);
        while (getline(ein2, line)) {
            istringstream iss(line);
            string p, ch;
            double len;
            iss >> p >> ch >> len;
            appears_as_parent.insert(p);
            appears_as_child.insert(ch);
        }
        ein2.close();

        for (const auto& name : appears_as_parent) {
            if (appears_as_child.find(name) == appears_as_child.end()) {
                root_name = name;
                break;
            }
        }
        cout << "Auto-detected root: " << root_name << endl;
    }

    tree.root_node = tree.name_to_id[root_name];
    tree.root_name = root_name;

    // Build parent map and branch lengths using BFS from root
    tree.parent_map.resize(tree.num_nodes, -1);
    tree.branch_lengths.resize(tree.num_nodes, 0.0);
    tree.node_names.resize(tree.num_nodes);

    for (const auto& name : all_nodes) {
        tree.node_names[tree.name_to_id[name]] = name;
    }

    // BFS to establish parent-child relationships from the chosen root
    queue<string> q;
    set<string> visited;
    q.push(root_name);
    visited.insert(root_name);

    while (!q.empty()) {
        string current = q.front();
        q.pop();
        int current_id = tree.name_to_id[current];

        for (const auto& neighbor : adj_list[current]) {
            string neighbor_name = neighbor.first;
            double edge_length = neighbor.second;

            if (visited.find(neighbor_name) == visited.end()) {
                visited.insert(neighbor_name);
                int neighbor_id = tree.name_to_id[neighbor_name];

                // current is parent of neighbor
                tree.parent_map[neighbor_id] = current_id;
                tree.branch_lengths[neighbor_id] = edge_length;

                q.push(neighbor_name);
            }
        }
    }

    // Build clique tree
    map<int, int> child_to_clique;
    tree.num_cliques = 0;

    for (int child = 0; child < tree.num_nodes; child++) {
        int parent = tree.parent_map[child];
        if (parent != -1) {
            tree.clique_x_node.push_back(parent);
            tree.clique_y_node.push_back(child);
            tree.clique_parent.push_back(-1);
            tree.clique_children.push_back(vector<int>());
            child_to_clique[child] = tree.num_cliques;
            tree.num_cliques++;
        }
    }

    // Connect cliques
    for (int c = 0; c < tree.num_cliques; c++) {
        int x_node = tree.clique_x_node[c];
        if (child_to_clique.count(x_node)) {
            int parent_clique = child_to_clique[x_node];
            tree.clique_parent[c] = parent_clique;
            tree.clique_children[parent_clique].push_back(c);
        }
    }

    // Handle multiple root cliques
    vector<int> root_cliques;
    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_parent[c] == -1) {
            root_cliques.push_back(c);
        }
    }

    if (root_cliques.size() > 1) {
        int main_root = root_cliques[0];
        for (size_t i = 1; i < root_cliques.size(); i++) {
            tree.clique_parent[root_cliques[i]] = main_root;
            tree.clique_children[main_root].push_back(root_cliques[i]);
        }
    }

    // Build traversal orders
    vector<bool> clique_visited(tree.num_cliques, false);
    int root_clique = root_cliques[0];

    function<void(int)> postorder = [&](int c) {
        clique_visited[c] = true;
        for (int child : tree.clique_children[c]) {
            if (!clique_visited[child]) {
                postorder(child);
                tree.postorder_edges.push_back({c, child});
            }
        }
    };

    postorder(root_clique);

    tree.preorder_edges = tree.postorder_edges;
    reverse(tree.preorder_edges.begin(), tree.preorder_edges.end());

    // Load base composition
    ifstream bin(basecomp_file);
    tree.root_prior = {0.25, 0.25, 0.25, 0.25};
    while (getline(bin, line)) {
        if (line.empty() || !isdigit(line[0])) continue;
        istringstream iss(line);
        int state;
        double freq;
        if (iss >> state >> freq) {
            if (state >= 0 && state < 4) {
                tree.root_prior[state] = freq;
            }
        }
    }
    bin.close();

    // Normalize
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += tree.root_prior[i];
    for (int i = 0; i < 4; i++) tree.root_prior[i] /= sum;

    compute_f81_mu(tree);

    // Load patterns
    ifstream pin(pattern_file);
    while (getline(pin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;
        tree.pattern_weights.push_back(weight);

        for (int i = 0; i < tree.num_leaves; i++) {
            int state;
            iss >> state;
            tree.pattern_data.push_back(state);
        }
    }
    pin.close();
    tree.num_patterns = tree.pattern_weights.size();

    return tree;
}

void prepare_gpu_tree(const CliqueTreeEM& tree, GPUCliqueTree& gpu_tree) {
    gpu_tree.num_cliques = tree.num_cliques;
    gpu_tree.num_nodes = tree.num_nodes;
    gpu_tree.num_leaves = tree.num_leaves;
    gpu_tree.root_node = tree.root_node;

    for (int c = 0; c < tree.num_cliques; c++) {
        gpu_tree.clique_x_node[c] = tree.clique_x_node[c];
        gpu_tree.clique_y_node[c] = tree.clique_y_node[c];
        gpu_tree.clique_parent[c] = tree.clique_parent[c];
        gpu_tree.clique_num_children[c] = tree.clique_children[c].size();

        for (int ch = 0; ch < (int)tree.clique_children[c].size(); ch++) {
            gpu_tree.clique_children[c][ch] = tree.clique_children[c][ch];
        }
    }

    gpu_tree.num_postorder_edges = tree.postorder_edges.size();
    for (int e = 0; e < gpu_tree.num_postorder_edges; e++) {
        gpu_tree.postorder_edges[e][0] = tree.postorder_edges[e].first;
        gpu_tree.postorder_edges[e][1] = tree.postorder_edges[e].second;
    }

    gpu_tree.num_preorder_edges = tree.preorder_edges.size();
    for (int e = 0; e < gpu_tree.num_preorder_edges; e++) {
        gpu_tree.preorder_edges[e][0] = tree.preorder_edges[e].first;
        gpu_tree.preorder_edges[e][1] = tree.preorder_edges[e].second;
    }

    for (int i = 0; i < 4; i++) {
        gpu_tree.root_prior[i] = tree.root_prior[i];
    }
}

// M-step: Update branch lengths from expected counts
// For F81 model: P(X,Y) = pi_X * P(Y|X) where P(Y|X) = exp(-t*mu) if X==Y, pi_Y*(1-exp(-t*mu)) otherwise
// Expected count N(X,Y) gives estimate of P(X,Y|data)
// We use: t = -log(1 - P_diff/beta) / mu where P_diff = sum_{i!=j} N(i,j) / total, beta = 1 - sum_i pi_i^2
void m_step_branch_lengths(CliqueTreeEM& tree, const vector<double>& counts) {
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += tree.root_prior[i] * tree.root_prior[i];
    }
    double beta = 1.0 - sum_pi_sq;  // Maximum expected difference under F81

    for (int c = 0; c < tree.num_cliques; c++) {
        int child_node = tree.clique_y_node[c];

        // Get expected counts for this clique
        double total = 0.0;
        double diff = 0.0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double n = counts[c * 16 + i * 4 + j];
                total += n;
                if (i != j) diff += n;
            }
        }

        // Estimate branch length: P_diff = beta * (1 - exp(-t*mu))
        // => t = -log(1 - P_diff/beta) / mu
        double p_diff = diff / total;

        // Safeguard: p_diff must be < beta
        if (p_diff >= beta * 0.999) {
            p_diff = beta * 0.999;  // Cap at 99.9% of maximum
        }
        if (p_diff < 1e-10) {
            p_diff = 1e-10;  // Minimum difference
        }

        double new_branch_length = -log(1.0 - p_diff / beta) / tree.f81_mu;

        // Safeguard: keep branch length in reasonable range
        new_branch_length = max(1e-8, min(10.0, new_branch_length));

        tree.branch_lengths[child_node] = new_branch_length;
    }
}

// Aitken's delta-squared method for log-likelihood convergence estimation
// Returns estimated limiting LL value: ℓ∞ = ℓ(k) + (Δℓ(k))² / Δ²ℓ(k)
double aitken_ll_estimate(double ll_k_minus_2, double ll_k_minus_1, double ll_k) {
    double delta_ll_k_minus_1 = ll_k_minus_1 - ll_k_minus_2;
    double delta_ll_k = ll_k - ll_k_minus_1;
    double delta2_ll = delta_ll_k - delta_ll_k_minus_1;

    // Check for convergence (denominator near zero means already converged)
    if (abs(delta2_ll) < 1e-12) {
        return ll_k;
    }

    // Aitken's estimate
    double ll_infinity = ll_k + (delta_ll_k * delta_ll_k) / (-delta2_ll);

    return ll_infinity;
}

// Aitken's parameter acceleration with damping
// theta* = theta(k) + damping * delta_theta(k) / (1 - lambda)
// lambda = (delta_theta(k+1) . delta_theta(k)) / (delta_theta(k) . delta_theta(k))
struct AitkenState {
    vector<double> theta_k_minus_2;  // Parameters at iteration k-2
    vector<double> theta_k_minus_1;  // Parameters at iteration k-1
    vector<double> theta_k;          // Parameters at iteration k (current)
    double ll_k_minus_2;
    double ll_k_minus_1;
    double ll_k;
    int iteration;
    bool has_history;
};

// Structure to hold acceleration info for logging
struct AitkenAccelInfo {
    int iteration;
    double lambda;
    double accel_factor;
    bool applied;
};

double aitken_accelerate_parameters(
    CliqueTreeEM& tree,
    AitkenState& state,
    double damping = 0.8,      // Damping factor (0 < damping <= 1)
    double min_improvement = 1e-4  // Minimum LL improvement to keep accelerated params
) {
    if (!state.has_history || state.iteration < 3) {
        return -1.0;  // Need at least 3 iterations to apply acceleration
    }

    int num_params = state.theta_k.size();

    // Compute delta vectors
    vector<double> delta_k_minus_1(num_params);
    vector<double> delta_k(num_params);

    for (int i = 0; i < num_params; i++) {
        delta_k_minus_1[i] = state.theta_k_minus_1[i] - state.theta_k_minus_2[i];
        delta_k[i] = state.theta_k[i] - state.theta_k_minus_1[i];
    }

    // Compute lambda = (delta_k . delta_k_minus_1) / (delta_k_minus_1 . delta_k_minus_1)
    double dot_k_km1 = 0.0;
    double dot_km1_km1 = 0.0;

    for (int i = 0; i < num_params; i++) {
        dot_k_km1 += delta_k[i] * delta_k_minus_1[i];
        dot_km1_km1 += delta_k_minus_1[i] * delta_k_minus_1[i];
    }

    if (dot_km1_km1 < 1e-20) {
        return -1.0;  // No change in parameters, skip acceleration
    }

    double lambda = dot_k_km1 / dot_km1_km1;

    // Check if lambda is valid for acceleration (should be 0 < lambda < 1)
    if (lambda <= 0.0 || lambda >= 1.0) {
        // Invalid lambda - EM not contracting or converged
        cout << "    Aitken: lambda=" << lambda << " (invalid, skipping acceleration)" << endl;
        return lambda;
    }

    // Compute accelerated parameters: theta* = theta(k) + damping * delta_k / (1 - lambda)
    vector<double> theta_accelerated(num_params);
    double acceleration_factor = damping / (1.0 - lambda);

    for (int i = 0; i < num_params; i++) {
        theta_accelerated[i] = state.theta_k[i] + acceleration_factor * delta_k[i];

        // Safeguard: enforce constraints (branch lengths must be positive)
        theta_accelerated[i] = max(1e-8, min(10.0, theta_accelerated[i]));
    }

    cout << "    Aitken: lambda=" << fixed << setprecision(6) << lambda
         << ", accel_factor=" << acceleration_factor << endl;

    // Apply accelerated parameters
    for (int c = 0; c < tree.num_cliques; c++) {
        int child_node = tree.clique_y_node[c];
        tree.branch_lengths[child_node] = theta_accelerated[c];
    }

    // Update state for next iteration
    state.theta_k = theta_accelerated;

    return lambda;
}

// Structure to hold EM results for file output
struct EMResults {
    vector<double> ll_history;
    vector<double> aitken_ll_estimates;
    vector<double> iter_times;
    vector<AitkenAccelInfo> accel_info;
    double total_time;
    bool converged;
    double final_ll;
};

void run_clique_tree_em(CliqueTreeEM& tree, int max_iter, double tol = 1e-6,
                         bool use_aitken_accel = true, double aitken_damping = 0.8,
                         const string& output_file = "") {
    cout << "\n=== GPU EMBH-EM using Clique Tree Structure ===" << endl;
    cout << "Nodes: " << tree.num_nodes << ", Leaves: " << tree.num_leaves << endl;
    cout << "Cliques: " << tree.num_cliques << ", Patterns: " << tree.num_patterns << endl;
    cout << "Root prior: [" << tree.root_prior[0] << ", " << tree.root_prior[1]
         << ", " << tree.root_prior[2] << ", " << tree.root_prior[3] << "]" << endl;
    cout << "F81 mu: " << tree.f81_mu << endl;

    // Debug: sum of branch lengths (should be same regardless of root)
    double total_branch_length = 0.0;
    for (int c = 0; c < tree.num_cliques; c++) {
        int y_node = tree.clique_y_node[c];
        total_branch_length += tree.branch_lengths[y_node];
    }
    cout << "Total branch length: " << fixed << setprecision(6) << total_branch_length << endl;

    cout << "Max iterations: " << max_iter << ", Tolerance: " << scientific << tol << endl;
    cout << "Aitken acceleration: " << (use_aitken_accel ? "enabled" : "disabled");
    if (use_aitken_accel) cout << " (damping=" << fixed << setprecision(2) << aitken_damping << ")";
    cout << endl;

    EMResults results;
    results.converged = false;

    // Prepare GPU data
    GPUCliqueTree gpu_tree;
    prepare_gpu_tree(tree, gpu_tree);

    CUDA_CHECK(cudaMemcpyToSymbol(d_tree, &gpu_tree, sizeof(GPUCliqueTree)));

    // Allocate GPU memory
    int* d_patterns;
    int* d_weights;
    double* d_branch_lengths;
    double* d_expected_counts;
    double* d_pattern_lls;

    CUDA_CHECK(cudaMalloc(&d_patterns, tree.num_patterns * tree.num_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, tree.num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_branch_lengths, tree.num_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_expected_counts, tree.num_cliques * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pattern_lls, tree.num_patterns * sizeof(double)));

    // Copy static data
    CUDA_CHECK(cudaMemcpy(d_patterns, tree.pattern_data.data(),
                          tree.num_patterns * tree.num_leaves * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, tree.pattern_weights.data(),
                          tree.num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    // Launch configuration
    int block_size = 128;
    int num_blocks = (tree.num_patterns + block_size - 1) / block_size;

    // Initialize Aitken state
    AitkenState aitken_state;
    aitken_state.iteration = 0;
    aitken_state.has_history = false;
    aitken_state.theta_k.resize(tree.num_cliques);
    aitken_state.theta_k_minus_1.resize(tree.num_cliques);
    aitken_state.theta_k_minus_2.resize(tree.num_cliques);

    // Store initial branch lengths
    for (int c = 0; c < tree.num_cliques; c++) {
        int child_node = tree.clique_y_node[c];
        aitken_state.theta_k[c] = tree.branch_lengths[child_node];
    }

    vector<double> ll_history;
    vector<double> expected_counts(tree.num_cliques * 16);
    vector<double> pattern_lls(tree.num_patterns);

    auto total_start = chrono::high_resolution_clock::now();

    cout << "\n=== Starting EM Iterations ===" << endl;
    cout << setw(6) << "Iter" << setw(20) << "Log-Likelihood" << setw(16) << "LL Change"
         << setw(20) << "Aitken LL Est" << setw(16) << "Time (ms)" << endl;
    cout << string(78, '-') << endl;

    double prev_ll = -1e20;
    bool converged = false;

    for (int iter = 0; iter < max_iter; iter++) {
        auto iter_start = chrono::high_resolution_clock::now();

        // === E-Step: Compute expected counts given current parameters ===
        CUDA_CHECK(cudaMemcpy(d_branch_lengths, tree.branch_lengths.data(),
                              tree.num_nodes * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_expected_counts, 0, tree.num_cliques * 16 * sizeof(double)));

        clique_tree_estep_kernel<<<num_blocks, block_size>>>(
            d_patterns, d_weights, d_branch_lengths, tree.f81_mu,
            tree.num_patterns, d_expected_counts, d_pattern_lls
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get log-likelihood
        CUDA_CHECK(cudaMemcpy(pattern_lls.data(), d_pattern_lls,
                              tree.num_patterns * sizeof(double), cudaMemcpyDeviceToHost));

        double current_ll = 0.0;
        for (double ll : pattern_lls) current_ll += ll;

        ll_history.push_back(current_ll);

        // Compute LL change
        double ll_change = current_ll - prev_ll;

        // Aitken LL estimation
        double aitken_ll_est = current_ll;
        if (ll_history.size() >= 3) {
            int n = ll_history.size();
            aitken_ll_est = aitken_ll_estimate(ll_history[n-3], ll_history[n-2], ll_history[n-1]);
        }

        auto iter_end = chrono::high_resolution_clock::now();
        double iter_time = chrono::duration<double, milli>(iter_end - iter_start).count();

        // Print iteration info
        cout << setw(6) << iter + 1
             << setw(20) << fixed << setprecision(6) << current_ll
             << setw(16) << scientific << setprecision(4) << ll_change
             << setw(20) << fixed << setprecision(6) << aitken_ll_est
             << setw(16) << setprecision(2) << iter_time << endl;

        // Store results
        results.aitken_ll_estimates.push_back(aitken_ll_est);
        results.iter_times.push_back(iter_time);

        // Check convergence using Aitken estimate
        if (iter >= 2) {
            double estimated_remaining = abs(aitken_ll_est - current_ll);
            if (estimated_remaining < tol || abs(ll_change) < tol * 0.1) {
                cout << "\nConverged! Estimated remaining improvement: " << scientific << estimated_remaining << endl;
                converged = true;
                results.converged = true;
                break;
            }
        }

        // Update Aitken state for parameter acceleration
        aitken_state.theta_k_minus_2 = aitken_state.theta_k_minus_1;
        aitken_state.theta_k_minus_1 = aitken_state.theta_k;

        // Get expected counts
        CUDA_CHECK(cudaMemcpy(expected_counts.data(), d_expected_counts,
                              tree.num_cliques * 16 * sizeof(double), cudaMemcpyDeviceToHost));

        // === M-Step: Update parameters from expected counts ===
        m_step_branch_lengths(tree, expected_counts);

        // Store new parameters
        for (int c = 0; c < tree.num_cliques; c++) {
            int child_node = tree.clique_y_node[c];
            aitken_state.theta_k[c] = tree.branch_lengths[child_node];
        }

        aitken_state.ll_k_minus_2 = aitken_state.ll_k_minus_1;
        aitken_state.ll_k_minus_1 = aitken_state.ll_k;
        aitken_state.ll_k = current_ll;
        aitken_state.iteration = iter + 1;
        aitken_state.has_history = (iter >= 2);

        // === Aitken Parameter Acceleration ===
        if (use_aitken_accel && iter >= 2 && iter % 3 == 2) {
            // Apply every 3rd iteration after warmup
            double lambda = aitken_accelerate_parameters(tree, aitken_state, aitken_damping);
            AitkenAccelInfo info;
            info.iteration = iter + 1;
            info.lambda = lambda;
            info.accel_factor = (lambda > 0 && lambda < 1) ? aitken_damping / (1.0 - lambda) : 0.0;
            info.applied = (lambda > 0 && lambda < 1);
            results.accel_info.push_back(info);
        }

        prev_ll = current_ll;
    }

    auto total_end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double, milli>(total_end - total_start).count();

    // Final E-step to get final LL and counts
    CUDA_CHECK(cudaMemcpy(d_branch_lengths, tree.branch_lengths.data(),
                          tree.num_nodes * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_expected_counts, 0, tree.num_cliques * 16 * sizeof(double)));

    clique_tree_estep_kernel<<<num_blocks, block_size>>>(
        d_patterns, d_weights, d_branch_lengths, tree.f81_mu,
        tree.num_patterns, d_expected_counts, d_pattern_lls
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pattern_lls.data(), d_pattern_lls,
                          tree.num_patterns * sizeof(double), cudaMemcpyDeviceToHost));

    double final_ll = 0.0;
    for (double ll : pattern_lls) final_ll += ll;

    CUDA_CHECK(cudaMemcpy(expected_counts.data(), d_expected_counts,
                          tree.num_cliques * 16 * sizeof(double), cudaMemcpyDeviceToHost));

    cout << "\n=== Final Results ===" << endl;
    cout << "Converged: " << (converged ? "Yes" : "No") << endl;
    cout << "Total iterations: " << ll_history.size() << endl;
    cout << "Total time: " << fixed << setprecision(2) << total_time << " ms" << endl;
    cout << "Final log-likelihood: " << setprecision(10) << final_ll << endl;
    cout << "Initial log-likelihood: " << setprecision(10) << ll_history[0] << endl;
    cout << "LL improvement: " << scientific << setprecision(6) << (final_ll - ll_history[0]) << endl;

    // Print some final branch lengths
    cout << "\nFinal branch lengths (first 5 cliques):" << endl;
    for (int c = 0; c < min(5, tree.num_cliques); c++) {
        int x_node = tree.clique_x_node[c];
        int y_node = tree.clique_y_node[c];
        cout << "  Clique " << c << " (" << x_node << " -> " << y_node << "): "
             << fixed << setprecision(8) << tree.branch_lengths[y_node] << endl;
    }

    // Print expected counts for first few cliques
    cout << "\nExpected counts for first 3 cliques:" << endl;
    for (int c = 0; c < min(3, tree.num_cliques); c++) {
        cout << "Clique " << c << " (" << tree.clique_x_node[c] << " -> " << tree.clique_y_node[c] << "):" << endl;
        double sum = 0.0;
        for (int i = 0; i < 4; i++) {
            cout << "  ";
            for (int j = 0; j < 4; j++) {
                double val = expected_counts[c * 16 + i * 4 + j];
                cout << setw(10) << fixed << setprecision(2) << val << " ";
                sum += val;
            }
            cout << endl;
        }
        cout << "  Sum: " << sum << endl;
    }

    // Store final results
    results.ll_history = ll_history;
    results.total_time = total_time;
    results.final_ll = final_ll;

    // Write results to file if requested
    if (!output_file.empty()) {
        ofstream out(output_file);
        if (out.is_open()) {
            out << "# EM Convergence Results\n";
            out << "# Aitken acceleration: " << (use_aitken_accel ? "enabled" : "disabled") << "\n";
            if (use_aitken_accel) {
                out << "# Aitken damping: " << aitken_damping << "\n";
            }
            out << "# Converged: " << (results.converged ? "yes" : "no") << "\n";
            out << "# Total iterations: " << ll_history.size() << "\n";
            out << "# Total time (ms): " << fixed << setprecision(2) << total_time << "\n";
            out << "# Final LL: " << setprecision(10) << final_ll << "\n";
            out << "\n";

            out << "=== Iteration Data ===\n";
            out << "Iter\tLL\tAitken_LL_Est\tTime_ms\n";
            for (size_t i = 0; i < ll_history.size(); i++) {
                out << (i + 1) << "\t"
                    << fixed << setprecision(10) << ll_history[i] << "\t"
                    << setprecision(10) << results.aitken_ll_estimates[i] << "\t"
                    << setprecision(4) << results.iter_times[i] << "\n";
            }

            if (!results.accel_info.empty()) {
                out << "\n=== Aitken Acceleration Events ===\n";
                out << "Iter\tLambda\tAccel_Factor\tApplied\n";
                for (const auto& info : results.accel_info) {
                    out << info.iteration << "\t"
                        << fixed << setprecision(8) << info.lambda << "\t"
                        << setprecision(8) << info.accel_factor << "\t"
                        << (info.applied ? "yes" : "no") << "\n";
                }
            }

            out.close();
            cout << "\nResults saved to: " << output_file << endl;
        } else {
            cerr << "Warning: Could not open output file: " << output_file << endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_patterns));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_branch_lengths));
    CUDA_CHECK(cudaFree(d_expected_counts));
    CUDA_CHECK(cudaFree(d_pattern_lls));
}

// Save BH parameters to file
void save_bh_parameters(const CliqueTreeEM& tree, const string& filename) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }

    out << fixed << setprecision(15);

    // Header
    out << "# BH Model Parameters\n";
    out << "# Generated by embh_clique_tree_gpu\n\n";

    // Root information
    out << "[ROOT]\n";
    out << "name=" << tree.root_name << "\n";
    out << "node_id=" << tree.root_node << "\n\n";

    // Root probability (pi)
    out << "[ROOT_PROBABILITY]\n";
    for (int i = 0; i < 4; i++) {
        out << tree.root_prior[i];
        if (i < 3) out << ",";
    }
    out << "\n\n";

    // F81 mu parameter
    out << "[F81_MU]\n";
    out << tree.f81_mu << "\n\n";

    // Tree structure
    out << "[TREE_STRUCTURE]\n";
    out << "num_nodes=" << tree.num_nodes << "\n";
    out << "num_leaves=" << tree.num_leaves << "\n";
    out << "num_cliques=" << tree.num_cliques << "\n\n";

    // Node names
    out << "[NODE_NAMES]\n";
    for (int i = 0; i < tree.num_nodes; i++) {
        out << i << "," << tree.node_names[i] << "\n";
    }
    out << "\n";

    // Branch lengths (transition parameters) - stored at child node
    out << "[BRANCH_LENGTHS]\n";
    out << "# parent_node,child_node,branch_length\n";
    for (int child = 0; child < tree.num_nodes; child++) {
        int parent = tree.parent_map[child];
        if (parent != -1) {
            out << tree.node_names[parent] << "," << tree.node_names[child] << "," << tree.branch_lengths[child] << "\n";
        }
    }
    out << "\n";

    // Parent map (for tree reconstruction)
    out << "[PARENT_MAP]\n";
    for (int i = 0; i < tree.num_nodes; i++) {
        out << i << "," << tree.parent_map[i] << "\n";
    }

    out.close();
    cout << "BH parameters saved to: " << filename << endl;
}

// Re-root the tree at a specified node
void reroot_tree(CliqueTreeEM& tree, const string& new_root_name) {
    if (tree.name_to_id.find(new_root_name) == tree.name_to_id.end()) {
        cerr << "Error: Root node '" << new_root_name << "' not found in tree!" << endl;
        exit(1);
    }

    int new_root = tree.name_to_id[new_root_name];
    if (new_root == tree.root_node) {
        return;  // Already rooted at this node
    }

    cout << "Re-rooting tree from " << tree.root_name << " to " << new_root_name << endl;

    // Find path from new root to old root
    vector<int> path;
    int current = new_root;
    while (current != -1) {
        path.push_back(current);
        current = tree.parent_map[current];
    }

    // Reverse edges along the path - branch length stays with edge (child node in original direction)
    // When we reverse parent->child to child->parent, the branch length moves to the new child (old parent)
    for (size_t i = 0; i < path.size() - 1; i++) {
        int child = path[i];
        int parent = path[i + 1];

        // The branch length for edge (parent, child) was stored at child
        // After reversing, edge becomes (child, parent), so length should be at parent
        double edge_length = tree.branch_lengths[child];

        // Swap parent-child relationship
        tree.parent_map[parent] = child;

        // Move branch length to the new child (old parent)
        tree.branch_lengths[parent] = edge_length;
    }

    // New root has no parent and no branch length to parent
    tree.parent_map[new_root] = -1;
    tree.branch_lengths[new_root] = 0.0;

    tree.root_node = new_root;
    tree.root_name = new_root_name;

    // Rebuild clique tree structure
    tree.clique_x_node.clear();
    tree.clique_y_node.clear();
    tree.clique_parent.clear();
    tree.clique_children.clear();

    map<int, int> child_to_clique;
    tree.num_cliques = 0;

    for (int child = 0; child < tree.num_nodes; child++) {
        int parent = tree.parent_map[child];
        if (parent != -1) {
            tree.clique_x_node.push_back(parent);
            tree.clique_y_node.push_back(child);
            tree.clique_parent.push_back(-1);
            tree.clique_children.push_back(vector<int>());
            child_to_clique[child] = tree.num_cliques;
            tree.num_cliques++;
        }
    }

    // Connect cliques
    for (int c = 0; c < tree.num_cliques; c++) {
        int x_node = tree.clique_x_node[c];
        if (child_to_clique.count(x_node)) {
            int parent_clique = child_to_clique[x_node];
            tree.clique_parent[c] = parent_clique;
            tree.clique_children[parent_clique].push_back(c);
        }
    }

    // Rebuild traversal orders
    tree.postorder_edges.clear();
    tree.preorder_edges.clear();

    vector<int> root_cliques;
    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_parent[c] == -1) {
            root_cliques.push_back(c);
        }
    }

    function<void(int)> build_postorder = [&](int clique) {
        for (int child : tree.clique_children[clique]) {
            build_postorder(child);
            tree.postorder_edges.push_back({child, clique});
        }
    };

    for (int root : root_cliques) {
        build_postorder(root);
    }

    for (auto it = tree.postorder_edges.rbegin(); it != tree.postorder_edges.rend(); ++it) {
        tree.preorder_edges.push_back({it->second, it->first});
    }

    cout << "Re-rooted tree has " << tree.num_cliques << " cliques" << endl;
}

int main(int argc, char** argv) {
    string edge_file = "data/tree_edges.txt";
    string pattern_file = "data/patterns_1000.pat";
    string taxon_file = "data/patterns_1000.taxon_order";
    string basecomp_file = "data/patterns_1000.basecomp";
    string output_file = "";
    string root_name = "";  // Empty means use default root
    string save_params_file = "";  // File to save BH parameters
    int max_iter = 100;
    double tol = 1e-6;
    bool use_aitken = true;
    double aitken_damping = 0.8;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--tree" && i + 1 < argc) {
            edge_file = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            pattern_file = argv[++i];
        } else if (arg == "--taxon" && i + 1 < argc) {
            taxon_file = argv[++i];
        } else if (arg == "--basecomp" && i + 1 < argc) {
            basecomp_file = argv[++i];
        } else if (arg == "--iter" && i + 1 < argc) {
            max_iter = atoi(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (arg == "--no-aitken") {
            use_aitken = false;
        } else if (arg == "--aitken-damping" && i + 1 < argc) {
            aitken_damping = atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--root" && i + 1 < argc) {
            root_name = argv[++i];
        } else if (arg == "--save-params" && i + 1 < argc) {
            save_params_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            cout << "GPU EMBH-EM with Aitken Acceleration" << endl;
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --tree FILE       Edge file (default: data/tree_edges.txt)" << endl;
            cout << "  --patterns FILE   Pattern file (default: data/patterns_1000.pat)" << endl;
            cout << "  --taxon FILE      Taxon order file (default: data/patterns_1000.taxon_order)" << endl;
            cout << "  --basecomp FILE   Base composition file (default: data/patterns_1000.basecomp)" << endl;
            cout << "  --iter N          Max EM iterations (default: 100)" << endl;
            cout << "  --tol TOL         Convergence tolerance (default: 1e-6)" << endl;
            cout << "  --no-aitken       Disable Aitken parameter acceleration" << endl;
            cout << "  --aitken-damping D  Damping factor for Aitken acceleration (default: 0.8)" << endl;
            cout << "  --output FILE     Output file for convergence results" << endl;
            cout << "  --root NAME       Root node name (default: auto-detect)" << endl;
            cout << "  --save-params FILE  Save BH parameters to file after EM" << endl;
            cout << "  --help, -h        Show this help message" << endl;
            return 0;
        }
    }

    cout << "Loading clique tree..." << endl;
    CliqueTreeEM tree = load_clique_tree(edge_file, pattern_file, taxon_file, basecomp_file, root_name);

    // Auto-generate output filename with pattern count and root name
    string final_output_file = output_file;
    if (!output_file.empty() && output_file.find("_NPATTERNS") != string::npos) {
        // Replace _NPATTERNS placeholder with actual pattern count and root name
        size_t pos = output_file.find("_NPATTERNS");
        final_output_file = output_file.substr(0, pos) + "_root_" + tree.root_name + "_num_patterns" + to_string(tree.num_patterns) + output_file.substr(pos + 10);
    } else if (!output_file.empty()) {
        // Insert root name and pattern count before file extension
        size_t dot_pos = output_file.rfind('.');
        if (dot_pos != string::npos) {
            final_output_file = output_file.substr(0, dot_pos) + "_root_" + tree.root_name + "_num_patterns" + to_string(tree.num_patterns) + output_file.substr(dot_pos);
        } else {
            final_output_file = output_file + "_root_" + tree.root_name + "_num_patterns" + to_string(tree.num_patterns);
        }
    }

    run_clique_tree_em(tree, max_iter, tol, use_aitken, aitken_damping, final_output_file);

    // Save BH parameters if requested
    if (!save_params_file.empty()) {
        save_bh_parameters(tree, save_params_file);
    }

    return 0;
}
