/*
 * clique_ll_consistency_test.cu
 *
 * GPU implementation of clique tree belief propagation matching the C++ code structure.
 *
 * In the C++ code (embh_core.cpp), each clique is a pair of variables (X, Y) where:
 * - X is the parent variable (separator with parent clique)
 * - Y is the child variable
 * - belief is a 4x4 matrix P(X, Y | data)
 * - logScalingFactorForClique accumulates all scaling factors
 *
 * After calibration, computing LL at ANY clique should give the same result:
 * LL = logScalingFactorForClique + log(sum_x pi[x] * P(X=x | data))
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <functional>
#include <map>
#include <set>
#include <queue>
#include <array>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Data structures matching C++ clique tree
struct TreeNode {
    int id;
    string name;
    bool is_leaf;
    int taxon_index;
    int parent_node;  // -1 for root
    vector<int> child_nodes;
};

struct Clique {
    int id;
    int x_node;  // parent variable (separator)
    int y_node;  // child variable
    int parent_clique;  // -1 for root clique
    vector<int> child_cliques;
};

struct TestTree {
    vector<TreeNode> nodes;
    vector<Clique> cliques;
    int root_clique;
    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<double> pattern_weights;
    vector<double> root_probabilities;
    vector<array<array<double, 4>, 4>> transition_matrices;  // M[x][y] per clique
    int root_node;
};

// GPU kernel: Initialize clique beliefs from initial potentials
// Initial potential for clique (X, Y) is M[x][y] for non-leaf Y
// For leaf Y with observed base b: potential[x][y] = M[x][b] if y==b, else 0
__global__ void initialize_clique_beliefs_kernel(
    int num_patterns,
    int num_cliques,
    int num_taxa,
    const int* d_clique_x_nodes,
    const int* d_clique_y_nodes,
    const char* d_node_is_leaf,
    const int* d_leaf_taxon_indices,
    const uint8_t* d_pattern_bases,
    const double* d_transition_matrices,  // [num_cliques * 16]
    double* d_beliefs,                     // [num_patterns * num_cliques * 16]
    double* d_log_scale_factors            // [num_patterns * num_cliques]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_patterns * num_cliques;
    if (idx >= total) return;

    int pid = idx / num_cliques;
    int cid = idx % num_cliques;

    int y_node = d_clique_y_nodes[cid];
    const double* M = d_transition_matrices + cid * 16;
    double* belief = d_beliefs + pid * num_cliques * 16 + cid * 16;

    if (d_node_is_leaf[y_node]) {
        // Leaf clique: Y is observed
        int taxon = d_leaf_taxon_indices[y_node];
        uint8_t obs_base = d_pattern_bases[pid * num_taxa + taxon];

        if (obs_base < 4) {
            // Observed base
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    if (y == obs_base) {
                        belief[x * 4 + y] = M[x * 4 + y];
                    } else {
                        belief[x * 4 + y] = 0.0;
                    }
                }
            }
        } else {
            // Gap: treat as all bases possible
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    belief[x * 4 + y] = M[x * 4 + y];
                }
            }
        }
    } else {
        // Internal clique: Y is unobserved, start with transition matrix
        for (int i = 0; i < 16; i++) {
            belief[i] = M[i];
        }
    }

    // Initialize log scale factor to 0
    d_log_scale_factors[pid * num_cliques + cid] = 0.0;
}

// GPU kernel: Send message from ONE specific child clique to parent clique (upward pass)
// Message from C_child to C_parent is computed by:
// 1. Start with C_child's current belief (4x4)
// 2. Marginalize over Y to get message over X (4-vector)
__global__ void upward_message_single_clique_kernel(
    int num_patterns,
    int num_cliques,
    int child_clique_id,  // The specific clique sending the message
    const double* d_beliefs,
    double* d_messages,           // [num_patterns * num_cliques * 4]
    double* d_message_log_scales  // [num_patterns * num_cliques]
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    const double* belief = d_beliefs + pid * num_cliques * 16 + child_clique_id * 16;
    double* msg = d_messages + pid * num_cliques * 4 + child_clique_id * 4;

    // Marginalize over Y to get message over X: msg[x] = sum_y belief[x][y]
    double sum = 0.0;
    for (int x = 0; x < 4; x++) {
        double val = 0.0;
        for (int y = 0; y < 4; y++) {
            val += belief[x * 4 + y];
        }
        msg[x] = val;
        sum += val;
    }

    // Normalize message
    double log_scale = 0.0;
    if (sum > 0.0) {
        for (int x = 0; x < 4; x++) {
            msg[x] /= sum;
        }
        log_scale = log(sum);
    }

    d_message_log_scales[pid * num_cliques + child_clique_id] = log_scale;
}

// GPU kernel: Absorb child messages into parent clique belief (upward pass)
// Parent clique absorbs message from child by multiplying belief by message
__global__ void absorb_child_message_kernel(
    int num_patterns,
    int num_cliques,
    int child_clique_id,
    const int* d_clique_x_nodes,
    const int* d_clique_y_nodes,
    const int* d_clique_parents,
    double* d_beliefs,
    double* d_log_scale_factors,
    const double* d_messages,
    const double* d_message_log_scales
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    int parent_id = d_clique_parents[child_clique_id];
    if (parent_id < 0) return;  // Root clique has no parent

    double* parent_belief = d_beliefs + pid * num_cliques * 16 + parent_id * 16;
    const double* child_msg = d_messages + pid * num_cliques * 4 + child_clique_id * 4;

    // The separator between child and parent cliques is the X variable of child clique
    // which is also the Y variable of parent clique (since child clique's X is parent's Y)
    int child_x = d_clique_x_nodes[child_clique_id];
    int parent_y = d_clique_y_nodes[parent_id];

    // Message from child is over child's X, which should match parent's Y
    if (child_x == parent_y) {
        // Multiply parent belief by child message: belief[x][y] *= msg[y]
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                parent_belief[x * 4 + y] *= child_msg[y];
            }
        }
    } else {
        // This shouldn't happen if clique tree is built correctly
        // Handle case where separator is parent's X
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                parent_belief[x * 4 + y] *= child_msg[x];
            }
        }
    }

    // Rescale to prevent underflow
    double max_val = 0.0;
    for (int i = 0; i < 16; i++) {
        max_val = fmax(max_val, parent_belief[i]);
    }
    if (max_val > 0.0) {
        for (int i = 0; i < 16; i++) {
            parent_belief[i] /= max_val;
        }
        d_log_scale_factors[pid * num_cliques + parent_id] += log(max_val);
    }

    // Add child's message log scale to parent's log scale factor
    d_log_scale_factors[pid * num_cliques + parent_id] +=
        d_message_log_scales[pid * num_cliques + child_clique_id];
}

// GPU kernel: Send message from parent clique to child clique (downward pass)
// This incorporates root prior and sibling information
__global__ void downward_message_kernel(
    int num_patterns,
    int num_cliques,
    int parent_clique_id,
    int child_clique_id,
    const int* d_clique_x_nodes,
    const int* d_clique_y_nodes,
    const int* d_clique_parents,
    const int* d_clique_num_children,
    const int* d_clique_child_offsets,
    const int* d_clique_children,
    const double* d_root_probs,
    int root_clique_id,
    double* d_beliefs,
    double* d_log_scale_factors,
    double* d_messages,
    double* d_message_log_scales
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    double* parent_belief = d_beliefs + pid * num_cliques * 16 + parent_clique_id * 16;
    double* child_belief = d_beliefs + pid * num_cliques * 16 + child_clique_id * 16;

    // Compute message from parent to child
    // Start with parent's belief, divide out child's contribution
    double temp_belief[16];
    for (int i = 0; i < 16; i++) {
        temp_belief[i] = parent_belief[i];
    }

    // If this is root clique, incorporate root prior
    double msg[4];
    int parent_x = d_clique_x_nodes[parent_clique_id];
    int child_x = d_clique_x_nodes[child_clique_id];
    int parent_y = d_clique_y_nodes[parent_clique_id];

    // Message is over the separator variable (child's X = parent's Y usually)
    if (child_x == parent_y) {
        // Standard case: marginalize parent over X to get message over Y
        // msg[y] = sum_x belief[x][y] * (root_prior[x] if parent is root)
        if (parent_clique_id == root_clique_id) {
            // Root clique: weight by root prior
            for (int y = 0; y < 4; y++) {
                double val = 0.0;
                for (int x = 0; x < 4; x++) {
                    val += temp_belief[x * 4 + y] * d_root_probs[x];
                }
                msg[y] = val;
            }
        } else {
            // Non-root clique
            for (int y = 0; y < 4; y++) {
                double val = 0.0;
                for (int x = 0; x < 4; x++) {
                    val += temp_belief[x * 4 + y];
                }
                msg[y] = val;
            }
        }
    } else {
        // Alternative: separator is parent's X
        if (parent_clique_id == root_clique_id) {
            for (int x = 0; x < 4; x++) {
                double val = 0.0;
                for (int y = 0; y < 4; y++) {
                    val += temp_belief[x * 4 + y] * d_root_probs[x];
                }
                msg[x] = val;
            }
        } else {
            for (int x = 0; x < 4; x++) {
                double val = 0.0;
                for (int y = 0; y < 4; y++) {
                    val += temp_belief[x * 4 + y];
                }
                msg[x] = val;
            }
        }
    }

    // Normalize
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += msg[i];
    double log_scale = 0.0;
    if (sum > 0.0) {
        for (int i = 0; i < 4; i++) msg[i] /= sum;
        log_scale = log(sum);
    }

    // Absorb message into child's belief
    if (child_x == parent_y) {
        // Multiply child belief by message: belief[x][y] *= msg[x]
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                child_belief[x * 4 + y] *= msg[x];
            }
        }
    } else {
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                child_belief[x * 4 + y] *= msg[x];
            }
        }
    }

    // Rescale child belief
    double max_val = 0.0;
    for (int i = 0; i < 16; i++) {
        max_val = fmax(max_val, child_belief[i]);
    }
    if (max_val > 0.0) {
        for (int i = 0; i < 16; i++) {
            child_belief[i] /= max_val;
        }
        d_log_scale_factors[pid * num_cliques + child_clique_id] += log(max_val);
    }

    // Add message log scale to child's log scale factor
    d_log_scale_factors[pid * num_cliques + child_clique_id] += log_scale;
}

// GPU kernel: Compute LL at each clique
// LL = logScalingFactorForClique + log(sum_x pi[x] * P(X=x | data))
__global__ void compute_clique_ll_kernel(
    int num_patterns,
    int num_cliques,
    const int* d_clique_x_nodes,
    const int* d_clique_y_nodes,
    int root_node_id,
    int root_clique_id,
    const double* d_root_probs,
    const double* d_beliefs,
    const double* d_log_scale_factors,
    const double* d_weights,
    double* d_clique_lls  // [num_cliques * num_patterns]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cliques * num_patterns;
    if (idx >= total) return;

    int cid = idx / num_patterns;
    int pid = idx % num_patterns;

    const double* belief = d_beliefs + pid * num_cliques * 16 + cid * 16;
    double log_scale = d_log_scale_factors[pid * num_cliques + cid];
    double weight = d_weights[pid];

    // Marginalize belief to get P(X | data) or P(Y | data)
    // Then weight by root prior
    double site_ll;

    int x_node = d_clique_x_nodes[cid];
    int y_node = d_clique_y_nodes[cid];

    if (x_node == root_node_id || y_node == root_node_id) {
        // Clique contains root variable
        if (x_node == root_node_id) {
            // X is root, marginalize over Y
            double marginal[4];
            for (int x = 0; x < 4; x++) {
                double val = 0.0;
                for (int y = 0; y < 4; y++) {
                    val += belief[x * 4 + y];
                }
                marginal[x] = val;
            }
            site_ll = 0.0;
            for (int x = 0; x < 4; x++) {
                site_ll += d_root_probs[x] * marginal[x];
            }
        } else {
            // Y is root, marginalize over X
            double marginal[4];
            for (int y = 0; y < 4; y++) {
                double val = 0.0;
                for (int x = 0; x < 4; x++) {
                    val += belief[x * 4 + y];
                }
                marginal[y] = val;
            }
            site_ll = 0.0;
            for (int y = 0; y < 4; y++) {
                site_ll += d_root_probs[y] * marginal[y];
            }
        }
    } else {
        // Clique does NOT contain root variable
        // Use root clique's marginal to get root prior contribution
        // This is the key insight from C++ code: for non-root cliques,
        // we still need the root prior contribution
        const double* root_belief = d_beliefs + pid * num_cliques * 16 + root_clique_id * 16;
        int root_x = d_clique_x_nodes[root_clique_id];

        double marginal_root[4];
        if (root_x == root_node_id) {
            for (int x = 0; x < 4; x++) {
                double val = 0.0;
                for (int y = 0; y < 4; y++) {
                    val += root_belief[x * 4 + y];
                }
                marginal_root[x] = val;
            }
        } else {
            for (int y = 0; y < 4; y++) {
                double val = 0.0;
                for (int x = 0; x < 4; x++) {
                    val += root_belief[x * 4 + y];
                }
                marginal_root[y] = val;
            }
        }

        site_ll = 0.0;
        for (int i = 0; i < 4; i++) {
            site_ll += d_root_probs[i] * marginal_root[i];
        }
    }

    if (site_ll < 1e-300) site_ll = 1e-300;
    d_clique_lls[cid * num_patterns + pid] = weight * (log_scale + log(site_ll));
}

// GPU kernel: Sum pattern LLs for each clique
__global__ void sum_clique_ll_kernel(
    int num_cliques,
    int num_patterns,
    const double* d_clique_pattern_lls,
    double* d_clique_lls
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cliques) return;

    double sum = 0.0;
    for (int p = 0; p < num_patterns; p++) {
        sum += d_clique_pattern_lls[cid * num_patterns + p];
    }
    d_clique_lls[cid] = sum;
}

// File loading functions
void load_edge_file(const string& filename, TestTree& tree) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open edge file: " << filename << endl;
        exit(1);
    }

    vector<array<string, 2>> edge_pairs;
    vector<double> lengths;
    string line;

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;
        edge_pairs.push_back({parent_name, child_name});
        lengths.push_back(branch_length);
    }

    // Build nodes
    set<string> node_names;
    for (auto& e : edge_pairs) {
        node_names.insert(e[0]);
        node_names.insert(e[1]);
    }

    map<string, int> name_to_id;
    int nid = 0;
    for (const string& name : node_names) {
        TreeNode node;
        node.id = nid;
        node.name = name;
        node.is_leaf = true;
        node.taxon_index = -1;
        node.parent_node = -1;
        tree.nodes.push_back(node);
        name_to_id[name] = nid;
        nid++;
    }

    // Build edges and set parent-child relationships
    for (size_t i = 0; i < edge_pairs.size(); i++) {
        int parent_id = name_to_id[edge_pairs[i][0]];
        int child_id = name_to_id[edge_pairs[i][1]];

        tree.nodes[parent_id].is_leaf = false;
        tree.nodes[parent_id].child_nodes.push_back(child_id);
        tree.nodes[child_id].parent_node = parent_id;

        // Create clique for this edge (X=parent, Y=child)
        Clique clique;
        clique.id = tree.cliques.size();
        clique.x_node = parent_id;
        clique.y_node = child_id;
        clique.parent_clique = -1;  // Will set later
        tree.cliques.push_back(clique);
    }

    // Store branch lengths and compute transition matrices later
    tree.transition_matrices.resize(tree.cliques.size());

    // Find root node (node with no parent)
    for (auto& node : tree.nodes) {
        if (node.parent_node == -1 && !node.child_nodes.empty()) {
            tree.root_node = node.id;
            break;
        }
    }

    // Build clique tree structure
    // Cliques are adjacent if they share a variable
    // Parent-child relationship: clique_i is parent of clique_j if clique_i.y == clique_j.x
    for (size_t i = 0; i < tree.cliques.size(); i++) {
        for (size_t j = 0; j < tree.cliques.size(); j++) {
            if (i != j) {
                // Check if clique_i.y == clique_j.x (clique_i is parent of clique_j)
                if (tree.cliques[i].y_node == tree.cliques[j].x_node) {
                    tree.cliques[j].parent_clique = i;
                    tree.cliques[i].child_cliques.push_back(j);
                }
            }
        }
    }

    // Find all root cliques (cliques whose X is the root node and have no parent yet)
    vector<int> root_clique_ids;
    for (auto& clique : tree.cliques) {
        if (clique.x_node == tree.root_node && clique.parent_clique == -1) {
            root_clique_ids.push_back(clique.id);
        }
    }

    // Designate the first root clique as THE root clique
    // Make other root cliques its children in the clique tree
    if (!root_clique_ids.empty()) {
        tree.root_clique = root_clique_ids[0];
        for (size_t i = 1; i < root_clique_ids.size(); i++) {
            int other_root = root_clique_ids[i];
            tree.cliques[other_root].parent_clique = tree.root_clique;
            tree.cliques[tree.root_clique].child_cliques.push_back(other_root);
        }
    } else {
        // No root cliques found, pick first clique
        tree.root_clique = 0;
    }

    cout << "Root cliques: ";
    for (int rc : root_clique_ids) {
        cout << rc << " (X=" << tree.cliques[rc].x_node << ", Y=" << tree.cliques[rc].y_node << ") ";
    }
    cout << endl;

    // Store branch lengths for transition matrix computation
    for (size_t i = 0; i < edge_pairs.size(); i++) {
        // Clique i corresponds to edge i
        tree.transition_matrices[i] = array<array<double, 4>, 4>();
    }

    // Store lengths temporarily
    vector<double> saved_lengths = lengths;

    // Now set transition matrices using F81 model
    // First load other data, then compute matrices
}

void load_pattern_file(const string& filename, TestTree& tree) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        double weight;
        iss >> weight;
        tree.pattern_weights.push_back(weight);
        int base;
        while (iss >> base) {
            tree.pattern_bases.push_back((uint8_t)base);
        }
    }
    tree.num_patterns = tree.pattern_weights.size();
    if (tree.num_patterns > 0) {
        tree.num_taxa = tree.pattern_bases.size() / tree.num_patterns;
    }
}

void load_taxon_order(const string& filename, TestTree& tree) {
    ifstream file(filename);
    vector<string> taxon_names;
    string line;
    getline(file, line);  // Skip header
    while (getline(file, line)) {
        if (!line.empty()) {
            size_t comma = line.find(',');
            if (comma != string::npos) {
                taxon_names.push_back(line.substr(0, comma));
            }
        }
    }
    tree.num_taxa = taxon_names.size();

    for (auto& node : tree.nodes) {
        if (node.is_leaf) {
            for (size_t i = 0; i < taxon_names.size(); i++) {
                if (taxon_names[i] == node.name) {
                    node.taxon_index = i;
                    break;
                }
            }
        }
    }
}

void load_base_composition(const string& filename, TestTree& tree) {
    ifstream file(filename);
    tree.root_probabilities.resize(4);
    string line;
    for (int i = 0; i < 4 && getline(file, line); i++) {
        istringstream iss(line);
        int idx;
        double prob;
        iss >> idx >> prob;
        tree.root_probabilities[idx] = prob;
    }
}

void load_branch_lengths_and_init_matrices(const string& filename, TestTree& tree) {
    ifstream file(filename);
    vector<double> lengths;
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        string p, c;
        double len;
        iss >> p >> c >> len;
        lengths.push_back(len);
    }

    // Compute F81 mu
    double S2 = 0.0;
    for (int k = 0; k < 4; k++) {
        S2 += tree.root_probabilities[k] * tree.root_probabilities[k];
    }
    double mu = 1.0 / max(1e-14, 1.0 - S2);

    // Compute transition matrices for each clique/edge
    for (size_t e = 0; e < tree.cliques.size(); e++) {
        double t = lengths[e];
        double exp_term = exp(-mu * t);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    tree.transition_matrices[e][i][j] = exp_term + tree.root_probabilities[j] * (1.0 - exp_term);
                } else {
                    tree.transition_matrices[e][i][j] = tree.root_probabilities[j] * (1.0 - exp_term);
                }
            }
        }
    }
}

vector<int> compute_post_order_cliques(const TestTree& tree) {
    vector<int> order;
    vector<bool> visited(tree.cliques.size(), false);

    function<void(int)> visit = [&](int cid) {
        for (int child_cid : tree.cliques[cid].child_cliques) {
            if (!visited[child_cid]) {
                visit(child_cid);
            }
        }
        if (!visited[cid]) {
            order.push_back(cid);
            visited[cid] = true;
        }
    };

    visit(tree.root_clique);
    return order;
}

// Compute distances between cliques
vector<vector<int>> compute_clique_distances(const TestTree& tree) {
    int num_cliques = tree.cliques.size();
    vector<vector<int>> dist(num_cliques, vector<int>(num_cliques, 0));

    // Build adjacency
    vector<vector<int>> adj(num_cliques);
    for (auto& c : tree.cliques) {
        for (int child : c.child_cliques) {
            adj[c.id].push_back(child);
            adj[child].push_back(c.id);
        }
    }

    // BFS for each clique
    for (int start = 0; start < num_cliques; start++) {
        queue<int> q;
        vector<bool> vis(num_cliques, false);
        q.push(start);
        vis[start] = true;
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            for (int next : adj[curr]) {
                if (!vis[next]) {
                    vis[next] = true;
                    dist[start][next] = dist[start][curr] + 1;
                    q.push(next);
                }
            }
        }
    }

    return dist;
}

void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " -e <edge_file> -p <pattern_file> -x <taxon_order> "
         << "-b <base_comp>" << endl;
}

int main(int argc, char** argv) {
    string edge_file, pattern_file, taxon_order_file, basecomp_file;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-e" && i + 1 < argc) edge_file = argv[++i];
        else if (arg == "-p" && i + 1 < argc) pattern_file = argv[++i];
        else if (arg == "-x" && i + 1 < argc) taxon_order_file = argv[++i];
        else if (arg == "-b" && i + 1 < argc) basecomp_file = argv[++i];
    }

    if (edge_file.empty() || pattern_file.empty() ||
        taxon_order_file.empty() || basecomp_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    cout << "=== Clique Tree Log-Likelihood Consistency Test ===" << endl;
    cout << "(Matching C++ junction tree structure)" << endl << endl;

    TestTree tree;
    cout << "Loading data..." << endl;
    load_edge_file(edge_file, tree);
    load_pattern_file(pattern_file, tree);
    load_taxon_order(taxon_order_file, tree);
    load_base_composition(basecomp_file, tree);
    load_branch_lengths_and_init_matrices(edge_file, tree);

    int num_cliques = tree.cliques.size();
    cout << "Loaded: " << tree.num_taxa << " taxa, " << num_cliques
         << " cliques, " << tree.num_patterns << " patterns" << endl;
    cout << "Root node: " << tree.root_node << ", Root clique: " << tree.root_clique << endl;

    // Compute clique distances
    vector<vector<int>> clique_distances = compute_clique_distances(tree);

    // Post-order traversal for upward pass
    vector<int> post_order = compute_post_order_cliques(tree);
    cout << "Post-order cliques: ";
    for (int c : post_order) cout << c << " ";
    cout << endl;

    // Prepare GPU data
    vector<int> clique_x(num_cliques), clique_y(num_cliques);
    vector<int> clique_parents(num_cliques);
    vector<int> clique_num_children(num_cliques), clique_child_offsets(num_cliques);
    vector<int> clique_children_flat;
    vector<bool> node_is_leaf_vec(tree.nodes.size());
    vector<int> leaf_taxon_vec(tree.nodes.size());
    vector<double> trans_flat(num_cliques * 16);

    for (auto& c : tree.cliques) {
        clique_x[c.id] = c.x_node;
        clique_y[c.id] = c.y_node;
        clique_parents[c.id] = c.parent_clique;
        clique_num_children[c.id] = c.child_cliques.size();
        clique_child_offsets[c.id] = clique_children_flat.size();
        for (int child : c.child_cliques) {
            clique_children_flat.push_back(child);
        }
    }

    vector<char> node_is_leaf_char(tree.nodes.size());  // Use char instead of bool for data()
    for (auto& n : tree.nodes) {
        node_is_leaf_char[n.id] = n.is_leaf ? 1 : 0;
        node_is_leaf_vec[n.id] = n.is_leaf;
        leaf_taxon_vec[n.id] = n.taxon_index;
    }

    for (int c = 0; c < num_cliques; c++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                trans_flat[c * 16 + i * 4 + j] = tree.transition_matrices[c][i][j];
            }
        }
    }

    // Allocate GPU memory
    cout << "Allocating GPU memory..." << endl;
    int* d_clique_x, *d_clique_y, *d_clique_parents;
    int* d_clique_num_children, *d_clique_child_offsets, *d_clique_children;
    char* d_node_is_leaf;
    int* d_leaf_taxon;
    uint8_t* d_pattern_bases;
    double* d_trans, *d_weights, *d_root_probs;
    double* d_beliefs, *d_log_scales;
    double* d_messages, *d_msg_log_scales;
    double* d_clique_pattern_lls, *d_clique_lls;

    CUDA_CHECK(cudaMalloc(&d_clique_x, num_cliques * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clique_y, num_cliques * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clique_parents, num_cliques * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clique_num_children, num_cliques * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clique_child_offsets, num_cliques * sizeof(int)));
    if (!clique_children_flat.empty()) {
        CUDA_CHECK(cudaMalloc(&d_clique_children, clique_children_flat.size() * sizeof(int)));
    } else {
        d_clique_children = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&d_node_is_leaf, tree.nodes.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_leaf_taxon, tree.nodes.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_bases, tree.num_patterns * tree.num_taxa * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_trans, num_cliques * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights, tree.num_patterns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_root_probs, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beliefs, tree.num_patterns * num_cliques * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_log_scales, tree.num_patterns * num_cliques * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_messages, tree.num_patterns * num_cliques * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_msg_log_scales, tree.num_patterns * num_cliques * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clique_pattern_lls, num_cliques * tree.num_patterns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clique_lls, num_cliques * sizeof(double)));

    // Copy data to GPU
    cout << "Copying data to GPU..." << endl;
    CUDA_CHECK(cudaMemcpy(d_clique_x, clique_x.data(), num_cliques * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clique_y, clique_y.data(), num_cliques * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clique_parents, clique_parents.data(), num_cliques * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clique_num_children, clique_num_children.data(), num_cliques * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clique_child_offsets, clique_child_offsets.data(), num_cliques * sizeof(int), cudaMemcpyHostToDevice));
    if (!clique_children_flat.empty()) {
        CUDA_CHECK(cudaMemcpy(d_clique_children, clique_children_flat.data(), clique_children_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_node_is_leaf, node_is_leaf_char.data(), tree.nodes.size() * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_taxon, leaf_taxon_vec.data(), tree.nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pattern_bases, tree.pattern_bases.data(), tree.num_patterns * tree.num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_trans, trans_flat.data(), num_cliques * 16 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, tree.pattern_weights.data(), tree.num_patterns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_root_probs, tree.root_probabilities.data(), 4 * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize beliefs
    cout << "Initializing clique beliefs..." << endl;
    int total_work = tree.num_patterns * num_cliques;
    int bs = 256;
    int nb = (total_work + bs - 1) / bs;
    initialize_clique_beliefs_kernel<<<nb, bs>>>(
        tree.num_patterns, num_cliques, tree.num_taxa,
        d_clique_x, d_clique_y, d_node_is_leaf, d_leaf_taxon,
        d_pattern_bases, d_trans, d_beliefs, d_log_scales
    );
    cudaDeviceSynchronize();

    // Upward pass (post-order)
    cout << "Running upward pass..." << endl;
    for (int cid : post_order) {
        int nb_p = (tree.num_patterns + bs - 1) / bs;
        // Only send message if this clique has a parent
        if (tree.cliques[cid].parent_clique >= 0) {
            // Compute message from this specific clique
            upward_message_single_clique_kernel<<<nb_p, bs>>>(
                tree.num_patterns, num_cliques, cid,
                d_beliefs, d_messages, d_msg_log_scales
            );
            cudaDeviceSynchronize();

            // Parent clique absorbs the message
            absorb_child_message_kernel<<<nb_p, bs>>>(
                tree.num_patterns, num_cliques, cid,
                d_clique_x, d_clique_y, d_clique_parents,
                d_beliefs, d_log_scales, d_messages, d_msg_log_scales
            );
            cudaDeviceSynchronize();
        }
    }

    // Downward pass (pre-order, skip root)
    cout << "Running downward pass..." << endl;
    vector<int> pre_order;
    pre_order.push_back(tree.root_clique);
    for (int i = post_order.size() - 1; i >= 0; i--) {
        if (post_order[i] != tree.root_clique) {
            pre_order.push_back(post_order[i]);
        }
    }

    for (size_t i = 1; i < pre_order.size(); i++) {  // Skip root
        int child_cid = pre_order[i];
        int parent_cid = tree.cliques[child_cid].parent_clique;
        if (parent_cid >= 0) {
            int nb_p = (tree.num_patterns + bs - 1) / bs;
            downward_message_kernel<<<nb_p, bs>>>(
                tree.num_patterns, num_cliques,
                parent_cid, child_cid,
                d_clique_x, d_clique_y, d_clique_parents,
                d_clique_num_children, d_clique_child_offsets, d_clique_children,
                d_root_probs, tree.root_clique,
                d_beliefs, d_log_scales, d_messages, d_msg_log_scales
            );
            cudaDeviceSynchronize();
        }
    }

    // Compute LL at each clique
    cout << "Computing log-likelihood at each clique..." << endl;
    compute_clique_ll_kernel<<<nb, bs>>>(
        tree.num_patterns, num_cliques,
        d_clique_x, d_clique_y, tree.root_node, tree.root_clique,
        d_root_probs, d_beliefs, d_log_scales, d_weights,
        d_clique_pattern_lls
    );
    cudaDeviceSynchronize();

    // Sum pattern LLs
    int nb3 = (num_cliques + bs - 1) / bs;
    sum_clique_ll_kernel<<<nb3, bs>>>(num_cliques, tree.num_patterns, d_clique_pattern_lls, d_clique_lls);
    cudaDeviceSynchronize();

    // Copy results back
    vector<double> clique_lls(num_cliques);
    CUDA_CHECK(cudaMemcpy(clique_lls.data(), d_clique_lls, num_cliques * sizeof(double), cudaMemcpyDeviceToHost));

    // Analyze results
    cout << endl << "=== Results ===" << endl;

    double ref_ll = clique_lls[tree.root_clique];
    cout << "Reference LL (root clique " << tree.root_clique << "): "
         << fixed << setprecision(15) << ref_ll << endl;

    double min_ll = clique_lls[0], max_ll = clique_lls[0], sum_ll = 0.0;
    for (int c = 0; c < num_cliques; c++) {
        min_ll = min(min_ll, clique_lls[c]);
        max_ll = max(max_ll, clique_lls[c]);
        sum_ll += clique_lls[c];
    }
    double mean_ll = sum_ll / num_cliques;

    double variance = 0.0;
    for (int c = 0; c < num_cliques; c++) {
        variance += (clique_lls[c] - mean_ll) * (clique_lls[c] - mean_ll);
    }
    double std_ll = sqrt(variance / num_cliques);

    cout << "\nLL Statistics across all " << num_cliques << " cliques:" << endl;
    cout << "  Mean:     " << fixed << setprecision(15) << mean_ll << endl;
    cout << "  Std Dev:  " << scientific << setprecision(6) << std_ll << endl;
    cout << "  Range:    " << fixed << setprecision(15) << min_ll << " to " << max_ll << endl;
    cout << "  Max diff: " << scientific << setprecision(6) << (max_ll - min_ll) << endl;

    // Output clique LL values
    cout << "\n=== Clique LL Values ===" << endl;
    cout << "CliqueID  X_node  Y_node          LL_Value" << endl;
    for (int c = 0; c < num_cliques; c++) {
        cout << setw(8) << c << "  " << setw(6) << tree.cliques[c].x_node
             << "  " << setw(6) << tree.cliques[c].y_node << "  "
             << fixed << setprecision(15) << clique_lls[c] << endl;
    }

    // Output distance vs LL difference data
    cout << "\n=== Distance vs LL Difference ===" << endl;
    map<int, vector<double>> diff_by_distance;
    for (int c1 = 0; c1 < num_cliques; c1++) {
        for (int c2 = c1 + 1; c2 < num_cliques; c2++) {
            int dist = clique_distances[c1][c2];
            double diff = fabs(clique_lls[c1] - clique_lls[c2]);
            diff_by_distance[dist].push_back(diff);
        }
    }

    cout << "\n=== LL Difference Statistics by Distance ===" << endl;
    cout << "Distance    Count    Mean_Diff        Max_Diff         Std_Diff" << endl;
    for (auto& kv : diff_by_distance) {
        int dist = kv.first;
        vector<double>& diffs = kv.second;
        double sum = 0.0, max_val = 0.0;
        for (double d : diffs) {
            sum += d;
            max_val = max(max_val, d);
        }
        double mean = sum / diffs.size();
        double var = 0.0;
        for (double d : diffs) {
            var += (d - mean) * (d - mean);
        }
        double std = sqrt(var / diffs.size());
        cout << setw(8) << dist << "  " << setw(7) << diffs.size() << "  "
             << scientific << setprecision(6) << mean << "  "
             << max_val << "  " << std << endl;
    }

    // Cleanup
    cudaFree(d_clique_x);
    cudaFree(d_clique_y);
    cudaFree(d_clique_parents);
    cudaFree(d_clique_num_children);
    cudaFree(d_clique_child_offsets);
    if (d_clique_children) cudaFree(d_clique_children);
    cudaFree(d_node_is_leaf);
    cudaFree(d_leaf_taxon);
    cudaFree(d_pattern_bases);
    cudaFree(d_trans);
    cudaFree(d_weights);
    cudaFree(d_root_probs);
    cudaFree(d_beliefs);
    cudaFree(d_log_scales);
    cudaFree(d_messages);
    cudaFree(d_msg_log_scales);
    cudaFree(d_clique_pattern_lls);
    cudaFree(d_clique_lls);

    return 0;
}
