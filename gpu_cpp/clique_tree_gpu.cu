/**
 * GPU-accelerated clique tree belief propagation
 *
 * Parallelizes over site patterns - each thread processes one pattern.
 * Tree structure and model parameters are stored in constant memory.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <array>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>
#include <algorithm>
#include <functional>
#include <tuple>
#include <cuda_runtime.h>

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
struct GPUTreeData {
    int num_cliques;
    int num_nodes;
    int num_leaves;
    int root_node;

    // Clique data
    int clique_x_node[MAX_CLIQUES];
    int clique_y_node[MAX_CLIQUES];
    int clique_parent[MAX_CLIQUES];      // -1 if no parent
    int clique_children[MAX_CLIQUES][MAX_CHILDREN];
    int clique_num_children[MAX_CLIQUES];

    // Traversal orders (edges: parent->child format)
    int postorder_edges[MAX_CLIQUES][2];  // [i][0]=parent, [i][1]=child
    int num_postorder_edges;
    int preorder_edges[MAX_CLIQUES][2];
    int num_preorder_edges;

    // Model parameters
    double branch_lengths[MAX_NODES];
    double root_prior[4];
    double f81_mu;
};

__constant__ GPUTreeData d_tree;

// Compute F81 transition matrix on GPU
__device__ void compute_transition_matrix_gpu(double branch_length, double* P) {
    double t = branch_length * d_tree.f81_mu;
    double exp_term = exp(-t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i * 4 + j] = exp_term + d_tree.root_prior[j] * (1.0 - exp_term);
            } else {
                P[i * 4 + j] = d_tree.root_prior[j] * (1.0 - exp_term);
            }
        }
    }
}

/**
 * GPU kernel: Compute log-likelihood for multiple patterns
 *
 * Each thread processes one pattern:
 * 1. Initialize clique potentials
 * 2. Perform message passing (calibration)
 * 3. Compute log-likelihood at root clique
 */
__global__ void compute_pattern_lls_kernel(
    const int* patterns,      // [num_patterns * num_leaves]
    const int* weights,       // [num_patterns]
    double* pattern_lls,      // [num_patterns] output
    int num_patterns
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    // Local storage for this pattern's belief propagation
    double init_pot[MAX_CLIQUES][16];  // 4x4 matrix flattened
    double msg_from_parent[MAX_CLIQUES][4];
    double msg_from_child[MAX_CLIQUES][MAX_CHILDREN][4];
    double scale_from_parent[MAX_CLIQUES];
    double scale_from_child[MAX_CLIQUES][MAX_CHILDREN];

    int num_cliques = d_tree.num_cliques;
    int num_leaves = d_tree.num_leaves;

    // Get pattern data
    const int* pattern = &patterns[pid * num_leaves];

    // 1. Initialize clique potentials
    for (int c = 0; c < num_cliques; c++) {
        int y_node = d_tree.clique_y_node[c];
        double P[16];
        compute_transition_matrix_gpu(d_tree.branch_lengths[y_node], P);

        bool y_is_leaf = (y_node < num_leaves);
        int leaf_state = y_is_leaf ? pattern[y_node] : -1;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (y_is_leaf && leaf_state < 4) {
                    // Observed state
                    init_pot[c][i * 4 + j] = (j == leaf_state) ? P[i * 4 + j] : 0.0;
                } else {
                    // Internal node or unknown state
                    init_pot[c][i * 4 + j] = P[i * 4 + j];
                }
            }
        }

        // Initialize messages and scales
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

    // 2. Upward pass: leaves to root (send messages from child to parent)
    for (int e = 0; e < d_tree.num_postorder_edges; e++) {
        int parent_clique = d_tree.postorder_edges[e][0];
        int child_clique = d_tree.postorder_edges[e][1];

        // SendMessage: child_clique -> parent_clique
        // Product of initial potential and messages from neighbors (excluding recipient)
        double factor[16];
        for (int i = 0; i < 16; i++) {
            factor[i] = init_pot[child_clique][i];
        }

        double logScalingFactor = 0.0;

        // Multiply by messages from child's children
        for (int ch = 0; ch < d_tree.clique_num_children[child_clique]; ch++) {
            int grandchild_clique = d_tree.clique_children[child_clique][ch];

            // Determine shared variable: grandchild shares Y of child_clique
            int child_y = d_tree.clique_y_node[child_clique];
            int gc_x = d_tree.clique_x_node[grandchild_clique];
            int gc_y = d_tree.clique_y_node[grandchild_clique];

            bool gc_shares_child_y = (child_y == gc_x || child_y == gc_y);

            if (gc_shares_child_y) {
                // Message applies to Y dimension (rows in matrix)
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    for (int dna_y = 0; dna_y < 4; dna_y++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_child[child_clique][ch][dna_y];
                    }
                }
            } else {
                // Message applies to X dimension (columns)
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    for (int dna_x = 0; dna_x < 4; dna_x++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_child[child_clique][ch][dna_x];
                    }
                }
            }

            // Rescale
            double largestElement = 0.0;
            for (int i = 0; i < 16; i++) {
                if (largestElement < factor[i]) largestElement = factor[i];
            }
            for (int i = 0; i < 16; i++) {
                factor[i] /= largestElement;
            }
            logScalingFactor += log(largestElement);
            logScalingFactor += scale_from_child[child_clique][ch];
        }

        // Marginalize: sum over variable not shared with parent
        // Child shares X with parent (parent's Y = child's X)
        double message[4] = {0.0, 0.0, 0.0, 0.0};

        int child_x = d_tree.clique_x_node[child_clique];
        int child_y = d_tree.clique_y_node[child_clique];
        int parent_x = d_tree.clique_x_node[parent_clique];
        int parent_y = d_tree.clique_y_node[parent_clique];

        bool child_y_shared = (child_y == parent_x || child_y == parent_y);

        if (child_y_shared) {
            // Shared is Y, sum over X
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    message[dna_y] += factor[dna_x * 4 + dna_y];
                }
            }
        } else {
            // Shared is X, sum over Y
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    message[dna_x] += factor[dna_x * 4 + dna_y];
                }
            }
        }

        // Rescale message
        double largestElement = 0.0;
        for (int d = 0; d < 4; d++) {
            if (largestElement < message[d]) largestElement = message[d];
        }
        for (int d = 0; d < 4; d++) {
            message[d] /= largestElement;
        }
        logScalingFactor += log(largestElement);

        // Store message in parent
        // Find which child index this is in parent's children list
        int child_idx = -1;
        for (int ch = 0; ch < d_tree.clique_num_children[parent_clique]; ch++) {
            if (d_tree.clique_children[parent_clique][ch] == child_clique) {
                child_idx = ch;
                break;
            }
        }

        if (child_idx >= 0) {
            for (int d = 0; d < 4; d++) {
                msg_from_child[parent_clique][child_idx][d] = message[d];
            }
            scale_from_child[parent_clique][child_idx] = logScalingFactor;
        }
    }

    // 3. Downward pass: root to leaves (send messages from parent to child)
    for (int e = 0; e < d_tree.num_preorder_edges; e++) {
        int parent_clique = d_tree.preorder_edges[e][0];
        int child_clique = d_tree.preorder_edges[e][1];

        // SendMessage: parent_clique -> child_clique
        double factor[16];
        for (int i = 0; i < 16; i++) {
            factor[i] = init_pot[parent_clique][i];
        }

        double logScalingFactor = 0.0;

        // Multiply by message from parent's parent (if exists)
        if (d_tree.clique_parent[parent_clique] >= 0) {
            int grandparent = d_tree.clique_parent[parent_clique];

            int parent_x = d_tree.clique_x_node[parent_clique];
            int parent_y = d_tree.clique_y_node[parent_clique];
            int gp_x = d_tree.clique_x_node[grandparent];
            int gp_y = d_tree.clique_y_node[grandparent];

            bool gp_shares_parent_y = (parent_y == gp_x || parent_y == gp_y);

            if (gp_shares_parent_y) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    for (int dna_y = 0; dna_y < 4; dna_y++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_parent[parent_clique][dna_y];
                    }
                }
            } else {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    for (int dna_x = 0; dna_x < 4; dna_x++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_parent[parent_clique][dna_x];
                    }
                }
            }

            // Rescale
            double largestElement = 0.0;
            for (int i = 0; i < 16; i++) {
                if (largestElement < factor[i]) largestElement = factor[i];
            }
            for (int i = 0; i < 16; i++) {
                factor[i] /= largestElement;
            }
            logScalingFactor += log(largestElement);
            logScalingFactor += scale_from_parent[parent_clique];
        }

        // Multiply by messages from parent's other children (not child_clique)
        for (int ch = 0; ch < d_tree.clique_num_children[parent_clique]; ch++) {
            int sibling = d_tree.clique_children[parent_clique][ch];
            if (sibling == child_clique) continue;

            int parent_y = d_tree.clique_y_node[parent_clique];
            int sib_x = d_tree.clique_x_node[sibling];
            int sib_y = d_tree.clique_y_node[sibling];

            bool sib_shares_parent_y = (parent_y == sib_x || parent_y == sib_y);

            if (sib_shares_parent_y) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    for (int dna_y = 0; dna_y < 4; dna_y++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_child[parent_clique][ch][dna_y];
                    }
                }
            } else {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    for (int dna_x = 0; dna_x < 4; dna_x++) {
                        factor[dna_x * 4 + dna_y] *= msg_from_child[parent_clique][ch][dna_x];
                    }
                }
            }

            // Rescale
            double largestElement = 0.0;
            for (int i = 0; i < 16; i++) {
                if (largestElement < factor[i]) largestElement = factor[i];
            }
            for (int i = 0; i < 16; i++) {
                factor[i] /= largestElement;
            }
            logScalingFactor += log(largestElement);
            logScalingFactor += scale_from_child[parent_clique][ch];
        }

        // Marginalize: sum over variable not shared with child
        double message[4] = {0.0, 0.0, 0.0, 0.0};

        int parent_x = d_tree.clique_x_node[parent_clique];
        int parent_y = d_tree.clique_y_node[parent_clique];
        int child_x = d_tree.clique_x_node[child_clique];
        int child_y = d_tree.clique_y_node[child_clique];

        bool parent_y_shared = (parent_y == child_x || parent_y == child_y);

        if (parent_y_shared) {
            // Shared is Y, sum over X
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    message[dna_y] += factor[dna_x * 4 + dna_y];
                }
            }
        } else {
            // Shared is X, sum over Y
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    message[dna_x] += factor[dna_x * 4 + dna_y];
                }
            }
        }

        // Rescale message
        double largestElement = 0.0;
        for (int d = 0; d < 4; d++) {
            if (largestElement < message[d]) largestElement = message[d];
        }
        for (int d = 0; d < 4; d++) {
            message[d] /= largestElement;
        }
        logScalingFactor += log(largestElement);

        // Store message in child
        for (int d = 0; d < 4; d++) {
            msg_from_parent[child_clique][d] = message[d];
        }
        scale_from_parent[child_clique] = logScalingFactor;
    }

    // 4. Compute log-likelihood at root clique (clique 0 - first root clique)
    // Find a clique containing the root node
    int target_clique = 0;
    for (int c = 0; c < num_cliques; c++) {
        if (d_tree.clique_x_node[c] == d_tree.root_node ||
            d_tree.clique_y_node[c] == d_tree.root_node) {
            target_clique = c;
            break;
        }
    }

    // ComputeBelief for target clique
    double belief[16];
    for (int i = 0; i < 16; i++) {
        belief[i] = init_pot[target_clique][i];
    }

    double logScalingFactor = 0.0;

    // Multiply by message from parent (if exists)
    if (d_tree.clique_parent[target_clique] >= 0) {
        int parent_x = d_tree.clique_x_node[d_tree.clique_parent[target_clique]];
        int parent_y = d_tree.clique_y_node[d_tree.clique_parent[target_clique]];
        int target_y = d_tree.clique_y_node[target_clique];

        bool parent_shares_target_y = (target_y == parent_x || target_y == parent_y);

        if (parent_shares_target_y) {
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    belief[dna_x * 4 + dna_y] *= msg_from_parent[target_clique][dna_y];
                }
            }
        } else {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    belief[dna_x * 4 + dna_y] *= msg_from_parent[target_clique][dna_x];
                }
            }
        }

        // Scale
        double scalingFactor = 0.0;
        for (int i = 0; i < 16; i++) {
            scalingFactor += belief[i];
        }
        for (int i = 0; i < 16; i++) {
            belief[i] /= scalingFactor;
        }
        logScalingFactor += log(scalingFactor);
        logScalingFactor += scale_from_parent[target_clique];
    }

    // Multiply by messages from all children
    for (int ch = 0; ch < d_tree.clique_num_children[target_clique]; ch++) {
        int child = d_tree.clique_children[target_clique][ch];
        int child_x = d_tree.clique_x_node[child];
        int child_y = d_tree.clique_y_node[child];
        int target_y = d_tree.clique_y_node[target_clique];

        bool child_shares_target_y = (target_y == child_x || target_y == child_y);

        if (child_shares_target_y) {
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    belief[dna_x * 4 + dna_y] *= msg_from_child[target_clique][ch][dna_y];
                }
            }
        } else {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    belief[dna_x * 4 + dna_y] *= msg_from_child[target_clique][ch][dna_x];
                }
            }
        }

        // Scale
        double scalingFactor = 0.0;
        for (int i = 0; i < 16; i++) {
            scalingFactor += belief[i];
        }
        for (int i = 0; i < 16; i++) {
            belief[i] /= scalingFactor;
        }
        logScalingFactor += log(scalingFactor);
        logScalingFactor += scale_from_child[target_clique][ch];
    }

    // Final normalization
    double finalScale = 0.0;
    for (int i = 0; i < 16; i++) {
        finalScale += belief[i];
    }
    for (int i = 0; i < 16; i++) {
        belief[i] /= finalScale;
    }
    logScalingFactor += log(finalScale);

    // Compute P(data) = sum_r P(r) * P(r | data)
    // Get marginal over root variable
    double marginalRoot[4] = {0.0, 0.0, 0.0, 0.0};

    if (d_tree.clique_x_node[target_clique] == d_tree.root_node) {
        // X is root, marginalize over Y
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                marginalRoot[dna_x] += belief[dna_x * 4 + dna_y];
            }
        }
    } else {
        // Y is root, marginalize over X
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                marginalRoot[dna_y] += belief[dna_x * 4 + dna_y];
            }
        }
    }

    double siteLikelihood = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        siteLikelihood += d_tree.root_prior[dna] * marginalRoot[dna];
    }

    double ll = logScalingFactor + log(siteLikelihood);

    // Store result (multiply by weight later on CPU)
    pattern_lls[pid] = ll;
}

// ========== CPU-side code (mostly copied from clique_tree_cpu.cpp) ==========

using Matrix4x4 = array<array<double, 4>, 4>;

struct Clique {
    int id;
    int x_node;
    int y_node;
    Clique* parent;
    vector<Clique*> children;
    Matrix4x4 initialPotential;

    Clique(int id_, int x_, int y_) : id(id_), x_node(x_), y_node(y_), parent(nullptr) {}
};

struct CliqueTree {
    vector<Clique*> cliques;
    Clique* root_clique;
    int root_node;
    vector<pair<Clique*, Clique*>> edgesForPostOrderTreeTraversal;
    vector<pair<Clique*, Clique*>> edgesForPreOrderTreeTraversal;

    ~CliqueTree() {
        for (auto* c : cliques) delete c;
    }
};

// Global data
int num_nodes;
int num_leaves;
vector<int> parent_map;
vector<double> branch_lengths;
array<double, 4> root_prior;
double f81_mu = 1.0;

void compute_f81_mu() {
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += root_prior[i] * root_prior[i];
    }
    f81_mu = 1.0 / (1.0 - sum_pi_sq);
    cout << "F81 mu is " << f81_mu << endl;
}

void load_tree_from_edges(const string& edge_file, const string& taxon_order_file = "") {
    ifstream fin(edge_file);
    if (!fin) {
        cerr << "Error opening tree edge file: " << edge_file << endl;
        exit(1);
    }

    vector<tuple<string, string, double>> edges;
    set<string> all_nodes;
    set<string> has_parent;

    string parent_name, child_name;
    double branch_len;

    while (fin >> parent_name >> child_name >> branch_len) {
        edges.push_back({parent_name, child_name, branch_len});
        all_nodes.insert(parent_name);
        all_nodes.insert(child_name);
        has_parent.insert(child_name);
    }
    fin.close();

    vector<string> leaf_names;
    vector<string> internal_names;

    for (const auto& name : all_nodes) {
        if (name.substr(0, 2) == "h_") {
            internal_names.push_back(name);
        } else {
            leaf_names.push_back(name);
        }
    }

    num_leaves = leaf_names.size();
    num_nodes = all_nodes.size();

    cout << "Tree has " << num_leaves << " leaves and " << num_nodes << " total nodes" << endl;

    map<string, int> name_to_id;
    if (!taxon_order_file.empty()) {
        ifstream tof(taxon_order_file);
        if (tof) {
            string line;
            getline(tof, line);
            while (getline(tof, line)) {
                size_t comma = line.find(',');
                if (comma != string::npos) {
                    string taxon = line.substr(0, comma);
                    int pos = stoi(line.substr(comma + 1));
                    if (name_to_id.find(taxon) == name_to_id.end()) {
                        name_to_id[taxon] = pos;
                    }
                }
            }
            tof.close();
            cout << "Loaded taxon order from " << taxon_order_file << endl;

            leaf_names.clear();
            leaf_names.resize(num_leaves);
            for (const auto& p : name_to_id) {
                if (p.second < num_leaves) {
                    leaf_names[p.second] = p.first;
                }
            }
        } else {
            sort(leaf_names.begin(), leaf_names.end());
            for (int i = 0; i < (int)leaf_names.size(); i++) {
                name_to_id[leaf_names[i]] = i;
            }
        }
    } else {
        sort(leaf_names.begin(), leaf_names.end());
        for (int i = 0; i < (int)leaf_names.size(); i++) {
            name_to_id[leaf_names[i]] = i;
        }
    }

    sort(internal_names.begin(), internal_names.end());
    int next_id = num_leaves;
    for (const auto& name : internal_names) {
        name_to_id[name] = next_id++;
    }

    string root_name;
    for (const auto& name : all_nodes) {
        if (has_parent.find(name) == has_parent.end()) {
            root_name = name;
            break;
        }
    }
    cout << "Root node: " << root_name << " (ID: " << name_to_id[root_name] << ")" << endl;

    parent_map.resize(num_nodes, -1);
    branch_lengths.resize(num_nodes, 0.0);

    for (const auto& edge : edges) {
        string p_name = get<0>(edge);
        string c_name = get<1>(edge);
        double b_len = get<2>(edge);

        int p_id = name_to_id[p_name];
        int c_id = name_to_id[c_name];

        parent_map[c_id] = p_id;
        branch_lengths[c_id] = b_len;
    }

    root_prior = {0.25, 0.25, 0.25, 0.25};
}

void load_base_composition(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening base composition file: " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == 'T' || line[0] == 'N' || line[0] == 'G') continue;
        istringstream iss(line);
        int state;
        double freq;
        if (iss >> state >> freq) {
            if (state >= 0 && state < 4) {
                root_prior[state] = freq;
            }
        }
    }
    fin.close();

    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += root_prior[i];
    for (int i = 0; i < 4; i++) root_prior[i] /= sum;

    cout << "Base composition (A, C, G, T): " << root_prior[0] << ", "
         << root_prior[1] << ", " << root_prior[2] << ", " << root_prior[3] << endl;
}

vector<pair<vector<int>, int>> load_patterns(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening pattern file: " << filename << endl;
        exit(1);
    }

    vector<pair<vector<int>, int>> patterns;
    string line;

    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;

        vector<int> pattern(num_leaves);
        for (int i = 0; i < num_leaves; i++) {
            iss >> pattern[i];
        }
        patterns.push_back({pattern, weight});
    }

    fin.close();
    return patterns;
}

CliqueTree* build_clique_tree() {
    CliqueTree* tree = new CliqueTree();
    tree->root_node = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (parent_map[i] == -1) {
            tree->root_node = i;
            break;
        }
    }

    map<int, vector<Clique*>> cliques_by_child;

    int clique_id = 0;
    for (int child = 0; child < num_nodes; child++) {
        int parent = parent_map[child];
        if (parent != -1) {
            Clique* c = new Clique(clique_id++, parent, child);
            tree->cliques.push_back(c);
            cliques_by_child[child].push_back(c);
        }
    }

    for (Clique* c : tree->cliques) {
        for (Clique* potential_parent : cliques_by_child[c->x_node]) {
            c->parent = potential_parent;
            potential_parent->children.push_back(c);
            break;
        }
    }

    vector<Clique*> root_cliques;
    for (Clique* c : tree->cliques) {
        if (c->parent == nullptr) {
            root_cliques.push_back(c);
        }
    }

    if (!root_cliques.empty()) {
        tree->root_clique = root_cliques[0];
        for (size_t i = 1; i < root_cliques.size(); i++) {
            root_cliques[i]->parent = tree->root_clique;
            tree->root_clique->children.push_back(root_cliques[i]);
        }
    }

    // Build traversal orders
    tree->edgesForPostOrderTreeTraversal.clear();
    tree->edgesForPreOrderTreeTraversal.clear();

    vector<bool> visited(tree->cliques.size(), false);

    function<void(Clique*)> postorder = [&](Clique* c) {
        visited[c->id] = true;
        for (Clique* child : c->children) {
            if (!visited[child->id]) {
                postorder(child);
                tree->edgesForPostOrderTreeTraversal.push_back({c, child});
            }
        }
    };

    postorder(tree->root_clique);

    for (auto& edge : tree->edgesForPostOrderTreeTraversal) {
        tree->edgesForPreOrderTreeTraversal.push_back({edge.first, edge.second});
    }
    reverse(tree->edgesForPreOrderTreeTraversal.begin(), tree->edgesForPreOrderTreeTraversal.end());

    return tree;
}

/**
 * Prepare GPU tree data structure
 */
void prepare_gpu_tree_data(CliqueTree* tree, GPUTreeData& gpu_tree) {
    gpu_tree.num_cliques = tree->cliques.size();
    gpu_tree.num_nodes = num_nodes;
    gpu_tree.num_leaves = num_leaves;
    gpu_tree.root_node = tree->root_node;

    // Copy clique data
    for (int c = 0; c < gpu_tree.num_cliques; c++) {
        Clique* clique = tree->cliques[c];
        gpu_tree.clique_x_node[c] = clique->x_node;
        gpu_tree.clique_y_node[c] = clique->y_node;
        gpu_tree.clique_parent[c] = clique->parent ? clique->parent->id : -1;
        gpu_tree.clique_num_children[c] = clique->children.size();

        for (int ch = 0; ch < (int)clique->children.size(); ch++) {
            gpu_tree.clique_children[c][ch] = clique->children[ch]->id;
        }
    }

    // Copy traversal orders
    gpu_tree.num_postorder_edges = tree->edgesForPostOrderTreeTraversal.size();
    for (int e = 0; e < gpu_tree.num_postorder_edges; e++) {
        gpu_tree.postorder_edges[e][0] = tree->edgesForPostOrderTreeTraversal[e].first->id;
        gpu_tree.postorder_edges[e][1] = tree->edgesForPostOrderTreeTraversal[e].second->id;
    }

    gpu_tree.num_preorder_edges = tree->edgesForPreOrderTreeTraversal.size();
    for (int e = 0; e < gpu_tree.num_preorder_edges; e++) {
        gpu_tree.preorder_edges[e][0] = tree->edgesForPreOrderTreeTraversal[e].first->id;
        gpu_tree.preorder_edges[e][1] = tree->edgesForPreOrderTreeTraversal[e].second->id;
    }

    // Copy model parameters
    for (int n = 0; n < num_nodes; n++) {
        gpu_tree.branch_lengths[n] = branch_lengths[n];
    }

    for (int i = 0; i < 4; i++) {
        gpu_tree.root_prior[i] = root_prior[i];
    }

    gpu_tree.f81_mu = f81_mu;
}

int main(int argc, char** argv) {
    cout << "=== GPU Clique Tree Log-Likelihood Computation ===" << endl;

    string tree_file = "data/tree_edges.txt";
    string pattern_file = "data/patterns_1000.pat";
    string basecomp_file = "data/patterns_1000.basecomp";
    string taxon_order_file = "data/patterns_1000.taxon_order";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--tree" && i + 1 < argc) {
            tree_file = argv[++i];
        } else if (arg == "--patterns" && i + 1 < argc) {
            pattern_file = argv[++i];
        } else if (arg == "--basecomp" && i + 1 < argc) {
            basecomp_file = argv[++i];
        } else if (arg == "--taxon_order" && i + 1 < argc) {
            taxon_order_file = argv[++i];
        }
    }

    cout << "Loading data..." << endl;
    cout << "Tree file: " << tree_file << endl;
    cout << "Pattern file: " << pattern_file << endl;
    cout << "Base comp file: " << basecomp_file << endl;
    cout << "Taxon order file: " << taxon_order_file << endl;

    // Load tree structure
    load_tree_from_edges(tree_file, taxon_order_file);

    // Load base composition
    load_base_composition(basecomp_file);
    compute_f81_mu();

    // Build clique tree
    CliqueTree* tree = build_clique_tree();
    cout << "\nBuilt clique tree with " << tree->cliques.size() << " cliques" << endl;

    if (tree->cliques.size() > MAX_CLIQUES) {
        cerr << "Error: Too many cliques (" << tree->cliques.size() << "), max is " << MAX_CLIQUES << endl;
        exit(1);
    }

    // Load patterns
    auto patterns = load_patterns(pattern_file);
    cout << "Loaded " << patterns.size() << " patterns" << endl;

    // Prepare GPU tree data
    GPUTreeData gpu_tree;
    prepare_gpu_tree_data(tree, gpu_tree);

    // Copy tree data to GPU constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_tree, &gpu_tree, sizeof(GPUTreeData)));

    // Prepare pattern data for GPU
    int num_patterns = patterns.size();
    vector<int> flat_patterns(num_patterns * num_leaves);
    vector<int> weights(num_patterns);

    for (int p = 0; p < num_patterns; p++) {
        weights[p] = patterns[p].second;
        for (int l = 0; l < num_leaves; l++) {
            flat_patterns[p * num_leaves + l] = patterns[p].first[l];
        }
    }

    // Allocate GPU memory
    int* d_patterns;
    int* d_weights;
    double* d_pattern_lls;

    CUDA_CHECK(cudaMalloc(&d_patterns, num_patterns * num_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_lls, num_patterns * sizeof(double)));

    // Copy pattern data to GPU
    CUDA_CHECK(cudaMemcpy(d_patterns, flat_patterns.data(),
                          num_patterns * num_leaves * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(),
                          num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 128;
    int num_blocks = (num_patterns + block_size - 1) / block_size;

    cout << "\nLaunching GPU kernel..." << endl;
    cout << "Patterns: " << num_patterns << ", Blocks: " << num_blocks << ", Threads/block: " << block_size << endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    compute_pattern_lls_kernel<<<num_blocks, block_size>>>(d_patterns, d_weights, d_pattern_lls, num_patterns);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // Copy results back
    vector<double> pattern_lls(num_patterns);
    CUDA_CHECK(cudaMemcpy(pattern_lls.data(), d_pattern_lls,
                          num_patterns * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute total LL
    double total_ll = 0.0;
    for (int p = 0; p < num_patterns; p++) {
        total_ll += pattern_lls[p] * weights[p];
    }

    cout << "\n=== Results ===" << endl;
    cout << "GPU kernel time: " << gpu_time << " ms" << endl;
    cout << "Total log-likelihood: " << setprecision(10) << total_ll << endl;
    cout << "Expected LL (CPU reference): -19310.06331" << endl;

    double diff = abs(total_ll - (-19310.06331));
    cout << "Difference from reference: " << setprecision(10) << diff << endl;

    if (diff < 1e-3) {
        cout << "\nSUCCESS: GPU matches CPU reference!" << endl;
    } else {
        cout << "\nWARNING: GPU does not match CPU reference" << endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_patterns));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_pattern_lls));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete tree;

    return 0;
}
