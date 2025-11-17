/*
 * precision_memory_timing_test
 *
 * Tests numerical precision by randomizing the order in which pattern
 * log-likelihoods are summed on GPU. Different summation orders can reveal
 * the number of significant digits in the computation.
 *
 * Also measures GPU execution time and memory requirements for:
 * 1. Pruning algorithm (forward pass only - Felsenstein's algorithm)
 * 2. Propagation algorithm (forward + backward passes)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <functional>
#include <map>
#include <set>
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

// Data structures
struct TreeNode {
    int id;
    string name;
    bool is_leaf;
    int taxon_index;
    vector<int> child_edge_ids;
};

struct TreeEdge {
    int id;
    int parent_node;
    int child_node;
    double branch_length;
};

struct PrecisionTestTree {
    vector<TreeNode> nodes;
    vector<TreeEdge> edges;
    int root_node;
    vector<int> root_edge_ids;
    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<double> pattern_weights;
    vector<double> root_probabilities;
    vector<vector<vector<double>>> transition_matrices;
};

// GPU Kernels

// Forward pass kernel (Pruning algorithm - computes upward messages)
__global__ void forward_pass_kernel(
    int num_patterns,
    int num_edges,
    int num_taxa,
    const int* d_post_order_edges,
    const int* d_edge_child_nodes,
    const char* d_node_is_leaf,
    const int* d_leaf_taxon_indices,
    const uint8_t* d_pattern_bases,
    const double* d_transition_matrices,
    const int* d_node_num_children,
    const int* d_node_child_offsets,
    const int* d_node_child_edges,
    double* d_upward_messages,
    double* d_upward_log_scales
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    double* msgs = d_upward_messages + pid * num_edges * 4;
    double* scales = d_upward_log_scales + pid * num_edges;

    for (int e = 0; e < num_edges; e++) scales[e] = 0.0;

    for (int idx = 0; idx < num_edges; idx++) {
        int eid = d_post_order_edges[idx];
        int child = d_edge_child_nodes[eid];
        const double* P = d_transition_matrices + eid * 16;

        double msg[4] = {1.0, 1.0, 1.0, 1.0};

        if (d_node_is_leaf[child]) {
            int taxon = d_leaf_taxon_indices[child];
            uint8_t base = d_pattern_bases[pid * num_taxa + taxon];
            if (base < 4) {
                for (int ps = 0; ps < 4; ps++) {
                    msg[ps] = P[ps * 4 + base];
                }
            }
        } else {
            int nch = d_node_num_children[child];
            int off = d_node_child_offsets[child];
            for (int ps = 0; ps < 4; ps++) {
                double sum = 0.0;
                for (int cs = 0; cs < 4; cs++) {
                    double prod = P[ps * 4 + cs];
                    for (int c = 0; c < nch && c < 8; c++) {
                        int ce = d_node_child_edges[off + c];
                        prod *= msgs[ce * 4 + cs];
                    }
                    sum += prod;
                }
                msg[ps] = sum;
            }
        }

        double mx = fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3]));
        if (mx > 0.0) {
            for (int i = 0; i < 4; i++) msg[i] /= mx;
            scales[eid] = log(mx);
        }

        for (int i = 0; i < 4; i++) msgs[eid * 4 + i] = msg[i];
    }
}

// Backward pass kernel (for propagation algorithm)
__global__ void backward_pass_kernel(
    int num_patterns,
    int num_edges,
    const int* d_post_order_edges,
    const int* d_edge_child_nodes,
    const int* d_edge_parent_nodes,
    const double* d_transition_matrices,
    const int* d_node_num_children,
    const int* d_node_child_offsets,
    const int* d_node_child_edges,
    const int* d_root_edge_ids,
    int num_root_edges,
    const double* d_root_probs,
    const double* d_upward_messages,
    double* d_downward_messages,
    double* d_downward_log_scales
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    const double* up = d_upward_messages + pid * num_edges * 4;
    double* down = d_downward_messages + pid * num_edges * 4;
    double* scales = d_downward_log_scales + pid * num_edges;

    for (int e = 0; e < num_edges; e++) {
        scales[e] = 0.0;
        for (int i = 0; i < 4; i++) down[e * 4 + i] = 1.0;
    }

    for (int idx = num_edges - 1; idx >= 0; idx--) {
        int eid = d_post_order_edges[idx];
        int parent = d_edge_parent_nodes[eid];

        double msg[4];

        bool is_root = false;
        for (int r = 0; r < num_root_edges; r++) {
            if (d_root_edge_ids[r] == eid) {
                is_root = true;
                break;
            }
        }

        if (is_root) {
            for (int i = 0; i < 4; i++) msg[i] = d_root_probs[i];
            for (int r = 0; r < num_root_edges; r++) {
                int sib = d_root_edge_ids[r];
                if (sib != eid) {
                    for (int i = 0; i < 4; i++) msg[i] *= up[sib * 4 + i];
                }
            }
        } else {
            int parent_edge = -1;
            for (int e = 0; e < num_edges; e++) {
                if (d_edge_child_nodes[e] == parent) {
                    parent_edge = e;
                    break;
                }
            }

            if (parent_edge >= 0) {
                const double* P_par = d_transition_matrices + parent_edge * 16;
                for (int cs = 0; cs < 4; cs++) {
                    double sum = 0.0;
                    for (int ps = 0; ps < 4; ps++) {
                        sum += P_par[ps * 4 + cs] * down[parent_edge * 4 + ps];
                    }
                    msg[cs] = sum;
                }

                int nch = d_node_num_children[parent];
                int off = d_node_child_offsets[parent];
                for (int c = 0; c < nch && c < 8; c++) {
                    int sib = d_node_child_edges[off + c];
                    if (sib != eid) {
                        for (int i = 0; i < 4; i++) msg[i] *= up[sib * 4 + i];
                    }
                }
            } else {
                for (int i = 0; i < 4; i++) msg[i] = 1.0;
            }
        }

        double mx = fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3]));
        if (mx > 0.0) {
            for (int i = 0; i < 4; i++) msg[i] /= mx;
            scales[eid] = log(mx);
        }

        for (int i = 0; i < 4; i++) down[eid * 4 + i] = msg[i];
    }
}

// Compute pattern LL using pruning algorithm (root likelihood)
__global__ void compute_pattern_ll_pruning_kernel(
    int num_patterns,
    const int* d_pattern_indices,  // Randomized order
    const int* d_root_edges,
    int num_root_edges,
    const double* d_root_probs,
    const double* d_upward_messages,
    const double* d_upward_log_scales,
    int num_edges,
    const double* d_weights,
    double* d_pattern_ll
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_patterns) return;

    int pid = d_pattern_indices[idx];  // Use randomized index

    const double* msgs = d_upward_messages + pid * num_edges * 4;
    const double* scales = d_upward_log_scales + pid * num_edges;

    // Sum all log scales from all edges (not just root edges)
    double log_scale = 0.0;
    for (int e = 0; e < num_edges; e++) {
        log_scale += scales[e];
    }

    // Combine root edge messages
    double combined[4] = {1.0, 1.0, 1.0, 1.0};
    for (int r = 0; r < num_root_edges; r++) {
        int re = d_root_edges[r];
        for (int i = 0; i < 4; i++) {
            combined[i] *= msgs[re * 4 + i];
        }
    }

    // Scale combined messages
    double mx = fmax(fmax(combined[0], combined[1]), fmax(combined[2], combined[3]));
    if (mx > 0.0) {
        for (int i = 0; i < 4; i++) combined[i] /= mx;
        log_scale += log(mx);
    }

    // Compute site likelihood
    double ll = 0.0;
    for (int i = 0; i < 4; i++) {
        ll += d_root_probs[i] * combined[i];
    }

    double weight = d_weights[pid];
    d_pattern_ll[idx] = weight * (log(ll) + log_scale);  // Store in randomized position
}

// Compute pattern LL using propagation algorithm (edge likelihood)
// After forward (upward) pass: up[e] = P(data_below_e | parent_state), already transformed through P[e]
// After backward (downward) pass: down[e] = P(parent_state) * P(data_not_below_e | parent_state)
//
// At any edge e, the total likelihood is:
// L = sum_{parent_state} down[e][ps] * up[e][ps]
//
// This works because:
// - up[e][ps] contains P(data_below_e | parent_state=ps) after transformation through P[e]
// - down[e][ps] contains P(parent_state=ps) * P(data_from_rest_of_tree | parent_state=ps)
// The product gives the joint probability P(all_data, parent_state=ps), and sum gives P(all_data)
__global__ void compute_pattern_ll_propagation_kernel(
    int num_patterns,
    const int* d_pattern_indices,  // Randomized order
    int edge_id,  // Which edge to compute likelihood at
    const double* d_transition_matrices,
    const double* d_upward_messages,
    const double* d_upward_log_scales,
    const double* d_downward_messages,
    const double* d_downward_log_scales,
    int num_edges,
    const double* d_root_probs,
    const int* d_root_edges,
    int num_root_edges,
    const double* d_weights,
    double* d_pattern_ll
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_patterns) return;

    int pid = d_pattern_indices[idx];

    const double* up = d_upward_messages + pid * num_edges * 4;
    const double* down = d_downward_messages + pid * num_edges * 4;
    const double* up_scales = d_upward_log_scales + pid * num_edges;
    const double* down_scales = d_downward_log_scales + pid * num_edges;

    // Sum all log scales from all edges (for consistency with pruning)
    double log_scale = 0.0;
    for (int e = 0; e < num_edges; e++) {
        log_scale += up_scales[e];
    }

    // Compute likelihood at this edge
    // L = sum_{parent_state} down[e][ps] * up[e][ps]
    // NO transition matrix here - it's already incorporated in up[e]
    double ll = 0.0;
    for (int ps = 0; ps < 4; ps++) {
        ll += down[edge_id * 4 + ps] * up[edge_id * 4 + ps];
    }

    // Scale the likelihood product
    double mx = fmax(fmax(ll, 1e-300), 1e-300);  // Avoid log(0)
    log_scale += log(mx);
    ll = 1.0;  // Already accounted for in log_scale

    // Add the downward scale for this edge
    log_scale += down_scales[edge_id];

    double weight = d_weights[pid];
    d_pattern_ll[idx] = weight * log_scale;
}

// Simple reduction kernel to sum pattern log-likelihoods
__global__ void sum_ll_kernel(const double* d_pattern_ll, int n, double* d_total) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_pattern_ll[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_total, sdata[0]);
    }
}

// File loading functions (simplified versions)
void load_edge_file(const string& filename, PrecisionTestTree& tree) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open edge file: " << filename << endl;
        exit(1);
    }

    // Read all edges first (format: parent_name child_name branch_length)
    vector<pair<string, string>> edge_pairs;
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

    // Build node set
    set<string> node_names;
    for (auto& e : edge_pairs) {
        node_names.insert(e.first);
        node_names.insert(e.second);
    }

    // Create node ID mapping
    map<string, int> name_to_id;
    int nid = 0;
    for (const string& name : node_names) {
        TreeNode node;
        node.id = nid;
        node.name = name;
        node.is_leaf = true;  // Will be set to false if it has children
        node.taxon_index = -1;
        tree.nodes.push_back(node);
        name_to_id[name] = nid;
        nid++;
    }

    // Create edges
    for (size_t i = 0; i < edge_pairs.size(); i++) {
        int parent_id = name_to_id[edge_pairs[i].first];
        int child_id = name_to_id[edge_pairs[i].second];

        TreeEdge edge;
        edge.id = tree.edges.size();
        edge.parent_node = parent_id;
        edge.child_node = child_id;
        edge.branch_length = lengths[i];
        tree.edges.push_back(edge);

        tree.nodes[parent_id].is_leaf = false;
        tree.nodes[parent_id].child_edge_ids.push_back(edge.id);
    }

    // Find root (node with children but no parent)
    tree.root_node = -1;
    for (auto& node : tree.nodes) {
        if (!node.child_edge_ids.empty()) {
            bool is_root = true;
            for (auto& e : tree.edges) {
                if (e.child_node == node.id) {
                    is_root = false;
                    break;
                }
            }
            if (is_root) {
                tree.root_node = node.id;
                break;
            }
        }
    }

    if (tree.root_node >= 0) {
        for (int eid : tree.nodes[tree.root_node].child_edge_ids) {
            tree.root_edge_ids.push_back(eid);
        }
    }
}

void load_pattern_file(const string& filename, PrecisionTestTree& tree) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open pattern file: " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        double weight;
        iss >> weight;
        tree.pattern_weights.push_back(weight);

        // Read integer bases (0=A, 1=C, 2=G, 3=T, 4=N)
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

void load_taxon_order(const string& filename, PrecisionTestTree& tree) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open taxon order file: " << filename << endl;
        exit(1);
    }

    vector<string> taxon_names;
    string line;
    getline(file, line);  // Skip header line (taxon_name,position)

    while (getline(file, line)) {
        if (!line.empty()) {
            // CSV format: taxon_name,position
            size_t comma = line.find(',');
            if (comma != string::npos) {
                taxon_names.push_back(line.substr(0, comma));
            }
        }
    }

    tree.num_taxa = taxon_names.size();

    int taxon_idx = 0;
    for (auto& node : tree.nodes) {
        if (node.is_leaf) {
            for (size_t i = 0; i < taxon_names.size(); i++) {
                if (taxon_names[i] == node.name) {
                    node.taxon_index = i;
                    taxon_idx++;
                    break;
                }
            }
        }
    }
}

void load_base_composition(const string& filename, PrecisionTestTree& tree) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open base composition file: " << filename << endl;
        exit(1);
    }

    tree.root_probabilities.resize(4);
    // Format: index<tab>probability<tab>(count)
    // Example: 0	0.2831852914	(13645409)
    string line;
    for (int i = 0; i < 4 && getline(file, line); i++) {
        istringstream iss(line);
        int idx;
        double prob;
        iss >> idx >> prob;
        tree.root_probabilities[idx] = prob;
    }
}

void compute_f81_transition_matrix(double mu, double t,
                                   const vector<double>& pi,
                                   vector<vector<double>>& P) {
    P.resize(4, vector<double>(4));
    double exp_term = exp(-mu * t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i][j] = exp_term + pi[j] * (1.0 - exp_term);
            } else {
                P[i][j] = pi[j] * (1.0 - exp_term);
            }
        }
    }
}

void initialize_transition_matrices(PrecisionTestTree& tree, double mu) {
    tree.transition_matrices.resize(tree.edges.size());
    for (auto& edge : tree.edges) {
        compute_f81_transition_matrix(mu, edge.branch_length,
                                      tree.root_probabilities,
                                      tree.transition_matrices[edge.id]);
    }
}

vector<int> compute_post_order(const PrecisionTestTree& tree) {
    vector<int> order;
    vector<bool> visited(tree.edges.size(), false);

    function<void(int)> visit = [&](int node_id) {
        for (int eid : tree.nodes[node_id].child_edge_ids) {
            int child = tree.edges[eid].child_node;
            visit(child);
            if (!visited[eid]) {
                order.push_back(eid);
                visited[eid] = true;
            }
        }
    };

    visit(tree.root_node);
    return order;
}

void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " -e <edge_file> -p <pattern_file> -x <taxon_order> "
         << "-b <base_comp> [-n <num_trials>] [-s <seed>]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -e  Edge list file" << endl;
    cerr << "  -p  Pattern file" << endl;
    cerr << "  -x  Taxon order file" << endl;
    cerr << "  -b  Base composition file" << endl;
    cerr << "  -n  Number of randomization trials (default: 10)" << endl;
    cerr << "  -s  Random seed (default: current time)" << endl;
}

int main(int argc, char** argv) {
    string edge_file, pattern_file, taxon_order_file, basecomp_file;
    int num_trials = 10;
    unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
    double mu = 0.0;  // F81 rate - will be computed from base composition

    // Parse command line
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-e" && i + 1 < argc) edge_file = argv[++i];
        else if (arg == "-p" && i + 1 < argc) pattern_file = argv[++i];
        else if (arg == "-x" && i + 1 < argc) taxon_order_file = argv[++i];
        else if (arg == "-b" && i + 1 < argc) basecomp_file = argv[++i];
        else if (arg == "-n" && i + 1 < argc) num_trials = atoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) seed = atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (edge_file.empty() || pattern_file.empty() ||
        taxon_order_file.empty() || basecomp_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    cout << "=== Precision, Memory, and Timing Test ===" << endl;
    cout << "Number of trials: " << num_trials << endl;
    cout << "Random seed: " << seed << endl;
    cout << endl;

    // Load data
    PrecisionTestTree tree;
    cout << "Loading data..." << endl;
    load_edge_file(edge_file, tree);
    load_pattern_file(pattern_file, tree);
    load_taxon_order(taxon_order_file, tree);
    load_base_composition(basecomp_file, tree);

    // Compute F81 mu from base composition (same as CPU code)
    // mu = 1.0 / (1.0 - sum(pi_i^2))
    double S2 = 0.0;
    for (int k = 0; k < 4; k++) {
        S2 += tree.root_probabilities[k] * tree.root_probabilities[k];
    }
    mu = 1.0 / max(1e-14, 1.0 - S2);
    cout << "F81 mu is " << mu << endl;

    initialize_transition_matrices(tree, mu);

    cout << "Loaded: " << tree.num_taxa << " taxa, " << tree.edges.size()
         << " edges, " << tree.num_patterns << " patterns" << endl;
    cout << "Root probs: [" << tree.root_probabilities[0] << ", "
         << tree.root_probabilities[1] << ", " << tree.root_probabilities[2]
         << ", " << tree.root_probabilities[3] << "]" << endl;

    // Compute post-order traversal
    vector<int> post_order = compute_post_order(tree);
    int num_edges = tree.edges.size();

    // Prepare GPU data structures
    vector<int> edge_child(num_edges), edge_parent(num_edges);
    vector<char> node_is_leaf(tree.nodes.size());  // Use char instead of bool for GPU compatibility
    vector<int> leaf_taxon(tree.nodes.size());
    vector<double> trans_flat(num_edges * 16);

    for (auto& edge : tree.edges) {
        edge_child[edge.id] = edge.child_node;
        edge_parent[edge.id] = edge.parent_node;
    }

    for (auto& node : tree.nodes) {
        node_is_leaf[node.id] = node.is_leaf ? 1 : 0;
        leaf_taxon[node.id] = node.taxon_index;
    }

    for (int e = 0; e < num_edges; e++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                trans_flat[e * 16 + i * 4 + j] = tree.transition_matrices[e][i][j];
            }
        }
    }

    // Node children info
    vector<int> node_nch(tree.nodes.size()), node_off(tree.nodes.size());
    vector<int> node_edges;
    int offset = 0;
    for (auto& node : tree.nodes) {
        node_nch[node.id] = node.child_edge_ids.size();
        node_off[node.id] = offset;
        for (int eid : node.child_edge_ids) {
            node_edges.push_back(eid);
        }
        offset += node.child_edge_ids.size();
    }

    // Allocate GPU memory
    cout << "Allocating GPU memory..." << endl;
    cout << "  Post order size: " << post_order.size() << endl;
    cout << "  Node edges size: " << node_edges.size() << endl;
    cout << "  Root edges: " << tree.root_edge_ids.size() << endl;

    int* d_post_order, *d_edge_child, *d_edge_parent;
    char* d_is_leaf;
    int* d_leaf_taxon;
    uint8_t* d_bases;
    double* d_trans, *d_weights, *d_root_probs;
    int* d_node_nch, *d_node_off, *d_node_edges;
    int* d_root_edges;
    double* d_up_msg, *d_up_scale, *d_down_msg, *d_down_scale;
    double* d_pattern_ll, *d_total_ll;
    int* d_pattern_indices;

    cout << "  Allocating basic arrays..." << endl;
    CUDA_CHECK(cudaMalloc(&d_post_order, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_child, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_parent, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_leaf, tree.nodes.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_leaf_taxon, tree.nodes.size() * sizeof(int)));
    cout << "  Allocating pattern bases..." << endl;
    CUDA_CHECK(cudaMalloc(&d_bases, tree.num_patterns * tree.num_taxa * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_trans, num_edges * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights, tree.num_patterns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_root_probs, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_node_nch, tree.nodes.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_off, tree.nodes.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_edges, node_edges.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_root_edges, tree.root_edge_ids.size() * sizeof(int)));
    cout << "  Allocating upward messages (" << (tree.num_patterns * num_edges * 4 * sizeof(double) / 1e9) << " GB)..." << endl;
    CUDA_CHECK(cudaMalloc(&d_up_msg, tree.num_patterns * num_edges * 4 * sizeof(double)));
    cout << "  Allocating upward scales..." << endl;
    CUDA_CHECK(cudaMalloc(&d_up_scale, tree.num_patterns * num_edges * sizeof(double)));
    cout << "  Allocating downward messages..." << endl;
    CUDA_CHECK(cudaMalloc(&d_down_msg, tree.num_patterns * num_edges * 4 * sizeof(double)));
    cout << "  Allocating downward scales..." << endl;
    CUDA_CHECK(cudaMalloc(&d_down_scale, tree.num_patterns * num_edges * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pattern_ll, tree.num_patterns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_total_ll, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pattern_indices, tree.num_patterns * sizeof(int)));
    cout << "  GPU memory allocation complete." << endl;

    // Calculate memory requirements for each algorithm
    size_t shared_memory = 0;
    shared_memory += num_edges * sizeof(int);  // d_post_order
    shared_memory += num_edges * sizeof(int);  // d_edge_child
    shared_memory += num_edges * sizeof(int);  // d_edge_parent
    shared_memory += tree.nodes.size() * sizeof(char);  // d_is_leaf
    shared_memory += tree.nodes.size() * sizeof(int);   // d_leaf_taxon
    shared_memory += tree.num_patterns * tree.num_taxa * sizeof(uint8_t);  // d_bases
    shared_memory += num_edges * 16 * sizeof(double);   // d_trans
    shared_memory += tree.num_patterns * sizeof(double);  // d_weights
    shared_memory += 4 * sizeof(double);  // d_root_probs
    shared_memory += tree.nodes.size() * sizeof(int);   // d_node_nch
    shared_memory += tree.nodes.size() * sizeof(int);   // d_node_off
    shared_memory += node_edges.size() * sizeof(int);   // d_node_edges
    shared_memory += tree.root_edge_ids.size() * sizeof(int);  // d_root_edges
    shared_memory += tree.num_patterns * sizeof(double);  // d_pattern_ll
    shared_memory += sizeof(double);  // d_total_ll
    shared_memory += tree.num_patterns * sizeof(int);  // d_pattern_indices

    size_t pruning_memory = shared_memory;
    pruning_memory += tree.num_patterns * num_edges * 4 * sizeof(double);  // d_up_msg
    pruning_memory += tree.num_patterns * num_edges * sizeof(double);       // d_up_scale

    size_t propagation_memory = pruning_memory;
    propagation_memory += tree.num_patterns * num_edges * 4 * sizeof(double);  // d_down_msg
    propagation_memory += tree.num_patterns * num_edges * sizeof(double);       // d_down_scale

    cout << "\n=== Memory Requirements ===" << endl;
    cout << "Shared (tree structure, patterns):  " << fixed << setprecision(6) << (shared_memory / 1e9) << " GB" << endl;
    cout << "Pruning algorithm total:            " << fixed << setprecision(6) << (pruning_memory / 1e9) << " GB" << endl;
    cout << "Propagation algorithm total:        " << fixed << setprecision(6) << (propagation_memory / 1e9) << " GB" << endl;
    cout << "Additional memory for propagation:  " << fixed << setprecision(6) << ((propagation_memory - pruning_memory) / 1e9) << " GB" << endl;
    cout << "Propagation / Pruning memory ratio: " << fixed << setprecision(4) << ((double)propagation_memory / pruning_memory) << endl;
    cout << endl;

    // Copy static data to GPU
    cout << "Copying data to GPU..." << endl;
    cout << "  post_order..." << endl;
    CUDA_CHECK(cudaMemcpy(d_post_order, post_order.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  edge_child..." << endl;
    CUDA_CHECK(cudaMemcpy(d_edge_child, edge_child.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  edge_parent..." << endl;
    CUDA_CHECK(cudaMemcpy(d_edge_parent, edge_parent.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  node_is_leaf (size=" << node_is_leaf.size() << ")..." << endl;
    CUDA_CHECK(cudaMemcpy(d_is_leaf, node_is_leaf.data(), tree.nodes.size() * sizeof(char), cudaMemcpyHostToDevice));
    cout << "  leaf_taxon (size=" << leaf_taxon.size() << ")..." << endl;
    CUDA_CHECK(cudaMemcpy(d_leaf_taxon, leaf_taxon.data(), tree.nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  pattern_bases (size=" << tree.pattern_bases.size() << ", expect=" << (tree.num_patterns * tree.num_taxa) << ")..." << endl;
    CUDA_CHECK(cudaMemcpy(d_bases, tree.pattern_bases.data(), tree.num_patterns * tree.num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice));
    cout << "  trans_flat..." << endl;
    CUDA_CHECK(cudaMemcpy(d_trans, trans_flat.data(), num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
    cout << "  weights..." << endl;
    CUDA_CHECK(cudaMemcpy(d_weights, tree.pattern_weights.data(), tree.num_patterns * sizeof(double), cudaMemcpyHostToDevice));
    cout << "  root_probs..." << endl;
    CUDA_CHECK(cudaMemcpy(d_root_probs, tree.root_probabilities.data(), 4 * sizeof(double), cudaMemcpyHostToDevice));
    cout << "  node_nch..." << endl;
    CUDA_CHECK(cudaMemcpy(d_node_nch, node_nch.data(), tree.nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  node_off..." << endl;
    CUDA_CHECK(cudaMemcpy(d_node_off, node_off.data(), tree.nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  node_edges..." << endl;
    CUDA_CHECK(cudaMemcpy(d_node_edges, node_edges.data(), node_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
    cout << "  root_edges..." << endl;
    CUDA_CHECK(cudaMemcpy(d_root_edges, tree.root_edge_ids.data(), tree.root_edge_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    cout << "Data copy complete." << endl;

    // Compute forward and backward passes once (these don't depend on summation order)
    int bs = 256;
    int nb = (tree.num_patterns + bs - 1) / bs;

    // Create CUDA events for timing
    cudaEvent_t start_fwd, end_fwd, start_bwd, end_bwd;
    cudaEventCreate(&start_fwd);
    cudaEventCreate(&end_fwd);
    cudaEventCreate(&start_bwd);
    cudaEventCreate(&end_bwd);

    // Prepare random number generator
    mt19937 rng(seed);
    vector<int> pattern_indices(tree.num_patterns);
    iota(pattern_indices.begin(), pattern_indices.end(), 0);

    // Choose an edge for propagation algorithm (use first root edge)
    int prop_edge = tree.root_edge_ids[0];
    cout << "Propagation edge ID: " << prop_edge << " (root edge)" << endl;

    // Storage for results - combined LL, timing, and memory in each trial
    vector<double> pruning_lls(num_trials);
    vector<double> propagation_lls(num_trials);
    vector<float> fwd_times(num_trials);
    vector<float> bwd_times(num_trials);
    vector<float> time_ratios(num_trials);

    cout << endl << "Running " << num_trials << " trials (each measures LL, timing, and memory)..." << endl;
    cout << string(160, '-') << endl;
    cout << setw(10) << "Trial"
         << setw(30) << "Pruning LL"
         << setw(30) << "Propagation LL"
         << setw(18) << "Difference"
         << setw(14) << "Fwd_ms"
         << setw(14) << "Bwd_ms"
         << setw(14) << "Time_Ratio"
         << setw(14) << "Mem_Ratio" << endl;
    cout << string(160, '-') << endl;

    for (int trial = 0; trial < num_trials; trial++) {
        // Shuffle pattern indices for this trial
        shuffle(pattern_indices.begin(), pattern_indices.end(), rng);

        // Copy shuffled indices to GPU
        CUDA_CHECK(cudaMemcpy(d_pattern_indices, pattern_indices.data(),
                              tree.num_patterns * sizeof(int), cudaMemcpyHostToDevice));

        // Forward pass with timing (Pruning algorithm)
        cudaEventRecord(start_fwd);
        forward_pass_kernel<<<nb, bs>>>(tree.num_patterns, num_edges, tree.num_taxa,
            d_post_order, d_edge_child, d_is_leaf, d_leaf_taxon, d_bases, d_trans,
            d_node_nch, d_node_off, d_node_edges, d_up_msg, d_up_scale);
        cudaEventRecord(end_fwd);
        cudaDeviceSynchronize();

        float fwd_time_ms = 0;
        cudaEventElapsedTime(&fwd_time_ms, start_fwd, end_fwd);
        fwd_times[trial] = fwd_time_ms;

        // Compute pruning LL with randomized summation order
        CUDA_CHECK(cudaMemset(d_total_ll, 0, sizeof(double)));
        compute_pattern_ll_pruning_kernel<<<nb, bs>>>(tree.num_patterns, d_pattern_indices,
            d_root_edges, tree.root_edge_ids.size(), d_root_probs,
            d_up_msg, d_up_scale, num_edges, d_weights, d_pattern_ll);
        sum_ll_kernel<<<nb, bs, bs * sizeof(double)>>>(d_pattern_ll, tree.num_patterns, d_total_ll);
        cudaDeviceSynchronize();

        double pruning_ll;
        CUDA_CHECK(cudaMemcpy(&pruning_ll, d_total_ll, sizeof(double), cudaMemcpyDeviceToHost));
        pruning_lls[trial] = pruning_ll;

        // Backward pass with timing (for Propagation algorithm)
        cudaEventRecord(start_bwd);
        backward_pass_kernel<<<nb, bs>>>(tree.num_patterns, num_edges,
            d_post_order, d_edge_child, d_edge_parent, d_trans,
            d_node_nch, d_node_off, d_node_edges,
            d_root_edges, tree.root_edge_ids.size(), d_root_probs,
            d_up_msg, d_down_msg, d_down_scale);
        cudaEventRecord(end_bwd);
        cudaDeviceSynchronize();

        float bwd_time_ms = 0;
        cudaEventElapsedTime(&bwd_time_ms, start_bwd, end_bwd);
        bwd_times[trial] = bwd_time_ms;

        time_ratios[trial] = (fwd_time_ms + bwd_time_ms) / fwd_time_ms;

        // Compute propagation LL with randomized summation order
        CUDA_CHECK(cudaMemset(d_total_ll, 0, sizeof(double)));
        compute_pattern_ll_propagation_kernel<<<nb, bs>>>(tree.num_patterns, d_pattern_indices,
            prop_edge, d_trans, d_up_msg, d_up_scale, d_down_msg, d_down_scale,
            num_edges, d_root_probs, d_root_edges, tree.root_edge_ids.size(),
            d_weights, d_pattern_ll);
        sum_ll_kernel<<<nb, bs, bs * sizeof(double)>>>(d_pattern_ll, tree.num_patterns, d_total_ll);
        cudaDeviceSynchronize();

        double propagation_ll;
        CUDA_CHECK(cudaMemcpy(&propagation_ll, d_total_ll, sizeof(double), cudaMemcpyDeviceToHost));
        propagation_lls[trial] = propagation_ll;

        double diff = pruning_ll - propagation_ll;
        double mem_ratio = (double)propagation_memory / pruning_memory;

        cout << setw(10) << (trial + 1)
             << setw(30) << fixed << setprecision(15) << pruning_ll
             << setw(30) << propagation_ll
             << setw(18) << scientific << setprecision(6) << diff
             << setw(14) << fixed << setprecision(6) << fwd_time_ms
             << setw(14) << bwd_time_ms
             << setw(14) << setprecision(6) << time_ratios[trial]
             << setw(14) << setprecision(4) << mem_ratio << endl;
    }

    cout << string(160, '-') << endl;

    cudaEventDestroy(start_fwd);
    cudaEventDestroy(end_fwd);
    cudaEventDestroy(start_bwd);
    cudaEventDestroy(end_bwd);

    // Compute statistics
    double pruning_sum = 0.0, propagation_sum = 0.0;
    double pruning_min = pruning_lls[0], pruning_max = pruning_lls[0];
    double propagation_min = propagation_lls[0], propagation_max = propagation_lls[0];

    for (int i = 0; i < num_trials; i++) {
        pruning_sum += pruning_lls[i];
        propagation_sum += propagation_lls[i];
        pruning_min = min(pruning_min, pruning_lls[i]);
        pruning_max = max(pruning_max, pruning_lls[i]);
        propagation_min = min(propagation_min, propagation_lls[i]);
        propagation_max = max(propagation_max, propagation_lls[i]);
    }

    double pruning_avg = pruning_sum / num_trials;
    double propagation_avg = propagation_sum / num_trials;

    // Compute standard deviation
    double pruning_var = 0.0, propagation_var = 0.0;
    for (int i = 0; i < num_trials; i++) {
        pruning_var += (pruning_lls[i] - pruning_avg) * (pruning_lls[i] - pruning_avg);
        propagation_var += (propagation_lls[i] - propagation_avg) * (propagation_lls[i] - propagation_avg);
    }
    double pruning_std = sqrt(pruning_var / num_trials);
    double propagation_std = sqrt(propagation_var / num_trials);

    cout << endl << "=== Summary ===" << endl;
    cout << "Pruning Algorithm:" << endl;
    cout << "  Average LL:  " << fixed << setprecision(10) << pruning_avg << endl;
    cout << "  Std Dev:     " << scientific << setprecision(4) << pruning_std << endl;
    cout << "  Range:       " << fixed << setprecision(10) << pruning_min << " to " << pruning_max << endl;
    cout << "  Max diff:    " << scientific << setprecision(4) << (pruning_max - pruning_min) << endl;
    cout << endl;

    cout << "Propagation Algorithm:" << endl;
    cout << "  Average LL:  " << fixed << setprecision(10) << propagation_avg << endl;
    cout << "  Std Dev:     " << scientific << setprecision(4) << propagation_std << endl;
    cout << "  Range:       " << fixed << setprecision(10) << propagation_min << " to " << propagation_max << endl;
    cout << "  Max diff:    " << scientific << setprecision(4) << (propagation_max - propagation_min) << endl;
    cout << endl;

    // Estimate significant digits
    double pruning_rel_err = pruning_std / fabs(pruning_avg);
    double propagation_rel_err = propagation_std / fabs(propagation_avg);
    int pruning_digits = (pruning_rel_err > 0) ? (int)(-log10(pruning_rel_err)) : 15;
    int propagation_digits = (propagation_rel_err > 0) ? (int)(-log10(propagation_rel_err)) : 15;

    cout << "Estimated significant digits:" << endl;
    cout << "  Pruning:     ~" << pruning_digits << " digits" << endl;
    cout << "  Propagation: ~" << propagation_digits << " digits" << endl;
    cout << endl;

    // Compute timing statistics
    float fwd_sum = 0, bwd_sum = 0, ratio_sum = 0;
    float fwd_min = fwd_times[0], fwd_max = fwd_times[0];
    float bwd_min = bwd_times[0], bwd_max = bwd_times[0];
    float ratio_min = time_ratios[0], ratio_max = time_ratios[0];

    for (int t = 0; t < num_trials; t++) {
        fwd_sum += fwd_times[t];
        bwd_sum += bwd_times[t];
        ratio_sum += time_ratios[t];
        fwd_min = min(fwd_min, fwd_times[t]);
        fwd_max = max(fwd_max, fwd_times[t]);
        bwd_min = min(bwd_min, bwd_times[t]);
        bwd_max = max(bwd_max, bwd_times[t]);
        ratio_min = min(ratio_min, time_ratios[t]);
        ratio_max = max(ratio_max, time_ratios[t]);
    }

    float fwd_avg = fwd_sum / num_trials;
    float bwd_avg = bwd_sum / num_trials;
    float ratio_avg = ratio_sum / num_trials;

    // Compute standard deviations
    float fwd_var = 0, bwd_var = 0, ratio_var = 0;
    for (int t = 0; t < num_trials; t++) {
        fwd_var += (fwd_times[t] - fwd_avg) * (fwd_times[t] - fwd_avg);
        bwd_var += (bwd_times[t] - bwd_avg) * (bwd_times[t] - bwd_avg);
        ratio_var += (time_ratios[t] - ratio_avg) * (time_ratios[t] - ratio_avg);
    }
    float fwd_std = sqrt(fwd_var / num_trials);
    float bwd_std = sqrt(bwd_var / num_trials);
    float ratio_std = sqrt(ratio_var / num_trials);

    cout << "=== Timing Summary ===" << endl;
    cout << "Forward pass (Pruning):" << endl;
    cout << "  Average: " << fixed << setprecision(3) << fwd_avg << " ms" << endl;
    cout << "  Std Dev: " << scientific << setprecision(4) << fwd_std << " ms" << endl;
    cout << "  Range:   " << fixed << setprecision(3) << fwd_min << " to " << fwd_max << " ms" << endl;

    cout << "\nBackward pass:" << endl;
    cout << "  Average: " << fixed << setprecision(3) << bwd_avg << " ms" << endl;
    cout << "  Std Dev: " << scientific << setprecision(4) << bwd_std << " ms" << endl;
    cout << "  Range:   " << fixed << setprecision(3) << bwd_min << " to " << bwd_max << " ms" << endl;

    cout << "\nPropagation / Pruning time ratio:" << endl;
    cout << "  Average: " << fixed << setprecision(6) << ratio_avg << endl;
    cout << "  Std Dev: " << scientific << setprecision(4) << ratio_std << endl;
    cout << "  Range:   " << fixed << setprecision(6) << ratio_min << " to " << ratio_max << endl;
    cout << endl;

    cout << "=== Memory Summary ===" << endl;
    cout << "Pruning algorithm:     " << fixed << setprecision(6) << (pruning_memory / 1e9) << " GB" << endl;
    cout << "Propagation algorithm: " << fixed << setprecision(6) << (propagation_memory / 1e9) << " GB" << endl;
    cout << "Memory ratio (Prop/Prune): " << fixed << setprecision(4) << ((double)propagation_memory / pruning_memory) << endl;

    // Cleanup
    cudaFree(d_post_order);
    cudaFree(d_edge_child);
    cudaFree(d_edge_parent);
    cudaFree(d_is_leaf);
    cudaFree(d_leaf_taxon);
    cudaFree(d_bases);
    cudaFree(d_trans);
    cudaFree(d_weights);
    cudaFree(d_root_probs);
    cudaFree(d_node_nch);
    cudaFree(d_node_off);
    cudaFree(d_node_edges);
    cudaFree(d_root_edges);
    cudaFree(d_up_msg);
    cudaFree(d_up_scale);
    cudaFree(d_down_msg);
    cudaFree(d_down_scale);
    cudaFree(d_pattern_ll);
    cudaFree(d_total_ll);
    cudaFree(d_pattern_indices);

    return 0;
}
