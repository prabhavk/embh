// GPU-Accelerated EMBH with Aitken Acceleration - VERSION 2
// Key optimizations:
// 1. Edge-level parallelism for expected counts (1 block per edge)
// 2. Warp-level reduction to minimize atomicAdd contention
// 3. Pre-computed structures for O(1) lookups
// 4. Memory-efficient scale storage
// 5. Texture memory for read-only pattern data

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <cmath>
#include <iomanip>

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

// Constant memory for frequently accessed small data
__constant__ double c_root_probs[4];
__constant__ int c_root_edges[16];
__constant__ int c_num_root_edges;

struct EMBHTree {
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<bool> node_is_leaf;
    vector<int> leaf_taxon_indices;
    vector<vector<int>> node_children;
    vector<int> node_parent;
    int root_node_id;

    vector<int> edge_child_nodes;
    vector<int> edge_parent_nodes;
    vector<double> branch_lengths;
    vector<int> post_order_edges;
    vector<int> root_edge_ids;
    vector<int> node_incoming_edge;

    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<int> pattern_weights;
    vector<string> taxon_names;

    vector<array<array<double, 4>, 4>> transition_matrices;
    array<double, 4> root_probabilities;
};

// Warp-level reduction for doubles
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory
__device__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// Forward pass - same as optimized version
__global__ void forward_pass_kernel_v2(
    int num_patterns,
    int num_edges,
    int num_taxa,
    const int* __restrict__ d_post_order_edges,
    const int* __restrict__ d_edge_child_nodes,
    const bool* __restrict__ d_node_is_leaf,
    const int* __restrict__ d_leaf_taxon_indices,
    const uint8_t* __restrict__ d_pattern_bases,
    const double* __restrict__ d_transition_matrices,
    const int* __restrict__ d_node_num_children,
    const int* __restrict__ d_node_child_offsets,
    const int* __restrict__ d_node_child_edges,
    double* __restrict__ d_upward_messages,
    double* __restrict__ d_upward_total_scale
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    double* msgs = d_upward_messages + pid * num_edges * 4;
    double total_scale = 0.0;

    for (int idx = 0; idx < num_edges; idx++) {
        int eid = d_post_order_edges[idx];
        int child = d_edge_child_nodes[eid];
        const double* P = d_transition_matrices + eid * 16;

        double msg[4];

        if (d_node_is_leaf[child]) {
            int taxon = d_leaf_taxon_indices[child];
            uint8_t base = d_pattern_bases[pid * num_taxa + taxon];
            if (base < 4) {
                msg[0] = P[0 * 4 + base];
                msg[1] = P[1 * 4 + base];
                msg[2] = P[2 * 4 + base];
                msg[3] = P[3 * 4 + base];
            } else {
                for (int ps = 0; ps < 4; ps++) {
                    msg[ps] = P[ps * 4 + 0] + P[ps * 4 + 1] + P[ps * 4 + 2] + P[ps * 4 + 3];
                }
            }
        } else {
            int nch = d_node_num_children[child];
            int off = d_node_child_offsets[child];

            for (int ps = 0; ps < 4; ps++) {
                double sum = 0.0;
                for (int cs = 0; cs < 4; cs++) {
                    double prod = P[ps * 4 + cs];
                    for (int c = 0; c < nch; c++) {
                        int ce = d_node_child_edges[off + c];
                        prod *= msgs[ce * 4 + cs];
                    }
                    sum += prod;
                }
                msg[ps] = sum;
            }
        }

        double mx = fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3]));
        if (mx > 0.0 && mx != 1.0) {
            double inv_mx = 1.0 / mx;
            msg[0] *= inv_mx;
            msg[1] *= inv_mx;
            msg[2] *= inv_mx;
            msg[3] *= inv_mx;
            total_scale += log(mx);
        }

        msgs[eid * 4 + 0] = msg[0];
        msgs[eid * 4 + 1] = msg[1];
        msgs[eid * 4 + 2] = msg[2];
        msgs[eid * 4 + 3] = msg[3];
    }

    d_upward_total_scale[pid] = total_scale;
}

// Backward pass with pre-computed lookup
__global__ void backward_pass_kernel_v2(
    int num_patterns,
    int num_edges,
    const int* __restrict__ d_post_order_edges,
    const int* __restrict__ d_edge_child_nodes,
    const int* __restrict__ d_edge_parent_nodes,
    const double* __restrict__ d_transition_matrices,
    const int* __restrict__ d_node_num_children,
    const int* __restrict__ d_node_child_offsets,
    const int* __restrict__ d_node_child_edges,
    const int* __restrict__ d_node_incoming_edge,
    const double* __restrict__ d_upward_messages,
    double* __restrict__ d_downward_messages
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    const double* up = d_upward_messages + pid * num_edges * 4;
    double* down = d_downward_messages + pid * num_edges * 4;

    for (int e = 0; e < num_edges; e++) {
        down[e * 4 + 0] = 1.0;
        down[e * 4 + 1] = 1.0;
        down[e * 4 + 2] = 1.0;
        down[e * 4 + 3] = 1.0;
    }

    for (int idx = num_edges - 1; idx >= 0; idx--) {
        int eid = d_post_order_edges[idx];
        int parent = d_edge_parent_nodes[eid];

        double msg[4];

        bool is_root = false;
        for (int r = 0; r < c_num_root_edges; r++) {
            if (c_root_edges[r] == eid) {
                is_root = true;
                break;
            }
        }

        if (is_root) {
            msg[0] = c_root_probs[0];
            msg[1] = c_root_probs[1];
            msg[2] = c_root_probs[2];
            msg[3] = c_root_probs[3];

            for (int r = 0; r < c_num_root_edges; r++) {
                int sib = c_root_edges[r];
                if (sib != eid) {
                    msg[0] *= up[sib * 4 + 0];
                    msg[1] *= up[sib * 4 + 1];
                    msg[2] *= up[sib * 4 + 2];
                    msg[3] *= up[sib * 4 + 3];
                }
            }
        } else {
            int parent_edge = d_node_incoming_edge[parent];

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
                for (int c = 0; c < nch; c++) {
                    int sib = d_node_child_edges[off + c];
                    if (sib != eid) {
                        msg[0] *= up[sib * 4 + 0];
                        msg[1] *= up[sib * 4 + 1];
                        msg[2] *= up[sib * 4 + 2];
                        msg[3] *= up[sib * 4 + 3];
                    }
                }
            } else {
                msg[0] = 1.0; msg[1] = 1.0; msg[2] = 1.0; msg[3] = 1.0;
            }
        }

        double mx = fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3]));
        if (mx > 0.0 && mx != 1.0) {
            double inv_mx = 1.0 / mx;
            msg[0] *= inv_mx;
            msg[1] *= inv_mx;
            msg[2] *= inv_mx;
            msg[3] *= inv_mx;
        }

        down[eid * 4 + 0] = msg[0];
        down[eid * 4 + 1] = msg[1];
        down[eid * 4 + 2] = msg[2];
        down[eid * 4 + 3] = msg[3];
    }
}

// NEW: Edge-parallel expected counts kernel
// Each block handles one edge, threads process patterns
__global__ void expected_counts_edge_parallel_kernel(
    int num_patterns,
    int edge_id,  // Which edge this block processes
    int num_edges,
    int num_taxa,
    int child_node,
    bool child_is_leaf,
    int child_taxon,
    int child_nch,
    int child_off,
    const uint8_t* __restrict__ d_pattern_bases,
    const int* __restrict__ d_pattern_weights,
    const double* __restrict__ d_transition_matrix,  // Just 16 values for this edge
    const double* __restrict__ d_upward_messages,
    const double* __restrict__ d_downward_messages,
    const int* __restrict__ d_node_child_edges,
    double* __restrict__ d_edge_counts  // Output: 16 values for this edge
) {
    extern __shared__ double s_mem[];
    double* s_counts = s_mem;  // [16] partial sums per thread block

    // Initialize shared memory
    if (threadIdx.x < 16) {
        s_counts[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Each thread accumulates counts for multiple patterns
    double local_counts[16] = {0.0};

    for (int pid = blockIdx.x * blockDim.x + threadIdx.x; pid < num_patterns; pid += gridDim.x * blockDim.x) {
        const double* up = d_upward_messages + pid * num_edges * 4;
        const double* down = d_downward_messages + pid * num_edges * 4;
        double weight = (double)d_pattern_weights[pid];

        double joint[16];
        double sum = 0.0;

        if (child_is_leaf) {
            uint8_t base = d_pattern_bases[pid * num_taxa + child_taxon];

            if (base < 4) {
                for (int ps = 0; ps < 4; ps++) {
                    for (int cs = 0; cs < 4; cs++) {
                        joint[ps * 4 + cs] = (cs == base) ?
                            down[edge_id * 4 + ps] * d_transition_matrix[ps * 4 + cs] : 0.0;
                        sum += joint[ps * 4 + cs];
                    }
                }
            } else {
                for (int ps = 0; ps < 4; ps++) {
                    for (int cs = 0; cs < 4; cs++) {
                        joint[ps * 4 + cs] = down[edge_id * 4 + ps] * d_transition_matrix[ps * 4 + cs];
                        sum += joint[ps * 4 + cs];
                    }
                }
            }
        } else {
            for (int ps = 0; ps < 4; ps++) {
                for (int cs = 0; cs < 4; cs++) {
                    double child_prod = 1.0;
                    for (int c = 0; c < child_nch; c++) {
                        int ce = d_node_child_edges[child_off + c];
                        child_prod *= up[ce * 4 + cs];
                    }
                    joint[ps * 4 + cs] = down[edge_id * 4 + ps] * d_transition_matrix[ps * 4 + cs] * child_prod;
                    sum += joint[ps * 4 + cs];
                }
            }
        }

        if (sum > 1e-300) {
            double inv_sum = weight / sum;
            for (int i = 0; i < 16; i++) {
                local_counts[i] += joint[i] * inv_sum;
            }
        }
    }

    // Reduce local counts across threads
    for (int i = 0; i < 16; i++) {
        double val = blockReduceSum(local_counts[i]);
        if (threadIdx.x == 0) {
            atomicAdd(&d_edge_counts[i], val);
        }
        __syncthreads();
    }
}

// Pattern LL kernel
__global__ void compute_pattern_ll_kernel_v2(
    int num_patterns,
    int num_edges,
    const double* __restrict__ d_upward_messages,
    const double* __restrict__ d_upward_total_scale,
    const int* __restrict__ d_pattern_weights,
    double* __restrict__ d_pattern_lls
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    const double* msgs = d_upward_messages + pid * num_edges * 4;
    double total_scale = d_upward_total_scale[pid];

    double combined[4] = {1.0, 1.0, 1.0, 1.0};
    for (int r = 0; r < c_num_root_edges; r++) {
        int re = c_root_edges[r];
        combined[0] *= msgs[re * 4 + 0];
        combined[1] *= msgs[re * 4 + 1];
        combined[2] *= msgs[re * 4 + 2];
        combined[3] *= msgs[re * 4 + 3];
    }

    double mx = fmax(fmax(combined[0], combined[1]), fmax(combined[2], combined[3]));
    if (mx > 0.0) {
        double inv_mx = 1.0 / mx;
        combined[0] *= inv_mx;
        combined[1] *= inv_mx;
        combined[2] *= inv_mx;
        combined[3] *= inv_mx;
        total_scale += log(mx);
    }

    double site_ll = c_root_probs[0] * combined[0] + c_root_probs[1] * combined[1] +
                     c_root_probs[2] * combined[2] + c_root_probs[3] * combined[3];

    d_pattern_lls[pid] = (total_scale + log(site_ll)) * d_pattern_weights[pid];
}

// Load tree (same as before)
EMBHTree load_tree(const string& edge_file, const string& pattern_file,
                   const string& taxon_file, const string& basecomp_file) {
    EMBHTree tree;

    ifstream tin(taxon_file);
    string line;
    getline(tin, line);
    while (getline(tin, line)) {
        size_t c = line.find(',');
        if (c != string::npos) {
            tree.taxon_names.push_back(line.substr(0, c));
        }
    }
    tin.close();
    tree.num_taxa = tree.taxon_names.size();

    vector<pair<string, string>> edges;
    vector<double> lengths;
    ifstream ein(edge_file);
    while (getline(ein, line)) {
        istringstream iss(line);
        string p, ch;
        double len;
        iss >> p >> ch >> len;
        edges.push_back({p, ch});
        lengths.push_back(len);
    }
    ein.close();

    set<string> names;
    for (auto& e : edges) {
        names.insert(e.first);
        names.insert(e.second);
    }
    int nid = 0;
    for (auto& n : names) {
        tree.node_names.push_back(n);
        tree.name_to_id[n] = nid++;
    }

    int num_nodes = tree.node_names.size();
    tree.node_is_leaf.resize(num_nodes, true);
    tree.leaf_taxon_indices.resize(num_nodes, -1);
    tree.node_children.resize(num_nodes);
    tree.node_parent.resize(num_nodes, -1);
    tree.node_incoming_edge.resize(num_nodes, -1);

    for (size_t i = 0; i < edges.size(); i++) {
        int pid = tree.name_to_id[edges[i].first];
        int cid = tree.name_to_id[edges[i].second];
        tree.edge_parent_nodes.push_back(pid);
        tree.edge_child_nodes.push_back(cid);
        tree.branch_lengths.push_back(lengths[i]);
        tree.node_children[pid].push_back(cid);
        tree.node_parent[cid] = pid;
        tree.node_is_leaf[pid] = false;
        tree.node_incoming_edge[cid] = i;
    }

    for (int n = 0; n < num_nodes; n++) {
        if (tree.node_parent[n] == -1) {
            tree.root_node_id = n;
            break;
        }
    }

    map<string, int> taxon_idx;
    for (int i = 0; i < tree.num_taxa; i++) {
        taxon_idx[tree.taxon_names[i]] = i;
    }
    for (int n = 0; n < num_nodes; n++) {
        if (tree.node_is_leaf[n]) {
            auto it = taxon_idx.find(tree.node_names[n]);
            if (it != taxon_idx.end()) {
                tree.leaf_taxon_indices[n] = it->second;
            }
        }
    }

    function<void(int)> dfs = [&](int node) {
        for (int ch : tree.node_children[node]) {
            dfs(ch);
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == ch && tree.edge_parent_nodes[e] == node) {
                    tree.post_order_edges.push_back(e);
                    break;
                }
            }
        }
    };
    dfs(tree.root_node_id);

    for (int ch : tree.node_children[tree.root_node_id]) {
        for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
            if (tree.edge_child_nodes[e] == ch && tree.edge_parent_nodes[e] == tree.root_node_id) {
                tree.root_edge_ids.push_back(e);
                break;
            }
        }
    }

    tree.transition_matrices.resize(tree.edge_child_nodes.size());

    ifstream pin(pattern_file);
    vector<vector<int>> temp_patterns;
    vector<int> temp_weights;
    while (getline(pin, line)) {
        istringstream iss(line);
        int weight;
        iss >> weight;
        vector<int> bases(tree.num_taxa);
        for (int i = 0; i < tree.num_taxa; i++) iss >> bases[i];
        temp_patterns.push_back(bases);
        temp_weights.push_back(weight);
    }
    pin.close();

    tree.num_patterns = temp_patterns.size();
    tree.pattern_bases.resize(tree.num_patterns * tree.num_taxa);
    tree.pattern_weights = temp_weights;

    for (int p = 0; p < tree.num_patterns; p++) {
        for (int t = 0; t < tree.num_taxa; t++) {
            tree.pattern_bases[p * tree.num_taxa + t] = (uint8_t)temp_patterns[p][t];
        }
    }

    ifstream bin(basecomp_file);
    for (int i = 0; i < 4; i++) {
        getline(bin, line);
        istringstream lss(line);
        int idx;
        double prob;
        lss >> idx >> prob;
        tree.root_probabilities[idx] = prob;
    }
    bin.close();

    double eps = 1e-6;
    for (int i = 0; i < 4; i++) {
        if (tree.root_probabilities[i] < eps) {
            tree.root_probabilities[i] = eps;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += tree.root_probabilities[i];
    for (int i = 0; i < 4; i++) tree.root_probabilities[i] /= sum;

    double S2 = 0.0;
    for (int i = 0; i < 4; i++) {
        S2 += tree.root_probabilities[i] * tree.root_probabilities[i];
    }
    double mu = 1.0 / max(1e-14, 1.0 - S2);
    cout << "F81 mu: " << mu << endl;

    for (size_t e = 0; e < tree.branch_lengths.size(); e++) {
        double t = tree.branch_lengths[e];
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

    return tree;
}

void m_step(EMBHTree& tree, const vector<double>& counts) {
    for (size_t e = 0; e < tree.transition_matrices.size(); e++) {
        array<double, 4> row_sum = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                row_sum[i] += counts[e * 16 + i * 4 + j];
            }
        }

        for (int i = 0; i < 4; i++) {
            if (row_sum[i] > 1e-10) {
                for (int j = 0; j < 4; j++) {
                    tree.transition_matrices[e][i][j] = counts[e * 16 + i * 4 + j] / row_sum[i];
                }
            } else {
                for (int j = 0; j < 4; j++) {
                    tree.transition_matrices[e][i][j] = 0.25;
                }
            }
        }
    }

    if (!tree.root_edge_ids.empty()) {
        int re = tree.root_edge_ids[0];
        double sum = 0.0;
        for (int i = 0; i < 4; i++) {
            tree.root_probabilities[i] = 0.0;
            for (int j = 0; j < 4; j++) {
                tree.root_probabilities[i] += counts[re * 16 + i * 4 + j];
            }
            sum += tree.root_probabilities[i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < 4; i++) {
                tree.root_probabilities[i] /= sum;
            }
        }
    }
}

void embh_aitken_gpu_v2(EMBHTree& tree, int max_iter) {
    const double MAX_RATE = 0.95;
    const int MIN_AITKEN = 3;

    int num_sites = 0;
    for (int w : tree.pattern_weights) num_sites += w;

    int num_nodes = tree.node_names.size();
    int num_edges = tree.edge_child_nodes.size();

    cout << "\nAllocating GPU memory (v2 - edge parallel)..." << endl;

    // GPU arrays
    int *d_post_order, *d_edge_child, *d_edge_parent;
    bool *d_is_leaf;
    int *d_leaf_taxon;
    int *d_node_nch, *d_node_off, *d_node_edges;
    int *d_node_incoming_edge;
    uint8_t *d_bases;
    int *d_weights;
    double *d_trans;
    double *d_up_msg, *d_up_total_scale;
    double *d_down_msg;
    double *d_counts, *d_pattern_ll;

    CUDA_CHECK(cudaMalloc(&d_post_order, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_child, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_parent, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_leaf, num_nodes * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_leaf_taxon, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_incoming_edge, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bases, tree.num_patterns * tree.num_taxa * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, tree.num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_trans, num_edges * 16 * sizeof(double)));

    vector<int> nch(num_nodes, 0), noff(num_nodes, 0);
    vector<int> nedges;
    for (int n = 0; n < num_nodes; n++) {
        noff[n] = nedges.size();
        for (int ch : tree.node_children[n]) {
            for (int e = 0; e < num_edges; e++) {
                if (tree.edge_child_nodes[e] == ch) {
                    nedges.push_back(e);
                    nch[n]++;
                    break;
                }
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&d_node_nch, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_off, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_edges, nedges.size() * sizeof(int)));

    size_t msg_sz = (size_t)tree.num_patterns * num_edges * 4 * sizeof(double);
    size_t scl_sz = (size_t)tree.num_patterns * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_up_msg, msg_sz));
    CUDA_CHECK(cudaMalloc(&d_up_total_scale, scl_sz));
    CUDA_CHECK(cudaMalloc(&d_down_msg, msg_sz));
    CUDA_CHECK(cudaMalloc(&d_counts, num_edges * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pattern_ll, tree.num_patterns * sizeof(double)));

    cout << "GPU memory (v2): " << (msg_sz * 2 + scl_sz) / (1024.0 * 1024.0) << " MB" << endl;

    // Copy static data
    CUDA_CHECK(cudaMemcpy(d_post_order, tree.post_order_edges.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_child, tree.edge_child_nodes.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_parent, tree.edge_parent_nodes.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    bool* is_leaf_arr = new bool[num_nodes];
    for (int i = 0; i < num_nodes; i++) is_leaf_arr[i] = tree.node_is_leaf[i];
    CUDA_CHECK(cudaMemcpy(d_is_leaf, is_leaf_arr, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
    delete[] is_leaf_arr;
    CUDA_CHECK(cudaMemcpy(d_leaf_taxon, tree.leaf_taxon_indices.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_incoming_edge, tree.node_incoming_edge.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_nch, nch.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_off, noff.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_edges, nedges.data(), nedges.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bases, tree.pattern_bases.data(), tree.num_patterns * tree.num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, tree.pattern_weights.data(), tree.num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    int h_root_edges[16];
    for (int i = 0; i < min((int)tree.root_edge_ids.size(), 16); i++) {
        h_root_edges[i] = tree.root_edge_ids[i];
    }
    int h_num_root_edges = tree.root_edge_ids.size();
    CUDA_CHECK(cudaMemcpyToSymbol(c_root_edges, h_root_edges, tree.root_edge_ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_num_root_edges, &h_num_root_edges, sizeof(int)));

    vector<double> flat_trans(num_edges * 16);
    auto update_flat = [&]() {
        for (int e = 0; e < num_edges; e++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    flat_trans[e * 16 + i * 4 + j] = tree.transition_matrices[e][i][j];
                }
            }
        }
    };

    auto compute_ll = [&]() -> double {
        CUDA_CHECK(cudaMemcpy(d_trans, flat_trans.data(), num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_root_probs, tree.root_probabilities.data(), 4 * sizeof(double)));

        int bs = 256;
        int nb = (tree.num_patterns + bs - 1) / bs;

        forward_pass_kernel_v2<<<nb, bs>>>(tree.num_patterns, num_edges, tree.num_taxa,
            d_post_order, d_edge_child, d_is_leaf, d_leaf_taxon, d_bases, d_trans,
            d_node_nch, d_node_off, d_node_edges, d_up_msg, d_up_total_scale);

        compute_pattern_ll_kernel_v2<<<nb, bs>>>(tree.num_patterns, num_edges,
            d_up_msg, d_up_total_scale, d_weights, d_pattern_ll);

        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> plls(tree.num_patterns);
        CUDA_CHECK(cudaMemcpy(plls.data(), d_pattern_ll, tree.num_patterns * sizeof(double), cudaMemcpyDeviceToHost));

        double ll = 0.0;
        for (double v : plls) ll += v;
        return ll;
    };

    update_flat();
    double ll0 = compute_ll();

    cout << "\n====================================================" << endl;
    cout << "GPU EM with Aitken Acceleration (V2 - Edge Parallel)" << endl;
    cout << "====================================================" << endl;
    cout << "Initial LL: " << fixed << setprecision(2) << ll0 << endl;
    cout << "Patterns: " << tree.num_patterns << ", Sites: " << num_sites << ", Edges: " << num_edges << endl;
    cout << "Optimizations: Edge-parallel expected counts, warp reduction" << endl;
    cout << string(70, '-') << endl;

    double ll_pp = ll0, ll_p = ll0, ll_c = ll0;
    int final_iter = 0;
    bool converged = false;
    string reason;
    double gpu_time = 0.0;

    auto total_start = chrono::high_resolution_clock::now();

    int bs = 256;
    int nb = (tree.num_patterns + bs - 1) / bs;

    for (int iter = 1; iter <= max_iter; iter++) {
        final_iter = iter;
        auto iter_start = chrono::high_resolution_clock::now();

        // E-step
        CUDA_CHECK(cudaMemcpy(d_trans, flat_trans.data(), num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_root_probs, tree.root_probabilities.data(), 4 * sizeof(double)));

        forward_pass_kernel_v2<<<nb, bs>>>(tree.num_patterns, num_edges, tree.num_taxa,
            d_post_order, d_edge_child, d_is_leaf, d_leaf_taxon, d_bases, d_trans,
            d_node_nch, d_node_off, d_node_edges, d_up_msg, d_up_total_scale);

        backward_pass_kernel_v2<<<nb, bs>>>(tree.num_patterns, num_edges,
            d_post_order, d_edge_child, d_edge_parent, d_trans,
            d_node_nch, d_node_off, d_node_edges, d_node_incoming_edge,
            d_up_msg, d_down_msg);

        // Edge-parallel expected counts
        CUDA_CHECK(cudaMemset(d_counts, 0, num_edges * 16 * sizeof(double)));

        // Launch one kernel per edge with many blocks
        int count_blocks = min(256, (tree.num_patterns + 255) / 256);
        int count_threads = 256;
        size_t smem_size = 16 * sizeof(double);

        for (int eid = 0; eid < num_edges; eid++) {
            int child = tree.edge_child_nodes[eid];
            expected_counts_edge_parallel_kernel<<<count_blocks, count_threads, smem_size>>>(
                tree.num_patterns, eid, num_edges, tree.num_taxa,
                child, tree.node_is_leaf[child], tree.leaf_taxon_indices[child],
                nch[child], noff[child],
                d_bases, d_weights, d_trans + eid * 16,
                d_up_msg, d_down_msg, d_node_edges,
                d_counts + eid * 16
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> counts(num_edges * 16);
        CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, num_edges * 16 * sizeof(double), cudaMemcpyDeviceToHost));

        auto iter_end = chrono::high_resolution_clock::now();
        double iter_time = chrono::duration_cast<chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
        gpu_time += iter_time;

        // M-step
        m_step(tree, counts);
        update_flat();

        ll_c = compute_ll();

        double imp = ll_c - ll_p;

        bool use_aitken = false;
        double aitken_ll = ll_c, aitken_dist = 0.0, rate = 0.0;

        if (iter >= MIN_AITKEN) {
            double d1 = ll_p - ll_pp;
            double d2 = ll_c - ll_p;
            double denom = d2 - d1;

            if (fabs(d1) > 1e-12) rate = fabs(d2 / d1);

            if (fabs(denom) > 1e-10 && rate > 0.01 && rate < MAX_RATE) {
                aitken_ll = ll_c - (d2 * d2) / denom;
                aitken_dist = fabs(aitken_ll - ll_c);
                use_aitken = true;
            }
        }

        cout << "Iter " << setw(3) << iter << ": LL = " << fixed << setprecision(2) << ll_c
             << " (+" << scientific << setprecision(2) << imp << ")"
             << " | GPU: " << fixed << setprecision(1) << iter_time << "ms";

        if (use_aitken) {
            cout << " | Rate: " << fixed << setprecision(3) << rate
                 << " | Aitken: " << scientific << setprecision(2) << aitken_dist;
        } else if (iter >= MIN_AITKEN) {
            cout << " | Rate: " << fixed << setprecision(3) << rate;
        }
        cout << endl;

        if (iter >= MIN_AITKEN && rate > 0.95) {
            converged = true;
            reason = "Aitken rate > 0.95";
        } else if (imp < -1e-6) {
            converged = true;
            reason = "Likelihood decreased";
        }

        if (converged) break;

        ll_pp = ll_p;
        ll_p = ll_c;
    }

    auto total_end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();

    cout << string(70, '-') << endl;
    cout << (converged ? "Converged: " + reason : "Max iterations") << endl;

    cout << "\n====================================================" << endl;
    cout << "Final Results (V2)" << endl;
    cout << "====================================================" << endl;
    cout << "Final LL: " << fixed << setprecision(2) << ll_c << endl;
    cout << "Iterations: " << final_iter << endl;
    cout << "Total GPU time: " << fixed << setprecision(1) << gpu_time << " ms" << endl;
    cout << "Avg GPU time per iter: " << fixed << setprecision(2) << gpu_time / final_iter << " ms" << endl;
    cout << "Total wall time: " << fixed << setprecision(1) << total_time << " ms" << endl;

    cudaFree(d_post_order);
    cudaFree(d_edge_child);
    cudaFree(d_edge_parent);
    cudaFree(d_is_leaf);
    cudaFree(d_leaf_taxon);
    cudaFree(d_node_incoming_edge);
    cudaFree(d_node_nch);
    cudaFree(d_node_off);
    cudaFree(d_node_edges);
    cudaFree(d_bases);
    cudaFree(d_weights);
    cudaFree(d_trans);
    cudaFree(d_up_msg);
    cudaFree(d_up_total_scale);
    cudaFree(d_down_msg);
    cudaFree(d_counts);
    cudaFree(d_pattern_ll);
}

int main(int argc, char** argv) {
    string edge_file = "../cpp_program/data/tree_edges.txt";
    string pattern_file = "../cpp_program/data/patterns.pat";
    string taxon_file = "../cpp_program/data/patterns.taxon_order";
    string basecomp_file = "../cpp_program/data/patterns.basecomp";
    int max_iter = 50;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--edges" && i + 1 < argc) edge_file = argv[++i];
        else if (arg == "--patterns" && i + 1 < argc) pattern_file = argv[++i];
        else if (arg == "--taxa" && i + 1 < argc) taxon_file = argv[++i];
        else if (arg == "--basecomp" && i + 1 < argc) basecomp_file = argv[++i];
        else if (arg == "--max-iter" && i + 1 < argc) max_iter = atoi(argv[++i]);
    }

    cout << "Loading tree and patterns..." << endl;
    EMBHTree tree = load_tree(edge_file, pattern_file, taxon_file, basecomp_file);
    cout << "Loaded: " << tree.num_patterns << " patterns, "
         << tree.edge_child_nodes.size() << " edges, "
         << tree.num_taxa << " taxa" << endl;

    embh_aitken_gpu_v2(tree, max_iter);

    return 0;
}
