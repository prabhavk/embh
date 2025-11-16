// GPU-accelerated log-likelihood computation for EMBH
// Implementation file

#include "gpu_likelihood.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            return false; \
        } \
    } while(0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            return; \
        } \
    } while(0)

// GPU tree data structure
struct GPUTreeData {
    int num_edges;
    int num_nodes;
    int num_taxa;
    int num_root_edges;

    // Node info [num_nodes]
    int* d_node_is_leaf;
    int* d_node_taxon_index;
    int* d_node_num_children;
    int* d_node_child_offsets;
    int* d_node_children;  // Flattened child EDGE IDs

    // Edge info [num_edges]
    int* d_edge_child_node;
    double* d_trans_matrices;  // [num_edges * 16]

    // Post-order traversal [num_edges]
    int* d_post_order;

    // Root edges
    int* d_root_edges;

    // Root probabilities [4]
    double* d_root_prob;

    // Pattern data (allocated per compute call)
    uint8_t* d_pattern_bases;
    int* d_pattern_weights;
};

// Device function: compute Jukes-Cantor transition probability
__device__ double d_jc_prob_lib(int parent_state, int child_state, double t) {
    double exp_term = exp(-4.0 * t / 3.0);
    if (parent_state == child_state) {
        return 0.25 + 0.75 * exp_term;
    } else {
        return 0.25 - 0.25 * exp_term;
    }
}

// Main kernel: compute log-likelihood for each pattern
__global__ void compute_pattern_ll_kernel(
    int num_patterns,
    int num_edges,
    int num_taxa,
    int num_root_edges,
    const int* d_node_is_leaf,
    const int* d_node_taxon_index,
    const int* d_node_num_children,
    const int* d_node_child_offsets,
    const int* d_node_children,
    const int* d_edge_child_node,
    const double* d_trans_matrices,
    const int* d_post_order,
    const int* d_root_edges,
    const double* d_root_prob,
    const uint8_t* d_pattern_bases,
    double* d_edge_messages,
    double* d_edge_log_scales,
    double* d_pattern_log_likelihoods
) {
    int pattern_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pattern_id >= num_patterns) return;

    // Offset into global arrays for this pattern
    double* edge_messages = d_edge_messages + pattern_id * num_edges * 4;
    double* edge_log_scales = d_edge_log_scales + pattern_id * num_edges;

    // Initialize log scales to 0
    for (int e = 0; e < num_edges; e++) {
        edge_log_scales[e] = 0.0;
    }

    // Process edges in post-order (leaves to root)
    for (int order_idx = 0; order_idx < num_edges; order_idx++) {
        int edge_id = d_post_order[order_idx];
        int child_node = d_edge_child_node[edge_id];

        // Get transition matrix for this edge
        const double* P = d_trans_matrices + edge_id * 16;

        // Compute message
        double message[4] = {1.0, 1.0, 1.0, 1.0};

        if (d_node_is_leaf[child_node]) {
            // Leaf edge: message based on observed base
            int taxon_idx = d_node_taxon_index[child_node];
            uint8_t observed_base = d_pattern_bases[pattern_id * num_taxa + taxon_idx];

            if (observed_base < 4) {
                // Observed nucleotide
                for (int parent_state = 0; parent_state < 4; parent_state++) {
                    message[parent_state] = P[parent_state * 4 + observed_base];
                }
            }
            // else: gap, message stays at 1.0 for all states
        } else {
            // Internal edge: combine child messages
            int num_children = d_node_num_children[child_node];
            int child_offset = d_node_child_offsets[child_node];

            for (int parent_state = 0; parent_state < 4; parent_state++) {
                double sum = 0.0;
                for (int child_state = 0; child_state < 4; child_state++) {
                    double prod = P[parent_state * 4 + child_state];

                    // Multiply by messages from child edges
                    for (int c = 0; c < num_children && c < 8; c++) {
                        int child_edge = d_node_children[child_offset + c];
                        prod *= edge_messages[child_edge * 4 + child_state];
                    }
                    sum += prod;
                }
                message[parent_state] = sum;
            }
        }

        // Scale message to avoid underflow
        double max_val = message[0];
        for (int i = 1; i < 4; i++) {
            if (message[i] > max_val) max_val = message[i];
        }

        if (max_val > 0.0) {
            for (int i = 0; i < 4; i++) {
                message[i] /= max_val;
                edge_messages[edge_id * 4 + i] = message[i];
            }
            edge_log_scales[edge_id] = log(max_val);
        } else {
            for (int i = 0; i < 4; i++) {
                edge_messages[edge_id * 4 + i] = message[i];
            }
        }
    }

    // Combine root edge messages
    double combined_msg[4] = {1.0, 1.0, 1.0, 1.0};
    double total_log_scale = 0.0;

    // Accumulate all log scales
    for (int e = 0; e < num_edges; e++) {
        total_log_scale += edge_log_scales[e];
    }

    // Multiply messages from root edges
    for (int r = 0; r < num_root_edges; r++) {
        int root_edge = d_root_edges[r];
        for (int i = 0; i < 4; i++) {
            combined_msg[i] *= edge_messages[root_edge * 4 + i];
        }
    }

    // Normalize combined message
    double max_val = combined_msg[0];
    for (int i = 1; i < 4; i++) {
        if (combined_msg[i] > max_val) max_val = combined_msg[i];
    }
    if (max_val > 0.0) {
        for (int i = 0; i < 4; i++) {
            combined_msg[i] /= max_val;
        }
        total_log_scale += log(max_val);
    }

    // Compute site likelihood
    double site_likelihood = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        site_likelihood += d_root_prob[dna] * combined_msg[dna];
    }

    // Store log-likelihood (not yet weighted by pattern count)
    d_pattern_log_likelihoods[pattern_id] = total_log_scale + log(site_likelihood);
}

// GPULikelihoodComputer implementation

GPULikelihoodComputer::GPULikelihoodComputer()
    : gpu_data(nullptr), num_patterns_allocated(0),
      d_edge_messages(nullptr), d_edge_log_scales(nullptr),
      d_pattern_log_likelihoods(nullptr), tree_initialized(false),
      _num_edges(0), _num_nodes(0), _num_taxa(0) {
}

GPULikelihoodComputer::~GPULikelihoodComputer() {
    FreePatternBuffers();

    if (gpu_data != nullptr) {
        cudaFree(gpu_data->d_node_is_leaf);
        cudaFree(gpu_data->d_node_taxon_index);
        cudaFree(gpu_data->d_node_num_children);
        cudaFree(gpu_data->d_node_child_offsets);
        cudaFree(gpu_data->d_node_children);
        cudaFree(gpu_data->d_edge_child_node);
        cudaFree(gpu_data->d_trans_matrices);
        cudaFree(gpu_data->d_post_order);
        cudaFree(gpu_data->d_root_edges);
        cudaFree(gpu_data->d_root_prob);
        cudaFree(gpu_data->d_pattern_bases);
        cudaFree(gpu_data->d_pattern_weights);
        delete gpu_data;
    }
}

bool GPULikelihoodComputer::InitializeTree(
    int num_nodes,
    int num_edges,
    int num_taxa,
    const vector<int>& edge_child_nodes,
    const vector<double>& branch_lengths,
    const vector<bool>& node_is_leaf,
    const vector<int>& leaf_taxon_indices,
    const vector<vector<int>>& node_children,
    const vector<int>& post_order_edges,
    const vector<int>& root_edge_ids
) {
    if (gpu_data != nullptr) {
        // Free existing data
        cudaFree(gpu_data->d_node_is_leaf);
        cudaFree(gpu_data->d_node_taxon_index);
        cudaFree(gpu_data->d_node_num_children);
        cudaFree(gpu_data->d_node_child_offsets);
        cudaFree(gpu_data->d_node_children);
        cudaFree(gpu_data->d_edge_child_node);
        cudaFree(gpu_data->d_trans_matrices);
        cudaFree(gpu_data->d_post_order);
        cudaFree(gpu_data->d_root_edges);
        cudaFree(gpu_data->d_root_prob);
        delete gpu_data;
    }

    gpu_data = new GPUTreeData();
    gpu_data->num_nodes = num_nodes;
    gpu_data->num_edges = num_edges;
    gpu_data->num_taxa = num_taxa;
    gpu_data->num_root_edges = root_edge_ids.size();

    _num_nodes = num_nodes;
    _num_edges = num_edges;
    _num_taxa = num_taxa;

    // Prepare host data
    vector<int> h_node_is_leaf(num_nodes);
    vector<int> h_node_taxon_index(num_nodes);
    vector<int> h_node_num_children(num_nodes);
    vector<int> h_node_child_offsets(num_nodes);
    vector<int> h_node_children_flat;

    int offset = 0;
    for (int n = 0; n < num_nodes; n++) {
        h_node_is_leaf[n] = node_is_leaf[n] ? 1 : 0;
        h_node_taxon_index[n] = leaf_taxon_indices[n];
        h_node_num_children[n] = node_children[n].size();
        h_node_child_offsets[n] = offset;

        // Store EDGE IDs for children
        for (int child_node_id : node_children[n]) {
            // Find edge that has this child_node_id as its child
            int child_edge_id = -1;
            for (int e = 0; e < num_edges; e++) {
                if (edge_child_nodes[e] == child_node_id) {
                    child_edge_id = e;
                    break;
                }
            }
            h_node_children_flat.push_back(child_edge_id);
        }
        offset += node_children[n].size();
    }

    // Compute transition matrices
    vector<double> h_trans_matrices(num_edges * 16);
    for (int e = 0; e < num_edges; e++) {
        double t = branch_lengths[e];
        double exp_term = exp(-4.0 * t / 3.0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    h_trans_matrices[e * 16 + i * 4 + j] = 0.25 + 0.75 * exp_term;
                } else {
                    h_trans_matrices[e * 16 + i * 4 + j] = 0.25 - 0.25 * exp_term;
                }
            }
        }
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&gpu_data->d_node_is_leaf, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_node_taxon_index, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_node_num_children, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_node_child_offsets, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_node_children, h_node_children_flat.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_edge_child_node, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_trans_matrices, num_edges * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_post_order, post_order_edges.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_root_edges, root_edge_ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_data->d_root_prob, 4 * sizeof(double)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(gpu_data->d_node_is_leaf, h_node_is_leaf.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_node_taxon_index, h_node_taxon_index.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_node_num_children, h_node_num_children.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_node_child_offsets, h_node_child_offsets.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_node_children, h_node_children_flat.data(),
                          h_node_children_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_edge_child_node, edge_child_nodes.data(),
                          num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_trans_matrices, h_trans_matrices.data(),
                          num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_post_order, post_order_edges.data(),
                          post_order_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_data->d_root_edges, root_edge_ids.data(),
                          root_edge_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Default root probabilities (uniform)
    double root_prob[4] = {0.25, 0.25, 0.25, 0.25};
    CUDA_CHECK(cudaMemcpy(gpu_data->d_root_prob, root_prob, 4 * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize pattern buffers to nullptr
    gpu_data->d_pattern_bases = nullptr;
    gpu_data->d_pattern_weights = nullptr;

    tree_initialized = true;
    return true;
}

void GPULikelihoodComputer::SetBranchLengths(const vector<double>& branch_lengths) {
    if (!tree_initialized || gpu_data == nullptr) {
        cerr << "Error: Tree not initialized" << endl;
        return;
    }

    // Recompute transition matrices
    vector<double> h_trans_matrices(_num_edges * 16);
    for (int e = 0; e < _num_edges; e++) {
        double t = branch_lengths[e];
        double exp_term = exp(-4.0 * t / 3.0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    h_trans_matrices[e * 16 + i * 4 + j] = 0.25 + 0.75 * exp_term;
                } else {
                    h_trans_matrices[e * 16 + i * 4 + j] = 0.25 - 0.25 * exp_term;
                }
            }
        }
    }

    CUDA_CHECK_VOID(cudaMemcpy(gpu_data->d_trans_matrices, h_trans_matrices.data(),
                               _num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
}

void GPULikelihoodComputer::SetRootProbabilities(const array<double, 4>& root_prob) {
    if (!tree_initialized || gpu_data == nullptr) {
        cerr << "Error: Tree not initialized" << endl;
        return;
    }

    CUDA_CHECK_VOID(cudaMemcpy(gpu_data->d_root_prob, root_prob.data(),
                               4 * sizeof(double), cudaMemcpyHostToDevice));
}

void GPULikelihoodComputer::AllocatePatternBuffers(int num_patterns) {
    if (num_patterns_allocated >= num_patterns) return;

    FreePatternBuffers();

    size_t edge_msg_size = (size_t)num_patterns * _num_edges * 4 * sizeof(double);
    size_t edge_scale_size = (size_t)num_patterns * _num_edges * sizeof(double);

    cudaMalloc(&d_edge_messages, edge_msg_size);
    cudaMalloc(&d_edge_log_scales, edge_scale_size);
    cudaMalloc(&d_pattern_log_likelihoods, num_patterns * sizeof(double));

    if (gpu_data != nullptr) {
        cudaMalloc(&gpu_data->d_pattern_bases, num_patterns * _num_taxa * sizeof(uint8_t));
        cudaMalloc(&gpu_data->d_pattern_weights, num_patterns * sizeof(int));
    }

    num_patterns_allocated = num_patterns;
}

void GPULikelihoodComputer::FreePatternBuffers() {
    if (d_edge_messages != nullptr) cudaFree(d_edge_messages);
    if (d_edge_log_scales != nullptr) cudaFree(d_edge_log_scales);
    if (d_pattern_log_likelihoods != nullptr) cudaFree(d_pattern_log_likelihoods);

    d_edge_messages = nullptr;
    d_edge_log_scales = nullptr;
    d_pattern_log_likelihoods = nullptr;

    if (gpu_data != nullptr) {
        if (gpu_data->d_pattern_bases != nullptr) cudaFree(gpu_data->d_pattern_bases);
        if (gpu_data->d_pattern_weights != nullptr) cudaFree(gpu_data->d_pattern_weights);
        gpu_data->d_pattern_bases = nullptr;
        gpu_data->d_pattern_weights = nullptr;
    }

    num_patterns_allocated = 0;
}

double GPULikelihoodComputer::ComputeLogLikelihood(
    const vector<vector<uint8_t>>& patterns,
    const vector<int>& pattern_weights
) {
    if (!tree_initialized || gpu_data == nullptr) {
        cerr << "Error: Tree not initialized" << endl;
        return 0.0;
    }

    int num_patterns = patterns.size();
    if (num_patterns == 0) return 0.0;

    // Flatten patterns
    vector<uint8_t> flat_patterns(num_patterns * _num_taxa);
    for (int p = 0; p < num_patterns; p++) {
        for (int t = 0; t < _num_taxa; t++) {
            flat_patterns[p * _num_taxa + t] = patterns[p][t];
        }
    }

    return ComputeLogLikelihoodFlat(num_patterns, flat_patterns.data(), pattern_weights.data());
}

double GPULikelihoodComputer::ComputeLogLikelihoodFlat(
    int num_patterns,
    const uint8_t* pattern_bases,
    const int* pattern_weights
) {
    if (!tree_initialized || gpu_data == nullptr) {
        cerr << "Error: Tree not initialized" << endl;
        return 0.0;
    }

    // Allocate buffers if needed
    AllocatePatternBuffers(num_patterns);

    // Copy pattern data to GPU
    cudaMemcpy(gpu_data->d_pattern_bases, pattern_bases,
               num_patterns * _num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data->d_pattern_weights, pattern_weights,
               num_patterns * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_patterns + block_size - 1) / block_size;

    compute_pattern_ll_kernel<<<num_blocks, block_size>>>(
        num_patterns,
        _num_edges,
        _num_taxa,
        gpu_data->num_root_edges,
        gpu_data->d_node_is_leaf,
        gpu_data->d_node_taxon_index,
        gpu_data->d_node_num_children,
        gpu_data->d_node_child_offsets,
        gpu_data->d_node_children,
        gpu_data->d_edge_child_node,
        gpu_data->d_trans_matrices,
        gpu_data->d_post_order,
        gpu_data->d_root_edges,
        gpu_data->d_root_prob,
        gpu_data->d_pattern_bases,
        d_edge_messages,
        d_edge_log_scales,
        d_pattern_log_likelihoods
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
        return 0.0;
    }
    cudaDeviceSynchronize();

    // Copy results back and sum
    vector<double> pattern_lls(num_patterns);
    cudaMemcpy(pattern_lls.data(), d_pattern_log_likelihoods,
               num_patterns * sizeof(double), cudaMemcpyDeviceToHost);

    double total_ll = 0.0;
    for (int p = 0; p < num_patterns; p++) {
        total_ll += pattern_lls[p] * pattern_weights[p];
    }

    return total_ll;
}

size_t GPULikelihoodComputer::GetGPUMemoryUsage() const {
    if (!tree_initialized) return 0;

    size_t total = 0;
    total += _num_nodes * sizeof(int) * 4;  // node info
    total += _num_edges * sizeof(int);      // edge child nodes
    total += _num_edges * 16 * sizeof(double);  // transition matrices
    total += _num_edges * sizeof(int);      // post order
    total += gpu_data->num_root_edges * sizeof(int);  // root edges
    total += 4 * sizeof(double);  // root prob

    if (num_patterns_allocated > 0) {
        total += (size_t)num_patterns_allocated * _num_edges * 4 * sizeof(double);  // messages
        total += (size_t)num_patterns_allocated * _num_edges * sizeof(double);  // scales
        total += num_patterns_allocated * sizeof(double);  // pattern LLs
        total += num_patterns_allocated * _num_taxa * sizeof(uint8_t);  // pattern bases
        total += num_patterns_allocated * sizeof(int);  // pattern weights
    }

    return total;
}

bool GPULikelihoodComputer::IsGPUAvailable() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}
