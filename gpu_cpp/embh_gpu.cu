#include "embh_gpu.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// GPU Kernels for Message Passing with Signature-Based Reuse
// ============================================================================

// Kernel 1: Compute upward message for each unique signature group
// Each thread processes one (edge, group) pair
// This is the key optimization: only 7086 unique computations instead of 47304
__global__ void compute_upward_messages_for_groups(
    int num_edges,
    const int* edge_num_groups,
    const int* edge_group_offsets,
    const int* edge_subtree_sizes,
    const uint8_t* group_signatures,
    int max_subtree_size,
    const int* edge_leaf_indices,
    const double* trans_matrices,  // [num_edges * 16]
    const int* post_order,          // [num_edges] edges in post-order
    const int* child_edges,         // [num_edges * 2]
    double* group_messages,         // [total_groups * 4]
    double* group_log_scales        // [total_groups]
) {
    // Each block processes one edge in post-order
    int edge_order_idx = blockIdx.x;
    if (edge_order_idx >= num_edges) return;

    int edge_id = post_order[edge_order_idx];
    int num_groups_for_edge = edge_num_groups[edge_id];
    int group_offset = edge_group_offsets[edge_id];
    int subtree_size = edge_subtree_sizes[edge_id];

    // Each thread in block processes one group for this edge
    int local_group_id = threadIdx.x;
    if (local_group_id >= num_groups_for_edge) return;

    int global_group_id = group_offset + local_group_id;

    // Get signature for this group
    const uint8_t* sig = &group_signatures[global_group_id * max_subtree_size];

    // Get transition matrix for this edge (P(child|parent))
    const double* trans = &trans_matrices[edge_id * 16];

    // Compute message based on subtree structure
    double message[4] = {1.0, 1.0, 1.0, 1.0};

    // Check if this is a leaf edge (subtree_size == 1)
    if (subtree_size == 1) {
        // Leaf edge: message is indicator for observed base
        uint8_t base = sig[0];
        if (base < 4) {
            // Observed base: message[i] = trans[i][base]
            for (int i = 0; i < 4; i++) {
                message[i] = trans[i * 4 + base];
            }
        } else {
            // Gap: message[i] = sum_j trans[i][j] = 1.0 (already set)
        }
    } else {
        // Internal edge: combine messages from children
        // For now, we need child messages which are computed in previous levels
        // This requires synchronization across edges - handled by processing in post-order

        // Get child edges
        int child1 = child_edges[edge_id * 2];
        int child2 = child_edges[edge_id * 2 + 1];

        // Find which groups in children correspond to this signature
        // This is the tricky part - we need to match sub-signatures
        // For efficiency, we'll compute this during preprocessing

        // Simplified for initial implementation: compute from scratch
        // TODO: Use precomputed child group mappings for full optimization

        // For leaf edges, the signature directly gives the message
        // For internal edges, we marginalize over child
        for (int parent_state = 0; parent_state < 4; parent_state++) {
            message[parent_state] = 0.0;
            for (int child_state = 0; child_state < 4; child_state++) {
                message[parent_state] += trans[parent_state * 4 + child_state];
            }
        }
    }

    // Scale message to avoid underflow
    double max_val = 0.0;
    for (int i = 0; i < 4; i++) {
        if (message[i] > max_val) max_val = message[i];
    }

    double log_scale = 0.0;
    if (max_val > 0.0) {
        for (int i = 0; i < 4; i++) {
            message[i] /= max_val;
        }
        log_scale = log(max_val);
    }

    // Store results
    for (int i = 0; i < 4; i++) {
        group_messages[global_group_id * 4 + i] = message[i];
    }
    group_log_scales[global_group_id] = log_scale;
}

// Kernel 2: Broadcast computed messages to all patterns
__global__ void broadcast_messages_to_patterns(
    int num_patterns,
    int num_edges,
    const int* pattern_to_group,    // [num_patterns * num_edges]
    const int* edge_group_offsets,  // [num_edges + 1]
    const double* group_messages,   // [total_groups * 4]
    const double* group_log_scales, // [total_groups]
    double* pattern_log_scales      // [num_patterns]
) {
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    double total_log_scale = 0.0;

    // Accumulate log scales from all edges for this pattern
    for (int e = 0; e < num_edges; e++) {
        int group_id = pattern_to_group[pat_idx * num_edges + e];
        int global_group_id = edge_group_offsets[e] + group_id;
        total_log_scale += group_log_scales[global_group_id];
    }

    pattern_log_scales[pat_idx] = total_log_scale;
}

// Kernel 3: Compute final log-likelihood per pattern
__global__ void compute_pattern_log_likelihoods(
    int num_patterns,
    const double* pattern_log_scales,
    const double* root_messages,  // [num_patterns * 4] - computed from root edge
    const double* root_prob,      // [4]
    double* pattern_log_likelihoods
) {
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    // Marginalize over root state
    double site_likelihood = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        site_likelihood += root_prob[dna] * root_messages[pat_idx * 4 + dna];
    }

    // Combine with accumulated log scaling factors
    double log_likelihood = pattern_log_scales[pat_idx] + log(site_likelihood);
    pattern_log_likelihoods[pat_idx] = log_likelihood;
}

// Kernel 4: Weighted sum reduction
__global__ void weighted_sum_reduction(
    int num_patterns,
    const double* pattern_log_likelihoods,
    const int* pattern_weights,
    double* partial_sums,
    int num_blocks
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute weighted contribution
    double val = 0.0;
    if (i < num_patterns) {
        val = pattern_log_likelihoods[i] * pattern_weights[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Host-side loader and memory management
// ============================================================================

GPUPatternGroups GPUPatternGroupsLoader::load_from_files(
    const std::string& prefix,
    const std::string& tree_edges_file,
    const std::string& pattern_file
) {
    GPUPatternGroups groups;

    // Read group index file
    std::string index_file = prefix + ".group_index";
    std::ifstream idx_in(index_file);
    if (!idx_in.is_open()) {
        std::cerr << "Error: Cannot open " << index_file << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(idx_in, line)) {
        if (line.find("NUM_PATTERNS") != std::string::npos) {
            sscanf(line.c_str(), "NUM_PATTERNS %d", &groups.num_patterns);
        } else if (line.find("NUM_EDGES") != std::string::npos) {
            sscanf(line.c_str(), "NUM_EDGES %d", &groups.num_edges);
        } else if (line.find("TOTAL_GROUPS") != std::string::npos) {
            sscanf(line.c_str(), "TOTAL_GROUPS %d", &groups.total_groups);
        }
    }
    idx_in.close();

    std::cout << "Loading " << groups.num_patterns << " patterns, "
              << groups.num_edges << " edges, " << groups.total_groups << " groups" << std::endl;

    // Allocate host arrays
    std::vector<int> h_edge_num_groups(groups.num_edges);
    std::vector<int> h_edge_group_offsets(groups.num_edges + 1);
    std::vector<int> h_edge_subtree_sizes(groups.num_edges);

    // Re-read index file for edge info
    idx_in.open(index_file);
    int cumulative_offset = 0;
    while (std::getline(idx_in, line)) {
        if (line.substr(0, 4) == "EDGE") {
            int edge_id, subtree_size, num_groups;
            char parent_name[64], child_name[64];
            sscanf(line.c_str(), "EDGE %d %s %s %d %d",
                   &edge_id, parent_name, child_name, &subtree_size, &num_groups);
            h_edge_num_groups[edge_id] = num_groups;
            h_edge_subtree_sizes[edge_id] = subtree_size;
            h_edge_group_offsets[edge_id] = cumulative_offset;
            cumulative_offset += num_groups;
        }
    }
    h_edge_group_offsets[groups.num_edges] = cumulative_offset;
    idx_in.close();

    // Find max subtree size
    groups.max_subtree_size = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        if (h_edge_subtree_sizes[e] > groups.max_subtree_size) {
            groups.max_subtree_size = h_edge_subtree_sizes[e];
        }
    }

    // Read pattern-to-group mapping
    std::string map_file = prefix + ".group_map";
    std::ifstream map_in(map_file);
    std::vector<int> h_pattern_to_group(groups.num_patterns * groups.num_edges);

    std::getline(map_in, line); // skip header
    for (int p = 0; p < groups.num_patterns; p++) {
        std::getline(map_in, line);
        std::istringstream iss(line);
        int pat_idx;
        iss >> pat_idx;
        for (int e = 0; e < groups.num_edges; e++) {
            int group_id;
            iss >> group_id;
            h_pattern_to_group[p * groups.num_edges + e] = group_id;
        }
    }
    map_in.close();

    // Read group signatures (binary)
    std::string sig_file = prefix + ".group_signatures";
    std::ifstream sig_in(sig_file, std::ios::binary);
    std::vector<uint8_t> h_group_signatures(groups.total_groups * groups.max_subtree_size, 4); // default to gap

    for (int e = 0; e < groups.num_edges; e++) {
        int num_groups = h_edge_num_groups[e];
        int subtree_size = h_edge_subtree_sizes[e];
        int offset = h_edge_group_offsets[e];

        for (int g = 0; g < num_groups; g++) {
            int global_idx = offset + g;
            sig_in.read(reinterpret_cast<char*>(&h_group_signatures[global_idx * groups.max_subtree_size]),
                       subtree_size);
        }
    }
    sig_in.close();

    // Read pattern weights from pattern file
    std::vector<int> h_pattern_weights(groups.num_patterns);
    std::ifstream pat_in(pattern_file);
    std::getline(pat_in, line); // skip header
    for (int p = 0; p < groups.num_patterns; p++) {
        std::getline(pat_in, line);
        int weight;
        sscanf(line.c_str(), "%d", &weight);
        h_pattern_weights[p] = weight;
    }
    pat_in.close();

    // Get num_taxa from pattern file header
    pat_in.open(pattern_file);
    std::getline(pat_in, line);
    int file_patterns, file_taxa;
    sscanf(line.c_str(), "%d %d", &file_patterns, &file_taxa);
    groups.num_taxa = file_taxa;
    pat_in.close();

    // Read pattern bases
    std::vector<uint8_t> h_pattern_bases(groups.num_patterns * groups.num_taxa);
    pat_in.open(pattern_file);
    std::getline(pat_in, line); // skip header
    for (int p = 0; p < groups.num_patterns; p++) {
        std::getline(pat_in, line);
        std::istringstream iss(line);
        int weight;
        iss >> weight;
        for (int t = 0; t < groups.num_taxa; t++) {
            int base;
            iss >> base;
            h_pattern_bases[p * groups.num_taxa + t] = (uint8_t)base;
        }
    }
    pat_in.close();

    // Compute group weights
    std::vector<int> h_group_weights(groups.total_groups, 0);
    for (int p = 0; p < groups.num_patterns; p++) {
        int weight = h_pattern_weights[p];
        for (int e = 0; e < groups.num_edges; e++) {
            int group_id = h_pattern_to_group[p * groups.num_edges + e];
            int global_idx = h_edge_group_offsets[e] + group_id;
            h_group_weights[global_idx] += weight;
        }
    }

    // Build tree structure (post-order, parent/child relationships)
    std::vector<int> h_parent_edge(groups.num_edges, -1);
    std::vector<int> h_child_edges(groups.num_edges * 2, -1);
    std::vector<int> h_post_order(groups.num_edges);
    std::vector<int> h_edge_leaf_indices(groups.num_edges * groups.max_subtree_size, -1);

    // Read tree edges to build structure
    std::ifstream tree_in(tree_edges_file);
    std::map<std::string, int> name_to_edge;
    std::vector<std::string> edge_child_names(groups.num_edges);

    int edge_idx = 0;
    while (std::getline(tree_in, line) && edge_idx < groups.num_edges) {
        std::istringstream iss(line);
        std::string parent_name, child_name;
        double weight;
        iss >> parent_name >> child_name >> weight;
        edge_child_names[edge_idx] = child_name;
        name_to_edge[child_name] = edge_idx;
        edge_idx++;
    }
    tree_in.close();

    // Build parent-child relationships based on naming
    // (This is simplified - ideally we'd have explicit tree structure)
    // For now, use post-order from file
    std::string schedule_file = prefix + ".group_schedule";
    std::ifstream sched_in(schedule_file);
    std::set<int> seen_edges;
    int order_idx = 0;

    while (std::getline(sched_in, line)) {
        if (line.substr(0, 4) == "WORK") {
            int edge_id, group_id, num_pats, total_weight;
            sscanf(line.c_str(), "WORK %d %d %d %d", &edge_id, &group_id, &num_pats, &total_weight);
            if (seen_edges.find(edge_id) == seen_edges.end()) {
                h_post_order[order_idx++] = edge_id;
                seen_edges.insert(edge_id);
            }
        }
    }
    sched_in.close();

    // Allocate GPU memory and copy data
    CUDA_CHECK(cudaMalloc(&groups.d_edge_num_groups, groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_edge_group_offsets, (groups.num_edges + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_edge_subtree_sizes, groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_group_signatures, groups.total_groups * groups.max_subtree_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&groups.d_pattern_to_group, groups.num_patterns * groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_group_weights, groups.total_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_pattern_weights, groups.num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_pattern_bases, groups.num_patterns * groups.num_taxa * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&groups.d_parent_edge, groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_child_edges, groups.num_edges * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_edge_depth, groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_post_order, groups.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&groups.d_edge_leaf_indices, groups.num_edges * groups.max_subtree_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(groups.d_edge_num_groups, h_edge_num_groups.data(), groups.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_edge_group_offsets, h_edge_group_offsets.data(), (groups.num_edges + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_edge_subtree_sizes, h_edge_subtree_sizes.data(), groups.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_group_signatures, h_group_signatures.data(), groups.total_groups * groups.max_subtree_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_pattern_to_group, h_pattern_to_group.data(), groups.num_patterns * groups.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_group_weights, h_group_weights.data(), groups.total_groups * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_pattern_weights, h_pattern_weights.data(), groups.num_patterns * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_pattern_bases, h_pattern_bases.data(), groups.num_patterns * groups.num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_parent_edge, h_parent_edge.data(), groups.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_child_edges, h_child_edges.data(), groups.num_edges * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_post_order, h_post_order.data(), groups.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(groups.d_edge_leaf_indices, h_edge_leaf_indices.data(), groups.num_edges * groups.max_subtree_size * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "GPU memory allocated for pattern groups" << std::endl;
    return groups;
}

void GPUPatternGroupsLoader::free_gpu_memory(GPUPatternGroups& groups) {
    cudaFree(groups.d_edge_num_groups);
    cudaFree(groups.d_edge_group_offsets);
    cudaFree(groups.d_edge_subtree_sizes);
    cudaFree(groups.d_group_signatures);
    cudaFree(groups.d_pattern_to_group);
    cudaFree(groups.d_group_weights);
    cudaFree(groups.d_pattern_weights);
    cudaFree(groups.d_pattern_bases);
    cudaFree(groups.d_parent_edge);
    cudaFree(groups.d_child_edges);
    cudaFree(groups.d_edge_depth);
    cudaFree(groups.d_post_order);
    cudaFree(groups.d_edge_leaf_indices);
}

void GPUPatternGroupsLoader::free_gpu_memory(GPUTransitionMatrices& matrices) {
    cudaFree(matrices.d_trans_matrices);
    cudaFree(matrices.d_root_prob);
}

void GPUPatternGroupsLoader::free_gpu_memory(GPUMessageBuffers& buffers) {
    cudaFree(buffers.d_group_messages);
    cudaFree(buffers.d_group_log_scales);
    cudaFree(buffers.d_pattern_beliefs);
    cudaFree(buffers.d_pattern_log_scales);
    cudaFree(buffers.d_pattern_log_likelihoods);
}

GPUMessageBuffers GPUPatternGroupsLoader::allocate_message_buffers(const GPUPatternGroups& groups) {
    GPUMessageBuffers buffers;

    CUDA_CHECK(cudaMalloc(&buffers.d_group_messages, groups.total_groups * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buffers.d_group_log_scales, groups.total_groups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buffers.d_pattern_beliefs, groups.num_patterns * groups.num_edges * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buffers.d_pattern_log_scales, groups.num_patterns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buffers.d_pattern_log_likelihoods, groups.num_patterns * sizeof(double)));

    return buffers;
}

GPUTransitionMatrices GPUPatternGroupsLoader::load_transition_matrices(
    const std::string& tree_edges_file,
    const double* root_prob
) {
    GPUTransitionMatrices matrices;

    // Count edges
    std::ifstream tree_in(tree_edges_file);
    std::string line;
    int num_edges = 0;
    while (std::getline(tree_in, line)) {
        if (!line.empty()) num_edges++;
    }
    tree_in.close();

    // Read edge weights and compute transition matrices
    std::vector<double> h_trans_matrices(num_edges * 16);

    tree_in.open(tree_edges_file);
    int edge_idx = 0;
    while (std::getline(tree_in, line)) {
        std::istringstream iss(line);
        std::string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;

        // Compute Jukes-Cantor transition matrix: P(child|parent)
        // P_ij = 0.25 + 0.75*exp(-4*t/3) if i==j
        // P_ij = 0.25 - 0.25*exp(-4*t/3) if i!=j
        double t = branch_length;
        double exp_term = exp(-4.0 * t / 3.0);
        double p_same = 0.25 + 0.75 * exp_term;
        double p_diff = 0.25 - 0.25 * exp_term;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                h_trans_matrices[edge_idx * 16 + i * 4 + j] = (i == j) ? p_same : p_diff;
            }
        }
        edge_idx++;
    }
    tree_in.close();

    // Allocate and copy to GPU
    CUDA_CHECK(cudaMalloc(&matrices.d_trans_matrices, num_edges * 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&matrices.d_root_prob, 4 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(matrices.d_trans_matrices, h_trans_matrices.data(), num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrices.d_root_prob, root_prob, 4 * sizeof(double), cudaMemcpyHostToDevice));

    return matrices;
}

// Host function to compute log-likelihood
double compute_log_likelihood_gpu(
    const GPUPatternGroups& groups,
    const GPUTransitionMatrices& matrices,
    GPUMessageBuffers& buffers
) {
    // Kernel 1: Compute upward messages for each group
    // One block per edge, threads per group within edge
    int max_groups_per_edge = 1024;  // Adjust based on actual data
    dim3 grid1(groups.num_edges);
    dim3 block1(max_groups_per_edge);

    compute_upward_messages_for_groups<<<grid1, block1>>>(
        groups.num_edges,
        groups.d_edge_num_groups,
        groups.d_edge_group_offsets,
        groups.d_edge_subtree_sizes,
        groups.d_group_signatures,
        groups.max_subtree_size,
        groups.d_edge_leaf_indices,
        matrices.d_trans_matrices,
        groups.d_post_order,
        groups.d_child_edges,
        buffers.d_group_messages,
        buffers.d_group_log_scales
    );
    CUDA_CHECK(cudaGetLastError());

    // Kernel 2: Broadcast to all patterns
    int threads_per_block = 256;
    int num_blocks = (groups.num_patterns + threads_per_block - 1) / threads_per_block;

    broadcast_messages_to_patterns<<<num_blocks, threads_per_block>>>(
        groups.num_patterns,
        groups.num_edges,
        groups.d_pattern_to_group,
        groups.d_edge_group_offsets,
        buffers.d_group_messages,
        buffers.d_group_log_scales,
        buffers.d_pattern_log_scales
    );
    CUDA_CHECK(cudaGetLastError());

    // Kernel 3: Compute per-pattern log-likelihoods
    // For now, use a simplified version that computes root beliefs
    // TODO: Implement full tree propagation with group-level optimization

    // Kernel 4: Weighted sum reduction
    double* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(double)));

    weighted_sum_reduction<<<num_blocks, threads_per_block, threads_per_block * sizeof(double)>>>(
        groups.num_patterns,
        buffers.d_pattern_log_likelihoods,
        groups.d_pattern_weights,
        d_partial_sums,
        num_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back and finish reduction on CPU
    std::vector<double> h_partial_sums(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double total_log_likelihood = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_log_likelihood += h_partial_sums[i];
    }

    cudaFree(d_partial_sums);
    return total_log_likelihood;
}
