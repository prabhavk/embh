// GPU-accelerated log-likelihood computation for EMBH
// Header file for integration with C++ codebase

#ifndef GPU_LIKELIHOOD_CUH
#define GPU_LIKELIHOOD_CUH

#include <vector>
#include <array>
#include <cstdint>

// Forward declaration of GPU tree structure
struct GPUTreeData;

// GPU Likelihood Computer class
// Manages GPU memory and provides interface for likelihood computation
class GPULikelihoodComputer {
public:
    GPULikelihoodComputer();
    ~GPULikelihoodComputer();

    // Initialize GPU tree structure from edge-based representation
    // edge_parent_nodes[i] and edge_child_nodes[i] define edge i
    // branch_lengths[i] is the branch length for edge i
    // node_is_leaf[n] indicates if node n is a leaf
    // leaf_taxon_indices[n] is the taxon index for leaf node n (-1 for internal)
    // node_children[n] is list of child node IDs for node n
    // post_order_edges is the post-order traversal of edges
    // root_edge_ids are edges from root's children
    bool InitializeTree(
        int num_nodes,
        int num_edges,
        int num_taxa,
        const std::vector<int>& edge_child_nodes,
        const std::vector<double>& branch_lengths,
        const std::vector<bool>& node_is_leaf,
        const std::vector<int>& leaf_taxon_indices,
        const std::vector<std::vector<int>>& node_children,
        const std::vector<int>& post_order_edges,
        const std::vector<int>& root_edge_ids
    );

    // Set substitution model (currently Jukes-Cantor)
    // Recomputes transition matrices for all edges
    void SetBranchLengths(const std::vector<double>& branch_lengths);

    // Set root state probabilities
    void SetRootProbabilities(const std::array<double, 4>& root_prob);

    // Compute log-likelihood for given patterns
    // patterns: [num_patterns x num_taxa] matrix of DNA bases (0=A, 1=C, 2=G, 3=T, 4=gap)
    // pattern_weights: weight for each pattern
    // Returns total log-likelihood
    double ComputeLogLikelihood(
        const std::vector<std::vector<uint8_t>>& patterns,
        const std::vector<int>& pattern_weights
    );

    // Alternative: compute with flat pattern array (for packed storage)
    // pattern_bases: [num_patterns * num_taxa] flat array
    double ComputeLogLikelihoodFlat(
        int num_patterns,
        const uint8_t* pattern_bases,
        const int* pattern_weights
    );

    // Get GPU memory usage in bytes
    size_t GetGPUMemoryUsage() const;

    // Check if GPU is available
    static bool IsGPUAvailable();

private:
    GPUTreeData* gpu_data;
    int num_patterns_allocated;
    double* d_edge_messages;
    double* d_edge_log_scales;
    double* d_pattern_log_likelihoods;

    bool tree_initialized;
    int _num_edges;
    int _num_nodes;
    int _num_taxa;

    void AllocatePatternBuffers(int num_patterns);
    void FreePatternBuffers();
};

#endif // GPU_LIKELIHOOD_CUH
