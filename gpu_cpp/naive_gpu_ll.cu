// Naive GPU acceleration for pruning-based log-likelihood
// Each CUDA thread processes one pattern independently
// This is simpler than grouped approach and avoids log-scale accumulation issues

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>

using namespace std;

// Tree node structure (host-side)
struct TreeNode {
    string name;
    int id;
    int parent_id;
    vector<int> child_ids;
    int edge_id;
    bool is_leaf;
    int taxon_index;
};

// Tree edge structure (host-side)
struct TreeEdge {
    int id;
    int parent_node;
    int child_node;
    string parent_name;
    string child_name;
    double branch_length;
};

// GPU-friendly tree structure (flattened)
struct GPUTree {
    int num_edges;
    int num_nodes;
    int num_taxa;
    int num_patterns;
    int max_children;  // Maximum number of children per node

    // Node info [num_nodes]
    int* d_node_is_leaf;          // 1 if leaf, 0 otherwise
    int* d_node_taxon_index;      // Taxon index for leaves (-1 for internal)
    int* d_node_num_children;     // Number of children
    int* d_node_child_offsets;    // Offset into d_node_children array
    int* d_node_children;         // Flattened children list

    // Edge info [num_edges]
    int* d_edge_child_node;       // Child node for each edge
    double* d_trans_matrices;     // 4x4 transition matrices [num_edges * 16]

    // Post-order traversal [num_edges]
    int* d_post_order;

    // Pattern data [num_patterns * num_taxa]
    uint8_t* d_pattern_bases;

    // Pattern weights [num_patterns]
    int* d_pattern_weights;

    // Root edge indices [num_root_edges]
    int* d_root_edges;
    int num_root_edges;

    // Root probabilities [4]
    double* d_root_prob;
};

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Device function: compute Jukes-Cantor transition probability
__device__ double d_jc_prob(int parent_state, int child_state, double t) {
    double exp_term = exp(-4.0 * t / 3.0);
    if (parent_state == child_state) {
        return 0.25 + 0.75 * exp_term;
    } else {
        return 0.25 - 0.25 * exp_term;
    }
}

// Main kernel: compute log-likelihood for each pattern
// One thread per pattern
__global__ void compute_pattern_log_likelihoods(
    const GPUTree tree,
    double* d_pattern_log_likelihoods,
    double* d_edge_messages,      // [num_patterns * num_edges * 4]
    double* d_edge_log_scales     // [num_patterns * num_edges]
) {
    int pattern_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pattern_id >= tree.num_patterns) return;

    // Each thread uses global memory for edge messages and log scales
    // Offset into global arrays for this pattern
    double* edge_messages = d_edge_messages + pattern_id * tree.num_edges * 4;
    double* edge_log_scales = d_edge_log_scales + pattern_id * tree.num_edges;

    // Initialize log scales to 0
    for (int e = 0; e < tree.num_edges; e++) {
        edge_log_scales[e] = 0.0;
    }

    // Process edges in post-order (leaves to root)
    for (int order_idx = 0; order_idx < tree.num_edges; order_idx++) {
        int edge_id = tree.d_post_order[order_idx];
        int child_node = tree.d_edge_child_node[edge_id];

        // Get transition matrix for this edge
        const double* P = tree.d_trans_matrices + edge_id * 16;

        // Compute message
        double message[4] = {1.0, 1.0, 1.0, 1.0};

        if (tree.d_node_is_leaf[child_node]) {
            // Leaf edge: message based on observed base
            int taxon_idx = tree.d_node_taxon_index[child_node];
            uint8_t observed_base = tree.d_pattern_bases[pattern_id * tree.num_taxa + taxon_idx];

            if (observed_base < 4) {
                // Observed nucleotide
                for (int parent_state = 0; parent_state < 4; parent_state++) {
                    message[parent_state] = P[parent_state * 4 + observed_base];
                }
            }
            // else: gap, message stays at 1.0 for all states
        } else {
            // Internal edge: combine child messages
            int num_children = tree.d_node_num_children[child_node];
            int child_offset = tree.d_node_child_offsets[child_node];

            // d_node_children stores EDGE IDs (not node IDs) for each child
            // This was set up in allocate_gpu_tree

            for (int parent_state = 0; parent_state < 4; parent_state++) {
                double sum = 0.0;
                for (int child_state = 0; child_state < 4; child_state++) {
                    double prod = P[parent_state * 4 + child_state];

                    // Multiply by messages from child edges
                    for (int c = 0; c < num_children && c < 8; c++) {
                        int child_edge = tree.d_node_children[child_offset + c];
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
    for (int e = 0; e < tree.num_edges; e++) {
        total_log_scale += edge_log_scales[e];
    }

    // Multiply messages from root edges
    for (int r = 0; r < tree.num_root_edges; r++) {
        int root_edge = tree.d_root_edges[r];
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
        site_likelihood += tree.d_root_prob[dna] * combined_msg[dna];
    }

    // Store log-likelihood (not yet weighted by pattern count)
    d_pattern_log_likelihoods[pattern_id] = total_log_scale + log(site_likelihood);
}

// Reduction kernel to sum weighted log-likelihoods
__global__ void reduce_log_likelihoods(
    const double* d_pattern_log_likelihoods,
    const int* d_pattern_weights,
    int num_patterns,
    double* d_partial_sums,
    int num_blocks
) {
    extern __shared__ double shared_sum[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute local weighted sum
    double local_sum = 0.0;
    for (int i = global_id; i < num_patterns; i += blockDim.x * num_blocks) {
        local_sum += d_pattern_log_likelihoods[i] * d_pattern_weights[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// Host structures for loading data
struct HostTree {
    vector<TreeNode> nodes;
    vector<TreeEdge> edges;
    map<string, int> name_to_node;
    int root_node_id;
    vector<int> post_order;
    vector<int> root_edges;

    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<int> pattern_weights;
};

// Load tree from edge file
HostTree load_tree_data(const string& tree_file, const string& pattern_file,
                         const string& taxon_order_file) {
    HostTree tree;

    // Read taxon order
    vector<string> taxon_names;
    ifstream taxon_in(taxon_order_file);
    string line;
    getline(taxon_in, line);  // Skip header
    while (getline(taxon_in, line)) {
        size_t comma = line.find(',');
        if (comma != string::npos) {
            string name = line.substr(0, comma);
            taxon_names.push_back(name);
        }
    }
    taxon_in.close();
    tree.num_taxa = taxon_names.size();

    cout << "Loaded " << tree.num_taxa << " taxa" << endl;

    // Read tree edges
    ifstream tree_in(tree_file);
    int edge_idx = 0;
    while (getline(tree_in, line)) {
        istringstream iss(line);
        string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;

        TreeEdge edge;
        edge.id = edge_idx;
        edge.parent_name = parent_name;
        edge.child_name = child_name;
        edge.branch_length = branch_length;
        tree.edges.push_back(edge);
        edge_idx++;
    }
    tree_in.close();

    cout << "Loaded " << tree.edges.size() << " edges" << endl;

    // Build node structure from edges
    set<string> all_names;
    for (const auto& e : tree.edges) {
        all_names.insert(e.parent_name);
        all_names.insert(e.child_name);
    }

    int node_id = 0;
    for (const string& name : all_names) {
        TreeNode node;
        node.name = name;
        node.id = node_id;
        node.parent_id = -1;
        node.edge_id = -1;
        node.is_leaf = true;
        node.taxon_index = -1;
        tree.nodes.push_back(node);
        tree.name_to_node[name] = node_id;
        node_id++;
    }

    // Set parent-child relationships
    for (auto& e : tree.edges) {
        int parent_id = tree.name_to_node[e.parent_name];
        int child_id = tree.name_to_node[e.child_name];
        e.parent_node = parent_id;
        e.child_node = child_id;
        tree.nodes[child_id].parent_id = parent_id;
        tree.nodes[child_id].edge_id = e.id;
        tree.nodes[parent_id].child_ids.push_back(child_id);
        tree.nodes[parent_id].is_leaf = false;
    }

    // Mark leaves and assign taxon indices
    map<string, int> taxon_name_to_index;
    for (int i = 0; i < tree.num_taxa; i++) {
        taxon_name_to_index[taxon_names[i]] = i;
    }

    for (auto& node : tree.nodes) {
        if (node.is_leaf) {
            auto it = taxon_name_to_index.find(node.name);
            if (it != taxon_name_to_index.end()) {
                node.taxon_index = it->second;
            }
        }
    }

    // Find root (node with no parent)
    tree.root_node_id = -1;
    for (const auto& node : tree.nodes) {
        if (node.parent_id == -1) {
            tree.root_node_id = node.id;
            break;
        }
    }

    cout << "Root node: " << tree.nodes[tree.root_node_id].name << endl;

    // Find root edges (edges from root's children)
    for (int child_id : tree.nodes[tree.root_node_id].child_ids) {
        tree.root_edges.push_back(tree.nodes[child_id].edge_id);
    }

    cout << "Root edges: " << tree.root_edges.size() << endl;

    // Compute post-order traversal
    function<void(int)> compute_post_order = [&](int node_id) {
        for (int child_id : tree.nodes[node_id].child_ids) {
            compute_post_order(child_id);
            tree.post_order.push_back(tree.nodes[child_id].edge_id);
        }
    };
    compute_post_order(tree.root_node_id);

    cout << "Post-order edges: " << tree.post_order.size() << endl;

    // Load pattern data
    ifstream pat_in(pattern_file);
    getline(pat_in, line);
    int file_patterns, file_taxa;
    sscanf(line.c_str(), "%d %d", &file_patterns, &file_taxa);
    tree.num_patterns = file_patterns;

    tree.pattern_bases.resize(tree.num_patterns * tree.num_taxa);
    tree.pattern_weights.resize(tree.num_patterns);

    for (int p = 0; p < tree.num_patterns; p++) {
        getline(pat_in, line);
        istringstream iss(line);
        int weight;
        iss >> weight;
        tree.pattern_weights[p] = weight;
        for (int t = 0; t < tree.num_taxa; t++) {
            int base;
            iss >> base;
            tree.pattern_bases[p * tree.num_taxa + t] = (uint8_t)base;
        }
    }
    pat_in.close();

    cout << "Loaded " << tree.num_patterns << " patterns" << endl;

    return tree;
}

// Allocate and copy data to GPU
GPUTree allocate_gpu_tree(const HostTree& host_tree) {
    GPUTree gpu_tree;

    gpu_tree.num_edges = host_tree.edges.size();
    gpu_tree.num_nodes = host_tree.nodes.size();
    gpu_tree.num_taxa = host_tree.num_taxa;
    gpu_tree.num_patterns = host_tree.num_patterns;
    gpu_tree.num_root_edges = host_tree.root_edges.size();

    // Compute max children
    gpu_tree.max_children = 0;
    for (const auto& node : host_tree.nodes) {
        if (node.child_ids.size() > gpu_tree.max_children) {
            gpu_tree.max_children = node.child_ids.size();
        }
    }

    // Flatten node children (store EDGE IDs, not node IDs)
    vector<int> node_child_edges;
    vector<int> node_child_offsets(gpu_tree.num_nodes);
    vector<int> node_num_children(gpu_tree.num_nodes);
    vector<int> node_is_leaf(gpu_tree.num_nodes);
    vector<int> node_taxon_index(gpu_tree.num_nodes);

    int offset = 0;
    for (int n = 0; n < gpu_tree.num_nodes; n++) {
        const TreeNode& node = host_tree.nodes[n];
        node_is_leaf[n] = node.is_leaf ? 1 : 0;
        node_taxon_index[n] = node.taxon_index;
        node_num_children[n] = node.child_ids.size();
        node_child_offsets[n] = offset;

        // Store edge IDs instead of node IDs
        for (int child_node_id : node.child_ids) {
            int child_edge_id = host_tree.nodes[child_node_id].edge_id;
            node_child_edges.push_back(child_edge_id);
        }
        offset += node.child_ids.size();
    }

    // Edge info
    vector<int> edge_child_node(gpu_tree.num_edges);
    vector<double> trans_matrices(gpu_tree.num_edges * 16);

    for (int e = 0; e < gpu_tree.num_edges; e++) {
        const TreeEdge& edge = host_tree.edges[e];
        edge_child_node[e] = edge.child_node;

        // Compute JC transition matrix
        double t = edge.branch_length;
        double exp_term = exp(-4.0 * t / 3.0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    trans_matrices[e * 16 + i * 4 + j] = 0.25 + 0.75 * exp_term;
                } else {
                    trans_matrices[e * 16 + i * 4 + j] = 0.25 - 0.25 * exp_term;
                }
            }
        }
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_node_is_leaf, gpu_tree.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_node_taxon_index, gpu_tree.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_node_num_children, gpu_tree.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_node_child_offsets, gpu_tree.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_node_children, node_child_edges.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&gpu_tree.d_edge_child_node, gpu_tree.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_trans_matrices, gpu_tree.num_edges * 16 * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&gpu_tree.d_post_order, host_tree.post_order.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&gpu_tree.d_pattern_bases, host_tree.pattern_bases.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_pattern_weights, host_tree.pattern_weights.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&gpu_tree.d_root_edges, host_tree.root_edges.size() * sizeof(int)));

    double root_prob[4] = {0.25, 0.25, 0.25, 0.25};
    CUDA_CHECK(cudaMalloc(&gpu_tree.d_root_prob, 4 * sizeof(double)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_node_is_leaf, node_is_leaf.data(),
                          gpu_tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_node_taxon_index, node_taxon_index.data(),
                          gpu_tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_node_num_children, node_num_children.data(),
                          gpu_tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_node_child_offsets, node_child_offsets.data(),
                          gpu_tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_node_children, node_child_edges.data(),
                          node_child_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_tree.d_edge_child_node, edge_child_node.data(),
                          gpu_tree.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_trans_matrices, trans_matrices.data(),
                          gpu_tree.num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_tree.d_post_order, host_tree.post_order.data(),
                          host_tree.post_order.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_tree.d_pattern_bases, host_tree.pattern_bases.data(),
                          host_tree.pattern_bases.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_tree.d_pattern_weights, host_tree.pattern_weights.data(),
                          host_tree.pattern_weights.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_tree.d_root_edges, host_tree.root_edges.data(),
                          host_tree.root_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_tree.d_root_prob, root_prob, 4 * sizeof(double), cudaMemcpyHostToDevice));

    return gpu_tree;
}

// Free GPU memory
void free_gpu_tree(GPUTree& gpu_tree) {
    cudaFree(gpu_tree.d_node_is_leaf);
    cudaFree(gpu_tree.d_node_taxon_index);
    cudaFree(gpu_tree.d_node_num_children);
    cudaFree(gpu_tree.d_node_child_offsets);
    cudaFree(gpu_tree.d_node_children);
    cudaFree(gpu_tree.d_edge_child_node);
    cudaFree(gpu_tree.d_trans_matrices);
    cudaFree(gpu_tree.d_post_order);
    cudaFree(gpu_tree.d_pattern_bases);
    cudaFree(gpu_tree.d_pattern_weights);
    cudaFree(gpu_tree.d_root_edges);
    cudaFree(gpu_tree.d_root_prob);
}

// CPU reference implementation for validation
double compute_ll_cpu(const HostTree& tree, const array<double, 4>& root_prob) {
    auto start = chrono::high_resolution_clock::now();

    // Precompute transition matrices
    vector<array<array<double, 4>, 4>> trans_matrices(tree.edges.size());
    for (size_t e = 0; e < tree.edges.size(); e++) {
        double t = tree.edges[e].branch_length;
        double exp_term = exp(-4.0 * t / 3.0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    trans_matrices[e][i][j] = 0.25 + 0.75 * exp_term;
                } else {
                    trans_matrices[e][i][j] = 0.25 - 0.25 * exp_term;
                }
            }
        }
    }

    double total_ll = 0.0;

    for (int p = 0; p < tree.num_patterns; p++) {
        vector<array<double, 4>> edge_messages(tree.edges.size());
        vector<double> edge_log_scales(tree.edges.size(), 0.0);

        for (int edge_id : tree.post_order) {
            int child_node = tree.edges[edge_id].child_node;
            const auto& P = trans_matrices[edge_id];

            array<double, 4> message = {1.0, 1.0, 1.0, 1.0};

            if (tree.nodes[child_node].is_leaf) {
                int taxon_idx = tree.nodes[child_node].taxon_index;
                uint8_t observed_base = tree.pattern_bases[p * tree.num_taxa + taxon_idx];

                if (observed_base < 4) {
                    for (int parent_state = 0; parent_state < 4; parent_state++) {
                        message[parent_state] = P[parent_state][observed_base];
                    }
                }
            } else {
                for (int parent_state = 0; parent_state < 4; parent_state++) {
                    double sum = 0.0;
                    for (int child_state = 0; child_state < 4; child_state++) {
                        double prod = P[parent_state][child_state];
                        for (int child_node_id : tree.nodes[child_node].child_ids) {
                            int child_edge = tree.nodes[child_node_id].edge_id;
                            prod *= edge_messages[child_edge][child_state];
                        }
                        sum += prod;
                    }
                    message[parent_state] = sum;
                }
            }

            // Scale
            double max_val = *max_element(message.begin(), message.end());
            if (max_val > 0.0) {
                for (int i = 0; i < 4; i++) {
                    message[i] /= max_val;
                }
                edge_log_scales[edge_id] = log(max_val);
            }

            edge_messages[edge_id] = message;
        }

        // Combine root edges
        array<double, 4> combined_msg = {1.0, 1.0, 1.0, 1.0};
        double total_log_scale = 0.0;

        for (size_t e = 0; e < tree.edges.size(); e++) {
            total_log_scale += edge_log_scales[e];
        }

        for (int root_edge : tree.root_edges) {
            for (int i = 0; i < 4; i++) {
                combined_msg[i] *= edge_messages[root_edge][i];
            }
        }

        double max_val = *max_element(combined_msg.begin(), combined_msg.end());
        if (max_val > 0.0) {
            for (int i = 0; i < 4; i++) {
                combined_msg[i] /= max_val;
            }
            total_log_scale += log(max_val);
        }

        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += root_prob[dna] * combined_msg[dna];
        }

        total_ll += (total_log_scale + log(site_likelihood)) * tree.pattern_weights[p];
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "CPU computation time: " << duration.count() / 1000.0 << " ms" << endl;

    return total_ll;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <tree_edges.txt> <patterns.pat> <taxon_order.csv>" << endl;
        return 1;
    }

    string tree_file = argv[1];
    string pattern_file = argv[2];
    string taxon_order_file = argv[3];

    cout << "=== Naive GPU Log-Likelihood Computation ===" << endl;

    // Load data
    HostTree host_tree = load_tree_data(tree_file, pattern_file, taxon_order_file);

    // CPU reference
    array<double, 4> root_prob = {0.25, 0.25, 0.25, 0.25};
    cout << "\nComputing CPU reference..." << endl;
    double cpu_ll = compute_ll_cpu(host_tree, root_prob);
    cout << "CPU Log-likelihood: " << cpu_ll << endl;

    // Allocate GPU structures
    cout << "\nAllocating GPU memory..." << endl;
    GPUTree gpu_tree = allocate_gpu_tree(host_tree);

    // Allocate output buffer
    double* d_pattern_lls;
    CUDA_CHECK(cudaMalloc(&d_pattern_lls, gpu_tree.num_patterns * sizeof(double)));

    // Allocate working memory for edge messages and log scales
    double* d_edge_messages;
    double* d_edge_log_scales;
    size_t edge_msg_size = (size_t)gpu_tree.num_patterns * gpu_tree.num_edges * 4 * sizeof(double);
    size_t edge_scale_size = (size_t)gpu_tree.num_patterns * gpu_tree.num_edges * sizeof(double);

    cout << "Allocating GPU working memory..." << endl;
    cout << "  Edge messages: " << edge_msg_size / (1024.0 * 1024.0) << " MB" << endl;
    cout << "  Edge log scales: " << edge_scale_size / (1024.0 * 1024.0) << " MB" << endl;

    CUDA_CHECK(cudaMalloc(&d_edge_messages, edge_msg_size));
    CUDA_CHECK(cudaMalloc(&d_edge_log_scales, edge_scale_size));

    // Launch kernel
    cout << "Launching GPU kernel..." << endl;
    int block_size = 256;
    int num_blocks = (gpu_tree.num_patterns + block_size - 1) / block_size;
    cout << "Grid: " << num_blocks << " blocks x " << block_size << " threads" << endl;

    auto start = chrono::high_resolution_clock::now();

    compute_pattern_log_likelihoods<<<num_blocks, block_size>>>(
        gpu_tree, d_pattern_lls, d_edge_messages, d_edge_log_scales
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();
    auto kernel_duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "GPU kernel time: " << kernel_duration.count() / 1000.0 << " ms" << endl;

    // Copy results back and sum
    vector<double> pattern_lls(gpu_tree.num_patterns);
    CUDA_CHECK(cudaMemcpy(pattern_lls.data(), d_pattern_lls,
                          gpu_tree.num_patterns * sizeof(double), cudaMemcpyDeviceToHost));

    double gpu_ll = 0.0;
    for (int p = 0; p < gpu_tree.num_patterns; p++) {
        gpu_ll += pattern_lls[p] * host_tree.pattern_weights[p];
    }

    cout << "GPU Log-likelihood: " << gpu_ll << endl;
    cout << "Difference: " << abs(gpu_ll - cpu_ll) << endl;

    if (abs(gpu_ll - cpu_ll) < 1e-6) {
        cout << "\nVALIDATION PASSED: GPU matches CPU!" << endl;
    } else {
        cout << "\nVALIDATION FAILED: Results differ!" << endl;
    }

    // Cleanup
    cudaFree(d_pattern_lls);
    cudaFree(d_edge_messages);
    cudaFree(d_edge_log_scales);
    free_gpu_tree(gpu_tree);

    cout << "\n=== Done ===" << endl;
    return 0;
}
