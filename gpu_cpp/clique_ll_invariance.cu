#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <random>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// Clique Tree structure for verification
struct CliqueTree {
    int num_nodes;
    int num_leaves;
    int root_node;

    vector<string> node_names;
    map<string, int> name_to_id;
    vector<int> parent_map;
    vector<double> branch_lengths;

    int num_cliques;
    vector<int> clique_x_node;  // Parent node in clique edge
    vector<int> clique_y_node;  // Child node in clique edge
    vector<int> clique_parent;
    vector<vector<int>> clique_children;

    int num_taxa;
    int num_patterns;
    vector<int> pattern_data;
    vector<int> pattern_weights;
    vector<string> taxon_names;

    array<double, 4> root_prior;
    double f81_mu;
};

// GPU kernel for computing log-likelihood with proper clique tree belief propagation
__global__ void compute_ll_at_clique_kernel(
    int num_patterns,
    int num_cliques,
    const int* d_clique_y_node,
    const int* d_postorder_cliques,
    const int* d_pattern_data,
    const int* d_pattern_weights,
    const double* d_branch_lengths,
    const double* d_root_prior,
    double f81_mu,
    int num_taxa,
    const int* d_leaf_taxon_idx,  // Maps node -> taxon index (-1 if not leaf)
    const int* d_node_is_leaf,
    const int* d_clique_num_children,
    const int* d_clique_child_offsets,
    const int* d_clique_child_ids,
    const int* d_root_cliques,
    int num_root_cliques,
    double* d_pattern_lls
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    // Local belief arrays - each clique has 4x4 beliefs
    // Use smaller max to reduce register pressure (128 * 16 * 8 = 16KB)
    double clique_beliefs[128][4][4];  // Max 128 cliques
    double clique_scales[128];

    if (num_cliques > 128) return;  // Safety check

    // Initialize all beliefs to 1.0 and scales to 0
    for (int c = 0; c < num_cliques; c++) {
        clique_scales[c] = 0.0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                clique_beliefs[c][i][j] = 1.0;
            }
        }
    }

    // Process cliques in postorder (leaves to root)
    for (int idx = 0; idx < num_cliques; idx++) {
        int c = d_postorder_cliques[idx];
        int y = d_clique_y_node[c];
        double t = d_branch_lengths[y];

        // Compute F81 transition matrix
        double exp_term = exp(-f81_mu * t);
        double P[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    P[i][j] = exp_term + (1.0 - exp_term) * d_root_prior[j];
                } else {
                    P[i][j] = (1.0 - exp_term) * d_root_prior[j];
                }
            }
        }

        // Check if y is a leaf
        if (d_node_is_leaf[y]) {
            // Get taxon index directly from precomputed map
            int taxon_idx = d_leaf_taxon_idx[y];

            if (taxon_idx >= 0) {
                int base = d_pattern_data[pid * num_taxa + taxon_idx];
                if (base < 4) {
                    for (int xstate = 0; xstate < 4; xstate++) {
                        for (int ystate = 0; ystate < 4; ystate++) {
                            if (ystate == base) {
                                clique_beliefs[c][xstate][ystate] = P[xstate][ystate];
                            } else {
                                clique_beliefs[c][xstate][ystate] = 0.0;
                            }
                        }
                    }
                }
            }
        } else {
            // Internal node - multiply by transition and child messages
            int num_children = d_clique_num_children[c];
            int child_offset = d_clique_child_offsets[c];

            for (int xstate = 0; xstate < 4; xstate++) {
                for (int ystate = 0; ystate < 4; ystate++) {
                    double val = P[xstate][ystate];

                    // Multiply by messages from children
                    for (int ci = 0; ci < num_children; ci++) {
                        int child_c = d_clique_child_ids[child_offset + ci];
                        double child_sum = 0.0;
                        for (int zstate = 0; zstate < 4; zstate++) {
                            child_sum += clique_beliefs[child_c][ystate][zstate];
                        }
                        val *= child_sum;
                    }
                    clique_beliefs[c][xstate][ystate] = val;
                }
            }

            // Scale to prevent underflow
            double max_val = 0.0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    max_val = fmax(max_val, clique_beliefs[c][i][j]);
                }
            }
            if (max_val > 0.0 && max_val < 1e-100) {
                double scale = 1.0 / max_val;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        clique_beliefs[c][i][j] *= scale;
                    }
                }
                clique_scales[c] = log(max_val);
            }
        }
    }

    // Compute likelihood at first root clique only (all should give same result due to invariance)
    double ll = 0.0;
    double total_scale = 0.0;

    // Use first root clique
    int rc = d_root_cliques[0];
    for (int xstate = 0; xstate < 4; xstate++) {
        double sum_y = 0.0;
        for (int ystate = 0; ystate < 4; ystate++) {
            sum_y += clique_beliefs[rc][xstate][ystate];
        }
        ll += d_root_prior[xstate] * sum_y;
    }

    // Sum all scales
    for (int c = 0; c < num_cliques; c++) {
        total_scale += clique_scales[c];
    }

    d_pattern_lls[pid] = d_pattern_weights[pid] * (log(ll) + total_scale);
}

CliqueTree load_tree(const string& edge_file, const string& pattern_file,
                     const string& taxon_file, const string& basecomp_file) {
    CliqueTree tree;

    // Load base composition (format: "index value (count)")
    ifstream bc_in(basecomp_file);
    if (!bc_in.is_open()) {
        cerr << "Error: Could not open basecomp file " << basecomp_file << endl;
        exit(1);
    }
    for (int i = 0; i < 4; i++) {
        int idx;
        double val;
        string rest;
        bc_in >> idx >> val;
        getline(bc_in, rest);  // Skip rest of line
        tree.root_prior[idx] = val;
    }
    bc_in.close();

    // Compute F81 mu
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += tree.root_prior[i] * tree.root_prior[i];
    }
    tree.f81_mu = 1.0 / (1.0 - sum_pi_sq);

    // Load edges
    ifstream edge_in(edge_file);
    if (!edge_in.is_open()) {
        cerr << "Error: Could not open edge file " << edge_file << endl;
        exit(1);
    }

    string line;
    vector<pair<string, string>> edges;
    map<string, double> branch_map;

    while (getline(edge_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string parent, child;
        double branch_len;
        iss >> parent >> child >> branch_len;
        edges.push_back({parent, child});
        branch_map[child] = branch_len;

        if (!tree.name_to_id.count(parent)) {
            int id = tree.node_names.size();
            tree.name_to_id[parent] = id;
            tree.node_names.push_back(parent);
        }
        if (!tree.name_to_id.count(child)) {
            int id = tree.node_names.size();
            tree.name_to_id[child] = id;
            tree.node_names.push_back(child);
        }
    }
    edge_in.close();

    tree.num_nodes = tree.node_names.size();
    tree.parent_map.resize(tree.num_nodes, -1);
    tree.branch_lengths.resize(tree.num_nodes, 0.0);

    // Build parent map
    for (const auto& edge : edges) {
        int parent_id = tree.name_to_id[edge.first];
        int child_id = tree.name_to_id[edge.second];
        tree.parent_map[child_id] = parent_id;
        tree.branch_lengths[child_id] = branch_map[edge.second];
    }

    // Find root
    for (int i = 0; i < tree.num_nodes; i++) {
        if (tree.parent_map[i] == -1) {
            tree.root_node = i;
            break;
        }
    }

    // Load taxon order
    ifstream tax_in(taxon_file);
    getline(tax_in, line);  // Skip header
    while (getline(tax_in, line)) {
        if (!line.empty()) {
            size_t comma_pos = line.find(',');
            if (comma_pos != string::npos) {
                string taxon_name = line.substr(0, comma_pos);
                tree.taxon_names.push_back(taxon_name);
            }
        }
    }
    tax_in.close();
    tree.num_taxa = tree.taxon_names.size();

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

    // Build clique tree (each edge becomes a clique)
    tree.num_cliques = edges.size();
    tree.clique_x_node.resize(tree.num_cliques);
    tree.clique_y_node.resize(tree.num_cliques);
    tree.clique_parent.resize(tree.num_cliques, -1);
    tree.clique_children.resize(tree.num_cliques);

    for (size_t i = 0; i < edges.size(); i++) {
        int parent_id = tree.name_to_id[edges[i].first];
        int child_id = tree.name_to_id[edges[i].second];
        tree.clique_x_node[i] = parent_id;
        tree.clique_y_node[i] = child_id;
    }

    // Build clique tree parent-child relationships
    map<int, int> node_to_clique;
    for (int c = 0; c < tree.num_cliques; c++) {
        node_to_clique[tree.clique_y_node[c]] = c;
    }

    for (int c = 0; c < tree.num_cliques; c++) {
        int x = tree.clique_x_node[c];
        if (node_to_clique.count(x)) {
            int parent_clique = node_to_clique[x];
            tree.clique_parent[c] = parent_clique;
            tree.clique_children[parent_clique].push_back(c);
        }
    }

    // Count leaves
    tree.num_leaves = tree.num_taxa;

    cout << "Loaded: " << tree.num_nodes << " nodes, " << tree.num_cliques << " cliques, "
         << tree.num_patterns << " patterns" << endl;

    return tree;
}

// Compute log-likelihood at specified root clique (CPU version)
double compute_ll_at_clique_cpu(const CliqueTree& tree, int root_clique) {
    double total_ll = 0.0;

    // Build postorder traversal from root_clique
    vector<int> postorder;
    function<void(int)> build_postorder = [&](int c) {
        for (int child : tree.clique_children[c]) {
            build_postorder(child);
        }
        postorder.push_back(c);
    };

    // Find all root cliques (cliques with no parent)
    vector<int> root_cliques;
    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_parent[c] == -1) {
            root_cliques.push_back(c);
        }
    }

    for (int rc : root_cliques) {
        build_postorder(rc);
    }

    // Compute LL for each pattern
    for (int p = 0; p < tree.num_patterns; p++) {
        vector<array<array<double, 4>, 4>> clique_beliefs(tree.num_cliques);

        // Initialize beliefs
        for (int c = 0; c < tree.num_cliques; c++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    clique_beliefs[c][i][j] = 1.0;
                }
            }
        }

        // Process in postorder
        for (int c : postorder) {
            int y = tree.clique_y_node[c];
            double t = tree.branch_lengths[y];

            // Compute F81 transition matrix
            double exp_term = exp(-tree.f81_mu * t);
            array<array<double, 4>, 4> P;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j) {
                        P[i][j] = exp_term + (1.0 - exp_term) * tree.root_prior[j];
                    } else {
                        P[i][j] = (1.0 - exp_term) * tree.root_prior[j];
                    }
                }
            }

            // Check if y is a leaf
            bool is_leaf = true;
            for (int child_c = 0; child_c < tree.num_cliques; child_c++) {
                if (tree.clique_x_node[child_c] == y) {
                    is_leaf = false;
                    break;
                }
            }

            if (is_leaf && y < tree.num_taxa) {
                // Find taxon index
                int taxon_idx = -1;
                for (size_t t = 0; t < tree.taxon_names.size(); t++) {
                    if (tree.name_to_id.count(tree.taxon_names[t]) &&
                        tree.name_to_id.at(tree.taxon_names[t]) == y) {
                        taxon_idx = t;
                        break;
                    }
                }

                if (taxon_idx >= 0) {
                    int base = tree.pattern_data[p * tree.num_taxa + taxon_idx];
                    if (base < 4) {
                        for (int xstate = 0; xstate < 4; xstate++) {
                            for (int ystate = 0; ystate < 4; ystate++) {
                                if (ystate == base) {
                                    clique_beliefs[c][xstate][ystate] = P[xstate][ystate];
                                } else {
                                    clique_beliefs[c][xstate][ystate] = 0.0;
                                }
                            }
                        }
                    }
                }
            } else {
                // Internal node - multiply children beliefs and transition
                for (int xstate = 0; xstate < 4; xstate++) {
                    for (int ystate = 0; ystate < 4; ystate++) {
                        double val = P[xstate][ystate];
                        for (int child_c : tree.clique_children[c]) {
                            double child_sum = 0.0;
                            for (int zstate = 0; zstate < 4; zstate++) {
                                child_sum += clique_beliefs[child_c][ystate][zstate];
                            }
                            val *= child_sum;
                        }
                        clique_beliefs[c][xstate][ystate] = val;
                    }
                }
            }
        }

        // Compute likelihood at first root clique only
        double ll = 0.0;
        int rc = root_cliques[0];  // Use first root clique
        for (int xstate = 0; xstate < 4; xstate++) {
            double sum_y = 0.0;
            for (int ystate = 0; ystate < 4; ystate++) {
                sum_y += clique_beliefs[rc][xstate][ystate];
            }
            ll += tree.root_prior[xstate] * sum_y;
        }

        total_ll += tree.pattern_weights[p] * log(ll);
    }

    return total_ll;
}

// Compute log-likelihood using GPU
double compute_ll_gpu(const CliqueTree& tree) {
    // Build postorder traversal
    vector<int> postorder;
    function<void(int)> build_postorder = [&](int c) {
        for (int child : tree.clique_children[c]) {
            build_postorder(child);
        }
        postorder.push_back(c);
    };

    // Find all root cliques
    vector<int> root_cliques;
    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_parent[c] == -1) {
            root_cliques.push_back(c);
        }
    }

    for (int rc : root_cliques) {
        build_postorder(rc);
    }

    // Identify leaf nodes and build leaf taxon index map
    vector<int> node_is_leaf(tree.num_nodes, 0);
    vector<int> leaf_taxon_idx(tree.num_nodes, -1);  // Maps node -> taxon index

    for (int c = 0; c < tree.num_cliques; c++) {
        int y = tree.clique_y_node[c];
        bool is_leaf = true;
        for (int cc = 0; cc < tree.num_cliques; cc++) {
            if (tree.clique_x_node[cc] == y) {
                is_leaf = false;
                break;
            }
        }
        if (is_leaf) {
            node_is_leaf[y] = 1;
            // Find taxon index for this leaf
            for (size_t t = 0; t < tree.taxon_names.size(); t++) {
                if (tree.name_to_id.count(tree.taxon_names[t]) &&
                    tree.name_to_id.at(tree.taxon_names[t]) == y) {
                    leaf_taxon_idx[y] = t;
                    break;
                }
            }
        }
    }

    // Build taxon to node mapping (for reference)
    vector<int> taxon_to_node(tree.num_taxa);
    for (int t = 0; t < tree.num_taxa; t++) {
        if (tree.name_to_id.count(tree.taxon_names[t])) {
            taxon_to_node[t] = tree.name_to_id.at(tree.taxon_names[t]);
        }
    }

    // Build clique children arrays for GPU
    vector<int> clique_num_children(tree.num_cliques);
    vector<int> clique_child_offsets(tree.num_cliques);
    vector<int> clique_child_ids;

    int offset = 0;
    for (int c = 0; c < tree.num_cliques; c++) {
        clique_num_children[c] = tree.clique_children[c].size();
        clique_child_offsets[c] = offset;
        for (int child : tree.clique_children[c]) {
            clique_child_ids.push_back(child);
        }
        offset += tree.clique_children[c].size();
    }
    if (clique_child_ids.empty()) clique_child_ids.push_back(0);  // Avoid empty allocation

    // Allocate GPU memory
    int* d_clique_y_node;
    int* d_postorder_cliques;
    int* d_pattern_data;
    int* d_pattern_weights;
    double* d_branch_lengths;
    double* d_root_prior;
    int* d_leaf_taxon_idx;
    int* d_node_is_leaf;
    int* d_clique_num_children;
    int* d_clique_child_offsets;
    int* d_clique_child_ids;
    int* d_root_cliques;
    double* d_pattern_lls;

    cudaMalloc(&d_clique_y_node, tree.num_cliques * sizeof(int));
    cudaMalloc(&d_postorder_cliques, postorder.size() * sizeof(int));
    cudaMalloc(&d_pattern_data, tree.pattern_data.size() * sizeof(int));
    cudaMalloc(&d_pattern_weights, tree.num_patterns * sizeof(int));
    cudaMalloc(&d_branch_lengths, tree.num_nodes * sizeof(double));
    cudaMalloc(&d_root_prior, 4 * sizeof(double));
    cudaMalloc(&d_leaf_taxon_idx, tree.num_nodes * sizeof(int));
    cudaMalloc(&d_node_is_leaf, tree.num_nodes * sizeof(int));
    cudaMalloc(&d_clique_num_children, tree.num_cliques * sizeof(int));
    cudaMalloc(&d_clique_child_offsets, tree.num_cliques * sizeof(int));
    cudaMalloc(&d_clique_child_ids, clique_child_ids.size() * sizeof(int));
    cudaMalloc(&d_root_cliques, root_cliques.size() * sizeof(int));
    cudaMalloc(&d_pattern_lls, tree.num_patterns * sizeof(double));

    // Copy data to GPU
    cudaMemcpy(d_clique_y_node, tree.clique_y_node.data(), tree.num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_postorder_cliques, postorder.data(), postorder.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_data, tree.pattern_data.data(), tree.pattern_data.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_weights, tree.pattern_weights.data(), tree.num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_branch_lengths, tree.branch_lengths.data(), tree.num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_prior, tree.root_prior.data(), 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaf_taxon_idx, leaf_taxon_idx.data(), tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_is_leaf, node_is_leaf.data(), tree.num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clique_num_children, clique_num_children.data(), tree.num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clique_child_offsets, clique_child_offsets.data(), tree.num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clique_child_ids, clique_child_ids.data(), clique_child_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_cliques, root_cliques.data(), root_cliques.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int grid_size = (tree.num_patterns + block_size - 1) / block_size;

    compute_ll_at_clique_kernel<<<grid_size, block_size>>>(
        tree.num_patterns,
        tree.num_cliques,
        d_clique_y_node,
        d_postorder_cliques,
        d_pattern_data,
        d_pattern_weights,
        d_branch_lengths,
        d_root_prior,
        tree.f81_mu,
        tree.num_taxa,
        d_leaf_taxon_idx,
        d_node_is_leaf,
        d_clique_num_children,
        d_clique_child_offsets,
        d_clique_child_ids,
        d_root_cliques,
        (int)root_cliques.size(),
        d_pattern_lls
    );

    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }

    // Copy results back
    vector<double> pattern_lls(tree.num_patterns);
    cudaMemcpy(pattern_lls.data(), d_pattern_lls, tree.num_patterns * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum up log-likelihoods
    double total_ll = 0.0;
    for (int p = 0; p < tree.num_patterns; p++) {
        total_ll += pattern_lls[p];
    }

    // Free GPU memory
    cudaFree(d_clique_y_node);
    cudaFree(d_postorder_cliques);
    cudaFree(d_pattern_data);
    cudaFree(d_pattern_weights);
    cudaFree(d_branch_lengths);
    cudaFree(d_root_prior);
    cudaFree(d_leaf_taxon_idx);
    cudaFree(d_node_is_leaf);
    cudaFree(d_clique_num_children);
    cudaFree(d_clique_child_offsets);
    cudaFree(d_clique_child_ids);
    cudaFree(d_root_cliques);
    cudaFree(d_pattern_lls);

    return total_ll;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " -e <edge_file> -p <patterns> -x <taxon_order> -b <basecomp> [-n num_cliques]" << endl;
        cerr << "Verifies that log-likelihood is the same when computed at different cliques" << endl;
        return 1;
    }

    string edge_file, pattern_file, taxon_file, basecomp_file;
    int num_test_cliques = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) basecomp_file = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) num_test_cliques = atoi(argv[++i]);
    }

    if (edge_file.empty() || pattern_file.empty() || taxon_file.empty() || basecomp_file.empty()) {
        cerr << "Missing required arguments" << endl;
        return 1;
    }

    cout << "=== Clique Tree LL Invariance Test (GPU Accelerated) ===" << endl;

    CliqueTree tree = load_tree(edge_file, pattern_file, taxon_file, basecomp_file);

    // Compute LL using CPU
    cout << "\nComputing log-likelihood (CPU)..." << endl;
    double ll_cpu = compute_ll_at_clique_cpu(tree, 0);
    cout << fixed << setprecision(14);
    cout << "CPU Log-likelihood: " << ll_cpu << endl;

    // Compute LL using GPU
    cout << "\nComputing log-likelihood (GPU)..." << endl;
    double ll_gpu = compute_ll_gpu(tree);
    cout << "GPU Log-likelihood: " << ll_gpu << endl;

    // Compare CPU and GPU
    double cpu_gpu_diff = abs(ll_cpu - ll_gpu);
    cout << "\nCPU vs GPU difference: " << scientific << cpu_gpu_diff << endl;

    if (cpu_gpu_diff < 1e-8) {
        cout << "CPU-GPU MATCH VERIFIED!" << endl;
    } else {
        cout << "WARNING: CPU and GPU results differ!" << endl;
    }

    // Test at multiple random cliques (CPU only for clique invariance)
    cout << "\n=== Testing LL Invariance at Different Cliques (CPU) ===" << endl;
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, tree.num_cliques - 1);

    vector<double> lls;
    lls.push_back(ll_cpu);

    for (int i = 0; i < num_test_cliques; i++) {
        int test_clique = dist(rng);
        double ll = compute_ll_at_clique_cpu(tree, test_clique);
        lls.push_back(ll);
        cout << "Clique " << test_clique << " (edge " << tree.clique_x_node[test_clique]
             << " -> " << tree.clique_y_node[test_clique] << "): LL = " << ll << endl;
    }

    // Check invariance
    double max_diff = 0.0;
    for (size_t i = 1; i < lls.size(); i++) {
        double diff = abs(lls[i] - lls[0]);
        max_diff = max(max_diff, diff);
    }

    cout << "\n=== Results ===" << endl;
    cout << fixed << setprecision(14);
    cout << "Maximum LL difference across cliques: " << scientific << max_diff << endl;
    cout << "CPU-GPU difference: " << cpu_gpu_diff << endl;

    if (max_diff < 1e-8 && cpu_gpu_diff < 1e-8) {
        cout << "\nSUCCESS: Clique LL invariance verified and GPU matches CPU!" << endl;
    } else if (max_diff >= 1e-8) {
        cout << "\nWARNING: LL values differ across cliques!" << endl;
    } else {
        cout << "\nWARNING: GPU results don't match CPU!" << endl;
    }

    return 0;
}
