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

// GPU kernel: Compute log-likelihood for each pattern in parallel
__global__ void compute_pattern_ll_kernel(
    int num_patterns,
    int num_edges,
    int num_taxa,
    const int* d_post_order_edges,
    const int* d_edge_child_nodes,
    const int* d_edge_parent_nodes,
    const bool* d_node_is_leaf,
    const int* d_leaf_taxon_indices,
    const uint8_t* d_pattern_bases,
    const int* d_pattern_weights,
    const double* d_transition_matrices,  // [num_edges * 16]
    const double* d_root_probabilities,   // [4]
    const int* d_root_edge_ids,
    int num_root_edges,
    const int* d_node_num_children,
    const int* d_node_child_offsets,
    const int* d_node_child_edges,
    double* d_pattern_lls  // [num_patterns]
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    // Local arrays for messages and scales
    double msgs[256 * 4];  // Max 256 edges
    double scales[256];

    for (int e = 0; e < num_edges; e++) scales[e] = 0.0;

    // Compute messages in post-order
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

        // Scale to prevent underflow
        double mx = fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3]));
        if (mx > 0.0) {
            for (int i = 0; i < 4; i++) msg[i] /= mx;
            scales[eid] = log(mx);
        }

        for (int i = 0; i < 4; i++) msgs[eid * 4 + i] = msg[i];
    }

    // Combine at root
    double root_msg[4] = {1.0, 1.0, 1.0, 1.0};
    double root_log_scale = 0.0;

    for (int r = 0; r < num_root_edges; r++) {
        int eid = d_root_edge_ids[r];
        for (int i = 0; i < 4; i++) {
            root_msg[i] *= msgs[eid * 4 + i];
        }
    }

    // Sum all scales from all edges (they accumulate along the tree)
    for (int e = 0; e < num_edges; e++) {
        root_log_scale += scales[e];
    }

    double ll = 0.0;
    for (int i = 0; i < 4; i++) {
        ll += d_root_probabilities[i] * root_msg[i];
    }

    d_pattern_lls[pid] = d_pattern_weights[pid] * (log(ll) + root_log_scale);
}

struct BHTree {
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<bool> node_is_leaf;
    vector<int> leaf_taxon_indices;
    vector<vector<int>> node_children;
    vector<int> node_parent;
    int root_node_id;

    vector<int> edge_child_nodes;
    vector<int> edge_parent_nodes;
    vector<double> branch_lengths;  // branch length for each node (stored at child)
    vector<int> post_order_edges;
    vector<int> root_edge_ids;

    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<int> pattern_weights;
    vector<string> taxon_names;

    // Model parameters
    vector<array<array<double, 4>, 4>> transition_matrices;
    array<double, 4> root_probabilities;
    double f81_mu;

    // HSS model parameters for re-rooting
    map<pair<int,int>, array<array<double,4>,4>> M_hss;
    vector<array<double, 4>> node_root_prob_hss;
};

// Compute F81 transition matrix for a given branch length
array<array<double, 4>, 4> compute_f81_matrix(double branch_len, double f81_mu, const array<double, 4>& pi) {
    array<array<double, 4>, 4> P;
    double exp_term = exp(-f81_mu * branch_len);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i][j] = exp_term + (1.0 - exp_term) * pi[j];
            } else {
                P[i][j] = (1.0 - exp_term) * pi[j];
            }
        }
    }
    return P;
}

BHTree load_bh_parameters(const string& filename) {
    BHTree tree;
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Could not open BH parameters file " << filename << endl;
        exit(1);
    }

    string line;
    string current_section;
    int num_nodes = 0, num_edges = 0;
    int node_idx = 0;
    int edge_idx = 0;
    int matrix_row = 0;
    int current_matrix_id = -1;
    int root_line_counter = 0;
    bool in_edges_block = false;

    while (getline(in, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Check for section headers
        if (line[0] == '[') {
            current_section = line;

            // Check for TRANSITION_MATRIX sections
            if (current_section.find("[TRANSITION_MATRIX_") == 0) {
                size_t start = 19;  // length of "[TRANSITION_MATRIX_"
                size_t end = current_section.find(']');
                current_matrix_id = stoi(current_section.substr(start, end - start));
                matrix_row = 0;
            }

            // Reset counters for new sections
            if (current_section == "[ROOT]") {
                root_line_counter = 0;
            }

            // Track edges block
            if (current_section == "[EDGES]") {
                in_edges_block = true;
            } else if (current_section == "[PARENT_MAP]") {
                in_edges_block = false;
            }
            continue;
        }

        if (current_section == "[ROOT]") {
            // First line: root name, second line: root node id
            if (root_line_counter == 0) {
                // root name - ignore, we'll set later
                root_line_counter++;
            } else {
                tree.root_node_id = stoi(line);
            }
        } else if (current_section == "[ROOT_PROBABILITY]") {
            // Space-separated values
            istringstream iss(line);
            for (int i = 0; i < 4; i++) {
                iss >> tree.root_probabilities[i];
            }
        } else if (current_section == "[TREE_STRUCTURE]") {
            // num_nodes num_taxa num_edges
            istringstream iss(line);
            iss >> num_nodes >> tree.num_taxa >> num_edges;
            tree.node_names.resize(num_nodes);
            tree.node_is_leaf.resize(num_nodes, false);
            tree.node_children.resize(num_nodes);
            tree.node_parent.resize(num_nodes, -1);
            tree.leaf_taxon_indices.resize(num_nodes, -1);
            tree.branch_lengths.resize(num_nodes, 0.0);
            tree.transition_matrices.resize(num_edges);
        } else if (current_section == "[NODE_NAMES]") {
            // Just node names, one per line
            tree.node_names[node_idx] = line;
            tree.name_to_id[line] = node_idx;
            node_idx++;
        } else if (in_edges_block && current_section.find("[TRANSITION_MATRIX_") == 0) {
            // We're in the edges block, but after a TRANSITION_MATRIX section
            // Check if this line is an edge definition (contains spaces)
            if (line.find(' ') != string::npos && isalpha(line[0])) {
                // This is an edge line: parent child branch_length
                istringstream iss(line);
                string parent_name, child_name;
                double branch_len;
                iss >> parent_name >> child_name >> branch_len;

                if (tree.name_to_id.count(parent_name) && tree.name_to_id.count(child_name)) {
                    int parent_id = tree.name_to_id[parent_name];
                    int child_id = tree.name_to_id[child_name];

                    tree.edge_parent_nodes.push_back(parent_id);
                    tree.edge_child_nodes.push_back(child_id);
                    tree.branch_lengths[child_id] = branch_len;
                    tree.node_children[parent_id].push_back(child_id);
                    tree.node_parent[child_id] = parent_id;
                    edge_idx++;
                }
            } else if (matrix_row < 4) {
                // Matrix data
                istringstream iss(line);
                for (int j = 0; j < 4; j++) {
                    iss >> tree.transition_matrices[current_matrix_id][matrix_row][j];
                }
                matrix_row++;
            }
        } else if (current_section == "[EDGES]") {
            // First edge right after [EDGES] header
            if (line.find(' ') != string::npos && isalpha(line[0])) {
                istringstream iss(line);
                string parent_name, child_name;
                double branch_len;
                iss >> parent_name >> child_name >> branch_len;

                if (tree.name_to_id.count(parent_name) && tree.name_to_id.count(child_name)) {
                    int parent_id = tree.name_to_id[parent_name];
                    int child_id = tree.name_to_id[child_name];

                    tree.edge_parent_nodes.push_back(parent_id);
                    tree.edge_child_nodes.push_back(child_id);
                    tree.branch_lengths[child_id] = branch_len;
                    tree.node_children[parent_id].push_back(child_id);
                    tree.node_parent[child_id] = parent_id;
                    edge_idx++;
                }
            }
        } else if (current_section == "[PARENT_MAP]") {
            // Space-separated parent ids
            istringstream iss(line);
            int parent_id;
            int idx = 0;
            while (iss >> parent_id) {
                tree.node_parent[idx++] = parent_id;
            }
        }
    }

    in.close();

    // Mark leaf nodes
    for (size_t i = 0; i < tree.node_names.size(); i++) {
        if (tree.node_children[i].empty()) {
            tree.node_is_leaf[i] = true;
        }
    }

    // Build post-order traversal
    function<void(int)> post_order = [&](int node_id) {
        for (int child_id : tree.node_children[node_id]) {
            post_order(child_id);
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == child_id) {
                    tree.post_order_edges.push_back(e);
                    break;
                }
            }
        }
    };
    post_order(tree.root_node_id);

    // Build root edge ids
    for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
        if (tree.edge_parent_nodes[e] == tree.root_node_id) {
            tree.root_edge_ids.push_back(e);
        }
    }

    cout << "Loaded BH parameters: " << tree.node_names.size() << " nodes, "
         << tree.edge_child_nodes.size() << " edges, root at "
         << tree.node_names[tree.root_node_id] << endl;

    return tree;
}

void load_patterns(BHTree& tree, const string& pattern_file, const string& taxon_file) {
    // Load taxon order (CSV format with header: taxon_name,position)
    ifstream tax_in(taxon_file);
    if (!tax_in.is_open()) {
        cerr << "Error: Could not open taxon file " << taxon_file << endl;
        exit(1);
    }

    string line;
    // Skip header line
    getline(tax_in, line);

    while (getline(tax_in, line)) {
        if (!line.empty()) {
            // Parse CSV: taxon_name,position
            size_t comma_pos = line.find(',');
            if (comma_pos != string::npos) {
                string taxon_name = line.substr(0, comma_pos);
                tree.taxon_names.push_back(taxon_name);
            }
        }
    }
    tax_in.close();

    tree.num_taxa = tree.taxon_names.size();

    // Map taxon names to leaf nodes
    // taxon_idx is the position in the pattern file (column index)
    int taxon_idx = 0;
    for (const auto& taxon_name : tree.taxon_names) {
        if (tree.name_to_id.count(taxon_name)) {
            int node_id = tree.name_to_id[taxon_name];
            tree.leaf_taxon_indices[node_id] = taxon_idx;
        } else {
            cerr << "Warning: Taxon " << taxon_name << " not found in tree" << endl;
        }
        taxon_idx++;
    }

    // Load patterns
    ifstream pat_in(pattern_file);
    if (!pat_in.is_open()) {
        cerr << "Error: Could not open pattern file " << pattern_file << endl;
        exit(1);
    }

    while (getline(pat_in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;
        tree.pattern_weights.push_back(weight);

        // Read numeric base values (0=A, 1=C, 2=G, 3=T, 4=unknown)
        int base;
        while (iss >> base) {
            tree.pattern_bases.push_back((uint8_t)base);
        }
    }

    tree.num_patterns = tree.pattern_weights.size();
    pat_in.close();

    cout << "Loaded " << tree.num_patterns << " patterns" << endl;
}

// Reparameterize BH model using Bayes rule
void reparameterize_bh(BHTree& tree) {
    int num_nodes = tree.node_names.size();
    tree.node_root_prob_hss.resize(num_nodes);
    tree.M_hss.clear();

    // Set root probability at root node
    tree.node_root_prob_hss[tree.root_node_id] = tree.root_probabilities;

    // Traverse tree in pre-order to compute pi at each node and HSS matrices
    vector<int> to_visit;
    to_visit.push_back(tree.root_node_id);

    while (!to_visit.empty()) {
        int parent_id = to_visit.back();
        to_visit.pop_back();

        array<double, 4> pi_p = tree.node_root_prob_hss[parent_id];

        for (int child_id : tree.node_children[parent_id]) {
            // Find edge from parent to child
            int edge_id = -1;
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == child_id && tree.edge_parent_nodes[e] == parent_id) {
                    edge_id = e;
                    break;
                }
            }
            if (edge_id < 0) continue;

            // Get forward transition matrix (parent -> child)
            array<array<double, 4>, 4> M_pc = tree.transition_matrices[edge_id];
            tree.M_hss[{parent_id, child_id}] = M_pc;

            // Compute pi_c (root probability at child)
            array<double, 4> pi_c = {0.0, 0.0, 0.0, 0.0};
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    pi_c[x] += pi_p[y] * M_pc[y][x];
                }
            }
            tree.node_root_prob_hss[child_id] = pi_c;

            // Compute backward matrix using Bayes rule
            // M_cp[y][x] = P(parent=x | child=y) = P(child=y | parent=x) * P(parent=x) / P(child=y)
            array<array<double, 4>, 4> M_cp;
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    M_cp[y][x] = M_pc[x][y] * pi_p[x] / max(1e-14, pi_c[y]);
                }
            }
            tree.M_hss[{child_id, parent_id}] = M_cp;

            to_visit.push_back(child_id);
        }
    }
}

// Re-root tree at new vertex
void reroot_tree(BHTree& tree, int new_root_id) {
    if (new_root_id == tree.root_node_id) return;

    // Build path from new root to old root
    vector<int> path;
    int current = new_root_id;
    while (current != tree.root_node_id) {
        path.push_back(current);
        current = tree.node_parent[current];
        if (current < 0) break;
    }
    path.push_back(tree.root_node_id);

    // Reverse edges along the path
    for (size_t i = 0; i < path.size() - 1; i++) {
        int child = path[i];
        int parent = path[i + 1];

        auto& p_children = tree.node_children[parent];
        p_children.erase(remove(p_children.begin(), p_children.end(), child), p_children.end());
        tree.node_children[child].push_back(parent);
        tree.node_parent[parent] = child;
    }
    tree.node_parent[new_root_id] = -1;

    // Rebuild edges with new directions
    tree.edge_child_nodes.clear();
    tree.edge_parent_nodes.clear();
    tree.transition_matrices.clear();

    function<void(int)> rebuild_edges = [&](int node_id) {
        for (int child_id : tree.node_children[node_id]) {
            tree.edge_parent_nodes.push_back(node_id);
            tree.edge_child_nodes.push_back(child_id);

            if (tree.M_hss.count({node_id, child_id})) {
                tree.transition_matrices.push_back(tree.M_hss[{node_id, child_id}]);
            } else {
                cerr << "Warning: Missing HSS matrix for edge " << node_id << " -> " << child_id << endl;
                array<array<double, 4>, 4> identity;
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        identity[i][j] = (i == j) ? 1.0 : 0.0;
                tree.transition_matrices.push_back(identity);
            }

            rebuild_edges(child_id);
        }
    };
    rebuild_edges(new_root_id);

    tree.root_node_id = new_root_id;
    tree.root_probabilities = tree.node_root_prob_hss[new_root_id];

    // Rebuild post-order traversal
    tree.post_order_edges.clear();
    function<void(int)> post_order = [&](int node_id) {
        for (int child_id : tree.node_children[node_id]) {
            post_order(child_id);
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == child_id) {
                    tree.post_order_edges.push_back(e);
                    break;
                }
            }
        }
    };
    post_order(new_root_id);

    tree.root_edge_ids.clear();
    for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
        if (tree.edge_parent_nodes[e] == new_root_id) {
            tree.root_edge_ids.push_back(e);
        }
    }
}

// GPU-accelerated log-likelihood computation
double compute_log_likelihood_gpu(const BHTree& tree) {
    int num_edges = tree.edge_child_nodes.size();
    int num_nodes = tree.node_names.size();
    int num_patterns = tree.num_patterns;
    int num_taxa = tree.num_taxa;

    // Build node child edge mapping
    vector<int> node_num_children(num_nodes, 0);
    vector<int> node_child_offsets(num_nodes, 0);
    vector<int> node_child_edges;

    for (int n = 0; n < num_nodes; n++) {
        node_num_children[n] = tree.node_children[n].size();
        node_child_offsets[n] = node_child_edges.size();
        for (int child_id : tree.node_children[n]) {
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == child_id) {
                    node_child_edges.push_back(e);
                    break;
                }
            }
        }
    }

    // Flatten transition matrices
    vector<double> flat_matrices(num_edges * 16);
    for (int e = 0; e < num_edges; e++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                flat_matrices[e * 16 + i * 4 + j] = tree.transition_matrices[e][i][j];
            }
        }
    }

    // Allocate GPU memory
    int *d_post_order_edges, *d_edge_child_nodes, *d_edge_parent_nodes;
    bool *d_node_is_leaf;
    int *d_leaf_taxon_indices;
    uint8_t *d_pattern_bases;
    int *d_pattern_weights;
    double *d_transition_matrices, *d_root_probabilities;
    int *d_root_edge_ids;
    int *d_node_num_children, *d_node_child_offsets, *d_node_child_edges;
    double *d_pattern_lls;

    cudaMalloc(&d_post_order_edges, num_edges * sizeof(int));
    cudaMalloc(&d_edge_child_nodes, num_edges * sizeof(int));
    cudaMalloc(&d_edge_parent_nodes, num_edges * sizeof(int));
    cudaMalloc(&d_node_is_leaf, num_nodes * sizeof(bool));
    cudaMalloc(&d_leaf_taxon_indices, num_nodes * sizeof(int));
    cudaMalloc(&d_pattern_bases, num_patterns * num_taxa * sizeof(uint8_t));
    cudaMalloc(&d_pattern_weights, num_patterns * sizeof(int));
    cudaMalloc(&d_transition_matrices, num_edges * 16 * sizeof(double));
    cudaMalloc(&d_root_probabilities, 4 * sizeof(double));
    cudaMalloc(&d_root_edge_ids, tree.root_edge_ids.size() * sizeof(int));
    cudaMalloc(&d_node_num_children, num_nodes * sizeof(int));
    cudaMalloc(&d_node_child_offsets, num_nodes * sizeof(int));
    cudaMalloc(&d_node_child_edges, node_child_edges.size() * sizeof(int));
    cudaMalloc(&d_pattern_lls, num_patterns * sizeof(double));

    // Copy data to GPU
    // Convert vector<bool> to char array (vector<bool> is specialized and doesn't have .data())
    vector<char> node_is_leaf_chars(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        node_is_leaf_chars[i] = tree.node_is_leaf[i] ? 1 : 0;
    }

    cudaMemcpy(d_post_order_edges, tree.post_order_edges.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_child_nodes, tree.edge_child_nodes.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_parent_nodes, tree.edge_parent_nodes.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_is_leaf, node_is_leaf_chars.data(), num_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaf_taxon_indices, tree.leaf_taxon_indices.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_bases, tree.pattern_bases.data(), num_patterns * num_taxa * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_weights, tree.pattern_weights.data(), num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_matrices, flat_matrices.data(), num_edges * 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_probabilities, tree.root_probabilities.data(), 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_edge_ids, tree.root_edge_ids.data(), tree.root_edge_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_num_children, node_num_children.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_child_offsets, node_child_offsets.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_child_edges, node_child_edges.data(), node_child_edges.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_patterns + block_size - 1) / block_size;
    compute_pattern_ll_kernel<<<num_blocks, block_size>>>(
        num_patterns, num_edges, num_taxa,
        d_post_order_edges, d_edge_child_nodes, d_edge_parent_nodes,
        d_node_is_leaf, d_leaf_taxon_indices, d_pattern_bases, d_pattern_weights,
        d_transition_matrices, d_root_probabilities,
        d_root_edge_ids, tree.root_edge_ids.size(),
        d_node_num_children, d_node_child_offsets, d_node_child_edges,
        d_pattern_lls
    );

    // Copy results back and sum
    vector<double> pattern_lls(num_patterns);
    cudaMemcpy(pattern_lls.data(), d_pattern_lls, num_patterns * sizeof(double), cudaMemcpyDeviceToHost);

    double total_ll = 0.0;
    for (int p = 0; p < num_patterns; p++) {
        total_ll += pattern_lls[p];
    }

    // Free GPU memory
    cudaFree(d_post_order_edges);
    cudaFree(d_edge_child_nodes);
    cudaFree(d_edge_parent_nodes);
    cudaFree(d_node_is_leaf);
    cudaFree(d_leaf_taxon_indices);
    cudaFree(d_pattern_bases);
    cudaFree(d_pattern_weights);
    cudaFree(d_transition_matrices);
    cudaFree(d_root_probabilities);
    cudaFree(d_root_edge_ids);
    cudaFree(d_node_num_children);
    cudaFree(d_node_child_offsets);
    cudaFree(d_node_child_edges);
    cudaFree(d_pattern_lls);

    return total_ll;
}

// CPU fallback for log-likelihood (used for verification)
double compute_log_likelihood_cpu(const BHTree& tree) {
    double total_ll = 0.0;

    for (int p = 0; p < tree.num_patterns; p++) {
        vector<array<double, 4>> edge_messages(tree.edge_child_nodes.size());
        vector<double> log_scales(tree.edge_child_nodes.size(), 0.0);

        // Compute messages for each edge in post-order
        for (int eid : tree.post_order_edges) {
            int child = tree.edge_child_nodes[eid];
            const auto& P = tree.transition_matrices[eid];

            array<double, 4> msg = {1.0, 1.0, 1.0, 1.0};

            if (tree.node_is_leaf[child]) {
                int taxon = tree.leaf_taxon_indices[child];
                uint8_t base = tree.pattern_bases[p * tree.num_taxa + taxon];
                if (base < 4) {
                    for (int ps = 0; ps < 4; ps++) {
                        msg[ps] = P[ps][base];
                    }
                }
            } else {
                // Internal node: combine messages from children
                for (int ps = 0; ps < 4; ps++) {
                    double sum = 0.0;
                    for (int cs = 0; cs < 4; cs++) {
                        double prod = P[ps][cs];
                        for (int child_id : tree.node_children[child]) {
                            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                                if (tree.edge_child_nodes[e] == child_id) {
                                    prod *= edge_messages[e][cs];
                                    break;
                                }
                            }
                        }
                        sum += prod;
                    }
                    msg[ps] = sum;
                }
            }

            // Scale to prevent underflow
            double mx = *max_element(msg.begin(), msg.end());
            if (mx > 0.0) {
                for (int i = 0; i < 4; i++) msg[i] /= mx;
                log_scales[eid] = log(mx);
            }

            edge_messages[eid] = msg;
        }

        // Combine at root
        array<double, 4> root_msg = {1.0, 1.0, 1.0, 1.0};
        double root_log_scale = 0.0;

        for (int eid : tree.root_edge_ids) {
            for (int i = 0; i < 4; i++) {
                root_msg[i] *= edge_messages[eid][i];
            }
            root_log_scale += log_scales[eid];

            // Add accumulated scales from children
            int child = tree.edge_child_nodes[eid];
            function<double(int)> sum_scales = [&](int node_id) -> double {
                if (tree.node_is_leaf[node_id]) return 0.0;
                double s = 0.0;
                for (int ch : tree.node_children[node_id]) {
                    for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                        if (tree.edge_child_nodes[e] == ch) {
                            s += log_scales[e] + sum_scales(ch);
                            break;
                        }
                    }
                }
                return s;
            };
            root_log_scale += sum_scales(child);
        }

        double ll = 0.0;
        for (int i = 0; i < 4; i++) {
            ll += tree.root_probabilities[i] * root_msg[i];
        }

        total_ll += tree.pattern_weights[p] * (log(ll) + root_log_scale);
    }

    return total_ll;
}

// Main compute function - uses CPU for accuracy in BH invariance testing
double compute_log_likelihood(const BHTree& tree) {
    return compute_log_likelihood_cpu(tree);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " -p <bh_params_file> -t <patterns_file> -x <taxon_order> [-s seed]" << endl;
        cerr << "Tests BH invariance by comparing log-likelihood at original and alternate root" << endl;
        return 1;
    }

    string params_file, pattern_file, taxon_file;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) params_file = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) pattern_file = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_file = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
    }

    if (params_file.empty() || pattern_file.empty() || taxon_file.empty()) {
        cerr << "Missing required arguments" << endl;
        return 1;
    }

    cout << "=== BH Invariance Test ===" << endl;
    cout << "BH parameters: " << params_file << endl;
    cout << "Patterns: " << pattern_file << endl;
    cout << "Taxon order: " << taxon_file << endl;
    cout << "Random seed: " << seed << endl;

    // Load BH parameters and patterns
    BHTree tree = load_bh_parameters(params_file);
    load_patterns(tree, pattern_file, taxon_file);

    string original_root_name = tree.node_names[tree.root_node_id];

    // Compute log-likelihood at original root
    cout << "\n=== Computing LL at original root: " << original_root_name << " ===" << endl;
    double ll_original_cpu = compute_log_likelihood_cpu(tree);
    double ll_original_gpu = compute_log_likelihood_gpu(tree);
    cout << fixed << setprecision(14);
    cout << "Log-likelihood (CPU) at " << original_root_name << ": " << ll_original_cpu << endl;
    cout << "Log-likelihood (GPU) at " << original_root_name << ": " << ll_original_gpu << endl;
    cout << "CPU-GPU difference: " << scientific << abs(ll_original_cpu - ll_original_gpu) << endl;
    double ll_original = ll_original_cpu;  // Use CPU for comparison

    // Select random alternate root (different from original, must be internal node)
    mt19937 rng(seed);
    vector<int> candidate_nodes;
    for (size_t i = 0; i < tree.node_names.size(); i++) {
        if ((int)i != tree.root_node_id && !tree.node_is_leaf[i]) {
            candidate_nodes.push_back(i);
        }
    }

    if (candidate_nodes.empty()) {
        cerr << "Error: No internal nodes available for re-rooting" << endl;
        return 1;
    }

    uniform_int_distribution<int> dist(0, candidate_nodes.size() - 1);
    int new_root_id = candidate_nodes[dist(rng)];
    string new_root_name = tree.node_names[new_root_id];

    cout << "\n=== Re-rooting tree at: " << new_root_name << " using Bayes rule ===" << endl;

    // Reparameterize using Bayes rule
    reparameterize_bh(tree);

    // Re-root tree
    reroot_tree(tree, new_root_id);

    // Compute log-likelihood at new root
    double ll_rerooted_cpu = compute_log_likelihood_cpu(tree);
    double ll_rerooted_gpu = compute_log_likelihood_gpu(tree);
    cout << "Log-likelihood (CPU) at " << new_root_name << ": " << ll_rerooted_cpu << endl;
    cout << "Log-likelihood (GPU) at " << new_root_name << ": " << ll_rerooted_gpu << endl;
    cout << "CPU-GPU difference: " << scientific << abs(ll_rerooted_cpu - ll_rerooted_gpu) << endl;
    double ll_rerooted = ll_rerooted_cpu;

    // Compare
    double diff = abs(ll_original - ll_rerooted);
    double rel_diff = diff / abs(ll_original);

    cout << "\n=== BH Invariance Results ===" << endl;
    cout << "Original root: " << original_root_name << ", LL = " << ll_original << endl;
    cout << "Alternate root: " << new_root_name << ", LL = " << ll_rerooted << endl;
    cout << "Absolute difference: " << scientific << diff << endl;
    cout << "Relative difference: " << rel_diff << endl;

    if (diff < 1e-8) {
        cout << "\nBH INVARIANCE VERIFIED: Log-likelihoods match within numerical precision!" << endl;
    } else if (rel_diff < 1e-10) {
        cout << "\nBH INVARIANCE VERIFIED: Relative difference is negligible." << endl;
    } else {
        cout << "\nWARNING: Log-likelihoods differ significantly! BH invariance may not hold." << endl;
    }

    return 0;
}
