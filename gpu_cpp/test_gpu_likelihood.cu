// Test program for GPULikelihoodComputer library
// Validates against CPU reference implementation

#include "gpu_likelihood.cuh"
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

using namespace std;

struct TestTree {
    vector<string> node_names;
    map<string, int> name_to_id;
    vector<bool> node_is_leaf;
    vector<int> leaf_taxon_indices;
    vector<vector<int>> node_children;
    int root_node_id;

    vector<int> edge_child_nodes;
    vector<double> branch_lengths;
    vector<int> post_order_edges;
    vector<int> root_edge_ids;

    int num_taxa;
    int num_patterns;
    vector<uint8_t> pattern_bases;
    vector<int> pattern_weights;
};

TestTree load_test_data(const string& tree_file, const string& pattern_file,
                         const string& taxon_order_file) {
    TestTree tree;

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
    vector<pair<string, string>> edge_pairs;
    vector<double> edge_lengths;
    ifstream tree_in(tree_file);
    while (getline(tree_in, line)) {
        istringstream iss(line);
        string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;
        edge_pairs.push_back({parent_name, child_name});
        edge_lengths.push_back(branch_length);
    }
    tree_in.close();

    cout << "Loaded " << edge_pairs.size() << " edges" << endl;

    // Build node structure
    set<string> all_names;
    for (const auto& e : edge_pairs) {
        all_names.insert(e.first);
        all_names.insert(e.second);
    }

    int node_id = 0;
    for (const string& name : all_names) {
        tree.node_names.push_back(name);
        tree.name_to_id[name] = node_id;
        tree.node_is_leaf.push_back(true);
        tree.leaf_taxon_indices.push_back(-1);
        tree.node_children.push_back(vector<int>());
        node_id++;
    }

    // Set parent-child relationships
    tree.edge_child_nodes.resize(edge_pairs.size());
    tree.branch_lengths.resize(edge_pairs.size());

    for (size_t e = 0; e < edge_pairs.size(); e++) {
        int parent_id = tree.name_to_id[edge_pairs[e].first];
        int child_id = tree.name_to_id[edge_pairs[e].second];
        tree.edge_child_nodes[e] = child_id;
        tree.branch_lengths[e] = edge_lengths[e];
        tree.node_children[parent_id].push_back(child_id);
        tree.node_is_leaf[parent_id] = false;
    }

    // Mark leaves and assign taxon indices
    map<string, int> taxon_name_to_index;
    for (int i = 0; i < tree.num_taxa; i++) {
        taxon_name_to_index[taxon_names[i]] = i;
    }

    for (int n = 0; n < (int)tree.node_names.size(); n++) {
        if (tree.node_is_leaf[n]) {
            auto it = taxon_name_to_index.find(tree.node_names[n]);
            if (it != taxon_name_to_index.end()) {
                tree.leaf_taxon_indices[n] = it->second;
            }
        }
    }

    // Find root (node with no parent)
    set<int> has_parent;
    for (int child : tree.edge_child_nodes) {
        has_parent.insert(child);
    }
    tree.root_node_id = -1;
    for (int n = 0; n < (int)tree.node_names.size(); n++) {
        if (has_parent.find(n) == has_parent.end()) {
            tree.root_node_id = n;
            break;
        }
    }

    cout << "Root node: " << tree.node_names[tree.root_node_id] << endl;

    // Find root edges
    for (int child_node_id : tree.node_children[tree.root_node_id]) {
        for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
            if (tree.edge_child_nodes[e] == child_node_id) {
                tree.root_edge_ids.push_back(e);
                break;
            }
        }
    }

    cout << "Root edges: " << tree.root_edge_ids.size() << endl;

    // Compute post-order traversal
    function<void(int)> compute_post_order = [&](int node_id) {
        for (int child_node_id : tree.node_children[node_id]) {
            compute_post_order(child_node_id);
            // Find edge to this child
            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                if (tree.edge_child_nodes[e] == child_node_id) {
                    tree.post_order_edges.push_back(e);
                    break;
                }
            }
        }
    };
    compute_post_order(tree.root_node_id);

    cout << "Post-order edges: " << tree.post_order_edges.size() << endl;

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

// CPU reference implementation
double compute_ll_cpu_ref(const TestTree& tree, const array<double, 4>& root_prob) {
    auto start = chrono::high_resolution_clock::now();

    // Precompute transition matrices
    vector<array<array<double, 4>, 4>> trans_matrices(tree.edge_child_nodes.size());
    for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
        double t = tree.branch_lengths[e];
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
        vector<array<double, 4>> edge_messages(tree.edge_child_nodes.size());
        vector<double> edge_log_scales(tree.edge_child_nodes.size(), 0.0);

        for (int edge_id : tree.post_order_edges) {
            int child_node = tree.edge_child_nodes[edge_id];
            const auto& P = trans_matrices[edge_id];

            array<double, 4> message = {1.0, 1.0, 1.0, 1.0};

            if (tree.node_is_leaf[child_node]) {
                int taxon_idx = tree.leaf_taxon_indices[child_node];
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
                        for (int child_node_id : tree.node_children[child_node]) {
                            // Find edge to this child
                            for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
                                if (tree.edge_child_nodes[e] == child_node_id) {
                                    prod *= edge_messages[e][child_state];
                                    break;
                                }
                            }
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

        for (size_t e = 0; e < tree.edge_child_nodes.size(); e++) {
            total_log_scale += edge_log_scales[e];
        }

        for (int root_edge : tree.root_edge_ids) {
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

    cout << "=== GPU Likelihood Library Test ===" << endl;

    if (!GPULikelihoodComputer::IsGPUAvailable()) {
        cerr << "No GPU available!" << endl;
        return 1;
    }
    cout << "GPU available: Yes" << endl;

    // Load data
    TestTree tree = load_test_data(tree_file, pattern_file, taxon_order_file);

    // CPU reference
    array<double, 4> root_prob = {0.25, 0.25, 0.25, 0.25};
    cout << "\nComputing CPU reference..." << endl;
    double cpu_ll = compute_ll_cpu_ref(tree, root_prob);
    cout << "CPU Log-likelihood: " << cpu_ll << endl;

    // GPU computation
    cout << "\nInitializing GPU..." << endl;
    GPULikelihoodComputer gpu_computer;

    bool init_ok = gpu_computer.InitializeTree(
        tree.node_names.size(),
        tree.edge_child_nodes.size(),
        tree.num_taxa,
        tree.edge_child_nodes,
        tree.branch_lengths,
        tree.node_is_leaf,
        tree.leaf_taxon_indices,
        tree.node_children,
        tree.post_order_edges,
        tree.root_edge_ids
    );

    if (!init_ok) {
        cerr << "Failed to initialize GPU tree!" << endl;
        return 1;
    }

    gpu_computer.SetRootProbabilities(root_prob);

    cout << "Computing GPU log-likelihood..." << endl;
    auto start = chrono::high_resolution_clock::now();

    double gpu_ll = gpu_computer.ComputeLogLikelihoodFlat(
        tree.num_patterns,
        tree.pattern_bases.data(),
        tree.pattern_weights.data()
    );

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "GPU computation time: " << duration.count() / 1000.0 << " ms" << endl;

    cout << "GPU Log-likelihood: " << gpu_ll << endl;
    cout << "Difference: " << abs(gpu_ll - cpu_ll) << endl;

    size_t gpu_mem = gpu_computer.GetGPUMemoryUsage();
    cout << "GPU memory usage: " << gpu_mem / (1024.0 * 1024.0) << " MB" << endl;

    if (abs(gpu_ll - cpu_ll) < 1e-6) {
        cout << "\nVALIDATION PASSED: GPU matches CPU!" << endl;
    } else {
        cout << "\nVALIDATION FAILED: Results differ!" << endl;
    }

    // Test multiple calls (branch length update)
    cout << "\nTesting branch length update..." << endl;

    // Modify branch lengths
    vector<double> new_branch_lengths = tree.branch_lengths;
    for (auto& bl : new_branch_lengths) {
        bl *= 1.1;  // Increase by 10%
    }

    gpu_computer.SetBranchLengths(new_branch_lengths);

    start = chrono::high_resolution_clock::now();
    double gpu_ll2 = gpu_computer.ComputeLogLikelihoodFlat(
        tree.num_patterns,
        tree.pattern_bases.data(),
        tree.pattern_weights.data()
    );
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "GPU Log-likelihood (modified branches): " << gpu_ll2 << endl;
    cout << "GPU computation time: " << duration.count() / 1000.0 << " ms" << endl;

    cout << "\n=== Test Complete ===" << endl;
    return 0;
}
