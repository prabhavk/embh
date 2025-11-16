// CPU reference implementation of grouped log-likelihood computation
// Uses pattern grouping for message reuse optimization
// This validates the approach before GPU implementation

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <set>
#include <array>
#include <chrono>
#include <cassert>
#include <functional>
#include <iomanip>
#include <utility>

using namespace std;

// Tree node structure
struct TreeNode {
    string name;
    int id;
    int parent_id;
    vector<int> child_ids;
    int edge_id;  // Edge from parent to this node
    bool is_leaf;
    int taxon_index;  // For leaves: index into pattern
};

// Tree edge structure
struct TreeEdge {
    int id;
    int parent_node;
    int child_node;
    string parent_name;
    string child_name;
    double branch_length;
    int subtree_size;
    vector<int> subtree_leaf_taxon_indices;  // Taxon indices in subtree
};

struct PatternGroups {
    int num_edges;
    int num_patterns;
    int num_taxa;
    int total_groups;
    int max_subtree_size;

    vector<int> edge_num_groups;
    vector<int> edge_group_offsets;
    vector<int> edge_subtree_sizes;
    vector<uint8_t> group_signatures;
    vector<int> pattern_to_group;
    vector<int> group_weights;
    vector<int> pattern_weights;
    vector<uint8_t> pattern_bases;
    vector<int> post_order;

    // Tree structure
    vector<TreeEdge> edges;
    vector<TreeNode> nodes;
    map<string, int> name_to_node;
    int root_node_id;
};

// Jukes-Cantor transition probability P(child_state | parent_state, t)
double jc_prob(int parent_state, int child_state, double t) {
    double exp_term = exp(-4.0 * t / 3.0);
    if (parent_state == child_state) {
        return 0.25 + 0.75 * exp_term;
    } else {
        return 0.25 - 0.25 * exp_term;
    }
}

// 4x4 transition matrix
using TransMatrix = array<array<double, 4>, 4>;

TransMatrix compute_jc_matrix(double t) {
    TransMatrix P;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            P[i][j] = jc_prob(i, j, t);
        }
    }
    return P;
}

PatternGroups load_pattern_groups(const string& prefix, const string& tree_file,
                                   const string& pattern_file, const string& taxon_order_file) {
    PatternGroups groups;

    // Read group index file
    string index_file = prefix + ".group_index";
    ifstream idx_in(index_file);
    if (!idx_in.is_open()) {
        cerr << "Error: Cannot open " << index_file << endl;
        exit(1);
    }

    string line;
    while (getline(idx_in, line)) {
        if (line.find("NUM_PATTERNS") != string::npos) {
            sscanf(line.c_str(), "NUM_PATTERNS %d", &groups.num_patterns);
        } else if (line.find("NUM_EDGES") != string::npos && line.find("TOTAL") == string::npos) {
            sscanf(line.c_str(), "NUM_EDGES %d", &groups.num_edges);
        } else if (line.find("TOTAL_GROUPS") != string::npos) {
            sscanf(line.c_str(), "TOTAL_GROUPS %d", &groups.total_groups);
        }
    }
    idx_in.close();

    // Allocate
    groups.edge_num_groups.resize(groups.num_edges);
    groups.edge_group_offsets.resize(groups.num_edges + 1);
    groups.edge_subtree_sizes.resize(groups.num_edges);
    groups.edges.resize(groups.num_edges);

    // Re-read index file for edge info
    idx_in.open(index_file);
    int cumulative_offset = 0;
    while (getline(idx_in, line)) {
        if (line.substr(0, 4) == "EDGE") {
            int edge_id, subtree_size, num_groups;
            char parent_name[64], child_name[64];
            sscanf(line.c_str(), "EDGE %d %s %s %d %d",
                   &edge_id, parent_name, child_name, &subtree_size, &num_groups);
            groups.edge_num_groups[edge_id] = num_groups;
            groups.edge_subtree_sizes[edge_id] = subtree_size;
            groups.edge_group_offsets[edge_id] = cumulative_offset;
            groups.edges[edge_id].id = edge_id;
            groups.edges[edge_id].parent_name = parent_name;
            groups.edges[edge_id].child_name = child_name;
            groups.edges[edge_id].subtree_size = subtree_size;
            cumulative_offset += num_groups;
        }
    }
    groups.edge_group_offsets[groups.num_edges] = cumulative_offset;
    idx_in.close();

    // Find max subtree size
    groups.max_subtree_size = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        if (groups.edge_subtree_sizes[e] > groups.max_subtree_size) {
            groups.max_subtree_size = groups.edge_subtree_sizes[e];
        }
    }

    // Read pattern-to-group mapping (binary file)
    string map_file = prefix + ".group_map";
    ifstream map_in(map_file, ios::binary);
    groups.pattern_to_group.resize(groups.num_patterns * groups.num_edges);

    // Read header
    int file_num_patterns, file_num_edges;
    map_in.read(reinterpret_cast<char*>(&file_num_patterns), sizeof(int));
    map_in.read(reinterpret_cast<char*>(&file_num_edges), sizeof(int));

    if (file_num_patterns != groups.num_patterns || file_num_edges != groups.num_edges) {
        cerr << "Warning: Map file has " << file_num_patterns << " patterns and "
             << file_num_edges << " edges, expected " << groups.num_patterns << " and " << groups.num_edges << endl;
    }

    // Data is organized by edge: for each edge, read all pattern->group mappings
    vector<int> temp_map(groups.num_patterns);
    for (int e = 0; e < groups.num_edges; e++) {
        map_in.read(reinterpret_cast<char*>(temp_map.data()), groups.num_patterns * sizeof(int));
        // Transpose to pattern-major order
        for (int p = 0; p < groups.num_patterns; p++) {
            groups.pattern_to_group[p * groups.num_edges + e] = temp_map[p];
        }
    }
    map_in.close();

    // Read group signatures (binary)
    string sig_file = prefix + ".group_signatures";
    ifstream sig_in(sig_file, ios::binary);
    groups.group_signatures.resize(groups.total_groups * groups.max_subtree_size, 4);

    int sig_file_num_edges;
    sig_in.read(reinterpret_cast<char*>(&sig_file_num_edges), sizeof(int));

    for (int e = 0; e < groups.num_edges; e++) {
        int file_num_groups, file_subtree_size;
        sig_in.read(reinterpret_cast<char*>(&file_num_groups), sizeof(int));
        sig_in.read(reinterpret_cast<char*>(&file_subtree_size), sizeof(int));

        int offset = groups.edge_group_offsets[e];
        for (int g = 0; g < file_num_groups; g++) {
            int global_idx = offset + g;
            sig_in.read(reinterpret_cast<char*>(&groups.group_signatures[global_idx * groups.max_subtree_size]),
                       file_subtree_size);
        }
    }
    sig_in.close();

    // Read pattern weights and bases
    groups.pattern_weights.resize(groups.num_patterns);
    ifstream pat_in(pattern_file);
    getline(pat_in, line);
    int file_patterns, file_taxa;
    sscanf(line.c_str(), "%d %d", &file_patterns, &file_taxa);
    groups.num_taxa = file_taxa;

    groups.pattern_bases.resize(groups.num_patterns * groups.num_taxa);
    for (int p = 0; p < groups.num_patterns; p++) {
        getline(pat_in, line);
        istringstream iss(line);
        int weight;
        iss >> weight;
        groups.pattern_weights[p] = weight;
        for (int t = 0; t < groups.num_taxa; t++) {
            int base;
            iss >> base;
            groups.pattern_bases[p * groups.num_taxa + t] = (uint8_t)base;
        }
    }
    pat_in.close();

    // Read taxon order to get names (CSV format with header)
    vector<string> taxon_names;
    ifstream tax_in(taxon_order_file);
    bool first_line = true;
    while (getline(tax_in, line)) {
        if (first_line) {
            first_line = false;
            continue; // skip header
        }
        if (!line.empty() && line[0] != '#') {
            // Parse CSV: name,position
            size_t comma_pos = line.find(',');
            string name = (comma_pos != string::npos) ? line.substr(0, comma_pos) : line;
            // Trim whitespace
            while (!name.empty() && isspace(name.back())) name.pop_back();
            while (!name.empty() && isspace(name.front())) name.erase(0, 1);
            if (!name.empty()) {
                taxon_names.push_back(name);
            }
        }
    }
    tax_in.close();

    cout << "Loaded " << taxon_names.size() << " taxon names" << endl;

    // Build tree structure from edges file
    ifstream tree_in(tree_file);
    int edge_idx = 0;
    set<string> all_names;
    while (getline(tree_in, line) && edge_idx < groups.num_edges) {
        istringstream iss(line);
        string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;
        groups.edges[edge_idx].branch_length = branch_length;
        all_names.insert(parent_name);
        all_names.insert(child_name);
        edge_idx++;
    }
    tree_in.close();

    // Create nodes
    int node_id = 0;
    for (const string& name : all_names) {
        TreeNode node;
        node.name = name;
        node.id = node_id;
        node.parent_id = -1;
        node.edge_id = -1;
        node.is_leaf = false;
        node.taxon_index = -1;

        // Check if this is a leaf (in taxon list)
        for (int t = 0; t < (int)taxon_names.size(); t++) {
            if (taxon_names[t] == name) {
                node.is_leaf = true;
                node.taxon_index = t;
                break;
            }
        }

        groups.name_to_node[name] = node_id;
        groups.nodes.push_back(node);
        node_id++;
    }

    // Set parent-child relationships from edges
    for (int e = 0; e < groups.num_edges; e++) {
        int parent_id = groups.name_to_node[groups.edges[e].parent_name];
        int child_id = groups.name_to_node[groups.edges[e].child_name];

        groups.edges[e].parent_node = parent_id;
        groups.edges[e].child_node = child_id;

        groups.nodes[child_id].parent_id = parent_id;
        groups.nodes[child_id].edge_id = e;
        groups.nodes[parent_id].child_ids.push_back(child_id);
    }

    // Find root (node with no parent)
    for (auto& node : groups.nodes) {
        if (node.parent_id == -1) {
            groups.root_node_id = node.id;
            break;
        }
    }

    // Compute subtree leaves for each edge
    for (int e = 0; e < groups.num_edges; e++) {
        int child_node = groups.edges[e].child_node;
        vector<int>& leaves = groups.edges[e].subtree_leaf_taxon_indices;

        // BFS to find all leaves in subtree
        vector<int> stack;
        stack.push_back(child_node);
        while (!stack.empty()) {
            int n = stack.back();
            stack.pop_back();
            if (groups.nodes[n].is_leaf) {
                leaves.push_back(groups.nodes[n].taxon_index);
            } else {
                for (int c : groups.nodes[n].child_ids) {
                    stack.push_back(c);
                }
            }
        }
        sort(leaves.begin(), leaves.end());
    }

    // Compute post-order traversal of edges
    groups.post_order.clear();
    set<int> processed_edges;

    function<void(int)> dfs_postorder = [&](int node_id) {
        for (int child_id : groups.nodes[node_id].child_ids) {
            dfs_postorder(child_id);
        }
        int edge_id = groups.nodes[node_id].edge_id;
        if (edge_id >= 0 && processed_edges.find(edge_id) == processed_edges.end()) {
            groups.post_order.push_back(edge_id);
            processed_edges.insert(edge_id);
        }
    };
    dfs_postorder(groups.root_node_id);

    return groups;
}

// Compute upward message for a single group
// Returns: (message, log_scale for this edge only - NOT accumulated)
pair<array<double, 4>, double> compute_group_message(
    const PatternGroups& groups,
    int edge_id,
    int group_id,
    const vector<TransMatrix>& trans_matrices,
    const vector<vector<array<double, 4>>>& child_group_messages
) {
    array<double, 4> message = {1.0, 1.0, 1.0, 1.0};

    int child_node = groups.edges[edge_id].child_node;
    const TransMatrix& P = trans_matrices[edge_id];

    if (groups.nodes[child_node].is_leaf) {
        // Leaf edge: message is P(parent|observed child)
        int global_group_idx = groups.edge_group_offsets[edge_id] + group_id;
        uint8_t observed_base = groups.group_signatures[global_group_idx * groups.max_subtree_size];

        if (observed_base < 4) {
            // Observed base
            for (int parent_state = 0; parent_state < 4; parent_state++) {
                message[parent_state] = P[parent_state][observed_base];
            }
        }
        // else gap: message stays {1,1,1,1}
    } else {
        // Internal edge: combine messages from children
        // message[parent_state] = sum_{child_state} P[parent][child] * product of child messages

        // Get children edges
        vector<int> child_edges;
        for (int c : groups.nodes[child_node].child_ids) {
            child_edges.push_back(groups.nodes[c].edge_id);
        }

        // For this group, we need to determine which child groups correspond to this signature
        // The signature for this edge is the concatenation of subtree leaves
        // Each child edge has its own signature based on its subtree

        int global_group_idx = groups.edge_group_offsets[edge_id] + group_id;
        const uint8_t* full_sig = &groups.group_signatures[global_group_idx * groups.max_subtree_size];
        int subtree_size = groups.edge_subtree_sizes[edge_id];

        // Match sub-signatures to child groups
        vector<int> child_group_ids(child_edges.size());
        for (int ci = 0; ci < (int)child_edges.size(); ci++) {
            int ce = child_edges[ci];
            int child_subtree_size = groups.edge_subtree_sizes[ce];

            // Build expected signature for child from parent signature
            // Need to know which positions in parent signature correspond to child subtree
            vector<uint8_t> child_sig(child_subtree_size);

            // Map child leaves to parent signature positions
            const vector<int>& child_leaves = groups.edges[ce].subtree_leaf_taxon_indices;
            const vector<int>& parent_leaves = groups.edges[edge_id].subtree_leaf_taxon_indices;

            if (child_leaves.empty() || parent_leaves.empty()) {
                cerr << "Error: Empty leaves for edge " << edge_id << " or child " << ce << endl;
                exit(1);
            }

            for (int i = 0; i < child_subtree_size; i++) {
                int child_taxon = child_leaves[i];
                // Find position in parent leaves
                auto it = lower_bound(parent_leaves.begin(), parent_leaves.end(), child_taxon);
                if (it == parent_leaves.end() || *it != child_taxon) {
                    cerr << "Error: Child taxon " << child_taxon << " not found in parent leaves" << endl;
                    exit(1);
                }
                int parent_pos = it - parent_leaves.begin();
                if (parent_pos >= subtree_size) {
                    cerr << "Error: parent_pos " << parent_pos << " >= subtree_size " << subtree_size << endl;
                    exit(1);
                }
                child_sig[i] = full_sig[parent_pos];
            }

            // Find matching group in child edge
            int num_child_groups = groups.edge_num_groups[ce];
            int child_offset = groups.edge_group_offsets[ce];
            bool found = false;
            for (int cg = 0; cg < num_child_groups; cg++) {
                const uint8_t* candidate_sig = &groups.group_signatures[(child_offset + cg) * groups.max_subtree_size];
                bool match = true;
                for (int i = 0; i < child_subtree_size; i++) {
                    if (candidate_sig[i] != child_sig[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    child_group_ids[ci] = cg;
                    found = true;
                    break;
                }
            }
            if (!found) {
                cerr << "Error: Could not find matching child group for edge " << edge_id << " group " << group_id << endl;
                cerr << "Child edge " << ce << " (" << groups.edges[ce].parent_name << " -> " << groups.edges[ce].child_name << ")" << endl;
                cerr << "Expected signature: ";
                for (int i = 0; i < child_subtree_size; i++) {
                    cerr << (int)child_sig[i] << " ";
                }
                cerr << endl;
                exit(1);
            }
        }

        // Compute message by marginalizing over child state
        for (int parent_state = 0; parent_state < 4; parent_state++) {
            double sum = 0.0;
            for (int child_state = 0; child_state < 4; child_state++) {
                double prod = P[parent_state][child_state];
                for (int ci = 0; ci < (int)child_edges.size(); ci++) {
                    int ce = child_edges[ci];
                    prod *= child_group_messages[ce][child_group_ids[ci]][child_state];
                }
                sum += prod;
            }
            message[parent_state] = sum;
        }
    }

    // Scale message to avoid underflow and return scale factor
    double max_val = *max_element(message.begin(), message.end());
    double log_scale = 0.0;
    if (max_val > 0 && max_val != 1.0) {
        for (int i = 0; i < 4; i++) {
            message[i] /= max_val;
        }
        log_scale = log(max_val);
    }

    return make_pair(message, log_scale);
}

// Compute log-likelihood using grouped message passing
double compute_log_likelihood_grouped(
    const PatternGroups& groups,
    const array<double, 4>& root_prob
) {
    auto start = chrono::high_resolution_clock::now();

    // Compute transition matrices for all edges
    vector<TransMatrix> trans_matrices(groups.num_edges);
    for (int e = 0; e < groups.num_edges; e++) {
        trans_matrices[e] = compute_jc_matrix(groups.edges[e].branch_length);
    }

    // Compute messages for each group in post-order
    // child_group_messages[edge][group] = message vector [4]
    vector<vector<array<double, 4>>> group_messages(groups.num_edges);
    vector<vector<double>> group_log_scales(groups.num_edges);

    for (int e = 0; e < groups.num_edges; e++) {
        group_messages[e].resize(groups.edge_num_groups[e]);
        group_log_scales[e].resize(groups.edge_num_groups[e], 0.0);
    }

    int total_message_computations = 0;

    // Process edges in post-order (leaves to root)
    for (int edge_id : groups.post_order) {
        int num_groups = groups.edge_num_groups[edge_id];
        cout << "Processing edge " << edge_id << " (" << groups.edges[edge_id].parent_name << " -> "
             << groups.edges[edge_id].child_name << ") with " << num_groups << " groups" << endl;

        for (int g = 0; g < num_groups; g++) {
            try {
                auto result = compute_group_message(groups, edge_id, g, trans_matrices, group_messages);
                array<double, 4> msg = result.first;
                double log_scale = result.second;

                group_messages[edge_id][g] = msg;
                group_log_scales[edge_id][g] = log_scale;  // Just this edge's scale
                total_message_computations++;
            } catch (const exception& e) {
                cerr << "Error at edge " << edge_id << " group " << g << ": " << e.what() << endl;
                throw;
            }
        }
    }

    cout << "Total unique message computations: " << total_message_computations << endl;

    // Now compute log-likelihood for each pattern by looking up group messages
    double total_log_likelihood = 0.0;

    // Find root edges (edges from root node)
    vector<int> root_edges;
    for (int c : groups.nodes[groups.root_node_id].child_ids) {
        root_edges.push_back(groups.nodes[c].edge_id);
    }

    cout << "Root edges: ";
    for (int e : root_edges) {
        cout << e << " ";
    }
    cout << endl;

    cout << "Computing per-pattern log-likelihoods..." << endl;

    for (int p = 0; p < groups.num_patterns; p++) {
        if (p % 100 == 0) {
            cout << "  Pattern " << p << "/" << groups.num_patterns << endl;
        }

        // Accumulate log scales from ALL edges for this pattern
        double total_log_scale = 0.0;
        for (int e = 0; e < groups.num_edges; e++) {
            int group_id = groups.pattern_to_group[p * groups.num_edges + e];
            total_log_scale += group_log_scales[e][group_id];
        }

        // Combine messages from all root edges (these are the final upward messages)
        array<double, 4> combined_msg = {1.0, 1.0, 1.0, 1.0};

        for (int root_edge : root_edges) {
            int group_id = groups.pattern_to_group[p * groups.num_edges + root_edge];
            if (group_id < 0 || group_id >= (int)group_messages[root_edge].size()) {
                cerr << "Error: Invalid group_id " << group_id << " for edge " << root_edge
                     << " (size " << group_messages[root_edge].size() << ")" << endl;
                exit(1);
            }
            const array<double, 4>& msg = group_messages[root_edge][group_id];

            for (int i = 0; i < 4; i++) {
                combined_msg[i] *= msg[i];
            }
        }

        // Normalize combined message
        double max_val = *max_element(combined_msg.begin(), combined_msg.end());
        if (max_val > 0) {
            for (int i = 0; i < 4; i++) {
                combined_msg[i] /= max_val;
            }
            total_log_scale += log(max_val);
        }

        // Weight by root probability and marginalize
        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += root_prob[dna] * combined_msg[dna];
        }

        double site_log_likelihood = total_log_scale + log(site_likelihood);

        // Weight by pattern count
        total_log_likelihood += site_log_likelihood * groups.pattern_weights[p];
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Grouped computation time: " << duration.count() / 1000.0 << " ms" << endl;

    return total_log_likelihood;
}

// Naive computation (no grouping) for comparison
double compute_log_likelihood_naive(
    const PatternGroups& groups,
    const array<double, 4>& root_prob
) {
    auto start = chrono::high_resolution_clock::now();

    vector<TransMatrix> trans_matrices(groups.num_edges);
    for (int e = 0; e < groups.num_edges; e++) {
        trans_matrices[e] = compute_jc_matrix(groups.edges[e].branch_length);
    }

    int total_message_computations = 0;
    double total_log_likelihood = 0.0;

    // Find root edges
    vector<int> root_edges;
    for (int c : groups.nodes[groups.root_node_id].child_ids) {
        root_edges.push_back(groups.nodes[c].edge_id);
    }

    // Process each pattern independently
    for (int p = 0; p < groups.num_patterns; p++) {
        // Compute messages for this pattern
        vector<array<double, 4>> edge_messages(groups.num_edges);
        vector<double> edge_log_scales(groups.num_edges, 0.0);

        // Process in post-order
        for (int edge_id : groups.post_order) {
            int child_node = groups.edges[edge_id].child_node;
            const TransMatrix& P = trans_matrices[edge_id];

            array<double, 4> message = {1.0, 1.0, 1.0, 1.0};

            if (groups.nodes[child_node].is_leaf) {
                // Leaf edge
                int taxon_idx = groups.nodes[child_node].taxon_index;
                uint8_t observed_base = groups.pattern_bases[p * groups.num_taxa + taxon_idx];

                if (observed_base < 4) {
                    for (int parent_state = 0; parent_state < 4; parent_state++) {
                        message[parent_state] = P[parent_state][observed_base];
                    }
                }
            } else {
                // Internal edge
                vector<int> child_edges;
                for (int c : groups.nodes[child_node].child_ids) {
                    child_edges.push_back(groups.nodes[c].edge_id);
                }

                for (int parent_state = 0; parent_state < 4; parent_state++) {
                    double sum = 0.0;
                    for (int child_state = 0; child_state < 4; child_state++) {
                        double prod = P[parent_state][child_state];
                        for (int ce : child_edges) {
                            prod *= edge_messages[ce][child_state];
                        }
                        sum += prod;
                    }
                    message[parent_state] = sum;
                }
            }

            // Scale
            double max_val = *max_element(message.begin(), message.end());
            if (max_val > 0) {
                for (int i = 0; i < 4; i++) {
                    message[i] /= max_val;
                }
                edge_log_scales[edge_id] = log(max_val);
            }

            edge_messages[edge_id] = message;
            total_message_computations++;
        }

        // Combine root edge messages
        array<double, 4> combined_msg = {1.0, 1.0, 1.0, 1.0};
        double total_log_scale = 0.0;

        for (int root_edge : root_edges) {
            for (int i = 0; i < 4; i++) {
                combined_msg[i] *= edge_messages[root_edge][i];
            }
            total_log_scale += edge_log_scales[root_edge];
        }

        double max_val = *max_element(combined_msg.begin(), combined_msg.end());
        if (max_val > 0) {
            for (int i = 0; i < 4; i++) {
                combined_msg[i] /= max_val;
            }
            total_log_scale += log(max_val);
        }

        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += root_prob[dna] * combined_msg[dna];
        }

        double site_log_likelihood = total_log_scale + log(site_likelihood);
        total_log_likelihood += site_log_likelihood * groups.pattern_weights[p];
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Naive computation time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "Total message computations (naive): " << total_message_computations << endl;

    return total_log_likelihood;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <group_prefix> <tree_edges.txt> <patterns.pat> <taxon_order.csv>" << endl;
        return 1;
    }

    string group_prefix = argv[1];
    string tree_file = argv[2];
    string pattern_file = argv[3];
    string taxon_order_file = argv[4];

    cout << "Loading pattern groups..." << endl;
    PatternGroups groups = load_pattern_groups(group_prefix, tree_file, pattern_file, taxon_order_file);

    cout << "Loaded " << groups.num_patterns << " patterns, "
         << groups.num_edges << " edges, " << groups.total_groups << " groups" << endl;
    cout << "Tree has " << groups.nodes.size() << " nodes, root: "
         << groups.nodes[groups.root_node_id].name << endl;
    cout << "Post-order has " << groups.post_order.size() << " edges" << endl;

    // Debug: print first few edges and their subtree leaves
    cout << "\nEdge subtree leaves:" << endl;
    for (int e = 0; e < min(10, groups.num_edges); e++) {
        cout << "Edge " << e << " (" << groups.edges[e].parent_name << " -> " << groups.edges[e].child_name << "): ";
        for (int l : groups.edges[e].subtree_leaf_taxon_indices) {
            cout << l << " ";
        }
        cout << endl;
    }
    cout << endl;

    // Use uniform root probability
    array<double, 4> root_prob = {0.25, 0.25, 0.25, 0.25};

    cout << "\n=== Computing log-likelihood ===" << endl;

    cout << "\n--- Grouped approach ---" << endl;
    double ll_grouped = compute_log_likelihood_grouped(groups, root_prob);
    cout << "Log-likelihood (grouped): " << fixed << setprecision(6) << ll_grouped << endl;

    cout << "\n--- Naive approach (for comparison) ---" << endl;
    double ll_naive = compute_log_likelihood_naive(groups, root_prob);
    cout << "Log-likelihood (naive): " << fixed << setprecision(6) << ll_naive << endl;

    cout << "\n--- Comparison ---" << endl;
    double diff = abs(ll_grouped - ll_naive);
    cout << "Difference: " << scientific << diff << endl;
    if (diff < 1e-10) {
        cout << "PASS: Results match!" << endl;
    } else if (diff < 1e-6) {
        cout << "PASS: Results match within numerical precision" << endl;
    } else {
        cout << "FAIL: Results differ significantly" << endl;
    }

    // Debug: compare per-site log-likelihoods for first pattern
    cout << "\n--- Debug: First pattern analysis ---" << endl;
    cout << "Pattern 0 weight: " << groups.pattern_weights[0] << endl;

    // Count leaf vs internal edges
    int num_leaf_edges = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        if (groups.nodes[groups.edges[e].child_node].is_leaf) {
            num_leaf_edges++;
        }
    }
    cout << "Leaf edges: " << num_leaf_edges << " / " << groups.num_edges << endl;

    double speedup = 1.0; // Would need timing info
    int naive_computations = groups.num_patterns * groups.num_edges;
    cout << "\nComputation savings: " << groups.total_groups << " vs " << naive_computations
         << " (" << 100.0 * (1.0 - (double)groups.total_groups / naive_computations) << "% reduction)" << endl;

    return 0;
}
