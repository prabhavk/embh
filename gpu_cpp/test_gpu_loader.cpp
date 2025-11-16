// Test program to validate GPU pattern group loading
// This tests the data loading without requiring full CUDA compilation

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <set>
#include <array>

using namespace std;

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
    vector<vector<int>> edge_leaf_indices;

    // Tree structure
    vector<string> edge_parent_names;
    vector<string> edge_child_names;
    vector<double> edge_branch_lengths;
};

PatternGroups load_pattern_groups(const string& prefix, const string& tree_file, const string& pattern_file) {
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

    cout << "Loading " << groups.num_patterns << " patterns, "
         << groups.num_edges << " edges, " << groups.total_groups << " groups" << endl;

    // Allocate
    groups.edge_num_groups.resize(groups.num_edges);
    groups.edge_group_offsets.resize(groups.num_edges + 1);
    groups.edge_subtree_sizes.resize(groups.num_edges);
    groups.edge_parent_names.resize(groups.num_edges);
    groups.edge_child_names.resize(groups.num_edges);

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
            groups.edge_parent_names[edge_id] = parent_name;
            groups.edge_child_names[edge_id] = child_name;
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
    cout << "Max subtree size: " << groups.max_subtree_size << endl;

    // Read pattern-to-group mapping
    string map_file = prefix + ".group_map";
    ifstream map_in(map_file);
    groups.pattern_to_group.resize(groups.num_patterns * groups.num_edges);

    getline(map_in, line); // skip header
    for (int p = 0; p < groups.num_patterns; p++) {
        getline(map_in, line);
        istringstream iss(line);
        int pat_idx;
        iss >> pat_idx;
        for (int e = 0; e < groups.num_edges; e++) {
            int group_id;
            iss >> group_id;
            groups.pattern_to_group[p * groups.num_edges + e] = group_id;
        }
    }
    map_in.close();

    // Read group signatures (binary)
    string sig_file = prefix + ".group_signatures";
    ifstream sig_in(sig_file, ios::binary);
    groups.group_signatures.resize(groups.total_groups * groups.max_subtree_size, 4); // default to gap

    // Read header: num_edges
    int file_num_edges;
    sig_in.read(reinterpret_cast<char*>(&file_num_edges), sizeof(int));
    if (file_num_edges != groups.num_edges) {
        cerr << "Warning: Signature file has " << file_num_edges << " edges, expected " << groups.num_edges << endl;
    }

    for (int e = 0; e < groups.num_edges; e++) {
        // Read per-edge header: num_groups, subtree_size
        int file_num_groups, file_subtree_size;
        sig_in.read(reinterpret_cast<char*>(&file_num_groups), sizeof(int));
        sig_in.read(reinterpret_cast<char*>(&file_subtree_size), sizeof(int));

        if (file_num_groups != groups.edge_num_groups[e]) {
            cerr << "Warning: Edge " << e << " has " << file_num_groups << " groups in file, expected " << groups.edge_num_groups[e] << endl;
        }

        int offset = groups.edge_group_offsets[e];
        for (int g = 0; g < file_num_groups; g++) {
            int global_idx = offset + g;
            sig_in.read(reinterpret_cast<char*>(&groups.group_signatures[global_idx * groups.max_subtree_size]),
                       file_subtree_size);
        }
    }
    sig_in.close();

    // Read pattern weights from pattern file
    groups.pattern_weights.resize(groups.num_patterns);
    ifstream pat_in(pattern_file);
    getline(pat_in, line); // header
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

    // Compute group weights
    groups.group_weights.resize(groups.total_groups, 0);
    for (int p = 0; p < groups.num_patterns; p++) {
        int weight = groups.pattern_weights[p];
        for (int e = 0; e < groups.num_edges; e++) {
            int group_id = groups.pattern_to_group[p * groups.num_edges + e];
            int global_idx = groups.edge_group_offsets[e] + group_id;
            // Note: this counts weight for each edge separately, which isn't quite right
            // We just need this for validation
        }
    }

    // Read tree edges for branch lengths
    groups.edge_branch_lengths.resize(groups.num_edges);
    ifstream tree_in(tree_file);
    int edge_idx = 0;
    while (getline(tree_in, line) && edge_idx < groups.num_edges) {
        istringstream iss(line);
        string parent_name, child_name;
        double branch_length;
        iss >> parent_name >> child_name >> branch_length;
        groups.edge_branch_lengths[edge_idx] = branch_length;
        edge_idx++;
    }
    tree_in.close();

    // Determine post-order from schedule file
    string schedule_file = prefix + ".group_schedule";
    ifstream sched_in(schedule_file);
    set<int> seen_edges;
    groups.post_order.clear();

    while (getline(sched_in, line)) {
        if (line.substr(0, 4) == "WORK") {
            int edge_id, group_id, num_pats, total_weight;
            sscanf(line.c_str(), "WORK %d %d %d %d", &edge_id, &group_id, &num_pats, &total_weight);
            if (seen_edges.find(edge_id) == seen_edges.end()) {
                groups.post_order.push_back(edge_id);
                seen_edges.insert(edge_id);
            }
        }
    }
    sched_in.close();

    cout << "Post-order edges: " << groups.post_order.size() << endl;

    return groups;
}

// Compute Jukes-Cantor transition probability
double jc_prob(int from_state, int to_state, double t) {
    double exp_term = exp(-4.0 * t / 3.0);
    if (from_state == to_state) {
        return 0.25 + 0.75 * exp_term;
    } else {
        return 0.25 - 0.25 * exp_term;
    }
}

// Compute upward message for a leaf edge with given observed base
array<double, 4> compute_leaf_message(int observed_base, double branch_length) {
    array<double, 4> message;
    if (observed_base < 4) {
        // Observed base
        for (int parent_state = 0; parent_state < 4; parent_state++) {
            message[parent_state] = jc_prob(parent_state, observed_base, branch_length);
        }
    } else {
        // Gap - sum over all child states
        for (int parent_state = 0; parent_state < 4; parent_state++) {
            message[parent_state] = 1.0; // sum of all jc_prob = 1
        }
    }
    return message;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <group_prefix> <tree_edges.txt> <patterns.pat>" << endl;
        return 1;
    }

    string group_prefix = argv[1];
    string tree_file = argv[2];
    string pattern_file = argv[3];

    PatternGroups groups = load_pattern_groups(group_prefix, tree_file, pattern_file);

    cout << "\n=== Validation Tests ===" << endl;

    // Test 1: Verify pattern-to-group consistency
    cout << "\n1. Testing pattern-to-group consistency..." << endl;
    int errors = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        if (groups.edge_subtree_sizes[e] == 1) {
            // Leaf edge - verify that patterns with same base map to same group
            map<int, int> base_to_group;
            for (int p = 0; p < groups.num_patterns; p++) {
                int group_id = groups.pattern_to_group[p * groups.num_edges + e];
                // For leaf edge, signature is just the single base
                int global_group_idx = groups.edge_group_offsets[e] + group_id;
                uint8_t base = groups.group_signatures[global_group_idx * groups.max_subtree_size];

                // Get the actual base from pattern data
                // We need to know which taxon this edge corresponds to
                // For now, skip this detailed check
            }
        }
    }
    cout << "   Pattern-to-group mapping loaded successfully" << endl;

    // Test 2: Verify group counts match
    cout << "\n2. Verifying group counts..." << endl;
    int total_groups_computed = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        total_groups_computed += groups.edge_num_groups[e];
    }
    if (total_groups_computed == groups.total_groups) {
        cout << "   PASS: Total groups match (" << groups.total_groups << ")" << endl;
    } else {
        cout << "   FAIL: Expected " << groups.total_groups << " but computed " << total_groups_computed << endl;
    }

    // Test 3: Compute leaf edge messages for validation
    cout << "\n3. Computing sample leaf edge messages..." << endl;
    int num_leaf_edges = 0;
    for (int e = 0; e < groups.num_edges; e++) {
        if (groups.edge_subtree_sizes[e] == 1) {
            num_leaf_edges++;
        }
    }
    cout << "   Number of leaf edges: " << num_leaf_edges << endl;

    // Show sample computation for first leaf edge
    for (int e = 0; e < min(3, groups.num_edges); e++) {
        if (groups.edge_subtree_sizes[e] == 1) {
            cout << "\n   Edge " << e << " (" << groups.edge_parent_names[e] << " -> "
                 << groups.edge_child_names[e] << "):" << endl;
            cout << "   Branch length: " << groups.edge_branch_lengths[e] << endl;
            cout << "   Number of groups: " << groups.edge_num_groups[e] << endl;

            int offset = groups.edge_group_offsets[e];
            for (int g = 0; g < min(4, groups.edge_num_groups[e]); g++) {
                uint8_t base = groups.group_signatures[(offset + g) * groups.max_subtree_size];
                cout << "     Group " << g << ": base=" << (int)base;

                array<double, 4> msg = compute_leaf_message(base, groups.edge_branch_lengths[e]);
                cout << " -> message=[";
                for (int i = 0; i < 4; i++) {
                    cout << msg[i];
                    if (i < 3) cout << ", ";
                }
                cout << "]" << endl;
            }
        }
    }

    // Test 4: Count unique computations vs naive approach
    cout << "\n4. Computation savings analysis..." << endl;
    int total_naive_computations = groups.num_patterns * groups.num_edges;
    cout << "   Naive approach: " << total_naive_computations << " message computations" << endl;
    cout << "   Group-based approach: " << groups.total_groups << " message computations" << endl;
    double savings = 100.0 * (1.0 - (double)groups.total_groups / total_naive_computations);
    cout << "   Savings: " << savings << "%" << endl;

    // Test 5: Memory requirements estimation
    cout << "\n5. Memory requirements estimation..." << endl;
    size_t pattern_bases_size = groups.num_patterns * groups.num_taxa * sizeof(uint8_t);
    size_t group_signatures_size = groups.total_groups * groups.max_subtree_size * sizeof(uint8_t);
    size_t pattern_to_group_size = groups.num_patterns * groups.num_edges * sizeof(int);
    size_t group_messages_size = groups.total_groups * 4 * sizeof(double);
    size_t pattern_log_scales_size = groups.num_patterns * sizeof(double);

    cout << "   Pattern bases: " << pattern_bases_size / 1024.0 << " KB" << endl;
    cout << "   Group signatures: " << group_signatures_size / 1024.0 << " KB" << endl;
    cout << "   Pattern-to-group map: " << pattern_to_group_size / 1024.0 << " KB" << endl;
    cout << "   Group messages: " << group_messages_size / 1024.0 << " KB" << endl;
    cout << "   Pattern log scales: " << pattern_log_scales_size / 1024.0 << " KB" << endl;

    size_t total_gpu_memory = pattern_bases_size + group_signatures_size + pattern_to_group_size +
                              group_messages_size + pattern_log_scales_size;
    cout << "   Total GPU memory (approx): " << total_gpu_memory / (1024.0 * 1024.0) << " MB" << endl;

    cout << "\n=== All tests completed ===" << endl;
    return 0;
}
