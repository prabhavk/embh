// pattern_groups_precompute.cpp
// Pre-compute pattern groupings by subtree signature for GPU message reuse
//
// This tool analyzes patterns and tree structure to group patterns that share
// the same subtree signature at each edge. This enables computing each unique
// message only ONCE on the GPU, then broadcasting to all patterns with that signature.
//
// Compile: g++ -O3 -std=c++11 pattern_groups_precompute.cpp -o pattern_groups_precompute
// Usage:   ./pattern_groups_precompute <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]
//
// Output files:
//   <prefix>.group_index   - Summary of groups per edge
//   <prefix>.group_map     - Binary: pattern_idx -> group_id for each edge
//   <prefix>.group_members - Binary: list of pattern indices per group
//   <prefix>.group_schedule - Processing schedule for GPU kernels

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <string>
#include <chrono>
#include <climits>

#define MAX_TAXA 256
#define MAX_VERTICES 512
#define MAX_EDGES 511
#define MAX_NAME_LEN 256

/* ========== Data Structures ========== */

struct Vertex {
    int id;
    char name[MAX_NAME_LEN];
    int parent_id;
    int children[10];
    int num_children;
    bool is_leaf;
    int pattern_index;
};

struct Edge {
    int parent_id;
    int child_id;
    std::vector<int> subtree_leaves;
    char parent_name[MAX_NAME_LEN];
    char child_name[MAX_NAME_LEN];
};

struct PatternData {
    int num_patterns;
    int num_taxa;
    std::vector<int> weights;
    std::vector<uint8_t> bases;
};

struct TreeData {
    int num_vertices;
    int num_edges;
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    int root_id;
    std::vector<std::string> taxon_names;
    std::vector<int> taxon_to_vertex;
};

struct SignatureGroup {
    std::vector<uint8_t> signature;  // The actual signature values
    std::vector<int> pattern_indices;  // Patterns with this signature
    int total_weight;  // Sum of pattern weights
};

struct EdgeGroups {
    int edge_id;
    int subtree_size;
    std::vector<SignatureGroup> groups;  // Unique signatures for this edge
    std::vector<int> pattern_to_group;  // For each pattern, which group it belongs to
};

/* ========== File I/O Functions ========== */

static void die(const char* msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

PatternData* read_patterns(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open pattern file");

    PatternData* pd = new PatternData();

    char line[65536];
    int num_patterns = 0;
    int num_taxa = -1;

    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) < 2) continue;

        int count = 0;
        char* line_copy = strdup(line);
        char* tok = strtok(line_copy, " \t\n");
        while (tok) {
            count++;
            tok = strtok(NULL, " \t\n");
        }
        free(line_copy);

        if (count < 2) continue;

        if (num_taxa < 0) {
            num_taxa = count - 1;
        } else if (count - 1 != num_taxa) {
            die("Inconsistent number of taxa in pattern file");
        }

        num_patterns++;
    }

    if (num_patterns == 0) die("No patterns found");
    if (num_taxa <= 0) die("No taxa found");

    pd->num_patterns = num_patterns;
    pd->num_taxa = num_taxa;
    pd->weights.resize(num_patterns);
    pd->bases.resize(num_patterns * num_taxa);

    rewind(f);
    int pat_idx = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) < 2) continue;

        char* tok = strtok(line, " \t\n");
        if (!tok) continue;

        pd->weights[pat_idx] = atoi(tok);

        for (int t = 0; t < num_taxa; t++) {
            tok = strtok(NULL, " \t\n");
            if (!tok) die("Missing base in pattern");
            pd->bases[pat_idx * num_taxa + t] = (uint8_t)atoi(tok);
        }

        pat_idx++;
        if (pat_idx >= num_patterns) break;
    }

    fclose(f);

    printf("Read %d patterns with %d taxa\n", num_patterns, num_taxa);
    return pd;
}

void read_taxon_order(const char* path, TreeData* td) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open taxon order file");

    char line[4096];
    int count = 0;
    int max_pos = -1;

    long pos = ftell(f);
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "taxon_name") != NULL) {
            // Header, skip
        } else {
            fseek(f, pos, SEEK_SET);
        }
    }

    while (fgets(line, sizeof(line), f)) {
        char name[MAX_NAME_LEN];
        int position;

        if (sscanf(line, "%[^,],%d", name, &position) == 2) {
            if (position > max_pos) max_pos = position;
            count++;
        }
    }

    td->taxon_names.resize(max_pos + 1);
    td->taxon_to_vertex.resize(max_pos + 1, -1);

    rewind(f);
    pos = ftell(f);
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "taxon_name") == NULL) {
            fseek(f, pos, SEEK_SET);
        }
    }

    while (fgets(line, sizeof(line), f)) {
        char name[MAX_NAME_LEN];
        int position;

        if (sscanf(line, "%[^,],%d", name, &position) == 2) {
            td->taxon_names[position] = name;
        }
    }

    fclose(f);
    printf("Read %d taxon names from order file\n", count);
}

void read_tree_edges(const char* path, TreeData* td) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open tree edges file");

    std::unordered_map<std::string, int> name_to_id;

    char line[4096];
    std::vector<std::pair<std::string, std::string>> edge_pairs;

    while (fgets(line, sizeof(line), f)) {
        char parent_name[MAX_NAME_LEN], child_name[MAX_NAME_LEN];
        float branch_len;

        int items = sscanf(line, "%s %s %f", parent_name, child_name, &branch_len);
        if (items >= 2) {
            edge_pairs.push_back({parent_name, child_name});

            if (name_to_id.find(parent_name) == name_to_id.end()) {
                int id = td->num_vertices++;
                name_to_id[parent_name] = id;
                if (id >= (int)td->vertices.size()) {
                    td->vertices.resize(id + 1);
                }
                strncpy(td->vertices[id].name, parent_name, MAX_NAME_LEN - 1);
                td->vertices[id].id = id;
                td->vertices[id].parent_id = -1;
                td->vertices[id].num_children = 0;
                td->vertices[id].is_leaf = true;
                td->vertices[id].pattern_index = -1;
            }

            if (name_to_id.find(child_name) == name_to_id.end()) {
                int id = td->num_vertices++;
                name_to_id[child_name] = id;
                if (id >= (int)td->vertices.size()) {
                    td->vertices.resize(id + 1);
                }
                strncpy(td->vertices[id].name, child_name, MAX_NAME_LEN - 1);
                td->vertices[id].id = id;
                td->vertices[id].parent_id = -1;
                td->vertices[id].num_children = 0;
                td->vertices[id].is_leaf = true;
                td->vertices[id].pattern_index = -1;
            }
        }
    }

    fclose(f);

    for (auto& ep : edge_pairs) {
        int p_id = name_to_id[ep.first];
        int c_id = name_to_id[ep.second];

        td->vertices[c_id].parent_id = p_id;
        td->vertices[p_id].children[td->vertices[p_id].num_children++] = c_id;
        td->vertices[p_id].is_leaf = false;
    }

    td->root_id = -1;
    for (int i = 0; i < td->num_vertices; i++) {
        if (td->vertices[i].parent_id < 0) {
            td->root_id = i;
            break;
        }
    }

    if (td->root_id < 0) die("No root found in tree");

    for (size_t t = 0; t < td->taxon_names.size(); t++) {
        if (!td->taxon_names[t].empty()) {
            auto it = name_to_id.find(td->taxon_names[t]);
            if (it != name_to_id.end()) {
                td->taxon_to_vertex[t] = it->second;
                td->vertices[it->second].pattern_index = t;
            } else {
                fprintf(stderr, "Warning: taxon '%s' not found in tree\n", td->taxon_names[t].c_str());
                td->taxon_to_vertex[t] = -1;
            }
        }
    }

    printf("Read tree with %d vertices, root at %s (id=%d)\n",
           td->num_vertices, td->vertices[td->root_id].name, td->root_id);
}

void compute_post_order_edges(TreeData* td) {
    std::vector<int> post_order;
    std::vector<int> stack;
    std::vector<int> visited(td->num_vertices, 0);

    stack.push_back(td->root_id);

    while (!stack.empty()) {
        int v = stack.back();

        if (visited[v] == 0) {
            visited[v] = 1;
            for (int i = td->vertices[v].num_children - 1; i >= 0; i--) {
                stack.push_back(td->vertices[v].children[i]);
            }
        } else {
            stack.pop_back();
            post_order.push_back(v);
        }
    }

    td->num_edges = 0;
    for (int v_id : post_order) {
        if (v_id != td->root_id) {
            int p_id = td->vertices[v_id].parent_id;
            td->edges.resize(td->num_edges + 1);
            td->edges[td->num_edges].parent_id = p_id;
            td->edges[td->num_edges].child_id = v_id;
            strncpy(td->edges[td->num_edges].parent_name, td->vertices[p_id].name, MAX_NAME_LEN);
            strncpy(td->edges[td->num_edges].child_name, td->vertices[v_id].name, MAX_NAME_LEN);
            td->num_edges++;
        }
    }

    printf("Built %d edges in post-order\n", td->num_edges);
}

void collect_subtree_leaves(TreeData* td, int v_id, std::vector<int>& leaves) {
    Vertex* v = &td->vertices[v_id];

    if (v->is_leaf && v->pattern_index >= 0) {
        leaves.push_back(v->pattern_index);
        return;
    }

    for (int i = 0; i < v->num_children; i++) {
        collect_subtree_leaves(td, v->children[i], leaves);
    }
}

void compute_edge_subtrees(TreeData* td) {
    for (int e = 0; e < td->num_edges; e++) {
        int child_id = td->edges[e].child_id;
        collect_subtree_leaves(td, child_id, td->edges[e].subtree_leaves);
        std::sort(td->edges[e].subtree_leaves.begin(), td->edges[e].subtree_leaves.end());
    }

    int total_leaves = 0;
    for (int e = 0; e < td->num_edges; e++) {
        total_leaves += td->edges[e].subtree_leaves.size();
    }
    printf("Total subtree leaf entries: %d (avg %.1f per edge)\n",
           total_leaves, (double)total_leaves / td->num_edges);
}

/* ========== Pattern Grouping ========== */

struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& v) const {
        size_t hash = 2166136261u;
        for (uint8_t val : v) {
            hash ^= val;
            hash *= 16777619u;
        }
        return hash;
    }
};

std::vector<EdgeGroups> compute_pattern_groups(PatternData* pd, TreeData* td) {
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;

    auto start_time = std::chrono::high_resolution_clock::now();

    printf("\nGrouping %d patterns by subtree signature for %d edges...\n", num_patterns, num_edges);

    std::vector<EdgeGroups> all_edge_groups(num_edges);

    long long total_groups = 0;
    long long total_pattern_group_pairs = 0;

    for (int e = 0; e < num_edges; e++) {
        EdgeGroups& eg = all_edge_groups[e];
        eg.edge_id = e;
        eg.subtree_size = td->edges[e].subtree_leaves.size();
        eg.pattern_to_group.resize(num_patterns, -1);

        std::vector<int>& subtree_leaves = td->edges[e].subtree_leaves;
        int subtree_size = subtree_leaves.size();

        if (subtree_size == 0) {
            // All patterns have empty signature - one group
            SignatureGroup sg;
            sg.total_weight = 0;
            for (int p = 0; p < num_patterns; p++) {
                sg.pattern_indices.push_back(p);
                sg.total_weight += pd->weights[p];
                eg.pattern_to_group[p] = 0;
            }
            eg.groups.push_back(std::move(sg));
            total_groups++;
            total_pattern_group_pairs += num_patterns;
            continue;
        }

        // Group patterns by signature
        std::unordered_map<std::vector<uint8_t>, int, VectorHash> sig_to_group_id;

        for (int p = 0; p < num_patterns; p++) {
            // Compute signature for this pattern
            std::vector<uint8_t> sig(subtree_size);
            for (int i = 0; i < subtree_size; i++) {
                int leaf_idx = subtree_leaves[i];
                sig[i] = pd->bases[p * pd->num_taxa + leaf_idx];
            }

            // Check if signature already has a group
            auto it = sig_to_group_id.find(sig);
            if (it != sig_to_group_id.end()) {
                // Add to existing group
                int group_id = it->second;
                eg.groups[group_id].pattern_indices.push_back(p);
                eg.groups[group_id].total_weight += pd->weights[p];
                eg.pattern_to_group[p] = group_id;
            } else {
                // Create new group
                int new_group_id = eg.groups.size();
                sig_to_group_id[sig] = new_group_id;

                SignatureGroup sg;
                sg.signature = std::move(sig);
                sg.pattern_indices.push_back(p);
                sg.total_weight = pd->weights[p];
                eg.groups.push_back(std::move(sg));

                eg.pattern_to_group[p] = new_group_id;
            }
        }

        total_groups += eg.groups.size();
        total_pattern_group_pairs += num_patterns;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    printf("Grouping completed in %ld ms\n", duration.count());
    printf("Total unique signature groups: %lld (vs %lld pattern-edge pairs)\n",
           total_groups, total_pattern_group_pairs);
    printf("Reduction: %.2f%% fewer message computations\n",
           100.0 * (1.0 - (double)total_groups / total_pattern_group_pairs));

    return all_edge_groups;
}

/* ========== GPU Schedule Generation ========== */

struct GPUWorkItem {
    int edge_id;
    int group_id;
    int num_patterns;  // How many patterns share this signature
    int total_weight;  // Sum of pattern weights
};

void generate_gpu_schedule(std::vector<EdgeGroups>& all_edge_groups, const char* output_prefix) {
    printf("\nGenerating GPU processing schedule...\n");

    // Strategy 1: Process edges in post-order (leaves to root)
    // This ensures parent messages are computed after children

    std::vector<GPUWorkItem> work_items;
    long long total_work = 0;
    int max_group_size = 0;
    int min_group_size = INT_MAX;

    for (auto& eg : all_edge_groups) {
        for (int g = 0; g < (int)eg.groups.size(); g++) {
            GPUWorkItem item;
            item.edge_id = eg.edge_id;
            item.group_id = g;
            item.num_patterns = eg.groups[g].pattern_indices.size();
            item.total_weight = eg.groups[g].total_weight;
            work_items.push_back(item);

            total_work += item.num_patterns;
            max_group_size = std::max(max_group_size, item.num_patterns);
            min_group_size = std::min(min_group_size, item.num_patterns);
        }
    }

    printf("Total work items (unique signatures): %zu\n", work_items.size());
    printf("Group sizes: min=%d, max=%d, avg=%.1f\n",
           min_group_size, max_group_size, (double)total_work / work_items.size());

    // Write schedule file
    char path_schedule[1024];
    snprintf(path_schedule, sizeof(path_schedule), "%s.group_schedule", output_prefix);
    FILE* f_schedule = fopen(path_schedule, "w");
    if (!f_schedule) die("Cannot write schedule file");

    fprintf(f_schedule, "# GPU Processing Schedule\n");
    fprintf(f_schedule, "# Format: edge_id group_id num_patterns total_weight\n");
    fprintf(f_schedule, "NUM_WORK_ITEMS %zu\n", work_items.size());
    fprintf(f_schedule, "TOTAL_PATTERNS_PROCESSED %lld\n", total_work);

    for (const auto& item : work_items) {
        fprintf(f_schedule, "WORK %d %d %d %d\n",
                item.edge_id, item.group_id, item.num_patterns, item.total_weight);
    }

    fclose(f_schedule);
    printf("Schedule written to %s\n", path_schedule);
}

void write_group_files(std::vector<EdgeGroups>& all_edge_groups, PatternData* pd,
                       TreeData* td, const char* output_prefix) {
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;

    printf("\nWriting group files...\n");

    // 1. Index file (human-readable summary)
    char path_index[1024];
    snprintf(path_index, sizeof(path_index), "%s.group_index", output_prefix);
    FILE* f_index = fopen(path_index, "w");
    if (!f_index) die("Cannot write index file");

    fprintf(f_index, "# Pattern Group Index\n");
    fprintf(f_index, "NUM_PATTERNS %d\n", num_patterns);
    fprintf(f_index, "NUM_EDGES %d\n", num_edges);

    long long total_groups = 0;
    for (auto& eg : all_edge_groups) {
        total_groups += eg.groups.size();
    }
    fprintf(f_index, "TOTAL_GROUPS %lld\n", total_groups);

    fprintf(f_index, "# edge_id parent_name child_name subtree_size num_groups max_group_size avg_group_size\n");

    for (auto& eg : all_edge_groups) {
        int max_size = 0;
        for (auto& g : eg.groups) {
            max_size = std::max(max_size, (int)g.pattern_indices.size());
        }
        double avg_size = (double)num_patterns / eg.groups.size();

        fprintf(f_index, "EDGE %d %s %s %d %zu %d %.2f\n",
                eg.edge_id,
                td->edges[eg.edge_id].parent_name,
                td->edges[eg.edge_id].child_name,
                eg.subtree_size,
                eg.groups.size(),
                max_size,
                avg_size);
    }

    fclose(f_index);
    printf("Index written to %s\n", path_index);

    // 2. Binary map file: pattern -> group_id for each edge
    char path_map[1024];
    snprintf(path_map, sizeof(path_map), "%s.group_map", output_prefix);
    FILE* f_map = fopen(path_map, "wb");
    if (!f_map) die("Cannot write map file");

    fwrite(&num_patterns, sizeof(int), 1, f_map);
    fwrite(&num_edges, sizeof(int), 1, f_map);

    for (auto& eg : all_edge_groups) {
        fwrite(eg.pattern_to_group.data(), sizeof(int), num_patterns, f_map);
    }

    fclose(f_map);
    printf("Map written to %s\n", path_map);

    // 3. Binary members file: list of patterns per group
    char path_members[1024];
    snprintf(path_members, sizeof(path_members), "%s.group_members", output_prefix);
    FILE* f_members = fopen(path_members, "wb");
    if (!f_members) die("Cannot write members file");

    fwrite(&num_patterns, sizeof(int), 1, f_members);
    fwrite(&num_edges, sizeof(int), 1, f_members);

    for (auto& eg : all_edge_groups) {
        int num_groups = eg.groups.size();
        fwrite(&num_groups, sizeof(int), 1, f_members);

        // Write offset table for this edge
        std::vector<int> offsets(num_groups + 1);
        offsets[0] = 0;
        for (int g = 0; g < num_groups; g++) {
            offsets[g + 1] = offsets[g] + eg.groups[g].pattern_indices.size();
        }
        fwrite(offsets.data(), sizeof(int), num_groups + 1, f_members);

        // Write all pattern indices
        for (int g = 0; g < num_groups; g++) {
            fwrite(eg.groups[g].pattern_indices.data(), sizeof(int),
                   eg.groups[g].pattern_indices.size(), f_members);
        }
    }

    fclose(f_members);
    printf("Members written to %s\n", path_members);

    // 4. Write compact signature data for GPU
    char path_sigs[1024];
    snprintf(path_sigs, sizeof(path_sigs), "%s.group_signatures", output_prefix);
    FILE* f_sigs = fopen(path_sigs, "wb");
    if (!f_sigs) die("Cannot write signatures file");

    fwrite(&num_edges, sizeof(int), 1, f_sigs);

    for (auto& eg : all_edge_groups) {
        int num_groups = eg.groups.size();
        int subtree_size = eg.subtree_size;

        fwrite(&num_groups, sizeof(int), 1, f_sigs);
        fwrite(&subtree_size, sizeof(int), 1, f_sigs);

        // Write all signatures for this edge
        for (auto& g : eg.groups) {
            if (subtree_size > 0) {
                fwrite(g.signature.data(), sizeof(uint8_t), subtree_size, f_sigs);
            }
        }
    }

    fclose(f_sigs);
    printf("Signatures written to %s\n", path_sigs);
}

/* ========== Statistics ========== */

void print_statistics(std::vector<EdgeGroups>& all_edge_groups, PatternData* pd) {
    printf("\n=== Pattern Group Statistics ===\n");

    // Find edges with most/least reuse
    std::vector<std::pair<int, double>> edge_reuse;
    for (auto& eg : all_edge_groups) {
        int num_patterns = pd->num_patterns;
        int num_groups = eg.groups.size();
        double reuse_rate = 100.0 * (num_patterns - num_groups) / num_patterns;
        edge_reuse.push_back({eg.edge_id, reuse_rate});
    }

    std::sort(edge_reuse.begin(), edge_reuse.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });

    printf("\nTop 10 edges by message reuse rate:\n");
    printf("%-8s %-12s %-10s\n", "Edge", "Groups", "Reuse Rate");
    for (int i = 0; i < std::min(10, (int)edge_reuse.size()); i++) {
        int e = edge_reuse[i].first;
        printf("%-8d %-12zu %-9.2f%%\n", e, all_edge_groups[e].groups.size(), edge_reuse[i].second);
    }

    printf("\nBottom 10 edges by message reuse rate:\n");
    printf("%-8s %-12s %-10s\n", "Edge", "Groups", "Reuse Rate");
    for (int i = std::max(0, (int)edge_reuse.size() - 10); i < (int)edge_reuse.size(); i++) {
        int e = edge_reuse[i].first;
        printf("%-8d %-12zu %-9.2f%%\n", e, all_edge_groups[e].groups.size(), edge_reuse[i].second);
    }

    // Group size distribution
    std::map<int, int> size_distribution;
    for (auto& eg : all_edge_groups) {
        for (auto& g : eg.groups) {
            int size = g.pattern_indices.size();
            size_distribution[size]++;
        }
    }

    printf("\nGroup size distribution (sample):\n");
    printf("%-12s %-12s\n", "Group Size", "Count");
    int shown = 0;
    for (auto& kv : size_distribution) {
        if (shown < 20 || kv.second > 100) {
            printf("%-12d %-12d\n", kv.first, kv.second);
            shown++;
        }
    }
}

/* ========== Main Function ========== */

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -o <prefix>    Output file prefix (default: pattern_groups)\n");
        fprintf(stderr, "\nOutput files:\n");
        fprintf(stderr, "  <prefix>.group_index      - Human-readable summary\n");
        fprintf(stderr, "  <prefix>.group_map        - Pattern to group mapping\n");
        fprintf(stderr, "  <prefix>.group_members    - Patterns in each group\n");
        fprintf(stderr, "  <prefix>.group_signatures - Signature data for each group\n");
        fprintf(stderr, "  <prefix>.group_schedule   - GPU processing schedule\n");
        return 1;
    }

    const char* pattern_file = argv[1];
    const char* edge_file = argv[2];
    const char* taxon_file = argv[3];
    const char* output_prefix = "pattern_groups";

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        }
    }

    printf("=== Pattern Group Pre-computation Tool ===\n");
    printf("Pattern file: %s\n", pattern_file);
    printf("Edge file: %s\n", edge_file);
    printf("Taxon order file: %s\n", taxon_file);
    printf("Output prefix: %s\n", output_prefix);
    printf("\n");

    // Read input data
    PatternData* pd = read_patterns(pattern_file);
    TreeData* td = new TreeData();
    td->num_vertices = 0;
    td->num_edges = 0;

    read_taxon_order(taxon_file, td);
    read_tree_edges(edge_file, td);
    compute_post_order_edges(td);
    compute_edge_subtrees(td);

    // Compute pattern groups
    std::vector<EdgeGroups> all_edge_groups = compute_pattern_groups(pd, td);

    // Generate GPU schedule
    generate_gpu_schedule(all_edge_groups, output_prefix);

    // Write output files
    write_group_files(all_edge_groups, pd, td, output_prefix);

    // Print statistics
    print_statistics(all_edge_groups, pd);

    // Cleanup
    delete pd;
    delete td;

    printf("\n=== Pre-computation Complete ===\n");
    return 0;
}
