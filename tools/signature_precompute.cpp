// signature_precompute.cpp
// Pre-compute all subtree signatures for O(1) lookup during memoization
//
// This tool computes and stores the actual signature vectors (not just hashes)
// so they can be loaded once and used for instant cache key generation.
//
// Compile: g++ -O3 -std=c++11 signature_precompute.cpp -o signature_precompute
// Usage:   ./signature_precompute <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]
//
// Output files:
//   <prefix>.sig_index  - Index file: edge -> signature mapping info
//   <prefix>.sig_data   - Binary signature data (packed uint8 arrays)
//   <prefix>.sig_map    - Pattern -> signature hash mapping for each edge

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
    int pattern_index;  // Index in pattern array (-1 if not leaf)
};

struct Edge {
    int parent_id;
    int child_id;
    std::vector<int> subtree_leaves;  // Pattern indices of leaves in subtree
    char parent_name[MAX_NAME_LEN];
    char child_name[MAX_NAME_LEN];
};

struct PatternData {
    int num_patterns;
    int num_taxa;
    std::vector<int> weights;
    std::vector<uint8_t> bases;  // [num_patterns * num_taxa]
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

/* ========== File I/O Functions ========== */

static void die(const char* msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

// Read pattern file: each line is "<weight> <base0> <base1> ... <baseN-1>"
PatternData* read_patterns(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open pattern file");

    PatternData* pd = new PatternData();

    // First pass: count patterns and determine num_taxa
    char line[65536];
    int num_patterns = 0;
    int num_taxa = -1;

    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) < 2) continue;

        // Count tokens
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
            num_taxa = count - 1;  // First token is weight
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

    // Second pass: read data
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

// Read taxon order file: CSV with taxon_name,position
void read_taxon_order(const char* path, TreeData* td) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open taxon order file");

    char line[4096];
    int count = 0;
    int max_pos = -1;

    // Skip header if present
    long pos = ftell(f);
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "taxon_name") != NULL) {
            // This is a header, skip it
        } else {
            // Not a header, rewind
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

    // Re-read to populate
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

// Read edge list file: each line is "parent_name child_name [branch_length]"
void read_tree_edges(const char* path, TreeData* td) {
    FILE* f = fopen(path, "r");
    if (!f) die("Cannot open tree edges file");

    // Map from name to vertex id
    std::unordered_map<std::string, int> name_to_id;

    char line[4096];
    std::vector<std::pair<std::string, std::string>> edge_pairs;

    while (fgets(line, sizeof(line), f)) {
        char parent_name[MAX_NAME_LEN], child_name[MAX_NAME_LEN];
        float branch_len;

        int items = sscanf(line, "%s %s %f", parent_name, child_name, &branch_len);
        if (items >= 2) {
            edge_pairs.push_back({parent_name, child_name});

            // Add vertices if not seen
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
                td->vertices[id].is_leaf = true;  // Assume leaf until proven otherwise
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

    // Build tree structure
    for (auto& ep : edge_pairs) {
        int p_id = name_to_id[ep.first];
        int c_id = name_to_id[ep.second];

        td->vertices[c_id].parent_id = p_id;
        td->vertices[p_id].children[td->vertices[p_id].num_children++] = c_id;
        td->vertices[p_id].is_leaf = false;  // Has children, not a leaf
    }

    // Find root (vertex with no parent)
    td->root_id = -1;
    for (int i = 0; i < td->num_vertices; i++) {
        if (td->vertices[i].parent_id < 0) {
            td->root_id = i;
            break;
        }
    }

    if (td->root_id < 0) die("No root found in tree");

    // Map taxa to vertices
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

// Build post-order edge list
void compute_post_order_edges(TreeData* td) {
    // Use post-order traversal to build edge list
    std::vector<int> post_order;
    std::vector<int> stack;
    std::vector<int> visited(td->num_vertices, 0);

    stack.push_back(td->root_id);

    while (!stack.empty()) {
        int v = stack.back();

        if (visited[v] == 0) {
            // First visit: push children
            visited[v] = 1;
            for (int i = td->vertices[v].num_children - 1; i >= 0; i--) {
                stack.push_back(td->vertices[v].children[i]);
            }
        } else {
            // Second visit: add to post-order
            stack.pop_back();
            post_order.push_back(v);
        }
    }

    // Build edges in post-order (child before parent)
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

// Recursively collect leaves in subtree
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

// Compute subtree leaves for each edge
void compute_edge_subtrees(TreeData* td) {
    for (int e = 0; e < td->num_edges; e++) {
        int child_id = td->edges[e].child_id;
        collect_subtree_leaves(td, child_id, td->edges[e].subtree_leaves);
        // Sort for consistent signature ordering
        std::sort(td->edges[e].subtree_leaves.begin(), td->edges[e].subtree_leaves.end());
    }

    int total_leaves = 0;
    for (int e = 0; e < td->num_edges; e++) {
        total_leaves += td->edges[e].subtree_leaves.size();
    }
    printf("Total subtree leaf entries: %d (avg %.1f per edge)\n",
           total_leaves, (double)total_leaves / td->num_edges);
}

/* ========== Signature Computation ========== */

// Hash function for signature vector (FNV-1a)
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

// Compute signatures and build mapping
void compute_and_write_signatures(PatternData* pd, TreeData* td, const char* output_prefix) {
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;

    auto start_time = std::chrono::high_resolution_clock::now();

    printf("\nComputing signatures for %d patterns x %d edges...\n", num_patterns, num_edges);

    // For each edge, build: signature -> signature_id mapping
    // And pattern -> signature_id mapping

    std::vector<std::unordered_map<std::vector<uint8_t>, int, VectorHash>> edge_sig_maps(num_edges);
    std::vector<std::vector<int>> edge_pattern_to_sig_id(num_edges);
    std::vector<std::vector<std::vector<uint8_t>>> edge_unique_sigs(num_edges);

    long long total_signatures = 0;
    size_t total_memory = 0;

    for (int e = 0; e < num_edges; e++) {
        edge_pattern_to_sig_id[e].resize(num_patterns);

        std::vector<int>& subtree_leaves = td->edges[e].subtree_leaves;
        int subtree_size = subtree_leaves.size();

        if (subtree_size == 0) {
            // No leaves in subtree - all patterns map to empty signature
            edge_unique_sigs[e].push_back(std::vector<uint8_t>());
            for (int p = 0; p < num_patterns; p++) {
                edge_pattern_to_sig_id[e][p] = 0;
            }
            continue;
        }

        // Compute signature for each pattern
        for (int p = 0; p < num_patterns; p++) {
            std::vector<uint8_t> sig(subtree_size);
            for (int i = 0; i < subtree_size; i++) {
                int leaf_idx = subtree_leaves[i];
                sig[i] = pd->bases[p * pd->num_taxa + leaf_idx];
            }

            // Check if signature already exists
            auto it = edge_sig_maps[e].find(sig);
            if (it != edge_sig_maps[e].end()) {
                // Existing signature
                edge_pattern_to_sig_id[e][p] = it->second;
            } else {
                // New signature
                int new_id = edge_unique_sigs[e].size();
                edge_sig_maps[e][sig] = new_id;
                edge_pattern_to_sig_id[e][p] = new_id;
                edge_unique_sigs[e].push_back(std::move(sig));
            }
        }

        total_signatures += edge_unique_sigs[e].size();
        total_memory += edge_unique_sigs[e].size() * subtree_size * sizeof(uint8_t);
        total_memory += num_patterns * sizeof(int);  // pattern_to_sig_id
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    printf("Signature computation completed in %ld ms\n", duration.count());
    printf("Total unique signatures: %lld\n", total_signatures);
    printf("Estimated memory for signatures: %.2f MB\n", total_memory / (1024.0 * 1024.0));

    // Calculate reuse statistics
    long long total_reuse = 0;
    for (int e = 0; e < num_edges; e++) {
        int num_unique = edge_unique_sigs[e].size();
        total_reuse += num_patterns - num_unique;
    }
    printf("Total message reuse: %lld (%.2f%% reduction)\n",
           total_reuse, 100.0 * total_reuse / ((long long)num_patterns * num_edges));

    // Write output files
    printf("\nWriting output files...\n");

    // 1. Write index file (human-readable)
    char path_index[1024];
    snprintf(path_index, sizeof(path_index), "%s.sig_index", output_prefix);
    FILE* f_index = fopen(path_index, "w");
    if (!f_index) die("Cannot write index file");

    fprintf(f_index, "# Subtree Signature Index\n");
    fprintf(f_index, "NUM_PATTERNS %d\n", num_patterns);
    fprintf(f_index, "NUM_EDGES %d\n", num_edges);
    fprintf(f_index, "TOTAL_UNIQUE_SIGNATURES %lld\n", total_signatures);
    fprintf(f_index, "# edge_id parent_name child_name subtree_size num_unique_sigs reuse_rate\n");

    for (int e = 0; e < num_edges; e++) {
        int num_unique = edge_unique_sigs[e].size();
        double reuse_rate = 100.0 * (num_patterns - num_unique) / num_patterns;
        fprintf(f_index, "EDGE %d %s %s %d %d %.2f\n",
                e, td->edges[e].parent_name, td->edges[e].child_name,
                (int)td->edges[e].subtree_leaves.size(), num_unique, reuse_rate);
    }

    fclose(f_index);
    printf("Index file written: %s\n", path_index);

    // 2. Write binary data file
    char path_data[1024];
    snprintf(path_data, sizeof(path_data), "%s.sig_data", output_prefix);
    FILE* f_data = fopen(path_data, "wb");
    if (!f_data) die("Cannot write data file");

    // Header
    fwrite(&num_patterns, sizeof(int), 1, f_data);
    fwrite(&num_edges, sizeof(int), 1, f_data);

    // For each edge: subtree_size, num_unique_sigs, then all unique signatures
    for (int e = 0; e < num_edges; e++) {
        int subtree_size = td->edges[e].subtree_leaves.size();
        int num_unique = edge_unique_sigs[e].size();

        fwrite(&subtree_size, sizeof(int), 1, f_data);
        fwrite(&num_unique, sizeof(int), 1, f_data);

        // Write subtree leaf indices (for verification)
        fwrite(td->edges[e].subtree_leaves.data(), sizeof(int), subtree_size, f_data);

        // Write all unique signatures for this edge
        for (int s = 0; s < num_unique; s++) {
            fwrite(edge_unique_sigs[e][s].data(), sizeof(uint8_t), subtree_size, f_data);
        }
    }

    fclose(f_data);
    printf("Signature data written: %s\n", path_data);

    // 3. Write mapping file (pattern_idx -> sig_id for each edge)
    char path_map[1024];
    snprintf(path_map, sizeof(path_map), "%s.sig_map", output_prefix);
    FILE* f_map = fopen(path_map, "wb");
    if (!f_map) die("Cannot write mapping file");

    // Header
    fwrite(&num_patterns, sizeof(int), 1, f_map);
    fwrite(&num_edges, sizeof(int), 1, f_map);

    // For each edge, write pattern -> sig_id mapping
    for (int e = 0; e < num_edges; e++) {
        fwrite(edge_pattern_to_sig_id[e].data(), sizeof(int), num_patterns, f_map);
    }

    fclose(f_map);
    printf("Mapping file written: %s\n", path_map);

    // Summary
    long file_size = 0;
    FILE* f_check = fopen(path_data, "rb");
    if (f_check) {
        fseek(f_check, 0, SEEK_END);
        file_size = ftell(f_check);
        fclose(f_check);
    }

    printf("\n=== Summary ===\n");
    printf("Signature data file size: %.2f MB\n", file_size / (1024.0 * 1024.0));

    f_check = fopen(path_map, "rb");
    if (f_check) {
        fseek(f_check, 0, SEEK_END);
        file_size = ftell(f_check);
        fclose(f_check);
    }
    printf("Mapping file size: %.2f MB\n", file_size / (1024.0 * 1024.0));
}

/* ========== Main Function ========== */

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -o <prefix>    Output file prefix (default: subtree_signatures)\n");
        fprintf(stderr, "\nOutput files:\n");
        fprintf(stderr, "  <prefix>.sig_index  - Human-readable index\n");
        fprintf(stderr, "  <prefix>.sig_data   - Binary signature data\n");
        fprintf(stderr, "  <prefix>.sig_map    - Pattern to signature mapping\n");
        return 1;
    }

    const char* pattern_file = argv[1];
    const char* edge_file = argv[2];
    const char* taxon_file = argv[3];
    const char* output_prefix = "subtree_signatures";

    // Parse options
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        }
    }

    printf("=== Subtree Signature Pre-computation Tool ===\n");
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

    // Compute and write signatures
    compute_and_write_signatures(pd, td, output_prefix);

    // Cleanup
    delete pd;
    delete td;

    printf("\n=== Pre-computation Complete ===\n");
    return 0;
}
