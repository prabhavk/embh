// subtree_cache_precompute.cu
// Pre-compute subtree signatures and identify high-reuse patterns for selective caching
//
// This tool analyzes pattern files and tree topology to determine which subtree
// signatures have high reuse rates, enabling selective caching for large datasets.
//
// Compile: nvcc -O3 -std=c++11 subtree_cache_precompute.cu -o subtree_cache_precompute
// Usage:   ./subtree_cache_precompute <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]
//
// Output files:
//   <prefix>.cache_spec    - Cache specification (which edges to cache)
//   <prefix>.cache_stats   - Detailed statistics per edge
//   <prefix>.signatures    - Pre-computed signatures for cached edges

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

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
    int subtree_leaves[MAX_TAXA];
    int num_subtree_leaves;
};

struct PatternData {
    int num_patterns;
    int num_taxa;
    int* weights;           // [num_patterns]
    uint8_t* bases;         // [num_patterns * num_taxa]
};

struct TreeData {
    int num_vertices;
    int num_edges;
    Vertex vertices[MAX_VERTICES];
    Edge edges[MAX_EDGES];
    int root_id;
    char taxon_names[MAX_TAXA][MAX_NAME_LEN];
    int taxon_to_vertex[MAX_TAXA];
};

struct CacheStats {
    int edge_id;
    int num_unique_signatures;
    int reuse_count;
    double reuse_rate;
    int subtree_size;
    bool should_cache;
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

    PatternData* pd = (PatternData*)calloc(1, sizeof(PatternData));

    // First pass: count patterns and determine num_taxa
    char line[65536];
    int num_patterns = 0;
    int num_taxa = -1;

    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) < 2) continue;

        // Count tokens
        int count = 0;
        char* tok = strtok(line, " \t\n");
        while (tok) {
            count++;
            tok = strtok(NULL, " \t\n");
        }

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
    pd->weights = (int*)malloc(num_patterns * sizeof(int));
    pd->bases = (uint8_t*)malloc(num_patterns * num_taxa * sizeof(uint8_t));

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

    // Skip header if present
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "taxon_name") != NULL) {
            // This is a header, skip it
        } else {
            // Not a header, rewind
            rewind(f);
        }
    }

    while (fgets(line, sizeof(line), f) && count < MAX_TAXA) {
        char name[MAX_NAME_LEN];
        int pos;

        if (sscanf(line, "%[^,],%d", name, &pos) == 2) {
            strncpy(td->taxon_names[pos], name, MAX_NAME_LEN - 1);
            td->taxon_names[pos][MAX_NAME_LEN - 1] = '\0';
            count++;
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
    for (int t = 0; t < MAX_TAXA; t++) {
        if (strlen(td->taxon_names[t]) > 0) {
            auto it = name_to_id.find(td->taxon_names[t]);
            if (it != name_to_id.end()) {
                td->taxon_to_vertex[t] = it->second;
                td->vertices[it->second].pattern_index = t;
            } else {
                fprintf(stderr, "Warning: taxon '%s' not found in tree\n", td->taxon_names[t]);
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
            td->edges[td->num_edges].parent_id = p_id;
            td->edges[td->num_edges].child_id = v_id;
            td->edges[td->num_edges].num_subtree_leaves = 0;
            td->num_edges++;
        }
    }

    printf("Built %d edges in post-order\n", td->num_edges);
}

// Recursively collect leaves in subtree
void collect_subtree_leaves(TreeData* td, int v_id, int* leaves, int* count) {
    Vertex* v = &td->vertices[v_id];

    if (v->is_leaf && v->pattern_index >= 0) {
        leaves[(*count)++] = v->pattern_index;
        return;
    }

    for (int i = 0; i < v->num_children; i++) {
        collect_subtree_leaves(td, v->children[i], leaves, count);
    }
}

// Compute subtree leaves for each edge
void compute_edge_subtrees(TreeData* td) {
    for (int e = 0; e < td->num_edges; e++) {
        int child_id = td->edges[e].child_id;
        td->edges[e].num_subtree_leaves = 0;
        collect_subtree_leaves(td, child_id, td->edges[e].subtree_leaves,
                              &td->edges[e].num_subtree_leaves);
    }

    int total_leaves = 0;
    for (int e = 0; e < td->num_edges; e++) {
        total_leaves += td->edges[e].num_subtree_leaves;
    }
    printf("Total subtree leaf entries: %d (avg %.1f per edge)\n",
           total_leaves, (double)total_leaves / td->num_edges);
}

/* ========== CUDA Kernels ========== */

// FNV-1a hash function
__device__ __host__ unsigned int fnv1a_hash(const uint8_t* data, int len) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash & 0x7FFFFFFF;
}

// Compute subtree signature for one (pattern, edge) pair
__global__ void compute_subtree_signatures_kernel(
    int num_patterns,
    int num_edges,
    int num_taxa,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ edge_subtree_offsets,
    const int* __restrict__ edge_subtree_leaves,
    int* __restrict__ signatures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_patterns * num_edges;
    if (idx >= total_pairs) return;

    int pat_idx = idx / num_edges;
    int edge_idx = idx % num_edges;

    // Get subtree leaves for this edge
    int start_offset = edge_subtree_offsets[edge_idx];
    int end_offset = edge_subtree_offsets[edge_idx + 1];
    int num_leaves = end_offset - start_offset;

    // Compute signature using FNV-1a hash
    unsigned int hash = 2166136261u;
    for (int i = start_offset; i < end_offset; i++) {
        int leaf_idx = edge_subtree_leaves[i];
        uint8_t base = patterns[pat_idx * num_taxa + leaf_idx];
        hash ^= base;
        hash *= 16777619u;
    }

    signatures[pat_idx * num_edges + edge_idx] = (int)(hash & 0x7FFFFFFF);
}

/* ========== Main Analysis Functions ========== */

// Analyze signatures to find unique counts per edge
CacheStats* analyze_signatures(int* h_signatures, PatternData* pd, TreeData* td,
                               double min_reuse_rate) {
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;

    CacheStats* stats = (CacheStats*)calloc(num_edges, sizeof(CacheStats));

    printf("\nAnalyzing subtree signatures for %d patterns x %d edges...\n",
           num_patterns, num_edges);

    long long total_reuse = 0;
    int edges_to_cache = 0;

    for (int e = 0; e < num_edges; e++) {
        // Count unique signatures for this edge
        std::unordered_set<int> unique_sigs;
        for (int p = 0; p < num_patterns; p++) {
            int sig = h_signatures[p * num_edges + e];
            unique_sigs.insert(sig);
        }

        int num_unique = unique_sigs.size();
        int reuse_count = num_patterns - num_unique;
        double reuse_rate = 100.0 * reuse_count / num_patterns;

        stats[e].edge_id = e;
        stats[e].num_unique_signatures = num_unique;
        stats[e].reuse_count = reuse_count;
        stats[e].reuse_rate = reuse_rate;
        stats[e].subtree_size = td->edges[e].num_subtree_leaves;
        stats[e].should_cache = (reuse_rate >= min_reuse_rate);

        total_reuse += reuse_count;
        if (stats[e].should_cache) edges_to_cache++;
    }

    printf("Total message reuse: %lld times (%.1f%% reduction)\n",
           total_reuse, 100.0 * total_reuse / (num_patterns * num_edges));
    printf("Edges to cache (reuse rate >= %.1f%%): %d / %d\n",
           min_reuse_rate, edges_to_cache, num_edges);

    return stats;
}

// Build compact cache representation for high-reuse edges
void build_cache_specification(CacheStats* stats, int* h_signatures,
                               PatternData* pd, TreeData* td,
                               const char* output_prefix) {
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;

    // Count edges to cache
    int num_cached_edges = 0;
    for (int e = 0; e < num_edges; e++) {
        if (stats[e].should_cache) num_cached_edges++;
    }

    // Write cache specification file
    char path_spec[1024];
    snprintf(path_spec, sizeof(path_spec), "%s.cache_spec", output_prefix);
    FILE* f_spec = fopen(path_spec, "w");
    if (!f_spec) die("Cannot write cache spec file");

    fprintf(f_spec, "# Cache Specification for Selective Subtree Memoization\n");
    fprintf(f_spec, "# Format: edge_id num_unique_signatures reuse_rate subtree_size\n");
    fprintf(f_spec, "NUM_PATTERNS %d\n", num_patterns);
    fprintf(f_spec, "NUM_EDGES %d\n", num_edges);
    fprintf(f_spec, "NUM_CACHED_EDGES %d\n", num_cached_edges);

    int total_cache_slots = 0;
    for (int e = 0; e < num_edges; e++) {
        if (stats[e].should_cache) {
            fprintf(f_spec, "CACHE_EDGE %d %d %.2f %d\n",
                    e, stats[e].num_unique_signatures,
                    stats[e].reuse_rate, stats[e].subtree_size);
            total_cache_slots += stats[e].num_unique_signatures;
        }
    }

    fprintf(f_spec, "TOTAL_CACHE_SLOTS %d\n", total_cache_slots);
    fclose(f_spec);

    printf("Cache specification written to %s\n", path_spec);
    printf("Total cache slots needed: %d (%.2f MB for messages)\n",
           total_cache_slots, total_cache_slots * 4.0 * 8.0 / 1048576.0);

    // Write detailed statistics
    char path_stats[1024];
    snprintf(path_stats, sizeof(path_stats), "%s.cache_stats", output_prefix);
    FILE* f_stats = fopen(path_stats, "w");
    if (!f_stats) die("Cannot write cache stats file");

    fprintf(f_stats, "edge_id,unique_sigs,reuse_count,reuse_rate,subtree_size,should_cache\n");
    for (int e = 0; e < num_edges; e++) {
        fprintf(f_stats, "%d,%d,%d,%.2f,%d,%d\n",
                stats[e].edge_id, stats[e].num_unique_signatures,
                stats[e].reuse_count, stats[e].reuse_rate,
                stats[e].subtree_size, stats[e].should_cache ? 1 : 0);
    }
    fclose(f_stats);

    printf("Detailed statistics written to %s\n", path_stats);

    // Write pre-computed signatures for cached edges only
    char path_sigs[1024];
    snprintf(path_sigs, sizeof(path_sigs), "%s.signatures", output_prefix);
    FILE* f_sigs = fopen(path_sigs, "wb");
    if (!f_sigs) die("Cannot write signatures file");

    // Header
    fwrite(&num_patterns, sizeof(int), 1, f_sigs);
    fwrite(&num_cached_edges, sizeof(int), 1, f_sigs);

    // For each cached edge, write: edge_id, then [pattern_to_cache_slot] mapping
    int cache_slot_offset = 0;
    for (int e = 0; e < num_edges; e++) {
        if (!stats[e].should_cache) continue;

        fwrite(&e, sizeof(int), 1, f_sigs);

        // Build signature to cache slot mapping for this edge
        std::unordered_map<int, int> sig_to_slot;
        int* pattern_to_slot = (int*)malloc(num_patterns * sizeof(int));

        int next_slot = cache_slot_offset;
        for (int p = 0; p < num_patterns; p++) {
            int sig = h_signatures[p * num_edges + e];
            auto it = sig_to_slot.find(sig);
            if (it == sig_to_slot.end()) {
                sig_to_slot[sig] = next_slot;
                pattern_to_slot[p] = next_slot;
                next_slot++;
            } else {
                pattern_to_slot[p] = it->second;
            }
        }

        fwrite(pattern_to_slot, sizeof(int), num_patterns, f_sigs);
        free(pattern_to_slot);

        cache_slot_offset = next_slot;
    }

    fclose(f_sigs);
    printf("Pre-computed signatures written to %s\n", path_sigs);
}

// Print summary statistics
void print_summary(CacheStats* stats, int num_edges) {
    // Sort by reuse rate
    std::vector<int> sorted_edges(num_edges);
    for (int i = 0; i < num_edges; i++) sorted_edges[i] = i;

    std::sort(sorted_edges.begin(), sorted_edges.end(), [&](int a, int b) {
        return stats[a].reuse_rate > stats[b].reuse_rate;
    });

    printf("\n=== Top 10 Edges by Reuse Rate ===\n");
    printf("%-8s %-12s %-12s %-12s %-10s %-8s\n",
           "Edge", "Unique Sigs", "Reuse Count", "Reuse Rate", "Subtree", "Cache?");
    for (int i = 0; i < std::min(10, num_edges); i++) {
        int e = sorted_edges[i];
        printf("%-8d %-12d %-12d %-11.1f%% %-10d %-8s\n",
               stats[e].edge_id, stats[e].num_unique_signatures,
               stats[e].reuse_count, stats[e].reuse_rate,
               stats[e].subtree_size, stats[e].should_cache ? "YES" : "NO");
    }

    printf("\n=== Bottom 10 Edges by Reuse Rate ===\n");
    printf("%-8s %-12s %-12s %-12s %-10s %-8s\n",
           "Edge", "Unique Sigs", "Reuse Count", "Reuse Rate", "Subtree", "Cache?");
    for (int i = std::max(0, num_edges - 10); i < num_edges; i++) {
        int e = sorted_edges[i];
        printf("%-8d %-12d %-12d %-11.1f%% %-10d %-8s\n",
               stats[e].edge_id, stats[e].num_unique_signatures,
               stats[e].reuse_count, stats[e].reuse_rate,
               stats[e].subtree_size, stats[e].should_cache ? "YES" : "NO");
    }

    // Count by reuse rate thresholds
    int high_reuse = 0, medium_reuse = 0, low_reuse = 0;
    for (int e = 0; e < num_edges; e++) {
        if (stats[e].reuse_rate >= 50.0) high_reuse++;
        else if (stats[e].reuse_rate >= 10.0) medium_reuse++;
        else low_reuse++;
    }

    printf("\n=== Reuse Rate Distribution ===\n");
    printf("High reuse (>=50%%):   %d edges\n", high_reuse);
    printf("Medium reuse (10-50%%): %d edges\n", medium_reuse);
    printf("Low reuse (<10%%):     %d edges\n", low_reuse);
}

/* ========== Main Function ========== */

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <pattern.pat> <tree_edges.txt> <taxon_order.csv> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -o <prefix>    Output file prefix (default: subtree_cache)\n");
        fprintf(stderr, "  -r <rate>      Minimum reuse rate to cache (default: 50.0)\n");
        return 1;
    }

    const char* pattern_file = argv[1];
    const char* edge_file = argv[2];
    const char* taxon_file = argv[3];
    const char* output_prefix = "subtree_cache";
    double min_reuse_rate = 50.0;

    // Parse options
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            min_reuse_rate = atof(argv[++i]);
        }
    }

    printf("=== Subtree Cache Pre-computation Tool ===\n");
    printf("Pattern file: %s\n", pattern_file);
    printf("Edge file: %s\n", edge_file);
    printf("Taxon order file: %s\n", taxon_file);
    printf("Output prefix: %s\n", output_prefix);
    printf("Minimum reuse rate for caching: %.1f%%\n", min_reuse_rate);
    printf("\n");

    // Read input data
    PatternData* pd = read_patterns(pattern_file);
    TreeData* td = (TreeData*)calloc(1, sizeof(TreeData));
    read_taxon_order(taxon_file, td);
    read_tree_edges(edge_file, td);
    compute_post_order_edges(td);
    compute_edge_subtrees(td);

    // Prepare data for GPU
    int num_patterns = pd->num_patterns;
    int num_edges = td->num_edges;
    int num_taxa = pd->num_taxa;

    printf("\nPreparing GPU computation...\n");

    // Flatten subtree leaves data
    int total_subtree_entries = 0;
    for (int e = 0; e < num_edges; e++) {
        total_subtree_entries += td->edges[e].num_subtree_leaves;
    }

    int* h_edge_subtree_offsets = (int*)malloc((num_edges + 1) * sizeof(int));
    int* h_edge_subtree_leaves = (int*)malloc(total_subtree_entries * sizeof(int));

    h_edge_subtree_offsets[0] = 0;
    int offset = 0;
    for (int e = 0; e < num_edges; e++) {
        for (int i = 0; i < td->edges[e].num_subtree_leaves; i++) {
            h_edge_subtree_leaves[offset++] = td->edges[e].subtree_leaves[i];
        }
        h_edge_subtree_offsets[e + 1] = offset;
    }

    // Allocate GPU memory
    uint8_t* d_patterns;
    int* d_edge_subtree_offsets;
    int* d_edge_subtree_leaves;
    int* d_signatures;

    CUDA_CHECK(cudaMalloc(&d_patterns, num_patterns * num_taxa * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_edge_subtree_offsets, (num_edges + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_subtree_leaves, total_subtree_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_signatures, num_patterns * num_edges * sizeof(int)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_patterns, pd->bases, num_patterns * num_taxa * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_subtree_offsets, h_edge_subtree_offsets,
                          (num_edges + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_subtree_leaves, h_edge_subtree_leaves,
                          total_subtree_entries * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    printf("Computing %lld subtree signatures on GPU...\n",
           (long long)num_patterns * num_edges);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int block_size = 256;
    int total_pairs = num_patterns * num_edges;
    int num_blocks = (total_pairs + block_size - 1) / block_size;

    compute_subtree_signatures_kernel<<<num_blocks, block_size>>>(
        num_patterns, num_edges, num_taxa,
        d_patterns, d_edge_subtree_offsets, d_edge_subtree_leaves,
        d_signatures
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);
    printf("GPU computation time: %.2f ms\n", kernel_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back
    int* h_signatures = (int*)malloc(num_patterns * num_edges * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_signatures, d_signatures,
                          num_patterns * num_edges * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Analyze signatures
    CacheStats* stats = analyze_signatures(h_signatures, pd, td, min_reuse_rate);

    // Print summary
    print_summary(stats, num_edges);

    // Build and write cache specification
    build_cache_specification(stats, h_signatures, pd, td, output_prefix);

    // Cleanup
    free(h_signatures);
    free(stats);
    free(h_edge_subtree_offsets);
    free(h_edge_subtree_leaves);
    free(pd->weights);
    free(pd->bases);
    free(pd);
    free(td);

    cudaFree(d_patterns);
    cudaFree(d_edge_subtree_offsets);
    cudaFree(d_edge_subtree_leaves);
    cudaFree(d_signatures);

    printf("\n=== Pre-computation Complete ===\n");
    return 0;
}
