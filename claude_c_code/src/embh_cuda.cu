/*
 * CUDA kernels for EMBH phylogenetic likelihood computation
 * Parallelizes pattern-level computation for pruning algorithm
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
#include "embh_types.h"
}

/* GPU data structures */
typedef struct {
    /* Tree structure */
    int num_patterns;
    int num_taxa;
    int num_vertices;
    int num_edges;
    int root_id;

    /* Device pointers */
    uint8_t* d_patterns;           /* num_patterns x num_taxa */
    int* d_pattern_weights;        /* num_patterns */
    double* d_transition_matrices; /* num_edges x 16 */
    int* d_edge_parent_ids;        /* num_edges */
    int* d_edge_child_ids;         /* num_edges */
    uint8_t* d_vertex_observed;    /* num_vertices */
    int* d_vertex_out_degree;      /* num_vertices */
    int* d_pattern_idx_to_vertex;  /* num_taxa */
    double* d_root_probability;    /* 4 */

    /* Results */
    double* d_pattern_log_likes;   /* num_patterns */
    double* d_total_log_likelihood;/* 1 */

    bool initialized;
} CUDAData;

static CUDAData cuda_data = {0};

/* Forward declarations (need extern "C" since they're defined in that block) */
extern "C" void cuda_cleanup_memoization(void);
extern "C" void cuda_cleanup_subtree_memoization(void);
extern "C" void cuda_free_cache_spec(void);
extern "C" void cuda_cleanup_estep(void);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* Double-precision atomic add for compatibility with all compute capabilities */
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* CUDA kernel: compute log-likelihood for each pattern independently */
__global__ void cuda_pruning_kernel(
    int num_patterns,
    int num_taxa,
    int num_vertices,
    int num_edges,
    int root_id,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ pattern_weights,
    const double* __restrict__ transition_matrices,
    const int* __restrict__ edge_parent_ids,
    const int* __restrict__ edge_child_ids,
    const uint8_t* __restrict__ vertex_observed,
    const int* __restrict__ vertex_out_degree,
    const int* __restrict__ pattern_idx_to_vertex,
    const double* __restrict__ root_probability,
    double* pattern_log_likes)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    /* Thread-local storage for conditional likelihoods - use local memory (slower but correct) */
    double cond_like[128 * 4];  /* Max 128 vertices * 4 states */

    /* Local arrays (small, fit in registers) */
    uint8_t vertex_to_base[128];  /* Max 128 vertices */
    bool cond_like_initialized[128];

    /* Initialize */
    for (int i = 0; i < num_vertices; i++) {
        vertex_to_base[i] = 4;  /* Gap */
        cond_like_initialized[i] = false;
        /* Initialize conditional likelihoods to zero */
        cond_like[i * 4 + 0] = 0.0;
        cond_like[i * 4 + 1] = 0.0;
        cond_like[i * 4 + 2] = 0.0;
        cond_like[i * 4 + 3] = 0.0;
    }

    /* Build vertex_to_base from pattern */
    for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
        int vertex_id = pattern_idx_to_vertex[taxon_idx];
        if (vertex_id >= 0 && vertex_id < num_vertices) {
            vertex_to_base[vertex_id] = patterns[pat_idx * num_taxa + taxon_idx];
        }
    }

    /* Track scaling factors */
    double vertex_scaling[128];
    for (int i = 0; i < num_vertices; i++) {
        vertex_scaling[i] = 0.0;
    }

    /* Process edges in post-order */
    for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
        int p_id = edge_parent_ids[edge_idx];
        int c_id = edge_child_ids[edge_idx];
        const double* P = &transition_matrices[edge_idx * 16];

        /* Accumulate child's scaling factor to parent */
        vertex_scaling[p_id] += vertex_scaling[c_id];

        /* Initialize leaf child */
        if (vertex_out_degree[c_id] == 0) {
            uint8_t base = vertex_to_base[c_id];
            if (base < 4) {
                cond_like[c_id * 4 + 0] = 0.0;
                cond_like[c_id * 4 + 1] = 0.0;
                cond_like[c_id * 4 + 2] = 0.0;
                cond_like[c_id * 4 + 3] = 0.0;
                cond_like[c_id * 4 + base] = 1.0;
            } else {
                cond_like[c_id * 4 + 0] = 1.0;
                cond_like[c_id * 4 + 1] = 1.0;
                cond_like[c_id * 4 + 2] = 1.0;
                cond_like[c_id * 4 + 3] = 1.0;
            }
            cond_like_initialized[c_id] = true;
        }

        /* Initialize parent if needed */
        if (!cond_like_initialized[p_id]) {
            if (!vertex_observed[p_id]) {
                cond_like[p_id * 4 + 0] = 1.0;
                cond_like[p_id * 4 + 1] = 1.0;
                cond_like[p_id * 4 + 2] = 1.0;
                cond_like[p_id * 4 + 3] = 1.0;
            } else {
                uint8_t base = vertex_to_base[p_id];
                if (base < 4) {
                    cond_like[p_id * 4 + 0] = 0.0;
                    cond_like[p_id * 4 + 1] = 0.0;
                    cond_like[p_id * 4 + 2] = 0.0;
                    cond_like[p_id * 4 + 3] = 0.0;
                    cond_like[p_id * 4 + base] = 1.0;
                } else {
                    cond_like[p_id * 4 + 0] = 1.0;
                    cond_like[p_id * 4 + 1] = 1.0;
                    cond_like[p_id * 4 + 2] = 1.0;
                    cond_like[p_id * 4 + 3] = 1.0;
                }
            }
            cond_like_initialized[p_id] = true;
        }

        /* DP update: parent_cl[dna_p] *= sum_c P[dna_p, dna_c] * child_cl[dna_c] */
        double largest = 0.0;
        for (int dna_p = 0; dna_p < 4; dna_p++) {
            double partial = 0.0;
            /* Unrolled for performance */
            partial += P[dna_p * 4 + 0] * cond_like[c_id * 4 + 0];
            partial += P[dna_p * 4 + 1] * cond_like[c_id * 4 + 1];
            partial += P[dna_p * 4 + 2] * cond_like[c_id * 4 + 2];
            partial += P[dna_p * 4 + 3] * cond_like[c_id * 4 + 3];

            cond_like[p_id * 4 + dna_p] *= partial;

            if (cond_like[p_id * 4 + dna_p] > largest) {
                largest = cond_like[p_id * 4 + dna_p];
            }
        }

        /* Scale to prevent underflow */
        if (largest > 0.0) {
            cond_like[p_id * 4 + 0] /= largest;
            cond_like[p_id * 4 + 1] /= largest;
            cond_like[p_id * 4 + 2] /= largest;
            cond_like[p_id * 4 + 3] /= largest;
            vertex_scaling[p_id] += log(largest);
        }
    }

    /* Compute site likelihood at root */
    double site_likelihood = 0.0;
    site_likelihood += root_probability[0] * cond_like[root_id * 4 + 0];
    site_likelihood += root_probability[1] * cond_like[root_id * 4 + 1];
    site_likelihood += root_probability[2] * cond_like[root_id * 4 + 2];
    site_likelihood += root_probability[3] * cond_like[root_id * 4 + 3];

    /* Compute weighted log-likelihood for this pattern */
    if (site_likelihood > 0.0) {
        double log_site_like = vertex_scaling[root_id] + log(site_likelihood);
        pattern_log_likes[pat_idx] = log_site_like * pattern_weights[pat_idx];
    } else {
        pattern_log_likes[pat_idx] = -1e30;  /* Error indicator */
    }
}

/* Reduction kernel to sum pattern log-likelihoods */
__global__ void cuda_reduction_kernel(
    const double* pattern_log_likes,
    int num_patterns,
    double* total_log_likelihood)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    sdata[tid] = (i < num_patterns) ? pattern_log_likes[i] : 0.0;
    __syncthreads();

    /* Reduction in shared memory */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write result for this block */
    if (tid == 0) {
        atomicAddDouble(total_log_likelihood, sdata[0]);
    }
}

/* C wrapper functions */
extern "C" {

int cuda_init_pruning(SEM* sem) {
    if (cuda_data.initialized) {
        return 0;  /* Already initialized */
    }

    cuda_data.num_patterns = sem->num_patterns;
    cuda_data.num_taxa = sem->packed_patterns->num_taxa;
    cuda_data.num_vertices = sem->num_vertices;
    cuda_data.num_edges = sem->num_post_order_edges;
    cuda_data.root_id = sem->root->id;

    printf("CUDA: Initializing for %d patterns, %d taxa, %d vertices, %d edges\n",
           cuda_data.num_patterns, cuda_data.num_taxa,
           cuda_data.num_vertices, cuda_data.num_edges);

    /* Allocate device memory */
    size_t pattern_size = cuda_data.num_patterns * cuda_data.num_taxa * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&cuda_data.d_patterns, pattern_size));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_pattern_weights,
                          cuda_data.num_patterns * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_transition_matrices,
                          cuda_data.num_edges * 16 * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_edge_parent_ids,
                          cuda_data.num_edges * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_edge_child_ids,
                          cuda_data.num_edges * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_vertex_observed,
                          cuda_data.num_vertices * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_vertex_out_degree,
                          cuda_data.num_vertices * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_pattern_idx_to_vertex,
                          cuda_data.num_taxa * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_root_probability, 4 * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_pattern_log_likes,
                          cuda_data.num_patterns * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&cuda_data.d_total_log_likelihood, sizeof(double)));

    /* Copy pattern data (unpack from packed storage) */
    uint8_t* h_patterns = (uint8_t*)malloc(pattern_size);
    for (int p = 0; p < cuda_data.num_patterns; p++) {
        for (int t = 0; t < cuda_data.num_taxa; t++) {
            h_patterns[p * cuda_data.num_taxa + t] =
                packed_storage_get_base(sem->packed_patterns, p, t);
        }
    }
    CUDA_CHECK(cudaMemcpy(cuda_data.d_patterns, h_patterns, pattern_size,
                          cudaMemcpyHostToDevice));
    free(h_patterns);

    /* Copy pattern weights */
    CUDA_CHECK(cudaMemcpy(cuda_data.d_pattern_weights, sem->pattern_weights,
                          cuda_data.num_patterns * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* Copy tree structure (pattern_idx_to_vertex) */
    int* h_pattern_idx_to_vertex = (int*)malloc(cuda_data.num_taxa * sizeof(int));
    for (int i = 0; i < cuda_data.num_taxa; i++) {
        h_pattern_idx_to_vertex[i] = -1;
    }
    for (int i = 0; i < sem->num_vertices; i++) {
        SEM_vertex* v = sem->vertices[i];
        if (v && v->observed && v->pattern_index >= 0 &&
            v->pattern_index < cuda_data.num_taxa) {
            h_pattern_idx_to_vertex[v->pattern_index] = v->id;
        }
    }
    CUDA_CHECK(cudaMemcpy(cuda_data.d_pattern_idx_to_vertex, h_pattern_idx_to_vertex,
                          cuda_data.num_taxa * sizeof(int), cudaMemcpyHostToDevice));
    free(h_pattern_idx_to_vertex);

    /* Copy vertex properties */
    uint8_t* h_vertex_observed = (uint8_t*)malloc(cuda_data.num_vertices * sizeof(uint8_t));
    int* h_vertex_out_degree = (int*)malloc(cuda_data.num_vertices * sizeof(int));
    for (int i = 0; i < sem->num_vertices; i++) {
        h_vertex_observed[i] = sem->vertices[i]->observed ? 1 : 0;
        h_vertex_out_degree[i] = sem->vertices[i]->out_degree;
    }
    CUDA_CHECK(cudaMemcpy(cuda_data.d_vertex_observed, h_vertex_observed,
                          cuda_data.num_vertices * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_data.d_vertex_out_degree, h_vertex_out_degree,
                          cuda_data.num_vertices * sizeof(int),
                          cudaMemcpyHostToDevice));
    free(h_vertex_observed);
    free(h_vertex_out_degree);

    /* Copy edge structure */
    int* h_edge_parent_ids = (int*)malloc(cuda_data.num_edges * sizeof(int));
    int* h_edge_child_ids = (int*)malloc(cuda_data.num_edges * sizeof(int));
    double* h_transition_matrices = (double*)malloc(cuda_data.num_edges * 16 * sizeof(double));

    for (int e = 0; e < cuda_data.num_edges; e++) {
        h_edge_parent_ids[e] = sem->post_order_parent[e]->id;
        h_edge_child_ids[e] = sem->post_order_child[e]->id;
        memcpy(&h_transition_matrices[e * 16],
               sem->post_order_child[e]->transition_matrix,
               16 * sizeof(double));
    }

    CUDA_CHECK(cudaMemcpy(cuda_data.d_edge_parent_ids, h_edge_parent_ids,
                          cuda_data.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_data.d_edge_child_ids, h_edge_child_ids,
                          cuda_data.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_data.d_transition_matrices, h_transition_matrices,
                          cuda_data.num_edges * 16 * sizeof(double),
                          cudaMemcpyHostToDevice));

    free(h_edge_parent_ids);
    free(h_edge_child_ids);
    free(h_transition_matrices);

    /* Copy root probability */
    CUDA_CHECK(cudaMemcpy(cuda_data.d_root_probability, sem->root_probability,
                          4 * sizeof(double), cudaMemcpyHostToDevice));

    cuda_data.initialized = true;
    printf("CUDA: Initialization complete\n");

    return 0;
}

int cuda_update_transition_matrices(SEM* sem) {
    if (!cuda_data.initialized) return -1;

    /* Update transition matrices on GPU (for EM iterations) */
    double* h_transition_matrices = (double*)malloc(cuda_data.num_edges * 16 * sizeof(double));

    for (int e = 0; e < cuda_data.num_edges; e++) {
        memcpy(&h_transition_matrices[e * 16],
               sem->post_order_child[e]->transition_matrix,
               16 * sizeof(double));
    }

    CUDA_CHECK(cudaMemcpy(cuda_data.d_transition_matrices, h_transition_matrices,
                          cuda_data.num_edges * 16 * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Update root probability */
    CUDA_CHECK(cudaMemcpy(cuda_data.d_root_probability, sem->root_probability,
                          4 * sizeof(double), cudaMemcpyHostToDevice));

    free(h_transition_matrices);
    return 0;
}

double cuda_compute_log_likelihood(SEM* sem) {
    bool needs_init = !cuda_data.initialized;

    if (needs_init) {
        if (cuda_init_pruning(sem) != 0) {
            fprintf(stderr, "CUDA: Failed to initialize\n");
            return -1e30;
        }
    }

    /* Synchronize to get accurate timing */
    cudaDeviceSynchronize();

    /* Reset total log-likelihood */
    double zero = 0.0;
    cudaMemcpy(cuda_data.d_total_log_likelihood, &zero, sizeof(double),
               cudaMemcpyHostToDevice);

    /* Launch pruning kernel */
    int block_size = 128;  /* Threads per block */
    int num_blocks = (cuda_data.num_patterns + block_size - 1) / block_size;

    /* Record kernel timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* No shared memory needed - using local memory per thread */
    cuda_pruning_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        cuda_data.num_taxa,
        cuda_data.num_vertices,
        cuda_data.num_edges,
        cuda_data.root_id,
        cuda_data.d_patterns,
        cuda_data.d_pattern_weights,
        cuda_data.d_transition_matrices,
        cuda_data.d_edge_parent_ids,
        cuda_data.d_edge_child_ids,
        cuda_data.d_vertex_observed,
        cuda_data.d_vertex_out_degree,
        cuda_data.d_pattern_idx_to_vertex,
        cuda_data.d_root_probability,
        cuda_data.d_pattern_log_likes
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1e30;
    }

    /* Reduction to compute total log-likelihood */
    cuda_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
        cuda_data.d_pattern_log_likes,
        cuda_data.num_patterns,
        cuda_data.d_total_log_likelihood
    );

    /* Record end time */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);
    printf("CUDA kernel execution time: %.4f ms\n", kernel_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA reduction failed: %s\n", cudaGetErrorString(err));
        return -1e30;
    }

    /* Copy result back */
    double total_log_likelihood;
    cudaMemcpy(&total_log_likelihood, cuda_data.d_total_log_likelihood,
               sizeof(double), cudaMemcpyDeviceToHost);

    return total_log_likelihood;
}

void cuda_cleanup(void) {
    if (!cuda_data.initialized) return;

    /* Clean up memoization and E-step first */
    cuda_cleanup_memoization();
    cuda_cleanup_subtree_memoization();
    cuda_cleanup_estep();

    cudaFree(cuda_data.d_patterns);
    cudaFree(cuda_data.d_pattern_weights);
    cudaFree(cuda_data.d_transition_matrices);
    cudaFree(cuda_data.d_edge_parent_ids);
    cudaFree(cuda_data.d_edge_child_ids);
    cudaFree(cuda_data.d_vertex_observed);
    cudaFree(cuda_data.d_vertex_out_degree);
    cudaFree(cuda_data.d_pattern_idx_to_vertex);
    cudaFree(cuda_data.d_root_probability);
    cudaFree(cuda_data.d_pattern_log_likes);
    cudaFree(cuda_data.d_total_log_likelihood);

    cuda_data.initialized = false;
    printf("CUDA: Cleanup complete\n");
}

/* Check if CUDA is available */
int cuda_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

/* Get CUDA device info */
void cuda_print_device_info(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDA Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f MB\n", prop.totalGlobalMem / 1048576.0);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
    }
}

/* ========== CUDA-Accelerated EM with Aitken ========== */

/* Structure to hold memoization data for GPU */
typedef struct {
    int* d_pattern_signatures;      /* Signature hash per pattern */
    int* d_unique_signatures;       /* Unique signature values */
    int* d_signature_to_pattern;    /* Map signature index to first pattern with that signature */
    int* d_pattern_to_signature_idx;/* Map pattern to its signature index */
    double* d_cached_messages;      /* Cached messages per unique signature per edge */
    double* d_cached_scales;        /* Cached scaling factors */
    int num_unique_signatures;
    bool initialized;
} MemoizationData;

static MemoizationData memo_data = {0};

/* ========== SUBTREE-LEVEL MEMOIZATION ========== */

/* Structure for subtree-level memoization (more fine-grained than pattern-level) */
typedef struct {
    /* For each edge, store which leaf indices belong to its subtree */
    int* h_edge_subtree_leaves;     /* Flattened array of leaf indices */
    int* h_edge_subtree_offsets;    /* Offset into h_edge_subtree_leaves for each edge */
    int* h_edge_num_subtree_leaves; /* Number of leaves in each edge's subtree */

    /* For each (pattern, edge) pair, pre-computed subtree signature */
    int* h_subtree_signatures;      /* [num_patterns x num_edges] */

    /* Hash table: signature -> cache slot */
    int* h_signature_to_cache_slot; /* Hash table for lookup */
    int* h_cache_slot_to_signature; /* Reverse mapping */
    int hash_table_size;
    int num_cache_slots;            /* Number of unique (edge, signature) pairs */

    /* GPU arrays */
    int* d_edge_subtree_leaves;
    int* d_edge_subtree_offsets;
    int* d_edge_num_subtree_leaves;
    int* d_subtree_signatures;      /* [num_patterns x num_edges] */
    int* d_pattern_edge_to_cache;   /* [num_patterns x num_edges] -> cache slot */
    double* d_subtree_cached_messages; /* [num_cache_slots x 4] */
    int* d_cache_valid;             /* [num_cache_slots] - is cache entry computed? */

    bool initialized;
} SubtreeMemoData;

static SubtreeMemoData subtree_memo = {0};

/* Compute pattern signature (hash of DNA values at leaf positions) on CPU */
static int compute_pattern_signature(SEM* sem, int pat_idx) {
    int hash = 0;
    for (int i = 0; i < sem->num_leaves; i++) {
        SEM_vertex* leaf = sem->leaves[i];
        if (leaf && leaf->pattern_index >= 0) {
            uint8_t dna = packed_storage_get_base(sem->packed_patterns, pat_idx, leaf->pattern_index);
            hash = hash * 5 + dna;  /* Base-5 encoding */
        }
    }
    return hash;
}

/* Helper: recursively collect leaves in subtree rooted at vertex v */
static void collect_subtree_leaves_recursive(SEM* sem, SEM_vertex* v, int* leaf_indices, int* count) {
    if (!v) return;

    /* If this is a leaf (observed vertex with pattern_index), add it */
    if (v->out_degree == 0 && v->observed && v->pattern_index >= 0) {
        leaf_indices[(*count)++] = v->pattern_index;
        return;
    }

    /* Recursively visit children */
    for (int i = 0; i < v->out_degree; i++) {
        collect_subtree_leaves_recursive(sem, v->children[i], leaf_indices, count);
    }
}

/* Compute subtree signature for a specific edge and pattern */
static int compute_subtree_signature(int pat_idx, int num_taxa,
                                     const uint8_t* h_patterns,
                                     const int* subtree_leaves,
                                     int num_leaves) {
    /* Use FNV-1a hash for better distribution */
    unsigned int hash = 2166136261u;
    for (int i = 0; i < num_leaves; i++) {
        int leaf_idx = subtree_leaves[i];
        uint8_t dna = h_patterns[pat_idx * num_taxa + leaf_idx];
        hash ^= dna;
        hash *= 16777619u;
    }
    return (int)(hash & 0x7FFFFFFF);  /* Keep positive */
}

/* Initialize subtree-level memoization */
int cuda_init_subtree_memoization(SEM* sem) {
    if (subtree_memo.initialized) return 0;

    if (!cuda_data.initialized) {
        fprintf(stderr, "CUDA: Must initialize cuda_data first\n");
        return -1;
    }

    printf("CUDA: Initializing subtree-level memoization...\n");

    int num_edges = cuda_data.num_edges;
    int num_patterns = cuda_data.num_patterns;
    int num_taxa = cuda_data.num_taxa;

    /* Step 1: Compute subtree leaves for each edge */
    printf("  Computing subtree leaves for %d edges...\n", num_edges);

    subtree_memo.h_edge_num_subtree_leaves = (int*)malloc(num_edges * sizeof(int));
    subtree_memo.h_edge_subtree_offsets = (int*)malloc((num_edges + 1) * sizeof(int));

    /* First pass: count leaves per edge */
    int* temp_leaf_buffer = (int*)malloc(num_taxa * sizeof(int));
    int total_subtree_leaves = 0;

    subtree_memo.h_edge_subtree_offsets[0] = 0;
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        collect_subtree_leaves_recursive(sem, child, temp_leaf_buffer, &count);
        subtree_memo.h_edge_num_subtree_leaves[e] = count;
        total_subtree_leaves += count;
        subtree_memo.h_edge_subtree_offsets[e + 1] = total_subtree_leaves;
    }

    /* Second pass: store leaf indices */
    subtree_memo.h_edge_subtree_leaves = (int*)malloc(total_subtree_leaves * sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        int offset = subtree_memo.h_edge_subtree_offsets[e];
        collect_subtree_leaves_recursive(sem, child,
                                         &subtree_memo.h_edge_subtree_leaves[offset], &count);
    }
    free(temp_leaf_buffer);

    printf("  Total subtree leaf entries: %d (avg %.1f per edge)\n",
           total_subtree_leaves, (double)total_subtree_leaves / num_edges);

    /* Step 2: Unpack pattern data for CPU processing */
    uint8_t* h_patterns = (uint8_t*)malloc(num_patterns * num_taxa * sizeof(uint8_t));
    for (int p = 0; p < num_patterns; p++) {
        for (int t = 0; t < num_taxa; t++) {
            h_patterns[p * num_taxa + t] =
                packed_storage_get_base(sem->packed_patterns, p, t);
        }
    }

    /* Step 3: Compute subtree signatures for all (pattern, edge) pairs */
    printf("  Computing subtree signatures for %d patterns x %d edges = %d pairs...\n",
           num_patterns, num_edges, num_patterns * num_edges);

    subtree_memo.h_subtree_signatures = (int*)malloc(num_patterns * num_edges * sizeof(int));

    /* Progress tracking for large datasets */
    int progress_interval = num_patterns / 20;  /* Report every 5% */
    if (progress_interval < 1) progress_interval = 1;

    for (int p = 0; p < num_patterns; p++) {
        for (int e = 0; e < num_edges; e++) {
            int offset = subtree_memo.h_edge_subtree_offsets[e];
            int num_leaves = subtree_memo.h_edge_num_subtree_leaves[e];
            int sig = compute_subtree_signature(p, num_taxa, h_patterns,
                                               &subtree_memo.h_edge_subtree_leaves[offset],
                                               num_leaves);
            subtree_memo.h_subtree_signatures[p * num_edges + e] = sig;
        }

        /* Progress indicator for large datasets */
        if (num_patterns > 10000 && (p + 1) % progress_interval == 0) {
            printf("\r    Computing signatures: %d/%d patterns (%.0f%%)...",
                   p + 1, num_patterns, 100.0 * (p + 1) / num_patterns);
            fflush(stdout);
        }
    }
    if (num_patterns > 10000) {
        printf("\r    Computing signatures: %d/%d patterns (100%%)     \n", num_patterns, num_patterns);
    }

    /* Step 4: Build mapping from (edge_id, signature) -> cache slot */
    printf("  Building signature to cache slot mapping...\n");

    /* Count unique (edge, signature) pairs and assign cache slots */
    int* h_pattern_edge_to_cache = (int*)malloc(num_patterns * num_edges * sizeof(int));
    int num_cache_slots = 0;

    /* Use hash table for O(1) lookup instead of O(n) linear search */
    /* Hash table size: 2x expected unique signatures per edge */
    int hash_table_size = num_patterns * 2;
    if (hash_table_size < 1024) hash_table_size = 1024;

    int* hash_keys = (int*)malloc(hash_table_size * sizeof(int));
    int* hash_slots = (int*)malloc(hash_table_size * sizeof(int));

    /* Process all (pattern, edge) pairs - for each edge, track unique signatures */
    long long total_savings = 0;

    /* Track per-edge statistics */
    int* edge_unique_sigs = (int*)malloc(num_edges * sizeof(int));
    int* edge_reuse_count = (int*)malloc(num_edges * sizeof(int));

    for (int e = 0; e < num_edges; e++) {
        /* Reset hash table for this edge */
        for (int i = 0; i < hash_table_size; i++) {
            hash_keys[i] = -1;  /* -1 means empty */
        }

        int unique_this_edge = 0;
        int reuse_this_edge = 0;

        for (int p = 0; p < num_patterns; p++) {
            int sig = subtree_memo.h_subtree_signatures[p * num_edges + e];

            /* Hash table lookup with linear probing */
            unsigned int hash_idx = ((unsigned int)sig * 2654435761u) % hash_table_size;
            int found_slot = -1;
            int probes = 0;

            while (hash_keys[hash_idx] != -1 && probes < hash_table_size) {
                if (hash_keys[hash_idx] == sig) {
                    /* Found existing signature */
                    found_slot = hash_slots[hash_idx];
                    total_savings++;
                    reuse_this_edge++;
                    break;
                }
                hash_idx = (hash_idx + 1) % hash_table_size;
                probes++;
            }

            if (found_slot < 0) {
                /* New signature - insert into hash table */
                found_slot = num_cache_slots++;
                hash_keys[hash_idx] = sig;
                hash_slots[hash_idx] = found_slot;
                unique_this_edge++;
            }

            h_pattern_edge_to_cache[p * num_edges + e] = found_slot;
        }

        edge_unique_sigs[e] = unique_this_edge;
        edge_reuse_count[e] = reuse_this_edge;

        /* Progress indicator for large datasets */
        if (num_patterns > 10000 && (e + 1) % 10 == 0) {
            printf("\r    Edge %d/%d processed...", e + 1, num_edges);
            fflush(stdout);
        }
    }
    if (num_patterns > 10000) {
        printf("\r    All %d edges processed.                    \n", num_edges);
    }

    free(hash_keys);
    free(hash_slots);

    subtree_memo.num_cache_slots = num_cache_slots;

    printf("  Found %d unique (edge, signature) pairs\n", num_cache_slots);
    printf("  Total (pattern, edge) pairs: %d\n", num_patterns * num_edges);
    printf("  Message reuse: %lld times (%.1f%% reduction)\n",
           total_savings, 100.0 * total_savings / (num_patterns * num_edges));

    /* Print per-edge reuse statistics */
    printf("\n  Per-edge reuse analysis (top 10 by reuse rate):\n");
    printf("  %-8s %-12s %-10s %-12s %-10s\n", "Edge", "Unique Sigs", "Reuse Cnt", "Reuse Rate", "Subtree Sz");

    /* Sort edges by reuse rate */
    int* sorted_edges = (int*)malloc(num_edges * sizeof(int));
    for (int i = 0; i < num_edges; i++) sorted_edges[i] = i;

    /* Simple bubble sort for small arrays */
    for (int i = 0; i < num_edges - 1; i++) {
        for (int j = 0; j < num_edges - i - 1; j++) {
            double rate_j = (double)edge_reuse_count[sorted_edges[j]] / num_patterns;
            double rate_j1 = (double)edge_reuse_count[sorted_edges[j+1]] / num_patterns;
            if (rate_j < rate_j1) {
                int tmp = sorted_edges[j];
                sorted_edges[j] = sorted_edges[j+1];
                sorted_edges[j+1] = tmp;
            }
        }
    }

    /* Print top 10 edges by reuse rate */
    int num_to_print = num_edges < 10 ? num_edges : 10;
    for (int i = 0; i < num_to_print; i++) {
        int e = sorted_edges[i];
        double reuse_rate = 100.0 * edge_reuse_count[e] / num_patterns;
        printf("  %-8d %-12d %-10d %-11.1f%% %-10d\n",
               e, edge_unique_sigs[e], edge_reuse_count[e], reuse_rate,
               subtree_memo.h_edge_num_subtree_leaves[e]);
    }

    /* Print bottom 10 edges by reuse rate */
    printf("\n  Per-edge reuse analysis (bottom 10 by reuse rate):\n");
    printf("  %-8s %-12s %-10s %-12s %-10s\n", "Edge", "Unique Sigs", "Reuse Cnt", "Reuse Rate", "Subtree Sz");
    for (int i = num_edges - num_to_print; i < num_edges; i++) {
        int e = sorted_edges[i];
        double reuse_rate = 100.0 * edge_reuse_count[e] / num_patterns;
        printf("  %-8d %-12d %-10d %-11.1f%% %-10d\n",
               e, edge_unique_sigs[e], edge_reuse_count[e], reuse_rate,
               subtree_memo.h_edge_num_subtree_leaves[e]);
    }

    /* Summary statistics */
    int edges_with_high_reuse = 0;  /* >50% reuse rate */
    int edges_with_low_reuse = 0;   /* <10% reuse rate */
    for (int e = 0; e < num_edges; e++) {
        double reuse_rate = 100.0 * edge_reuse_count[e] / num_patterns;
        if (reuse_rate > 50.0) edges_with_high_reuse++;
        if (reuse_rate < 10.0) edges_with_low_reuse++;
    }
    printf("\n  Summary: %d edges with >50%% reuse, %d edges with <10%% reuse\n",
           edges_with_high_reuse, edges_with_low_reuse);

    free(sorted_edges);
    free(edge_unique_sigs);
    free(edge_reuse_count);

    /* Step 5: Allocate GPU memory */
    printf("  Allocating GPU memory for subtree memoization...\n");

    cudaMalloc(&subtree_memo.d_edge_subtree_leaves, total_subtree_leaves * sizeof(int));
    cudaMalloc(&subtree_memo.d_edge_subtree_offsets, (num_edges + 1) * sizeof(int));
    cudaMalloc(&subtree_memo.d_edge_num_subtree_leaves, num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_subtree_signatures, num_patterns * num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_pattern_edge_to_cache, num_patterns * num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_subtree_cached_messages, num_cache_slots * 4 * sizeof(double));
    cudaMalloc(&subtree_memo.d_cache_valid, num_cache_slots * sizeof(int));

    /* Copy to GPU */
    cudaMemcpy(subtree_memo.d_edge_subtree_leaves, subtree_memo.h_edge_subtree_leaves,
               total_subtree_leaves * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_edge_subtree_offsets, subtree_memo.h_edge_subtree_offsets,
               (num_edges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_edge_num_subtree_leaves, subtree_memo.h_edge_num_subtree_leaves,
               num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_subtree_signatures, subtree_memo.h_subtree_signatures,
               num_patterns * num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_pattern_edge_to_cache, h_pattern_edge_to_cache,
               num_patterns * num_edges * sizeof(int), cudaMemcpyHostToDevice);

    /* Initialize cache as invalid */
    cudaMemset(subtree_memo.d_cache_valid, 0, num_cache_slots * sizeof(int));

    free(h_patterns);
    free(h_pattern_edge_to_cache);

    double cache_memory_mb = (num_cache_slots * 4 * sizeof(double) +
                              num_cache_slots * sizeof(int)) / 1048576.0;
    printf("  Subtree memoization cache: %.2f MB\n", cache_memory_mb);

    subtree_memo.initialized = true;
    printf("CUDA: Subtree-level memoization initialized\n");

    return 0;
}

/* Cleanup subtree memoization */
void cuda_cleanup_subtree_memoization(void) {
    if (!subtree_memo.initialized) return;

    free(subtree_memo.h_edge_subtree_leaves);
    free(subtree_memo.h_edge_subtree_offsets);
    free(subtree_memo.h_edge_num_subtree_leaves);
    free(subtree_memo.h_subtree_signatures);

    cudaFree(subtree_memo.d_edge_subtree_leaves);
    cudaFree(subtree_memo.d_edge_subtree_offsets);
    cudaFree(subtree_memo.d_edge_num_subtree_leaves);
    cudaFree(subtree_memo.d_subtree_signatures);
    cudaFree(subtree_memo.d_pattern_edge_to_cache);
    cudaFree(subtree_memo.d_subtree_cached_messages);
    cudaFree(subtree_memo.d_cache_valid);

    subtree_memo.initialized = false;
    printf("CUDA: Subtree memoization cleanup complete\n");
}

/* ========== SELECTIVE SUBTREE MEMOIZATION WITH PRE-COMPUTED CACHE SPEC ========== */

/* Structure to hold pre-computed cache specification */
typedef struct {
    int num_patterns;           /* Expected number of patterns */
    int num_edges;              /* Expected number of edges */
    int num_cached_edges;       /* Number of edges to cache */
    int* cached_edge_ids;       /* Array of edge IDs to cache */
    int* edge_num_unique_sigs;  /* Number of unique signatures per cached edge */
    double* edge_reuse_rates;   /* Reuse rate per cached edge */
    int* edge_subtree_sizes;    /* Subtree size (num leaves) per cached edge */
    int total_cache_slots;      /* Total number of cache slots needed */
    bool loaded;
} CacheSpecification;

static CacheSpecification cache_spec = {0};

/* Load cache specification from file */
int cuda_load_cache_spec(const char* cache_spec_file) {
    if (cache_spec.loaded) {
        cuda_free_cache_spec();
    }

    FILE* fp = fopen(cache_spec_file, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open cache specification file: %s\n", cache_spec_file);
        return -1;
    }

    printf("Loading cache specification from: %s\n", cache_spec_file);

    char line[256];
    int num_cached_edges_allocated = 0;

    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') continue;

        if (strncmp(line, "NUM_PATTERNS", 12) == 0) {
            sscanf(line, "NUM_PATTERNS %d", &cache_spec.num_patterns);
        } else if (strncmp(line, "NUM_EDGES", 9) == 0) {
            sscanf(line, "NUM_EDGES %d", &cache_spec.num_edges);
        } else if (strncmp(line, "NUM_CACHED_EDGES", 16) == 0) {
            sscanf(line, "NUM_CACHED_EDGES %d", &cache_spec.num_cached_edges);
            /* Allocate arrays */
            cache_spec.cached_edge_ids = (int*)malloc(cache_spec.num_cached_edges * sizeof(int));
            cache_spec.edge_num_unique_sigs = (int*)malloc(cache_spec.num_cached_edges * sizeof(int));
            cache_spec.edge_reuse_rates = (double*)malloc(cache_spec.num_cached_edges * sizeof(double));
            cache_spec.edge_subtree_sizes = (int*)malloc(cache_spec.num_cached_edges * sizeof(int));
        } else if (strncmp(line, "CACHE_EDGE", 10) == 0) {
            if (num_cached_edges_allocated < cache_spec.num_cached_edges) {
                int edge_id, num_unique, subtree_size;
                double reuse_rate;
                sscanf(line, "CACHE_EDGE %d %d %lf %d",
                       &edge_id, &num_unique, &reuse_rate, &subtree_size);
                cache_spec.cached_edge_ids[num_cached_edges_allocated] = edge_id;
                cache_spec.edge_num_unique_sigs[num_cached_edges_allocated] = num_unique;
                cache_spec.edge_reuse_rates[num_cached_edges_allocated] = reuse_rate;
                cache_spec.edge_subtree_sizes[num_cached_edges_allocated] = subtree_size;
                num_cached_edges_allocated++;
            }
        } else if (strncmp(line, "TOTAL_CACHE_SLOTS", 17) == 0) {
            sscanf(line, "TOTAL_CACHE_SLOTS %d", &cache_spec.total_cache_slots);
        }
    }

    fclose(fp);

    if (num_cached_edges_allocated != cache_spec.num_cached_edges) {
        fprintf(stderr, "WARNING: Expected %d cached edges but found %d\n",
                cache_spec.num_cached_edges, num_cached_edges_allocated);
        cache_spec.num_cached_edges = num_cached_edges_allocated;
    }

    cache_spec.loaded = true;

    printf("  Cache specification loaded:\n");
    printf("    Expected patterns: %d\n", cache_spec.num_patterns);
    printf("    Expected edges: %d\n", cache_spec.num_edges);
    printf("    Cached edges: %d\n", cache_spec.num_cached_edges);
    printf("    Total cache slots: %d\n", cache_spec.total_cache_slots);
    printf("    Estimated memory: %.2f MB\n",
           cache_spec.total_cache_slots * 4 * sizeof(double) / 1048576.0);

    return 0;
}

/* Free cache specification memory */
void cuda_free_cache_spec(void) {
    if (!cache_spec.loaded) return;

    free(cache_spec.cached_edge_ids);
    free(cache_spec.edge_num_unique_sigs);
    free(cache_spec.edge_reuse_rates);
    free(cache_spec.edge_subtree_sizes);

    memset(&cache_spec, 0, sizeof(CacheSpecification));
    printf("Cache specification freed\n");
}

/* Initialize selective subtree memoization using pre-computed cache spec */
int cuda_init_selective_subtree_memoization(SEM* sem, const char* cache_spec_file) {
    if (subtree_memo.initialized) {
        printf("CUDA: Subtree memoization already initialized, cleaning up first...\n");
        cuda_cleanup_subtree_memoization();
    }

    if (!cuda_data.initialized) {
        fprintf(stderr, "CUDA: Must initialize cuda_data first\n");
        return -1;
    }

    /* Load cache specification if not already loaded */
    if (!cache_spec.loaded) {
        if (cuda_load_cache_spec(cache_spec_file) != 0) {
            return -1;
        }
    }

    printf("CUDA: Initializing SELECTIVE subtree-level memoization...\n");

    int num_edges = cuda_data.num_edges;
    int num_patterns = cuda_data.num_patterns;
    int num_taxa = cuda_data.num_taxa;

    /* Validate cache spec matches current SEM */
    if (cache_spec.num_edges != num_edges) {
        fprintf(stderr, "ERROR: Cache spec expects %d edges but SEM has %d edges\n",
                cache_spec.num_edges, num_edges);
        return -1;
    }

    if (cache_spec.num_patterns != num_patterns) {
        fprintf(stderr, "WARNING: Cache spec expects %d patterns but SEM has %d patterns\n",
                cache_spec.num_patterns, num_patterns);
        /* Continue anyway - patterns might be slightly different */
    }

    /* Step 1: Compute subtree leaves for each edge */
    printf("  Computing subtree leaves for %d edges...\n", num_edges);

    subtree_memo.h_edge_num_subtree_leaves = (int*)malloc(num_edges * sizeof(int));
    subtree_memo.h_edge_subtree_offsets = (int*)malloc((num_edges + 1) * sizeof(int));

    /* First pass: count leaves per edge */
    int* temp_leaf_buffer = (int*)malloc(num_taxa * sizeof(int));
    int total_subtree_leaves = 0;

    subtree_memo.h_edge_subtree_offsets[0] = 0;
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        collect_subtree_leaves_recursive(sem, child, temp_leaf_buffer, &count);
        subtree_memo.h_edge_num_subtree_leaves[e] = count;
        total_subtree_leaves += count;
        subtree_memo.h_edge_subtree_offsets[e + 1] = total_subtree_leaves;
    }

    /* Second pass: store leaf indices */
    subtree_memo.h_edge_subtree_leaves = (int*)malloc(total_subtree_leaves * sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        int offset = subtree_memo.h_edge_subtree_offsets[e];
        collect_subtree_leaves_recursive(sem, child,
                                         &subtree_memo.h_edge_subtree_leaves[offset], &count);
    }
    free(temp_leaf_buffer);

    /* Step 2: Unpack pattern data for CPU processing */
    uint8_t* h_patterns = (uint8_t*)malloc(num_patterns * num_taxa * sizeof(uint8_t));
    for (int p = 0; p < num_patterns; p++) {
        for (int t = 0; t < num_taxa; t++) {
            h_patterns[p * num_taxa + t] =
                packed_storage_get_base(sem->packed_patterns, p, t);
        }
    }

    /* Step 3: Create edge-to-cache-index mapping */
    /* Only edges in cache_spec.cached_edge_ids will have cache slots */
    int* edge_to_cache_start = (int*)malloc(num_edges * sizeof(int));  /* -1 if not cached */
    for (int e = 0; e < num_edges; e++) {
        edge_to_cache_start[e] = -1;  /* Mark as not cached */
    }

    /* Assign cache slot ranges for each cached edge */
    int current_cache_slot = 0;
    for (int i = 0; i < cache_spec.num_cached_edges; i++) {
        int edge_id = cache_spec.cached_edge_ids[i];
        if (edge_id >= 0 && edge_id < num_edges) {
            edge_to_cache_start[edge_id] = current_cache_slot;
            current_cache_slot += cache_spec.edge_num_unique_sigs[i];
        }
    }

    printf("  Caching %d edges (out of %d total)\n", cache_spec.num_cached_edges, num_edges);
    printf("  Total cache slots: %d\n", current_cache_slot);

    /* Step 4: Compute subtree signatures ONLY for cached edges */
    printf("  Computing subtree signatures for cached edges only...\n");

    subtree_memo.h_subtree_signatures = (int*)malloc(num_patterns * num_edges * sizeof(int));
    int* h_pattern_edge_to_cache = (int*)malloc(num_patterns * num_edges * sizeof(int));

    /* Initialize all to -1 (not cached) */
    for (int i = 0; i < num_patterns * num_edges; i++) {
        h_pattern_edge_to_cache[i] = -1;
        subtree_memo.h_subtree_signatures[i] = 0;
    }

    /* Hash table for signature deduplication */
    int hash_table_size = num_patterns * 2;
    if (hash_table_size < 1024) hash_table_size = 1024;
    int* hash_keys = (int*)malloc(hash_table_size * sizeof(int));
    int* hash_slots = (int*)malloc(hash_table_size * sizeof(int));

    int num_cache_slots = 0;
    long long total_savings = 0;

    /* Process only cached edges */
    for (int ci = 0; ci < cache_spec.num_cached_edges; ci++) {
        int e = cache_spec.cached_edge_ids[ci];
        if (e < 0 || e >= num_edges) continue;

        /* Reset hash table for this edge */
        for (int i = 0; i < hash_table_size; i++) {
            hash_keys[i] = -1;
        }

        int unique_this_edge = 0;

        for (int p = 0; p < num_patterns; p++) {
            /* Compute signature */
            int offset = subtree_memo.h_edge_subtree_offsets[e];
            int num_leaves = subtree_memo.h_edge_num_subtree_leaves[e];
            int sig = compute_subtree_signature(p, num_taxa, h_patterns,
                                               &subtree_memo.h_edge_subtree_leaves[offset],
                                               num_leaves);
            subtree_memo.h_subtree_signatures[p * num_edges + e] = sig;

            /* Hash table lookup with linear probing */
            unsigned int hash_idx = ((unsigned int)sig * 2654435761u) % hash_table_size;
            int found_slot = -1;
            int probes = 0;

            while (hash_keys[hash_idx] != -1 && probes < hash_table_size) {
                if (hash_keys[hash_idx] == sig) {
                    found_slot = hash_slots[hash_idx];
                    total_savings++;
                    break;
                }
                hash_idx = (hash_idx + 1) % hash_table_size;
                probes++;
            }

            if (found_slot < 0) {
                /* New signature */
                found_slot = num_cache_slots++;
                hash_keys[hash_idx] = sig;
                hash_slots[hash_idx] = found_slot;
                unique_this_edge++;
            }

            h_pattern_edge_to_cache[p * num_edges + e] = found_slot;
        }

        /* Progress indicator */
        if ((ci + 1) % 10 == 0 || ci == cache_spec.num_cached_edges - 1) {
            printf("\r    Processed %d/%d cached edges...", ci + 1, cache_spec.num_cached_edges);
            fflush(stdout);
        }
    }
    printf("\n");

    free(hash_keys);
    free(hash_slots);
    free(edge_to_cache_start);

    subtree_memo.num_cache_slots = num_cache_slots;

    printf("  Selective memoization results:\n");
    printf("    Cache slots used: %d\n", num_cache_slots);
    printf("    Total (pattern, edge) pairs for cached edges: %d\n",
           num_patterns * cache_spec.num_cached_edges);
    printf("    Message reuse: %lld times (%.1f%% of cached operations)\n",
           total_savings,
           100.0 * total_savings / (num_patterns * cache_spec.num_cached_edges));

    /* Step 5: Allocate GPU memory */
    printf("  Allocating GPU memory for selective subtree memoization...\n");

    cudaMalloc(&subtree_memo.d_edge_subtree_leaves, total_subtree_leaves * sizeof(int));
    cudaMalloc(&subtree_memo.d_edge_subtree_offsets, (num_edges + 1) * sizeof(int));
    cudaMalloc(&subtree_memo.d_edge_num_subtree_leaves, num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_subtree_signatures, num_patterns * num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_pattern_edge_to_cache, num_patterns * num_edges * sizeof(int));
    cudaMalloc(&subtree_memo.d_subtree_cached_messages, num_cache_slots * 4 * sizeof(double));
    cudaMalloc(&subtree_memo.d_cache_valid, num_cache_slots * sizeof(int));

    /* Copy to GPU */
    cudaMemcpy(subtree_memo.d_edge_subtree_leaves, subtree_memo.h_edge_subtree_leaves,
               total_subtree_leaves * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_edge_subtree_offsets, subtree_memo.h_edge_subtree_offsets,
               (num_edges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_edge_num_subtree_leaves, subtree_memo.h_edge_num_subtree_leaves,
               num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_subtree_signatures, subtree_memo.h_subtree_signatures,
               num_patterns * num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subtree_memo.d_pattern_edge_to_cache, h_pattern_edge_to_cache,
               num_patterns * num_edges * sizeof(int), cudaMemcpyHostToDevice);

    /* Initialize cache as invalid */
    cudaMemset(subtree_memo.d_cache_valid, 0, num_cache_slots * sizeof(int));

    free(h_patterns);
    free(h_pattern_edge_to_cache);

    double cache_memory_mb = (num_cache_slots * 4 * sizeof(double) +
                              num_cache_slots * sizeof(int)) / 1048576.0;
    printf("  Selective subtree memoization cache: %.2f MB\n", cache_memory_mb);
    printf("  Memory savings vs full cache: %.1f%%\n",
           100.0 * (1.0 - (double)num_cache_slots / (num_patterns * num_edges)));

    subtree_memo.initialized = true;
    printf("CUDA: Selective subtree-level memoization initialized\n");

    return 0;
}

/* CUDA kernel: compute unique cache slots only (subtree-level memoization)
 * This kernel processes CACHE SLOTS, not patterns.
 * Each cache slot represents a unique (edge, subtree_signature) pair.
 * The idea: instead of computing all pattern×edge = N×E operations,
 * we only compute unique_slots << N×E operations.
 */
__global__ void cuda_subtree_memoized_kernel(
    int num_cache_slots,
    int num_patterns,
    int num_taxa,
    int num_vertices,
    int num_edges,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ pattern_edge_to_cache,
    const double* __restrict__ transition_matrices,
    const int* __restrict__ edge_parent_ids,
    const int* __restrict__ edge_child_ids,
    const uint8_t* __restrict__ vertex_observed,
    const int* __restrict__ vertex_out_degree,
    const int* __restrict__ pattern_idx_to_vertex,
    double* cached_messages,
    int* cache_computed)
{
    int slot_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_idx >= num_cache_slots) return;

    /* Find which pattern and edge this slot corresponds to */
    /* We need to find ANY (pattern, edge) pair that maps to this slot */
    /* This is stored in the reverse mapping we'll create */
    /* For now, mark as computed - actual computation done at pattern level */
    cache_computed[slot_idx] = 1;
}

/* CUDA kernel: compute log-likelihood with subtree memoization
 * This processes each pattern, but reuses cached messages where possible.
 * Since the cache lookup is the expensive part, we parallelize over patterns.
 */
__global__ void cuda_pruning_with_subtree_memo_kernel(
    int num_patterns,
    int num_taxa,
    int num_vertices,
    int num_edges,
    int root_id,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ pattern_weights,
    const double* __restrict__ transition_matrices,
    const int* __restrict__ edge_parent_ids,
    const int* __restrict__ edge_child_ids,
    const uint8_t* __restrict__ vertex_observed,
    const int* __restrict__ vertex_out_degree,
    const int* __restrict__ pattern_idx_to_vertex,
    const double* __restrict__ root_probability,
    const int* __restrict__ pattern_edge_to_cache,
    double* __restrict__ cached_messages,
    int* __restrict__ cache_valid,
    double* pattern_log_likes)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    /* Thread-local storage */
    double cond_like[128 * 4];
    uint8_t vertex_to_base[128];
    bool cond_like_initialized[128];
    double vertex_scaling[128];

    /* Initialize */
    for (int i = 0; i < num_vertices; i++) {
        vertex_to_base[i] = 4;
        cond_like_initialized[i] = false;
        cond_like[i * 4 + 0] = 0.0;
        cond_like[i * 4 + 1] = 0.0;
        cond_like[i * 4 + 2] = 0.0;
        cond_like[i * 4 + 3] = 0.0;
        vertex_scaling[i] = 0.0;
    }

    /* Build vertex_to_base from pattern */
    for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
        int vertex_id = pattern_idx_to_vertex[taxon_idx];
        if (vertex_id >= 0 && vertex_id < num_vertices) {
            vertex_to_base[vertex_id] = patterns[pat_idx * num_taxa + taxon_idx];
        }
    }

    /* Process edges with memoization */
    for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
        int p_id = edge_parent_ids[edge_idx];
        int c_id = edge_child_ids[edge_idx];
        const double* P = &transition_matrices[edge_idx * 16];

        /* Accumulate child's scaling factor to parent */
        vertex_scaling[p_id] += vertex_scaling[c_id];

        /* Initialize leaf child */
        if (vertex_out_degree[c_id] == 0) {
            uint8_t base = vertex_to_base[c_id];
            if (base < 4) {
                cond_like[c_id * 4 + 0] = 0.0;
                cond_like[c_id * 4 + 1] = 0.0;
                cond_like[c_id * 4 + 2] = 0.0;
                cond_like[c_id * 4 + 3] = 0.0;
                cond_like[c_id * 4 + base] = 1.0;
            } else {
                cond_like[c_id * 4 + 0] = 1.0;
                cond_like[c_id * 4 + 1] = 1.0;
                cond_like[c_id * 4 + 2] = 1.0;
                cond_like[c_id * 4 + 3] = 1.0;
            }
            cond_like_initialized[c_id] = true;
        }

        /* Initialize parent if needed */
        if (!cond_like_initialized[p_id]) {
            if (!vertex_observed[p_id]) {
                cond_like[p_id * 4 + 0] = 1.0;
                cond_like[p_id * 4 + 1] = 1.0;
                cond_like[p_id * 4 + 2] = 1.0;
                cond_like[p_id * 4 + 3] = 1.0;
            } else {
                uint8_t base = vertex_to_base[p_id];
                if (base < 4) {
                    cond_like[p_id * 4 + 0] = 0.0;
                    cond_like[p_id * 4 + 1] = 0.0;
                    cond_like[p_id * 4 + 2] = 0.0;
                    cond_like[p_id * 4 + 3] = 0.0;
                    cond_like[p_id * 4 + base] = 1.0;
                } else {
                    cond_like[p_id * 4 + 0] = 1.0;
                    cond_like[p_id * 4 + 1] = 1.0;
                    cond_like[p_id * 4 + 2] = 1.0;
                    cond_like[p_id * 4 + 3] = 1.0;
                }
            }
            cond_like_initialized[p_id] = true;
        }

        /* Check if cached message is available */
        int cache_slot = pattern_edge_to_cache[pat_idx * num_edges + edge_idx];

        /* Always compute message (don't use cache for now - correctness first)
         * The cache hit/miss logic has race conditions that need careful handling
         */
        double message[4];
        message[0] = P[0] * cond_like[c_id * 4 + 0] +
                    P[1] * cond_like[c_id * 4 + 1] +
                    P[2] * cond_like[c_id * 4 + 2] +
                    P[3] * cond_like[c_id * 4 + 3];
        message[1] = P[4] * cond_like[c_id * 4 + 0] +
                    P[5] * cond_like[c_id * 4 + 1] +
                    P[6] * cond_like[c_id * 4 + 2] +
                    P[7] * cond_like[c_id * 4 + 3];
        message[2] = P[8] * cond_like[c_id * 4 + 0] +
                    P[9] * cond_like[c_id * 4 + 1] +
                    P[10] * cond_like[c_id * 4 + 2] +
                    P[11] * cond_like[c_id * 4 + 3];
        message[3] = P[12] * cond_like[c_id * 4 + 0] +
                    P[13] * cond_like[c_id * 4 + 1] +
                    P[14] * cond_like[c_id * 4 + 2] +
                    P[15] * cond_like[c_id * 4 + 3];

        /* Mark this cache slot as used (for statistics) */
        cache_valid[cache_slot] = 1;

        /* Apply message to parent */
        double largest = 0.0;
        for (int dna_p = 0; dna_p < 4; dna_p++) {
            cond_like[p_id * 4 + dna_p] *= message[dna_p];
            if (cond_like[p_id * 4 + dna_p] > largest) {
                largest = cond_like[p_id * 4 + dna_p];
            }
        }

        /* Scale to prevent underflow */
        if (largest > 0.0) {
            cond_like[p_id * 4 + 0] /= largest;
            cond_like[p_id * 4 + 1] /= largest;
            cond_like[p_id * 4 + 2] /= largest;
            cond_like[p_id * 4 + 3] /= largest;
            vertex_scaling[p_id] += log(largest);
        }
    }

    /* Compute site likelihood at root */
    double site_likelihood = 0.0;
    site_likelihood += root_probability[0] * cond_like[root_id * 4 + 0];
    site_likelihood += root_probability[1] * cond_like[root_id * 4 + 1];
    site_likelihood += root_probability[2] * cond_like[root_id * 4 + 2];
    site_likelihood += root_probability[3] * cond_like[root_id * 4 + 3];

    /* Compute weighted log-likelihood for this pattern */
    if (site_likelihood > 0.0) {
        double log_site_like = vertex_scaling[root_id] + log(site_likelihood);
        pattern_log_likes[pat_idx] = log_site_like * pattern_weights[pat_idx];
    } else {
        pattern_log_likes[pat_idx] = -1e30;
    }
}

/* Compute log-likelihood with subtree-level memoization */
double cuda_compute_log_likelihood_subtree_memoized(SEM* sem) {
    if (!cuda_data.initialized) {
        if (cuda_init_pruning(sem) != 0) {
            fprintf(stderr, "CUDA: Failed to initialize\n");
            return -1e30;
        }
    }

    if (!subtree_memo.initialized) {
        if (cuda_init_subtree_memoization(sem) != 0) {
            fprintf(stderr, "CUDA: Failed to initialize subtree memoization\n");
            return -1e30;
        }
    }

    cudaDeviceSynchronize();

    /* Reset total log-likelihood */
    double zero = 0.0;
    cudaMemcpy(cuda_data.d_total_log_likelihood, &zero, sizeof(double),
               cudaMemcpyHostToDevice);

    /* Reset cache validity (need fresh computation after parameter update) */
    cudaMemset(subtree_memo.d_cache_valid, 0,
               subtree_memo.num_cache_slots * sizeof(int));

    /* Record timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* Launch kernel with subtree memoization */
    int block_size = 128;
    int num_blocks = (cuda_data.num_patterns + block_size - 1) / block_size;

    cuda_pruning_with_subtree_memo_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        cuda_data.num_taxa,
        cuda_data.num_vertices,
        cuda_data.num_edges,
        cuda_data.root_id,
        cuda_data.d_patterns,
        cuda_data.d_pattern_weights,
        cuda_data.d_transition_matrices,
        cuda_data.d_edge_parent_ids,
        cuda_data.d_edge_child_ids,
        cuda_data.d_vertex_observed,
        cuda_data.d_vertex_out_degree,
        cuda_data.d_pattern_idx_to_vertex,
        cuda_data.d_root_probability,
        subtree_memo.d_pattern_edge_to_cache,
        subtree_memo.d_subtree_cached_messages,
        subtree_memo.d_cache_valid,
        cuda_data.d_pattern_log_likes
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1e30;
    }

    /* Reduction to compute total log-likelihood */
    cuda_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
        cuda_data.d_pattern_log_likes,
        cuda_data.num_patterns,
        cuda_data.d_total_log_likelihood
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);
    printf("CUDA subtree-memoized kernel time: %.4f ms (%d cache slots for %d patterns)\n",
           kernel_time_ms, subtree_memo.num_cache_slots, cuda_data.num_patterns);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA reduction failed: %s\n", cudaGetErrorString(err));
        return -1e30;
    }

    /* Copy result back */
    double total_log_likelihood;
    cudaMemcpy(&total_log_likelihood, cuda_data.d_total_log_likelihood,
               sizeof(double), cudaMemcpyDeviceToHost);

    return total_log_likelihood;
}

/* Initialize memoization structures */
int cuda_init_memoization(SEM* sem) {
    if (memo_data.initialized) return 0;

    printf("CUDA: Initializing memoization structures...\n");

    /* Compute signatures for all patterns */
    int* h_signatures = (int*)malloc(sem->num_patterns * sizeof(int));
    for (int p = 0; p < sem->num_patterns; p++) {
        h_signatures[p] = compute_pattern_signature(sem, p);
    }

    /* Find unique signatures and create mapping */
    int* h_unique_sigs = (int*)malloc(sem->num_patterns * sizeof(int));
    int* h_pattern_to_sig_idx = (int*)malloc(sem->num_patterns * sizeof(int));
    int* h_sig_to_pattern = (int*)malloc(sem->num_patterns * sizeof(int));

    int num_unique = 0;
    for (int p = 0; p < sem->num_patterns; p++) {
        int sig = h_signatures[p];
        int found_idx = -1;

        /* Check if signature already seen */
        for (int u = 0; u < num_unique; u++) {
            if (h_unique_sigs[u] == sig) {
                found_idx = u;
                break;
            }
        }

        if (found_idx < 0) {
            /* New unique signature */
            found_idx = num_unique;
            h_unique_sigs[num_unique] = sig;
            h_sig_to_pattern[num_unique] = p;  /* First pattern with this signature */
            num_unique++;
        }
        h_pattern_to_sig_idx[p] = found_idx;
    }

    memo_data.num_unique_signatures = num_unique;
    printf("CUDA: Found %d unique signatures from %d patterns (%.1f%% reduction)\n",
           num_unique, sem->num_patterns, 100.0 * (1.0 - (double)num_unique / sem->num_patterns));

    /* Allocate GPU memory for memoization */
    cudaMalloc(&memo_data.d_pattern_signatures, sem->num_patterns * sizeof(int));
    cudaMalloc(&memo_data.d_unique_signatures, num_unique * sizeof(int));
    cudaMalloc(&memo_data.d_pattern_to_signature_idx, sem->num_patterns * sizeof(int));
    cudaMalloc(&memo_data.d_signature_to_pattern, num_unique * sizeof(int));

    /* Cache storage: for each unique signature, store messages for all edges
     * Each edge message is 4 doubles + 1 double scaling factor
     */
    size_t cache_size = num_unique * cuda_data.num_edges * 4 * sizeof(double);
    size_t scale_size = num_unique * cuda_data.num_edges * sizeof(double);
    cudaMalloc(&memo_data.d_cached_messages, cache_size);
    cudaMalloc(&memo_data.d_cached_scales, scale_size);

    /* Copy data to GPU */
    cudaMemcpy(memo_data.d_pattern_signatures, h_signatures,
               sem->num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(memo_data.d_unique_signatures, h_unique_sigs,
               num_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(memo_data.d_pattern_to_signature_idx, h_pattern_to_sig_idx,
               sem->num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(memo_data.d_signature_to_pattern, h_sig_to_pattern,
               num_unique * sizeof(int), cudaMemcpyHostToDevice);

    /* Initialize cache to invalid state (-1 for signatures means not computed) */
    cudaMemset(memo_data.d_cached_messages, 0, cache_size);
    cudaMemset(memo_data.d_cached_scales, 0, scale_size);

    free(h_signatures);
    free(h_unique_sigs);
    free(h_pattern_to_sig_idx);
    free(h_sig_to_pattern);

    memo_data.initialized = true;
    printf("CUDA: Memoization initialized with %.2f MB cache\n",
           (cache_size + scale_size) / 1048576.0);

    return 0;
}

/* CUDA kernel: compute messages for unique signatures only (memoization) */
__global__ void cuda_memoized_pruning_kernel(
    int num_unique_signatures,
    int num_taxa,
    int num_vertices,
    int num_edges,
    int root_id,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ signature_to_pattern,
    const double* __restrict__ transition_matrices,
    const int* __restrict__ edge_parent_ids,
    const int* __restrict__ edge_child_ids,
    const uint8_t* __restrict__ vertex_observed,
    const int* __restrict__ vertex_out_degree,
    const int* __restrict__ pattern_idx_to_vertex,
    const double* __restrict__ root_probability,
    double* cached_messages,
    double* cached_scales,
    double* signature_log_likes)
{
    int sig_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sig_idx >= num_unique_signatures) return;

    /* Get representative pattern for this signature */
    int pat_idx = signature_to_pattern[sig_idx];

    /* Thread-local storage */
    double cond_like[128 * 4];
    uint8_t vertex_to_base[128];
    bool cond_like_initialized[128];
    double vertex_scaling[128];

    /* Initialize */
    for (int i = 0; i < num_vertices; i++) {
        vertex_to_base[i] = 4;
        cond_like_initialized[i] = false;
        cond_like[i * 4 + 0] = 0.0;
        cond_like[i * 4 + 1] = 0.0;
        cond_like[i * 4 + 2] = 0.0;
        cond_like[i * 4 + 3] = 0.0;
        vertex_scaling[i] = 0.0;
    }

    /* Build vertex_to_base from pattern */
    for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
        int vertex_id = pattern_idx_to_vertex[taxon_idx];
        if (vertex_id >= 0 && vertex_id < num_vertices) {
            vertex_to_base[vertex_id] = patterns[pat_idx * num_taxa + taxon_idx];
        }
    }

    /* Process edges and cache messages */
    for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
        int p_id = edge_parent_ids[edge_idx];
        int c_id = edge_child_ids[edge_idx];
        const double* P = &transition_matrices[edge_idx * 16];

        vertex_scaling[p_id] += vertex_scaling[c_id];

        /* Initialize leaf */
        if (vertex_out_degree[c_id] == 0) {
            uint8_t base = vertex_to_base[c_id];
            if (base < 4) {
                cond_like[c_id * 4 + 0] = 0.0;
                cond_like[c_id * 4 + 1] = 0.0;
                cond_like[c_id * 4 + 2] = 0.0;
                cond_like[c_id * 4 + 3] = 0.0;
                cond_like[c_id * 4 + base] = 1.0;
            } else {
                cond_like[c_id * 4 + 0] = 1.0;
                cond_like[c_id * 4 + 1] = 1.0;
                cond_like[c_id * 4 + 2] = 1.0;
                cond_like[c_id * 4 + 3] = 1.0;
            }
            cond_like_initialized[c_id] = true;
        }

        /* Initialize parent */
        if (!cond_like_initialized[p_id]) {
            if (!vertex_observed[p_id]) {
                cond_like[p_id * 4 + 0] = 1.0;
                cond_like[p_id * 4 + 1] = 1.0;
                cond_like[p_id * 4 + 2] = 1.0;
                cond_like[p_id * 4 + 3] = 1.0;
            } else {
                uint8_t base = vertex_to_base[p_id];
                if (base < 4) {
                    cond_like[p_id * 4 + 0] = 0.0;
                    cond_like[p_id * 4 + 1] = 0.0;
                    cond_like[p_id * 4 + 2] = 0.0;
                    cond_like[p_id * 4 + 3] = 0.0;
                    cond_like[p_id * 4 + base] = 1.0;
                } else {
                    cond_like[p_id * 4 + 0] = 1.0;
                    cond_like[p_id * 4 + 1] = 1.0;
                    cond_like[p_id * 4 + 2] = 1.0;
                    cond_like[p_id * 4 + 3] = 1.0;
                }
            }
            cond_like_initialized[p_id] = true;
        }

        /* Compute message (child -> parent) */
        double message[4];
        double largest = 0.0;
        for (int dna_p = 0; dna_p < 4; dna_p++) {
            double partial = 0.0;
            partial += P[dna_p * 4 + 0] * cond_like[c_id * 4 + 0];
            partial += P[dna_p * 4 + 1] * cond_like[c_id * 4 + 1];
            partial += P[dna_p * 4 + 2] * cond_like[c_id * 4 + 2];
            partial += P[dna_p * 4 + 3] * cond_like[c_id * 4 + 3];
            message[dna_p] = partial;
            cond_like[p_id * 4 + dna_p] *= partial;
            if (cond_like[p_id * 4 + dna_p] > largest) {
                largest = cond_like[p_id * 4 + dna_p];
            }
        }

        /* Cache the message for this signature and edge */
        int cache_idx = sig_idx * num_edges + edge_idx;
        cached_messages[cache_idx * 4 + 0] = message[0];
        cached_messages[cache_idx * 4 + 1] = message[1];
        cached_messages[cache_idx * 4 + 2] = message[2];
        cached_messages[cache_idx * 4 + 3] = message[3];

        /* Scale and cache scaling factor */
        if (largest > 0.0) {
            cond_like[p_id * 4 + 0] /= largest;
            cond_like[p_id * 4 + 1] /= largest;
            cond_like[p_id * 4 + 2] /= largest;
            cond_like[p_id * 4 + 3] /= largest;
            vertex_scaling[p_id] += log(largest);
        }
        cached_scales[cache_idx] = vertex_scaling[p_id];
    }

    /* Compute likelihood for this signature */
    double site_likelihood = 0.0;
    site_likelihood += root_probability[0] * cond_like[root_id * 4 + 0];
    site_likelihood += root_probability[1] * cond_like[root_id * 4 + 1];
    site_likelihood += root_probability[2] * cond_like[root_id * 4 + 2];
    site_likelihood += root_probability[3] * cond_like[root_id * 4 + 3];

    if (site_likelihood > 0.0) {
        signature_log_likes[sig_idx] = vertex_scaling[root_id] + log(site_likelihood);
    } else {
        signature_log_likes[sig_idx] = -1e30;
    }
}

/* CUDA kernel: apply cached results to all patterns */
__global__ void cuda_apply_memoized_results_kernel(
    int num_patterns,
    const int* __restrict__ pattern_to_signature_idx,
    const int* __restrict__ pattern_weights,
    const double* __restrict__ signature_log_likes,
    double* pattern_log_likes)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    int sig_idx = pattern_to_signature_idx[pat_idx];
    double log_like = signature_log_likes[sig_idx];
    pattern_log_likes[pat_idx] = log_like * pattern_weights[pat_idx];
}

/* Compute log-likelihood using memoized CUDA */
double cuda_compute_log_likelihood_memoized(SEM* sem) {
    if (!cuda_data.initialized) {
        if (cuda_init_pruning(sem) != 0) {
            fprintf(stderr, "CUDA: Failed to initialize\n");
            return -1e30;
        }
    }

    if (!memo_data.initialized) {
        if (cuda_init_memoization(sem) != 0) {
            fprintf(stderr, "CUDA: Failed to initialize memoization\n");
            return -1e30;
        }
    }

    cudaDeviceSynchronize();

    /* Allocate temporary storage for signature likelihoods */
    double* d_signature_log_likes;
    cudaMalloc(&d_signature_log_likes, memo_data.num_unique_signatures * sizeof(double));

    /* Reset total log-likelihood */
    double zero = 0.0;
    cudaMemcpy(cuda_data.d_total_log_likelihood, &zero, sizeof(double),
               cudaMemcpyHostToDevice);

    /* Record timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* Phase 1: Compute messages for unique signatures only */
    int block_size = 128;
    int num_blocks_sigs = (memo_data.num_unique_signatures + block_size - 1) / block_size;

    cuda_memoized_pruning_kernel<<<num_blocks_sigs, block_size>>>(
        memo_data.num_unique_signatures,
        cuda_data.num_taxa,
        cuda_data.num_vertices,
        cuda_data.num_edges,
        cuda_data.root_id,
        cuda_data.d_patterns,
        memo_data.d_signature_to_pattern,
        cuda_data.d_transition_matrices,
        cuda_data.d_edge_parent_ids,
        cuda_data.d_edge_child_ids,
        cuda_data.d_vertex_observed,
        cuda_data.d_vertex_out_degree,
        cuda_data.d_pattern_idx_to_vertex,
        cuda_data.d_root_probability,
        memo_data.d_cached_messages,
        memo_data.d_cached_scales,
        d_signature_log_likes
    );

    /* Phase 2: Apply cached results to all patterns */
    int num_blocks_pats = (cuda_data.num_patterns + block_size - 1) / block_size;
    cuda_apply_memoized_results_kernel<<<num_blocks_pats, block_size>>>(
        cuda_data.num_patterns,
        memo_data.d_pattern_to_signature_idx,
        cuda_data.d_pattern_weights,
        d_signature_log_likes,
        cuda_data.d_pattern_log_likes
    );

    /* Phase 3: Reduce to total log-likelihood */
    cuda_reduction_kernel<<<num_blocks_pats, block_size, block_size * sizeof(double)>>>(
        cuda_data.d_pattern_log_likes,
        cuda_data.num_patterns,
        cuda_data.d_total_log_likelihood
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);
    printf("CUDA memoized kernel time: %.4f ms (computed %d unique signatures)\n",
           kernel_time_ms, memo_data.num_unique_signatures);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Copy result back */
    double total_log_likelihood;
    cudaMemcpy(&total_log_likelihood, cuda_data.d_total_log_likelihood,
               sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_signature_log_likes);

    return total_log_likelihood;
}

/* CUDA-accelerated EM with Aitken and memoization */
void cuda_run_em_with_aitken(SEM* sem, int max_iterations) {
    if (!sem) return;

    printf("\n====================================================\n");
    printf("CUDA-Accelerated EM with Aitken and Memoization\n");
    printf("====================================================\n");

    /* Initialize CUDA if needed */
    if (!cuda_data.initialized) {
        cuda_init_pruning(sem);
    }
    if (!memo_data.initialized) {
        cuda_init_memoization(sem);
    }

    /* Compute initial log-likelihood using CUDA */
    double ll_prev = cuda_compute_log_likelihood_memoized(sem);
    sem->log_likelihood = ll_prev;
    printf("Initial LL (CUDA): %.6f\n", ll_prev);

    int num_sites = 0;
    for (int i = 0; i < sem->num_patterns; i++) {
        num_sites += sem->pattern_weights[i];
    }
    double conv_threshold = 1e-5 * num_sites;

    printf("Number of sites: %d\n", num_sites);
    printf("Convergence tolerance: %.2e (%.2e per site)\n", conv_threshold, 1e-5);
    printf("----------------------------------------------------------------------\n");

    double ll_prev_prev = ll_prev;
    int final_iter = 0;
    bool converged = false;

    for (int iter = 1; iter <= max_iterations; iter++) {
        /* E-step: Use CPU belief propagation for expected counts
         * (This is where we'd benefit most from GPU, but tree structure is complex)
         */
        sem_em_iteration(sem);

        /* Update transition matrices on GPU after M-step */
        cuda_update_transition_matrices(sem);

        /* Compute new log-likelihood using CUDA with memoization */
        double ll_current = cuda_compute_log_likelihood_memoized(sem);
        sem->log_likelihood = ll_current;

        double improvement = ll_current - ll_prev;
        printf("Iter %3d: LL = %.2f (+%.2e)", iter, ll_current, improvement);

        /* Aitken acceleration check */
        if (iter >= 3) {
            double rate = (ll_current - ll_prev) / (ll_prev - ll_prev_prev + 1e-10);
            printf(" | Rate: %.3f", rate);

            if (rate > 0.0 && rate < 0.95) {
                double aitken_ll = ll_current + (ll_current - ll_prev) / (1.0 - rate);
                double dist_to_aitken = fabs(aitken_ll - ll_current);
                printf(" | Aitken LL: %.2f | Dist: %.2e", aitken_ll, dist_to_aitken);
            }
        }
        printf("\n");

        /* Check convergence */
        if (improvement < conv_threshold && improvement >= 0) {
            converged = true;
            final_iter = iter;
            printf("Converged after %d iterations\n", iter);
            break;
        }

        ll_prev_prev = ll_prev;
        ll_prev = ll_current;
        final_iter = iter;
    }

    printf("----------------------------------------------------------------------\n");
    if (!converged) {
        printf("Maximum iterations (%d) reached\n", max_iterations);
    }
    printf("Final LL (CUDA): %.11f\n", sem->log_likelihood);
    printf("====================================================\n");
}

void cuda_cleanup_memoization(void) {
    if (!memo_data.initialized) return;

    cudaFree(memo_data.d_pattern_signatures);
    cudaFree(memo_data.d_unique_signatures);
    cudaFree(memo_data.d_pattern_to_signature_idx);
    cudaFree(memo_data.d_signature_to_pattern);
    cudaFree(memo_data.d_cached_messages);
    cudaFree(memo_data.d_cached_scales);

    memo_data.initialized = false;
    printf("CUDA: Memoization cleanup complete\n");
}

/* ========== GPU-Accelerated E-Step ========== */

/* E-step data structures */
typedef struct {
    int num_cliques;
    int* d_clique_x_ids;           /* num_cliques - parent vertex ID */
    int* d_clique_y_ids;           /* num_cliques - child vertex ID */
    int* d_clique_parent_idx;      /* num_cliques - parent clique index (-1 for root) */
    int* d_clique_child_indices;   /* num_cliques * MAX_CHILDREN - child clique indices */
    int* d_clique_num_children;    /* num_cliques */
    int* d_post_order_clique;      /* num_cliques - traversal order */
    int* d_clique_y_pattern_idx;   /* num_cliques - pattern index for observed Y (-1 if not observed) */

    /* Device memory for E-step computation */
    double* d_clique_base_potentials; /* num_cliques * 16 - transition matrices */
    double* d_clique_beliefs;          /* num_patterns * num_cliques * 16 */
    double* d_clique_messages;         /* num_patterns * num_cliques * MAX_NEIGHBORS * 4 */
    double* d_clique_log_scales;       /* num_patterns * num_cliques */

    /* Expected counts (output of E-step) */
    double* d_expected_counts_vertex;  /* num_vertices * 4 */
    double* d_expected_counts_edge;    /* num_edges * 16 */

    /* Edge to clique mapping */
    int* d_edge_to_clique_idx;         /* num_edges - which clique corresponds to each edge */

    bool initialized;
} CUDAEStepData;

static CUDAEStepData estep_data = {0};

#define MAX_CLIQUE_CHILDREN 10

/* E-step kernel: Apply evidence and initialize beliefs for all patterns */
__global__ void cuda_estep_apply_evidence_kernel(
    int num_patterns,
    int num_cliques,
    const uint8_t* __restrict__ patterns,
    const int* __restrict__ clique_y_pattern_idx,
    const double* __restrict__ clique_base_potentials,
    double* beliefs,
    double* log_scales)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    /* For each clique, apply evidence for this pattern */
    for (int c = 0; c < num_cliques; c++) {
        int belief_offset = (pat_idx * num_cliques + c) * 16;
        int base_offset = c * 16;

        /* Copy base potential to initial belief */
        for (int i = 0; i < 16; i++) {
            beliefs[belief_offset + i] = clique_base_potentials[base_offset + i];
        }

        /* Apply evidence if Y is observed */
        int y_pat_idx = clique_y_pattern_idx[c];
        if (y_pat_idx >= 0) {
            uint8_t dna_y = patterns[pat_idx * (num_cliques + 1) + y_pat_idx]; /* Simplified: needs proper taxon mapping */

            if (dna_y < 4) {
                /* Zero out columns that don't match observed value */
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (j != dna_y) {
                            beliefs[belief_offset + i * 4 + j] = 0.0;
                        }
                    }
                }
            }
            /* If dna_y == 4 (gap), keep full transition matrix */
        }

        /* Initialize scaling factor */
        log_scales[pat_idx * num_cliques + c] = 0.0;
    }
}

/* E-step kernel: Send message from child to parent (upward pass) */
__device__ void cuda_send_message_upward(
    const double* child_belief,     /* 16 doubles */
    const double* child_messages,   /* messages received by child */
    int child_num_msgs,
    double* message_to_parent,      /* 4 doubles - output */
    double* log_scale)              /* 1 double - output */
{
    /* Compute factor = belief * product of incoming messages (excluding parent) */
    double factor[16];
    for (int i = 0; i < 16; i++) {
        factor[i] = child_belief[i];
    }

    /* Multiply by messages from child's children */
    /* (For upward pass, we multiply belief rows by child messages) */
    for (int m = 0; m < child_num_msgs; m++) {
        const double* msg = &child_messages[m * 4];
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                factor[x * 4 + y] *= msg[y];
            }
        }
    }

    /* Marginalize over Y (sum columns) to get message to parent */
    double scale_factor = 0.0;
    for (int x = 0; x < 4; x++) {
        message_to_parent[x] = 0.0;
        for (int y = 0; y < 4; y++) {
            message_to_parent[x] += factor[x * 4 + y];
        }
        scale_factor += message_to_parent[x];
    }

    /* Normalize for numerical stability */
    if (scale_factor > 0.0) {
        for (int x = 0; x < 4; x++) {
            message_to_parent[x] /= scale_factor;
        }
        *log_scale = log(scale_factor);
    } else {
        *log_scale = 0.0;
    }
}

/* E-step kernel: Upward pass (leaves to root) - one thread per pattern */
__global__ void cuda_estep_upward_pass_kernel(
    int num_patterns,
    int num_cliques,
    const int* __restrict__ post_order_clique,
    const int* __restrict__ clique_parent_idx,
    const int* __restrict__ clique_child_indices,
    const int* __restrict__ clique_num_children,
    double* beliefs,
    double* messages,
    double* log_scales)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    /* Process cliques in post-order (leaves to root) */
    for (int ord = 0; ord < num_cliques; ord++) {
        int c = post_order_clique[ord];
        int parent_c = clique_parent_idx[c];

        if (parent_c >= 0) {
            /* Send message from child c to parent parent_c */
            int belief_offset = (pat_idx * num_cliques + c) * 16;
            int num_children = clique_num_children[c];

            /* Gather messages from this clique's children */
            double local_msgs[MAX_CLIQUE_CHILDREN * 4];
            for (int ch = 0; ch < num_children; ch++) {
                int child_c = clique_child_indices[c * MAX_CLIQUE_CHILDREN + ch];
                /* The message from child_c to c is stored at messages[pat_idx * num_cliques + child_c] */
                int msg_offset = (pat_idx * num_cliques + child_c) * 4;
                for (int d = 0; d < 4; d++) {
                    local_msgs[ch * 4 + d] = messages[msg_offset + d];
                }
            }

            /* Compute message from c to parent */
            double msg_to_parent[4];
            double log_scale;
            cuda_send_message_upward(&beliefs[belief_offset], local_msgs, num_children,
                                      msg_to_parent, &log_scale);

            /* Store message (at this clique's slot, will be read by parent) */
            int out_msg_offset = (pat_idx * num_cliques + c) * 4;
            for (int d = 0; d < 4; d++) {
                messages[out_msg_offset + d] = msg_to_parent[d];
            }

            /* Accumulate scaling factor */
            log_scales[pat_idx * num_cliques + c] += log_scale;
        }
    }
}

/* E-step kernel: Compute final beliefs after calibration */
__global__ void cuda_estep_compute_beliefs_kernel(
    int num_patterns,
    int num_cliques,
    const int* __restrict__ clique_child_indices,
    const int* __restrict__ clique_num_children,
    const int* __restrict__ clique_parent_idx,
    double* beliefs,
    const double* __restrict__ messages_up,
    const double* __restrict__ messages_down)
{
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    /* For each clique, multiply belief by all incoming messages */
    for (int c = 0; c < num_cliques; c++) {
        int belief_offset = (pat_idx * num_cliques + c) * 16;

        /* Multiply by messages from children (upward) */
        int num_children = clique_num_children[c];
        for (int ch = 0; ch < num_children; ch++) {
            int child_c = clique_child_indices[c * MAX_CLIQUE_CHILDREN + ch];
            int msg_offset = (pat_idx * num_cliques + child_c) * 4;

            /* Message from child affects Y dimension */
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    beliefs[belief_offset + x * 4 + y] *= messages_up[msg_offset + y];
                }
            }
        }

        /* Multiply by message from parent (downward) if exists */
        int parent_c = clique_parent_idx[c];
        if (parent_c >= 0) {
            int msg_offset = (pat_idx * num_cliques + c) * 4; /* Parent's message to us */
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    beliefs[belief_offset + x * 4 + y] *= messages_down[msg_offset + x];
                }
            }
        }

        /* Normalize belief */
        double sum = 0.0;
        for (int i = 0; i < 16; i++) {
            sum += beliefs[belief_offset + i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < 16; i++) {
                beliefs[belief_offset + i] /= sum;
            }
        }
    }
}

/* E-step kernel: Accumulate expected counts from beliefs */
__global__ void cuda_estep_accumulate_counts_kernel(
    int num_patterns,
    int num_cliques,
    int num_vertices,
    const int* __restrict__ pattern_weights,
    const int* __restrict__ clique_x_ids,
    const int* __restrict__ clique_y_ids,
    const int* __restrict__ edge_to_clique_idx,
    int num_edges,
    const double* __restrict__ beliefs,
    double* expected_counts_vertex,
    double* expected_counts_edge)
{
    /* Each thread processes one pattern */
    int pat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pat_idx >= num_patterns) return;

    int weight = pattern_weights[pat_idx];

    /* Track which vertices we've counted */
    bool vertex_counted[128];
    for (int i = 0; i < num_vertices; i++) {
        vertex_counted[i] = false;
    }

    /* For each clique, add marginal probabilities to vertex counts */
    for (int c = 0; c < num_cliques; c++) {
        int belief_offset = (pat_idx * num_cliques + c) * 16;
        int x_id = clique_x_ids[c];

        /* Add marginal for X (sum over Y) if not already counted */
        if (!vertex_counted[x_id]) {
            double marginal_x[4] = {0.0, 0.0, 0.0, 0.0};
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    marginal_x[x] += beliefs[belief_offset + x * 4 + y];
                }
            }

            /* Atomic add to global counts */
            for (int d = 0; d < 4; d++) {
                atomicAddDouble(&expected_counts_vertex[x_id * 4 + d], marginal_x[d] * weight);
            }
            vertex_counted[x_id] = true;
        }
    }

    /* Add edge counts from clique beliefs */
    for (int e = 0; e < num_edges; e++) {
        int c = edge_to_clique_idx[e];
        if (c >= 0 && c < num_cliques) {
            int belief_offset = (pat_idx * num_cliques + c) * 16;

            /* Atomic add joint probability to edge counts */
            for (int i = 0; i < 16; i++) {
                atomicAddDouble(&expected_counts_edge[e * 16 + i],
                               beliefs[belief_offset + i] * weight);
            }
        }
    }
}

/* Initialize E-step data structures on GPU */
int cuda_init_estep(SEM* sem) {
    if (estep_data.initialized) return 0;

    if (!sem->clique_tree) {
        fprintf(stderr, "CUDA E-step: No clique tree constructed\n");
        return -1;
    }

    CliqueTree* ct = sem->clique_tree;
    estep_data.num_cliques = ct->num_cliques;

    printf("CUDA E-step: Initializing for %d cliques, %d patterns\n",
           estep_data.num_cliques, sem->num_patterns);

    /* Allocate host arrays for clique tree structure */
    int* h_clique_x_ids = (int*)malloc(ct->num_cliques * sizeof(int));
    int* h_clique_y_ids = (int*)malloc(ct->num_cliques * sizeof(int));
    int* h_clique_parent_idx = (int*)malloc(ct->num_cliques * sizeof(int));
    int* h_clique_child_indices = (int*)calloc(ct->num_cliques * MAX_CLIQUE_CHILDREN, sizeof(int));
    int* h_clique_num_children = (int*)malloc(ct->num_cliques * sizeof(int));
    int* h_post_order_clique = (int*)malloc(ct->num_cliques * sizeof(int));
    int* h_clique_y_pattern_idx = (int*)malloc(ct->num_cliques * sizeof(int));
    double* h_clique_base_potentials = (double*)malloc(ct->num_cliques * 16 * sizeof(double));
    int* h_edge_to_clique_idx = (int*)malloc(sem->num_post_order_edges * sizeof(int));

    if (!h_clique_x_ids || !h_clique_y_ids || !h_clique_parent_idx ||
        !h_clique_child_indices || !h_clique_num_children || !h_post_order_clique ||
        !h_clique_y_pattern_idx || !h_clique_base_potentials || !h_edge_to_clique_idx) {
        fprintf(stderr, "CUDA E-step: Host allocation failed\n");
        return -1;
    }

    /* Build clique index mapping */
    int* clique_to_idx = (int*)calloc(ct->num_cliques, sizeof(int));
    for (int i = 0; i < ct->num_cliques; i++) {
        clique_to_idx[ct->cliques[i]->id] = i;
    }

    /* Fill clique tree structure arrays */
    for (int i = 0; i < ct->num_cliques; i++) {
        Clique* c = ct->cliques[i];
        h_clique_x_ids[i] = c->x->id;
        h_clique_y_ids[i] = c->y->id;

        /* Parent clique index */
        if (c->parent && c->parent != c) {
            h_clique_parent_idx[i] = clique_to_idx[c->parent->id];
        } else {
            h_clique_parent_idx[i] = -1; /* Root clique */
        }

        /* Children clique indices */
        h_clique_num_children[i] = c->num_children;
        for (int ch = 0; ch < c->num_children && ch < MAX_CLIQUE_CHILDREN; ch++) {
            h_clique_child_indices[i * MAX_CLIQUE_CHILDREN + ch] = clique_to_idx[c->children[ch]->id];
        }

        /* Pattern index for observed Y */
        if (c->y->observed && c->y->pattern_index >= 0) {
            h_clique_y_pattern_idx[i] = c->y->pattern_index;
        } else {
            h_clique_y_pattern_idx[i] = -1;
        }

        /* Base potential (transition matrix) */
        memcpy(&h_clique_base_potentials[i * 16], c->base_potential, 16 * sizeof(double));
    }

    /* Post-order traversal */
    for (int i = 0; i < ct->traversal_size && i < ct->num_cliques; i++) {
        h_post_order_clique[i] = clique_to_idx[ct->post_order_traversal[i]->id];
    }

    /* Edge to clique mapping */
    for (int e = 0; e < sem->num_post_order_edges; e++) {
        /* Find clique corresponding to this edge */
        SEM_vertex* child_v = sem->post_order_child[e];
        h_edge_to_clique_idx[e] = -1;
        for (int i = 0; i < ct->num_cliques; i++) {
            if (ct->cliques[i]->y == child_v) {
                h_edge_to_clique_idx[e] = i;
                break;
            }
        }
    }

    free(clique_to_idx);

    /* Allocate device memory */
    cudaMalloc(&estep_data.d_clique_x_ids, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_clique_y_ids, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_clique_parent_idx, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_clique_child_indices, ct->num_cliques * MAX_CLIQUE_CHILDREN * sizeof(int));
    cudaMalloc(&estep_data.d_clique_num_children, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_post_order_clique, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_clique_y_pattern_idx, ct->num_cliques * sizeof(int));
    cudaMalloc(&estep_data.d_clique_base_potentials, ct->num_cliques * 16 * sizeof(double));
    cudaMalloc(&estep_data.d_edge_to_clique_idx, sem->num_post_order_edges * sizeof(int));

    /* Large arrays for per-pattern computation */
    size_t beliefs_size = (size_t)sem->num_patterns * ct->num_cliques * 16 * sizeof(double);
    size_t messages_size = (size_t)sem->num_patterns * ct->num_cliques * 4 * sizeof(double);
    size_t scales_size = (size_t)sem->num_patterns * ct->num_cliques * sizeof(double);

    printf("CUDA E-step: Allocating %.2f MB for beliefs, %.2f MB for messages\n",
           beliefs_size / (1024.0 * 1024.0), messages_size / (1024.0 * 1024.0));

    cudaMalloc(&estep_data.d_clique_beliefs, beliefs_size);
    cudaMalloc(&estep_data.d_clique_messages, messages_size);
    cudaMalloc(&estep_data.d_clique_log_scales, scales_size);

    /* Expected counts (reduced across all patterns) */
    cudaMalloc(&estep_data.d_expected_counts_vertex, sem->num_vertices * 4 * sizeof(double));
    cudaMalloc(&estep_data.d_expected_counts_edge, sem->num_post_order_edges * 16 * sizeof(double));

    /* Copy data to device */
    cudaMemcpy(estep_data.d_clique_x_ids, h_clique_x_ids, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_y_ids, h_clique_y_ids, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_parent_idx, h_clique_parent_idx, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_child_indices, h_clique_child_indices, ct->num_cliques * MAX_CLIQUE_CHILDREN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_num_children, h_clique_num_children, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_post_order_clique, h_post_order_clique, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_y_pattern_idx, h_clique_y_pattern_idx, ct->num_cliques * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_clique_base_potentials, h_clique_base_potentials, ct->num_cliques * 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(estep_data.d_edge_to_clique_idx, h_edge_to_clique_idx, sem->num_post_order_edges * sizeof(int), cudaMemcpyHostToDevice);

    /* Cleanup host arrays */
    free(h_clique_x_ids);
    free(h_clique_y_ids);
    free(h_clique_parent_idx);
    free(h_clique_child_indices);
    free(h_clique_num_children);
    free(h_post_order_clique);
    free(h_clique_y_pattern_idx);
    free(h_clique_base_potentials);
    free(h_edge_to_clique_idx);

    estep_data.initialized = true;
    printf("CUDA E-step: Initialization complete\n");

    return 0;
}

/* Run GPU-accelerated E-step: compute expected counts */
int cuda_compute_expected_counts(SEM* sem) {
    if (!cuda_data.initialized) {
        if (cuda_init_pruning(sem) != 0) return -1;
    }
    if (!estep_data.initialized) {
        if (cuda_init_estep(sem) != 0) return -1;
    }

    int block_size = 128;
    int num_blocks = (cuda_data.num_patterns + block_size - 1) / block_size;

    /* Zero out expected counts */
    cudaMemset(estep_data.d_expected_counts_vertex, 0, sem->num_vertices * 4 * sizeof(double));
    cudaMemset(estep_data.d_expected_counts_edge, 0, sem->num_post_order_edges * 16 * sizeof(double));
    cudaMemset(estep_data.d_clique_messages, 0,
               (size_t)sem->num_patterns * estep_data.num_cliques * 4 * sizeof(double));

    /* Phase 1: Apply evidence and initialize beliefs */
    cuda_estep_apply_evidence_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        estep_data.num_cliques,
        cuda_data.d_patterns,
        estep_data.d_clique_y_pattern_idx,
        estep_data.d_clique_base_potentials,
        estep_data.d_clique_beliefs,
        estep_data.d_clique_log_scales
    );

    /* Phase 2: Upward pass (message passing leaves to root) */
    cuda_estep_upward_pass_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        estep_data.num_cliques,
        estep_data.d_post_order_clique,
        estep_data.d_clique_parent_idx,
        estep_data.d_clique_child_indices,
        estep_data.d_clique_num_children,
        estep_data.d_clique_beliefs,
        estep_data.d_clique_messages,
        estep_data.d_clique_log_scales
    );

    /* Phase 3: Compute final beliefs (simplified - using upward messages only) */
    /* Note: For full EM, we'd need downward pass too, but this gives approximate beliefs */
    cuda_estep_compute_beliefs_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        estep_data.num_cliques,
        estep_data.d_clique_child_indices,
        estep_data.d_clique_num_children,
        estep_data.d_clique_parent_idx,
        estep_data.d_clique_beliefs,
        estep_data.d_clique_messages,
        estep_data.d_clique_messages  /* Using same for up/down for now */
    );

    /* Phase 4: Accumulate expected counts from beliefs */
    cuda_estep_accumulate_counts_kernel<<<num_blocks, block_size>>>(
        cuda_data.num_patterns,
        estep_data.num_cliques,
        cuda_data.num_vertices,
        cuda_data.d_pattern_weights,
        estep_data.d_clique_x_ids,
        estep_data.d_clique_y_ids,
        estep_data.d_edge_to_clique_idx,
        cuda_data.num_edges,
        estep_data.d_clique_beliefs,
        estep_data.d_expected_counts_vertex,
        estep_data.d_expected_counts_edge
    );

    cudaDeviceSynchronize();

    /* Copy results back to host */
    if (sem->expected_counts_for_vertex) {
        double* h_vertex_counts = (double*)malloc(sem->num_vertices * 4 * sizeof(double));
        cudaMemcpy(h_vertex_counts, estep_data.d_expected_counts_vertex,
                   sem->num_vertices * 4 * sizeof(double), cudaMemcpyDeviceToHost);
        for (int v = 0; v < sem->num_vertices; v++) {
            for (int d = 0; d < 4; d++) {
                sem->expected_counts_for_vertex[v][d] = h_vertex_counts[v * 4 + d];
            }
        }
        free(h_vertex_counts);
    }

    if (sem->expected_counts_for_edge) {
        double* h_edge_counts = (double*)malloc(sem->num_post_order_edges * 16 * sizeof(double));
        cudaMemcpy(h_edge_counts, estep_data.d_expected_counts_edge,
                   sem->num_post_order_edges * 16 * sizeof(double), cudaMemcpyDeviceToHost);
        for (int e = 0; e < sem->num_post_order_edges; e++) {
            for (int i = 0; i < 16; i++) {
                sem->expected_counts_for_edge[e][i] = h_edge_counts[e * 16 + i];
            }
        }
        free(h_edge_counts);
    }

    return 0;
}

/* Update base potentials on GPU after M-step */
int cuda_update_base_potentials(SEM* sem) {
    if (!estep_data.initialized) return -1;

    CliqueTree* ct = sem->clique_tree;
    double* h_potentials = (double*)malloc(ct->num_cliques * 16 * sizeof(double));

    for (int i = 0; i < ct->num_cliques; i++) {
        memcpy(&h_potentials[i * 16], ct->cliques[i]->base_potential, 16 * sizeof(double));
    }

    cudaMemcpy(estep_data.d_clique_base_potentials, h_potentials,
               ct->num_cliques * 16 * sizeof(double), cudaMemcpyHostToDevice);

    free(h_potentials);
    return 0;
}

/* Cleanup E-step resources */
void cuda_cleanup_estep(void) {
    if (!estep_data.initialized) return;

    cudaFree(estep_data.d_clique_x_ids);
    cudaFree(estep_data.d_clique_y_ids);
    cudaFree(estep_data.d_clique_parent_idx);
    cudaFree(estep_data.d_clique_child_indices);
    cudaFree(estep_data.d_clique_num_children);
    cudaFree(estep_data.d_post_order_clique);
    cudaFree(estep_data.d_clique_y_pattern_idx);
    cudaFree(estep_data.d_clique_base_potentials);
    cudaFree(estep_data.d_clique_beliefs);
    cudaFree(estep_data.d_clique_messages);
    cudaFree(estep_data.d_clique_log_scales);
    cudaFree(estep_data.d_expected_counts_vertex);
    cudaFree(estep_data.d_expected_counts_edge);
    cudaFree(estep_data.d_edge_to_clique_idx);

    estep_data.initialized = false;
    printf("CUDA E-step: Cleanup complete\n");
}

} /* extern "C" */
