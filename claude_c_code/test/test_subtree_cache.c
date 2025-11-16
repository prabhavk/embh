#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "embh_types.h"
#include "embh_manager.h"

/*
 * Test program to measure actual performance of subtree caching
 * Compares: no caching vs selective subtree caching
 */

/* Helper: recursively collect leaves in subtree rooted at vertex v */
static void collect_subtree_leaves_recursive(SEM* sem, SEM_vertex* v, int* leaf_indices, int* count) {
    if (!v) return;

    if (v->out_degree == 0 && v->observed && v->pattern_index >= 0) {
        leaf_indices[(*count)++] = v->pattern_index;
        return;
    }

    for (int i = 0; i < v->out_degree; i++) {
        collect_subtree_leaves_recursive(sem, v->children[i], leaf_indices, count);
    }
}

/* Compute subtree signature using FNV-1a hash */
static int compute_subtree_signature(int pat_idx, int num_taxa,
                                     const uint8_t* patterns,
                                     const int* subtree_leaves,
                                     int num_leaves) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < num_leaves; i++) {
        int leaf_idx = subtree_leaves[i];
        uint8_t dna = patterns[pat_idx * num_taxa + leaf_idx];
        hash ^= dna;
        hash *= 16777619u;
    }
    return (int)(hash & 0x7FFFFFFF);
}

/* Structure for subtree cache */
typedef struct {
    int* edge_subtree_leaves;     /* Flattened array of leaf indices */
    int* edge_subtree_offsets;    /* Offset into leaves for each edge */
    int* edge_num_subtree_leaves; /* Number of leaves in each edge's subtree */

    int* pattern_edge_to_cache;   /* [num_patterns x num_edges] -> cache slot */
    double** cached_messages;     /* [num_cache_slots][4] */
    int num_cache_slots;

    bool initialized;
} SubtreeCache;

/* Initialize subtree cache from cache spec file */
static SubtreeCache* init_subtree_cache(SEM* sem, const char* cache_spec_file) {
    FILE* fp = fopen(cache_spec_file, "r");
    if (!fp) return NULL;

    SubtreeCache* cache = (SubtreeCache*)calloc(1, sizeof(SubtreeCache));
    if (!cache) {
        fclose(fp);
        return NULL;
    }

    int num_patterns = sem->num_patterns;
    int num_edges = sem->num_post_order_edges;
    int num_taxa = sem->packed_patterns->num_taxa;

    /* Read cache spec to get which edges to cache */
    int* cached_edge_ids = NULL;
    int* edge_num_unique_sigs = NULL;
    int num_cached_edges = 0;
    int spec_num_patterns = 0;
    int spec_num_edges = 0;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        if (strncmp(line, "NUM_PATTERNS", 12) == 0) {
            sscanf(line, "NUM_PATTERNS %d", &spec_num_patterns);
        } else if (strncmp(line, "NUM_EDGES", 9) == 0) {
            sscanf(line, "NUM_EDGES %d", &spec_num_edges);
        } else if (strncmp(line, "NUM_CACHED_EDGES", 16) == 0) {
            sscanf(line, "NUM_CACHED_EDGES %d", &num_cached_edges);
            cached_edge_ids = (int*)malloc(num_cached_edges * sizeof(int));
            edge_num_unique_sigs = (int*)malloc(num_cached_edges * sizeof(int));
        } else if (strncmp(line, "CACHE_EDGE", 10) == 0 && cached_edge_ids) {
            static int ci = 0;
            if (ci < num_cached_edges) {
                int edge_id, num_unique, subtree_size;
                double reuse_rate;
                sscanf(line, "CACHE_EDGE %d %d %lf %d",
                       &edge_id, &num_unique, &reuse_rate, &subtree_size);
                cached_edge_ids[ci] = edge_id;
                edge_num_unique_sigs[ci] = num_unique;
                ci++;
            }
        }
    }
    fclose(fp);

    printf("Cache spec: %d patterns, %d edges, %d cached edges\n",
           spec_num_patterns, spec_num_edges, num_cached_edges);

    /* Build subtree leaves for each edge */
    cache->edge_num_subtree_leaves = (int*)malloc(num_edges * sizeof(int));
    cache->edge_subtree_offsets = (int*)malloc((num_edges + 1) * sizeof(int));

    int* temp_leaf_buffer = (int*)malloc(num_taxa * sizeof(int));
    int total_subtree_leaves = 0;

    cache->edge_subtree_offsets[0] = 0;
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        collect_subtree_leaves_recursive(sem, child, temp_leaf_buffer, &count);
        cache->edge_num_subtree_leaves[e] = count;
        total_subtree_leaves += count;
        cache->edge_subtree_offsets[e + 1] = total_subtree_leaves;
    }

    cache->edge_subtree_leaves = (int*)malloc(total_subtree_leaves * sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        SEM_vertex* child = sem->post_order_child[e];
        int count = 0;
        int offset = cache->edge_subtree_offsets[e];
        collect_subtree_leaves_recursive(sem, child,
                                         &cache->edge_subtree_leaves[offset], &count);
    }
    free(temp_leaf_buffer);

    /* Unpack pattern data */
    uint8_t* patterns = (uint8_t*)malloc(num_patterns * num_taxa * sizeof(uint8_t));
    for (int p = 0; p < num_patterns; p++) {
        for (int t = 0; t < num_taxa; t++) {
            patterns[p * num_taxa + t] = packed_storage_get_base(sem->packed_patterns, p, t);
        }
    }

    /* Build cache mapping for cached edges only */
    cache->pattern_edge_to_cache = (int*)malloc(num_patterns * num_edges * sizeof(int));
    for (int i = 0; i < num_patterns * num_edges; i++) {
        cache->pattern_edge_to_cache[i] = -1;  /* -1 = not cached */
    }

    int hash_table_size = num_patterns * 2;
    if (hash_table_size < 1024) hash_table_size = 1024;
    int* hash_keys = (int*)malloc(hash_table_size * sizeof(int));
    int* hash_slots = (int*)malloc(hash_table_size * sizeof(int));

    cache->num_cache_slots = 0;

    for (int ci = 0; ci < num_cached_edges; ci++) {
        int e = cached_edge_ids[ci];
        if (e < 0 || e >= num_edges) continue;

        /* Reset hash table */
        for (int i = 0; i < hash_table_size; i++) {
            hash_keys[i] = -1;
        }

        for (int p = 0; p < num_patterns; p++) {
            int offset = cache->edge_subtree_offsets[e];
            int num_leaves = cache->edge_num_subtree_leaves[e];
            int sig = compute_subtree_signature(p, num_taxa, patterns,
                                               &cache->edge_subtree_leaves[offset],
                                               num_leaves);

            unsigned int hash_idx = ((unsigned int)sig * 2654435761u) % hash_table_size;
            int found_slot = -1;
            int probes = 0;

            while (hash_keys[hash_idx] != -1 && probes < hash_table_size) {
                if (hash_keys[hash_idx] == sig) {
                    found_slot = hash_slots[hash_idx];
                    break;
                }
                hash_idx = (hash_idx + 1) % hash_table_size;
                probes++;
            }

            if (found_slot < 0) {
                found_slot = cache->num_cache_slots++;
                hash_keys[hash_idx] = sig;
                hash_slots[hash_idx] = found_slot;
            }

            cache->pattern_edge_to_cache[p * num_edges + e] = found_slot;
        }
    }

    free(hash_keys);
    free(hash_slots);
    free(patterns);
    free(cached_edge_ids);
    free(edge_num_unique_sigs);

    /* Allocate cache storage */
    cache->cached_messages = (double**)malloc(cache->num_cache_slots * sizeof(double*));
    for (int i = 0; i < cache->num_cache_slots; i++) {
        cache->cached_messages[i] = (double*)malloc(4 * sizeof(double));
        cache->cached_messages[i][0] = -1.0;  /* Mark as invalid */
    }

    printf("Allocated %d cache slots (%.2f MB)\n", cache->num_cache_slots,
           cache->num_cache_slots * 4 * sizeof(double) / 1048576.0);

    cache->initialized = true;
    return cache;
}

/* Free subtree cache */
static void free_subtree_cache(SubtreeCache* cache) {
    if (!cache) return;

    free(cache->edge_subtree_leaves);
    free(cache->edge_subtree_offsets);
    free(cache->edge_num_subtree_leaves);
    free(cache->pattern_edge_to_cache);

    if (cache->cached_messages) {
        for (int i = 0; i < cache->num_cache_slots; i++) {
            free(cache->cached_messages[i]);
        }
        free(cache->cached_messages);
    }

    free(cache);
}

/* Pruning with subtree caching */
static double compute_ll_with_subtree_cache(SEM* sem, SubtreeCache* cache) {
    if (!sem || !sem->packed_patterns || !cache) return -1e30;

    double total_ll = 0.0;

    /* Invalidate all cache entries at start */
    for (int i = 0; i < cache->num_cache_slots; i++) {
        cache->cached_messages[i][0] = -1.0;
    }

    int num_edges = sem->num_post_order_edges;
    int cache_hits = 0;
    int cache_misses = 0;

    /* Process each pattern */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        /* Create conditional likelihood storage */
        double** cond_like = (double**)calloc(sem->num_vertices, sizeof(double*));
        bool* cond_like_initialized = (bool*)calloc(sem->num_vertices, sizeof(bool));
        double* vertex_scaling = (double*)calloc(sem->num_vertices, sizeof(double));

        for (int i = 0; i < sem->num_vertices; i++) {
            cond_like[i] = (double*)calloc(4, sizeof(double));
        }

        /* Process edges in post-order */
        for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            SEM_vertex* p = sem->post_order_parent[edge_idx];
            SEM_vertex* c = sem->post_order_child[edge_idx];

            vertex_scaling[p->id] += vertex_scaling[c->id];

            /* Initialize leaf child */
            if (c->out_degree == 0 && c->pattern_index >= 0) {
                uint8_t base = packed_storage_get_base(sem->packed_patterns, pat_idx, c->pattern_index);
                if (base < 4) {
                    cond_like[c->id][base] = 1.0;
                } else {
                    cond_like[c->id][0] = 1.0;
                    cond_like[c->id][1] = 1.0;
                    cond_like[c->id][2] = 1.0;
                    cond_like[c->id][3] = 1.0;
                }
                cond_like_initialized[c->id] = true;
            }

            /* Initialize parent */
            if (!cond_like_initialized[p->id]) {
                cond_like[p->id][0] = 1.0;
                cond_like[p->id][1] = 1.0;
                cond_like[p->id][2] = 1.0;
                cond_like[p->id][3] = 1.0;
                cond_like_initialized[p->id] = true;
            }

            /* Check cache */
            int cache_slot = cache->pattern_edge_to_cache[pat_idx * num_edges + edge_idx];
            double message[4];

            if (cache_slot >= 0 && cache->cached_messages[cache_slot][0] >= 0) {
                /* Cache hit! */
                message[0] = cache->cached_messages[cache_slot][0];
                message[1] = cache->cached_messages[cache_slot][1];
                message[2] = cache->cached_messages[cache_slot][2];
                message[3] = cache->cached_messages[cache_slot][3];
                cache_hits++;
            } else {
                /* Cache miss - compute message */
                for (int dna_p = 0; dna_p < 4; dna_p++) {
                    double partial = 0.0;
                    for (int dna_c = 0; dna_c < 4; dna_c++) {
                        partial += c->transition_matrix[dna_p * 4 + dna_c] * cond_like[c->id][dna_c];
                    }
                    message[dna_p] = partial;
                }

                /* Store in cache if slot available */
                if (cache_slot >= 0) {
                    cache->cached_messages[cache_slot][0] = message[0];
                    cache->cached_messages[cache_slot][1] = message[1];
                    cache->cached_messages[cache_slot][2] = message[2];
                    cache->cached_messages[cache_slot][3] = message[3];
                }
                cache_misses++;
            }

            /* Apply message to parent */
            double largest = 0.0;
            for (int dna_p = 0; dna_p < 4; dna_p++) {
                cond_like[p->id][dna_p] *= message[dna_p];
                if (cond_like[p->id][dna_p] > largest) {
                    largest = cond_like[p->id][dna_p];
                }
            }

            /* Scale */
            if (largest > 0.0) {
                for (int dna_p = 0; dna_p < 4; dna_p++) {
                    cond_like[p->id][dna_p] /= largest;
                }
                vertex_scaling[p->id] += log(largest);
            }
        }

        /* Compute site likelihood at root */
        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += sem->root_probability[dna] * cond_like[sem->root->id][dna];
        }

        if (site_likelihood > 0.0) {
            double log_site_like = vertex_scaling[sem->root->id] + log(site_likelihood);
            total_ll += log_site_like * sem->pattern_weights[pat_idx];
        }

        /* Cleanup */
        for (int i = 0; i < sem->num_vertices; i++) {
            free(cond_like[i]);
        }
        free(cond_like);
        free(cond_like_initialized);
        free(vertex_scaling);
    }

    printf("Cache stats: %d hits, %d misses (%.1f%% hit rate)\n",
           cache_hits, cache_misses, 100.0 * cache_hits / (cache_hits + cache_misses));

    return total_ll;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <edgelist> <patterns> <taxon_order> <basecomp> <root_name> <cache_spec>\n", argv[0]);
        return 1;
    }

    const char* edge_list_file = argv[1];
    const char* pattern_file = argv[2];
    const char* taxon_order_file = argv[3];
    const char* base_comp_file = argv[4];
    const char* root_name = argv[5];
    const char* cache_spec_file = argv[6];

    printf("=== Subtree Cache Performance Test ===\n\n");

    /* Create SEM instance */
    SEM* P = embh_sem_create(200, 1000);
    if (!P) {
        fprintf(stderr, "Failed to create SEM\n");
        return 1;
    }

    /* Setup SEM */
    sem_set_edges_from_topology_file(P, edge_list_file);
    sem_read_patterns_from_file(P, pattern_file, taxon_order_file);

    SEM_vertex* root_vertex = sem_get_vertex_by_name(P, root_name);
    if (!root_vertex) {
        fprintf(stderr, "Root vertex '%s' not found\n", root_name);
        embh_sem_destroy(P);
        return 1;
    }
    sem_root_tree_at_vertex(P, root_vertex);
    sem_set_leaves_from_pattern_indices(P);
    sem_set_vertex_vector_except_root(P);
    sem_set_f81_model(P, base_comp_file);

    for (int i = 0; i < P->num_non_root_vertices; i++) {
        SEM_vertex* v = P->non_root_vertices[i];
        if (v) sem_set_f81_matrix(P, v);
    }

    sem_compute_edges_for_post_order_traversal(P);

    printf("Dataset: %d patterns, %d edges, %d taxa\n\n",
           P->num_patterns, P->num_post_order_edges, P->packed_patterns->num_taxa);

    /* Test 1: Standard pruning (no caching) */
    printf("=== Test 1: Standard Pruning (no subtree caching) ===\n");
    clock_t start = clock();
    sem_compute_log_likelihood_using_patterns(P);
    clock_t end = clock();
    double time_no_cache = (double)(end - start) / CLOCKS_PER_SEC;
    double ll_no_cache = P->log_likelihood;
    printf("Log-likelihood: %.11f\n", ll_no_cache);
    printf("Time: %.6f seconds\n\n", time_no_cache);

    /* Test 2: Pruning with subtree caching */
    printf("=== Test 2: Pruning with Selective Subtree Caching ===\n");
    printf("Loading cache spec: %s\n", cache_spec_file);

    SubtreeCache* cache = init_subtree_cache(P, cache_spec_file);
    if (!cache) {
        fprintf(stderr, "Failed to initialize cache\n");
        embh_sem_destroy(P);
        return 1;
    }

    start = clock();
    double ll_with_cache = compute_ll_with_subtree_cache(P, cache);
    end = clock();
    double time_with_cache = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Log-likelihood: %.11f\n", ll_with_cache);
    printf("Time: %.6f seconds\n\n", time_with_cache);

    /* Results */
    printf("=== Performance Comparison ===\n");
    printf("No caching:      %.6f seconds\n", time_no_cache);
    printf("Subtree caching: %.6f seconds\n", time_with_cache);
    printf("Speedup: %.2fx\n", time_no_cache / time_with_cache);
    printf("Time saved: %.6f seconds\n", time_no_cache - time_with_cache);
    printf("\nLog-likelihood difference: %.2e\n", fabs(ll_with_cache - ll_no_cache));

    if (fabs(ll_with_cache - ll_no_cache) < 1e-8) {
        printf("Results VERIFIED - caching produces correct result!\n");
    } else {
        printf("WARNING: Results differ!\n");
    }

    /* Cleanup */
    free_subtree_cache(cache);
    embh_sem_destroy(P);

    return 0;
}
