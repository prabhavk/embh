#include "embh_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*
 * Pruning Algorithm (Felsenstein's algorithm) for phylogenetic likelihood
 *
 * This implements the classic tree-pruning algorithm for computing the
 * likelihood of observed DNA patterns given an evolutionary tree.
 */

/* Compute post-order edge traversal (leaves to root) */
void sem_compute_edges_for_post_order_traversal(SEM* sem) {
    if (!sem || !sem->root) return;

    /* Reset times_visited for all vertices */
    for (int i = 0; i < sem->num_vertices; i++) {
        sem->vertices[i]->times_visited = 0;
    }

    /* Use leaves as starting point */
    if (sem->num_leaves == 0) {
        /* Set leaves if not already done */
        sem_set_leaves_from_pattern_indices(sem);
    }

    /* Create work queue (stack-like, same as C++) */
    SEM_vertex** to_visit = (SEM_vertex**)malloc(sem->num_vertices * sizeof(SEM_vertex*));
    if (!to_visit) return;

    /* Initialize with leaves */
    int num_to_visit = 0;
    for (int i = 0; i < sem->num_leaves; i++) {
        to_visit[num_to_visit++] = sem->leaves[i];
    }

    sem->num_post_order_edges = 0;

    /* Process vertices from leaves to root */
    while (num_to_visit > 0) {
        SEM_vertex* c = to_visit[num_to_visit - 1];
        to_visit[num_to_visit - 1] = NULL;
        num_to_visit--;

        /* Add edge (parent, child) if not root */
        if (c->parent && c->parent != c) {
            SEM_vertex* p = c->parent;

            /* Store the edge */
            if (sem->num_post_order_edges < sem->max_post_order_edges) {
                sem->post_order_parent[sem->num_post_order_edges] = p;
                sem->post_order_child[sem->num_post_order_edges] = c;
                sem->num_post_order_edges++;
            }

            /* Increment parent's visit count */
            p->times_visited++;

            /* If parent has been visited by all children, add to queue */
            if (p->times_visited == p->out_degree) {
                to_visit[num_to_visit++] = p;
            }
        }
    }

    free(to_visit);
}

/* Reset log scaling factors for all vertices */
void sem_reset_log_scaling_factors(SEM* sem) {
    if (!sem) return;

    for (int i = 0; i < sem->num_vertices; i++) {
        sem->vertices[i]->log_scaling_factors = 0.0;
    }
}

/* Main pruning algorithm: compute log-likelihood using patterns */
void sem_compute_log_likelihood_using_patterns(SEM* sem) {
    if (!sem || !sem->packed_patterns) {
        fprintf(stderr, "Error: No patterns loaded. Call sem_read_patterns_from_file() first.\n");
        return;
    }

    if (sem->num_post_order_edges == 0) {
        sem_compute_edges_for_post_order_traversal(sem);
    }

    sem->log_likelihood = 0.0;

    /* Create conditional likelihood storage - maps vertex ID to array[4] */
    double** cond_like = (double**)calloc(sem->num_vertices, sizeof(double*));
    bool* cond_like_initialized = (bool*)calloc(sem->num_vertices, sizeof(bool));
    if (!cond_like || !cond_like_initialized) {
        free(cond_like);
        free(cond_like_initialized);
        return;
    }

    for (int i = 0; i < sem->num_vertices; i++) {
        cond_like[i] = (double*)malloc(4 * sizeof(double));
        if (!cond_like[i]) {
            for (int j = 0; j < i; j++) {
                free(cond_like[j]);
            }
            free(cond_like);
            free(cond_like_initialized);
            return;
        }
    }

    /* Create vertex_to_base mapping for current pattern */
    uint8_t* vertex_to_base = (uint8_t*)malloc(sem->num_vertices * sizeof(uint8_t));
    if (!vertex_to_base) {
        for (int i = 0; i < sem->num_vertices; i++) {
            free(cond_like[i]);
        }
        free(cond_like);
        free(cond_like_initialized);
        return;
    }

    /* Temporary pattern buffer */
    int num_taxa = packed_storage_get_num_taxa(sem->packed_patterns);
    uint8_t* pattern = (uint8_t*)malloc(num_taxa * sizeof(uint8_t));
    if (!pattern) {
        free(vertex_to_base);
        for (int i = 0; i < sem->num_vertices; i++) {
            free(cond_like[i]);
        }
        free(cond_like);
        free(cond_like_initialized);
        return;
    }

    /* Build pattern_index_to_vertex_id mapping */
    int* pattern_idx_to_vertex_id = (int*)malloc(num_taxa * sizeof(int));
    if (!pattern_idx_to_vertex_id) {
        free(pattern);
        free(vertex_to_base);
        for (int i = 0; i < sem->num_vertices; i++) {
            free(cond_like[i]);
        }
        free(cond_like);
        free(cond_like_initialized);
        return;
    }

    /* Initialize to -1 (not found) */
    for (int i = 0; i < num_taxa; i++) {
        pattern_idx_to_vertex_id[i] = -1;
    }

    /* Fill mapping from pattern index to vertex ID */
    for (int v = 0; v < sem->num_vertices; v++) {
        SEM_vertex* vertex = sem->vertices[v];
        if (vertex->observed && vertex->pattern_index >= 0 && vertex->pattern_index < num_taxa) {
            pattern_idx_to_vertex_id[vertex->pattern_index] = vertex->id;
        }
    }

    /* Iterate over each pattern */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        /* Clear conditional likelihood map */
        memset(cond_like_initialized, 0, sem->num_vertices * sizeof(bool));

        /* Reset log scaling factors */
        sem_reset_log_scaling_factors(sem);

        /* Get current pattern */
        packed_storage_get_pattern(sem->packed_patterns, pat_idx, pattern);

        /* Build vertex_to_base for this pattern */
        /* Initialize all to gap (4) */
        memset(vertex_to_base, 4, sem->num_vertices * sizeof(uint8_t));

        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            int vertex_id = pattern_idx_to_vertex_id[taxon_idx];
            if (vertex_id >= 0 && vertex_id < sem->num_vertices) {
                vertex_to_base[vertex_id] = pattern[taxon_idx];
            }
        }

        /* Traverse tree using post-order edges */
        for (int edge_idx = 0; edge_idx < sem->num_post_order_edges; edge_idx++) {
            SEM_vertex* p = sem->post_order_parent[edge_idx];
            SEM_vertex* c = sem->post_order_child[edge_idx];
            double* P = c->transition_matrix;  /* P(Y|X) for child */

            /* Accumulate scaling factors */
            p->log_scaling_factors += c->log_scaling_factors;

            /* Initialize leaf child (outDegree == 0) */
            if (c->out_degree == 0) {
                uint8_t base = vertex_to_base[c->id];

                if (base < 4) {  /* Valid DNA base (0-3) */
                    cond_like[c->id][0] = 0.0;
                    cond_like[c->id][1] = 0.0;
                    cond_like[c->id][2] = 0.0;
                    cond_like[c->id][3] = 0.0;
                    cond_like[c->id][base] = 1.0;
                } else {  /* Gap (4) or unknown - treat as missing data */
                    cond_like[c->id][0] = 1.0;
                    cond_like[c->id][1] = 1.0;
                    cond_like[c->id][2] = 1.0;
                    cond_like[c->id][3] = 1.0;
                }
                cond_like_initialized[c->id] = true;
            }

            /* Initialize parent p if not yet initialized */
            if (!cond_like_initialized[p->id]) {
                if (!p->observed) {
                    /* Latent (hidden) vertex - treat as missing data */
                    cond_like[p->id][0] = 1.0;
                    cond_like[p->id][1] = 1.0;
                    cond_like[p->id][2] = 1.0;
                    cond_like[p->id][3] = 1.0;
                } else {
                    /* Observed vertex as parent */
                    uint8_t base = vertex_to_base[p->id];

                    if (base < 4) {  /* Valid DNA base */
                        cond_like[p->id][0] = 0.0;
                        cond_like[p->id][1] = 0.0;
                        cond_like[p->id][2] = 0.0;
                        cond_like[p->id][3] = 0.0;
                        cond_like[p->id][base] = 1.0;
                    } else {  /* Gap - missing data */
                        cond_like[p->id][0] = 1.0;
                        cond_like[p->id][1] = 1.0;
                        cond_like[p->id][2] = 1.0;
                        cond_like[p->id][3] = 1.0;
                    }
                }
                cond_like_initialized[p->id] = true;
            }

            /* DP update: parent *= sum_child P(parent|child) * child_likelihood */
            double largest = 0.0;
            double* parent_cl = cond_like[p->id];
            double* child_cl = cond_like[c->id];

            for (int dna_p = 0; dna_p < 4; dna_p++) {
                double partial = 0.0;
                /* Sum over child states: P(dna_p -> dna_c) * childCL[dna_c] */
                for (int dna_c = 0; dna_c < 4; dna_c++) {
                    partial += P[dna_p * 4 + dna_c] * child_cl[dna_c];
                }
                parent_cl[dna_p] *= partial;

                if (parent_cl[dna_p] > largest) {
                    largest = parent_cl[dna_p];
                }
            }

            /* Scale to prevent underflow */
            if (largest > 0.0) {
                for (int dna_p = 0; dna_p < 4; dna_p++) {
                    parent_cl[dna_p] /= largest;
                }
                p->log_scaling_factors += log(largest);
            } else {
                /* Error: zero likelihood - debug output */
                fprintf(stderr, "Error: conditional likelihood is zero at pattern %d, edge %d->%d\n",
                        pat_idx, p->id, c->id);
                fprintf(stderr, "  Parent %s (id=%d, observed=%d, out_degree=%d)\n",
                        p->name, p->id, p->observed, p->out_degree);
                fprintf(stderr, "  Child %s (id=%d, observed=%d, out_degree=%d)\n",
                        c->name, c->id, c->observed, c->out_degree);
                fprintf(stderr, "  Parent CL before update: [%.6f, %.6f, %.6f, %.6f]\n",
                        parent_cl[0], parent_cl[1], parent_cl[2], parent_cl[3]);
                fprintf(stderr, "  Child CL: [%.6f, %.6f, %.6f, %.6f]\n",
                        child_cl[0], child_cl[1], child_cl[2], child_cl[3]);
                fprintf(stderr, "  Child base: %d, Parent base: %d\n",
                        vertex_to_base[c->id], vertex_to_base[p->id]);
                /* Clean up and return */
                free(pattern_idx_to_vertex_id);
                free(pattern);
                free(vertex_to_base);
                for (int i = 0; i < sem->num_vertices; i++) {
                    free(cond_like[i]);
                }
                free(cond_like);
                free(cond_like_initialized);
                return;
            }
        }

        /* Compute site likelihood at root */
        double site_likelihood = 0.0;
        double* root_cl = cond_like[sem->root->id];

        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += sem->root_probability[dna] * root_cl[dna];
        }

        if (site_likelihood <= 0.0) {
            fprintf(stderr, "Error: site likelihood <= 0 at pattern %d\n", pat_idx);
            free(pattern_idx_to_vertex_id);
            free(pattern);
            free(vertex_to_base);
            for (int i = 0; i < sem->num_vertices; i++) {
                free(cond_like[i]);
            }
            free(cond_like);
            free(cond_like_initialized);
            return;
        }

        /* Add weighted site log-likelihood */
        int weight = sem->pattern_weights[pat_idx];
        sem->log_likelihood += (sem->root->log_scaling_factors + log(site_likelihood)) * weight;
    }

    /* Clean up */
    free(pattern_idx_to_vertex_id);
    free(pattern);
    free(vertex_to_base);
    for (int i = 0; i < sem->num_vertices; i++) {
        free(cond_like[i]);
    }
    free(cond_like);
    free(cond_like_initialized);
}
