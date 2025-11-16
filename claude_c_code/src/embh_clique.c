#include "embh_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_CLIQUE_NEIGHBORS (MAX_CHILDREN + 1)  /* Parent + children */
#define MAX_CLIQUES 200

/* Create a new clique for edge (x, y) */
Clique* clique_create(SEM_vertex* x, SEM_vertex* y, PackedPatternStorage* patterns) {
    if (!x || !y) return NULL;

    Clique* c = (Clique*)calloc(1, sizeof(Clique));
    if (!c) return NULL;

    c->x = x;
    c->y = y;
    c->packed_patterns = patterns;

    /* Set name as "x_id-y_id" */
    snprintf(c->name, MAX_NAME_LEN, "%d-%d", x->id, y->id);

    c->site = -1;
    c->parent = c;  /* Point to self when no parent */
    c->children = (Clique**)calloc(MAX_CHILDREN, sizeof(Clique*));
    c->num_children = 0;
    c->in_degree = 0;
    c->out_degree = 0;
    c->times_visited = 0;

    if (!c->children) {
        free(c);
        return NULL;
    }

    /* Initialize matrices to zero */
    memset(c->initial_potential, 0, 16 * sizeof(double));
    memset(c->base_potential, 0, 16 * sizeof(double));
    memset(c->factor, 0, 16 * sizeof(double));
    memset(c->belief, 0, 16 * sizeof(double));

    c->scaling_factor = 0.0;
    c->log_scaling_factor_for_clique = 0.0;

    /* Allocate message storage */
    c->messages_from_neighbors = (double**)calloc(MAX_CLIQUE_NEIGHBORS, sizeof(double*));
    c->log_scaling_factors_for_messages = (double*)calloc(MAX_CLIQUE_NEIGHBORS, sizeof(double));
    c->neighbor_cliques = (Clique**)calloc(MAX_CLIQUE_NEIGHBORS, sizeof(Clique*));

    if (!c->messages_from_neighbors || !c->log_scaling_factors_for_messages || !c->neighbor_cliques) {
        free(c->children);
        free(c->messages_from_neighbors);
        free(c->log_scaling_factors_for_messages);
        free(c->neighbor_cliques);
        free(c);
        return NULL;
    }

    for (int i = 0; i < MAX_CLIQUE_NEIGHBORS; i++) {
        c->messages_from_neighbors[i] = (double*)calloc(4, sizeof(double));
        if (!c->messages_from_neighbors[i]) {
            for (int j = 0; j < i; j++) {
                free(c->messages_from_neighbors[j]);
            }
            free(c->messages_from_neighbors);
            free(c->log_scaling_factors_for_messages);
            free(c->neighbor_cliques);
            free(c->children);
            free(c);
            return NULL;
        }
    }
    c->num_neighbors = 0;

    c->root_variable = NULL;
    c->is_leaf_clique = false;
    c->post_order_edge_index = -1;  /* Will be set by sem_construct_clique_tree */

    /* Initialize memoization support */
    c->subtree_leaves = NULL;
    c->num_subtree_leaves = 0;
    c->complement_leaves = NULL;
    c->num_complement_leaves = 0;

    /* Message cache (will be allocated on demand) */
    c->upward_cache_signatures = NULL;
    c->upward_cache_messages = NULL;
    c->upward_cache_scales = NULL;
    c->upward_cache_size = 0;
    c->upward_cache_capacity = 0;

    c->downward_cache_signatures = NULL;
    c->downward_cache_messages = NULL;
    c->downward_cache_scales = NULL;
    c->downward_cache_size = 0;
    c->downward_cache_capacity = 0;

    /* Cache statistics */
    c->upward_cache_hits = 0;
    c->upward_cache_misses = 0;
    c->downward_cache_hits = 0;
    c->downward_cache_misses = 0;

    return c;
}

/* Free clique memory */
void clique_destroy(Clique* c) {
    if (!c) return;

    if (c->messages_from_neighbors) {
        for (int i = 0; i < MAX_CLIQUE_NEIGHBORS; i++) {
            free(c->messages_from_neighbors[i]);
        }
        free(c->messages_from_neighbors);
    }

    free(c->log_scaling_factors_for_messages);
    free(c->neighbor_cliques);
    free(c->children);

    /* Free memoization data */
    free(c->subtree_leaves);
    free(c->complement_leaves);

    /* Free upward cache */
    if (c->upward_cache_messages) {
        for (int i = 0; i < c->upward_cache_capacity; i++) {
            free(c->upward_cache_messages[i]);
        }
        free(c->upward_cache_messages);
    }
    free(c->upward_cache_signatures);
    free(c->upward_cache_scales);

    /* Free downward cache */
    if (c->downward_cache_messages) {
        for (int i = 0; i < c->downward_cache_capacity; i++) {
            free(c->downward_cache_messages[i]);
        }
        free(c->downward_cache_messages);
    }
    free(c->downward_cache_signatures);
    free(c->downward_cache_scales);

    free(c);
}

/* Add child to clique */
void clique_add_child(Clique* c, Clique* child) {
    if (!c || !child || c->num_children >= MAX_CHILDREN) return;

    c->children[c->num_children++] = child;
    c->out_degree++;
}

/* Set parent of clique */
void clique_set_parent(Clique* c, Clique* parent) {
    if (!c) return;
    c->parent = parent ? parent : c;
    if (parent && parent != c) {
        c->in_degree = 1;
    }
}

/* Set initial potential and belief for a given site */
void clique_set_initial_potential_and_belief(Clique* c, int site) {
    if (!c || !c->x || !c->y) return;

    c->site = site;

    /* Copy transition matrix from Y (child vertex) as base potential */
    /* P(Y|X) = y->transition_matrix */
    memcpy(c->initial_potential, c->y->transition_matrix, 16 * sizeof(double));

    /* If Y is observed, apply evidence */
    if (c->y->observed && c->y->pattern_index >= 0 && c->packed_patterns) {
        int dna_y = packed_storage_get_base(c->packed_patterns, site, c->y->pattern_index);

        if (dna_y >= 0 && dna_y < 4) {
            /* Set columns not matching observed value to zero */
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (j != dna_y) {
                        c->initial_potential[i * 4 + j] = 0.0;
                    }
                }
            }
        }
        /* If dna_y == 4 (gap), keep full transition matrix */
    }

    /* NOTE: Do NOT multiply by root probability here.
     * The root probability is applied at the final likelihood computation
     * in sem_compute_log_likelihood_using_propagation().
     * This ensures consistency regardless of whether Y is observed or not.
     */

    /* Initialize belief to initial potential */
    memcpy(c->belief, c->initial_potential, 16 * sizeof(double));

    /* Reset scaling factors */
    c->log_scaling_factor_for_clique = 0.0;

    /* Clear messages and neighbor mapping */
    c->num_neighbors = 0;
    for (int i = 0; i < MAX_CLIQUE_NEIGHBORS; i++) {
        memset(c->messages_from_neighbors[i], 0, 4 * sizeof(double));
        c->log_scaling_factors_for_messages[i] = 0.0;
        c->neighbor_cliques[i] = NULL;
    }
}

/* Set base potential from model parameters (done once before pattern loop)
 * This sets up the transition matrix and root prior, without site-specific evidence.
 */
void clique_set_base_potential(Clique* c) {
    if (!c || !c->x || !c->y) return;

    /* Copy transition matrix from Y (child vertex) as base potential */
    memcpy(c->base_potential, c->y->transition_matrix, 16 * sizeof(double));

    /* NOTE: Root probability is NOT multiplied here.
     * The root probability is applied at the final likelihood computation
     * in sem_compute_log_likelihood_using_propagation().
     * This ensures consistency regardless of tree structure.
     */
}

/* Apply site-specific evidence to the initial potential (inside pattern loop)
 * This applies the observed values for a given site.
 */
void clique_apply_evidence_for_site(Clique* c, int site) {
    if (!c || !c->y) return;

    c->site = site;

    /* Start with base potential (model parameters) */
    memcpy(c->initial_potential, c->base_potential, 16 * sizeof(double));

    /* If Y is observed, restrict to observed value at this site */
    if (c->y->observed && c->y->pattern_index >= 0 && c->packed_patterns) {
        int dna_y = packed_storage_get_base(c->packed_patterns, site, c->y->pattern_index);

        if (dna_y >= 0 && dna_y < 4) {
            /* Zero out columns that don't match observed value */
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (j != dna_y) {
                        c->initial_potential[i * 4 + j] = 0.0;
                    }
                }
            }
        }
        /* If dna_y == 4 (gap), keep full transition matrix (no restriction) */
    }

    /* Initialize belief to initial potential */
    memcpy(c->belief, c->initial_potential, 16 * sizeof(double));

    /* Reset scaling factors */
    c->log_scaling_factor_for_clique = 0.0;

    /* Clear messages and neighbor mapping */
    c->num_neighbors = 0;
    for (int i = 0; i < MAX_CLIQUE_NEIGHBORS; i++) {
        memset(c->messages_from_neighbors[i], 0, 4 * sizeof(double));
        c->log_scaling_factors_for_messages[i] = 0.0;
        c->neighbor_cliques[i] = NULL;
    }
}

/* clique_compute_belief is implemented in embh_propagation.c */

/* Marginalize belief over variable v to get message */
void clique_marginalize_over_variable(const Clique* c, const SEM_vertex* v, double* result) {
    if (!c || !v || !result) return;

    /* Initialize result to zero */
    memset(result, 0, 4 * sizeof(double));

    if (c->x == v) {
        /* Marginalize over X (sum over rows) to get P(Y) */
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                result[y] += c->belief[x * 4 + y];
            }
        }
    } else if (c->y == v) {
        /* Marginalize over Y (sum over columns) to get P(X) */
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                result[x] += c->belief[x * 4 + y];
            }
        }
    }
}

/* Create clique tree */
CliqueTree* clique_tree_create(void) {
    CliqueTree* ct = (CliqueTree*)calloc(1, sizeof(CliqueTree));
    if (!ct) return NULL;

    ct->cliques = (Clique**)calloc(MAX_CLIQUES, sizeof(Clique*));
    ct->leaves = (Clique**)calloc(MAX_CLIQUES, sizeof(Clique*));
    ct->post_order_traversal = (Clique**)calloc(MAX_CLIQUES, sizeof(Clique*));
    ct->pre_order_traversal = (Clique**)calloc(MAX_CLIQUES, sizeof(Clique*));

    if (!ct->cliques || !ct->leaves || !ct->post_order_traversal || !ct->pre_order_traversal) {
        free(ct->cliques);
        free(ct->leaves);
        free(ct->post_order_traversal);
        free(ct->pre_order_traversal);
        free(ct);
        return NULL;
    }

    ct->num_cliques = 0;
    ct->num_leaves = 0;
    ct->traversal_size = 0;
    ct->root = NULL;
    ct->root_set = false;
    ct->site = -1;

    /* Initialize memoization fields */
    ct->all_observed_leaves = NULL;
    ct->num_all_observed_leaves = 0;
    ct->cache_hits = 0;
    ct->cache_misses = 0;
    ct->downward_cache_hits = 0;
    ct->downward_cache_misses = 0;

    return ct;
}

/* Free clique tree (but not the cliques themselves) */
void clique_tree_destroy(CliqueTree* ct) {
    if (!ct) return;
    free(ct->cliques);
    free(ct->leaves);
    free(ct->post_order_traversal);
    free(ct->pre_order_traversal);
    free(ct->all_observed_leaves);
    free(ct);
}

/* Add clique to tree */
void clique_tree_add_clique(CliqueTree* ct, Clique* c) {
    if (!ct || !c || ct->num_cliques >= MAX_CLIQUES) return;

    c->id = ct->num_cliques;
    ct->cliques[ct->num_cliques++] = c;

    /* Check if leaf clique (Y is observed) */
    if (c->y && c->y->observed) {
        c->is_leaf_clique = true;
        if (ct->num_leaves < MAX_CLIQUES) {
            ct->leaves[ct->num_leaves++] = c;
        }
    }
}

/* Set root of clique tree */
void clique_tree_set_root(CliqueTree* ct, Clique* root) {
    if (!ct) return;
    ct->root = root;
    ct->root_set = (root != NULL);
}

/* Compute post-order and pre-order traversals */
void clique_tree_compute_traversal_orders(CliqueTree* ct) {
    if (!ct || !ct->root) return;

    /* Use topological sort for post-order (leaves to root) */
    int* in_degree = (int*)calloc(ct->num_cliques, sizeof(int));
    if (!in_degree) return;

    /* Count in-degrees (number of children) */
    for (int i = 0; i < ct->num_cliques; i++) {
        in_degree[i] = ct->cliques[i]->num_children;
    }

    /* Queue for BFS */
    Clique** queue = (Clique**)malloc(ct->num_cliques * sizeof(Clique*));
    if (!queue) {
        free(in_degree);
        return;
    }

    int front = 0, back = 0;
    int post_idx = 0;

    /* Start with leaf cliques (no children) */
    for (int i = 0; i < ct->num_cliques; i++) {
        if (in_degree[i] == 0) {
            queue[back++] = ct->cliques[i];
        }
    }

    /* Process cliques in post-order */
    while (front < back) {
        Clique* curr = queue[front++];
        ct->post_order_traversal[post_idx++] = curr;

        /* Decrease parent's in-degree */
        if (curr->parent && curr->parent != curr) {
            int parent_idx = curr->parent->id;
            in_degree[parent_idx]--;
            if (in_degree[parent_idx] == 0) {
                queue[back++] = curr->parent;
            }
        }
    }

    ct->traversal_size = post_idx;

    /* Pre-order is reverse of post-order */
    for (int i = 0; i < ct->traversal_size; i++) {
        ct->pre_order_traversal[i] = ct->post_order_traversal[ct->traversal_size - 1 - i];
    }

    free(in_degree);
    free(queue);
}

/* Set current site for tree */
void clique_tree_set_site(CliqueTree* ct, int site) {
    if (!ct) return;
    ct->site = site;
}

/* Initialize potentials and beliefs for all cliques */
void clique_tree_initialize_potentials_and_beliefs(CliqueTree* ct) {
    if (!ct) return;

    for (int i = 0; i < ct->num_cliques; i++) {
        clique_set_initial_potential_and_belief(ct->cliques[i], ct->site);
    }
}

/* Set base potentials for all cliques (done once before pattern loop) */
void clique_tree_set_base_potentials(CliqueTree* ct) {
    if (!ct) return;

    for (int i = 0; i < ct->num_cliques; i++) {
        clique_set_base_potential(ct->cliques[i]);
    }
}

/* Apply evidence and reset for a specific site (inside pattern loop) */
void clique_tree_apply_evidence_and_reset(CliqueTree* ct, int site) {
    if (!ct) return;

    ct->site = site;
    for (int i = 0; i < ct->num_cliques; i++) {
        clique_apply_evidence_for_site(ct->cliques[i], site);
    }
}

/* clique_tree_calibrate is implemented in embh_propagation.c */

/* ========== Memoization Support Functions ========== */

/* Compare vertices by ID for sorting */
static int compare_vertex_id(const void* a, const void* b) {
    SEM_vertex* va = *(SEM_vertex**)a;
    SEM_vertex* vb = *(SEM_vertex**)b;
    return va->id - vb->id;
}

/* Compute which observed variables are in this clique's subtree
 * This is computed recursively in post-order (leaves to root)
 */
void clique_compute_subtree_leaves(Clique* c) {
    if (!c) return;

    /* Count total leaves needed */
    int total_leaves = 0;

    /* If Y is observed, this clique has one leaf */
    if (c->y->observed) {
        total_leaves++;
    }

    /* Add leaves from all children */
    for (int i = 0; i < c->num_children; i++) {
        total_leaves += c->children[i]->num_subtree_leaves;
    }

    /* Allocate or reallocate subtree_leaves array */
    if (c->subtree_leaves) {
        free(c->subtree_leaves);
    }
    c->subtree_leaves = (SEM_vertex**)malloc(total_leaves * sizeof(SEM_vertex*));
    if (!c->subtree_leaves) {
        c->num_subtree_leaves = 0;
        return;
    }

    int idx = 0;

    /* Add this clique's Y if observed */
    if (c->y->observed) {
        c->subtree_leaves[idx++] = c->y;
    }

    /* Add leaves from children */
    for (int i = 0; i < c->num_children; i++) {
        Clique* child = c->children[i];
        for (int j = 0; j < child->num_subtree_leaves; j++) {
            c->subtree_leaves[idx++] = child->subtree_leaves[j];
        }
    }

    c->num_subtree_leaves = total_leaves;

    /* Sort by vertex ID for consistent signature */
    if (c->num_subtree_leaves > 1) {
        qsort(c->subtree_leaves, c->num_subtree_leaves, sizeof(SEM_vertex*), compare_vertex_id);
    }
}

/* Compute signature hash for subtree pattern
 * Returns a hash of the pattern values at all observed leaves in subtree
 */
int clique_get_subtree_signature(const Clique* c, int site) {
    if (!c || !c->packed_patterns || c->num_subtree_leaves == 0) return 0;

    /* Simple hash: treat pattern values as base-5 digits (0-4 for A,C,G,T,gap) */
    int hash = 0;
    for (int i = 0; i < c->num_subtree_leaves; i++) {
        SEM_vertex* leaf = c->subtree_leaves[i];
        int dna = packed_storage_get_base(c->packed_patterns, site, leaf->pattern_index);
        hash = hash * 5 + dna;
    }
    return hash;
}

/* Compute signature hash for complement pattern
 * Returns a hash of the pattern values at all observed leaves NOT in subtree
 */
int clique_get_complement_signature(const Clique* c, int site) {
    if (!c || !c->packed_patterns || c->num_complement_leaves == 0) return 0;

    int hash = 0;
    for (int i = 0; i < c->num_complement_leaves; i++) {
        SEM_vertex* leaf = c->complement_leaves[i];
        int dna = packed_storage_get_base(c->packed_patterns, site, leaf->pattern_index);
        hash = hash * 5 + dna;
    }
    return hash;
}

/* Clear all cached messages for a clique (but keep capacity) */
void clique_clear_caches(Clique* c) {
    if (!c) return;

    c->upward_cache_size = 0;
    c->downward_cache_size = 0;

    /* Reset statistics */
    c->upward_cache_hits = 0;
    c->upward_cache_misses = 0;
    c->downward_cache_hits = 0;
    c->downward_cache_misses = 0;
}

/* Compute subtree leaves for all cliques in post-order */
void clique_tree_compute_all_subtree_leaves(CliqueTree* ct) {
    if (!ct) return;

    /* Process in post-order (children before parents) */
    for (int i = 0; i < ct->traversal_size; i++) {
        clique_compute_subtree_leaves(ct->post_order_traversal[i]);
    }

    /* Store all observed leaves (from root's subtree) */
    if (ct->root && ct->root->num_subtree_leaves > 0) {
        free(ct->all_observed_leaves);
        ct->all_observed_leaves = (SEM_vertex**)malloc(
            ct->root->num_subtree_leaves * sizeof(SEM_vertex*));
        if (ct->all_observed_leaves) {
            memcpy(ct->all_observed_leaves, ct->root->subtree_leaves,
                   ct->root->num_subtree_leaves * sizeof(SEM_vertex*));
            ct->num_all_observed_leaves = ct->root->num_subtree_leaves;
        }
    }
}

/* Compute complement leaves for all cliques */
void clique_tree_compute_all_complement_leaves(CliqueTree* ct) {
    if (!ct || !ct->all_observed_leaves) return;

    /* For each clique, complement = all_leaves - subtree_leaves */
    for (int i = 0; i < ct->num_cliques; i++) {
        Clique* c = ct->cliques[i];

        /* Compute complement size */
        int complement_size = ct->num_all_observed_leaves - c->num_subtree_leaves;

        /* Allocate complement array */
        free(c->complement_leaves);
        if (complement_size > 0) {
            c->complement_leaves = (SEM_vertex**)malloc(complement_size * sizeof(SEM_vertex*));
            if (!c->complement_leaves) {
                c->num_complement_leaves = 0;
                continue;
            }

            /* Fill complement: leaves in all_observed but not in subtree */
            int comp_idx = 0;
            int sub_idx = 0;
            for (int all_idx = 0; all_idx < ct->num_all_observed_leaves; all_idx++) {
                SEM_vertex* leaf = ct->all_observed_leaves[all_idx];

                /* Check if this leaf is in subtree (both arrays are sorted by ID) */
                bool in_subtree = false;
                while (sub_idx < c->num_subtree_leaves &&
                       c->subtree_leaves[sub_idx]->id < leaf->id) {
                    sub_idx++;
                }
                if (sub_idx < c->num_subtree_leaves &&
                    c->subtree_leaves[sub_idx]->id == leaf->id) {
                    in_subtree = true;
                    sub_idx++;
                }

                if (!in_subtree) {
                    c->complement_leaves[comp_idx++] = leaf;
                }
            }
            c->num_complement_leaves = comp_idx;
        } else {
            c->complement_leaves = NULL;
            c->num_complement_leaves = 0;
        }
    }
}

/* Clear all caches in the clique tree */
void clique_tree_clear_all_caches(CliqueTree* ct) {
    if (!ct) return;

    for (int i = 0; i < ct->num_cliques; i++) {
        clique_clear_caches(ct->cliques[i]);
    }

    ct->cache_hits = 0;
    ct->cache_misses = 0;
    ct->downward_cache_hits = 0;
    ct->downward_cache_misses = 0;
}

/* Reset cache statistics */
void clique_tree_reset_cache_statistics(CliqueTree* ct) {
    if (!ct) return;

    for (int i = 0; i < ct->num_cliques; i++) {
        Clique* c = ct->cliques[i];
        c->upward_cache_hits = 0;
        c->upward_cache_misses = 0;
        c->downward_cache_hits = 0;
        c->downward_cache_misses = 0;
    }

    ct->cache_hits = 0;
    ct->cache_misses = 0;
    ct->downward_cache_hits = 0;
    ct->downward_cache_misses = 0;
}
