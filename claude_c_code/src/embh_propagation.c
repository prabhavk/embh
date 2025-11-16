#include "embh_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*
 * Belief Propagation Algorithm (Junction Tree Algorithm) for phylogenetic likelihood
 *
 * This implements the sum-product message passing algorithm on a clique tree
 * to compute the likelihood of observed DNA patterns given an evolutionary tree.
 *
 * Key concepts:
 * - Each edge (X->Y) in the SEM tree becomes a clique in the junction tree
 * - Messages are passed between adjacent cliques (sharing a variable)
 * - Calibration: send messages leaves->root, then root->leaves
 * - Final beliefs represent exact marginal probabilities
 */

/* Helper: Get neighbor index for a clique (returns -1 if not found, adds if not present) */
int clique_get_neighbor_index(Clique* c, Clique* neighbor) {
    if (!c || !neighbor) return -1;

    /* Search existing neighbors */
    for (int i = 0; i < c->num_neighbors; i++) {
        if (c->neighbor_cliques[i] == neighbor) {
            return i;
        }
    }

    /* Add new neighbor if space available */
    if (c->num_neighbors < MAX_CHILDREN + 1) {  /* MAX_CLIQUE_NEIGHBORS */
        int idx = c->num_neighbors;
        c->neighbor_cliques[idx] = neighbor;
        c->num_neighbors++;
        return idx;
    }

    return -1;  /* No space */
}

/* Store a message from one clique to another */
void clique_store_message(Clique* to, Clique* from, const double* message, double log_scaling) {
    if (!to || !from || !message) return;

    int idx = clique_get_neighbor_index(to, from);
    if (idx >= 0) {
        memcpy(to->messages_from_neighbors[idx], message, 4 * sizeof(double));
        to->log_scaling_factors_for_messages[idx] = log_scaling;
    }
}

/* Send message from one clique to another
 * This implements the sum-product message passing:
 * 1. Take initial potential of sender
 * 2. Multiply by messages from all neighbors except receiver
 * 3. Marginalize over variable not shared with receiver
 * 4. Rescale and send
 */
void clique_tree_send_message(CliqueTree* ct, Clique* from, Clique* to) {
    (void)ct;  /* Unused parameter */

    if (!from || !to) return;

    double log_scaling_factor = 0.0;
    double largest_element;
    double factor[16];
    double message_to_neighbor[4];

    /* Start with initial potential of sender */
    memcpy(factor, from->initial_potential, 16 * sizeof(double));

    /* Multiply by messages from all neighbors except 'to' */
    /* Build list of neighbors (parent + children, excluding 'to') */
    Clique* neighbors[MAX_CHILDREN + 1];
    int num_neighbors = 0;

    if (from->parent != from && from->parent != to) {
        neighbors[num_neighbors++] = from->parent;
    }

    for (int i = 0; i < from->num_children; i++) {
        if (from->children[i] != to) {
            neighbors[num_neighbors++] = from->children[i];
        }
    }

    /* A. PRODUCT: Multiply messages from neighbors */
    for (int n = 0; n < num_neighbors; n++) {
        Clique* neighbor = neighbors[n];
        int neighbor_idx = clique_get_neighbor_index(from, neighbor);
        if (neighbor_idx < 0) continue;

        double* message_from = from->messages_from_neighbors[neighbor_idx];
        log_scaling_factor += from->log_scaling_factors_for_messages[neighbor_idx];

        /* Determine if shared variable is X or Y of 'from' */
        if (from->y == neighbor->x || from->y == neighbor->y) {
            /* Shared variable is Y: row-wise multiplication */
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    factor[dna_x * 4 + dna_y] *= message_from[dna_y];
                }
            }
        } else if (from->x == neighbor->x || from->x == neighbor->y) {
            /* Shared variable is X: column-wise multiplication */
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    factor[dna_x * 4 + dna_y] *= message_from[dna_x];
                }
            }
        } else {
            fprintf(stderr, "Error: No shared variable in clique message passing\n");
            return;
        }

        /* Rescale factor to prevent underflow */
        largest_element = 0.0;
        for (int i = 0; i < 16; i++) {
            if (factor[i] > largest_element) {
                largest_element = factor[i];
            }
        }

        if (largest_element > 0.0) {
            for (int i = 0; i < 16; i++) {
                factor[i] /= largest_element;
            }
            log_scaling_factor += log(largest_element);
        } else {
            fprintf(stderr, "Error: All-zero factor in message passing\n");
            return;
        }
    }

    /* B. SUM: Marginalize over variable not shared with 'to' */
    largest_element = 0.0;

    if (from->y == to->x || from->y == to->y) {
        /* Shared variable is Y of 'from': sum over X (columns) */
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            message_to_neighbor[dna_y] = 0.0;
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                message_to_neighbor[dna_y] += factor[dna_x * 4 + dna_y];
            }
        }
    } else if (from->x == to->x || from->x == to->y) {
        /* Shared variable is X of 'from': sum over Y (rows) */
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            message_to_neighbor[dna_x] = 0.0;
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                message_to_neighbor[dna_x] += factor[dna_x * 4 + dna_y];
            }
        }
    } else {
        fprintf(stderr, "Error: No shared variable in marginalization\n");
        return;
    }

    /* Rescale message */
    for (int i = 0; i < 4; i++) {
        if (message_to_neighbor[i] > largest_element) {
            largest_element = message_to_neighbor[i];
        }
    }

    if (largest_element > 0.0) {
        for (int i = 0; i < 4; i++) {
            message_to_neighbor[i] /= largest_element;
        }
        log_scaling_factor += log(largest_element);
    } else {
        fprintf(stderr, "Error: All-zero message in marginalization\n");
        return;
    }

    /* C. TRANSMIT: Store message in receiver */
    clique_store_message(to, from, message_to_neighbor, log_scaling_factor);
}

/* Compute belief for a clique after receiving all messages */
void clique_compute_belief(Clique* c) {
    if (!c) return;

    /* Start with initial potential */
    memcpy(c->factor, c->initial_potential, 16 * sizeof(double));

    /* Build list of all neighbors (parent + children) */
    Clique* neighbors[MAX_CHILDREN + 1];
    int num_neighbors = 0;

    if (c->parent != c) {
        neighbors[num_neighbors++] = c->parent;
    }
    for (int i = 0; i < c->num_children; i++) {
        neighbors[num_neighbors++] = c->children[i];
    }

    c->log_scaling_factor_for_clique = 0.0;

    /* Multiply by messages from all neighbors */
    for (int n = 0; n < num_neighbors; n++) {
        Clique* neighbor = neighbors[n];
        int neighbor_idx = clique_get_neighbor_index(c, neighbor);
        if (neighbor_idx < 0) continue;

        double* message_from = c->messages_from_neighbors[neighbor_idx];
        c->log_scaling_factor_for_clique += c->log_scaling_factors_for_messages[neighbor_idx];

        /* Determine if shared variable is X or Y of 'c' */
        if (c->y == neighbor->x || c->y == neighbor->y) {
            /* Shared variable is Y: row-wise multiplication */
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    c->factor[dna_x * 4 + dna_y] *= message_from[dna_y];
                }
            }
        } else if (c->x == neighbor->x || c->x == neighbor->y) {
            /* Shared variable is X: column-wise multiplication */
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    c->factor[dna_x * 4 + dna_y] *= message_from[dna_x];
                }
            }
        }

        /* Rescale factor */
        double largest = 0.0;
        for (int i = 0; i < 16; i++) {
            if (c->factor[i] > largest) {
                largest = c->factor[i];
            }
        }

        if (largest > 0.0) {
            for (int i = 0; i < 16; i++) {
                c->factor[i] /= largest;
            }
            c->log_scaling_factor_for_clique += log(largest);
        }
    }

    /* Final rescaling */
    double scaling = 0.0;
    for (int i = 0; i < 16; i++) {
        scaling += c->factor[i];
    }

    if (scaling > 0.0) {
        for (int i = 0; i < 16; i++) {
            c->factor[i] /= scaling;
        }
        c->log_scaling_factor_for_clique += log(scaling);
    }

    /* Set belief to factor */
    memcpy(c->belief, c->factor, 16 * sizeof(double));
}

/* Calibrate tree: send messages leaves->root, then root->leaves, then compute beliefs */
void clique_tree_calibrate(CliqueTree* ct) {
    if (!ct || !ct->root) return;

    /* Send messages from leaves to root (post-order) */
    for (int i = 0; i < ct->traversal_size; i++) {
        Clique* child = ct->post_order_traversal[i];
        if (child->parent && child->parent != child) {
            clique_tree_send_message(ct, child, child->parent);
        }
    }

    /* Send messages from root to leaves (pre-order) */
    for (int i = 0; i < ct->traversal_size; i++) {
        Clique* parent = ct->pre_order_traversal[i];
        for (int j = 0; j < parent->num_children; j++) {
            clique_tree_send_message(ct, parent, parent->children[j]);
        }
    }

    /* Compute beliefs for all cliques */
    for (int i = 0; i < ct->num_cliques; i++) {
        clique_compute_belief(ct->cliques[i]);
    }
}

/* Construct clique tree from SEM tree
 * Each edge (parent -> child) in the directed SEM tree becomes a clique
 * Cliques are connected when they share a variable
 */
void sem_construct_clique_tree(SEM* sem) {
    if (!sem || !sem->root) return;

    /* Clean up existing clique tree */
    if (sem->clique_tree) {
        for (int i = 0; i < sem->clique_tree->num_cliques; i++) {
            clique_destroy(sem->clique_tree->cliques[i]);
        }
        clique_tree_destroy(sem->clique_tree);
    }

    /* Create new clique tree */
    sem->clique_tree = clique_tree_create();
    if (!sem->clique_tree) return;

    /* Create cliques for each edge in pre-order traversal
     * An edge is (parent, child) where parent -> child in the directed tree
     * We need to compute pre-order edges first
     */

    /* Use post-order edges (already computed) to determine tree structure */
    if (sem->num_post_order_edges == 0) {
        sem_compute_edges_for_post_order_traversal(sem);
    }

    /* Create cliques in the order of post-order edges (reversed gives pre-order) */
    /* Actually, we need to create cliques for all edges, maintaining parent-child relationship */

    /* First create all cliques (one per edge) */
    Clique** edge_to_clique = (Clique**)calloc(sem->num_post_order_edges, sizeof(Clique*));
    if (!edge_to_clique) {
        clique_tree_destroy(sem->clique_tree);
        sem->clique_tree = NULL;
        return;
    }

    /* Create cliques in reverse post-order (which is pre-order) */
    for (int i = sem->num_post_order_edges - 1; i >= 0; i--) {
        SEM_vertex* parent = sem->post_order_parent[i];
        SEM_vertex* child = sem->post_order_child[i];

        Clique* c = clique_create(parent, child, sem->packed_patterns);
        if (!c) {
            for (int j = sem->num_post_order_edges - 1; j > i; j--) {
                clique_destroy(edge_to_clique[j]);
            }
            free(edge_to_clique);
            clique_tree_destroy(sem->clique_tree);
            sem->clique_tree = NULL;
            return;
        }

        c->root_variable = sem->root;
        c->post_order_edge_index = i;  /* Store edge index for EM algorithm */
        edge_to_clique[i] = c;
        clique_tree_add_clique(sem->clique_tree, c);

        /* Set as root clique if parent is root of SEM tree */
        if (parent->parent == parent && !sem->clique_tree->root_set) {
            clique_tree_set_root(sem->clique_tree, c);
        }
    }

    /* Now connect cliques based on shared variables */
    /* Two cliques C_i = (X_i, Y_i) and C_j = (X_j, Y_j) are connected if:
     * - Y_i == X_j (C_i is parent of C_j)
     * - Y_j == X_i (C_j is parent of C_i)
     * - X_i == X_j and one is root clique (siblings under root)
     */
    for (int i = 0; i < sem->clique_tree->num_cliques; i++) {
        Clique* c_i = sem->clique_tree->cliques[i];

        for (int j = i + 1; j < sem->clique_tree->num_cliques; j++) {
            Clique* c_j = sem->clique_tree->cliques[j];

            if (c_i->y == c_j->x) {
                /* C_i.y == C_j.x: C_i is parent of C_j */
                clique_add_child(c_i, c_j);
                clique_set_parent(c_j, c_i);
            } else if (c_j->y == c_i->x) {
                /* C_j.y == C_i.x: C_j is parent of C_i */
                clique_add_child(c_j, c_i);
                clique_set_parent(c_i, c_j);
            } else if (c_i->x == c_j->x && c_i->parent == c_i) {
                /* Same X variable and C_i is root clique: C_i is parent of C_j */
                clique_add_child(c_i, c_j);
                clique_set_parent(c_j, c_i);
            }
        }
    }

    /* Recompute leaves (cliques with no children) */
    sem->clique_tree->num_leaves = 0;
    for (int i = 0; i < sem->clique_tree->num_cliques; i++) {
        Clique* c = sem->clique_tree->cliques[i];
        if (c->out_degree == 0) {
            sem->clique_tree->leaves[sem->clique_tree->num_leaves++] = c;
        }
    }

    /* Compute traversal orders */
    clique_tree_compute_traversal_orders(sem->clique_tree);

    free(edge_to_clique);

    printf("Constructed clique tree with %d cliques, %d leaves\n",
           sem->clique_tree->num_cliques, sem->clique_tree->num_leaves);
}

/* Main belief propagation algorithm: compute log-likelihood using clique tree */
void sem_compute_log_likelihood_using_propagation(SEM* sem) {
    if (!sem || !sem->packed_patterns) {
        fprintf(stderr, "Error: No patterns loaded. Call sem_read_patterns_from_file() first.\n");
        return;
    }

    /* Construct clique tree if not already done */
    if (!sem->clique_tree || !sem->clique_tree->root) {
        sem_construct_clique_tree(sem);
    }

    if (!sem->clique_tree || !sem->clique_tree->root) {
        fprintf(stderr, "Error: Failed to construct clique tree.\n");
        return;
    }

    sem->log_likelihood = 0.0;
    Clique* root_clique = sem->clique_tree->root;

    /* Set base potentials from model parameters (done once outside pattern loop) */
    clique_tree_set_base_potentials(sem->clique_tree);

    /* Iterate over each pattern */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        /* Apply evidence for this site and reset messages (inside pattern loop) */
        clique_tree_apply_evidence_and_reset(sem->clique_tree, pat_idx);

        /* Calibrate tree (message passing) */
        clique_tree_calibrate(sem->clique_tree);

        /* Compute site likelihood from root clique */
        /* Marginalize over Y to get P(data | X) for root variable X */
        double marginal_x[4];
        clique_marginalize_over_variable(root_clique, root_clique->y, marginal_x);

        /* Weight by root probabilities: P(data) = sum_x P(x) * P(data | x)
         * Note: The root clique's belief does NOT include P(X) multiplication
         * (that was removed from clique_set_initial_potential_and_belief).
         * So we must multiply here.
         */
        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += sem->root_probability[dna] * marginal_x[dna];
        }

        if (site_likelihood <= 0.0) {
            fprintf(stderr, "Error: site likelihood <= 0 at pattern %d\n", pat_idx);
            return;
        }

        /* Combine with accumulated log scaling factors */
        double site_log_likelihood = root_clique->log_scaling_factor_for_clique + log(site_likelihood);

        /* Use pattern weight */
        int weight = sem->pattern_weights[pat_idx];
        sem->log_likelihood += site_log_likelihood * weight;
    }
}

/* ========== Memoized Message Passing ========== */

#define INITIAL_CACHE_CAPACITY 64

/* Helper: Look up cached message by signature hash */
static int cache_lookup(int* signatures, int cache_size, int signature) {
    for (int i = 0; i < cache_size; i++) {
        if (signatures[i] == signature) {
            return i;
        }
    }
    return -1;  /* Not found */
}

/* Helper: Add message to cache */
static int cache_add(Clique* c, int signature, double* message, double log_scale, bool is_upward) {
    int* signatures = is_upward ? c->upward_cache_signatures : c->downward_cache_signatures;
    double** messages = is_upward ? c->upward_cache_messages : c->downward_cache_messages;
    double* scales = is_upward ? c->upward_cache_scales : c->downward_cache_scales;
    int* cache_size = is_upward ? &c->upward_cache_size : &c->downward_cache_size;
    int* capacity = is_upward ? &c->upward_cache_capacity : &c->downward_cache_capacity;

    /* Expand cache if needed */
    if (*cache_size >= *capacity) {
        int new_capacity = (*capacity == 0) ? INITIAL_CACHE_CAPACITY : (*capacity * 2);

        int* new_sigs = (int*)realloc(signatures, new_capacity * sizeof(int));
        double** new_msgs = (double**)realloc(messages, new_capacity * sizeof(double*));
        double* new_scales = (double*)realloc(scales, new_capacity * sizeof(double));

        if (!new_sigs || !new_msgs || !new_scales) {
            return -1;  /* Allocation failed */
        }

        /* Initialize new message slots */
        for (int i = *capacity; i < new_capacity; i++) {
            new_msgs[i] = (double*)calloc(4, sizeof(double));
            if (!new_msgs[i]) return -1;
        }

        if (is_upward) {
            c->upward_cache_signatures = new_sigs;
            c->upward_cache_messages = new_msgs;
            c->upward_cache_scales = new_scales;
            c->upward_cache_capacity = new_capacity;
        } else {
            c->downward_cache_signatures = new_sigs;
            c->downward_cache_messages = new_msgs;
            c->downward_cache_scales = new_scales;
            c->downward_cache_capacity = new_capacity;
        }

        signatures = new_sigs;
        messages = new_msgs;
        scales = new_scales;
        *capacity = new_capacity;
    }

    /* Add to cache */
    int idx = *cache_size;
    signatures[idx] = signature;
    memcpy(messages[idx], message, 4 * sizeof(double));
    scales[idx] = log_scale;
    (*cache_size)++;

    return idx;
}

/* Send message with memoization
 * For upward messages: cache based on subtree pattern signature
 * For downward messages: cache based on complement pattern signature
 */
void clique_tree_send_message_with_memoization(CliqueTree* ct, Clique* from, Clique* to) {
    if (!ct || !from || !to) return;

    bool is_upward = (from->parent == to);

    if (is_upward && from->num_subtree_leaves > 0) {
        /* UPWARD MESSAGE: depends on observed variables in from's subtree */
        int signature = clique_get_subtree_signature(from, ct->site);

        /* Check cache */
        int cache_idx = cache_lookup(from->upward_cache_signatures,
                                      from->upward_cache_size, signature);

        if (cache_idx >= 0) {
            /* Cache hit! Reuse the cached message */
            ct->cache_hits++;
            from->upward_cache_hits++;

            /* Get neighbor index in 'to' clique */
            int neighbor_idx = clique_get_neighbor_index(to, from);
            if (neighbor_idx >= 0) {
                memcpy(to->messages_from_neighbors[neighbor_idx],
                       from->upward_cache_messages[cache_idx], 4 * sizeof(double));
                to->log_scaling_factors_for_messages[neighbor_idx] =
                    from->upward_cache_scales[cache_idx];
            }
            return;
        }

        /* Cache miss - compute the message */
        ct->cache_misses++;
        from->upward_cache_misses++;
        clique_tree_send_message(ct, from, to);

        /* Cache the result */
        int neighbor_idx = clique_get_neighbor_index(to, from);
        if (neighbor_idx >= 0) {
            cache_add(from, signature, to->messages_from_neighbors[neighbor_idx],
                     to->log_scaling_factors_for_messages[neighbor_idx], true);
        }

    } else if (!is_upward && to->num_complement_leaves > 0) {
        /* DOWNWARD MESSAGE: depends on observed variables NOT in to's subtree */
        int signature = clique_get_complement_signature(to, ct->site);

        /* Check cache */
        int cache_idx = cache_lookup(to->downward_cache_signatures,
                                      to->downward_cache_size, signature);

        if (cache_idx >= 0) {
            /* Cache hit! */
            ct->downward_cache_hits++;
            to->downward_cache_hits++;

            int neighbor_idx = clique_get_neighbor_index(to, from);
            if (neighbor_idx >= 0) {
                memcpy(to->messages_from_neighbors[neighbor_idx],
                       to->downward_cache_messages[cache_idx], 4 * sizeof(double));
                to->log_scaling_factors_for_messages[neighbor_idx] =
                    to->downward_cache_scales[cache_idx];
            }
            return;
        }

        /* Cache miss */
        ct->downward_cache_misses++;
        to->downward_cache_misses++;
        clique_tree_send_message(ct, from, to);

        /* Cache the result */
        int neighbor_idx = clique_get_neighbor_index(to, from);
        if (neighbor_idx >= 0) {
            cache_add(to, signature, to->messages_from_neighbors[neighbor_idx],
                     to->log_scaling_factors_for_messages[neighbor_idx], false);
        }

    } else {
        /* No memoization possible (root clique or empty subtree/complement) */
        clique_tree_send_message(ct, from, to);
    }
}

/* Calibrate clique tree with memoization */
void clique_tree_calibrate_with_memoization(CliqueTree* ct) {
    if (!ct || !ct->root) return;

    /* Upward pass: leaves to root (post-order) */
    for (int i = 0; i < ct->traversal_size; i++) {
        Clique* child = ct->post_order_traversal[i];
        Clique* parent = child->parent;
        if (parent && parent != child) {
            clique_tree_send_message_with_memoization(ct, child, parent);
        }
    }

    /* Downward pass: root to leaves (pre-order = reverse post-order) */
    for (int i = ct->traversal_size - 1; i >= 0; i--) {
        Clique* child = ct->post_order_traversal[i];
        Clique* parent = child->parent;
        if (parent && parent != child) {
            clique_tree_send_message_with_memoization(ct, parent, child);
        }
    }

    /* Compute beliefs for all cliques */
    for (int i = 0; i < ct->num_cliques; i++) {
        clique_compute_belief(ct->cliques[i]);
    }
}

/* Compute log-likelihood using memoized propagation algorithm
 * This caches messages based on subtree pattern signatures for speedup
 */
void sem_compute_log_likelihood_with_memoization(SEM* sem) {
    if (!sem || !sem->packed_patterns) {
        fprintf(stderr, "Error: No patterns loaded.\n");
        return;
    }

    /* Construct clique tree if not already done */
    if (!sem->clique_tree || !sem->clique_tree->root) {
        sem_construct_clique_tree(sem);
    }

    if (!sem->clique_tree || !sem->clique_tree->root) {
        fprintf(stderr, "Error: Failed to construct clique tree.\n");
        return;
    }

    /* Compute subtree and complement leaves (once, before pattern loop) */
    clique_tree_compute_all_subtree_leaves(sem->clique_tree);
    clique_tree_compute_all_complement_leaves(sem->clique_tree);

    /* Reset cache statistics */
    clique_tree_reset_cache_statistics(sem->clique_tree);

    sem->log_likelihood = 0.0;
    Clique* root_clique = sem->clique_tree->root;

    /* Set base potentials from model parameters (done once outside pattern loop) */
    clique_tree_set_base_potentials(sem->clique_tree);

    /* Iterate over each pattern */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        /* Apply evidence for this site and reset messages (inside pattern loop) */
        clique_tree_apply_evidence_and_reset(sem->clique_tree, pat_idx);

        /* Calibrate tree using memoization */
        clique_tree_calibrate_with_memoization(sem->clique_tree);

        /* Compute site likelihood from root clique */
        double marginal_x[4];
        clique_marginalize_over_variable(root_clique, root_clique->y, marginal_x);

        /* Weight by root probabilities */
        double site_likelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            site_likelihood += sem->root_probability[dna] * marginal_x[dna];
        }

        if (site_likelihood <= 0.0) {
            fprintf(stderr, "Error: site likelihood <= 0 at pattern %d\n", pat_idx);
            return;
        }

        double site_log_likelihood = root_clique->log_scaling_factor_for_clique + log(site_likelihood);

        /* Use pattern weight */
        int weight = sem->pattern_weights[pat_idx];
        sem->log_likelihood += site_log_likelihood * weight;
    }

    /* Report memoization statistics */
    printf("\n=== Memoization Statistics ===\n");
    printf("Upward cache hits: %d\n", sem->clique_tree->cache_hits);
    printf("Upward cache misses: %d\n", sem->clique_tree->cache_misses);
    int total_upward = sem->clique_tree->cache_hits + sem->clique_tree->cache_misses;
    if (total_upward > 0) {
        printf("Upward hit rate: %.1f%%\n", 100.0 * sem->clique_tree->cache_hits / total_upward);
    }
    printf("Downward cache hits: %d\n", sem->clique_tree->downward_cache_hits);
    printf("Downward cache misses: %d\n", sem->clique_tree->downward_cache_misses);
    int total_downward = sem->clique_tree->downward_cache_hits + sem->clique_tree->downward_cache_misses;
    if (total_downward > 0) {
        printf("Downward hit rate: %.1f%%\n", 100.0 * sem->clique_tree->downward_cache_hits / total_downward);
    }
}
