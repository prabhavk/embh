#include "embh_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef USE_CUDA
#include "embh_cuda.h"
#endif

/*
 * EM Algorithm for phylogenetic parameter estimation
 *
 * This implements the Expectation-Maximization algorithm for estimating
 * transition matrices and root probabilities from DNA site patterns.
 *
 * Key steps:
 * E-step: Compute expected counts using belief propagation
 * M-step: Update parameters from normalized expected counts
 */

/* Initialize expected counts for all vertices and edges */
void sem_initialize_expected_counts(SEM* sem) {
    if (!sem) return;

    /* Allocate expected counts for vertices (only internal vertices) */
    if (!sem->expected_counts_for_vertex) {
        sem->expected_counts_for_vertex = (double**)calloc(sem->num_vertices, sizeof(double*));
        if (!sem->expected_counts_for_vertex) return;

        for (int i = 0; i < sem->num_vertices; i++) {
            sem->expected_counts_for_vertex[i] = (double*)calloc(4, sizeof(double));
            if (!sem->expected_counts_for_vertex[i]) return;
        }
    }

    /* Allocate expected counts for edges (parent-child pairs) */
    if (!sem->expected_counts_for_edge) {
        sem->expected_counts_for_edge = (double**)calloc(sem->num_post_order_edges, sizeof(double*));
        if (!sem->expected_counts_for_edge) return;

        for (int i = 0; i < sem->num_post_order_edges; i++) {
            sem->expected_counts_for_edge[i] = (double*)calloc(16, sizeof(double));
            if (!sem->expected_counts_for_edge[i]) return;
        }
    }

    /* Allocate posterior probabilities for vertices */
    if (!sem->posterior_prob_for_vertex) {
        sem->posterior_prob_for_vertex = (double**)calloc(sem->num_vertices, sizeof(double*));
        if (!sem->posterior_prob_for_vertex) return;

        for (int i = 0; i < sem->num_vertices; i++) {
            sem->posterior_prob_for_vertex[i] = (double*)calloc(4, sizeof(double));
            if (!sem->posterior_prob_for_vertex[i]) return;
        }
    }

    /* Allocate posterior probabilities for edges */
    if (!sem->posterior_prob_for_edge) {
        sem->posterior_prob_for_edge = (double**)calloc(sem->num_post_order_edges, sizeof(double*));
        if (!sem->posterior_prob_for_edge) return;

        for (int i = 0; i < sem->num_post_order_edges; i++) {
            sem->posterior_prob_for_edge[i] = (double*)calloc(16, sizeof(double));
            if (!sem->posterior_prob_for_edge[i]) return;
        }
    }

    /* Zero out all counts */
    for (int i = 0; i < sem->num_vertices; i++) {
        memset(sem->expected_counts_for_vertex[i], 0, 4 * sizeof(double));
    }

    for (int i = 0; i < sem->num_post_order_edges; i++) {
        memset(sem->expected_counts_for_edge[i], 0, 16 * sizeof(double));
    }
}

/* Add to expected counts after calibration for one site */
void sem_add_to_expected_counts(SEM* sem, int site_weight) {
    if (!sem || !sem->clique_tree) return;

    /* Track which vertices we've already counted (using a simple visited array) */
    bool* vertex_visited = (bool*)calloc(sem->num_vertices, sizeof(bool));
    if (!vertex_visited) return;

    /* For each clique, add marginal probabilities weighted by site weight */
    for (int c_idx = 0; c_idx < sem->clique_tree->num_cliques; c_idx++) {
        Clique* c = sem->clique_tree->cliques[c_idx];

        /* Add counts for X variable (parent vertex) if not already counted */
        if (!vertex_visited[c->x->id] && !c->x->observed) {
            double marginal_x[4];
            clique_marginalize_over_variable(c, c->y, marginal_x);

            for (int dna = 0; dna < 4; dna++) {
                sem->expected_counts_for_vertex[c->x->id][dna] += marginal_x[dna] * site_weight;
            }
            vertex_visited[c->x->id] = true;
        }
    }

    /* Add counts for edges (from clique beliefs) */
    /* Each clique corresponds to an edge (x->y) */
    for (int c_idx = 0; c_idx < sem->clique_tree->num_cliques; c_idx++) {
        Clique* c = sem->clique_tree->cliques[c_idx];

        /* Use cached edge index (O(1) lookup instead of O(n) search) */
        int edge_idx = c->post_order_edge_index;
        if (edge_idx >= 0 && edge_idx < sem->num_post_order_edges) {
            /* Add belief (joint probability) weighted by site weight */
            for (int i = 0; i < 16; i++) {
                sem->expected_counts_for_edge[edge_idx][i] += c->belief[i] * site_weight;
            }
        }
    }

    free(vertex_visited);
}

/* Compute marginal probabilities from expected counts */
void sem_compute_marginal_probabilities_from_counts(SEM* sem) {
    if (!sem) return;

    /* Normalize vertex counts to get posterior probabilities */
    for (int v_id = 0; v_id < sem->num_vertices; v_id++) {
        double sum = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            sum += sem->expected_counts_for_vertex[v_id][dna];
        }

        if (sum > 0.0) {
            for (int dna = 0; dna < 4; dna++) {
                sem->posterior_prob_for_vertex[v_id][dna] =
                    sem->expected_counts_for_vertex[v_id][dna] / sum;
            }
        }
    }

    /* Normalize edge counts to get joint posterior probabilities P(X,Y) */
    for (int edge_idx = 0; edge_idx < sem->num_post_order_edges; edge_idx++) {
        double sum = 0.0;
        for (int i = 0; i < 16; i++) {
            sum += sem->expected_counts_for_edge[edge_idx][i];
        }

        if (sum > 0.0) {
            for (int i = 0; i < 16; i++) {
                sem->posterior_prob_for_edge[edge_idx][i] =
                    sem->expected_counts_for_edge[edge_idx][i] / sum;
            }
        }
    }
}

/* Compute P(Y|X) from P(X,Y) */
static void compute_conditional_from_joint(const double* p_xy, double* p_y_given_x) {
    double p_x[4];

    /* Compute marginal P(X) by summing over Y */
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        p_x[dna_x] = 0.0;
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            p_x[dna_x] += p_xy[dna_x * 4 + dna_y];
        }
    }

    /* Compute P(Y|X) = P(X,Y) / P(X) */
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            if (p_x[dna_x] > 0.0) {
                p_y_given_x[dna_x * 4 + dna_y] = p_xy[dna_x * 4 + dna_y] / p_x[dna_x];
            } else {
                p_y_given_x[dna_x * 4 + dna_y] = 0.0;
            }
        }
    }
}

/* M-step: Update parameters from posterior probabilities */
void sem_update_parameters_from_posteriors(SEM* sem) {
    if (!sem || !sem->root) return;

    /* Update root probability from posterior of root vertex */
    double sum_root = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        sum_root += sem->posterior_prob_for_vertex[sem->root->id][dna];
    }

    if (sum_root > 0.0) {
        for (int dna = 0; dna < 4; dna++) {
            sem->root_probability[dna] = sem->posterior_prob_for_vertex[sem->root->id][dna];
            sem->root->root_probability[dna] = sem->root_probability[dna];
        }
    }
    /* If no posterior for root (shouldn't happen), keep old root probability */

    /* Update transition matrices from posterior joint probabilities */
    for (int edge_idx = 0; edge_idx < sem->num_post_order_edges; edge_idx++) {
        SEM_vertex* child = sem->post_order_child[edge_idx];

        /* Check if we have valid posterior for this edge */
        double sum_edge = 0.0;
        for (int i = 0; i < 16; i++) {
            sum_edge += sem->posterior_prob_for_edge[edge_idx][i];
        }

        if (sum_edge > 0.0) {
            /* Compute P(Y|X) from P(X,Y) */
            compute_conditional_from_joint(sem->posterior_prob_for_edge[edge_idx],
                                           child->transition_matrix);
        }
        /* If no posterior for edge (shouldn't happen), keep old transition matrix */
    }
}

/* Single EM iteration: E-step followed by M-step */
void sem_em_iteration(SEM* sem) {
    if (!sem || !sem->packed_patterns) {
        fprintf(stderr, "Error: No patterns loaded.\n");
        return;
    }

    /* Construct clique tree if needed */
    if (!sem->clique_tree || !sem->clique_tree->root) {
        sem_construct_clique_tree(sem);
    }

    /* Initialize expected counts */
    sem_initialize_expected_counts(sem);

    /* Set base potentials from model parameters (done once outside pattern loop) */
    clique_tree_set_base_potentials(sem->clique_tree);

    /* E-step: Accumulate expected counts over all patterns */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        /* Apply evidence for this site and reset messages (inside pattern loop) */
        clique_tree_apply_evidence_and_reset(sem->clique_tree, pat_idx);
        clique_tree_calibrate(sem->clique_tree);

        /* Add weighted counts */
        int weight = sem->pattern_weights[pat_idx];
        sem_add_to_expected_counts(sem, weight);
    }

    /* Compute posterior probabilities from counts */
    sem_compute_marginal_probabilities_from_counts(sem);

    /* M-step: Update parameters */
    sem_update_parameters_from_posteriors(sem);
}

/* Run EM with Aitken acceleration for convergence */
void sem_run_em_with_aitken(SEM* sem, int max_iterations) {
    if (!sem) return;

    const double MAX_RELIABLE_RATE = 0.95;
    const int MIN_ITER_FOR_AITKEN = 3;

    /* Compute number of sites */
    int num_sites = 0;
    for (int i = 0; i < sem->num_patterns; i++) {
        num_sites += sem->pattern_weights[i];
    }

    /* Tolerance: 1e-5 per site */
    const double PER_SITE_TOLERANCE = 1e-5;
    const double TOLERANCE = PER_SITE_TOLERANCE * num_sites;

#ifdef USE_CUDA
    /* Initialize CUDA pruning for GPU-accelerated log-likelihood */
    int cuda_available = 0;
    if (cuda_is_available()) {
        printf("Initializing CUDA for GPU-accelerated EM...\n");
        if (cuda_init_pruning(sem) == 0) {
            cuda_available = 1;
            printf("CUDA initialization successful\n");
        } else {
            printf("WARNING: CUDA initialization failed, falling back to CPU\n");
        }
    }

    /* Compute initial log-likelihood */
    if (cuda_available) {
        sem->log_likelihood = cuda_compute_log_likelihood(sem);
    } else {
        sem_compute_log_likelihood_using_propagation(sem);
    }
#else
    /* Compute initial log-likelihood */
    sem_compute_log_likelihood_using_propagation(sem);
#endif
    double ll_0 = sem->log_likelihood;

    printf("\n====================================================\n");
    printf("EM with Aitken Acceleration\n");
#ifdef USE_CUDA
    if (cuda_available) {
        printf("(GPU-accelerated)\n");
    }
#endif
    printf("====================================================\n");
    printf("Initial LL: %.2f\n", ll_0);
    printf("Number of sites: %d\n", num_sites);
    printf("Convergence tolerance: %.2e (%.2e per site)\n", TOLERANCE, PER_SITE_TOLERANCE);
    printf("----------------------------------------------------------------------\n");

    double ll_prev_prev = ll_0;
    double ll_prev = ll_0;
    double ll_curr = ll_0;

    int final_iter = 0;
    bool converged = false;

    for (int iter = 1; iter <= max_iterations; iter++) {
        final_iter = iter;

        /* E-step and M-step */
        sem_em_iteration(sem);

#ifdef USE_CUDA
        /* Update GPU transition matrices after M-step */
        if (cuda_available) {
            cuda_update_transition_matrices(sem);
        }

        /* Compute new log-likelihood */
        if (cuda_available) {
            sem->log_likelihood = cuda_compute_log_likelihood(sem);
        } else {
            sem_compute_log_likelihood_using_propagation(sem);
        }
#else
        /* Compute new log-likelihood */
        sem_compute_log_likelihood_using_propagation(sem);
#endif
        ll_curr = sem->log_likelihood;

        double improvement = ll_curr - ll_prev;

        /* Aitken acceleration check */
        bool use_aitken = false;
        double aitken_ll = ll_curr;
        double aitken_distance = 0.0;
        double rate_factor = 0.0;

        if (iter >= MIN_ITER_FOR_AITKEN) {
            double delta1 = ll_prev - ll_prev_prev;
            double delta2 = ll_curr - ll_prev;
            double denominator = delta2 - delta1;

            if (fabs(delta1) > 1e-12) {
                rate_factor = fabs(delta2 / delta1);
            }

            if (fabs(denominator) > 1e-10 &&
                rate_factor > 0.01 &&
                rate_factor < MAX_RELIABLE_RATE) {

                aitken_ll = ll_curr - (delta2 * delta2) / denominator;
                aitken_distance = fabs(aitken_ll - ll_curr);
                use_aitken = true;
            }
        }

        /* Display iteration info */
        printf("Iter %3d: LL = %.2f (+%.2e)", iter, ll_curr, improvement);

        if (use_aitken) {
            printf(" | Rate: %.3f | Aitken LL: %.2f | Dist: %.2e",
                   rate_factor, aitken_ll, aitken_distance);
        } else if (iter >= MIN_ITER_FOR_AITKEN) {
            printf(" | Rate: %.3f | Aitken: disabled", rate_factor);
        }
        printf("\n");

        /* Convergence check */
        if (use_aitken && aitken_distance < TOLERANCE) {
            converged = true;
            printf("Converged (Aitken criterion)\n");
        } else if (improvement < TOLERANCE && improvement > 0) {
            converged = true;
            printf("Converged (standard criterion)\n");
        } else if (improvement < 0) {
            printf("WARNING: Likelihood decreased! Stopping.\n");
            converged = true;
        }

        if (converged) {
            break;
        }

        /* Update for next iteration */
        ll_prev_prev = ll_prev;
        ll_prev = ll_curr;
    }

    printf("----------------------------------------------------------------------\n");
    if (converged) {
        printf("Converged after %d iterations\n", final_iter);
    } else {
        printf("Reached maximum iterations (%d)\n", max_iterations);
    }

    printf("\n====================================================\n");
    printf("Final Results:\n");
    printf("  Initial LL:  %.2f\n", ll_0);
    printf("  Final LL:    %.2f\n", sem->log_likelihood);
    printf("  Improvement: %.2f\n", sem->log_likelihood - ll_0);
    printf("  Iterations:  %d\n", final_iter);
    printf("  LL/site:     %.6f\n", sem->log_likelihood / num_sites);
    printf("====================================================\n");
}

/* Reparameterize BH model for root-invariant likelihood computation
 * Computes HSS parameters using Bayes' rule as described in HSS paper.
 * After this, we can re-root the tree and get the same likelihood.
 *
 * Key insight: Store transition matrices indexed by vertex ID pairs, so we can
 * look them up after re-rooting regardless of the new tree structure.
 */
void sem_reparameterize_bh(SEM* sem) {
    if (!sem || !sem->root || sem->num_post_order_edges == 0) return;

    /* Allocate HSS matrices if not already done
     * We use 2D array indexed by [parent_id * max_vertices + child_id]
     * This allows O(1) lookup of transition matrix for any directed edge
     */
    int total_size = sem->max_vertices * sem->max_vertices;
    if (!sem->hss_matrices_forward) {
        sem->hss_matrices_forward = (double**)calloc(total_size, sizeof(double*));
        sem->hss_matrices_reverse = (double**)calloc(total_size, sizeof(double*));
        if (!sem->hss_matrices_forward || !sem->hss_matrices_reverse) return;

        /* Only allocate for actual edges (will be populated during traversal) */
        for (int i = 0; i < total_size; i++) {
            sem->hss_matrices_forward[i] = NULL;
            sem->hss_matrices_reverse[i] = NULL;
        }
    }

    /* 1. Set root's HSS probability to current root probability */
    for (int x = 0; x < 4; x++) {
        sem->root->root_prob_hss[x] = sem->root_probability[x];
    }

    /* 2. Traverse tree in pre-order (root to leaves) to compute HSS parameters
     * Pre-order is reverse of post-order
     */
    for (int edge_idx = sem->num_post_order_edges - 1; edge_idx >= 0; edge_idx--) {
        SEM_vertex* p = sem->post_order_parent[edge_idx];
        SEM_vertex* c = sem->post_order_child[edge_idx];
        int p_id = p->id;
        int c_id = c->id;

        /* Compute index for this directed edge (p -> c) */
        int forward_idx = p_id * sem->max_vertices + c_id;
        int reverse_idx = c_id * sem->max_vertices + p_id;

        /* Allocate if needed */
        if (!sem->hss_matrices_forward[forward_idx]) {
            sem->hss_matrices_forward[forward_idx] = (double*)calloc(16, sizeof(double));
            if (!sem->hss_matrices_forward[forward_idx]) return;
        }
        if (!sem->hss_matrices_reverse[reverse_idx]) {
            sem->hss_matrices_reverse[reverse_idx] = (double*)calloc(16, sizeof(double));
            if (!sem->hss_matrices_reverse[reverse_idx]) return;
        }

        /* Store forward transition matrix P(child|parent) for edge p->c */
        memcpy(sem->hss_matrices_forward[forward_idx], c->transition_matrix, 16 * sizeof(double));

        /* Get parent's HSS root probability */
        double pi_p[4], pi_c[4];
        for (int x = 0; x < 4; x++) {
            pi_p[x] = p->root_prob_hss[x];
            pi_c[x] = 0.0;
        }

        /* Compute child's marginal: pi_c[x] = sum_y pi_p[y] * P(x|y) */
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                pi_c[x] += pi_p[y] * c->transition_matrix[y * 4 + x];
            }
        }

        /* Store child's HSS root probability */
        for (int x = 0; x < 4; x++) {
            c->root_prob_hss[x] = pi_c[x];
        }

        /* Compute reverse transition matrix using Bayes' rule:
         * P(parent|child)[y][x] = P(child|parent)[x][y] * pi_p[x] / pi_c[y]
         * M_cp[y][x] = M_pc[x][y] * pi_p[x] / pi_c[y]
         *
         * This is the transition matrix for edge c->p (when c becomes parent of p)
         */
        for (int x = 0; x < 4; x++) {  /* parent state (now p is child) */
            for (int y = 0; y < 4; y++) {  /* child state (now c is parent) */
                if (pi_c[y] > 0.0) {
                    /* c->p: P(p=x|c=y) = P(c=y|p=x) * P(p=x) / P(c=y) */
                    sem->hss_matrices_reverse[reverse_idx][y * 4 + x] =
                        c->transition_matrix[x * 4 + y] * pi_p[x] / pi_c[y];
                } else {
                    sem->hss_matrices_reverse[reverse_idx][y * 4 + x] = 0.0;
                }
            }
        }
    }

    sem->hss_computed = true;
}

/* Set model parameters using HSS reparameterization for current root */
void sem_set_model_parameters_using_hss(SEM* sem) {
    if (!sem || !sem->root || !sem->hss_computed) return;

    /* Set root probability from HSS */
    for (int x = 0; x < 4; x++) {
        sem->root_probability[x] = sem->root->root_prob_hss[x];
    }

    /* Set transition matrices from HSS
     * For each edge in current tree, look up the appropriate HSS matrix.
     * This requires matching the current edge direction with stored HSS matrices.
     */
    for (int edge_idx = 0; edge_idx < sem->num_post_order_edges; edge_idx++) {
        SEM_vertex* c = sem->post_order_child[edge_idx];

        /* Copy the appropriate transition matrix
         * Since we've recomputed the tree structure, use the child's stored matrix
         * The HSS reparameterization already computed the correct matrices
         */
        memcpy(c->transition_matrix, sem->hss_matrices_forward[edge_idx], 16 * sizeof(double));
    }
}

/* Evaluate BH model at a different root position using HSS reparameterization */
void sem_evaluate_bh_at_check_root(SEM* sem, const char* root_check_name) {
    if (!sem || !root_check_name) return;

    /* First, reparameterize current model to get HSS parameters */
    sem_reparameterize_bh(sem);

    /* Get the check root vertex */
    SEM_vertex* check_root = sem_get_vertex_by_name(sem, root_check_name);
    if (!check_root) {
        fprintf(stderr, "Error: Check root vertex '%s' not found\n", root_check_name);
        return;
    }

    /* Re-root the tree at check vertex */
    sem_root_tree_at_vertex(sem, check_root);
    sem_set_vertex_vector_except_root(sem);

    /* Recompute post-order edges for new tree structure */
    sem_compute_edges_for_post_order_traversal(sem);

    /* Set model parameters using HSS reparameterization
     * This ensures likelihood remains invariant under root change
     */
    for (int x = 0; x < 4; x++) {
        sem->root_probability[x] = sem->root->root_prob_hss[x];
        sem->root->root_probability[x] = sem->root->root_prob_hss[x];
    }

    /* For each edge in the new tree, look up the appropriate HSS transition matrix
     * Key insight: The HSS matrices are stored indexed by vertex ID pairs.
     * For edge (new_p -> new_c), we look up M_hss[new_p][new_c].
     * This matrix was computed during reparameterization as either:
     * - Forward matrix (if new_p was parent of new_c in original tree)
     * - Reverse matrix (if new_c was parent of new_p in original tree)
     */
    int num_forward = 0, num_reverse = 0, num_missing = 0;
    for (int edge_idx = 0; edge_idx < sem->num_post_order_edges; edge_idx++) {
        SEM_vertex* p = sem->post_order_parent[edge_idx];
        SEM_vertex* c = sem->post_order_child[edge_idx];
        int p_id = p->id;
        int c_id = c->id;

        /* Look up transition matrix for this directed edge (p -> c)
         *
         * The HSS matrices store:
         * - hss_matrices_forward[p*max+c] = P(c|p) for original edge p->c
         * - hss_matrices_reverse[c*max+p] = P(p|c) for original edge p->c (the reverse)
         *
         * So for new edge p->c:
         * - If original was p->c: use hss_matrices_forward[p*max+c]
         * - If original was c->p: use hss_matrices_reverse[p*max+c] (which is P(c|p))
         */
        int idx = p_id * sem->max_vertices + c_id;

        /* Check if this edge exists in the forward direction (original was p->c) */
        if (sem->hss_matrices_forward[idx] != NULL) {
            /* Edge p->c was in the original tree, use forward matrix */
            memcpy(c->transition_matrix, sem->hss_matrices_forward[idx], 16 * sizeof(double));
            num_forward++;
        } else if (sem->hss_matrices_reverse[idx] != NULL) {
            /* Original was c->p, so this edge flipped direction
             * Use the reverse matrix which gives P(c|p)
             */
            memcpy(c->transition_matrix, sem->hss_matrices_reverse[idx], 16 * sizeof(double));
            num_reverse++;
        } else {
            fprintf(stderr, "Warning: Edge (%d -> %d) not found in HSS matrices\n", p_id, c_id);
            num_missing++;
        }
    }

    /* Destroy old clique tree (built for previous root)
     * This is critical - the clique tree structure depends on the tree rooting
     */
    if (sem->clique_tree) {
        for (int i = 0; i < sem->clique_tree->num_cliques; i++) {
            clique_destroy(sem->clique_tree->cliques[i]);
        }
        clique_tree_destroy(sem->clique_tree);
        sem->clique_tree = NULL;
    }

    /* Compute log-likelihood at new root using PRUNING algorithm
     * The pruning algorithm is simpler and more robust for this purpose.
     * Reset post-order edges first to ensure they're correct for new tree
     */
    sem->num_post_order_edges = 0;  /* Force recomputation */
    sem_compute_log_likelihood_using_patterns(sem);

    printf("Log-likelihood with root at %s is %.11f\n", root_check_name, sem->log_likelihood);
}

