#define _POSIX_C_SOURCE 200809L  /* For strdup */
#include "embh_manager.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef USE_CUDA
#include "embh_cuda.h"
#endif

/* Create manager and run full pipeline */
Manager* manager_create(const char* edge_list_file,
                        const char* pattern_file,
                        const char* taxon_order_file,
                        const char* base_comp_file,
                        const char* root_optimize,
                        const char* root_check,
                        const char* cache_spec_file) {
    Manager* mgr = (Manager*)calloc(1, sizeof(Manager));
    if (!mgr) return NULL;

    /* Store file paths */
    mgr->edge_list_file = strdup(edge_list_file);
    mgr->pattern_file = strdup(pattern_file);
    mgr->taxon_order_file = strdup(taxon_order_file);
    mgr->base_comp_file = strdup(base_comp_file);
    mgr->root_optimize = strdup(root_optimize);
    mgr->root_check = strdup(root_check);
    mgr->cache_spec_file = cache_spec_file ? strdup(cache_spec_file) : NULL;

    if (!mgr->edge_list_file || !mgr->pattern_file || !mgr->taxon_order_file ||
        !mgr->base_comp_file || !mgr->root_optimize || !mgr->root_check) {
        manager_destroy(mgr);
        return NULL;
    }

    /* Configuration */
    mgr->verbose = false;
    mgr->max_em_iterations = 100;
    mgr->conv_threshold = 1e-5;

    /* Initialize results */
    mgr->max_log_likelihood = 0.0;
    mgr->max_log_likelihood_hss = 0.0;

    /* Create SEM instance */
    mgr->P = embh_sem_create(200, 1000);
    if (!mgr->P) {
        manager_destroy(mgr);
        return NULL;
    }

    mgr->P->verbose = mgr->verbose;
    mgr->P->max_iter = mgr->max_em_iterations;

    /* Run the full pipeline */
    manager_run_pipeline(mgr);

    return mgr;
}

/* Destroy manager and free resources */
void manager_destroy(Manager* mgr) {
    if (!mgr) return;

#ifdef USE_CUDA
    cuda_cleanup();
    cuda_free_cache_spec();
#endif

    if (mgr->P) embh_sem_destroy(mgr->P);

    free(mgr->edge_list_file);
    free(mgr->pattern_file);
    free(mgr->taxon_order_file);
    free(mgr->base_comp_file);
    free(mgr->root_optimize);
    free(mgr->root_check);
    free(mgr->cache_spec_file);

    free(mgr);
}

/* Run the full EMBH pipeline */
void manager_run_pipeline(Manager* mgr) {
    if (!mgr || !mgr->P) return;

    SEM* P = mgr->P;

    printf("\n=== EMBH C Implementation Pipeline ===\n\n");

    /* Step 1: Read topology */
    printf("Step 1: Reading tree topology from %s\n", mgr->edge_list_file);
    sem_set_edges_from_topology_file(P, mgr->edge_list_file);
    printf("  Created %d vertices\n", P->num_vertices);

    /* Step 2: Read patterns */
    printf("\nStep 2: Reading patterns from %s\n", mgr->pattern_file);
    sem_read_patterns_from_file(P, mgr->pattern_file, mgr->taxon_order_file);
    printf("  Loaded %d patterns\n", P->num_patterns);

    /* Step 3: Root tree */
    printf("\nStep 3: Rooting tree at %s\n", mgr->root_optimize);
    SEM_vertex* root_vertex = sem_get_vertex_by_name(P, mgr->root_optimize);
    if (!root_vertex) {
        fprintf(stderr, "Error: Root vertex '%s' not found\n", mgr->root_optimize);
        return;
    }
    sem_root_tree_at_vertex(P, root_vertex);
    printf("  Root set to %s (id=%d)\n", root_vertex->name, root_vertex->id);

    /* Step 4: Setup tree structure */
    printf("\nStep 4: Setting up tree structure\n");
    sem_set_leaves_from_pattern_indices(P);
    sem_set_vertex_vector_except_root(P);
    printf("  Number of leaves: %d\n", P->num_leaves);
    printf("  Number of non-root vertices: %d\n", P->num_non_root_vertices);

    /* Step 5: Setup F81 model */
    printf("\nStep 5: Setting F81 model from %s\n", mgr->base_comp_file);
    sem_set_f81_model(P, mgr->base_comp_file);
    printf("  Root probability (pi): [%.6f, %.6f, %.6f, %.6f]\n",
           P->root_probability[0], P->root_probability[1],
           P->root_probability[2], P->root_probability[3]);
    printf("  F81 mu: %.6f\n", P->F81_mu);

    /* Set transition matrices for all edges */
    for (int i = 0; i < P->num_non_root_vertices; i++) {
        SEM_vertex* v = P->non_root_vertices[i];
        if (v) {
            sem_set_f81_matrix(P, v);
        }
    }
    printf("  Set F81 matrices for %d edges\n", P->num_non_root_vertices);

    /* Step 6: Compute log-likelihood using pruning algorithm */
    printf("\n=== Computing log-likelihood using pruning algorithm ===\n");
    clock_t start_pruning = clock();
    sem_compute_log_likelihood_using_patterns(P);
    clock_t end_pruning = clock();
    double time_pruning = (double)(end_pruning - start_pruning) / CLOCKS_PER_SEC;
    printf("Log-likelihood (pruning): %.11f\n", P->log_likelihood);
    printf("Time: %.4f seconds\n", time_pruning);

#ifdef USE_CUDA
    /* Step 6b: Compute log-likelihood using CUDA-accelerated pruning */
    printf("\n=== Computing log-likelihood using CUDA ===\n");
    if (cuda_is_available()) {
        cuda_print_device_info();

        clock_t start_cuda = clock();
        double cuda_ll = cuda_compute_log_likelihood(P);
        clock_t end_cuda = clock();
        double time_cuda = (double)(end_cuda - start_cuda) / CLOCKS_PER_SEC;

        printf("Log-likelihood (CUDA): %.11f\n", cuda_ll);
        printf("Time: %.4f seconds\n", time_cuda);

        /* Verify correctness */
        double diff = fabs(cuda_ll - P->log_likelihood);
        printf("Difference from CPU: %.2e\n", diff);
        if (diff < 1e-8) {
            printf("CUDA result VERIFIED (matches CPU)\n");
        } else if (diff < 1e-5) {
            printf("CUDA result close to CPU (small numerical difference)\n");
        } else {
            printf("WARNING: CUDA result differs significantly from CPU!\n");
        }

        if (time_cuda > 0 && time_pruning > 0) {
            double speedup = time_pruning / time_cuda;
            printf("CUDA speedup over CPU pruning: %.2fx\n", speedup);
        }

        /* Step 6c: Compute log-likelihood using CUDA with subtree-level memoization */
        if (mgr->cache_spec_file) {
            /* Use selective caching with pre-computed cache specification */
            printf("\n=== Computing log-likelihood using CUDA with SELECTIVE subtree memoization ===\n");
            printf("Using cache specification: %s\n", mgr->cache_spec_file);
            clock_t start_cuda_subtree = clock();

            /* Initialize selective memoization with cache spec */
            if (cuda_init_selective_subtree_memoization(P, mgr->cache_spec_file) == 0) {
                double cuda_subtree_ll = cuda_compute_log_likelihood_subtree_memoized(P);
                clock_t end_cuda_subtree = clock();
                double time_cuda_subtree = (double)(end_cuda_subtree - start_cuda_subtree) / CLOCKS_PER_SEC;

                printf("Log-likelihood (CUDA selective subtree memo): %.11f\n", cuda_subtree_ll);
                printf("Time: %.4f seconds\n", time_cuda_subtree);

                /* Verify correctness */
                double diff_subtree = fabs(cuda_subtree_ll - P->log_likelihood);
                printf("Difference from CPU: %.2e\n", diff_subtree);
                if (diff_subtree < 1e-8) {
                    printf("CUDA selective subtree memo result VERIFIED (matches CPU)\n");
                } else if (diff_subtree < 1e-5) {
                    printf("CUDA selective subtree memo result close to CPU (small numerical difference)\n");
                } else {
                    printf("WARNING: CUDA selective subtree memo result differs significantly from CPU!\n");
                }

                if (time_cuda_subtree > 0 && time_cuda > 0) {
                    double subtree_speedup = time_cuda / time_cuda_subtree;
                    printf("Selective subtree memoization speedup over basic CUDA: %.2fx\n", subtree_speedup);
                }
            } else {
                printf("ERROR: Failed to initialize selective subtree memoization\n");
            }
        } else {
            /* Use standard (non-selective) subtree memoization */
            printf("\n=== Computing log-likelihood using CUDA with subtree memoization ===\n");
            clock_t start_cuda_subtree = clock();
            double cuda_subtree_ll = cuda_compute_log_likelihood_subtree_memoized(P);
            clock_t end_cuda_subtree = clock();
            double time_cuda_subtree = (double)(end_cuda_subtree - start_cuda_subtree) / CLOCKS_PER_SEC;

            printf("Log-likelihood (CUDA subtree memo): %.11f\n", cuda_subtree_ll);
            printf("Time: %.4f seconds\n", time_cuda_subtree);

            /* Verify correctness */
            double diff_subtree = fabs(cuda_subtree_ll - P->log_likelihood);
            printf("Difference from CPU: %.2e\n", diff_subtree);
            if (diff_subtree < 1e-8) {
                printf("CUDA subtree memo result VERIFIED (matches CPU)\n");
            } else if (diff_subtree < 1e-5) {
                printf("CUDA subtree memo result close to CPU (small numerical difference)\n");
            } else {
                printf("WARNING: CUDA subtree memo result differs significantly from CPU!\n");
            }

            if (time_cuda_subtree > 0 && time_cuda > 0) {
                double subtree_speedup = time_cuda / time_cuda_subtree;
                printf("Subtree memoization speedup over basic CUDA: %.2fx\n", subtree_speedup);
            }
        }
    } else {
        printf("CUDA not available on this system\n");
    }
#endif

#ifndef USE_CUDA
    /* Step 7: Compute log-likelihood using propagation algorithm (CPU only) */
    printf("\n=== Computing log-likelihood using propagation algorithm ===\n");
    clock_t start_prop = clock();
    sem_compute_log_likelihood_using_propagation(P);
    clock_t end_prop = clock();
    double time_propagation = (double)(end_prop - start_prop) / CLOCKS_PER_SEC;
    printf("Log-likelihood (propagation): %.11f\n", P->log_likelihood);
    printf("Time: %.4f seconds\n", time_propagation);

    /* Step 7b: Compute log-likelihood using memoized propagation */
    printf("\n=== Computing log-likelihood using MEMOIZED propagation ===\n");
    clock_t start_memo = clock();
    sem_compute_log_likelihood_with_memoization(P);
    clock_t end_memo = clock();
    double time_memoized = (double)(end_memo - start_memo) / CLOCKS_PER_SEC;
    printf("Log-likelihood (memoized): %.11f\n", P->log_likelihood);
    printf("Time: %.4f seconds\n", time_memoized);

    /* Performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Pruning algorithm:     %.4f seconds\n", time_pruning);
    printf("Propagation algorithm: %.4f seconds\n", time_propagation);
    printf("Memoized propagation:  %.4f seconds\n", time_memoized);
    if (time_propagation > 0 && time_memoized > 0) {
        double speedup = time_propagation / time_memoized;
        double time_saved = time_propagation - time_memoized;
        printf("Speedup from memoization: %.2fx (saved %.4f seconds)\n", speedup, time_saved);
    }
#endif

    /* Step 8: Run EM with Aitken acceleration */
#ifdef USE_CUDA
    printf("\n=== Running EM algorithm with Aitken acceleration (GPU-accelerated) ===\n");
#else
    printf("\n=== Running EM algorithm with Aitken acceleration ===\n");
#endif
    sem_run_em_with_aitken(P, mgr->max_em_iterations);
    mgr->max_log_likelihood = P->log_likelihood;
    printf("Final log-likelihood after EM: %.11f\n", P->log_likelihood);

    /* Step 9: Evaluate BH model at check root */
    if (mgr->root_check && strlen(mgr->root_check) > 0) {
        printf("\n=== Evaluating BH model at root %s ===\n", mgr->root_check);
        manager_evaluate_bh_at_check_root(mgr);
    }

    printf("\n=== Pipeline Complete ===\n");
}

/* Evaluate BH model with root at check vertex */
void manager_evaluate_bh_at_check_root(Manager* mgr) {
    if (!mgr || !mgr->P || !mgr->root_check) return;

    SEM* P = mgr->P;

    printf("Evaluating BH model at root %s using HSS reparameterization\n", mgr->root_check);

    /* Use HSS reparameterization to compute likelihood at different root
     * This should give the SAME likelihood as at the original root
     */
    sem_evaluate_bh_at_check_root(P, mgr->root_check);
    mgr->max_log_likelihood_hss = P->log_likelihood;
}
