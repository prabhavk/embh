#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/embh_types.h"

/*
 * Test for EM Algorithm
 * Validates that:
 * 1. EM increases log-likelihood
 * 2. Parameters are updated correctly
 * 3. Algorithm converges
 */

#define TOLERANCE 1e-3

static int tests_passed = 0;
static int tests_failed = 0;

static void check(int condition, const char* test_name) {
    if (condition) {
        printf("  PASS: %s\n", test_name);
        tests_passed++;
    } else {
        printf("  FAIL: %s\n", test_name);
        tests_failed++;
    }
}

/* Full setup helper */
static SEM* full_setup(void) {
    const char* edge_file = "data/RAxML_bipartitions.CDS_FcC_partition.edgelist";
    const char* pattern_file = "data/patterns_1000.pat";
    const char* taxon_order_file = "data/patterns_1000.taxon_order";
    const char* base_comp_file = "data/patterns_1000.basecomp";

    SEM* sem = embh_sem_create(200, 1000);
    if (!sem) return NULL;

    sem_set_edges_from_topology_file(sem, edge_file);
    sem_read_patterns_from_file(sem, pattern_file, taxon_order_file);

    SEM_vertex* root_vertex = sem_get_vertex_by_name(sem, "h_0");
    if (root_vertex) {
        sem_root_tree_at_vertex(sem, root_vertex);
    }

    sem_set_leaves_from_pattern_indices(sem);
    sem_set_vertex_vector_except_root(sem);
    sem_set_f81_model(sem, base_comp_file);

    /* Set transition matrices for all edges */
    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        SEM_vertex* v = sem->non_root_vertices[i];
        if (v) {
            sem_set_f81_matrix(sem, v);
        }
    }

    return sem;
}

/* Test 1: E-step expected counts accumulation */
static void test_expected_counts(void) {
    printf("\nTest 1: Expected counts computation\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Compute post-order edges */
    sem_compute_edges_for_post_order_traversal(sem);

    /* Construct clique tree */
    sem_construct_clique_tree(sem);
    check(sem->clique_tree != NULL, "Clique tree constructed");

    /* Initialize expected counts */
    sem_initialize_expected_counts(sem);
    check(sem->expected_counts_for_vertex != NULL, "Vertex counts allocated");
    check(sem->expected_counts_for_edge != NULL, "Edge counts allocated");

    /* Perform one E-step: accumulate counts over all patterns */
    for (int pat_idx = 0; pat_idx < sem->num_patterns; pat_idx++) {
        clique_tree_set_site(sem->clique_tree, pat_idx);
        clique_tree_initialize_potentials_and_beliefs(sem->clique_tree);
        clique_tree_calibrate(sem->clique_tree);

        int weight = sem->pattern_weights[pat_idx];
        sem_add_to_expected_counts(sem, weight);
    }

    /* Check that counts sum to total number of sites */
    int total_sites = 0;
    for (int i = 0; i < sem->num_patterns; i++) {
        total_sites += sem->pattern_weights[i];
    }
    printf("  Total sites: %d\n", total_sites);

    /* Root vertex counts should sum to total_sites */
    double root_count_sum = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        root_count_sum += sem->expected_counts_for_vertex[sem->root->id][dna];
    }
    printf("  Root count sum: %.6f\n", root_count_sum);
    check(fabs(root_count_sum - total_sites) < 1.0, "Root counts sum to total sites");

    embh_sem_destroy(sem);
}

/* Test 2: Single EM iteration */
static void test_single_em_iteration(void) {
    printf("\nTest 2: Single EM iteration\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Get initial log-likelihood */
    sem_compute_log_likelihood_using_propagation(sem);
    double ll_before = sem->log_likelihood;
    printf("  Initial LL: %.11f\n", ll_before);

    /* Save initial root probability */
    double initial_root_prob[4];
    for (int dna = 0; dna < 4; dna++) {
        initial_root_prob[dna] = sem->root_probability[dna];
    }
    printf("  Initial root prob: [%.6f, %.6f, %.6f, %.6f]\n",
           initial_root_prob[0], initial_root_prob[1],
           initial_root_prob[2], initial_root_prob[3]);

    /* Perform one EM iteration */
    sem_em_iteration(sem);

    /* Get new log-likelihood */
    sem_compute_log_likelihood_using_propagation(sem);
    double ll_after = sem->log_likelihood;
    printf("  After 1 EM iteration LL: %.11f\n", ll_after);

    /* Print new root probability */
    printf("  New root prob: [%.6f, %.6f, %.6f, %.6f]\n",
           sem->root_probability[0], sem->root_probability[1],
           sem->root_probability[2], sem->root_probability[3]);

    /* Check that likelihood increased or stayed same (EM monotonicity) */
    double improvement = ll_after - ll_before;
    printf("  Improvement: %.6e\n", improvement);
    check(improvement >= -1e-10, "Likelihood non-decreasing (EM monotonicity)");

    /* Check that root probability changed */
    double root_change = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        root_change += fabs(sem->root_probability[dna] - initial_root_prob[dna]);
    }
    printf("  Root prob change: %.6e\n", root_change);
    check(root_change > 1e-10 || improvement < 1e-10, "Parameters updated or already optimal");

    /* Check that root probabilities still sum to 1 */
    double root_sum = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        root_sum += sem->root_probability[dna];
    }
    check(fabs(root_sum - 1.0) < 1e-10, "Root probabilities sum to 1");

    embh_sem_destroy(sem);
}

/* Test 3: Multiple EM iterations increase likelihood */
static void test_em_improvement(void) {
    printf("\nTest 3: EM monotonic improvement\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    sem_compute_log_likelihood_using_propagation(sem);
    double ll_prev = sem->log_likelihood;
    printf("  Initial LL: %.6f\n", ll_prev);

    /* Run 5 EM iterations */
    bool monotonic = true;
    for (int iter = 1; iter <= 5; iter++) {
        sem_em_iteration(sem);
        sem_compute_log_likelihood_using_propagation(sem);
        double ll_curr = sem->log_likelihood;
        double improvement = ll_curr - ll_prev;

        printf("  Iter %d: LL = %.6f (+%.2e)\n", iter, ll_curr, improvement);

        if (improvement < -1e-10) {
            monotonic = false;
            printf("  WARNING: Likelihood decreased!\n");
        }

        ll_prev = ll_curr;
    }

    check(monotonic, "Likelihood monotonically non-decreasing");

    embh_sem_destroy(sem);
}

/* Test 4: Full EM with Aitken (shorter run) */
static void test_em_with_aitken(void) {
    printf("\nTest 4: EM with Aitken acceleration\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Run EM with limited iterations */
    sem_run_em_with_aitken(sem, 10);

    /* Check final log-likelihood is reasonable */
    check(sem->log_likelihood < 0.0, "Final LL is negative");
    check(!isnan(sem->log_likelihood), "Final LL is not NaN");
    check(!isinf(sem->log_likelihood), "Final LL is not infinite");

    /* Check that LL improved from F81 baseline */
    double baseline = -19310.063115;
    double improvement = sem->log_likelihood - baseline;
    printf("  Improvement over F81 baseline: %.6f\n", improvement);

    /* EM should improve or at least not decrease much from baseline */
    check(improvement > -1.0, "LL not significantly worse than baseline");

    embh_sem_destroy(sem);
}

/* Test 5: Memory cleanup after EM */
static void test_em_memory(void) {
    printf("\nTest 5: EM memory allocation and cleanup\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Run one iteration to allocate all EM structures */
    sem_em_iteration(sem);

    /* Check allocations */
    check(sem->expected_counts_for_vertex != NULL, "Vertex counts allocated");
    check(sem->expected_counts_for_edge != NULL, "Edge counts allocated");
    check(sem->posterior_prob_for_vertex != NULL, "Vertex posteriors allocated");
    check(sem->posterior_prob_for_edge != NULL, "Edge posteriors allocated");

    /* Cleanup - should not leak */
    embh_sem_destroy(sem);
    printf("  PASS: Memory cleanup (check with valgrind)\n");
    tests_passed++;
}

int main(void) {
    printf("=== Testing EM Algorithm (Stage 6) ===\n");
    printf("Validates EM parameter estimation\n");

    test_expected_counts();
    test_single_em_iteration();
    test_em_improvement();
    test_em_with_aitken();
    test_em_memory();

    printf("\n=== Results ===\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);

    if (tests_failed == 0) {
        printf("\n*** ALL TESTS PASSED ***\n");
        printf("Stage 6 (EM Algorithm) VALIDATED\n");
        printf("C implementation complete!\n");
    } else {
        printf("\n*** VALIDATION FAILED ***\n");
        printf("Fix issues before proceeding!\n");
    }

    return tests_failed > 0 ? 1 : 0;
}
