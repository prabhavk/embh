#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/embh_types.h"

/*
 * Test for belief propagation algorithm (Junction Tree Algorithm)
 * Critical validation: must match pruning algorithm log-likelihood
 * Both should give the same result for the same model and data
 *
 * Expected baseline: -19310.063115 (RAxML-NG value)
 * Data: 38 taxa, 648 patterns, 73 edges, rooted at h_0
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

static void check_double(double expected, double actual, double tol, const char* test_name) {
    double diff = fabs(expected - actual);
    if (diff < tol) {
        printf("  PASS: %s (expected=%.11f, got=%.11f, diff=%.2e)\n",
               test_name, expected, actual, diff);
        tests_passed++;
    } else {
        printf("  FAIL: %s (expected=%.11f, got=%.11f, diff=%.2e)\n",
               test_name, expected, actual, diff);
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

    /* First read tree topology */
    sem_set_edges_from_topology_file(sem, edge_file);

    /* Then read patterns */
    sem_read_patterns_from_file(sem, pattern_file, taxon_order_file);

    /* Root tree at h_0 */
    SEM_vertex* root_vertex = sem_get_vertex_by_name(sem, "h_0");
    if (root_vertex) {
        sem_root_tree_at_vertex(sem, root_vertex);
    }

    /* Set leaves from pattern indices */
    sem_set_leaves_from_pattern_indices(sem);

    /* Set non-root vertices list */
    sem_set_vertex_vector_except_root(sem);

    /* Set up F81 model */
    sem_set_f81_model(sem, base_comp_file);

    /* Set transition matrices for all edges */
    printf("  Setting F81 matrices for %d non-root vertices\n", sem->num_non_root_vertices);
    int matrices_set = 0;
    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        SEM_vertex* v = sem->non_root_vertices[i];
        if (v) {
            sem_set_f81_matrix(sem, v);
            matrices_set++;
        }
    }
    printf("  Set %d matrices\n", matrices_set);

    return sem;
}

/* Test 1: Clique tree construction */
static void test_clique_tree_construction(void) {
    printf("\nTest 1: Clique tree construction\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Construct clique tree */
    sem_construct_clique_tree(sem);

    check(sem->clique_tree != NULL, "Clique tree created");
    if (!sem->clique_tree) {
        embh_sem_destroy(sem);
        return;
    }

    /* Should have same number of cliques as edges */
    printf("  Number of cliques: %d\n", sem->clique_tree->num_cliques);
    printf("  Number of post-order edges: %d\n", sem->num_post_order_edges);
    check(sem->clique_tree->num_cliques == sem->num_post_order_edges,
          "Number of cliques equals number of edges");

    /* Should have a root */
    check(sem->clique_tree->root != NULL, "Clique tree has root");

    /* Root clique should have X = root vertex */
    if (sem->clique_tree->root) {
        check(sem->clique_tree->root->x == sem->root,
              "Root clique X is root vertex");
        printf("  Root clique: %s (X=%s, Y=%s)\n",
               sem->clique_tree->root->name,
               sem->clique_tree->root->x->name,
               sem->clique_tree->root->y->name);
    }

    /* Check leaves */
    printf("  Number of leaf cliques: %d\n", sem->clique_tree->num_leaves);
    check(sem->clique_tree->num_leaves > 0, "Has leaf cliques");

    /* Check traversal orders */
    printf("  Traversal size: %d\n", sem->clique_tree->traversal_size);
    check(sem->clique_tree->traversal_size == sem->clique_tree->num_cliques,
          "Traversal includes all cliques");

    embh_sem_destroy(sem);
}

/* Test 2: Basic propagation algorithm test */
static void test_propagation_basic(void) {
    printf("\nTest 2: Basic belief propagation algorithm test\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Verify setup */
    printf("  Number of patterns: %d\n", sem->num_patterns);
    printf("  Number of vertices: %d\n", sem->num_vertices);
    printf("  Root: %s (id=%d)\n", sem->root->name, sem->root->id);

    /* Compute log-likelihood using propagation */
    printf("  Computing log-likelihood using belief propagation...\n");
    sem_compute_log_likelihood_using_propagation(sem);

    printf("  Computed log-likelihood: %.11f\n", sem->log_likelihood);
    check(sem->log_likelihood < 0.0, "Log-likelihood is negative");
    check(!isnan(sem->log_likelihood), "Log-likelihood is not NaN");
    check(!isinf(sem->log_likelihood), "Log-likelihood is not infinite");

    embh_sem_destroy(sem);
}

/* Test 3: Critical validation against pruning algorithm */
static void test_propagation_vs_pruning(void) {
    printf("\nTest 3: CRITICAL - Validate propagation against pruning\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Compute using pruning algorithm */
    sem_compute_log_likelihood_using_patterns(sem);
    double ll_pruning = sem->log_likelihood;
    printf("  Pruning algorithm:     %.11f\n", ll_pruning);

    /* Compute using propagation algorithm */
    sem_compute_log_likelihood_using_propagation(sem);
    double ll_propagation = sem->log_likelihood;
    printf("  Propagation algorithm: %.11f\n", ll_propagation);

    printf("  Difference:            %.2e\n", fabs(ll_pruning - ll_propagation));

    /* This is the CRITICAL test - both should give same result */
    check_double(ll_pruning, ll_propagation, TOLERANCE,
                 "Propagation matches pruning within tolerance");

    embh_sem_destroy(sem);
}

/* Test 4: Validate against baseline */
static void test_propagation_baseline_validation(void) {
    printf("\nTest 4: Validate propagation against baseline\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Compute log-likelihood using propagation */
    sem_compute_log_likelihood_using_propagation(sem);

    double baseline = -19310.063115;
    printf("  RAxML-NG baseline:     %.11f\n", baseline);
    printf("  Propagation result:    %.11f\n", sem->log_likelihood);
    printf("  Difference:            %.2e\n", fabs(baseline - sem->log_likelihood));

    check_double(baseline, sem->log_likelihood, TOLERANCE,
                 "Propagation matches baseline within tolerance");

    embh_sem_destroy(sem);
}

/* Test 5: Internal consistency */
static void test_propagation_consistency(void) {
    printf("\nTest 5: Internal consistency checks\n");

    SEM* sem = full_setup();
    check(sem != NULL, "Full setup successful");
    if (!sem) return;

    /* Compute once */
    sem_compute_log_likelihood_using_propagation(sem);
    double ll1 = sem->log_likelihood;

    /* Compute again - should get same result */
    sem_compute_log_likelihood_using_propagation(sem);
    double ll2 = sem->log_likelihood;

    check_double(ll1, ll2, 1e-15, "Repeated computation gives same result");

    embh_sem_destroy(sem);
}

int main(void) {
    printf("=== Testing Belief Propagation Algorithm (Stage 5) ===\n");
    printf("Expected to match pruning algorithm: -19310.063115\n");
    printf("Tolerance: < 1e-3 difference\n");

    test_clique_tree_construction();
    test_propagation_basic();
    test_propagation_vs_pruning();
    test_propagation_baseline_validation();
    test_propagation_consistency();

    printf("\n=== Results ===\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);

    if (tests_failed == 0) {
        printf("\n*** ALL TESTS PASSED ***\n");
        printf("Stage 5 (Belief Propagation) VALIDATED\n");
        printf("You may proceed to Stage 6 (EM Algorithm)\n");
    } else {
        printf("\n*** VALIDATION FAILED ***\n");
        printf("DO NOT proceed to Stage 6 until all tests pass!\n");
    }

    return tests_failed > 0 ? 1 : 0;
}
