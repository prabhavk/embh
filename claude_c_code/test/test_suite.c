#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../include/embh_types.h"
#include "../include/embh_manager.h"

/*
 * Comprehensive Test Suite for EMBH C Implementation
 * Validates all stages of the conversion from C++ to C
 */

#define TOLERANCE 1e-6
#define STRICT_TOLERANCE 1e-10

/* Test result tracking */
typedef struct {
    int total_tests;
    int passed;
    int failed;
    int warnings;
} TestResults;

static TestResults results = {0, 0, 0, 0};

/* Test helper macros */
#define TEST_SECTION(name) printf("\n========== %s ==========\n", name)
#define TEST_PASS(msg) do { results.passed++; results.total_tests++; printf("  PASS: %s\n", msg); } while(0)
#define TEST_FAIL(msg) do { results.failed++; results.total_tests++; printf("  FAIL: %s\n", msg); } while(0)
#define TEST_WARN(msg) do { results.warnings++; printf("  WARN: %s\n", msg); } while(0)

#define ASSERT_TRUE(cond, msg) do { if (cond) TEST_PASS(msg); else TEST_FAIL(msg); } while(0)
#define ASSERT_FALSE(cond, msg) do { if (!(cond)) TEST_PASS(msg); else TEST_FAIL(msg); } while(0)
#define ASSERT_EQUAL_INT(a, b, msg) do { if ((a) == (b)) TEST_PASS(msg); else { TEST_FAIL(msg); printf("    Expected: %d, Got: %d\n", (b), (a)); } } while(0)
#define ASSERT_EQUAL_DOUBLE(a, b, tol, msg) do { if (fabs((a) - (b)) < (tol)) TEST_PASS(msg); else { TEST_FAIL(msg); printf("    Expected: %.11f, Got: %.11f, Diff: %.2e\n", (b), (a), fabs((a)-(b))); } } while(0)
#define ASSERT_NOT_NULL(ptr, msg) do { if ((ptr) != NULL) TEST_PASS(msg); else TEST_FAIL(msg); } while(0)

/* Test data paths */
static const char* EDGE_FILE = "data/RAxML_bipartitions.CDS_FcC_partition.edgelist";
static const char* PATTERN_FILE = "data/patterns_1000.pat";
static const char* TAXON_ORDER_FILE = "data/patterns_1000.taxon_order";
static const char* BASE_COMP_FILE = "data/patterns_1000.basecomp";

/* Expected baseline values */
static const double EXPECTED_LL_F81 = -19310.063315;
static const int EXPECTED_NUM_VERTICES = 74;
static const int EXPECTED_NUM_PATTERNS = 648;
static const int EXPECTED_NUM_LEAVES = 38;
static const int EXPECTED_NUM_EDGES = 73;
static const double EXPECTED_ROOT_PROB[] = {0.287675, 0.197608, 0.259430, 0.255286};

/*
 * Stage 1: Core Data Structures
 */
static void test_pattern_storage(void) {
    TEST_SECTION("Stage 1.1: PackedPatternStorage");

    /* Create storage */
    PackedPatternStorage* storage = packed_storage_create(100, 10);
    ASSERT_NOT_NULL(storage, "Create packed storage");

    /* Test set/get operations */
    packed_storage_set_base(storage, 0, 0, DNA_A);
    packed_storage_set_base(storage, 0, 1, DNA_C);
    packed_storage_set_base(storage, 0, 2, DNA_G);
    packed_storage_set_base(storage, 0, 3, DNA_T);
    packed_storage_set_base(storage, 0, 4, DNA_GAP);

    ASSERT_EQUAL_INT(packed_storage_get_base(storage, 0, 0), DNA_A, "Get base A");
    ASSERT_EQUAL_INT(packed_storage_get_base(storage, 0, 1), DNA_C, "Get base C");
    ASSERT_EQUAL_INT(packed_storage_get_base(storage, 0, 2), DNA_G, "Get base G");
    ASSERT_EQUAL_INT(packed_storage_get_base(storage, 0, 3), DNA_T, "Get base T");
    ASSERT_EQUAL_INT(packed_storage_get_base(storage, 0, 4), DNA_GAP, "Get base GAP");

    /* Test all patterns/taxa */
    bool all_correct = true;
    for (int pat = 0; pat < 100; pat++) {
        for (int tax = 0; tax < 10; tax++) {
            uint8_t base = (pat + tax) % 5;
            packed_storage_set_base(storage, pat, tax, base);
        }
    }
    for (int pat = 0; pat < 100; pat++) {
        for (int tax = 0; tax < 10; tax++) {
            uint8_t expected = (pat + tax) % 5;
            if (packed_storage_get_base(storage, pat, tax) != expected) {
                all_correct = false;
            }
        }
    }
    ASSERT_TRUE(all_correct, "Store and retrieve 1000 base values");

    /* Memory efficiency */
    size_t memory = packed_storage_get_memory_bytes(storage);
    size_t naive = 100 * 10;  /* 1 byte per base */
    ASSERT_TRUE(memory < naive, "Memory efficient storage");
    printf("    Memory: %zu bytes (%.1f%% of naive)\n", memory, 100.0 * memory / naive);

    packed_storage_destroy(storage);
}

static void test_vertex_operations(void) {
    TEST_SECTION("Stage 1.2: SEM_vertex");

    SEM_vertex* v1 = sem_vertex_create(0);
    SEM_vertex* v2 = sem_vertex_create(1);
    SEM_vertex* v3 = sem_vertex_create(2);

    ASSERT_NOT_NULL(v1, "Create vertex 1");
    ASSERT_NOT_NULL(v2, "Create vertex 2");
    ASSERT_NOT_NULL(v3, "Create vertex 3");

    /* Test neighbor operations */
    sem_vertex_add_neighbor(v1, v2);
    sem_vertex_add_neighbor(v1, v3);
    ASSERT_EQUAL_INT(v1->num_neighbors, 2, "Add 2 neighbors");

    sem_vertex_remove_neighbor(v1, v2);
    ASSERT_EQUAL_INT(v1->num_neighbors, 1, "Remove 1 neighbor");

    /* Test parent-child relationships */
    sem_vertex_set_parent(v2, v1);
    sem_vertex_add_child(v1, v2);
    ASSERT_TRUE(v2->parent == v1, "Set parent correctly");
    ASSERT_EQUAL_INT(v1->num_children, 1, "Add child");

    /* Test name */
    sem_vertex_set_name(v1, "test_vertex");
    ASSERT_TRUE(strcmp(v1->name, "test_vertex") == 0, "Set vertex name");

    sem_vertex_destroy(v1);
    sem_vertex_destroy(v2);
    sem_vertex_destroy(v3);
}

static void test_sem_structure(void) {
    TEST_SECTION("Stage 1.4: Main SEM Structure");

    SEM* sem = embh_sem_create(100, 500);
    ASSERT_NOT_NULL(sem, "Create SEM");

    /* Add vertices */
    int id1 = sem_add_vertex(sem, "taxon_1", true);
    int id2 = sem_add_vertex(sem, "hidden_1", false);
    ASSERT_TRUE(id1 >= 0, "Add observed vertex");
    ASSERT_TRUE(id2 >= 0, "Add hidden vertex");

    /* Retrieve vertices */
    SEM_vertex* v1 = sem_get_vertex_by_name(sem, "taxon_1");
    SEM_vertex* v2 = sem_get_vertex_by_id(sem, id2);
    ASSERT_NOT_NULL(v1, "Get vertex by name");
    ASSERT_NOT_NULL(v2, "Get vertex by id");
    ASSERT_TRUE(v1->observed, "Vertex observed flag");
    ASSERT_FALSE(v2->observed, "Hidden vertex flag");

    /* Check contains */
    ASSERT_TRUE(sem_contains_vertex(sem, "taxon_1"), "Contains existing vertex");
    ASSERT_FALSE(sem_contains_vertex(sem, "nonexistent"), "Does not contain missing vertex");

    embh_sem_destroy(sem);
}

/*
 * Stage 2: Utility Functions
 */
static void test_utilities(void) {
    TEST_SECTION("Stage 2: Utility Functions");

    /* Matrix operations */
    double M[16], I[16], R[16];
    matrix_identity_4x4(I);
    ASSERT_EQUAL_DOUBLE(I[0], 1.0, 1e-15, "Identity matrix [0,0]");
    ASSERT_EQUAL_DOUBLE(I[5], 1.0, 1e-15, "Identity matrix [1,1]");
    ASSERT_EQUAL_DOUBLE(I[1], 0.0, 1e-15, "Identity matrix [0,1]");

    /* Matrix copy */
    for (int i = 0; i < 16; i++) M[i] = i * 0.1;
    matrix_copy_4x4(M, R);
    bool copy_ok = true;
    for (int i = 0; i < 16; i++) {
        if (fabs(M[i] - R[i]) > 1e-15) copy_ok = false;
    }
    ASSERT_TRUE(copy_ok, "Matrix copy");

    /* DNA conversion */
    ASSERT_EQUAL_INT(convert_dna_to_index('A'), 0, "DNA A to index");
    ASSERT_EQUAL_INT(convert_dna_to_index('C'), 1, "DNA C to index");
    ASSERT_EQUAL_INT(convert_dna_to_index('G'), 2, "DNA G to index");
    ASSERT_EQUAL_INT(convert_dna_to_index('T'), 3, "DNA T to index");
    ASSERT_EQUAL_INT(convert_dna_to_index('-'), 4, "DNA gap to index");

    ASSERT_TRUE(convert_index_to_dna(0) == 'A', "Index 0 to DNA");
    ASSERT_TRUE(convert_index_to_dna(1) == 'C', "Index 1 to DNA");
    ASSERT_TRUE(convert_index_to_dna(2) == 'G', "Index 2 to DNA");
    ASSERT_TRUE(convert_index_to_dna(3) == 'T', "Index 3 to DNA");
    ASSERT_TRUE(convert_index_to_dna(4) == '-', "Index 4 to DNA");

    /* String utilities */
    ASSERT_TRUE(string_starts_with("hello_world", "hello"), "String starts with prefix");
    ASSERT_FALSE(string_starts_with("hello_world", "world"), "String does not start with suffix");
}

/*
 * Stage 3: File I/O
 */
static void test_file_io(void) {
    TEST_SECTION("Stage 3: File I/O");

    SEM* sem = embh_sem_create(200, 1000);
    ASSERT_NOT_NULL(sem, "Create SEM for I/O");

    /* Load topology */
    sem_set_edges_from_topology_file(sem, EDGE_FILE);
    ASSERT_EQUAL_INT(sem->num_vertices, EXPECTED_NUM_VERTICES, "Load correct number of vertices");

    /* Load patterns */
    sem_read_patterns_from_file(sem, PATTERN_FILE, TAXON_ORDER_FILE);
    ASSERT_EQUAL_INT(sem->num_patterns, EXPECTED_NUM_PATTERNS, "Load correct number of patterns");
    ASSERT_NOT_NULL(sem->packed_patterns, "Packed patterns allocated");

    /* Root tree */
    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    ASSERT_NOT_NULL(root, "Find root vertex");
    sem_root_tree_at_vertex(sem, root);
    ASSERT_TRUE(sem->root == root, "Tree rooted correctly");

    /* Set leaves */
    sem_set_leaves_from_pattern_indices(sem);
    ASSERT_EQUAL_INT(sem->num_leaves, EXPECTED_NUM_LEAVES, "Correct number of leaves");

    /* Set non-root vertices */
    sem_set_vertex_vector_except_root(sem);
    ASSERT_EQUAL_INT(sem->num_non_root_vertices, EXPECTED_NUM_EDGES, "Correct number of non-root vertices");

    /* Load F81 model */
    sem_set_f81_model(sem, BASE_COMP_FILE);
    for (int i = 0; i < 4; i++) {
        char msg[100];
        snprintf(msg, sizeof(msg), "Root probability[%d] matches baseline", i);
        ASSERT_EQUAL_DOUBLE(sem->root_probability[i], EXPECTED_ROOT_PROB[i], 1e-6, msg);
    }

    embh_sem_destroy(sem);
}

/*
 * Stage 4: Pruning Algorithm
 */
static void test_pruning_algorithm(void) {
    TEST_SECTION("Stage 4: Pruning Algorithm");

    SEM* sem = embh_sem_create(200, 1000);
    sem_set_edges_from_topology_file(sem, EDGE_FILE);
    sem_read_patterns_from_file(sem, PATTERN_FILE, TAXON_ORDER_FILE);

    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    sem_root_tree_at_vertex(sem, root);
    sem_set_leaves_from_pattern_indices(sem);
    sem_set_vertex_vector_except_root(sem);
    sem_set_f81_model(sem, BASE_COMP_FILE);

    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        sem_set_f81_matrix(sem, sem->non_root_vertices[i]);
    }

    /* Compute LL using pruning */
    clock_t start = clock();
    sem_compute_log_likelihood_using_patterns(sem);
    clock_t end = clock();
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    ASSERT_TRUE(sem->log_likelihood < 0.0, "Log-likelihood is negative");
    ASSERT_TRUE(!isnan(sem->log_likelihood), "Log-likelihood is not NaN");
    ASSERT_TRUE(!isinf(sem->log_likelihood), "Log-likelihood is not infinite");

    ASSERT_EQUAL_DOUBLE(sem->log_likelihood, EXPECTED_LL_F81, 1e-3,
                        "Log-likelihood matches C++ baseline (1e-3 tolerance)");

    printf("    Computed LL: %.11f in %.2f ms\n", sem->log_likelihood, time_ms);

    embh_sem_destroy(sem);
}

/*
 * Stage 5: Belief Propagation
 */
static void test_belief_propagation(void) {
    TEST_SECTION("Stage 5: Belief Propagation");

    SEM* sem = embh_sem_create(200, 1000);
    sem_set_edges_from_topology_file(sem, EDGE_FILE);
    sem_read_patterns_from_file(sem, PATTERN_FILE, TAXON_ORDER_FILE);

    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    sem_root_tree_at_vertex(sem, root);
    sem_set_leaves_from_pattern_indices(sem);
    sem_set_vertex_vector_except_root(sem);
    sem_set_f81_model(sem, BASE_COMP_FILE);

    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        sem_set_f81_matrix(sem, sem->non_root_vertices[i]);
    }

    /* Compute using pruning first */
    sem_compute_log_likelihood_using_patterns(sem);
    double ll_pruning = sem->log_likelihood;

    /* Compute using propagation */
    clock_t start = clock();
    sem_compute_log_likelihood_using_propagation(sem);
    clock_t end = clock();
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    double ll_prop = sem->log_likelihood;

    ASSERT_NOT_NULL(sem->clique_tree, "Clique tree constructed");
    ASSERT_EQUAL_INT(sem->clique_tree->num_cliques, EXPECTED_NUM_EDGES, "Correct number of cliques");

    /* Critical: must match pruning exactly */
    double diff = fabs(ll_pruning - ll_prop);
    ASSERT_EQUAL_DOUBLE(ll_prop, ll_pruning, 1e-10, "Propagation matches pruning (1e-10)");

    printf("    Pruning LL: %.11f\n", ll_pruning);
    printf("    Propagation LL: %.11f\n", ll_prop);
    printf("    Difference: %.2e (computed in %.2f ms)\n", diff, time_ms);

    embh_sem_destroy(sem);
}

/*
 * Stage 6: EM Algorithm
 */
static void test_em_algorithm(void) {
    TEST_SECTION("Stage 6: EM Algorithm");

    SEM* sem = embh_sem_create(200, 1000);
    sem_set_edges_from_topology_file(sem, EDGE_FILE);
    sem_read_patterns_from_file(sem, PATTERN_FILE, TAXON_ORDER_FILE);

    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    sem_root_tree_at_vertex(sem, root);
    sem_set_leaves_from_pattern_indices(sem);
    sem_set_vertex_vector_except_root(sem);
    sem_set_f81_model(sem, BASE_COMP_FILE);

    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        sem_set_f81_matrix(sem, sem->non_root_vertices[i]);
    }

    double ll_initial = EXPECTED_LL_F81;

    /* Run EM */
    clock_t start = clock();
    sem_run_em_with_aitken(sem, 100);
    clock_t end = clock();
    double time_sec = (double)(end - start) / CLOCKS_PER_SEC;

    double ll_final = sem->log_likelihood;
    double improvement = ll_final - ll_initial;

    ASSERT_TRUE(ll_final > ll_initial, "EM improves log-likelihood");
    ASSERT_TRUE(improvement > 1000.0, "EM improvement > 1000");
    ASSERT_TRUE(ll_final < 0.0, "Final LL is negative");
    ASSERT_TRUE(!isnan(ll_final), "Final LL is not NaN");

    /* Check root probabilities sum to 1 */
    double sum = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += sem->root_probability[i];
    }
    ASSERT_EQUAL_DOUBLE(sum, 1.0, 1e-10, "Root probabilities sum to 1");

    printf("    Initial LL: %.6f\n", ll_initial);
    printf("    Final LL: %.6f\n", ll_final);
    printf("    Improvement: %.2f\n", improvement);
    printf("    Time: %.2f seconds\n", time_sec);

    embh_sem_destroy(sem);
}

/*
 * Stage 7: Full Pipeline
 */
static void test_full_pipeline(void) {
    TEST_SECTION("Stage 7: Full Pipeline (Manager)");

    Manager* mgr = manager_create(
        EDGE_FILE,
        PATTERN_FILE,
        TAXON_ORDER_FILE,
        BASE_COMP_FILE,
        "h_0",
        "h_5"
    );

    ASSERT_NOT_NULL(mgr, "Create manager");
    ASSERT_NOT_NULL(mgr->P, "Manager has SEM instance");

    /* Check results */
    ASSERT_TRUE(mgr->max_log_likelihood > -18100.0, "EM converged to good solution");
    ASSERT_TRUE(fabs(mgr->max_log_likelihood - mgr->max_log_likelihood_hss) > 100.0,
                "Different roots have different likelihoods");

    printf("    Max LL at h_0: %.6f\n", mgr->max_log_likelihood);
    printf("    LL at h_5: %.6f\n", mgr->max_log_likelihood_hss);

    manager_destroy(mgr);
}

/*
 * Performance benchmarking
 */
static void test_performance(void) {
    TEST_SECTION("Performance Benchmarking");

    SEM* sem = embh_sem_create(200, 1000);
    sem_set_edges_from_topology_file(sem, EDGE_FILE);
    sem_read_patterns_from_file(sem, PATTERN_FILE, TAXON_ORDER_FILE);

    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    sem_root_tree_at_vertex(sem, root);
    sem_set_leaves_from_pattern_indices(sem);
    sem_set_vertex_vector_except_root(sem);
    sem_set_f81_model(sem, BASE_COMP_FILE);

    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        sem_set_f81_matrix(sem, sem->non_root_vertices[i]);
    }

    /* Benchmark pruning */
    int num_runs = 10;
    clock_t start = clock();
    for (int i = 0; i < num_runs; i++) {
        sem_compute_log_likelihood_using_patterns(sem);
    }
    clock_t end = clock();
    double pruning_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC / num_runs;

    /* Benchmark propagation */
    start = clock();
    for (int i = 0; i < num_runs; i++) {
        sem_compute_log_likelihood_using_propagation(sem);
    }
    end = clock();
    double prop_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC / num_runs;

    printf("    Pruning algorithm: %.2f ms/evaluation (avg of %d runs)\n", pruning_ms, num_runs);
    printf("    Propagation algorithm: %.2f ms/evaluation (avg of %d runs)\n", prop_ms, num_runs);
    printf("    Patterns: %d, Taxa: %d, Vertices: %d\n",
           sem->num_patterns, sem->num_leaves, sem->num_vertices);

    ASSERT_TRUE(pruning_ms < 1000.0, "Pruning completes in < 1 second");
    ASSERT_TRUE(prop_ms < 1000.0, "Propagation completes in < 1 second");

    embh_sem_destroy(sem);
}

/*
 * Main test runner
 */
int main(void) {
    printf("============================================\n");
    printf("EMBH C Implementation - Comprehensive Tests\n");
    printf("============================================\n");

    clock_t total_start = clock();

    /* Run all stage tests */
    test_pattern_storage();
    test_vertex_operations();
    test_sem_structure();
    test_utilities();
    test_file_io();
    test_pruning_algorithm();
    test_belief_propagation();
    test_em_algorithm();
    test_full_pipeline();
    test_performance();

    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    /* Summary */
    printf("\n============================================\n");
    printf("TEST SUMMARY\n");
    printf("============================================\n");
    printf("Total tests: %d\n", results.total_tests);
    printf("Passed: %d\n", results.passed);
    printf("Failed: %d\n", results.failed);
    printf("Warnings: %d\n", results.warnings);
    printf("Total time: %.2f seconds\n", total_time);
    printf("============================================\n");

    if (results.failed == 0) {
        printf("\n*** ALL TESTS PASSED ***\n");
        printf("EMBH C Implementation FULLY VALIDATED\n");
        return 0;
    } else {
        printf("\n*** %d TEST(S) FAILED ***\n", results.failed);
        return 1;
    }
}
