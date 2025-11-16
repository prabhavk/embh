#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

int test_read_edges(void) {
    printf("Test: sem_set_edges_from_topology_file\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    sem_set_edges_from_topology_file(sem, "data/RAxML_bipartitions.CDS_FcC_partition.edgelist");

    /* Check number of edges (73 edges for 38 taxa = 74 vertices) */
    if (sem->num_edges != 73) {
        printf("  FAIL: num_edges is %d, expected 73\n", sem->num_edges);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check number of vertices (74 = 38 taxa + 36 internal) */
    if (sem->num_vertices != 74) {
        printf("  FAIL: num_vertices is %d, expected 74\n", sem->num_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check that specific vertices exist */
    if (!sem_contains_vertex(sem, "h_0")) {
        printf("  FAIL: vertex h_0 not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (!sem_contains_vertex(sem, "Nag")) {
        printf("  FAIL: vertex Nag not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (!sem_contains_vertex(sem, "Pod")) {
        printf("  FAIL: vertex Pod not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check that observed taxa are marked as observed */
    SEM_vertex* nag = sem_get_vertex_by_name(sem, "Nag");
    if (!nag || !nag->observed) {
        printf("  FAIL: Nag should be observed\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check that internal nodes are not observed */
    SEM_vertex* h0 = sem_get_vertex_by_name(sem, "h_0");
    if (!h0 || h0->observed) {
        printf("  FAIL: h_0 should not be observed\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check edge length was stored */
    if (sem->num_edge_lengths != 73) {
        printf("  FAIL: num_edge_lengths is %d, expected 73\n", sem->num_edge_lengths);
        embh_sem_destroy(sem);
        return 1;
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_root_tree(void) {
    printf("Test: sem_root_tree_at_vertex\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    sem_set_edges_from_topology_file(sem, "data/RAxML_bipartitions.CDS_FcC_partition.edgelist");

    /* Root at h_0 */
    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    if (!root) {
        printf("  FAIL: h_0 not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    sem_root_tree_at_vertex(sem, root);

    /* Check root is set */
    if (sem->root != root) {
        printf("  FAIL: root not set correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check root's parent is itself */
    if (root->parent != root) {
        printf("  FAIL: root's parent should be itself\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check all non-root vertices have a parent different from themselves */
    for (int i = 0; i < sem->num_vertices; i++) {
        SEM_vertex* v = sem->vertices[i];
        if (v != root) {
            if (!v->parent || v->parent == v) {
                printf("  FAIL: vertex %s has no parent\n", v->name);
                embh_sem_destroy(sem);
                return 1;
            }
        }
    }

    /* Set non-root vertices */
    sem_set_vertex_vector_except_root(sem);

    if (sem->num_non_root_vertices != 73) {
        printf("  FAIL: num_non_root_vertices is %d, expected 73\n", sem->num_non_root_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_read_patterns(void) {
    printf("Test: sem_read_patterns_from_file\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* First read edges to create vertices */
    sem_set_edges_from_topology_file(sem, "data/RAxML_bipartitions.CDS_FcC_partition.edgelist");

    /* Then read patterns */
    sem_read_patterns_from_file(sem, "data/patterns_1000.pat", "data/patterns_1000.taxon_order");

    /* Check number of patterns (648 unique patterns in 1000 sites) */
    if (sem->num_patterns != 648) {
        printf("  FAIL: num_patterns is %d, expected 648\n", sem->num_patterns);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check pattern weights sum to 1000 */
    int total_weight = 0;
    for (int i = 0; i < sem->num_patterns; i++) {
        total_weight += sem->pattern_weights[i];
    }
    if (total_weight != 1000) {
        printf("  FAIL: total pattern weight is %d, expected 1000\n", total_weight);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check first pattern (weight 109, all G) */
    if (sem->pattern_weights[0] != 109) {
        printf("  FAIL: first pattern weight is %d, expected 109\n", sem->pattern_weights[0]);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check first pattern is all G (base 2) */
    for (int t = 0; t < 38; t++) {
        uint8_t base = packed_storage_get_base(sem->packed_patterns, 0, t);
        if (base != DNA_G) {
            printf("  FAIL: first pattern should be all G, got base %d at taxon %d\n", base, t);
            embh_sem_destroy(sem);
            return 1;
        }
    }

    /* Check pattern indices are set for observed taxa */
    SEM_vertex* nag = sem_get_vertex_by_name(sem, "Nag");
    if (!nag || nag->pattern_index != 0) {
        printf("  FAIL: Nag should have pattern_index 0, got %d\n", nag ? nag->pattern_index : -1);
        embh_sem_destroy(sem);
        return 1;
    }

    SEM_vertex* pod = sem_get_vertex_by_name(sem, "Pod");
    if (!pod || pod->pattern_index != 37) {
        printf("  FAIL: Pod should have pattern_index 37, got %d\n", pod ? pod->pattern_index : -1);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check packed storage memory matches C++ baseline */
    size_t mem = packed_storage_get_memory_bytes(sem->packed_patterns);
    if (mem != 9234) {
        printf("  FAIL: packed storage memory is %zu bytes, expected 9234\n", mem);
        embh_sem_destroy(sem);
        return 1;
    }

    packed_storage_destroy(sem->packed_patterns);
    sem->packed_patterns = NULL;
    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_f81_model(void) {
    printf("Test: sem_set_f81_model\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* Set up tree */
    sem_set_edges_from_topology_file(sem, "data/RAxML_bipartitions.CDS_FcC_partition.edgelist");
    SEM_vertex* root = sem_get_vertex_by_name(sem, "h_0");
    sem_root_tree_at_vertex(sem, root);
    sem_set_vertex_vector_except_root(sem);

    /* Set F81 model */
    sem_set_f81_model(sem, "data/patterns_1000.basecomp");

    /* Check root probabilities match base composition file */
    double expected_pi[] = {0.2876752105, 0.1976084259, 0.2594303513, 0.2552860122};
    for (int i = 0; i < 4; i++) {
        if (fabs(sem->root_probability[i] - expected_pi[i]) > 1e-10) {
            printf("  FAIL: root_probability[%d] = %.10f, expected %.10f\n",
                   i, sem->root_probability[i], expected_pi[i]);
            embh_sem_destroy(sem);
            return 1;
        }
    }

    /* Check F81 mu value */
    double S2 = 0.0;
    for (int i = 0; i < 4; i++) {
        S2 += expected_pi[i] * expected_pi[i];
    }
    double expected_mu = 1.0 / (1.0 - S2);

    if (fabs(sem->F81_mu - expected_mu) > 1e-10) {
        printf("  FAIL: F81_mu = %.10f, expected %.10f\n", sem->F81_mu, expected_mu);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check transition matrix for Nag (first child of h_0) */
    SEM_vertex* nag = sem_get_vertex_by_name(sem, "Nag");
    if (!nag) {
        printf("  FAIL: Nag not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Get branch length for Nag */
    double t = sem_get_length_of_subtending_branch(sem, nag);
    if (fabs(t - 0.013721) > 1e-6) {
        printf("  FAIL: Nag branch length = %.6f, expected 0.013721\n", t);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check transition matrix is valid (rows sum to 1) */
    for (int i = 0; i < 4; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < 4; j++) {
            row_sum += nag->transition_matrix[i * 4 + j];
        }
        if (fabs(row_sum - 1.0) > 1e-10) {
            printf("  FAIL: Nag transition matrix row %d sums to %.10f, expected 1.0\n", i, row_sum);
            embh_sem_destroy(sem);
            return 1;
        }
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_full_setup(void) {
    printf("Test: Full tree setup (like C++ baseline)\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* Set up tree exactly like C++ */
    sem_set_edges_from_topology_file(sem, "data/RAxML_bipartitions.CDS_FcC_partition.edgelist");

    if (sem->num_vertices != 74) {
        printf("  FAIL: num_vertices = %d, expected 74\n", sem->num_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    sem_set_vertex_vector(sem);

    SEM_vertex* root_optim = sem_get_vertex_by_name(sem, "h_0");
    if (!root_optim) {
        printf("  FAIL: h_0 not found\n");
        embh_sem_destroy(sem);
        return 1;
    }

    sem_root_tree_at_vertex(sem, root_optim);
    sem_set_vertex_vector_except_root(sem);

    if (sem->num_non_root_vertices != 73) {
        printf("  FAIL: num_non_root_vertices = %d, expected 73\n", sem->num_non_root_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Read patterns */
    sem_read_patterns_from_file(sem, "data/patterns_1000.pat", "data/patterns_1000.taxon_order");

    if (sem->num_patterns != 648) {
        printf("  FAIL: num_patterns = %d, expected 648\n", sem->num_patterns);
        packed_storage_destroy(sem->packed_patterns);
        sem->packed_patterns = NULL;
        embh_sem_destroy(sem);
        return 1;
    }

    /* Set F81 model */
    sem_set_f81_model(sem, "data/patterns_1000.basecomp");

    printf("  Setup complete:\n");
    printf("    - %d vertices (38 taxa, 36 internal)\n", sem->num_vertices);
    printf("    - %d edges\n", sem->num_edges);
    printf("    - %d unique patterns\n", sem->num_patterns);
    printf("    - Root at %s\n", sem->root->name);
    printf("    - F81 mu = %.10f\n", sem->F81_mu);

    packed_storage_destroy(sem->packed_patterns);
    sem->packed_patterns = NULL;
    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== I/O Function Unit Tests ===\n\n");

    int failures = 0;

    failures += test_read_edges();
    failures += test_root_tree();
    failures += test_read_patterns();
    failures += test_f81_model();
    failures += test_full_setup();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
