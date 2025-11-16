#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

int test_clique_create(void) {
    printf("Test: clique_create and destroy\n");

    SEM_vertex* x = sem_vertex_create(0);
    SEM_vertex* y = sem_vertex_create(1);

    if (!x || !y) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    sem_vertex_set_name(x, "h_0");
    sem_vertex_set_name(y, "Nag");

    Clique* c = clique_create(x, y, NULL);
    if (!c) {
        printf("  FAIL: clique_create returned NULL\n");
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    if (c->x != x || c->y != y) {
        printf("  FAIL: clique vertices incorrect\n");
        clique_destroy(c);
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    if (strcmp(c->name, "0-1") != 0) {
        printf("  FAIL: clique name is '%s', expected '0-1'\n", c->name);
        clique_destroy(c);
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    if (c->parent != c) {
        printf("  FAIL: clique parent should be self\n");
        clique_destroy(c);
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    clique_destroy(c);
    sem_vertex_destroy(x);
    sem_vertex_destroy(y);
    printf("  PASS\n");
    return 0;
}

int test_clique_tree_structure(void) {
    printf("Test: Clique tree structure for 4-taxa tree\n");

    /* Create a simple tree:
     *       h_0 (root)
     *      /   \
     *    A       h_1
     *           /   \
     *          B     C
     *
     * Edges (cliques):
     *   C0: h_0 -> A
     *   C1: h_0 -> h_1
     *   C2: h_1 -> B
     *   C3: h_1 -> C
     *
     * Clique tree structure:
     *   C1 is root
     *   C0 is child of C1 (shares h_0)
     *   C2 is child of C1 (shares h_1)
     *   C3 is child of C1 (shares h_1)
     */

    /* Create vertices */
    SEM_vertex* h0 = sem_vertex_create(0);
    SEM_vertex* h1 = sem_vertex_create(1);
    SEM_vertex* A = sem_vertex_create(2);
    SEM_vertex* B = sem_vertex_create(3);
    SEM_vertex* C = sem_vertex_create(4);

    if (!h0 || !h1 || !A || !B || !C) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    sem_vertex_set_name(h0, "h_0");
    sem_vertex_set_name(h1, "h_1");
    sem_vertex_set_name(A, "A");
    sem_vertex_set_name(B, "B");
    sem_vertex_set_name(C, "C_taxon");

    /* Mark leaves as observed */
    A->observed = true;
    B->observed = true;
    C->observed = true;

    /* Create cliques */
    Clique* c0 = clique_create(h0, A, NULL);
    Clique* c1 = clique_create(h0, h1, NULL);
    Clique* c2 = clique_create(h1, B, NULL);
    Clique* c3 = clique_create(h1, C, NULL);

    if (!c0 || !c1 || !c2 || !c3) {
        printf("  FAIL: Failed to create cliques\n");
        goto cleanup_vertices;
    }

    /* Build clique tree structure */
    clique_set_parent(c0, c1);
    clique_set_parent(c2, c1);
    clique_set_parent(c3, c1);

    clique_add_child(c1, c0);
    clique_add_child(c1, c2);
    clique_add_child(c1, c3);

    /* Create clique tree */
    CliqueTree* ct = clique_tree_create();
    if (!ct) {
        printf("  FAIL: clique_tree_create returned NULL\n");
        goto cleanup_cliques;
    }

    /* Add cliques */
    clique_tree_add_clique(ct, c0);
    clique_tree_add_clique(ct, c1);
    clique_tree_add_clique(ct, c2);
    clique_tree_add_clique(ct, c3);

    if (ct->num_cliques != 4) {
        printf("  FAIL: num_cliques is %d, expected 4\n", ct->num_cliques);
        goto cleanup_tree;
    }

    /* Check leaf cliques */
    if (ct->num_leaves != 3) {
        printf("  FAIL: num_leaves is %d, expected 3\n", ct->num_leaves);
        goto cleanup_tree;
    }

    /* Set root */
    clique_tree_set_root(ct, c1);
    if (ct->root != c1 || !ct->root_set) {
        printf("  FAIL: root not set correctly\n");
        goto cleanup_tree;
    }

    /* Compute traversal orders */
    clique_tree_compute_traversal_orders(ct);

    if (ct->traversal_size != 4) {
        printf("  FAIL: traversal_size is %d, expected 4\n", ct->traversal_size);
        goto cleanup_tree;
    }

    /* Post-order should be: c0, c2, c3, c1 (leaves first, then root) */
    /* Check that root is last in post-order */
    if (ct->post_order_traversal[ct->traversal_size - 1] != c1) {
        printf("  FAIL: root should be last in post-order\n");
        goto cleanup_tree;
    }

    /* Check that root is first in pre-order */
    if (ct->pre_order_traversal[0] != c1) {
        printf("  FAIL: root should be first in pre-order\n");
        goto cleanup_tree;
    }

    printf("  Post-order traversal:\n");
    for (int i = 0; i < ct->traversal_size; i++) {
        printf("    %d: %s\n", i, ct->post_order_traversal[i]->name);
    }

    clique_tree_destroy(ct);
    clique_destroy(c0);
    clique_destroy(c1);
    clique_destroy(c2);
    clique_destroy(c3);
    sem_vertex_destroy(h0);
    sem_vertex_destroy(h1);
    sem_vertex_destroy(A);
    sem_vertex_destroy(B);
    sem_vertex_destroy(C);
    printf("  PASS\n");
    return 0;

cleanup_tree:
    clique_tree_destroy(ct);
cleanup_cliques:
    clique_destroy(c0);
    clique_destroy(c1);
    clique_destroy(c2);
    clique_destroy(c3);
cleanup_vertices:
    sem_vertex_destroy(h0);
    sem_vertex_destroy(h1);
    sem_vertex_destroy(A);
    sem_vertex_destroy(B);
    sem_vertex_destroy(C);
    return 1;
}

int test_clique_initial_potential(void) {
    printf("Test: clique_set_initial_potential_and_belief\n");

    /* Create a simple clique with observed Y */
    SEM_vertex* x = sem_vertex_create(0);
    SEM_vertex* y = sem_vertex_create(1);

    if (!x || !y) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    y->observed = true;
    y->pattern_index = 0;

    /* Create packed storage with one pattern */
    PackedPatternStorage* patterns = packed_storage_create(1, 1);
    if (!patterns) {
        printf("  FAIL: Failed to create packed storage\n");
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    /* Set pattern: taxon 0 is 'G' (index 2) */
    packed_storage_set_base(patterns, 0, 0, DNA_G);

    /* Set transition matrix in Y */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            y->transition_matrix[i * 4 + j] = 0.25;  /* Uniform for test */
        }
    }

    Clique* c = clique_create(x, y, patterns);
    if (!c) {
        printf("  FAIL: clique_create returned NULL\n");
        packed_storage_destroy(patterns);
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    /* Initialize potential for site 0 */
    clique_set_initial_potential_and_belief(c, 0);

    /* Check that only column 2 (G) is non-zero */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = (j == DNA_G) ? 0.25 : 0.0;
            double actual = c->initial_potential[i * 4 + j];
            if (fabs(actual - expected) > 1e-10) {
                printf("  FAIL: initial_potential[%d][%d] = %f, expected %f\n",
                       i, j, actual, expected);
                clique_destroy(c);
                packed_storage_destroy(patterns);
                sem_vertex_destroy(x);
                sem_vertex_destroy(y);
                return 1;
            }
        }
    }

    clique_destroy(c);
    packed_storage_destroy(patterns);
    sem_vertex_destroy(x);
    sem_vertex_destroy(y);
    printf("  PASS\n");
    return 0;
}

int test_clique_marginalize(void) {
    printf("Test: clique_marginalize_over_variable\n");

    SEM_vertex* x = sem_vertex_create(0);
    SEM_vertex* y = sem_vertex_create(1);

    if (!x || !y) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    Clique* c = clique_create(x, y, NULL);
    if (!c) {
        printf("  FAIL: clique_create returned NULL\n");
        sem_vertex_destroy(x);
        sem_vertex_destroy(y);
        return 1;
    }

    /* Set belief to known values (row-major: belief[i*4+j] = P(X=i, Y=j)) */
    /* Set P(X,Y) such that:
     *   P(X=0) = 0.4, P(X=1) = 0.3, P(X=2) = 0.2, P(X=3) = 0.1
     *   P(Y=0) = 0.1, P(Y=1) = 0.2, P(Y=2) = 0.3, P(Y=3) = 0.4
     */
    double p_x[4] = {0.4, 0.3, 0.2, 0.1};
    double p_y[4] = {0.1, 0.2, 0.3, 0.4};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            c->belief[i * 4 + j] = p_x[i] * p_y[j];  /* Independent for simplicity */
        }
    }

    /* Marginalize over X to get P(Y) */
    double result_y[4];
    clique_marginalize_over_variable(c, x, result_y);

    for (int j = 0; j < 4; j++) {
        if (fabs(result_y[j] - p_y[j]) > 1e-10) {
            printf("  FAIL: P(Y=%d) = %f, expected %f\n", j, result_y[j], p_y[j]);
            clique_destroy(c);
            sem_vertex_destroy(x);
            sem_vertex_destroy(y);
            return 1;
        }
    }

    /* Marginalize over Y to get P(X) */
    double result_x[4];
    clique_marginalize_over_variable(c, y, result_x);

    for (int i = 0; i < 4; i++) {
        if (fabs(result_x[i] - p_x[i]) > 1e-10) {
            printf("  FAIL: P(X=%d) = %f, expected %f\n", i, result_x[i], p_x[i]);
            clique_destroy(c);
            sem_vertex_destroy(x);
            sem_vertex_destroy(y);
            return 1;
        }
    }

    clique_destroy(c);
    sem_vertex_destroy(x);
    sem_vertex_destroy(y);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== Clique and CliqueTree Unit Tests ===\n\n");

    int failures = 0;

    failures += test_clique_create();
    failures += test_clique_tree_structure();
    failures += test_clique_initial_potential();
    failures += test_clique_marginalize();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
