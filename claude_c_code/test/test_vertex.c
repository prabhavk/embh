#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

int test_vertex_create(void) {
    printf("Test: sem_vertex_create and destroy\n");

    SEM_vertex* v = sem_vertex_create(42);
    if (!v) {
        printf("  FAIL: sem_vertex_create returned NULL\n");
        return 1;
    }

    if (v->id != 42) {
        printf("  FAIL: id is %d, expected 42\n", v->id);
        sem_vertex_destroy(v);
        return 1;
    }

    if (v->parent != v) {
        printf("  FAIL: parent should be self\n");
        sem_vertex_destroy(v);
        return 1;
    }

    if (v->num_neighbors != 0 || v->num_children != 0) {
        printf("  FAIL: should have no neighbors/children initially\n");
        sem_vertex_destroy(v);
        return 1;
    }

    if (v->observed != false || v->pattern_index != -1) {
        printf("  FAIL: observation status incorrect\n");
        sem_vertex_destroy(v);
        return 1;
    }

    /* Check identity matrix (row-major: M[i][j] = M[i*4+j]) */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            double actual = v->transition_matrix[i * 4 + j];
            if (fabs(actual - expected) > 1e-10) {
                printf("  FAIL: transition_matrix[%d][%d] = %f, expected %f\n",
                       i, j, actual, expected);
                sem_vertex_destroy(v);
                return 1;
            }
        }
    }

    /* Check uniform probabilities */
    for (int i = 0; i < 4; i++) {
        if (fabs(v->root_probability[i] - 0.25) > 1e-10) {
            printf("  FAIL: root_probability[%d] = %f, expected 0.25\n",
                   i, v->root_probability[i]);
            sem_vertex_destroy(v);
            return 1;
        }
    }

    sem_vertex_destroy(v);
    printf("  PASS\n");
    return 0;
}

int test_vertex_name(void) {
    printf("Test: sem_vertex_set_name\n");

    SEM_vertex* v = sem_vertex_create(1);
    if (!v) {
        printf("  FAIL: sem_vertex_create returned NULL\n");
        return 1;
    }

    sem_vertex_set_name(v, "h_0");
    if (strcmp(v->name, "h_0") != 0) {
        printf("  FAIL: name is '%s', expected 'h_0'\n", v->name);
        sem_vertex_destroy(v);
        return 1;
    }

    sem_vertex_set_name(v, "Arabidopsis_thaliana");
    if (strcmp(v->name, "Arabidopsis_thaliana") != 0) {
        printf("  FAIL: name is '%s', expected 'Arabidopsis_thaliana'\n", v->name);
        sem_vertex_destroy(v);
        return 1;
    }

    sem_vertex_destroy(v);
    printf("  PASS\n");
    return 0;
}

int test_vertex_tree_structure(void) {
    printf("Test: Tree structure (parent-child relationships)\n");

    /* Create 5 vertices: root with 2 children, each child has 1 grandchild */
    /*      root (0)
     *      /    \
     *   c1(1)  c2(2)
     *    |       |
     *  g1(3)   g2(4)
     */

    SEM_vertex* root = sem_vertex_create(0);
    SEM_vertex* c1 = sem_vertex_create(1);
    SEM_vertex* c2 = sem_vertex_create(2);
    SEM_vertex* g1 = sem_vertex_create(3);
    SEM_vertex* g2 = sem_vertex_create(4);

    if (!root || !c1 || !c2 || !g1 || !g2) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    sem_vertex_set_name(root, "root");
    sem_vertex_set_name(c1, "child1");
    sem_vertex_set_name(c2, "child2");
    sem_vertex_set_name(g1, "grandchild1");
    sem_vertex_set_name(g2, "grandchild2");

    /* Set up parent-child relationships */
    sem_vertex_set_parent(c1, root);
    sem_vertex_set_parent(c2, root);
    sem_vertex_set_parent(g1, c1);
    sem_vertex_set_parent(g2, c2);

    sem_vertex_add_child(root, c1);
    sem_vertex_add_child(root, c2);
    sem_vertex_add_child(c1, g1);
    sem_vertex_add_child(c2, g2);

    /* Verify relationships */
    if (root->parent != root) {
        printf("  FAIL: root parent should be self\n");
        goto cleanup;
    }

    if (c1->parent != root || c2->parent != root) {
        printf("  FAIL: c1 and c2 should have root as parent\n");
        goto cleanup;
    }

    if (g1->parent != c1) {
        printf("  FAIL: g1 parent should be c1\n");
        goto cleanup;
    }

    if (g2->parent != c2) {
        printf("  FAIL: g2 parent should be c2\n");
        goto cleanup;
    }

    if (root->num_children != 2) {
        printf("  FAIL: root should have 2 children, has %d\n", root->num_children);
        goto cleanup;
    }

    if (c1->num_children != 1 || c2->num_children != 1) {
        printf("  FAIL: c1 and c2 should each have 1 child\n");
        goto cleanup;
    }

    if (g1->num_children != 0 || g2->num_children != 0) {
        printf("  FAIL: grandchildren should have no children\n");
        goto cleanup;
    }

    /* Test neighbor relationships */
    sem_vertex_add_neighbor(root, c1);
    sem_vertex_add_neighbor(root, c2);
    sem_vertex_add_neighbor(c1, root);
    sem_vertex_add_neighbor(c1, g1);
    sem_vertex_add_neighbor(c2, root);
    sem_vertex_add_neighbor(c2, g2);
    sem_vertex_add_neighbor(g1, c1);
    sem_vertex_add_neighbor(g2, c2);

    if (root->num_neighbors != 2) {
        printf("  FAIL: root should have 2 neighbors, has %d\n", root->num_neighbors);
        goto cleanup;
    }

    if (c1->num_neighbors != 2) {
        printf("  FAIL: c1 should have 2 neighbors, has %d\n", c1->num_neighbors);
        goto cleanup;
    }

    /* Clean up */
    sem_vertex_destroy(root);
    sem_vertex_destroy(c1);
    sem_vertex_destroy(c2);
    sem_vertex_destroy(g1);
    sem_vertex_destroy(g2);
    printf("  PASS\n");
    return 0;

cleanup:
    sem_vertex_destroy(root);
    sem_vertex_destroy(c1);
    sem_vertex_destroy(c2);
    sem_vertex_destroy(g1);
    sem_vertex_destroy(g2);
    return 1;
}

int test_vertex_remove_operations(void) {
    printf("Test: Remove neighbor and child operations\n");

    SEM_vertex* v1 = sem_vertex_create(1);
    SEM_vertex* v2 = sem_vertex_create(2);
    SEM_vertex* v3 = sem_vertex_create(3);

    if (!v1 || !v2 || !v3) {
        printf("  FAIL: Failed to create vertices\n");
        return 1;
    }

    /* Add neighbors */
    sem_vertex_add_neighbor(v1, v2);
    sem_vertex_add_neighbor(v1, v3);

    if (v1->num_neighbors != 2) {
        printf("  FAIL: v1 should have 2 neighbors\n");
        goto cleanup;
    }

    /* Remove one neighbor */
    sem_vertex_remove_neighbor(v1, v2);
    if (v1->num_neighbors != 1) {
        printf("  FAIL: v1 should have 1 neighbor after removal\n");
        goto cleanup;
    }

    if (v1->neighbors[0] != v3) {
        printf("  FAIL: remaining neighbor should be v3\n");
        goto cleanup;
    }

    /* Add children */
    sem_vertex_add_child(v1, v2);
    sem_vertex_add_child(v1, v3);

    if (v1->num_children != 2) {
        printf("  FAIL: v1 should have 2 children\n");
        goto cleanup;
    }

    /* Remove one child */
    sem_vertex_remove_child(v1, v3);
    if (v1->num_children != 1) {
        printf("  FAIL: v1 should have 1 child after removal\n");
        goto cleanup;
    }

    if (v1->children[0] != v2) {
        printf("  FAIL: remaining child should be v2\n");
        goto cleanup;
    }

    sem_vertex_destroy(v1);
    sem_vertex_destroy(v2);
    sem_vertex_destroy(v3);
    printf("  PASS\n");
    return 0;

cleanup:
    sem_vertex_destroy(v1);
    sem_vertex_destroy(v2);
    sem_vertex_destroy(v3);
    return 1;
}

int test_vertex_matrices(void) {
    printf("Test: Matrix initialization and modification\n");

    SEM_vertex* v = sem_vertex_create(1);
    if (!v) {
        printf("  FAIL: sem_vertex_create returned NULL\n");
        return 1;
    }

    /* Modify transition matrix */
    /* Set to F81-like matrix (not exact, just for testing) */
    double pi[4] = {0.25, 0.25, 0.25, 0.25};
    double beta = 0.75;
    double exp_term = 0.5;  /* exp(-mu*t/beta) */

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                v->transition_matrix[i * 4 + j] = pi[j] + (1.0 - pi[j]) * exp_term;
            } else {
                v->transition_matrix[i * 4 + j] = pi[j] * (1.0 - exp_term);
            }
        }
    }

    /* Verify row sums = 1 */
    for (int i = 0; i < 4; i++) {
        double sum = 0.0;
        for (int j = 0; j < 4; j++) {
            sum += v->transition_matrix[i * 4 + j];
        }
        if (fabs(sum - 1.0) > 1e-10) {
            printf("  FAIL: row %d sum is %f, expected 1.0\n", i, sum);
            sem_vertex_destroy(v);
            return 1;
        }
    }

    sem_vertex_destroy(v);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== SEM_vertex Unit Tests ===\n\n");

    int failures = 0;

    failures += test_vertex_create();
    failures += test_vertex_name();
    failures += test_vertex_tree_structure();
    failures += test_vertex_remove_operations();
    failures += test_vertex_matrices();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
