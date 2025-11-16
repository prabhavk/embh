#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

int test_sem_create(void) {
    printf("Test: embh_sem_create and destroy\n");

    SEM* sem = embh_sem_create(100, 50);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    if (sem->max_vertices != 100) {
        printf("  FAIL: max_vertices is %d, expected 100\n", sem->max_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    if (sem->num_vertices != 0) {
        printf("  FAIL: num_vertices is %d, expected 0\n", sem->num_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check root probability is uniform */
    for (int i = 0; i < 4; i++) {
        if (fabs(sem->root_probability[i] - 0.25) > 1e-10) {
            printf("  FAIL: root_probability[%d] = %f, expected 0.25\n",
                   i, sem->root_probability[i]);
            embh_sem_destroy(sem);
            return 1;
        }
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_sem_add_vertices(void) {
    printf("Test: sem_add_vertex\n");

    SEM* sem = embh_sem_create(100, 50);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* Add some vertices */
    int id0 = sem_add_vertex(sem, "h_0", false);  /* Internal */
    int id1 = sem_add_vertex(sem, "h_1", false);
    int id2 = sem_add_vertex(sem, "Arabidopsis", true);  /* Leaf */
    int id3 = sem_add_vertex(sem, "Rice", true);
    int id4 = sem_add_vertex(sem, "Maize", true);

    if (id0 != 0 || id1 != 1 || id2 != 2 || id3 != 3 || id4 != 4) {
        printf("  FAIL: vertex IDs not assigned correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (sem->num_vertices != 5) {
        printf("  FAIL: num_vertices is %d, expected 5\n", sem->num_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    if (sem->num_observed_vertices != 3) {
        printf("  FAIL: num_observed_vertices is %d, expected 3\n",
               sem->num_observed_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Check vertices are retrievable */
    SEM_vertex* v0 = sem_get_vertex_by_id(sem, 0);
    if (!v0 || v0->id != 0 || strcmp(v0->name, "h_0") != 0) {
        printf("  FAIL: vertex 0 not retrieved correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    SEM_vertex* v2 = sem_get_vertex_by_id(sem, 2);
    if (!v2 || !v2->observed || strcmp(v2->name, "Arabidopsis") != 0) {
        printf("  FAIL: vertex 2 not retrieved correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_sem_name_lookup(void) {
    printf("Test: sem_get_vertex_by_name and sem_contains_vertex\n");

    SEM* sem = embh_sem_create(100, 50);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    sem_add_vertex(sem, "h_0", false);
    sem_add_vertex(sem, "h_1", false);
    sem_add_vertex(sem, "Arabidopsis", true);
    sem_add_vertex(sem, "Rice", true);

    /* Test containment */
    if (!sem_contains_vertex(sem, "h_0")) {
        printf("  FAIL: sem_contains_vertex returned false for 'h_0'\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (!sem_contains_vertex(sem, "Rice")) {
        printf("  FAIL: sem_contains_vertex returned false for 'Rice'\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (sem_contains_vertex(sem, "NotAVertex")) {
        printf("  FAIL: sem_contains_vertex returned true for non-existent vertex\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Test retrieval by name */
    SEM_vertex* v = sem_get_vertex_by_name(sem, "Arabidopsis");
    if (!v) {
        printf("  FAIL: sem_get_vertex_by_name returned NULL\n");
        embh_sem_destroy(sem);
        return 1;
    }

    if (strcmp(v->name, "Arabidopsis") != 0 || !v->observed) {
        printf("  FAIL: retrieved vertex has wrong attributes\n");
        embh_sem_destroy(sem);
        return 1;
    }

    SEM_vertex* vnull = sem_get_vertex_by_name(sem, "NonExistent");
    if (vnull != NULL) {
        printf("  FAIL: sem_get_vertex_by_name should return NULL for non-existent\n");
        embh_sem_destroy(sem);
        return 1;
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_sem_tree_structure(void) {
    printf("Test: SEM tree structure and post-order traversal\n");

    SEM* sem = embh_sem_create(100, 50);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* Create a tree:
     *       root (0)
     *      /    \
     *    A (1)   h1 (2)
     *           /   \
     *         B (3)  C (4)
     */
    int id_root = sem_add_vertex(sem, "root", false);
    int id_A = sem_add_vertex(sem, "A", true);
    int id_h1 = sem_add_vertex(sem, "h1", false);
    int id_B = sem_add_vertex(sem, "B", true);
    int id_C = sem_add_vertex(sem, "C", true);

    SEM_vertex* root = sem_get_vertex_by_id(sem, id_root);
    SEM_vertex* A = sem_get_vertex_by_id(sem, id_A);
    SEM_vertex* h1 = sem_get_vertex_by_id(sem, id_h1);
    SEM_vertex* B = sem_get_vertex_by_id(sem, id_B);
    SEM_vertex* C = sem_get_vertex_by_id(sem, id_C);

    /* Set up parent-child relationships */
    sem_vertex_set_parent(A, root);
    sem_vertex_set_parent(h1, root);
    sem_vertex_set_parent(B, h1);
    sem_vertex_set_parent(C, h1);

    sem_vertex_add_child(root, A);
    sem_vertex_add_child(root, h1);
    sem_vertex_add_child(h1, B);
    sem_vertex_add_child(h1, C);

    sem_set_root(sem, root);

    /* Add leaves */
    sem_add_leaf(sem, A);
    sem_add_leaf(sem, B);
    sem_add_leaf(sem, C);

    if (sem->num_leaves != 3) {
        printf("  FAIL: num_leaves is %d, expected 3\n", sem->num_leaves);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Compute post-order */
    sem_compute_post_order(sem);

    if (sem->num_post_order != 5) {
        printf("  FAIL: num_post_order is %d, expected 5\n", sem->num_post_order);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Root should be last in post-order */
    if (sem->post_order_vertices[sem->num_post_order - 1] != root) {
        printf("  FAIL: root should be last in post-order\n");
        embh_sem_destroy(sem);
        return 1;
    }

    /* Print post-order for verification */
    printf("  Post-order traversal:\n");
    for (int i = 0; i < sem->num_post_order; i++) {
        printf("    %d: %s\n", i, sem->post_order_vertices[i]->name);
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int test_sem_large_scale(void) {
    printf("Test: SEM with 74 vertices (like baseline)\n");

    SEM* sem = embh_sem_create(100, 700);
    if (!sem) {
        printf("  FAIL: embh_sem_create returned NULL\n");
        return 1;
    }

    /* Add 38 observed vertices (leaves) and 36 internal vertices */
    for (int i = 0; i < 36; i++) {
        char name[32];
        snprintf(name, sizeof(name), "h_%d", i);
        sem_add_vertex(sem, name, false);
    }

    for (int i = 0; i < 38; i++) {
        char name[32];
        snprintf(name, sizeof(name), "taxon_%d", i);
        sem_add_vertex(sem, name, true);
    }

    if (sem->num_vertices != 74) {
        printf("  FAIL: num_vertices is %d, expected 74\n", sem->num_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    if (sem->num_observed_vertices != 38) {
        printf("  FAIL: num_observed_vertices is %d, expected 38\n",
               sem->num_observed_vertices);
        embh_sem_destroy(sem);
        return 1;
    }

    /* Test retrieval */
    SEM_vertex* h10 = sem_get_vertex_by_name(sem, "h_10");
    if (!h10 || h10->observed) {
        printf("  FAIL: h_10 not retrieved correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    SEM_vertex* t20 = sem_get_vertex_by_name(sem, "taxon_20");
    if (!t20 || !t20->observed) {
        printf("  FAIL: taxon_20 not retrieved correctly\n");
        embh_sem_destroy(sem);
        return 1;
    }

    embh_sem_destroy(sem);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== SEM Unit Tests ===\n\n");

    int failures = 0;

    failures += test_sem_create();
    failures += test_sem_add_vertices();
    failures += test_sem_name_lookup();
    failures += test_sem_tree_structure();
    failures += test_sem_large_scale();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
