#include "embh_types.h"
#include <stdlib.h>
#include <string.h>

/* Create main SEM structure */
SEM* embh_sem_create(int max_vertices, int max_patterns) {
    SEM* sem = (SEM*)calloc(1, sizeof(SEM));
    if (!sem) return NULL;

    sem->max_vertices = max_vertices;

    /* Allocate vertex array */
    sem->vertices = (SEM_vertex**)calloc(max_vertices, sizeof(SEM_vertex*));
    sem->vertex_names = (char**)calloc(max_vertices, sizeof(char*));
    sem->vertex_ids = (int*)calloc(max_vertices, sizeof(int));
    sem->leaves = (SEM_vertex**)calloc(max_vertices, sizeof(SEM_vertex*));
    sem->non_root_vertices = (SEM_vertex**)calloc(max_vertices, sizeof(SEM_vertex*));
    sem->post_order_vertices = (SEM_vertex**)calloc(max_vertices, sizeof(SEM_vertex*));

    if (!sem->vertices || !sem->vertex_names || !sem->vertex_ids ||
        !sem->leaves || !sem->non_root_vertices || !sem->post_order_vertices) {
        free(sem->vertices);
        free(sem->vertex_names);
        free(sem->vertex_ids);
        free(sem->leaves);
        free(sem->non_root_vertices);
        free(sem->post_order_vertices);
        free(sem);
        return NULL;
    }

    /* Allocate name strings */
    for (int i = 0; i < max_vertices; i++) {
        sem->vertex_names[i] = (char*)calloc(MAX_NAME_LEN, sizeof(char));
        if (!sem->vertex_names[i]) {
            for (int j = 0; j < i; j++) {
                free(sem->vertex_names[j]);
            }
            free(sem->vertices);
            free(sem->vertex_names);
            free(sem->vertex_ids);
            free(sem->leaves);
            free(sem->non_root_vertices);
            free(sem->post_order_vertices);
            free(sem);
            return NULL;
        }
    }

    sem->num_vertices = 0;
    sem->num_named_vertices = 0;
    sem->num_leaves = 0;
    sem->num_non_root_vertices = 0;
    sem->num_post_order = 0;

    /* Allocate pattern weights if needed */
    if (max_patterns > 0) {
        sem->pattern_weights = (int*)calloc(max_patterns, sizeof(int));
        if (!sem->pattern_weights) {
            for (int i = 0; i < max_vertices; i++) {
                free(sem->vertex_names[i]);
            }
            free(sem->vertices);
            free(sem->vertex_names);
            free(sem->vertex_ids);
            free(sem->leaves);
            free(sem->non_root_vertices);
            free(sem->post_order_vertices);
            free(sem);
            return NULL;
        }
    }
    sem->num_patterns = 0;

    /* Initialize root probability to uniform */
    for (int i = 0; i < 4; i++) {
        sem->root_probability[i] = 0.25;
        sem->root_probability_stored[i] = 0.25;
    }

    sem->packed_patterns = NULL;
    sem->root = NULL;
    sem->clique_tree = NULL;
    sem->expected_counts_for_vertex = NULL;
    sem->expected_counts_for_edge = NULL;
    sem->posterior_prob_for_vertex = NULL;
    sem->posterior_prob_for_edge = NULL;
    sem->F81_mu = 1.0;
    sem->log_likelihood = 0.0;
    sem->maximum_log_likelihood = 0.0;
    sem->verbose = false;
    sem->debug = false;
    sem->max_iter = 100;
    sem->num_observed_vertices = 0;
    sem->num_edges = 0;

    /* Allocate edge length storage (max_vertices - 1 edges in a tree) */
    sem->max_edge_lengths = max_vertices;
    sem->edge_u = (SEM_vertex**)calloc(sem->max_edge_lengths, sizeof(SEM_vertex*));
    sem->edge_v = (SEM_vertex**)calloc(sem->max_edge_lengths, sizeof(SEM_vertex*));
    sem->edge_lengths = (double*)calloc(sem->max_edge_lengths, sizeof(double));
    sem->num_edge_lengths = 0;

    if (!sem->edge_u || !sem->edge_v || !sem->edge_lengths) {
        free(sem->edge_u);
        free(sem->edge_v);
        free(sem->edge_lengths);
        /* Clean up all previously allocated memory */
        for (int i = 0; i < max_vertices; i++) {
            free(sem->vertex_names[i]);
        }
        free(sem->vertices);
        free(sem->vertex_names);
        free(sem->vertex_ids);
        free(sem->leaves);
        free(sem->non_root_vertices);
        free(sem->post_order_vertices);
        free(sem->pattern_weights);
        free(sem);
        return NULL;
    }

    /* Allocate post-order edge traversal (for pruning algorithm) */
    sem->max_post_order_edges = max_vertices;
    sem->post_order_parent = (SEM_vertex**)calloc(sem->max_post_order_edges, sizeof(SEM_vertex*));
    sem->post_order_child = (SEM_vertex**)calloc(sem->max_post_order_edges, sizeof(SEM_vertex*));
    sem->num_post_order_edges = 0;

    if (!sem->post_order_parent || !sem->post_order_child) {
        free(sem->post_order_parent);
        free(sem->post_order_child);
        free(sem->edge_u);
        free(sem->edge_v);
        free(sem->edge_lengths);
        for (int i = 0; i < max_vertices; i++) {
            free(sem->vertex_names[i]);
        }
        free(sem->vertices);
        free(sem->vertex_names);
        free(sem->vertex_ids);
        free(sem->leaves);
        free(sem->non_root_vertices);
        free(sem->post_order_vertices);
        free(sem->pattern_weights);
        free(sem);
        return NULL;
    }

    /* Initialize HSS matrices to NULL (allocated on demand) */
    sem->hss_matrices_forward = NULL;
    sem->hss_matrices_reverse = NULL;
    sem->hss_computed = false;

    return sem;
}

/* Free SEM structure (including vertices) */
void embh_sem_destroy(SEM* sem) {
    if (!sem) return;

    /* Free vertices */
    for (int i = 0; i < sem->num_vertices; i++) {
        sem_vertex_destroy(sem->vertices[i]);
    }

    /* Free name strings */
    for (int i = 0; i < sem->max_vertices; i++) {
        free(sem->vertex_names[i]);
    }

    /* Free expected counts */
    if (sem->expected_counts_for_vertex) {
        for (int i = 0; i < sem->num_vertices; i++) {
            free(sem->expected_counts_for_vertex[i]);
        }
        free(sem->expected_counts_for_vertex);
    }

    if (sem->expected_counts_for_edge) {
        for (int i = 0; i < sem->num_post_order_edges; i++) {
            free(sem->expected_counts_for_edge[i]);
        }
        free(sem->expected_counts_for_edge);
    }

    if (sem->posterior_prob_for_vertex) {
        for (int i = 0; i < sem->num_vertices; i++) {
            free(sem->posterior_prob_for_vertex[i]);
        }
        free(sem->posterior_prob_for_vertex);
    }

    if (sem->posterior_prob_for_edge) {
        for (int i = 0; i < sem->num_post_order_edges; i++) {
            free(sem->posterior_prob_for_edge[i]);
        }
        free(sem->posterior_prob_for_edge);
    }

    free(sem->vertices);
    free(sem->vertex_names);
    free(sem->vertex_ids);
    free(sem->leaves);
    free(sem->non_root_vertices);
    free(sem->post_order_vertices);
    free(sem->pattern_weights);

    /* Free edge length storage */
    free(sem->edge_u);
    free(sem->edge_v);
    free(sem->edge_lengths);

    /* Free post-order edge traversal */
    free(sem->post_order_parent);
    free(sem->post_order_child);

    /* Free HSS matrices if allocated
     * HSS matrices are indexed by vertex ID pairs: [p_id * max_vertices + c_id]
     */
    if (sem->hss_matrices_forward) {
        int total_size = sem->max_vertices * sem->max_vertices;
        for (int i = 0; i < total_size; i++) {
            if (sem->hss_matrices_forward[i]) {
                free(sem->hss_matrices_forward[i]);
            }
        }
        free(sem->hss_matrices_forward);
    }
    if (sem->hss_matrices_reverse) {
        int total_size = sem->max_vertices * sem->max_vertices;
        for (int i = 0; i < total_size; i++) {
            if (sem->hss_matrices_reverse[i]) {
                free(sem->hss_matrices_reverse[i]);
            }
        }
        free(sem->hss_matrices_reverse);
    }

    /* Free packed patterns if present */
    if (sem->packed_patterns) {
        packed_storage_destroy(sem->packed_patterns);
    }

    /* Free clique tree if present */
    if (sem->clique_tree) {
        for (int i = 0; i < sem->clique_tree->num_cliques; i++) {
            clique_destroy(sem->clique_tree->cliques[i]);
        }
        clique_tree_destroy(sem->clique_tree);
    }

    free(sem);
}

/* Add vertex to SEM, returns vertex ID or -1 on error */
int sem_add_vertex(SEM* sem, const char* name, bool observed) {
    if (!sem || !name || sem->num_vertices >= sem->max_vertices) return -1;

    int id = sem->num_vertices;
    SEM_vertex* v = sem_vertex_create(id);
    if (!v) return -1;

    sem_vertex_set_name(v, name);
    v->observed = observed;

    sem->vertices[id] = v;
    sem->num_vertices++;

    /* Add to name mapping */
    if (sem->num_named_vertices < sem->max_vertices) {
        strncpy(sem->vertex_names[sem->num_named_vertices], name, MAX_NAME_LEN - 1);
        sem->vertex_names[sem->num_named_vertices][MAX_NAME_LEN - 1] = '\0';
        sem->vertex_ids[sem->num_named_vertices] = id;
        sem->num_named_vertices++;
    }

    if (observed) {
        sem->num_observed_vertices++;
    }

    return id;
}

/* Get vertex by name using linear search */
SEM_vertex* sem_get_vertex_by_name(SEM* sem, const char* name) {
    if (!sem || !name) return NULL;

    for (int i = 0; i < sem->num_named_vertices; i++) {
        if (strcmp(sem->vertex_names[i], name) == 0) {
            int id = sem->vertex_ids[i];
            if (id >= 0 && id < sem->num_vertices) {
                return sem->vertices[id];
            }
        }
    }
    return NULL;
}

/* Get vertex by ID */
SEM_vertex* sem_get_vertex_by_id(SEM* sem, int id) {
    if (!sem || id < 0 || id >= sem->num_vertices) return NULL;
    return sem->vertices[id];
}

/* Check if vertex with name exists */
bool sem_contains_vertex(SEM* sem, const char* name) {
    return sem_get_vertex_by_name(sem, name) != NULL;
}

/* Set root vertex */
void sem_set_root(SEM* sem, SEM_vertex* root) {
    if (!sem) return;
    sem->root = root;
}

/* Add leaf vertex to leaves list */
void sem_add_leaf(SEM* sem, SEM_vertex* leaf) {
    if (!sem || !leaf || sem->num_leaves >= sem->max_vertices) return;
    sem->leaves[sem->num_leaves++] = leaf;
}

/* Add non-root vertex to list */
void sem_add_non_root_vertex(SEM* sem, SEM_vertex* v) {
    if (!sem || !v || sem->num_non_root_vertices >= sem->max_vertices) return;
    sem->non_root_vertices[sem->num_non_root_vertices++] = v;
}

/* Compute post-order traversal of tree */
void sem_compute_post_order(SEM* sem) {
    if (!sem || !sem->root) return;

    /* Use BFS with in-degree counting (same as clique tree) */
    int* in_degree = (int*)calloc(sem->num_vertices, sizeof(int));
    if (!in_degree) return;

    /* Count in-degrees (number of children) */
    for (int i = 0; i < sem->num_vertices; i++) {
        in_degree[i] = sem->vertices[i]->num_children;
    }

    /* Queue for BFS */
    SEM_vertex** queue = (SEM_vertex**)malloc(sem->num_vertices * sizeof(SEM_vertex*));
    if (!queue) {
        free(in_degree);
        return;
    }

    int front = 0, back = 0;
    sem->num_post_order = 0;

    /* Start with leaves (no children) */
    for (int i = 0; i < sem->num_vertices; i++) {
        if (in_degree[i] == 0) {
            queue[back++] = sem->vertices[i];
        }
    }

    /* Process vertices in post-order */
    while (front < back) {
        SEM_vertex* curr = queue[front++];
        sem->post_order_vertices[sem->num_post_order++] = curr;

        /* Decrease parent's in-degree */
        if (curr->parent && curr->parent != curr) {
            int parent_id = curr->parent->id;
            in_degree[parent_id]--;
            if (in_degree[parent_id] == 0) {
                queue[back++] = curr->parent;
            }
        }
    }

    free(in_degree);
    free(queue);
}
