#include "embh_types.h"
#include <stdlib.h>
#include <string.h>

/* Create a new SEM vertex */
SEM_vertex* sem_vertex_create(int id) {
    SEM_vertex* v = (SEM_vertex*)calloc(1, sizeof(SEM_vertex));
    if (!v) return NULL;

    v->id = id;
    v->global_id = id;
    v->name[0] = '\0';

    /* Tree structure */
    v->parent = v;  /* Point to self when no parent (like C++) */
    v->neighbors = (SEM_vertex**)calloc(MAX_NEIGHBORS, sizeof(SEM_vertex*));
    v->children = (SEM_vertex**)calloc(MAX_CHILDREN, sizeof(SEM_vertex*));

    if (!v->neighbors || !v->children) {
        free(v->neighbors);
        free(v->children);
        free(v);
        return NULL;
    }

    v->num_neighbors = 0;
    v->num_children = 0;
    v->in_degree = 0;
    v->out_degree = 0;
    v->degree = 0;

    /* Observation */
    v->observed = false;
    v->pattern_index = -1;

    /* DNA data */
    v->dna_compressed = NULL;
    v->dna_compressed_size = 0;

    /* Initialize matrices to identity */
    sem_vertex_initialize_matrices(v);

    /* Likelihood values */
    v->log_scaling_factors = 0.0;
    v->vertex_log_likelihood = 0.0;
    v->sum_of_edge_log_likelihoods = 0.0;

    /* Other */
    v->times_visited = 0;
    v->rate_category = 0;
    v->gc_content = 0;
    v->dna_cond_like_initialized = false;

    return v;
}

/* Free vertex memory */
void sem_vertex_destroy(SEM_vertex* v) {
    if (!v) return;
    free(v->neighbors);
    free(v->children);
    free(v->dna_compressed);
    free(v);
}

/* Add neighbor to vertex */
void sem_vertex_add_neighbor(SEM_vertex* v, SEM_vertex* neighbor) {
    if (!v || !neighbor || v->num_neighbors >= MAX_NEIGHBORS) return;

    /* Check if already a neighbor */
    for (int i = 0; i < v->num_neighbors; i++) {
        if (v->neighbors[i] == neighbor) return;
    }

    v->neighbors[v->num_neighbors++] = neighbor;
    v->degree++;
}

/* Remove neighbor from vertex */
void sem_vertex_remove_neighbor(SEM_vertex* v, SEM_vertex* neighbor) {
    if (!v || !neighbor) return;

    for (int i = 0; i < v->num_neighbors; i++) {
        if (v->neighbors[i] == neighbor) {
            /* Shift remaining neighbors */
            for (int j = i; j < v->num_neighbors - 1; j++) {
                v->neighbors[j] = v->neighbors[j + 1];
            }
            v->num_neighbors--;
            v->degree--;
            v->neighbors[v->num_neighbors] = NULL;
            return;
        }
    }
}

/* Add child to vertex */
void sem_vertex_add_child(SEM_vertex* v, SEM_vertex* child) {
    if (!v || !child || v->num_children >= MAX_CHILDREN) return;

    /* Check if already a child */
    for (int i = 0; i < v->num_children; i++) {
        if (v->children[i] == child) return;
    }

    v->children[v->num_children++] = child;
    v->out_degree++;
}

/* Remove child from vertex */
void sem_vertex_remove_child(SEM_vertex* v, SEM_vertex* child) {
    if (!v || !child) return;

    for (int i = 0; i < v->num_children; i++) {
        if (v->children[i] == child) {
            /* Shift remaining children */
            for (int j = i; j < v->num_children - 1; j++) {
                v->children[j] = v->children[j + 1];
            }
            v->num_children--;
            v->out_degree--;
            v->children[v->num_children] = NULL;
            return;
        }
    }
}

/* Set parent of vertex */
void sem_vertex_set_parent(SEM_vertex* v, SEM_vertex* parent) {
    if (!v) return;
    v->parent = parent ? parent : v;  /* If NULL, point to self */
    if (parent && parent != v) {
        v->in_degree = 1;
    }
}

/* Remove parent (set to self) */
void sem_vertex_remove_parent(SEM_vertex* v) {
    if (!v) return;
    v->parent = v;
    v->in_degree = 0;
}

/* Initialize matrices to identity and probabilities to uniform */
void sem_vertex_initialize_matrices(SEM_vertex* v) {
    if (!v) return;

    /* Initialize 4x4 transition matrix to identity (row-major) */
    memset(v->transition_matrix, 0, 16 * sizeof(double));
    memset(v->transition_matrix_stored, 0, 16 * sizeof(double));

    for (int i = 0; i < 4; i++) {
        v->transition_matrix[i * 4 + i] = 1.0;         /* Diagonal = 1 */
        v->transition_matrix_stored[i * 4 + i] = 1.0;
    }

    /* Initialize probability vectors to uniform */
    for (int i = 0; i < 4; i++) {
        v->root_probability[i] = 0.25;
        v->posterior_probability[i] = 0.25;
        v->root_prob_hss[i] = 0.25;
    }
}

/* Set vertex name */
void sem_vertex_set_name(SEM_vertex* v, const char* name) {
    if (!v || !name) return;
    strncpy(v->name, name, MAX_NAME_LEN - 1);
    v->name[MAX_NAME_LEN - 1] = '\0';
}
