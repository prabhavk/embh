#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

#define MAX_LINE_LEN 4096

/* Add edge length to SEM storage */
void sem_add_edge_length(SEM* sem, SEM_vertex* u, SEM_vertex* v, double length) {
    if (!sem || !u || !v) return;
    if (sem->num_edge_lengths >= sem->max_edge_lengths) return;

    /* Store with smaller id first for consistency */
    if (u->id < v->id) {
        sem->edge_u[sem->num_edge_lengths] = u;
        sem->edge_v[sem->num_edge_lengths] = v;
    } else {
        sem->edge_u[sem->num_edge_lengths] = v;
        sem->edge_v[sem->num_edge_lengths] = u;
    }
    sem->edge_lengths[sem->num_edge_lengths] = length;
    sem->num_edge_lengths++;
}

/* Get length of branch from parent to vertex v */
double sem_get_length_of_subtending_branch(SEM* sem, SEM_vertex* v) {
    if (!sem || !v || !v->parent || v->parent == v) return 0.0;

    SEM_vertex* parent = v->parent;
    SEM_vertex* u1, *v1;

    /* Look for edge with smaller id first */
    if (parent->id < v->id) {
        u1 = parent;
        v1 = v;
    } else {
        u1 = v;
        v1 = parent;
    }

    /* Linear search for edge */
    for (int i = 0; i < sem->num_edge_lengths; i++) {
        if (sem->edge_u[i] == u1 && sem->edge_v[i] == v1) {
            return sem->edge_lengths[i];
        }
    }

    return 0.0;  /* Edge not found */
}

/* Read patterns from file */
void sem_read_patterns_from_file(SEM* sem, const char* pattern_filename,
                                  const char* taxon_order_filename) {
    if (!sem || !pattern_filename || !taxon_order_filename) return;

    FILE* taxon_file = fopen(taxon_order_filename, "r");
    if (!taxon_file) {
        fprintf(stderr, "Error: Cannot open taxon order file %s\n", taxon_order_filename);
        return;
    }

    /* Read taxon order file to get taxon names and their pattern indices */
    /* Format: taxon_name,position (CSV with header) */
    char line[MAX_LINE_LEN];
    char taxon_names[100][MAX_NAME_LEN];  /* Max 100 taxa */
    int num_taxa = 0;

    /* Skip header */
    if (fgets(line, sizeof(line), taxon_file) == NULL) {
        fclose(taxon_file);
        return;
    }

    /* Read taxa */
    while (fgets(line, sizeof(line), taxon_file) && num_taxa < 100) {
        char* comma = strchr(line, ',');
        if (comma) {
            int name_len = comma - line;
            if (name_len >= MAX_NAME_LEN) name_len = MAX_NAME_LEN - 1;
            strncpy(taxon_names[num_taxa], line, name_len);
            taxon_names[num_taxa][name_len] = '\0';
            num_taxa++;
        }
    }
    fclose(taxon_file);

    printf("Read %d taxa from taxon order file\n", num_taxa);

    /* Read pattern file to count patterns */
    FILE* pattern_file = fopen(pattern_filename, "r");
    if (!pattern_file) {
        fprintf(stderr, "Error: Cannot open pattern file %s\n", pattern_filename);
        return;
    }

    int num_patterns = 0;
    while (fgets(line, sizeof(line), pattern_file)) {
        num_patterns++;
    }
    rewind(pattern_file);

    printf("Found %d patterns in file\n", num_patterns);

    /* Create packed storage */
    sem->packed_patterns = packed_storage_create(num_patterns, num_taxa);
    if (!sem->packed_patterns) {
        fprintf(stderr, "Error: Failed to create packed storage\n");
        fclose(pattern_file);
        return;
    }

    /* Reallocate pattern weights if needed */
    if (sem->pattern_weights) {
        free(sem->pattern_weights);
    }
    sem->pattern_weights = (int*)calloc(num_patterns, sizeof(int));
    if (!sem->pattern_weights) {
        fprintf(stderr, "Error: Failed to allocate pattern weights\n");
        packed_storage_destroy(sem->packed_patterns);
        sem->packed_patterns = NULL;
        fclose(pattern_file);
        return;
    }

    /* Set pattern indices for observed vertices */
    for (int t = 0; t < num_taxa; t++) {
        SEM_vertex* v = sem_get_vertex_by_name(sem, taxon_names[t]);
        if (v && v->observed) {
            v->pattern_index = t;
        }
    }

    /* Read patterns */
    int pattern_idx = 0;
    while (fgets(line, sizeof(line), pattern_file) && pattern_idx < num_patterns) {
        int num_tokens;
        char** tokens = split_whitespace(line, &num_tokens);

        if (num_tokens >= num_taxa + 1) {
            /* First token is weight */
            int weight = atoi(tokens[0]);
            sem->pattern_weights[pattern_idx] = weight;

            /* Remaining tokens are bases (0-4) */
            for (int t = 0; t < num_taxa; t++) {
                int base = atoi(tokens[t + 1]);
                if (base < 0 || base > 4) base = DNA_GAP;
                packed_storage_set_base(sem->packed_patterns, pattern_idx, t, (uint8_t)base);
            }
        }

        free_string_array(tokens, num_tokens);
        pattern_idx++;
    }

    fclose(pattern_file);
    sem->num_patterns = num_patterns;

    printf("Number of unique site patterns is %d\n", num_patterns);
    printf("Packed storage memory: %zu bytes (%.2f%% savings)\n",
           packed_storage_get_memory_bytes(sem->packed_patterns),
           (1.0 - (double)packed_storage_get_memory_bytes(sem->packed_patterns) /
            (num_patterns * num_taxa)) * 100.0);
}

/* Set leaves from vertices that have pattern indices set */
void sem_set_leaves_from_pattern_indices(SEM* sem) {
    if (!sem) return;

    sem->num_leaves = 0;
    for (int i = 0; i < sem->num_vertices; i++) {
        SEM_vertex* v = sem->vertices[i];
        if (v->observed && v->pattern_index >= 0) {
            sem_add_leaf(sem, v);
        }
    }
}

/* Read edge list and build tree topology */
void sem_set_edges_from_topology_file(SEM* sem, const char* edge_list_filename) {
    if (!sem || !edge_list_filename) return;

    FILE* file = fopen(edge_list_filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open edge list file %s\n", edge_list_filename);
        return;
    }

    char line[MAX_LINE_LEN];
    int num_edges = 0;

    while (fgets(line, sizeof(line), file)) {
        int num_tokens;
        char** tokens = split_whitespace(line, &num_tokens);

        if (num_tokens >= 3) {
            const char* u_name = tokens[0];
            const char* v_name = tokens[1];
            double t = atof(tokens[2]);

            SEM_vertex* u;
            SEM_vertex* v;

            /* Get or create vertex u */
            if (sem_contains_vertex(sem, u_name)) {
                u = sem_get_vertex_by_name(sem, u_name);
            } else {
                /* Create new internal vertex */
                int id = sem_add_vertex(sem, u_name, false);
                u = sem_get_vertex_by_id(sem, id);
            }

            /* Get or create vertex v */
            if (sem_contains_vertex(sem, v_name)) {
                v = sem_get_vertex_by_name(sem, v_name);
            } else {
                /* Check if this is an observed taxon (leaf) */
                bool is_observed = !string_starts_with(v_name, "h_");
                int id = sem_add_vertex(sem, v_name, is_observed);
                v = sem_get_vertex_by_id(sem, id);
            }

            /* Add neighbor relationships (undirected edge) */
            sem_vertex_add_neighbor(u, v);
            sem_vertex_add_neighbor(v, u);

            /* Store edge length */
            sem_add_edge_length(sem, u, v, t);

            num_edges++;
        }

        free_string_array(tokens, num_tokens);
    }

    fclose(file);
    sem->num_edges = num_edges;
    printf("Number of edges in topology file is %d\n", num_edges);
}

/* Root tree at specified vertex using BFS to assign parent/child relationships */
void sem_root_tree_at_vertex(SEM* sem, SEM_vertex* root) {
    if (!sem || !root) return;

    /* Clear old parent/child relationships for ALL vertices
     * This is critical for re-rooting - we need a clean slate
     */
    for (int i = 0; i < sem->num_vertices; i++) {
        SEM_vertex* v = sem->vertices[i];
        v->parent = NULL;
        v->num_children = 0;
        v->out_degree = 0;
        v->in_degree = 0;
    }

    sem_set_root(sem, root);
    root->parent = root;  /* Root is its own parent */

    /* BFS to set parent/child relationships */
    SEM_vertex** queue = (SEM_vertex**)malloc(sem->num_vertices * sizeof(SEM_vertex*));
    bool* visited = (bool*)calloc(sem->num_vertices, sizeof(bool));
    if (!queue || !visited) {
        free(queue);
        free(visited);
        return;
    }

    int front = 0, back = 0;
    queue[back++] = root;
    visited[root->id] = true;

    while (front < back) {
        SEM_vertex* curr = queue[front++];

        /* Process all neighbors */
        for (int i = 0; i < curr->num_neighbors; i++) {
            SEM_vertex* neighbor = curr->neighbors[i];
            if (!visited[neighbor->id]) {
                visited[neighbor->id] = true;

                /* Set parent/child relationship */
                sem_vertex_set_parent(neighbor, curr);
                sem_vertex_add_child(curr, neighbor);

                queue[back++] = neighbor;
            }
        }
    }

    free(queue);
    free(visited);
}

/* Set vertex vector (all vertices) */
void sem_set_vertex_vector(SEM* sem) {
    /* Already have vertices array, just ensure it's consistent */
    if (!sem) return;
    /* Nothing extra needed - sem->vertices already contains all vertices */
}

/* Set non-root vertices vector */
void sem_set_vertex_vector_except_root(SEM* sem) {
    if (!sem || !sem->root) return;

    sem->num_non_root_vertices = 0;
    for (int i = 0; i < sem->num_vertices; i++) {
        if (sem->vertices[i] != sem->root) {
            sem_add_non_root_vertex(sem, sem->vertices[i]);
        }
    }
    printf("Number of non-root vertices is %d\n", sem->num_non_root_vertices);
}

/* Set F81 model from base composition file */
void sem_set_f81_model(SEM* sem, const char* base_comp_filename) {
    if (!sem || !base_comp_filename) return;

    FILE* file = fopen(base_comp_filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open base composition file %s\n", base_comp_filename);
        return;
    }

    char line[MAX_LINE_LEN];

    /* Read base frequencies */
    while (fgets(line, sizeof(line), file)) {
        int num_tokens;
        char** tokens = split_whitespace(line, &num_tokens);

        if (num_tokens >= 2) {
            /* Check if first token is a base index (0, 1, 2, or 3) */
            if (strcmp(tokens[0], "0") == 0 || strcmp(tokens[0], "1") == 0 ||
                strcmp(tokens[0], "2") == 0 || strcmp(tokens[0], "3") == 0) {
                int base_idx = atoi(tokens[0]);
                double freq = atof(tokens[1]);
                sem->root_probability[base_idx] = freq;
                sem->root_probability_stored[base_idx] = freq;
            }
        }

        free_string_array(tokens, num_tokens);
    }

    fclose(file);

    printf("Root probability (pi): [%.6f, %.6f, %.6f, %.6f]\n",
           sem->root_probability[0], sem->root_probability[1],
           sem->root_probability[2], sem->root_probability[3]);

    /* Compute F81 mu */
    sem_set_f81_mu(sem);

    /* Set F81 transition matrix for all non-root vertices */
    for (int i = 0; i < sem->num_non_root_vertices; i++) {
        sem_set_f81_matrix(sem, sem->non_root_vertices[i]);
    }
}

/* Compute F81 mu parameter */
void sem_set_f81_mu(SEM* sem) {
    if (!sem) return;

    double S2 = 0.0;
    for (int k = 0; k < 4; k++) {
        S2 += sem->root_probability[k] * sem->root_probability[k];
    }

    /* mu chosen so that expected rate = 1 */
    double denom = (1.0 - S2);
    if (denom < 1e-14) denom = 1e-14;  /* Guard against division by zero */

    sem->F81_mu = 1.0 / denom;
    printf("F81 mu is %f\n", sem->F81_mu);
}

/* Set F81 transition matrix for vertex v */
void sem_set_f81_matrix(SEM* sem, SEM_vertex* v) {
    if (!sem || !v) return;

    double t = sem_get_length_of_subtending_branch(sem, v);
    double* pi = sem->root_probability;
    double mu = sem->F81_mu;

    /* F81 transition matrix:
     * P_ij(t) = e^(-mu*t) * delta_ij + (1 - e^(-mu*t)) * pi_j
     */
    double exp_term = exp(-mu * t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            v->transition_matrix[i * 4 + j] = exp_term * delta_ij + (1.0 - exp_term) * pi[j];
        }
    }

    /* Store a backup copy */
    matrix_copy_4x4(v->transition_matrix, v->transition_matrix_stored);
}
