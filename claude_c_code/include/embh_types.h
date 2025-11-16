#ifndef EMBH_TYPES_H
#define EMBH_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* DNA base encoding: A=0, C=1, G=2, T=3, gap=4 */
#define DNA_A 0
#define DNA_C 1
#define DNA_G 2
#define DNA_T 3
#define DNA_GAP 4
#define NUM_BASES 4

/* Pattern structure - single pattern with weight */
typedef struct {
    int weight;
    uint8_t* characters;  /* Array of size num_taxa */
    int num_taxa;
} Pattern;

/* Packed pattern storage - 3-bit encoding for memory efficiency */
typedef struct {
    uint8_t* packed_data;  /* Bit-packed array */
    int num_patterns;
    int num_taxa;
    size_t data_size;      /* Size of packed_data in bytes */
} PackedPatternStorage;

/* Function declarations for pattern operations */
Pattern* pattern_create(int weight, const uint8_t* characters, int num_taxa);
void pattern_destroy(Pattern* p);

PackedPatternStorage* packed_storage_create(int num_patterns, int num_taxa);
void packed_storage_destroy(PackedPatternStorage* storage);
uint8_t packed_storage_get_base(const PackedPatternStorage* storage, int pattern_idx, int taxon_idx);
void packed_storage_set_base(PackedPatternStorage* storage, int pattern_idx, int taxon_idx, uint8_t base);
void packed_storage_store_pattern(PackedPatternStorage* storage, int pattern_idx, const uint8_t* pattern, int pattern_size);
void packed_storage_get_pattern(const PackedPatternStorage* storage, int pattern_idx, uint8_t* pattern_out);
size_t packed_storage_get_memory_bytes(const PackedPatternStorage* storage);
int packed_storage_get_num_patterns(const PackedPatternStorage* storage);
int packed_storage_get_num_taxa(const PackedPatternStorage* storage);

/* Maximum neighbors/children per vertex */
#define MAX_NEIGHBORS 10
#define MAX_CHILDREN 10
#define MAX_NAME_LEN 256

/* Forward declarations */
struct SEM_vertex;
typedef struct SEM_vertex SEM_vertex;

/* SEM Vertex structure - node in phylogenetic tree */
struct SEM_vertex {
    int id;
    int global_id;
    char name[MAX_NAME_LEN];

    /* Tree structure */
    SEM_vertex* parent;
    SEM_vertex** neighbors;
    SEM_vertex** children;
    int num_neighbors;
    int num_children;
    int in_degree;
    int out_degree;
    int degree;

    /* Observation status */
    bool observed;
    int pattern_index;  /* -1 if not observed leaf */

    /* DNA data (may not be needed with packed storage) */
    int* dna_compressed;
    int dna_compressed_size;

    /* Matrices for likelihood computation - 4x4 stored row-major */
    double transition_matrix[16];         /* P(Y|X) */
    double transition_matrix_stored[16];  /* Backup copy */
    double root_probability[4];           /* Prior at root */
    double posterior_probability[4];      /* Posterior */
    double root_prob_hss[4];              /* HSS root probability */

    /* Likelihood values */
    double log_scaling_factors;
    double vertex_log_likelihood;
    double sum_of_edge_log_likelihoods;

    /* Other attributes */
    int times_visited;
    int rate_category;
    int gc_content;
    bool dna_cond_like_initialized;
};

/* SEM_vertex function declarations */
SEM_vertex* sem_vertex_create(int id);
void sem_vertex_destroy(SEM_vertex* v);
void sem_vertex_add_neighbor(SEM_vertex* v, SEM_vertex* neighbor);
void sem_vertex_remove_neighbor(SEM_vertex* v, SEM_vertex* neighbor);
void sem_vertex_add_child(SEM_vertex* v, SEM_vertex* child);
void sem_vertex_remove_child(SEM_vertex* v, SEM_vertex* child);
void sem_vertex_set_parent(SEM_vertex* v, SEM_vertex* parent);
void sem_vertex_remove_parent(SEM_vertex* v);
void sem_vertex_initialize_matrices(SEM_vertex* v);
void sem_vertex_set_name(SEM_vertex* v, const char* name);

/* Forward declarations for Clique */
struct Clique;
typedef struct Clique Clique;

/* Clique structure - edge in phylogenetic tree = clique in junction tree */
struct Clique {
    int id;
    char name[MAX_NAME_LEN];
    int site;  /* Current pattern/site index */

    /* The two vertices defining this clique (edge X->Y) */
    SEM_vertex* x;  /* Parent vertex */
    SEM_vertex* y;  /* Child vertex */

    /* Tree structure in clique tree */
    Clique* parent;
    Clique** children;
    int num_children;
    int in_degree;
    int out_degree;
    int times_visited;

    /* Matrices for belief propagation - 4x4 stored row-major */
    double initial_potential[16];  /* P(X,Y) with evidence */
    double base_potential[16];     /* Transition matrix P(Y|X) */
    double factor[16];             /* Working matrix */
    double belief[16];             /* Final belief after calibration */

    /* Scaling factors */
    double scaling_factor;
    double log_scaling_factor_for_clique;

    /* Messages from neighbors (max neighbors = 1 parent + MAX_CHILDREN children) */
    double** messages_from_neighbors;     /* [neighbor_idx][4] */
    double* log_scaling_factors_for_messages;  /* [neighbor_idx] */
    Clique** neighbor_cliques;            /* Map from neighbor index to clique pointer */
    int num_neighbors;

    /* Root variable of the SEM tree (for root clique handling) */
    SEM_vertex* root_variable;

    /* Leaf status */
    bool is_leaf_clique;

    /* Edge index in post_order_edges array (-1 if not set) */
    int post_order_edge_index;

    /* Reference to packed patterns */
    PackedPatternStorage* packed_patterns;

    /* Memoization support */
    SEM_vertex** subtree_leaves;      /* Observed variables in this clique's subtree */
    int num_subtree_leaves;
    SEM_vertex** complement_leaves;   /* Observed variables NOT in subtree */
    int num_complement_leaves;

    /* Message cache: stores computed messages indexed by pattern signature hash
     * For upward messages: signature = pattern values at subtree leaves
     * For downward messages: signature = pattern values at complement leaves
     */
    int* upward_cache_signatures;     /* Array of signature hashes */
    double** upward_cache_messages;   /* [cache_idx][4] cached messages */
    double* upward_cache_scales;      /* Cached log scaling factors */
    int upward_cache_size;
    int upward_cache_capacity;

    int* downward_cache_signatures;
    double** downward_cache_messages;
    double* downward_cache_scales;
    int downward_cache_size;
    int downward_cache_capacity;

    /* Cache statistics */
    int upward_cache_hits;
    int upward_cache_misses;
    int downward_cache_hits;
    int downward_cache_misses;
};

/* CliqueTree structure - junction tree for belief propagation */
typedef struct {
    Clique** cliques;
    int num_cliques;
    Clique* root;
    bool root_set;

    /* Leaf cliques (those with observed variables) */
    Clique** leaves;
    int num_leaves;

    /* Traversal orders - arrays of clique pointers */
    Clique** post_order_traversal;  /* Leaves to root */
    Clique** pre_order_traversal;   /* Root to leaves */
    int traversal_size;

    /* Current site */
    int site;

    /* All observed leaves (for complement computation) */
    SEM_vertex** all_observed_leaves;
    int num_all_observed_leaves;

    /* Memoization statistics */
    int cache_hits;
    int cache_misses;
    int downward_cache_hits;
    int downward_cache_misses;
} CliqueTree;

/* Clique function declarations */
Clique* clique_create(SEM_vertex* x, SEM_vertex* y, PackedPatternStorage* patterns);
void clique_destroy(Clique* c);
void clique_add_child(Clique* c, Clique* child);
void clique_set_parent(Clique* c, Clique* parent);
void clique_set_initial_potential_and_belief(Clique* c, int site);
void clique_set_base_potential(Clique* c);
void clique_apply_evidence_for_site(Clique* c, int site);
void clique_compute_belief(Clique* c);
void clique_marginalize_over_variable(const Clique* c, const SEM_vertex* v, double* result);
void clique_compute_subtree_leaves(Clique* c);
int clique_get_subtree_signature(const Clique* c, int site);
int clique_get_complement_signature(const Clique* c, int site);
void clique_clear_caches(Clique* c);

/* CliqueTree function declarations */
CliqueTree* clique_tree_create(void);
void clique_tree_destroy(CliqueTree* ct);
void clique_tree_add_clique(CliqueTree* ct, Clique* c);
void clique_tree_set_root(CliqueTree* ct, Clique* root);
void clique_tree_compute_traversal_orders(CliqueTree* ct);
void clique_tree_calibrate(CliqueTree* ct);
void clique_tree_set_site(CliqueTree* ct, int site);
void clique_tree_initialize_potentials_and_beliefs(CliqueTree* ct);
void clique_tree_set_base_potentials(CliqueTree* ct);
void clique_tree_apply_evidence_and_reset(CliqueTree* ct, int site);
void clique_tree_compute_all_subtree_leaves(CliqueTree* ct);
void clique_tree_compute_all_complement_leaves(CliqueTree* ct);
void clique_tree_calibrate_with_memoization(CliqueTree* ct);
void clique_tree_clear_all_caches(CliqueTree* ct);
void clique_tree_reset_cache_statistics(CliqueTree* ct);

/* Main SEM (Structural EM) structure - manages entire phylogenetic model */
typedef struct {
    /* Vertices */
    SEM_vertex** vertices;
    int num_vertices;
    int max_vertices;

    /* Name to ID mapping (simple linear search for now) */
    char** vertex_names;
    int* vertex_ids;
    int num_named_vertices;

    /* Patterns */
    PackedPatternStorage* packed_patterns;
    int* pattern_weights;
    int num_patterns;

    /* Tree structure */
    SEM_vertex* root;
    SEM_vertex** leaves;
    int num_leaves;
    SEM_vertex** non_root_vertices;
    int num_non_root_vertices;

    /* Edge traversals */
    SEM_vertex** post_order_vertices;
    int num_post_order;

    /* Clique tree */
    CliqueTree* clique_tree;

    /* F81 model parameters */
    double root_probability[4];       /* pi */
    double root_probability_stored[4];
    double F81_mu;

    /* Log-likelihood */
    double log_likelihood;
    double maximum_log_likelihood;

    /* Counts for EM */
    double** expected_counts_for_vertex;  /* [vertex_id][4] */
    double** expected_counts_for_edge;    /* [edge_idx][16] - 4x4 row-major */
    double** posterior_prob_for_vertex;   /* [vertex_id][4] */
    double** posterior_prob_for_edge;     /* [edge_idx][16] */

    /* Settings */
    bool verbose;
    bool debug;
    int max_iter;

    /* Statistics */
    int num_observed_vertices;
    int num_edges;

    /* Edge lengths - simple array storage */
    SEM_vertex** edge_u;  /* Parent of edge */
    SEM_vertex** edge_v;  /* Child of edge */
    double* edge_lengths;
    int num_edge_lengths;
    int max_edge_lengths;

    /* Post-order edge traversal for pruning algorithm */
    SEM_vertex** post_order_parent;  /* Parent of each edge */
    SEM_vertex** post_order_child;   /* Child of each edge */
    int num_post_order_edges;
    int max_post_order_edges;

    /* HSS (root-invariant) transition matrices for reparameterization
     * hss_matrices_forward[i] = P(child|parent) for edge i
     * hss_matrices_reverse[i] = P(parent|child) for edge i (Bayes rule)
     * Both stored as 16 doubles (4x4 row-major)
     */
    double** hss_matrices_forward;  /* [edge_idx][16] */
    double** hss_matrices_reverse;  /* [edge_idx][16] */
    bool hss_computed;
} SEM;

/* SEM function declarations */
SEM* embh_sem_create(int max_vertices, int max_patterns);
void embh_sem_destroy(SEM* sem);
int sem_add_vertex(SEM* sem, const char* name, bool observed);
SEM_vertex* sem_get_vertex_by_name(SEM* sem, const char* name);
SEM_vertex* sem_get_vertex_by_id(SEM* sem, int id);
bool sem_contains_vertex(SEM* sem, const char* name);
void sem_set_root(SEM* sem, SEM_vertex* root);
void sem_add_leaf(SEM* sem, SEM_vertex* leaf);
void sem_add_non_root_vertex(SEM* sem, SEM_vertex* v);
void sem_compute_post_order(SEM* sem);

/* Utility functions - Matrix operations (4x4 row-major) */
void matrix_transpose_4x4(const double* src, double* dst);
void matrix_multiply_4x4(const double* A, const double* B, double* R);
void matrix_identity_4x4(double* M);
void matrix_copy_4x4(const double* src, double* dst);

/* Utility functions - String operations */
bool string_starts_with(const char* str, const char* prefix);
char** split_whitespace(const char* str, int* num_tokens);
void free_string_array(char** arr, int num_tokens);

/* Utility functions - DNA conversion */
int convert_dna_to_index(char dna);
char convert_index_to_dna(int index);
double gap_proportion_in_pattern(const int* pattern, int length);
int unique_non_gap_count_in_pattern(const int* pattern, int length);

/* I/O functions - File reading */
void sem_read_patterns_from_file(SEM* sem, const char* pattern_filename,
                                  const char* taxon_order_filename);
void sem_set_leaves_from_pattern_indices(SEM* sem);
void sem_set_edges_from_topology_file(SEM* sem, const char* edge_list_filename);
void sem_root_tree_at_vertex(SEM* sem, SEM_vertex* root);
void sem_set_vertex_vector(SEM* sem);
void sem_set_vertex_vector_except_root(SEM* sem);
void sem_set_f81_model(SEM* sem, const char* base_comp_filename);
void sem_set_f81_matrix(SEM* sem, SEM_vertex* v);
void sem_set_f81_mu(SEM* sem);
double sem_get_length_of_subtending_branch(SEM* sem, SEM_vertex* v);
void sem_add_edge_length(SEM* sem, SEM_vertex* u, SEM_vertex* v, double length);

/* Pruning algorithm functions */
void sem_compute_edges_for_post_order_traversal(SEM* sem);
void sem_compute_log_likelihood_using_patterns(SEM* sem);
void sem_reset_log_scaling_factors(SEM* sem);

/* Belief propagation functions */
void sem_construct_clique_tree(SEM* sem);
void sem_compute_log_likelihood_using_propagation(SEM* sem);
void sem_compute_log_likelihood_with_memoization(SEM* sem);
void clique_tree_send_message(CliqueTree* ct, Clique* from, Clique* to);
void clique_tree_send_message_with_memoization(CliqueTree* ct, Clique* from, Clique* to);
int clique_get_neighbor_index(Clique* c, Clique* neighbor);
void clique_store_message(Clique* to, Clique* from, const double* message, double log_scaling);

/* EM algorithm functions */
void sem_initialize_expected_counts(SEM* sem);
void sem_add_to_expected_counts(SEM* sem, int site_weight);
void sem_compute_marginal_probabilities_from_counts(SEM* sem);
void sem_update_parameters_from_posteriors(SEM* sem);
void sem_em_iteration(SEM* sem);
void sem_run_em_with_aitken(SEM* sem, int max_iterations);

/* HSS (root-invariant) reparameterization functions */
void sem_reparameterize_bh(SEM* sem);
void sem_set_model_parameters_using_hss(SEM* sem);
void sem_evaluate_bh_at_check_root(SEM* sem, const char* root_check_name);

#endif /* EMBH_TYPES_H */
