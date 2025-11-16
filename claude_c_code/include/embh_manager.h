#ifndef EMBH_MANAGER_H
#define EMBH_MANAGER_H

#include "embh_types.h"

/* Manager structure - coordinates the full EMBH pipeline */
typedef struct {
    /* Main SEM instance */
    SEM* P;

    /* File paths */
    char* edge_list_file;
    char* pattern_file;
    char* taxon_order_file;
    char* base_comp_file;
    char* root_optimize;
    char* root_check;
    char* cache_spec_file;  /* Optional: pre-computed cache specification */

    /* Configuration */
    bool verbose;
    int max_em_iterations;
    double conv_threshold;

    /* Results */
    double max_log_likelihood;
    double max_log_likelihood_hss;
} Manager;

/* Manager function declarations */
Manager* manager_create(const char* edge_list_file,
                        const char* pattern_file,
                        const char* taxon_order_file,
                        const char* base_comp_file,
                        const char* root_optimize,
                        const char* root_check,
                        const char* cache_spec_file);  /* Optional: can be NULL */
void manager_destroy(Manager* mgr);

/* Pipeline functions */
void manager_run_pipeline(Manager* mgr);
void manager_evaluate_bh_at_check_root(Manager* mgr);

#endif /* EMBH_MANAGER_H */
