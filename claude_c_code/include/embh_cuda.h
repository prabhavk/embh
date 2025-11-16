#ifndef EMBH_CUDA_H
#define EMBH_CUDA_H

#include "embh_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize CUDA data structures for pruning algorithm */
int cuda_init_pruning(SEM* sem);

/* Update transition matrices on GPU (call after M-step) */
int cuda_update_transition_matrices(SEM* sem);

/* Compute log-likelihood using CUDA-accelerated pruning */
double cuda_compute_log_likelihood(SEM* sem);

/* Run EM algorithm with CUDA acceleration */
void cuda_run_em_with_aitken(SEM* sem, int max_iterations);

/* Compute log-likelihood using CUDA with memoization */
double cuda_compute_log_likelihood_memoized(SEM* sem);

/* Initialize memoization structures */
int cuda_init_memoization(SEM* sem);

/* Cleanup memoization resources */
void cuda_cleanup_memoization(void);

/* Subtree-level memoization (more efficient for unique patterns) */
int cuda_init_subtree_memoization(SEM* sem);
double cuda_compute_log_likelihood_subtree_memoized(SEM* sem);
void cuda_cleanup_subtree_memoization(void);

/* Selective subtree memoization with pre-computed cache specification */
int cuda_load_cache_spec(const char* cache_spec_file);
int cuda_init_selective_subtree_memoization(SEM* sem, const char* cache_spec_file);
void cuda_free_cache_spec(void);

/* GPU-accelerated E-step */
int cuda_init_estep(SEM* sem);
int cuda_compute_expected_counts(SEM* sem);
int cuda_update_base_potentials(SEM* sem);
void cuda_cleanup_estep(void);

/* Cleanup CUDA resources */
void cuda_cleanup(void);

/* Check if CUDA is available */
int cuda_is_available(void);

/* Print CUDA device information */
void cuda_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif /* EMBH_CUDA_H */
