# C++ to C Conversion Plan for EMBH Phylogenetic Software

## Project Overview
Convert EMBH (EM Barry-Hartigan) phylogenetic analysis software from C++ to C for CUDA parallelization. The project implements evolutionary maximum likelihood algorithms with both Felsenstein's pruning and Pearl's belief propagation algorithms.

**Key Components:**
- F81 evolutionary model
- Expectation-Maximization optimization
- PackedPatternStorage (3-bit DNA encoding)
- SEM (Statistical Evolutionary Model) class with clique tree belief propagation
- Pattern-based phylogenetic likelihood computation

## Pre-Conversion: Baseline Testing

### Step 0: Establish C++ Baseline
**Objective:** Create reference outputs that C code must match

**Actions:**
```bash
# Build and run baseline test
make clean
make test

# Capture baseline output
make test > baseline_output.txt 2>&1

# Key metrics to capture:
# 1. Log-likelihood values (should match to 11 decimal places)
# 2. EM convergence iterations and final parameters
# 3. Memory usage statistics
# 4. Timing information
# 5. Pattern compression statistics
```

**Expected Output Format:**
- Number of patterns: 1000
- Number of taxa: 38
- Log-likelihood from pruning algorithm: [exact value]
- Log-likelihood from propagation algorithm: [exact value]
- EM iterations and convergence values
- Final F81 model parameters (pi values, branch lengths)

**Validation Checkpoint 0:**
- [ ] Baseline test runs successfully
- [ ] Output saved to baseline_output.txt
- [ ] All log-likelihood values printed with precision >= 11
- [ ] No memory leaks detected (run with valgrind if available)

---

## Stage 1: Core Data Structures (C Structs)

### Objective
Convert C++ classes to C structs with function pointers where necessary. Focus on data layout that will be CUDA-friendly (Structure of Arrays where beneficial).

### Files to Create/Modify
- `embh_types.h` - All struct definitions
- `embh_types.c` - Constructor/destructor-like functions

### Conversion Tasks

#### 1.1: Basic Pattern and DNA Storage Structures
```c
// Pattern struct (simple, no dependencies)
typedef struct {
    int weight;
    char* characters;  // Dynamically allocated array
    int num_chars;
} Pattern;

// PackedPatternStorage struct
typedef struct {
    uint8_t* packed_data;
    int num_patterns;
    int num_taxa;
    size_t data_size;
} PackedPatternStorage;
```

**Functions to implement:**
- `Pattern* pattern_create(int weight, char* characters, int num_chars)`
- `void pattern_destroy(Pattern* p)`
- `PackedPatternStorage* packed_storage_create(int num_patterns, int num_taxa)`
- `void packed_storage_destroy(PackedPatternStorage* storage)`
- `uint8_t packed_storage_get_base(const PackedPatternStorage* storage, int pattern_idx, int taxon_idx)`
- `void packed_storage_set_base(PackedPatternStorage* storage, int pattern_idx, int taxon_idx, uint8_t base)`

**Validation Checkpoint 1.1:**
- [ ] Compile successfully with C compiler (gcc -std=c99)
- [ ] Test program creates and destroys patterns
- [ ] Test 3-bit packing/unpacking matches C++ behavior
- [ ] Memory leak check passes
- [ ] Unit test: Store 1000 patterns, retrieve all, verify correctness

#### 1.2: SEM_vertex Structure
```c
typedef struct SEM_vertex {
    int id;
    int global_id;
    int degree;
    int pattern_index;
    bool observed;
    
    // Neighbors and tree structure
    struct SEM_vertex** neighbors;
    struct SEM_vertex** children;
    struct SEM_vertex* parent;
    int num_neighbors;
    int num_children;
    int neighbors_capacity;
    int children_capacity;
    
    // DNA sequences
    int* DNArecoded;
    int* DNAcompressed;
    int dna_recoded_length;
    int dna_compressed_length;
    
    // Probabilities and matrices (4x4 for DNA)
    double transitionMatrix[4][4];
    double transitionMatrix_stored[4][4];
    double rootProbability[4];
    double posteriorProbability[4];
    double root_prob_hss[4];
    
    // Likelihood values
    double logScalingFactors;
    double vertexLogLikelihood;
    double sumOfEdgeLogLikelihoods;
    
    // Metadata
    char* name;
    char* newickLabel;
    int timesVisited;
    bool DNA_cond_like_initialized;
} SEM_vertex;
```

**Functions to implement:**
- `SEM_vertex* sem_vertex_create(int id, int* compressed_sequence, int seq_length)`
- `void sem_vertex_destroy(SEM_vertex* v)`
- `void sem_vertex_add_neighbor(SEM_vertex* v, SEM_vertex* neighbor)`
- `void sem_vertex_remove_neighbor(SEM_vertex* v, SEM_vertex* neighbor)`
- `void sem_vertex_add_child(SEM_vertex* v, SEM_vertex* child)`
- `void sem_vertex_set_parent(SEM_vertex* v, SEM_vertex* parent)`

**Validation Checkpoint 1.2:**
- [ ] Create vertex with sequence data
- [ ] Add/remove neighbors correctly
- [ ] Parent-child relationships work
- [ ] Dynamic array resizing for neighbors/children works
- [ ] No memory leaks
- [ ] Test with small tree (5-10 vertices)

#### 1.3: Clique and CliqueTree Structures
```c
typedef struct {
    int id;
    SEM_vertex** vertices;
    int num_vertices;
    int vertices_capacity;
    
    // Message passing
    double*** messages_to_neighbors;  // [neighbor_idx][pattern][4]
    int* neighbor_ids;
    int num_neighbors;
} Clique;

typedef struct {
    Clique** cliques;
    int num_cliques;
    int cliques_capacity;
    
    // Graph structure
    int** adjacency;  // adjacency[i][j] = 1 if cliques i and j are adjacent
    SEM_vertex* root;
} CliqueTree;
```

**Functions to implement:**
- `Clique* clique_create(int id)`
- `void clique_add_vertex(Clique* c, SEM_vertex* v)`
- `CliqueTree* clique_tree_create(SEM_vertex* root)`
- `void clique_tree_build(CliqueTree* tree, /* tree structure */)`
- `void clique_tree_destroy(CliqueTree* tree)`

**Validation Checkpoint 1.3:**
- [ ] Create clique tree from small test topology
- [ ] Verify clique structure matches C++ version
- [ ] Adjacency relationships correct
- [ ] Memory properly allocated and freed

#### 1.4: Main SEM Structure
```c
typedef struct {
    // Vertices and graph structure
    SEM_vertex** all_vertices;
    SEM_vertex** leaves;
    SEM_vertex** non_root_vertices;
    SEM_vertex* root;
    int num_vertices;
    int num_leaves;
    int num_non_root_vertices;
    
    // Hash maps (implement as simple arrays or use hash library)
    // For simplicity, use linear search or include uthash.h
    int* name_to_id_map;  // Or use uthash
    char** id_to_name_map;
    
    // Pattern data
    PackedPatternStorage* packed_patterns;
    int* DNAPatternWeights;
    bool* gapLessDNAFlag;
    double* DNAPatternGapProp;
    int* DNAPatternUniqueCount;
    int num_dna_patterns;
    
    // Model parameters
    double pi[4];  // Base frequencies
    double alpha;  // F81 parameter
    
    // Clique tree for belief propagation
    CliqueTree* clique_tree;
    
    // Results
    double logLikelihood;
    double* pattern_log_likelihoods;
    
    // Configuration
    bool verbose;
    double conv_thresh;
    int max_iterations;
} SEM;
```

**Functions to implement:**
- `SEM* sem_create(double alpha, int max_iter, bool verbose)`
- `void sem_destroy(SEM* sem)`
- `void sem_add_vertex(SEM* sem, const char* name, int* sequence, int seq_len)`
- `SEM_vertex* sem_get_vertex(SEM* sem, const char* name)`
- `bool sem_contains_vertex(SEM* sem, const char* name)`

**Validation Checkpoint 1.4:**
- [ ] Create SEM instance
- [ ] Add vertices by name
- [ ] Retrieve vertices by name
- [ ] All memory properly managed
- [ ] Test with pattern file from project

---

## Stage 2: Utility Functions and Helpers

### Objective
Implement namespace-like utility functions (emtr namespace in C++)

### Files to Create
- `embh_utils.h` - Utility function declarations
- `embh_utils.c` - Utility function implementations

### Functions to Implement

#### 2.1: Matrix Operations
```c
// 4x4 matrix helpers
void matrix_transpose_4x4(const double src[4][4], double dst[4][4]);
void matrix_multiply_4x4(const double A[4][4], const double B[4][4], double R[4][4]);
void matrix_identity_4x4(double M[4][4]);
void matrix_copy_4x4(const double src[4][4], double dst[4][4]);
```

#### 2.2: String Utilities
```c
bool string_starts_with(const char* str, const char* prefix);
char** split_whitespace(const char* str, int* num_tokens);
void free_string_array(char** arr, int num_tokens);
```

#### 2.3: DNA Conversion
```c
int convert_dna_to_index(char dna);
char convert_index_to_dna(int index);
double gap_proportion_in_pattern(const int* pattern, int length);
int unique_non_gap_count_in_pattern(const int* pattern, int length);
```

**Validation Checkpoint 2:**
- [ ] All matrix operations produce identical results to C++
- [ ] String utilities handle edge cases (empty strings, etc.)
- [ ] DNA conversion matches exactly
- [ ] Unit tests pass for all utility functions

---

## Stage 3: File I/O and Pattern Loading

### Objective
Convert file reading functions to C, handling FASTA and pattern files

### Files to Create
- `embh_io.h` - I/O function declarations
- `embh_io.c` - I/O implementations

### Functions to Implement

#### 3.1: Pattern File Reading
```c
void sem_read_patterns_from_file(SEM* sem, const char* pattern_filename, 
                                  const char* taxon_order_filename);
void sem_set_leaves_from_pattern_indices(SEM* sem);
```

#### 3.2: Topology File Reading
```c
void sem_set_edges_from_topology_file(SEM* sem, const char* edge_list_filename);
void sem_root_tree_at_vertex(SEM* sem, SEM_vertex* root);
void sem_set_vertex_vector(SEM* sem);
void sem_set_vertex_vector_except_root(SEM* sem);
```

#### 3.3: Base Composition Reading
```c
void sem_set_f81_model(SEM* sem, const char* base_comp_filename);
```

**Validation Checkpoint 3:**
- [ ] Read patterns_1000.pat successfully
- [ ] Read taxon_order file correctly
- [ ] Read edge list and build tree structure
- [ ] Verify tree topology matches C++ version
- [ ] F81 model parameters loaded correctly
- [ ] Compare with baseline: same number of patterns, taxa, tree structure

---

## Stage 4: Core Algorithm - Pruning Algorithm

### Objective
Implement Felsenstein's pruning algorithm for likelihood computation

### Files to Create
- `embh_pruning.h` - Pruning algorithm declarations
- `embh_pruning.c` - Pruning algorithm implementation

### Functions to Implement

#### 4.1: Conditional Likelihood Computation
```c
void compute_conditional_likelihoods_pruning(SEM* sem);
void init_leaf_conditionals(SEM_vertex* leaf, int pattern_idx, const PackedPatternStorage* patterns);
void compute_internal_conditionals(SEM_vertex* vertex, int pattern_idx);
```

#### 4.2: Likelihood Calculation
```c
double compute_log_likelihood_pruning(SEM* sem);
double compute_log_likelihood_at_root(SEM_vertex* root, const double pi[4], int pattern_idx);
```

**Validation Checkpoint 4:**
- [ ] Compute log-likelihood on test dataset
- [ ] Value matches C++ baseline (within 1e-10)
- [ ] Test on multiple root positions
- [ ] Performance benchmarked
- [ ] No numerical instabilities

**Critical Test:**
```bash
# Run C version
./embh_c -e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
         -p data/patterns_1000.pat \
         -x data/patterns_1000.taxon_order \
         -b data/patterns_1000.basecomp \
         -o h_0 -c h_5

# Compare log-likelihood output to baseline
# Must match to at least 10 decimal places
```

---

## Stage 5: Core Algorithm - Belief Propagation

### Objective
Implement Pearl's belief propagation algorithm

### Files to Create
- `embh_propagation.h` - Propagation algorithm declarations
- `embh_propagation.c` - Propagation implementation

### Functions to Implement

#### 5.1: Clique Tree Construction
```c
void construct_clique_tree(SEM* sem);
void identify_maximal_cliques(SEM* sem, CliqueTree* tree);
void build_clique_tree_structure(CliqueTree* tree);
```

#### 5.2: Message Passing
```c
void initialize_messages(CliqueTree* tree, int num_patterns);
void pass_messages_inward(CliqueTree* tree, Clique* current, Clique* parent, 
                         int pattern_idx, const PackedPatternStorage* patterns);
void pass_messages_outward(CliqueTree* tree, Clique* current, Clique* parent,
                          int pattern_idx);
```

#### 5.3: Likelihood Computation
```c
double compute_log_likelihood_propagation(SEM* sem);
void compute_log_likelihood_using_patterns_propagation_optimized(SEM* sem);
void compute_log_likelihood_using_patterns_propagation_memoized(SEM* sem);
```

**Validation Checkpoint 5:**
- [ ] Clique tree structure matches C++ version
- [ ] Message passing produces identical results
- [ ] Log-likelihood matches pruning algorithm (within 1e-10)
- [ ] Log-likelihood matches C++ baseline
- [ ] Optimized version faster than naive version
- [ ] Memoized version produces same results

**Critical Test:**
```bash
# Both algorithms must produce identical log-likelihoods
# Difference should be < 1e-10
```

---

## Stage 6: EM Algorithm and Optimization

### Objective
Implement Expectation-Maximization with Aitken acceleration

### Files to Create
- `embh_em.h` - EM algorithm declarations
- `embh_em.c` - EM implementation

### Functions to Implement

#### 6.1: E-Step
```c
void expectation_step(SEM* sem, double expected_counts[4][4]);
void compute_expected_sufficient_statistics(SEM* sem, double counts[4][4]);
```

#### 6.2: M-Step
```c
void maximization_step(SEM* sem, const double expected_counts[4][4]);
void update_model_parameters(SEM* sem, const double counts[4][4]);
void update_transition_matrices(SEM* sem);
```

#### 6.3: Convergence Acceleration
```c
void embh_aitken(SEM* sem, int max_iterations);
double compute_aitken_acceleration(double l0, double l1, double l2);
```

**Validation Checkpoint 6:**
- [ ] EM converges (monotonically increasing log-likelihood)
- [ ] Final parameters match C++ baseline
- [ ] Aitken acceleration works correctly
- [ ] Convergence speed comparable to C++
- [ ] Number of iterations matches baseline

**Critical Test:**
```bash
# Track log-likelihood over iterations
# Must be monotonically increasing
# Final value must match C++ baseline
```

---

## Stage 7: Manager and Main Program

### Objective
Implement top-level workflow coordination

### Files to Create
- `embh_manager.h` - Manager struct and functions
- `embh_manager.c` - Manager implementation
- `embh_main.c` - Main entry point

### Functions to Implement

#### 7.1: Manager Structure
```c
typedef struct {
    SEM* P;
    char* edge_list_file_name;
    char* pattern_file_name;
    char* base_comp_file_name;
    char* root_train;
    char* root_test;
    
    // Configuration
    bool verbose;
    int max_EM_iter;
    double conv_thresh;
    
    // Results
    double max_log_lik;
    double max_log_lik_ssh;
} Manager;

Manager* manager_create(const char* edge_list_file,
                       const char* pattern_file,
                       const char* taxon_order_file,
                       const char* base_comp_file,
                       const char* root_optim,
                       const char* root_check);
void manager_destroy(Manager* mgr);
void manager_run_analysis(Manager* mgr);
```

#### 7.2: BH Model Evaluation
```c
void evaluate_bh_model_with_root_at_check(SEM* sem, const char* root_check_name);
void verify_log_likelihood_at_all_cliques(SEM* sem);
```

**Validation Checkpoint 7:**
- [ ] Command-line parsing works
- [ ] Full pipeline executes successfully
- [ ] Output format matches C++ version
- [ ] All log-likelihoods match baseline

---

## Stage 8: Build System and Testing

### Objective
Create C build system and comprehensive test suite

### Files to Create
- `Makefile_c` - C build system
- `test_suite.c` - Automated tests
- `test_suite.h` - Test framework

### Makefile Structure
```makefile
CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra
LDFLAGS = -lm

# For CUDA preparation
CFLAGS_CUDA_READY = -std=c99 -O2 -fPIC

SOURCES = embh_types.c embh_utils.c embh_io.c \
          embh_pruning.c embh_propagation.c \
          embh_em.c embh_manager.c embh_main.c

OBJECTS = $(SOURCES:.c=.o)
TARGET = embh_c

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	./$(TARGET) -e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
	            -p data/patterns_1000.pat \
	            -x data/patterns_1000.taxon_order \
	            -b data/patterns_1000.basecomp \
	            -o h_0 -c h_5 > c_output.txt 2>&1
	diff -u baseline_output.txt c_output.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
```

### Test Suite Implementation
```c
// Test framework
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
} TestResults;

void test_pattern_storage(TestResults* results);
void test_sem_vertex_operations(TestResults* results);
void test_matrix_operations(TestResults* results);
void test_pruning_algorithm(TestResults* results);
void test_propagation_algorithm(TestResults* results);
void test_em_algorithm(TestResults* results);
void test_full_pipeline(TestResults* results);
```

**Validation Checkpoint 8:**
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Full pipeline test matches baseline
- [ ] Memory leak check passes (valgrind)
- [ ] Performance within 10% of C++ version

---

## Stage 9: CUDA Preparation Refactoring

### Objective
Restructure C code for optimal CUDA parallelization

### Refactoring Tasks

#### 9.1: Convert to Structure of Arrays (SoA)
```c
// BEFORE (Array of Structures - AoS)
typedef struct {
    double transition_matrix[4][4];
    double root_probability[4];
    // ... other fields
} SEM_vertex;
SEM_vertex* vertices[N];

// AFTER (Structure of Arrays - SoA)
typedef struct {
    double* transition_matrices;  // [N][4][4] contiguous
    double* root_probabilities;   // [N][4] contiguous
    int num_vertices;
} SEM_vertex_arrays;
```

#### 9.2: Identify Parallelizable Loops
Mark all loops that can be parallelized:
```c
// Pattern-level parallelism
#pragma PARALLEL_FOR_PATTERNS
for (int pattern_idx = 0; pattern_idx < num_patterns; pattern_idx++) {
    compute_likelihood_for_pattern(sem, pattern_idx);
}

// Vertex-level parallelism  
#pragma PARALLEL_FOR_VERTICES
for (int v = 0; v < num_vertices; v++) {
    update_vertex_conditionals(vertices, v, pattern_idx);
}
```

#### 9.3: Memory Layout Optimization
```c
// Align data structures for coalesced memory access
typedef struct {
    double* __attribute__((aligned(16))) conditional_likelihoods;
    double* __attribute__((aligned(16))) transition_matrices;
} GPU_ready_data;
```

**Validation Checkpoint 9:**
- [ ] SoA conversion maintains correctness
- [ ] All results still match baseline
- [ ] Memory layout is CUDA-friendly
- [ ] Identified at least 5 major parallel regions
- [ ] No performance regression from refactoring

---

## Stage 10: Final Validation and Documentation

### Objective
Comprehensive validation and documentation for CUDA development

### Validation Tasks

#### 10.1: Comprehensive Testing
```bash
# Run all test datasets
make test          # 1000 patterns
make test_2k       # 2000 patterns (if available)

# Memory validation
valgrind --leak-check=full --show-leak-kinds=all ./embh_c [args]

# Performance profiling
gprof ./embh_c gmon.out > profile.txt
```

#### 10.2: Numerical Validation
```python
# Python script to compare outputs
import numpy as np

cpp_ll = load_cpp_log_likelihood()
c_ll = load_c_log_likelihood()

assert np.abs(cpp_ll - c_ll) < 1e-10, "Log-likelihood mismatch!"
print(f"Log-likelihood match: {cpp_ll} vs {c_ll}")
print(f"Difference: {abs(cpp_ll - c_ll)}")
```

#### 10.3: Documentation
Create:
- `C_CONVERSION_NOTES.md` - Technical decisions and gotchas
- `CUDA_PARALLELIZATION_GUIDE.md` - Roadmap for CUDA implementation
- `API_REFERENCE.md` - All C function signatures
- Code comments explaining algorithm implementations

**Final Validation Checkpoint:**
- [ ] All tests pass with zero failures
- [ ] Log-likelihoods match to machine precision (< 1e-12)
- [ ] No memory leaks
- [ ] Performance within 5% of C++ version
- [ ] Code is fully documented
- [ ] CUDA parallelization points identified and documented
- [ ] Build system robust and tested

---

## CUDA Parallelization Strategy (Post-Conversion)

### High-Level Parallel Regions

1. **Pattern-Level Parallelism** (Primary)
   - Each pattern's likelihood can be computed independently
   - ~1000 patterns → 1000 CUDA threads
   - Speedup: Near-linear with pattern count

2. **Vertex-Level Parallelism** (Secondary)
   - Within each pattern, vertex operations can be parallel
   - Tree traversal needs synchronization
   - Useful for propagation algorithm

3. **Matrix Operations** (Tertiary)
   - 4x4 matrix multiplies can use cuBLAS
   - Small matrices, so benefit may be limited

### Memory Transfer Strategy
```
CPU (Host)           GPU (Device)
-----------          ------------
Pattern data   -->   Constant memory
Vertex data    -->   Global memory
Matrices       -->   Shared memory (per block)
Results        <--   Global memory
```

### Recommended CUDA Implementation Order
1. Parallelize pattern likelihood computation (pruning algorithm)
2. Parallelize message passing (propagation algorithm)
3. Parallelize EM sufficient statistics computation
4. Optimize memory transfers and coalescing
5. Use streams for overlapping computation and transfer

---

## Quality Metrics and Success Criteria

### Correctness
- ✅ All log-likelihoods match C++ baseline (< 1e-10 difference)
- ✅ EM convergence behavior identical
- ✅ Tree structure and cliques identical
- ✅ All intermediate results match

### Performance
- ✅ C version within 10% of C++ performance
- ✅ No memory leaks (valgrind clean)
- ✅ Memory usage comparable to C++

### Code Quality
- ✅ All functions documented
- ✅ Consistent naming conventions
- ✅ Error handling implemented
- ✅ Build system robust

### CUDA Readiness
- ✅ Data structures are SoA where beneficial
- ✅ No unnecessary dependencies between iterations
- ✅ Memory layout optimized for coalescing
- ✅ Parallel regions clearly identified

---

## Timeline Estimate

- **Stage 0**: 1 hour (baseline establishment)
- **Stage 1**: 1-2 days (data structures)
- **Stage 2**: 0.5 days (utilities)
- **Stage 3**: 1 day (I/O)
- **Stage 4**: 2-3 days (pruning algorithm)
- **Stage 5**: 3-4 days (belief propagation - most complex)
- **Stage 6**: 2-3 days (EM algorithm)
- **Stage 7**: 1 day (manager/main)
- **Stage 8**: 1-2 days (testing)
- **Stage 9**: 2-3 days (CUDA preparation)
- **Stage 10**: 1 day (validation/documentation)

**Total**: ~15-23 days for complete, validated conversion

---

## Key Risks and Mitigation

### Risk 1: Numerical Precision Differences
**Mitigation**: Use identical math library functions, validate intermediate results

### Risk 2: Memory Management Complexity
**Mitigation**: Implement clear ownership rules, use valgrind extensively

### Risk 3: Hash Map Implementation
**Mitigation**: Use uthash.h library or simple linear search for small N

### Risk 4: Complex Object Relationships
**Mitigation**: Document ownership and lifecycle clearly, use helper functions

### Risk 5: Performance Regression
**Mitigation**: Profile early and often, compare to C++ at each stage

---

## Tools and Resources

### Required Tools
- GCC or Clang (C99 support)
- Make
- Valgrind (memory debugging)
- GDB (debugging)
- Python (for validation scripts)

### Recommended Libraries
- `uthash.h` - Hash table in C
- `<math.h>` - Standard math functions

### Reference Documentation
- C99 standard
- CUDA Programming Guide
- RAxML-NG source (for reference implementation)

---

## Conclusion

This plan provides a systematic approach to converting the EMBH C++ codebase to C with CUDA parallelization as the end goal. Each stage has clear validation checkpoints ensuring correctness is maintained throughout the conversion. The final C code will be:

1. **Functionally identical** to the C++ version
2. **Well-tested** with comprehensive validation
3. **CUDA-ready** with optimal data structures and identified parallel regions
4. **Maintainable** with clear documentation

Upon completion, the codebase will be ready for CUDA kernel development with confidence that the serial C version is correct and optimized for GPU parallelization.
