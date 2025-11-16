# Claude Code Instructions for C++ to C Conversion

## Quick Start for Claude Code

### Initial Setup Command
```bash
# Run this first to establish baseline
cd /path/to/embh
make clean && make test > baseline_output.txt 2>&1
cat baseline_output.txt
```

### Stage-by-Stage Claude Code Prompts

## Stage 0: Baseline (START HERE)
**Prompt for Claude Code:**
```
I need to establish a baseline for C++ to C conversion. 

1. Run 'make clean && make test' and save output to baseline_output.txt
2. Extract and report these key values:
   - Number of patterns loaded
   - Number of taxa
   - Log-likelihood from pruning algorithm (to 11 decimal places)
   - Log-likelihood from propagation algorithm (to 11 decimal places)
   - EM convergence iterations
   - Any timing information
3. Confirm the test passes successfully
4. Create a file called baseline_metrics.json with the extracted values

This baseline will be our reference for validating the C conversion.
```

---

## Stage 1: Data Structures

### Stage 1.1: Pattern and PackedPatternStorage
**Prompt for Claude Code:**
```
Convert the PackedPatternStorage and pattern classes to C.

Reference files: embh_core.cpp (lines 77-181), embh_core.hpp

Create:
1. embh_types.h with:
   - Pattern struct definition
   - PackedPatternStorage struct definition
   - All function declarations

2. embh_pattern.c with:
   - pattern_create()
   - pattern_destroy()
   - packed_storage_create()
   - packed_storage_destroy()
   - packed_storage_get_base()
   - packed_storage_set_base()
   - packed_storage_store_pattern()
   - packed_storage_get_pattern()

3. test_pattern.c with unit tests that:
   - Create packed storage for 10 patterns, 5 taxa
   - Store test patterns using 3-bit encoding
   - Retrieve and verify all patterns match
   - Check memory usage (should be 3*10*5/8 = 19 bytes rounded up)
   - Free all memory

Validation:
- Compile with: gcc -std=c99 -Wall -Wextra test_pattern.c embh_pattern.c -o test_pattern
- Run and verify all tests pass
- Run valgrind: valgrind --leak-check=full ./test_pattern
```

### Stage 1.2: SEM_vertex
**Prompt for Claude Code:**
```
Convert the SEM_vertex class to a C struct.

Reference: embh_core.cpp lines 183-243

Add to embh_types.h:
- SEM_vertex struct (see conversion plan for complete definition)
- Include dynamic arrays for neighbors and children

Create embh_vertex.c with:
- sem_vertex_create()
- sem_vertex_destroy()
- sem_vertex_add_neighbor()
- sem_vertex_remove_neighbor()
- sem_vertex_add_child()
- sem_vertex_set_parent()
- sem_vertex_initialize_matrices()

Create test_vertex.c that:
- Creates 5 vertices
- Builds a small tree: root with 2 children, each child has 1 child
- Verifies parent-child relationships
- Verifies neighbor lists
- Tests matrix initialization (should be identity)
- Frees all memory

Validation: Compile, run, check with valgrind
```

### Stage 1.3: Clique and CliqueTree
**Prompt for Claude Code:**
```
Convert clique and cliqueTree classes to C.

Reference: embh_core.cpp lines 245-500 (approximate - search for "class clique" and "class cliqueTree")

Add to embh_types.h:
- Clique struct
- CliqueTree struct

Create embh_clique.c with:
- clique_create()
- clique_destroy()
- clique_add_vertex()
- clique_tree_create()
- clique_tree_destroy()
- clique_tree_add_clique()

Create test_clique.c that:
- Creates a simple clique tree for a 4-taxa tree
- Verifies clique structure
- Tests adjacency relationships
- Frees all memory

Validation: Compile, run, check with valgrind
```

### Stage 1.4: Main SEM Structure
**Prompt for Claude Code:**
```
Convert the main SEM class to a C struct.

Reference: embh_core.cpp, embh_core.hpp (SEM class definition)

Add to embh_types.h:
- SEM struct (see conversion plan)

Create embh_sem.c with:
- sem_create()
- sem_destroy()
- sem_add_vertex()
- sem_get_vertex_by_name()
- sem_get_vertex_by_id()
- sem_contains_vertex()

For the name-to-id mapping, use a simple linear search approach for now:
- char** vertex_names
- int* vertex_ids
- Linear search for lookups

Create test_sem.c that:
- Creates SEM instance
- Adds 10 vertices with names
- Retrieves vertices by name
- Verifies containment checks work
- Frees all memory

Validation: Compile, run, check with valgrind

Checkpoint: After this stage, all core data structures are defined in C.
```

---

## Stage 2: Utility Functions

**Prompt for Claude Code:**
```
Convert utility functions from the emtr namespace and other helpers to C.

Reference: embh_core.cpp lines 29-75

Create embh_utils.h with declarations for:
- matrix_transpose_4x4()
- matrix_multiply_4x4()
- matrix_identity_4x4()
- matrix_copy_4x4()
- string_starts_with()
- split_whitespace()
- free_string_array()
- convert_dna_to_index()
- convert_index_to_dna()

Create embh_utils.c with implementations.

Create test_utils.c that:
- Tests all matrix operations against known results
- Tests string operations
- Tests DNA conversion (A→0, C→1, G→2, T→3, gap→4)

Validation:
- Test matrix multiply: A*I = A
- Test transpose: transpose(transpose(A)) = A
- Test DNA conversion round-trip

Checkpoint: All utilities working correctly
```

---

## Stage 3: File I/O

**Prompt for Claude Code:**
```
Convert file reading functions to C.

Reference: 
- embh_core.cpp SetDNASequencesFromFile (lines 3984-4042)
- ReadPatternsFromFile (search for this function)
- SetEdgesFromTopologyFile (search for this function)
- SetF81Model (search for this function)

Create embh_io.h and embh_io.c with:
- sem_read_patterns_from_file()
- sem_set_leaves_from_pattern_indices()
- sem_set_edges_from_topology_file()
- sem_root_tree_at_vertex()
- sem_set_vertex_vector()
- sem_set_f81_model()

Test with actual project data:
- Read patterns_1000.pat
- Read patterns_1000.taxon_order  
- Read RAxML_bipartitions.CDS_FcC_partition.edgelist
- Read patterns_1000.basecomp

Validation:
- Number of patterns should match baseline
- Number of taxa should match baseline
- Tree structure should be identical (print tree in Newick format and compare)
- F81 parameters (pi values) should match baseline

Critical: After this stage, we should be able to load all input data.
```

---

## Stage 4: Pruning Algorithm

**Prompt for Claude Code:**
```
Convert Felsenstein's pruning algorithm to C.

Reference: Search embh_core.cpp for:
- ComputeLogLikelihoodUsingPatternsWithPruning
- Related functions for conditional likelihood computation

Create embh_pruning.h and embh_pruning.c with:
- compute_log_likelihood_pruning()
- init_leaf_conditionals()
- compute_internal_conditionals()
- compute_log_likelihood_at_root()

Implementation notes:
- Use packed patterns directly (no conversion to DNAcompressed)
- Traverse tree post-order for upward pass
- Apply transition matrices at each edge
- Sum over states at root using pi values

Create test_pruning.c that:
- Loads test data
- Computes log-likelihood
- Compares to baseline value

Validation:
- Log-likelihood must match C++ baseline to at least 10 decimal places
- Print: "Log-likelihood (C pruning): X.XXXXXXXXXXX"
- Print: "Log-likelihood (baseline):   Y.YYYYYYYYYYY"
- Print: "Difference: Z.ZZZZZZZZZZZZE-XX"

CRITICAL CHECKPOINT: This is the first full algorithm. If this works, we're on the right track.
```

---

## Stage 5: Belief Propagation

**Prompt for Claude Code:**
```
Convert Pearl's belief propagation algorithm to C.

This is the most complex stage. Reference:
- Search embh_core.cpp for:
  - ConstructCliqueTree
  - ComputeLogLikelihoodUsingPatternsWithPropagation
  - Message passing functions

Create embh_propagation.h and embh_propagation.c with:
- construct_clique_tree()
- initialize_messages()
- pass_messages_inward()
- pass_messages_outward()
- compute_log_likelihood_propagation()
- compute_log_likelihood_propagation_optimized()
- compute_log_likelihood_propagation_memoized()

Implementation approach:
1. First implement basic propagation algorithm
2. Test it matches pruning algorithm
3. Then implement optimizations
4. Test memoization

Validation:
- Log-likelihood from propagation must match pruning (< 1e-10 difference)
- Both must match C++ baseline
- Print timing comparison between optimized and memoized versions

CRITICAL: After this stage, both major algorithms work.
```

---

## Stage 6: EM Algorithm

**Prompt for Claude Code:**
```
Convert EM algorithm with Aitken acceleration to C.

Reference: Search embh_core.cpp for:
- embh_aitken
- E-step and M-step functions
- Expected sufficient statistics computation

Create embh_em.h and embh_em.c with:
- expectation_step()
- maximization_step()
- embh_aitken()
- compute_expected_sufficient_statistics()
- update_model_parameters()

Track convergence:
- Log-likelihood at each iteration
- Parameter changes
- Aitken acceleration factor

Validation:
- EM must converge (monotonically increasing log-likelihood)
- Final log-likelihood must match C++ baseline
- Number of iterations should be similar to C++
- Final parameter values (pi, alpha) must match

Print convergence table:
Iter | Log-likelihood | Delta LL    | pi[A]    | pi[C]    | pi[G]    | pi[T]
-----|----------------|-------------|----------|----------|----------|----------
0    | ...            | ...         | ...      | ...      | ...      | ...
...
```

---

## Stage 7: Manager and Main

**Prompt for Claude Code:**
```
Convert Manager class and create main() function.

Reference:
- embh_core.cpp manager::manager (lines 4061-4116)
- embh.cpp main() (lines 20-60)

Create embh_manager.h, embh_manager.c, and embh_main.c

Manager should:
1. Parse command-line arguments
2. Create SEM instance
3. Load all input files
4. Run pruning algorithm
5. Run propagation algorithms (optimized and memoized)
6. Run EM with Aitken acceleration
7. Evaluate BH model at check root
8. Verify log-likelihoods at all cliques
9. Print all results
10. Clean up memory

Main should:
- Parse arguments: -e, -p, -x, -b, -o, -c
- Create manager
- Run analysis
- Destroy manager
- Return 0 on success

Validation:
- Run: ./embh_c -e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
              -p data/patterns_1000.pat \
              -x data/patterns_1000.taxon_order \
              -b data/patterns_1000.basecomp \
              -o h_0 -c h_5

- All output should match baseline format and values
```

---

## Stage 8: Build System and Testing

**Prompt for Claude Code:**
```
Create comprehensive build system and test suite.

1. Create Makefile_c with:
   - All source files
   - Proper dependencies
   - test target that runs and compares to baseline
   - clean target
   - debug target (with -g -DDEBUG)
   - valgrind target

2. Create test_suite.c with:
   - Unit tests for each module
   - Integration test for full pipeline
   - Automated comparison with baseline

3. Create validate_conversion.sh script:
   - Runs C version
   - Compares output to baseline_output.txt
   - Checks log-likelihood values
   - Reports success/failure

Validation:
- make clean && make all
- make test (should pass)
- make valgrind (should show no leaks)
- ./validate_conversion.sh (should report SUCCESS)
```

---

## Stage 9: CUDA Preparation

**Prompt for Claude Code:**
```
Refactor code for optimal CUDA parallelization.

1. Convert key data structures to Structure of Arrays:
   - SEM vertex data (conditional likelihoods, matrices)
   - Pattern data already optimal
   
2. Identify and mark parallelizable loops:
   - Pattern-level loops (main parallelism)
   - Vertex-level loops (secondary)
   - Add comments: /* CUDA: Parallelize over patterns */

3. Ensure memory layout is contiguous:
   - All arrays should be single malloc() where possible
   - No pointer-chasing required

4. Create embh_cuda_prep.h with:
   - GPU-ready data structures
   - Memory layout specifications
   - Parallelization strategy documentation

5. Validate:
   - All refactoring must maintain correctness
   - Results must still match baseline
   - Add performance benchmarks

Document in CUDA_ROADMAP.md:
- Which loops will become CUDA kernels
- Expected speedup for each kernel
- Memory transfer strategy
- Optimization opportunities
```

---

## Stage 10: Final Validation

**Prompt for Claude Code:**
```
Comprehensive validation and documentation.

1. Run full test suite on multiple datasets:
   - patterns_1000.pat
   - patterns_2000.pat (if available)
   
2. Create validation report:
   - All log-likelihood comparisons
   - Performance comparison (C vs C++)
   - Memory usage comparison
   
3. Run extensive memory checks:
   - valgrind --leak-check=full
   - valgrind --tool=massif (memory profiling)
   
4. Create documentation:
   - C_API_REFERENCE.md (all functions documented)
   - CUDA_PARALLELIZATION_GUIDE.md
   - C_CONVERSION_NOTES.md (decisions, gotchas)

5. Final checklist:
   - [ ] All tests pass
   - [ ] No memory leaks
   - [ ] Log-likelihoods match (< 1e-12)
   - [ ] Performance within 5% of C++
   - [ ] Code documented
   - [ ] CUDA strategy clear

Create CONVERSION_COMPLETE.md summarizing:
- What was converted
- Validation results
- Known limitations
- Next steps for CUDA
```

---

## Quick Reference: Validation Commands

```bash
# Compile C version
make -f Makefile_c clean
make -f Makefile_c all

# Run test
./embh_c -e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
         -p data/patterns_1000.pat \
         -x data/patterns_1000.taxon_order \
         -b data/patterns_1000.basecomp \
         -o h_0 -c h_5 > c_output.txt 2>&1

# Compare with baseline
diff -u baseline_output.txt c_output.txt

# Extract and compare log-likelihoods
grep "log-likelihood" baseline_output.txt
grep "log-likelihood" c_output.txt

# Memory check
valgrind --leak-check=full --show-leak-kinds=all \
         --track-origins=yes --verbose \
         --log-file=valgrind_output.txt \
         ./embh_c [args]

# Check for no leaks
grep "definitely lost: 0 bytes" valgrind_output.txt
```

---

## Troubleshooting Common Issues

### Issue 1: Log-likelihood doesn't match
**Diagnosis:**
- Check matrix operations are bit-for-bit identical
- Verify loop order matches C++ version
- Check numerical precision in intermediates

**Fix:**
- Add debug prints for intermediate values
- Compare step-by-step with C++

### Issue 2: Memory leaks
**Diagnosis:**
- Use valgrind to identify leak source
- Check all malloc() have corresponding free()
- Check destroy functions are called

**Fix:**
- Add cleanup functions
- Use valgrind to verify

### Issue 3: Segmentation fault
**Diagnosis:**
- Check array bounds
- Verify pointers are initialized
- Use gdb to find crash location

**Fix:**
```bash
gdb ./embh_c
(gdb) run [args]
(gdb) backtrace
```

### Issue 4: Performance regression
**Diagnosis:**
- Profile with gprof
- Check for unnecessary copying
- Verify compiler optimization flags

**Fix:**
- Use -O2 or -O3
- Profile and optimize hotspots

---

## Success Criteria Summary

✅ **Correctness**
- Log-likelihoods match baseline (< 1e-10)
- EM converges identically
- All intermediate results match

✅ **Quality**
- No memory leaks (valgrind clean)
- No segfaults
- Robust error handling

✅ **Performance**
- Within 10% of C++ runtime
- Memory usage comparable

✅ **CUDA Readiness**
- Data structures are SoA
- Parallel regions identified
- Memory layout optimized

When all criteria are met, the conversion is complete and ready for CUDA implementation!
