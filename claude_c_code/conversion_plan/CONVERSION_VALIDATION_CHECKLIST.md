# C Conversion Validation Checklist

Use this checklist to track progress through the conversion. Check off each item as it's completed and validated.

---

## Stage 0: Baseline Establishment ✓/✗

- [ ] C++ code compiles successfully
- [ ] `make test` runs without errors
- [ ] Baseline output saved to `baseline_output.txt`
- [ ] Extracted metrics saved to `baseline_metrics.json`
- [ ] Key values recorded:
  - [ ] Number of patterns: ___________
  - [ ] Number of taxa: ___________
  - [ ] Log-likelihood (pruning): ___.___________
  - [ ] Log-likelihood (propagation optimized): ___.___________
  - [ ] Log-likelihood (propagation memoized): ___.___________
  - [ ] EM iterations: ___________
  - [ ] Final log-likelihood: ___.___________

**Stage 0 Complete**: Date ________ Time Spent ________

---

## Stage 1.1: Pattern Storage ✓/✗

### Implementation
- [ ] Created `embh_types.h` with Pattern and PackedPatternStorage structs
- [ ] Created `embh_pattern.c` with all functions
- [ ] Created `test_pattern.c` with unit tests

### Validation
- [ ] Compiles without warnings: `gcc -std=c99 -Wall -Wextra test_pattern.c embh_pattern.c -o test_pattern`
- [ ] Test program runs successfully
- [ ] 3-bit packing/unpacking verified correct
- [ ] Memory usage correct (3 bits per base)
- [ ] Valgrind shows no leaks: `valgrind --leak-check=full ./test_pattern`
- [ ] Test output:
  ```
  Expected: Pattern storage uses X bytes
  Actual: Pattern storage uses X bytes
  ✓ Pattern packing/unpacking correct
  ✓ All tests passed
  ```

**Blockers/Issues**: _______________________________________________

**Stage 1.1 Complete**: Date ________ Time Spent ________

---

## Stage 1.2: SEM_vertex Structure ✓/✗

### Implementation
- [ ] Added SEM_vertex struct to `embh_types.h`
- [ ] Created `embh_vertex.c` with all functions
- [ ] Created `test_vertex.c` with unit tests

### Validation
- [ ] Compiles without warnings
- [ ] Can create/destroy vertices
- [ ] Add/remove neighbors works correctly
- [ ] Parent-child relationships correct
- [ ] Dynamic arrays resize correctly
- [ ] Matrix initialization to identity verified
- [ ] Valgrind shows no leaks
- [ ] Test creates tree: root → [child1, child2] → [grandchild1, grandchild2]

**Blockers/Issues**: _______________________________________________

**Stage 1.2 Complete**: Date ________ Time Spent ________

---

## Stage 1.3: Clique and CliqueTree ✓/✗

### Implementation
- [ ] Added Clique and CliqueTree structs to `embh_types.h`
- [ ] Created `embh_clique.c` with all functions
- [ ] Created `test_clique.c` with unit tests

### Validation
- [ ] Compiles without warnings
- [ ] Can create/destroy cliques
- [ ] Can build clique tree for simple 4-taxa tree
- [ ] Clique adjacency relationships correct
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 1.3 Complete**: Date ________ Time Spent ________

---

## Stage 1.4: Main SEM Structure ✓/✗

### Implementation
- [ ] Added SEM struct to `embh_types.h`
- [ ] Created `embh_sem.c` with all functions
- [ ] Implemented name-to-id mapping (simple linear search)
- [ ] Created `test_sem.c` with unit tests

### Validation
- [ ] Compiles without warnings
- [ ] Can create/destroy SEM instances
- [ ] Can add vertices by name
- [ ] Can retrieve vertices by name
- [ ] Contains_vertex check works
- [ ] Valgrind shows no leaks
- [ ] Test with 10+ vertices passes

**Blockers/Issues**: _______________________________________________

**Stage 1.4 Complete**: Date ________ Time Spent ________

**MILESTONE 1**: All data structures defined in C ✓/✗

---

## Stage 2: Utility Functions ✓/✗

### Implementation
- [ ] Created `embh_utils.h` with function declarations
- [ ] Created `embh_utils.c` with implementations
- [ ] Created `test_utils.c` with unit tests

### Functions Implemented
- [ ] matrix_transpose_4x4()
- [ ] matrix_multiply_4x4()
- [ ] matrix_identity_4x4()
- [ ] matrix_copy_4x4()
- [ ] string_starts_with()
- [ ] split_whitespace()
- [ ] free_string_array()
- [ ] convert_dna_to_index()
- [ ] convert_index_to_dna()

### Validation
- [ ] All matrix operations produce correct results
- [ ] Identity matrix: M * I = M verified
- [ ] Transpose: transpose(transpose(M)) = M verified
- [ ] DNA conversion: A→0, C→1, G→2, T→3, gap→4
- [ ] Round-trip: index→char→index works
- [ ] String utilities handle edge cases
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 2 Complete**: Date ________ Time Spent ________

---

## Stage 3: File I/O ✓/✗

### Implementation
- [ ] Created `embh_io.h` with function declarations
- [ ] Created `embh_io.c` with implementations

### Functions Implemented
- [ ] sem_read_patterns_from_file()
- [ ] sem_set_leaves_from_pattern_indices()
- [ ] sem_set_edges_from_topology_file()
- [ ] sem_root_tree_at_vertex()
- [ ] sem_set_vertex_vector()
- [ ] sem_set_vertex_vector_except_root()
- [ ] sem_set_f81_model()

### Validation with Project Data
- [ ] Successfully reads `patterns_1000.pat`
- [ ] Successfully reads `patterns_1000.taxon_order`
- [ ] Successfully reads `RAxML_bipartitions.CDS_FcC_partition.edgelist`
- [ ] Successfully reads `patterns_1000.basecomp`

### Data Verification
- [ ] Number of patterns matches baseline: _____ (expected: 1000)
- [ ] Number of taxa matches baseline: _____ (expected: 38)
- [ ] Tree structure verified (print Newick and compare)
- [ ] F81 parameters loaded:
  - [ ] pi[A] = _____
  - [ ] pi[C] = _____
  - [ ] pi[G] = _____
  - [ ] pi[T] = _____
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 3 Complete**: Date ________ Time Spent ________

**MILESTONE 2**: All input data can be loaded ✓/✗

---

## Stage 4: Pruning Algorithm ✓/✗

### Implementation
- [ ] Created `embh_pruning.h` with function declarations
- [ ] Created `embh_pruning.c` with implementations

### Functions Implemented
- [ ] compute_log_likelihood_pruning()
- [ ] init_leaf_conditionals()
- [ ] compute_internal_conditionals()
- [ ] compute_log_likelihood_at_root()

### Validation
- [ ] Compiles without warnings
- [ ] Loads test data successfully
- [ ] Computes log-likelihood for 1000 patterns

### Critical Numerical Validation
**Baseline Log-Likelihood**: _________________ (from baseline_output.txt)
**C Implementation Result**: _________________

- [ ] **Difference < 1e-10**: _____ (MUST BE TRUE)
- [ ] Print comparison:
  ```
  Log-likelihood (C pruning):  X.XXXXXXXXXXX
  Log-likelihood (baseline):   Y.YYYYYYYYYYY
  Absolute difference:         Z.ZZE-XX
  ```
- [ ] Tested with different root positions (results should vary but be consistent)
- [ ] No numerical instabilities observed
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 4 Complete**: Date ________ Time Spent ________

**MILESTONE 3**: First complete algorithm working ✓/✗

---

## Stage 5: Belief Propagation ✓/✗

### Implementation
- [ ] Created `embh_propagation.h` with function declarations
- [ ] Created `embh_propagation.c` with implementations

### Functions Implemented
- [ ] construct_clique_tree()
- [ ] initialize_messages()
- [ ] pass_messages_inward()
- [ ] pass_messages_outward()
- [ ] compute_log_likelihood_propagation()
- [ ] compute_log_likelihood_propagation_optimized()
- [ ] compute_log_likelihood_propagation_memoized()

### Validation - Basic Propagation
**Pruning Log-Likelihood**: _________________ (from Stage 4)
**Propagation Log-Likelihood**: _________________

- [ ] **Propagation matches Pruning** (difference < 1e-10)
- [ ] Both match C++ baseline
- [ ] Clique tree structure verified

### Validation - Optimizations
**Optimized Version Log-Likelihood**: _________________
**Memoized Version Log-Likelihood**: _________________

- [ ] Optimized version matches basic version
- [ ] Memoized version matches basic version
- [ ] All three match baseline

### Performance
- [ ] Timing: Optimized version: _____ ms
- [ ] Timing: Memoized version: _____ ms
- [ ] Memoized should be faster than optimized
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 5 Complete**: Date ________ Time Spent ________

**MILESTONE 4**: Both major algorithms working ✓/✗

---

## Stage 6: EM Algorithm ✓/✗

### Implementation
- [ ] Created `embh_em.h` with function declarations
- [ ] Created `embh_em.c` with implementations

### Functions Implemented
- [ ] expectation_step()
- [ ] maximization_step()
- [ ] embh_aitken()
- [ ] compute_expected_sufficient_statistics()
- [ ] update_model_parameters()

### Validation - Convergence
- [ ] EM converges (monotonically increasing log-likelihood)
- [ ] Convergence table printed (iter, log-lik, delta, parameters)
- [ ] Number of iterations: _____ (compare to baseline: _____)

### Validation - Final Results
**Initial Log-Likelihood**: _________________
**Final Log-Likelihood**: _________________
**Baseline Final Log-Likelihood**: _________________

- [ ] **Final LL matches baseline** (difference < 1e-10)

**Final Parameters**:
- [ ] pi[A] = _____ (baseline: _____)
- [ ] pi[C] = _____ (baseline: _____)
- [ ] pi[G] = _____ (baseline: _____)
- [ ] pi[T] = _____ (baseline: _____)
- [ ] alpha = _____ (baseline: _____)

- [ ] All parameters match baseline (within 1e-6)
- [ ] Aitken acceleration factor calculated
- [ ] No numerical issues during EM
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 6 Complete**: Date ________ Time Spent ________

---

## Stage 7: Manager and Main ✓/✗

### Implementation
- [ ] Created `embh_manager.h` with Manager struct and functions
- [ ] Created `embh_manager.c` with implementation
- [ ] Created `embh_main.c` with main() function

### Manager Functions
- [ ] manager_create()
- [ ] manager_destroy()
- [ ] manager_run_analysis()
- [ ] evaluate_bh_model_with_root_at_check()
- [ ] verify_log_likelihood_at_all_cliques()

### Command-Line Interface
- [ ] Parses -e (edge list file)
- [ ] Parses -p (pattern file)
- [ ] Parses -x (taxon order file)
- [ ] Parses -b (base composition file)
- [ ] Parses -o (root optimize)
- [ ] Parses -c (root check)
- [ ] Shows usage if arguments missing

### Validation - Full Pipeline
- [ ] Compiles without warnings
- [ ] Runs successfully:
  ```bash
  ./embh_c -e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
           -p data/patterns_1000.pat \
           -x data/patterns_1000.taxon_order \
           -b data/patterns_1000.basecomp \
           -o h_0 -c h_5 > c_output.txt 2>&1
  ```

### Output Validation
- [ ] Output format matches C++ baseline
- [ ] All log-likelihoods printed and match baseline
- [ ] BH model evaluation runs
- [ ] Clique verification runs
- [ ] Valgrind shows no leaks

**Blockers/Issues**: _______________________________________________

**Stage 7 Complete**: Date ________ Time Spent ________

**MILESTONE 5**: Complete working C implementation ✓/✗

---

## Stage 8: Build System and Testing ✓/✗

### Build System
- [ ] Created `Makefile_c` with all targets
- [ ] `make all` compiles successfully
- [ ] `make clean` removes all artifacts
- [ ] `make test` runs and compares to baseline
- [ ] `make debug` compiles with -g -DDEBUG
- [ ] `make valgrind` runs memory check

### Test Suite
- [ ] Created `test_suite.c` with comprehensive tests
- [ ] Unit tests for each module
- [ ] Integration test for full pipeline
- [ ] Automated comparison with baseline

### Validation Script
- [ ] Created `validate_conversion.sh`
- [ ] Extracts log-likelihoods from both outputs
- [ ] Compares numerically (not just text)
- [ ] Reports PASS/FAIL clearly

### Validation Results
```bash
$ make clean && make all
[compilation output]

$ make test
[test output]

$ ./validate_conversion.sh
Comparing C implementation to C++ baseline...
✓ Number of patterns: 1000 (matches baseline)
✓ Number of taxa: 38 (matches baseline)
✓ Log-likelihood (pruning): matches (diff: X.XXE-XX)
✓ Log-likelihood (propagation opt): matches (diff: X.XXE-XX)
✓ Log-likelihood (propagation mem): matches (diff: X.XXE-XX)
✓ Final EM log-likelihood: matches (diff: X.XXE-XX)
✓ All parameters match within tolerance

VALIDATION: PASSED
```

- [ ] All validation checks pass
- [ ] No memory leaks in any test
- [ ] All tests documented

**Blockers/Issues**: _______________________________________________

**Stage 8 Complete**: Date ________ Time Spent ________

---

## Stage 9: CUDA Preparation ✓/✗

### Structure of Arrays Conversion
- [ ] SEM vertex data converted to SoA layout
- [ ] Conditional likelihoods in contiguous array
- [ ] Transition matrices in contiguous array
- [ ] All pattern data already optimal (PackedPatternStorage)

### Parallelization Identification
- [ ] Pattern-level loops marked: `/* CUDA: Parallelize over patterns */`
- [ ] Vertex-level loops marked: `/* CUDA: Parallelize over vertices */`
- [ ] Dependencies documented
- [ ] Synchronization points identified

### Memory Layout Optimization
- [ ] All major data structures use single malloc
- [ ] No pointer chasing required in hot loops
- [ ] Data aligned for coalesced access (16-byte alignment)
- [ ] Documented in code comments

### Documentation
- [ ] Created `embh_cuda_prep.h` with GPU-ready structures
- [ ] Created `CUDA_ROADMAP.md` with:
  - [ ] Which functions become kernels
  - [ ] Expected speedup estimates
  - [ ] Memory transfer strategy
  - [ ] Optimization opportunities

### Validation After Refactoring
- [ ] All tests still pass
- [ ] Log-likelihoods still match baseline
- [ ] No performance regression (within 5%)
- [ ] No new memory leaks

**Blockers/Issues**: _______________________________________________

**Stage 9 Complete**: Date ________ Time Spent ________

**MILESTONE 6**: CUDA-ready C implementation ✓/✗

---

## Stage 10: Final Validation ✓/✗

### Multi-Dataset Testing
- [ ] Test with patterns_1000.pat - PASSED
- [ ] Test with patterns_2000.pat - PASSED (if available)
- [ ] Test with different root positions - PASSED

### Memory Validation
```bash
$ valgrind --leak-check=full --show-leak-kinds=all ./embh_c [args]
```
- [ ] No definitely lost bytes
- [ ] No indirectly lost bytes
- [ ] No possibly lost bytes
- [ ] All heap blocks freed

### Performance Comparison
**C++ Version Runtime**: _____ seconds
**C Version Runtime**: _____ seconds
**Ratio (C/C++)**: _____ (should be < 1.10)

- [ ] C version within 10% of C++ performance
- [ ] Hot loops profiled with gprof
- [ ] No obvious bottlenecks

### Documentation
- [ ] Created `C_API_REFERENCE.md` - all functions documented
- [ ] Created `CUDA_PARALLELIZATION_GUIDE.md` - CUDA strategy clear
- [ ] Created `C_CONVERSION_NOTES.md` - decisions and gotchas documented
- [ ] All code has meaningful comments

### Final Checklist
- [ ] ✅ All log-likelihoods match baseline (< 1e-12 difference)
- [ ] ✅ EM convergence identical to C++
- [ ] ✅ All parameters match
- [ ] ✅ No memory leaks
- [ ] ✅ No segfaults
- [ ] ✅ Performance acceptable
- [ ] ✅ Code documented
- [ ] ✅ Build system robust
- [ ] ✅ CUDA strategy documented
- [ ] ✅ Ready for parallelization

### Validation Summary Report
```
=== C CONVERSION VALIDATION SUMMARY ===

Correctness:
✓ Log-likelihood (pruning):       Matches baseline (diff: X.XXE-XX)
✓ Log-likelihood (propagation):   Matches baseline (diff: X.XXE-XX)  
✓ EM final log-likelihood:        Matches baseline (diff: X.XXE-XX)
✓ All parameters:                 Match within tolerance

Quality:
✓ Memory leaks:                   None (valgrind clean)
✓ Segmentation faults:            None
✓ Warning-free compilation:       Yes

Performance:
✓ Runtime vs C++:                 XX% (within 10% threshold)
✓ Memory usage:                   Comparable to C++

CUDA Readiness:
✓ Data structures:                Structure of Arrays where beneficial
✓ Parallel regions:               Identified and documented
✓ Memory layout:                  Optimized for coalescing
✓ Strategy documented:            Yes

OVERALL STATUS: READY FOR CUDA IMPLEMENTATION
```

**Blockers/Issues**: _______________________________________________

**Stage 10 Complete**: Date ________ Time Spent ________

---

## Overall Project Completion ✓/✗

### Summary Statistics
- **Total Time Spent**: _____ days/hours
- **Lines of C Code**: _____
- **Number of Test Cases**: _____
- **Test Pass Rate**: _____%

### Key Achievements
- [ ] Complete functional C implementation
- [ ] Bit-for-bit numerical accuracy
- [ ] Memory-safe implementation
- [ ] CUDA-ready code structure
- [ ] Comprehensive documentation

### Next Steps for CUDA
1. Implement pattern-level parallel kernel (highest priority)
2. Implement message-passing parallel kernel
3. Optimize memory transfers
4. Benchmark GPU vs CPU
5. Iterate on optimizations

### Known Limitations
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________

### Lessons Learned
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________

---

**PROJECT STATUS**: 
- [ ] IN PROGRESS
- [ ] CONVERSION COMPLETE, READY FOR CUDA
- [ ] CUDA IMPLEMENTATION IN PROGRESS

**Date Completed**: ________________

**Next Action**: ____________________________________
