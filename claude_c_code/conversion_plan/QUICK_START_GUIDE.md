# Quick Start Guide: C++ to C Conversion

## 🚀 Getting Started in 5 Minutes

### Step 1: Establish Baseline (DO THIS FIRST!)

```bash
# Navigate to your project directory
cd /path/to/embh

# Run the C++ baseline test
make clean
make test > baseline_output.txt 2>&1

# Verify it worked
cat baseline_output.txt | grep "log-likelihood"
```

**Expected output should contain lines like:**
```
log-likelihood using OPTIMIZED propagation algorithm is 12345.67890123456
log-likelihood using MEMOIZED propagation algorithm is 12345.67890123456
```

### Step 2: Extract Key Baseline Values

Create a file `baseline_metrics.txt` with these values:
```
Number of patterns: [extract from output]
Number of taxa: [extract from output]
Log-likelihood (pruning): [extract value]
Log-likelihood (propagation optimized): [extract value]
Log-likelihood (propagation memoized): [extract value]
```

### Step 3: Choose Your Conversion Approach

**Option A: Full Systematic Conversion (Recommended)**
- Follow the complete conversion plan in `CPP_TO_C_CONVERSION_PLAN.md`
- Use stage-by-stage prompts from `CLAUDE_CODE_INSTRUCTIONS.md`
- Track progress with `CONVERSION_VALIDATION_CHECKLIST.md`

**Option B: Quick Prototype Approach**
- Start with Stage 4 (Pruning Algorithm)
- Use hardcoded test data first
- Validate numerical accuracy early
- Then backfill data structures

---

## 📋 Your First Claude Code Session

### Session 1: Data Structures (2-3 hours)

**Copy-paste this prompt to Claude Code:**

```
I'm converting a C++ phylogenetic analysis program to C for CUDA parallelization. 

Context:
- This is EMBH (EM Barry-Hartigan) phylogenetic software
- Key algorithms: Felsenstein's pruning, Pearl's belief propagation
- Uses 3-bit packed DNA pattern storage for memory efficiency
- Need exact numerical match to C++ baseline

First session goal: Create core data structures in C.

Please do the following:

1. Review the C++ code in embh_core.hpp and embh_core.cpp (lines 77-240)
   focusing on:
   - pattern class
   - PackedPatternStorage class  
   - SEM_vertex class

2. Create three new files:
   
   A) embh_types.h containing:
      - Pattern struct
      - PackedPatternStorage struct
      - SEM_vertex struct
      - All necessary #includes
      - Function declarations
   
   B) embh_pattern.c containing:
      - pattern_create()
      - pattern_destroy()
      - packed_storage_create()
      - packed_storage_destroy()
      - packed_storage_get_base()
      - packed_storage_set_base()
      - packed_storage_store_pattern()
      - packed_storage_get_pattern()
   
   C) test_pattern.c containing:
      - main() function
      - Create packed storage for 10 patterns, 5 taxa
      - Store test pattern: [0,1,2,3,4]
      - Retrieve and verify
      - Print memory usage
      - Free all memory

3. Create a simple Makefile to compile:
   gcc -std=c99 -Wall -Wextra test_pattern.c embh_pattern.c -o test_pattern

4. Ensure the 3-bit packing algorithm is identical to C++:
   - Each base takes exactly 3 bits
   - Bit positions calculated same way as C++
   - Handles cross-byte boundaries correctly

5. Run the test and verify:
   - Compiles without warnings
   - Runs without errors  
   - Memory usage is correct (3*num_patterns*num_taxa/8 bytes)
   - Values retrieved match values stored

Let me know if you need to see the relevant C++ code sections before starting.
```

---

## 🎯 Critical Success Criteria

### After Each Stage, Verify:

1. **Compilation**
   ```bash
   gcc -std=c99 -Wall -Wextra *.c -o test -lm
   # Should produce NO warnings
   ```

2. **Memory**
   ```bash
   valgrind --leak-check=full ./test
   # Should show: "All heap blocks were freed -- no leaks are possible"
   ```

3. **Numerical Accuracy**
   ```python
   # When you reach Stage 4+ (algorithms)
   import numpy as np
   baseline = 12345.67890123456  # Your actual baseline
   c_result = 12345.67890123457  # Your C implementation
   assert abs(baseline - c_result) < 1e-10
   ```

---

## 🔥 Common Pitfalls and Solutions

### Pitfall 1: "My log-likelihood doesn't match!"

**Check these in order:**
1. Is the tree topology identical?
   ```c
   // Print tree structure from both versions
   print_tree_newick(root);
   ```

2. Are matrices initialized correctly?
   ```c
   // Should be identity before SetF81Model
   for (int i = 0; i < 4; i++)
       for (int j = 0; j < 4; j++)
           printf("M[%d][%d] = %f\n", i, j, matrix[i][j]);
   ```

3. Is the loop order identical to C++?
   ```c
   // C++ uses this order - match it exactly!
   for (pattern) {
       for (vertex in post_order) {
           compute_conditional();
       }
   }
   ```

### Pitfall 2: "Segmentation fault!"

**Debug with GDB:**
```bash
gcc -g -O0 embh.c -o embh
gdb ./embh
(gdb) run -e edge.list -p patterns.pat ...
(gdb) backtrace
```

**Common causes:**
- Uninitialized pointer
- Array out of bounds
- Using freed memory

### Pitfall 3: "Memory leak detected!"

**Find the source:**
```bash
valgrind --leak-check=full --show-leak-kinds=all \
         --track-origins=yes --verbose \
         --log-file=leak.txt ./embh [args]

# Look for "definitely lost" first
grep "definitely lost" leak.txt
```

**Every malloc needs a free:**
```c
// Pattern 1: Constructor/Destructor
Pattern* p = pattern_create(10, data);
// ... use p ...
pattern_destroy(p);

// Pattern 2: Container ownership
SEM* sem = sem_create();
sem->vertices = malloc(10 * sizeof(SEM_vertex*));
// ... add vertices ...
sem_destroy(sem);  // This must free vertices array
```

---

## 📊 Progress Tracking Template

Create a file `conversion_progress.md`:

```markdown
# Conversion Progress

## Week 1
- [x] Established baseline (2024-XX-XX)
- [x] Stage 1.1: Pattern storage (2024-XX-XX) - 3 hours
- [ ] Stage 1.2: SEM_vertex (2024-XX-XX) - X hours
- [ ] Stage 1.3: Clique structures
- [ ] Stage 1.4: Main SEM

**Blockers:** None yet

**Next session:** Implement SEM_vertex structure

## Week 2
...
```

---

## 🧪 Testing Strategy

### Level 1: Unit Tests (Per-Function)
```c
void test_packed_storage() {
    PackedPatternStorage* ps = packed_storage_create(5, 3);
    
    // Test 1: Store and retrieve single base
    packed_storage_set_base(ps, 0, 0, 2);  // G
    uint8_t base = packed_storage_get_base(ps, 0, 0);
    assert(base == 2);
    
    // Test 2: Full pattern
    uint8_t pattern[] = {0, 1, 2};  // A, C, G
    packed_storage_store_pattern(ps, 1, pattern, 3);
    uint8_t* retrieved = packed_storage_get_pattern(ps, 1);
    assert(memcmp(pattern, retrieved, 3) == 0);
    
    packed_storage_destroy(ps);
    printf("✓ Packed storage tests passed\n");
}
```

### Level 2: Integration Tests (Multi-Component)
```c
void test_tree_building() {
    SEM* sem = sem_create(0.1, 100, false);
    
    // Add vertices
    int seq1[] = {0, 1, 2, 3};
    sem_add_vertex(sem, "leaf1", seq1, 4);
    sem_add_vertex(sem, "leaf2", seq1, 4);
    sem_add_vertex(sem, "internal1", NULL, 0);
    
    // Build tree structure
    SEM_vertex* leaf1 = sem_get_vertex(sem, "leaf1");
    SEM_vertex* leaf2 = sem_get_vertex(sem, "leaf2");
    SEM_vertex* internal = sem_get_vertex(sem, "internal1");
    
    sem_vertex_add_child(internal, leaf1);
    sem_vertex_add_child(internal, leaf2);
    
    assert(internal->num_children == 2);
    assert(leaf1->parent == internal);
    
    sem_destroy(sem);
    printf("✓ Tree building tests passed\n");
}
```

### Level 3: System Tests (Full Pipeline)
```bash
# Compare output file line-by-line
./embh_c [args] > c_output.txt
diff baseline_output.txt c_output.txt

# Or numerically
python compare_results.py baseline_output.txt c_output.txt
```

---

## 🎓 Learning Resources

### Understanding the Algorithms

**Felsenstein's Pruning:**
- Bottom-up tree traversal
- Each node computes conditional likelihoods
- Root combines all evidence
- O(n) per pattern

**Belief Propagation:**
- Message passing on clique tree
- Inward pass: leaves → root
- Outward pass: root → leaves
- Same result as pruning, but more flexible

**EM Algorithm:**
- E-step: Compute expected sufficient statistics
- M-step: Update parameters to maximize likelihood
- Aitken acceleration: Speeds up convergence
- Guarantees: Monotonic increase in log-likelihood

### C Programming Tips

**Memory Management:**
```c
// GOOD: Clear ownership
typedef struct {
    int* data;
    int size;
} Array;

Array* array_create(int size) {
    Array* a = malloc(sizeof(Array));
    a->data = malloc(size * sizeof(int));
    a->size = size;
    return a;
}

void array_destroy(Array* a) {
    free(a->data);  // Free contents first
    free(a);        // Then free container
}
```

**Error Handling:**
```c
// GOOD: Check all allocations
SEM* sem_create(double alpha, int max_iter, bool verbose) {
    SEM* sem = malloc(sizeof(SEM));
    if (!sem) {
        fprintf(stderr, "Failed to allocate SEM\n");
        return NULL;
    }
    
    sem->vertices = malloc(100 * sizeof(SEM_vertex*));
    if (!sem->vertices) {
        fprintf(stderr, "Failed to allocate vertex array\n");
        free(sem);
        return NULL;
    }
    
    // ... initialize ...
    return sem;
}
```

---

## ⚡ Performance Optimization (Post-Correctness)

### Only optimize after correctness is verified!

**Profile first:**
```bash
gcc -pg embh.c -o embh
./embh [args]
gprof embh gmon.out > profile.txt
```

**Common hotspots:**
1. Pattern likelihood loops (95% of time)
2. Matrix multiplications
3. Message passing

**Optimization checklist:**
- [ ] Use -O2 or -O3 compiler flag
- [ ] Ensure loop invariants are hoisted
- [ ] Use restrict keyword for non-aliasing pointers
- [ ] Consider loop unrolling for 4x4 matrices
- [ ] Profile before and after each optimization

---

## 📞 Getting Help

### When Stuck on a Bug

1. **Isolate the problem**
   - Create minimal test case
   - Binary search: which line causes the issue?

2. **Compare with C++**
   - Add debug prints to both versions
   - Compare intermediate values

3. **Ask Claude Code**
   ```
   I'm stuck on [specific issue]. Here's what I've tried:
   1. ...
   2. ...
   
   The C++ code does [X] (see lines Y-Z of embh_core.cpp)
   My C code does [A] (see my_file.c)
   
   The output differs: C++ gives X, C gives Y
   
   Can you help me identify the discrepancy?
   ```

### When Claude Code Gives Wrong Code

**Don't just accept it!**
- Ask for explanation of the approach
- Request comparison with C++ original
- Ask for validation logic

**Better prompt:**
```
The code you provided doesn't match the baseline. 

Expected log-likelihood: 12345.678
Actual log-likelihood: 12345.ABC (differs at 4th decimal)

Please:
1. Review the C++ implementation (embh_core.cpp lines X-Y)
2. Identify what's different in my C code
3. Explain why the difference causes numerical discrepancy
4. Provide corrected code with explanation
```

---

## ✅ Validation Milestones

### Milestone 1: Data Structures (Week 1)
- [ ] All structs compile
- [ ] Can create/destroy instances
- [ ] No memory leaks
- [ ] Unit tests pass

### Milestone 2: Data Loading (Week 1-2)
- [ ] Reads all input files
- [ ] Tree structure correct
- [ ] Pattern data correct
- [ ] F81 model loaded

### Milestone 3: First Algorithm (Week 2-3)
- [ ] Pruning algorithm computes
- [ ] Log-likelihood matches baseline
- [ ] Works on full 1000-pattern dataset

### Milestone 4: Second Algorithm (Week 3-4)
- [ ] Propagation algorithm computes
- [ ] Matches pruning algorithm
- [ ] Both match baseline

### Milestone 5: Optimization (Week 4-5)
- [ ] EM converges
- [ ] Final parameters match
- [ ] Full pipeline works

### Milestone 6: CUDA Ready (Week 5-6)
- [ ] SoA conversion complete
- [ ] Parallel regions identified
- [ ] All tests still pass
- [ ] Ready for CUDA

---

## 🎉 Success Indicators

**You know you're on track when:**
- ✅ Each stage passes validation before moving on
- ✅ Log-likelihoods match to > 10 decimal places
- ✅ Valgrind shows zero leaks
- ✅ Code is documented as you go
- ✅ You understand each algorithm you convert

**Warning signs:**
- ⚠️ Skipping validation steps
- ⚠️ Accumulating "fix later" items
- ⚠️ Not understanding why code works
- ⚠️ Numerical differences "close enough"
- ⚠️ Memory leaks "probably not important"

---

## 🚀 Ready to Start?

1. Set up your baseline (Step 1 above)
2. Open `CLAUDE_CODE_INSTRUCTIONS.md`
3. Start with Stage 0
4. Work through each stage systematically
5. Use the checklist to track progress

**First Claude Code prompt:** (from Session 1 above)

Good luck with your conversion! Remember:
- **Correctness first, performance second**
- **Validate at every stage**
- **When in doubt, match the C++ exactly**

You've got this! 💪
