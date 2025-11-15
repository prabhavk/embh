# Propagation Algorithm Comparison - Summary

## What This Adds

This enhancement adds the ability to compute log-likelihoods using **two different algorithms** and compare their results:

1. **Pruning Algorithm (Felsenstein)** - Post-order tree traversal, already implemented
2. **Propagation Algorithm (Belief Propagation)** - Message passing on clique tree, newly added

## Key Functions Added

### 1. `ComputeLogLikelihoodUsingPatternsWithPropagation()`
- Computes log-likelihood using belief propagation on the clique tree
- Uses the same pattern-based input as the pruning algorithm
- Should produce identical results to pruning (within numerical precision)

### 2. `ComparePruningAndPropagationOnPatterns(bool verbose)`
- Runs both algorithms and compares results
- Reports log-likelihoods, timing, and differences
- Validates that both methods agree

### 3. `CompareAlgorithms(bool verbose)`
- Manager-level wrapper for easy access
- Can be called from main() or within the manager constructor

## Files Modified

1. **embh_core.hpp** - Add 3 method declarations
2. **embh_core.cpp** - Add 3 method implementations
3. **embh_core.cpp (manager constructor)** - Add 1 function call (optional)
4. **embh.cpp** - Add 1 function call (alternative to #3)

## Quick Start

### Option A: Automatic Comparison (in manager constructor)
Add one line to the manager constructor:
```cpp
this->CompareAlgorithms(false);
```

### Option B: Manual Comparison (from main)
Call from main():
```cpp
embh_obj->CompareAlgorithms(false);
```

## What You'll See

After running, you'll see output like:

```
==================================================================
COMPARING PRUNING AND PROPAGATION ALGORITHMS ON PATTERNS
==================================================================

[1/2] Running PRUNING ALGORITHM...
  ✓ Complete
  Log-likelihood: -19310.0633113404
  Time:           0.0234 seconds

[2/2] Running PROPAGATION ALGORITHM...
  ✓ Complete
  Log-likelihood: -19310.0633113404
  Time:           0.0456 seconds

COMPARISON RESULTS
Pruning LL:       -19310.0633113404
Propagation LL:   -19310.0633113404
Difference:       -1.235e-10

✓ PASSED - Algorithms agree within tolerance
==================================================================
```

## Why This Is Useful

1. **Validation** - Confirms both implementations are correct
2. **Performance Comparison** - See which algorithm is faster for your data
3. **Debugging** - If results differ, indicates a bug in one implementation
4. **Flexibility** - Choose the best algorithm for your use case

## Technical Details

### Pruning Algorithm
- Works by computing conditional likelihoods from leaves to root
- Efficient for tree-structured models
- Standard approach in phylogenetics (Felsenstein 1981)

### Propagation Algorithm
- Uses belief propagation / sum-product algorithm
- Passes messages between cliques in the tree
- More general framework (works for any graphical model)
- Equivalent to pruning for tree models

### Expected Results
- **Log-likelihoods**: Should be identical within ~1e-10
- **Performance**: Pruning often faster for simple tree models
- **Tolerance**: Default 1e-6 for validation check

## Troubleshooting

### If algorithms disagree significantly:
1. Check that clique tree is properly constructed
2. Verify pattern data is correctly loaded
3. Ensure transition matrices are set correctly
4. Check for numerical underflow/overflow

### If propagation is much slower:
- This is normal - propagation has more overhead
- For simple trees, pruning is typically faster
- Propagation advantages appear in more complex graphical models

## References

- Felsenstein, J. (1981). "Evolutionary trees from DNA sequences"
- Pearl, J. (1988). "Probabilistic Reasoning in Intelligent Systems"
- Your SSH paper for the specific clique tree implementation

## Files Included

1. `CODE_SNIPPETS.cpp` - Ready-to-copy code
2. `INTEGRATION_GUIDE.md` - Step-by-step integration instructions
3. `propagation_implementation.cpp` - Complete documented implementation
4. `SUMMARY.md` - This file

## Quick Reference: Where to Add Code

```
embh_core.hpp
├── SEM class
│   ├── [ADD] void ComputeLogLikelihoodUsingPatternsWithPropagation();
│   └── [ADD] void ComparePruningAndPropagationOnPatterns(bool verbose = false);
└── manager class
    └── [ADD] void CompareAlgorithms(bool verbose = false);

embh_core.cpp
├── After ComputeLogLikelihoodUsingPatterns()
│   ├── [ADD] SEM::ComputeLogLikelihoodUsingPatternsWithPropagation() { ... }
│   └── [ADD] SEM::ComparePruningAndPropagationOnPatterns() { ... }
├── After EMhss()
│   └── [ADD] manager::CompareAlgorithms() { ... }
└── In manager::manager() constructor
    └── [ADD] this->CompareAlgorithms(false);  // Optional

embh.cpp (alternative)
└── In main(), after creating manager object
    └── [ADD] embh_obj->CompareAlgorithms(false);  // Alternative to above
```

## Next Steps

1. Copy code from `CODE_SNIPPETS.cpp`
2. Follow `INTEGRATION_GUIDE.md` for placement
3. Compile: `make clean && make all`
4. Test: `make test`
5. Verify output shows both algorithms agreeing

Good luck! 🚀
