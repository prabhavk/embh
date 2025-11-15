# Integration Guide: Adding Propagation Algorithm Comparison

This guide shows exactly what to add to your existing code to enable comparison
between pruning and propagation algorithms for pattern-based likelihood computation.

## Overview

You'll add:
1. **ComputeLogLikelihoodUsingPatternsWithPropagation()** - Propagation-based likelihood
2. **ComparePruningAndPropagationOnPatterns()** - Comparison function
3. **CompareAlgorithms()** - Manager wrapper

## STEP 1: Update embh_core.hpp

Add these method declarations to the `SEM` class (around line 1340):

```cpp
// In SEM class, add after ComputeLogLikelihoodUsingPatterns():
void ComputeLogLikelihoodUsingPatternsWithPropagation();
void ComparePruningAndPropagationOnPatterns(bool verbose = false);
```

Add this method declaration to the `manager` class:

```cpp
// In manager class public methods:
void CompareAlgorithms(bool verbose = false);
```

## STEP 2: Update embh_core.cpp

### 2.1: Add ComputeLogLikelihoodUsingPatternsWithPropagation()

Add this function after `ComputeLogLikelihoodUsingPatterns()` (around line 3700):

```cpp
void SEM::ComputeLogLikelihoodUsingPatternsWithPropagation() {
    if (this->cliqueT == nullptr) {
        cerr << "Error: Clique tree not constructed. Call ConstructCliqueTree() first." << endl;
        return;
    }
    
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }
    
    this->logLikelihood = 0;
    clique* rootClique = this->cliqueT->root;
    
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        this->cliqueT->SetSite(pattern_idx);
        this->cliqueT->InitializePotentialAndBeliefs();
        this->cliqueT->CalibrateTree();
        
        array<double, 4> marginalAtRoot = rootClique->MarginalizeOverVariable(rootClique->x);
        
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalAtRoot[dna];
        }
        
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
        this->logLikelihood += siteLogLikelihood * pattern_weights[pattern_idx];
    }
}
```

### 2.2: Add ComparePruningAndPropagationOnPatterns()

Add this function after the previous one:

```cpp
void SEM::ComparePruningAndPropagationOnPatterns(bool verbose) {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Cannot compare algorithms." << endl;
        return;
    }
    
    cout << "\n" << string(70, '=') << endl;
    cout << "COMPARING PRUNING AND PROPAGATION ALGORITHMS" << endl;
    cout << string(70, '=') << endl;
    
    // Pruning algorithm
    auto start_pruning = chrono::high_resolution_clock::now();
    this->ComputeLogLikelihoodUsingPatterns();
    double ll_pruning = this->logLikelihood;
    auto end_pruning = chrono::high_resolution_clock::now();
    chrono::duration<double> time_pruning = end_pruning - start_pruning;
    
    cout << "\nPRUNING ALGORITHM:" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_pruning << endl;
    cout << "  Time:           " << setprecision(4) << time_pruning.count() << " s" << endl;
    
    // Build clique tree if needed
    if (this->cliqueT == nullptr) {
        this->ConstructCliqueTree();
    }
    
    // Propagation algorithm
    auto start_propagation = chrono::high_resolution_clock::now();
    this->ComputeLogLikelihoodUsingPatternsWithPropagation();
    double ll_propagation = this->logLikelihood;
    auto end_propagation = chrono::high_resolution_clock::now();
    chrono::duration<double> time_propagation = end_propagation - start_propagation;
    
    cout << "\nPROPAGATION ALGORITHM:" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_propagation << endl;
    cout << "  Time:           " << setprecision(4) << time_propagation.count() << " s" << endl;
    
    // Comparison
    double difference = abs(ll_pruning - ll_propagation);
    double tolerance = 1e-6;
    
    cout << "\nCOMPARISON:" << endl;
    cout << "  Difference: " << scientific << setprecision(6) << difference << endl;
    cout << "  Speedup:    " << fixed << setprecision(2) 
         << (time_pruning.count() / time_propagation.count()) << "x" << endl;
    
    if (difference < tolerance) {
        cout << "  Status: ✓ PASSED (within tolerance " << scientific << tolerance << ")" << endl;
    } else {
        cout << "  Status: ✗ FAILED (exceeds tolerance)" << endl;
    }
    
    cout << string(70, '=') << endl;
}
```

### 2.3: Add CompareAlgorithms() to manager

Add this function in the manager section (around line 6376):

```cpp
void manager::CompareAlgorithms(bool verbose) {
    if (this->P == nullptr) {
        cerr << "Error: SEM object not initialized" << endl;
        return;
    }
    
    this->P->ComparePruningAndPropagationOnPatterns(verbose);
}
```

## STEP 3: Update manager constructor

In the manager constructor (around line 6316), add the comparison call:

```cpp
if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
    cout << "\n=== Computing log-likelihood using packed patterns ===" << endl;
    this->P->ReadPatternsFromFile(pattern_file_name, taxon_order_file_name);
    this->P->ComputeLogLikelihoodUsingPatterns();
    cout << "log-likelihood using pruning algorithm and packed patterns is "
         << setprecision(11) << this->P->logLikelihood << endl;
    
    // ADD THIS:
    this->CompareAlgorithms(false);  // Set to true for verbose output
}
```

## STEP 4: Alternative - Call from main()

Alternatively, you can call it from main() in embh.cpp:

```cpp
int main(int argc, char* argv[]) {
    // ... existing argument parsing ...
    
    manager * embh_obj = new manager(
        string(edge_list_file_name),
        string(fasta_file_name),
        string(pattern_file_name),
        string(taxon_order_file_name),
        string(base_comp_file_name),
        string(root_optimize_name),
        string(root_check_name)
    );
    
    // ADD THIS:
    embh_obj->CompareAlgorithms(false);
    
    delete embh_obj;
    return 0;
}
```

## Expected Output

When you run the program, you should see output like:

```
==================================================================
COMPARING PRUNING AND PROPAGATION ALGORITHMS
==================================================================

PRUNING ALGORITHM:
  Log-likelihood: -19310.0633113404
  Time:           0.0234 s

PROPAGATION ALGORITHM:
  Log-likelihood: -19310.0633113404
  Time:           0.0456 s

COMPARISON:
  Difference: 1.234567e-10
  Speedup:    0.51x
  Status: ✓ PASSED (within tolerance 1.000000e-06)
==================================================================
```

## Testing

After integration, compile and test:

```bash
make clean
make all
make test
```

You should see the comparison output showing that both algorithms produce
nearly identical log-likelihoods (within numerical precision).

## Notes

1. The propagation algorithm requires the clique tree to be constructed
2. Both algorithms should produce identical results (within ~1e-10 tolerance)
3. Performance may vary - pruning is often faster for tree-structured models
4. Set verbose=true to get detailed per-pattern analysis (when implemented)
