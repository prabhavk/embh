// ============================================================================
// READY-TO-COPY CODE SNIPPETS
// Copy and paste these into your files
// ============================================================================

// ============================================================================
// 1. ADD TO embh_core.hpp (in SEM class, public section)
// ============================================================================

void ComputeLogLikelihoodUsingPatternsWithPropagation();
void ComparePruningAndPropagationOnPatterns(bool verbose = false);


// ============================================================================
// 2. ADD TO embh_core.hpp (in manager class, public section)
// ============================================================================

void CompareAlgorithms(bool verbose = false);


// ============================================================================
// 3. ADD TO embh_core.cpp (after ComputeLogLikelihoodUsingPatterns function)
// ============================================================================

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
    
    // Iterate over each pattern
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        // Set current pattern as the "site" for the clique tree
        this->cliqueT->SetSite(pattern_idx);
        
        // Initialize potentials based on observed data
        this->cliqueT->InitializePotentialAndBeliefs();
        
        // Run belief propagation (message passing)
        this->cliqueT->CalibrateTree();
        
        // Marginalize at root clique
        array<double, 4> marginalAtRoot = rootClique->MarginalizeOverVariable(rootClique->x);
        
        // Compute site likelihood
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalAtRoot[dna];
        }
        
        // Add log-likelihood with scaling
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
        this->logLikelihood += siteLogLikelihood * pattern_weights[pattern_idx];
    }
}

void SEM::ComparePruningAndPropagationOnPatterns(bool verbose) {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Cannot compare algorithms." << endl;
        return;
    }
    
    cout << "\n" << string(70, '=') << endl;
    cout << "COMPARING PRUNING AND PROPAGATION ALGORITHMS ON PATTERNS" << endl;
    cout << string(70, '=') << endl;
    cout << "Number of patterns: " << num_patterns_from_file << endl;
    cout << "Total sites: " << accumulate(pattern_weights.begin(), pattern_weights.end(), 0) << endl;
    cout << string(70, '=') << endl;
    
    // ========================================================================
    // Compute using Pruning Algorithm
    // ========================================================================
    cout << "\n[1/2] Running PRUNING ALGORITHM..." << endl;
    auto start_pruning = chrono::high_resolution_clock::now();
    
    this->ComputeLogLikelihoodUsingPatterns();
    double ll_pruning = this->logLikelihood;
    
    auto end_pruning = chrono::high_resolution_clock::now();
    chrono::duration<double> time_pruning = end_pruning - start_pruning;
    
    cout << "  ✓ Complete" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_pruning << endl;
    cout << "  Time:           " << setprecision(4) << time_pruning.count() << " seconds" << endl;
    
    // ========================================================================
    // Build clique tree if needed
    // ========================================================================
    if (this->cliqueT == nullptr) {
        cout << "\n  Building clique tree for propagation..." << endl;
        this->ConstructCliqueTree();
        cout << "  ✓ Clique tree constructed" << endl;
    }
    
    // ========================================================================
    // Compute using Propagation Algorithm
    // ========================================================================
    cout << "\n[2/2] Running PROPAGATION ALGORITHM..." << endl;
    auto start_propagation = chrono::high_resolution_clock::now();
    
    this->ComputeLogLikelihoodUsingPatternsWithPropagation();
    double ll_propagation = this->logLikelihood;
    
    auto end_propagation = chrono::high_resolution_clock::now();
    chrono::duration<double> time_propagation = end_propagation - start_propagation;
    
    cout << "  ✓ Complete" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_propagation << endl;
    cout << "  Time:           " << setprecision(4) << time_propagation.count() << " seconds" << endl;
    
    // ========================================================================
    // Comparison
    // ========================================================================
    cout << "\n" << string(70, '=') << endl;
    cout << "COMPARISON RESULTS" << endl;
    cout << string(70, '=') << endl;
    
    double difference = ll_pruning - ll_propagation;
    double abs_difference = abs(difference);
    double relative_error = abs_difference / abs(ll_pruning);
    
    cout << fixed << setprecision(10);
    cout << "Pruning LL:       " << ll_pruning << endl;
    cout << "Propagation LL:   " << ll_propagation << endl;
    cout << scientific << setprecision(6);
    cout << "Difference:       " << difference << endl;
    cout << "Absolute diff:    " << abs_difference << endl;
    cout << "Relative error:   " << relative_error << endl;
    
    cout << "\n" << string(70, '-') << endl;
    cout << "PERFORMANCE" << endl;
    cout << string(70, '-') << endl;
    
    cout << fixed << setprecision(6);
    cout << "Pruning time:     " << time_pruning.count() << " s" << endl;
    cout << "Propagation time: " << time_propagation.count() << " s" << endl;
    
    if (time_propagation.count() > 0) {
        double speedup = time_pruning.count() / time_propagation.count();
        cout << "Speedup factor:   " << setprecision(2) << speedup << "x";
        if (speedup > 1.0) {
            cout << " (propagation faster)" << endl;
        } else {
            cout << " (pruning faster)" << endl;
        }
    }
    
    // Tolerance check
    double tolerance = 1e-6;
    cout << "\n" << string(70, '-') << endl;
    cout << "VALIDATION (tolerance = " << scientific << tolerance << ")" << endl;
    cout << string(70, '-') << endl;
    
    if (abs_difference < tolerance) {
        cout << "✓ PASSED - Algorithms agree within tolerance" << endl;
    } else {
        cout << "✗ FAILED - Difference exceeds tolerance: " << scientific << abs_difference << endl;
    }
    
    cout << string(70, '=') << endl;
}


// ============================================================================
// 4. ADD TO embh_core.cpp (in manager section, after EMhss function)
// ============================================================================

void manager::CompareAlgorithms(bool verbose) {
    if (this->P == nullptr) {
        cerr << "Error: SEM object not initialized" << endl;
        return;
    }
    
    this->P->ComparePruningAndPropagationOnPatterns(verbose);
}


// ============================================================================
// 5. MODIFY manager constructor in embh_core.cpp
// Find this section (around line 6310-6316) and ADD the comparison call:
// ============================================================================

if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
    cout << "\n=== Computing log-likelihood using packed patterns ===" << endl;
    this->P->ReadPatternsFromFile(pattern_file_name, taxon_order_file_name);
    this->P->ComputeLogLikelihoodUsingPatterns();
    cout << "log-likelihood using pruning algorithm and packed patterns is "
         << setprecision(11) << this->P->logLikelihood << endl;
    
    // *** ADD THIS LINE: ***
    this->CompareAlgorithms(false);
}


// ============================================================================
// ALTERNATIVE: Call from main() in embh.cpp
// ============================================================================

int main(int argc, char* argv[]) {
    // ... existing code for argument parsing ...
    
    manager * embh_obj = new manager(
        string(edge_list_file_name),
        string(fasta_file_name),
        string(pattern_file_name),
        string(taxon_order_file_name),
        string(base_comp_file_name),
        string(root_optimize_name),
        string(root_check_name)
    );
    
    // *** ADD THIS: ***
    embh_obj->CompareAlgorithms(false);  // Set true for verbose
    
    delete embh_obj;
    return 0;
}


// ============================================================================
// EXPECTED OUTPUT EXAMPLE
// ============================================================================

/*
==================================================================
COMPARING PRUNING AND PROPAGATION ALGORITHMS ON PATTERNS
==================================================================
Number of patterns: 648
Total sites: 1000
==================================================================

[1/2] Running PRUNING ALGORITHM...
  ✓ Complete
  Log-likelihood: -19310.0633113404
  Time:           0.0234 seconds

[2/2] Running PROPAGATION ALGORITHM...
  ✓ Complete
  Log-likelihood: -19310.0633113404
  Time:           0.0456 seconds

==================================================================
COMPARISON RESULTS
==================================================================
Pruning LL:       -19310.0633113404
Propagation LL:   -19310.0633113404
Difference:       -1.234567e-10
Absolute diff:    1.234567e-10
Relative error:   6.394210e-15

------------------------------------------------------------------
PERFORMANCE
------------------------------------------------------------------
Pruning time:     0.023400 s
Propagation time: 0.045600 s
Speedup factor:   0.51x (pruning faster)

------------------------------------------------------------------
VALIDATION (tolerance = 1.000000e-06)
------------------------------------------------------------------
✓ PASSED - Algorithms agree within tolerance
==================================================================
*/
