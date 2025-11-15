/*
 * Additions to embh_core.cpp for Propagation Algorithm Pattern Comparison
 * 
 * Add these functions to embh_core.cpp
 */

// ============================================================================
// STEP 1: Add to SEM class methods (in embh_core.cpp)
// ============================================================================

/**
 * Compute log-likelihood using the propagation (belief propagation) algorithm
 * with pattern-based input. This uses the clique tree for message passing.
 */
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
        // Set current pattern index for the clique tree
        this->cliqueT->SetSite(pattern_idx);
        
        // Initialize potentials based on observed data
        // This reads from DNAcompressed which should be set up for patterns
        this->cliqueT->InitializePotentialAndBeliefs();
        
        // Run belief propagation (message passing)
        // This performs:
        //   1. Post-order traversal (leaves -> root): collect evidence
        //   2. Pre-order traversal (root -> leaves): distribute beliefs
        //   3. Compute beliefs at each clique
        this->cliqueT->CalibrateTree();
        
        // Marginalize at root clique to get P(root_state | data)
        // The marginalization integrates out one variable from the belief
        array<double, 4> marginalAtRoot = rootClique->MarginalizeOverVariable(rootClique->x);
        
        // Compute site likelihood: sum over all possible root states
        // L(pattern) = Σ_i π[i] * P(pattern | root=i)
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalAtRoot[dna];
        }
        
        // Compute log-likelihood with scaling factor
        // The scaling factor accounts for numerical underflow prevention
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
        
        // Add weighted contribution (pattern frequency)
        this->logLikelihood += siteLogLikelihood * pattern_weights[pattern_idx];
    }
}

/**
 * Compare pruning and propagation algorithms on pattern-based data.
 * Computes likelihood using both methods and reports differences.
 */
void SEM::ComparePruningAndPropagationOnPatterns(bool verbose) {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Cannot compare algorithms." << endl;
        return;
    }
    
    cout << "\n" << string(70, '=') << endl;
    cout << "COMPARING PRUNING AND PROPAGATION ALGORITHMS ON PATTERNS" << endl;
    cout << string(70, '=') << endl;
    cout << "Number of patterns: " << num_patterns_from_file << endl;
    cout << "Total sites (sum of weights): " << accumulate(pattern_weights.begin(), pattern_weights.end(), 0) << endl;
    cout << string(70, '=') << endl;
    
    // ========================================================================
    // Compute using Pruning Algorithm (Felsenstein's algorithm)
    // ========================================================================
    cout << "\n[1/2] Computing with PRUNING ALGORITHM..." << endl;
    auto start_pruning = chrono::high_resolution_clock::now();
    
    this->ComputeLogLikelihoodUsingPatterns();
    double ll_pruning = this->logLikelihood;
    
    auto end_pruning = chrono::high_resolution_clock::now();
    chrono::duration<double> time_pruning = end_pruning - start_pruning;
    
    cout << "  ✓ Complete" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_pruning << endl;
    cout << "  Time:           " << setprecision(4) << time_pruning.count() << " seconds" << endl;
    
    // ========================================================================
    // Ensure clique tree is constructed for propagation
    // ========================================================================
    if (this->cliqueT == nullptr) {
        cout << "\n  Building clique tree for propagation algorithm..." << endl;
        this->ConstructCliqueTree();
        cout << "  ✓ Clique tree constructed" << endl;
    }
    
    // ========================================================================
    // Compute using Propagation Algorithm (Belief Propagation)
    // ========================================================================
    cout << "\n[2/2] Computing with PROPAGATION ALGORITHM..." << endl;
    auto start_propagation = chrono::high_resolution_clock::now();
    
    this->ComputeLogLikelihoodUsingPatternsWithPropagation();
    double ll_propagation = this->logLikelihood;
    
    auto end_propagation = chrono::high_resolution_clock::now();
    chrono::duration<double> time_propagation = end_propagation - start_propagation;
    
    cout << "  ✓ Complete" << endl;
    cout << "  Log-likelihood: " << fixed << setprecision(10) << ll_propagation << endl;
    cout << "  Time:           " << setprecision(4) << time_propagation.count() << " seconds" << endl;
    
    // ========================================================================
    // Statistical Comparison
    // ========================================================================
    cout << "\n" << string(70, '=') << endl;
    cout << "STATISTICAL COMPARISON" << endl;
    cout << string(70, '=') << endl;
    
    double difference = ll_pruning - ll_propagation;
    double abs_difference = abs(difference);
    double relative_error = abs_difference / abs(ll_pruning);
    
    cout << fixed << setprecision(10);
    cout << "Pruning LL:       " << ll_pruning << endl;
    cout << "Propagation LL:   " << ll_propagation << endl;
    cout << "Difference:       " << scientific << setprecision(6) << difference << endl;
    cout << "Absolute diff:    " << scientific << abs_difference << endl;
    cout << "Relative error:   " << scientific << relative_error << endl;
    
    // ========================================================================
    // Performance Comparison
    // ========================================================================
    cout << "\n" << string(70, '-') << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << string(70, '-') << endl;
    
    cout << fixed << setprecision(6);
    cout << "Pruning time:       " << time_pruning.count() << " s" << endl;
    cout << "Propagation time:   " << time_propagation.count() << " s" << endl;
    
    if (time_propagation.count() > 0) {
        double speedup = time_pruning.count() / time_propagation.count();
        cout << "Speedup factor:     " << setprecision(2) << speedup << "x";
        if (speedup > 1.0) {
            cout << " (propagation faster)" << endl;
        } else if (speedup < 1.0) {
            cout << " (pruning faster)" << endl;
        } else {
            cout << " (equal)" << endl;
        }
    }
    
    // ========================================================================
    // Tolerance Check
    // ========================================================================
    cout << "\n" << string(70, '-') << endl;
    cout << "VALIDATION" << endl;
    cout << string(70, '-') << endl;
    
    double tolerance = 1e-6;
    bool passed = abs_difference < tolerance;
    
    cout << "Tolerance threshold: " << scientific << tolerance << endl;
    
    if (passed) {
        cout << "Result: ✓ PASSED - Algorithms agree within tolerance" << endl;
        cout << "        Both implementations are numerically consistent." << endl;
    } else {
        cout << "Result: ✗ FAILED - Algorithms differ beyond tolerance" << endl;
        cout << "        Difference: " << scientific << abs_difference << endl;
        cout << "        This may indicate:" << endl;
        cout << "          - Implementation bug in one algorithm" << endl;
        cout << "          - Different numerical precision/rounding" << endl;
        cout << "          - Different handling of edge cases" << endl;
    }
    
    cout << string(70, '=') << endl;
    
    // ========================================================================
    // Verbose mode: per-pattern details
    // ========================================================================
    if (verbose && (abs_difference >= tolerance || verbose)) {
        cout << "\nVERBOSE MODE: Per-pattern comparison (first 5 patterns)" << endl;
        cout << string(70, '-') << endl;
        cout << "Pattern | Weight | Pruning LL | Propagation LL | Difference" << endl;
        cout << string(70, '-') << endl;
        
        // Note: To implement per-pattern comparison, we would need to
        // compute likelihoods pattern-by-pattern, which requires
        // modifying the existing functions or creating new ones.
        // For now, we note this as future enhancement.
        
        cout << "[Per-pattern comparison not yet implemented]" << endl;
        cout << "To enable: implement single-pattern likelihood computation" << endl;
        cout << string(70, '-') << endl;
    }
}

// ============================================================================
// STEP 2: Add to manager class methods (in embh_core.cpp)
// ============================================================================

/**
 * Run comparison between pruning and propagation algorithms.
 * Called from manager to test both implementations.
 */
void manager::CompareAlgorithms(bool verbose) {
    cout << "\n" << string(70, '=') << endl;
    cout << "ALGORITHM COMPARISON SUITE" << endl;
    cout << string(70, '=') << endl;
    
    if (this->P == nullptr) {
        cerr << "Error: SEM object not initialized" << endl;
        return;
    }
    
    // Run the comparison
    this->P->ComparePruningAndPropagationOnPatterns(verbose);
    
    cout << "\nComparison complete." << endl;
}

// ============================================================================
// STEP 3: Add these declarations to embh_core.hpp
// ============================================================================

/*
In the SEM class, add:

    void ComputeLogLikelihoodUsingPatternsWithPropagation();
    void ComparePruningAndPropagationOnPatterns(bool verbose = false);

In the manager class, add:

    void CompareAlgorithms(bool verbose = false);
*/

// ============================================================================
// STEP 4: Usage in embh.cpp or manager constructor
// ============================================================================

/*
To use in manager constructor (embh_core.cpp), add after computing likelihoods:

    // Compare pruning and propagation algorithms
    if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
        this->CompareAlgorithms(false);  // Set to true for verbose output
    }

Or call from main (embh.cpp):

    embh_obj->CompareAlgorithms(false);
*/
