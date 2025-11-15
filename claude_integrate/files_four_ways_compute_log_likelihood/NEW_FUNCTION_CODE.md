# New Function: ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm

## Complete Implementation

```cpp
void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm() {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }
    
    this->logLikelihood = 0;
    clique * rootClique = this->cliqueT->root;
    
    // Get pattern-to-taxon mapping (same as in ComputeLogLikelihoodUsingPatterns)
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }
    
    int num_taxa = packed_patterns->get_num_taxa();
    
    // Save original DNAcompressed data for observed vertices
    map<SEM_vertex*, vector<int>> saved_DNAcompressed;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed) {
            saved_DNAcompressed[v] = v->DNAcompressed;
            // Resize to hold pattern data (one element per pattern)
            v->DNAcompressed.resize(num_patterns_from_file, 4); // Initialize with gaps
        }
    }
    
    // Pre-populate all patterns into DNAcompressed
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);
        
        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                SEM_vertex* v = (*this->vertexMap)[it->second];
                v->DNAcompressed[pattern_idx] = pattern[taxon_idx];
            }
        }
    }
    
    // Now iterate over each pattern using standard propagation algorithm
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        this->cliqueT->SetSite(pattern_idx);
        this->cliqueT->InitializePotentialAndBeliefs();
        this->cliqueT->CalibrateTree();
        
        // Marginalize over child variable Y to get P(X | data) for root variable X
        std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);
        
        // Weight by root probabilities to get P(data, X) = P(data | X) * P(X)
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalX[dna];
        }
        
        // Combine with accumulated log scaling factors
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
        
        // Use pattern weight
        int weight = pattern_weights[pattern_idx];
        this->logLikelihood += siteLogLikelihood * weight;
    }
    
    // Restore original DNAcompressed data
    for (auto& pair : saved_DNAcompressed) {
        pair.first->DNAcompressed = pair.second;
    }
}
```

## Declaration (add to SEM class, around line 1302)

```cpp
void ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm();
```

## Usage in EMBH Constructor (add around line 6432)

```cpp
// Compute log-likelihood using propagation algorithm with patterns
if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
    this->P->ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm();
    cout << "log-likelihood using propagation algorithm and packed patterns is " 
         << setprecision(20) << this->P->logLikelihood << endl;
}
```

## How It Works

1. **Saves** original `DNAcompressed` data from observed vertices
2. **Resizes** `DNAcompressed` to hold one value per pattern (instead of per site)
3. **Pre-populates** all pattern data into the resized arrays
4. **Iterates** over patterns using the standard propagation algorithm
5. **Applies the critical fix**: Marginalizes belief and weights by root probabilities
6. **Restores** original `DNAcompressed` data

## Key Features

✅ Memory efficient (uses 3-bit packed patterns)
✅ Reuses existing clique tree infrastructure
✅ Includes the critical marginalization + root probability fix
✅ Non-invasive (restores original state)
✅ Produces identical results to other three methods

## Test Output

```
log-likelihood using pruning algorithm and compressed sequences is -27221.158997018177615
log-likelihood using pruning algorithm and packed patterns is -27221.158997018173977
log-likelihood using propagation algorithm and compressed sequences is -27221.158997018177615
log-likelihood using propagation algorithm and packed patterns is -27221.158997018173977
```

All four methods match within machine precision! ✅
