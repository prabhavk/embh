# New Function: ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm

## Overview

Added a new log-likelihood computation function that combines:
- **Pattern-based input** (like `ComputeLogLikelihoodUsingPatterns`)
- **Propagation algorithm** (like `ComputeLogLikelihoodUsingCompressedSequencesWithPropagationAlgorithm`)

## Function Signature

```cpp
void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm()
```

## Location

- **File**: `embh_core.cpp`
- **Lines**: Approximately 3733-3810
- **Declaration**: Line 1302

## Implementation Strategy

The function uses a clever approach to reuse existing clique tree infrastructure:

1. **Save original DNAcompressed data** for all observed vertices
2. **Resize DNAcompressed** to hold one value per pattern (instead of per site)
3. **Pre-populate** all pattern data into the resized DNAcompressed arrays
4. **Run standard propagation algorithm** using pattern indices as site indices
5. **Restore original DNAcompressed data** after computation

This approach avoids modifying the clique tree infrastructure while still enabling pattern-based computation.

## Key Code Sections

### 1. Save and Resize DNAcompressed

```cpp
map<SEM_vertex*, vector<int>> saved_DNAcompressed;
for (auto& pair : *this->vertexMap) {
    SEM_vertex* v = pair.second;
    if (v->observed) {
        saved_DNAcompressed[v] = v->DNAcompressed;
        v->DNAcompressed.resize(num_patterns_from_file, 4); // Initialize with gaps
    }
}
```

### 2. Pre-populate Patterns

```cpp
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
```

### 3. Run Propagation Algorithm

```cpp
for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
    this->cliqueT->SetSite(pattern_idx);
    this->cliqueT->InitializePotentialAndBeliefs();
    this->cliqueT->CalibrateTree();
    
    // Marginalize and weight by root probabilities (critical fix!)
    std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);
    
    double siteLikelihood = 0.0;
    for (int dna = 0; dna < 4; dna++) {
        siteLikelihood += this->rootProbability[dna] * marginalX[dna];
    }
    
    double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
    
    int weight = pattern_weights[pattern_idx];
    this->logLikelihood += siteLogLikelihood * weight;
}
```

### 4. Restore Original Data

```cpp
for (auto& pair : saved_DNAcompressed) {
    pair.first->DNAcompressed = pair.second;
}
```

## Integration with EMBH Constructor

The function is called in the EMBH constructor after pattern data is loaded:

```cpp
if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
    this->P->ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm();
    cout << "log-likelihood using propagation algorithm and packed patterns is " 
         << setprecision(20) << this->P->logLikelihood << endl;
}
```

## Test Results

All four log-likelihood computation methods now produce identical results:

```
=== Computing log-likelihood using compressed sequences ===
log-likelihood using pruning algorithm and compressed sequences is -27221.158997018177615

=== Computing log-likelihood using packed patterns ===
log-likelihood using pruning algorithm and packed patterns is -27221.158997018173977

log-likelihood using propagation algorithm and compressed sequences is -27221.158997018177615

log-likelihood using propagation algorithm and packed patterns is -27221.158997018173977
```

Differences are within machine precision (≈ 10^-12).

## Advantages

1. **Memory efficient**: Uses packed pattern storage (3 bits per base, 90% memory savings)
2. **Reuses infrastructure**: No modification to clique tree classes needed
3. **Correct computation**: Includes the critical fix (marginalization + root probability weighting)
4. **Non-invasive**: Saves and restores original data, leaving no side effects

## Comparison of All Four Methods

| Method | Input | Algorithm | Result |
|--------|-------|-----------|--------|
| ComputeLogLikelihood | Compressed sequences | Pruning | -27221.158997018177615 |
| ComputeLogLikelihoodUsingPatterns | Packed patterns | Pruning | -27221.158997018173977 |
| ComputeLogLikelihoodUsingCompressedSequencesWithPropagationAlgorithm | Compressed sequences | Propagation | -27221.158997018177615 |
| ComputeLogLikelihoodUsingPatternsWithPropagationAlgorithm | Packed patterns | Propagation | -27221.158997018173977 |

## Files

- **embh_core_WITH_PATTERN_PROPAGATION.cpp**: Complete source with all four functions
- Lines modified:
  - Line 1302: Function declaration
  - Lines 3733-3810: Function implementation
  - Lines 6432-6438: Call in EMBH constructor

## Usage

The function is automatically called when patterns are provided:

```bash
./embh -e edgelist.txt -f sequences.fas \
       -p patterns.pat -x taxon_order.txt \
       -b basecomp.txt -o root1 -c root2
```

Output includes log-likelihood from all four methods for verification.
