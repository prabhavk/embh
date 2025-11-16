# GPU Log-Likelihood Integration Guide for EMBH

This guide explains how to integrate the GPU-accelerated log-likelihood computation into the EMBH C++ codebase.

## Overview

The `GPULikelihoodComputer` class provides a high-level interface for computing phylogenetic log-likelihoods on the GPU using Felsenstein's pruning algorithm. It achieves up to **26x speedup** compared to CPU-only computation.

## Performance Results

| Dataset | Patterns | CPU Time | GPU Time | Speedup |
|---------|----------|----------|----------|---------|
| patterns.pat | 72,369 | 1157.4 ms | 44.4 ms | **26x** |
| patterns_10000.pat | 617 | 1.0 ms | 0.93 ms | 1.07x |

The GPU excels with large pattern counts (>1000 patterns).

## Files

- `gpu_likelihood.cuh` - Header file with class declaration
- `gpu_likelihood.cu` - CUDA implementation
- `test_gpu_likelihood.cu` - Test and validation program
- `Makefile.cuda` - Build system

## Integration Steps

### 1. Build the GPU Library

```bash
cd gpu_cpp
make -f Makefile.cuda libgpu_likelihood.a
```

This creates:
- `gpu_likelihood.o` - Object file
- `libgpu_likelihood.a` - Static library

### 2. Link with Your Code

Add to your Makefile:
```makefile
CUDA_LIB = -L/path/to/gpu_cpp -lgpu_likelihood -lcudart
CUDA_INC = -I/path/to/gpu_cpp

your_program: your_code.cpp
    $(CXX) $(CXXFLAGS) $(CUDA_INC) -o $@ $< $(CUDA_LIB)
```

### 3. Include Header

```cpp
#include "gpu_likelihood.cuh"
```

### 4. Initialize GPU Computer

Extract tree structure from your SEM object:

```cpp
// Create GPU computer
GPULikelihoodComputer gpu_computer;

// Extract tree info from SEM
vector<int> edge_child_nodes;
vector<double> branch_lengths;
vector<bool> node_is_leaf;
vector<int> leaf_taxon_indices;
vector<vector<int>> node_children;
vector<int> post_order_edges;
vector<int> root_edge_ids;

// ... populate these from your clique tree ...

// Initialize
bool ok = gpu_computer.InitializeTree(
    num_nodes,
    num_edges,
    num_taxa,
    edge_child_nodes,
    branch_lengths,
    node_is_leaf,
    leaf_taxon_indices,
    node_children,
    post_order_edges,
    root_edge_ids
);

if (!ok) {
    cerr << "GPU initialization failed" << endl;
}
```

### 5. Compute Log-Likelihood

```cpp
// Set root probabilities
array<double, 4> root_prob = {0.25, 0.25, 0.25, 0.25};
gpu_computer.SetRootProbabilities(root_prob);

// Compute with patterns
double logLL = gpu_computer.ComputeLogLikelihoodFlat(
    num_patterns,
    pattern_bases_ptr,  // uint8_t* [num_patterns * num_taxa]
    pattern_weights_ptr // int* [num_patterns]
);
```

### 6. Update Branch Lengths (for optimization)

When branch lengths change during EM:

```cpp
vector<double> new_branch_lengths = ...;
gpu_computer.SetBranchLengths(new_branch_lengths);
double new_logLL = gpu_computer.ComputeLogLikelihoodFlat(...);
```

This is efficient - only recomputes transition matrices, doesn't reallocate memory.

## Integration with SEM Class

Here's how to add GPU support to the SEM class:

```cpp
// In embh_core.cpp

#ifdef USE_GPU
#include "gpu_likelihood.cuh"
#endif

class SEM {
    // ... existing members ...

#ifdef USE_GPU
    GPULikelihoodComputer* gpu_computer;
    bool gpu_initialized;

    void InitializeGPUComputer();
    void ExtractTreeStructureForGPU(
        vector<int>& edge_child_nodes,
        vector<double>& branch_lengths,
        vector<bool>& node_is_leaf,
        vector<int>& leaf_taxon_indices,
        vector<vector<int>>& node_children,
        vector<int>& post_order_edges,
        vector<int>& root_edge_ids
    );
#endif

public:
    void ComputeLogLikelihoodUsingPatternsWithPropagationGPU();
};

#ifdef USE_GPU
void SEM::InitializeGPUComputer() {
    if (!GPULikelihoodComputer::IsGPUAvailable()) {
        cerr << "GPU not available, falling back to CPU" << endl;
        gpu_initialized = false;
        return;
    }

    // Extract tree structure
    vector<int> edge_child_nodes;
    vector<double> branch_lengths;
    vector<bool> node_is_leaf;
    vector<int> leaf_taxon_indices;
    vector<vector<int>> node_children;
    vector<int> post_order_edges;
    vector<int> root_edge_ids;

    ExtractTreeStructureForGPU(
        edge_child_nodes, branch_lengths, node_is_leaf,
        leaf_taxon_indices, node_children, post_order_edges, root_edge_ids
    );

    gpu_computer = new GPULikelihoodComputer();
    gpu_initialized = gpu_computer->InitializeTree(
        node_is_leaf.size(),
        edge_child_nodes.size(),
        packed_patterns->get_num_taxa(),
        edge_child_nodes,
        branch_lengths,
        node_is_leaf,
        leaf_taxon_indices,
        node_children,
        post_order_edges,
        root_edge_ids
    );

    if (gpu_initialized) {
        array<double, 4> root_prob;
        for (int i = 0; i < 4; i++) {
            root_prob[i] = this->rootProbability[i];
        }
        gpu_computer->SetRootProbabilities(root_prob);
    }
}

void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationGPU() {
    if (!gpu_initialized) {
        // Fall back to CPU
        ComputeLogLikelihoodUsingPatternsWithPropagation();
        return;
    }

    // Get pattern data in flat format
    int num_patterns = num_patterns_from_file;
    int num_taxa = packed_patterns->get_num_taxa();

    vector<uint8_t> flat_patterns(num_patterns * num_taxa);
    for (int p = 0; p < num_patterns; p++) {
        vector<uint8_t> pat = packed_patterns->get_pattern(p);
        for (int t = 0; t < num_taxa; t++) {
            flat_patterns[p * num_taxa + t] = pat[t];
        }
    }

    // Compute on GPU
    this->logLikelihood = gpu_computer->ComputeLogLikelihoodFlat(
        num_patterns,
        flat_patterns.data(),
        pattern_weights.data()
    );
}
#endif
```

## Memory Requirements

For N patterns with E edges:
- Edge messages: N × E × 4 × 8 bytes
- Edge log scales: N × E × 8 bytes
- Pattern data: N × T × 1 byte

Example (72K patterns, 73 edges, 38 taxa):
- Working memory: ~205 MB
- Tree structure: ~50 KB

## Compilation with GPU Support

```bash
# Compile with GPU support
nvcc -DUSE_GPU -O3 -std=c++14 -arch=sm_60 \
    -o embh_gpu embh_core.cpp gpu_likelihood.cu -lcudart

# Or compile without GPU (falls back to CPU)
g++ -O3 -std=c++14 -o embh_cpu embh_core.cpp
```

## Checking GPU Availability at Runtime

```cpp
if (GPULikelihoodComputer::IsGPUAvailable()) {
    cout << "GPU acceleration enabled" << endl;
    InitializeGPUComputer();
} else {
    cout << "Using CPU-only computation" << endl;
}
```

## Best Practices

1. **Initialize once**: Call `InitializeTree()` once at startup, not per likelihood call
2. **Reuse buffers**: The library automatically manages GPU memory
3. **Batch patterns**: Compute all patterns at once, not one at a time
4. **Update branches efficiently**: Use `SetBranchLengths()` instead of re-initializing
5. **Check availability**: Always check `IsGPUAvailable()` before initialization

## Limitations

- Currently only supports Jukes-Cantor model (can be extended)
- Maximum 8 children per node (can be increased in kernel)
- Requires CUDA-capable GPU with compute capability 6.0+
- Pattern data must fit in GPU memory

## Future Optimizations

1. **Shared memory**: Store frequently accessed data in shared memory
2. **Stream processing**: Overlap computation with memory transfers
3. **Mixed precision**: Use float instead of double where precision allows
4. **Grouped approach**: Combine with memoization for additional speedup

## Troubleshooting

1. **CUDA not found**: Ensure CUDA toolkit is installed and nvcc is in PATH
2. **Out of memory**: Reduce pattern batch size or use GPU with more memory
3. **Wrong results**: Verify tree structure extraction matches CPU implementation
4. **Slow performance**: Ensure using Release build (-O3 optimization)

---

*Created: November 16, 2025*
*Project: EMBH (Expected Mutations under Branch Heterogeneity)*
