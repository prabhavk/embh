# GPU EMBH Optimization Summary

## Implemented Optimizations

### Version 1: `embh_aitken_gpu_optimized.cu`
**Performance: 8% faster, 20% memory reduction**

1. **Pre-computed parent edge lookup table**
   - O(1) lookup instead of O(num_edges) scan in backward pass
   - Eliminates linear search in `backward_pass_kernel` (lines 190-195 of original)

2. **Constant memory for root data**
   - Root probabilities and root edge IDs stored in `__constant__` memory
   - Faster access than global memory (broadcast to all threads)

3. **Fused scale accumulation**
   - Single `d_upward_total_scale` per pattern instead of per-edge scales
   - Reduces memory from `num_patterns * num_edges` to `num_patterns` doubles
   - **Memory saved: 847 MB (20% reduction)**

### Version 2: `embh_aitken_gpu_v2.cu`
**Performance: 15% faster**

4. **Edge-parallel expected counts**
   - Separate kernel launch per edge (73 launches/iteration)
   - Each kernel processes all patterns for one edge
   - Reduces atomicAdd contention significantly

5. **Warp-level reduction**
   - Uses `__shfl_down_sync` for fast intra-warp communication
   - Block-level reduction with shared memory
   - Only 1 atomicAdd per block instead of per thread

---

## Performance Results (766K patterns, 73 edges, 38 taxa)

| Version | Avg Time/Iter | Memory | Speedup |
|---------|--------------|--------|---------|
| Original | 915 ms | 4267 MB | baseline |
| V1 Optimized | 844 ms | 3420 MB | 1.08x |
| V2 Edge-Parallel | 778 ms | 3420 MB | 1.18x |

**Correctness**: All versions produce identical log-likelihood values.

---

## Future Optimizations to Test

### High Priority (Likely Impact)

1. **CUDA Streams for Overlapping**
   - Overlap edge kernel executions (V2 launches 73 sequential kernels)
   - Expected: Reduce kernel launch overhead

2. **Batched Edge Kernels**
   - Process multiple edges per kernel (e.g., 8-16 edges)
   - Use 2D grid: (patterns, edges)
   - Reduce launch overhead while maintaining parallelism

3. **Profile with Nsight/nvprof**
   - Identify actual bottlenecks (memory bandwidth vs compute)
   - Check occupancy and register usage
   - Find memory access inefficiencies

4. **Shared Memory for Transition Matrices**
   - Load 16 doubles (128 bytes) per edge into shared memory
   - Broadcast to all threads in block
   - Reduce global memory bandwidth

### Medium Priority

5. **Texture Memory for Pattern Data**
   - Read-only pattern access with hardware caching
   - May improve memory bandwidth for random access

6. **Coalesced Memory Access**
   - Restructure `d_upward_messages` layout
   - Current: `[pattern][edge][state]`
   - Consider: `[edge][pattern][state]` or `[state][pattern][edge]`

7. **Persistent Kernels**
   - Single kernel launch that loops internally
   - Avoid kernel launch overhead entirely
   - Requires careful synchronization

8. **Loop Unrolling**
   - `#pragma unroll` for 4x4 matrix operations
   - Compiler hints for state loops

### Lower Priority (Experimental)

9. **Half-Precision (FP16)**
   - Use `half` or `half2` for messages
   - 2x memory bandwidth, potential accuracy loss
   - Requires careful numerical analysis

10. **Cooperative Groups**
    - Modern CUDA synchronization primitives
    - Better control over thread cooperation

11. **Dynamic Parallelism**
    - Kernels launch child kernels
    - Could handle tree structure more naturally

12. **Graph-based Execution (CUDA Graphs)**
    - Pre-record kernel execution pattern
    - Reduce CPU overhead for repeated EM iterations

---

## Testing Commands

```bash
# Original
./embh_aitken_gpu -e ../cpp_program/data/tree_edges.txt \
  -p ../cpp_program/data/patterns.pat \
  -x ../cpp_program/data/patterns.taxon_order \
  -b ../cpp_program/data/patterns.basecomp 50

# V1 Optimized
./embh_aitken_gpu_optimized --edges ../cpp_program/data/tree_edges.txt \
  --patterns ../cpp_program/data/patterns.pat \
  --taxa ../cpp_program/data/patterns.taxon_order \
  --basecomp ../cpp_program/data/patterns.basecomp --max-iter 50

# V2 Edge-Parallel
./embh_aitken_gpu_v2 --max-iter 50

# Profile with nvprof
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency \
  ./embh_aitken_gpu_v2 --max-iter 5
```

---

## Key Bottleneck Analysis

Current bottleneck breakdown (estimated):
- **Forward pass**: ~40% (sequential edge processing within thread)
- **Backward pass**: ~25% (sequential edge processing)
- **Expected counts**: ~30% (atomicAdd contention in original, kernel overhead in V2)
- **Memory transfers**: ~5% (per-iteration transition matrix copies)

The main limitation is that each thread must process edges sequentially due to tree dependencies. Future optimizations should focus on:
1. Reducing per-iteration overhead
2. Improving memory access patterns
3. Better utilizing GPU parallelism across the (pattern × edge) space
