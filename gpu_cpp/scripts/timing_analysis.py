#!/usr/bin/env python3
"""
Analyze actual vs theoretical timing ratios for Pruning vs Propagation algorithms.
"""

def main():
    print("=" * 70)
    print("TIMING ANALYSIS: PRUNING vs PROPAGATION")
    print("=" * 70)

    # Actual timing results
    tests = [
        {
            'name': 'Small (648 patterns)',
            'fwd_ms': 0.987,
            'bwd_ms': 0.863,
        },
        {
            'name': 'Full (766,213 patterns)',
            'fwd_ms': 468.135,
            'bwd_ms': 214.671,
        }
    ]

    print("\nActual GPU Timing Results:")
    print("-" * 70)
    for t in tests:
        total = t['fwd_ms'] + t['bwd_ms']
        ratio = total / t['fwd_ms']
        bwd_fwd_ratio = t['bwd_ms'] / t['fwd_ms']
        print(f"\n{t['name']}:")
        print(f"  Forward pass (Pruning):  {t['fwd_ms']:10.3f} ms")
        print(f"  Backward pass:           {t['bwd_ms']:10.3f} ms")
        print(f"  Total (Propagation):     {total:10.3f} ms")
        print(f"  Propagation/Pruning:     {ratio:10.3f}")
        print(f"  Backward/Forward:        {bwd_fwd_ratio:10.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nTheoretical Estimates:")
    print("  - My initial estimate: Propagation/Pruning = 2.13x")
    print("    (Assumed backward pass = 100% of forward pass)")

    print("\n  - Actual results:")
    for t in tests:
        bwd_fwd = t['bwd_ms'] / t['fwd_ms']
        ratio = (t['fwd_ms'] + t['bwd_ms']) / t['fwd_ms']
        print(f"    {t['name']:30s} Backward/Forward = {bwd_fwd:.3f}, Total ratio = {ratio:.3f}")

    print("\nWhy is backward pass faster than forward pass?")
    print("-" * 70)

    print("""
1. Forward pass complexity (per internal edge):
   - For each of 4 parent states:
     - For each of 4 child states:
       - Read P[ps,cs] and child messages
       - Multiply: prod *= msg_child[cs]  (for each child)
       - Accumulate: sum += prod
   - Scale: find max, divide by max, log(max)
   - Memory access pattern: Read upward messages from multiple child edges

2. Backward pass complexity (per edge):
   - Read parent's downward message (4 values)
   - Matrix-vector product (4x4): P^T * down_parent
   - Multiply by sibling upward messages
   - Scale: find max, divide by max, log(max)
   - Memory access pattern: Read from single parent + siblings

Key Differences:
   - Forward pass does MORE computation per edge:
     * Must combine messages from ALL children (2+ multiplications per state)
     * Nested loops over parent and child states

   - Backward pass does LESS computation per edge:
     * Only combines message from ONE parent
     * Single matrix-vector product (transpose)
     * Sibling multiplication is simpler (just 1-2 siblings)

3. Memory Bandwidth:
   - Forward pass: Higher memory pressure due to multiple child message reads
   - Backward pass: More cache-friendly sequential access

Corrected Operation Count:
   Forward pass (internal edge, binary tree):
     4 parent states × (4 child states × (1 mul for P + 2 muls for 2 children) + 3 adds)
     = 4 × (4×3 + 3) = 4 × 15 = 60 ops

   Backward pass (non-root edge):
     4 child states × (4 parent states × 2 ops + 3 adds) + sibling muls
     = 4 × (4×2 + 3) + 4 = 4 × 11 + 4 = 48 ops

   Ratio: 48/60 = 0.80 (theoretical)

   But actual ratio is ~0.46 for large dataset!

4. GPU-specific effects:
   - Larger dataset = better GPU utilization
   - Backward pass may have better memory coalescing
   - Forward pass may have more register pressure
""")

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    avg_ratio = sum((t['fwd_ms'] + t['bwd_ms']) / t['fwd_ms'] for t in tests) / len(tests)
    avg_bwd_fwd = sum(t['bwd_ms'] / t['fwd_ms'] for t in tests) / len(tests)

    print(f"""
Measured Results:
  Average Propagation/Pruning ratio: {avg_ratio:.3f}x
  Average Backward/Forward ratio:    {avg_bwd_fwd:.3f}

This means Propagation algorithm takes about {avg_ratio:.1f}x longer than Pruning,
NOT the 2.13x I originally estimated.

The backward pass is more efficient than expected because:
  1. Simpler computation pattern (single parent vs multiple children)
  2. Better memory access pattern
  3. More efficient GPU utilization

Theoretical vs Actual:
  - Expected ratio: 2.13x (100% overhead for backward pass)
  - Actual ratio:   {avg_ratio:.3f}x (~{100*(avg_ratio-1):.0f}% overhead)

The ratio is {avg_ratio:.3f}, NOT equal to 2.13x, because the backward pass
is computationally cheaper than the forward pass.
""")


if __name__ == "__main__":
    main()
