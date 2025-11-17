#!/usr/bin/env python3
"""
Analyze computational complexity of Pruning vs Propagation algorithms.
Counts operations in each kernel for accurate comparison.
"""

def count_forward_pass_operations(num_edges, num_root_edges, avg_children=2):
    """
    Forward pass kernel - for each edge, each pattern:
    - Leaf nodes: just copy P[ps,base] (4 assignments)
    - Internal nodes: matrix-vector product with message multiplication
    """
    # For internal nodes (assume ~half are internal in binary tree)
    # Each internal node: 4 parent states x (4 child states x (1 mul for P + avg_children muls for messages) + 3 adds)
    # = 4 * (4 * (1 + avg_children) + 3) = 4 * (4*(1+2) + 3) = 4 * 15 = 60 ops per internal edge
    # Plus scaling: find max (3 comparisons), 4 divides, 1 log

    # Actually, let's be more precise from the code:
    # Lines 114-125 for internal node:
    ops_per_internal_edge = {
        'assignments': 4,  # msg[ps] = sum
        'additions': 4 * 3,  # 4 parent states * 3 additions per sum
        'multiplications': 0,  # count separately
    }

    # For each of 4 parent states:
    # for cs in 0..3: sum += prod where prod = P[ps,cs] * msg_child1[cs] * msg_child2[cs]...
    # That's 4 muls (P * child1 * child2 for each cs) + 3 adds
    # With avg_children=2: P * c1 * c2 = 2 muls per cs, 4 cs = 8 muls per ps, 4 ps = 32 muls
    ops_per_internal_edge['multiplications'] = 4 * 4 * avg_children  # = 32

    # Scaling per edge (lines 128-132):
    # fmax(fmax(msg[0], msg[1]), fmax(msg[2], msg[3])) = 3 comparisons
    # if mx > 0: 1 comparison
    # 4 divisions: msg[i] /= mx
    # 1 log: log(mx)
    scaling_ops = {
        'comparisons': 4,
        'divisions': 4,
        'logs': 1
    }

    # Total for forward pass (assuming 50% internal edges)
    # Actually, in a bifurcating tree with n leaves, there are n-1 internal nodes
    # and n + (n-1) = 2n-1 edges. About half are internal.

    num_internal_edges = num_edges // 2
    num_leaf_edges = num_edges - num_internal_edges

    total_ops = {
        'assignments': num_leaf_edges * 4 + num_internal_edges * 4,
        'additions': num_internal_edges * 12,  # 4 ps * 3 adds
        'multiplications': num_internal_edges * 32,  # 4 ps * 4 cs * 2 children
        'comparisons': num_edges * 4,  # scaling
        'divisions': num_edges * 4,  # scaling
        'logs': num_edges * 1,  # scaling
    }

    return total_ops


def count_backward_pass_operations(num_edges, num_root_edges, avg_siblings=1):
    """
    Backward pass kernel - for each edge (in reverse post order):
    """
    # For root edges (lines 182-189):
    # msg[i] = root_probs[i] for 4 states (4 assignments)
    # multiply by sibling messages: (num_root_edges-1) * 4 muls
    root_edge_ops = {
        'assignments': 4,
        'multiplications': (num_root_edges - 1) * 4,
    }

    # For non-root edges (lines 190-220):
    # Finding parent edge: O(num_edges) loop - but this is inefficient, let's count it
    # Matrix-vector product: 4 child states * (4 parent states * (1 mul + 1 add))
    # = 4 * 4 * 2 = 32 ops (16 muls + 16 adds, but actually 12 adds due to accumulation)
    # Then multiply by sibling messages: avg_siblings * 4 muls
    non_root_ops_per_edge = {
        'assignments': 4,  # msg[cs] = sum
        'additions': 4 * 3,  # 4 cs * 3 adds per sum
        'multiplications': 4 * 4 + avg_siblings * 4,  # 16 for P*down + sibling muls
    }

    # Scaling per edge (lines 222-226):
    scaling_ops = {
        'comparisons': 4,
        'divisions': 4,
        'logs': 1
    }

    total_ops = {
        'assignments': num_edges * 4,
        'additions': (num_edges - num_root_edges) * 12,
        'multiplications': num_root_edges * (num_root_edges - 1) * 4 + (num_edges - num_root_edges) * (16 + avg_siblings * 4),
        'comparisons': num_edges * 4,
        'divisions': num_edges * 4,
        'logs': num_edges * 1,
    }

    return total_ops


def count_pruning_final_ops(num_edges, num_root_edges):
    """
    Final likelihood computation for pruning algorithm (lines 253-282):
    """
    ops = {
        'additions': num_edges,  # sum all log scales
        'multiplications': num_root_edges * 4 + 4,  # combine root edges + root_probs * combined
        'comparisons': 3 + 1,  # fmax and if
        'divisions': 4,  # normalize combined
        'logs': 2,  # log(mx) and log(ll)
    }
    # Also: 3 additions for final ll sum
    ops['additions'] += 3
    return ops


def count_propagation_final_ops(num_edges):
    """
    Final likelihood computation for propagation algorithm (lines 323-345):
    """
    ops = {
        'additions': num_edges + 3 + 1,  # sum up_scales + dot product (3 adds) + 1 for down_scales
        'multiplications': 4,  # dot product: down * up
        'comparisons': 1,  # fmax (actually just ensuring not zero)
        'divisions': 0,
        'logs': 1,  # log(mx)
    }
    return ops


def main():
    # From test results: 73 edges, 3 root edges
    num_edges = 73
    num_root_edges = 3

    print("=" * 70)
    print("DETAILED OPERATION COUNT ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {num_edges} edges, {num_root_edges} root edges")
    print()

    # Forward pass (shared by both)
    fwd_ops = count_forward_pass_operations(num_edges, num_root_edges)
    print("FORWARD PASS (shared by both algorithms):")
    for op, count in fwd_ops.items():
        print(f"  {op:15s}: {count:6d}")
    print()

    # Backward pass (propagation only)
    bwd_ops = count_backward_pass_operations(num_edges, num_root_edges)
    print("BACKWARD PASS (propagation only):")
    for op, count in bwd_ops.items():
        print(f"  {op:15s}: {count:6d}")
    print()

    # Final likelihood computation
    pruning_final = count_pruning_final_ops(num_edges, num_root_edges)
    prop_final = count_propagation_final_ops(num_edges)

    print("PRUNING FINAL LIKELIHOOD:")
    for op, count in pruning_final.items():
        print(f"  {op:15s}: {count:6d}")
    print()

    print("PROPAGATION FINAL LIKELIHOOD:")
    for op, count in prop_final.items():
        print(f"  {op:15s}: {count:6d}")
    print()

    # Total operations
    print("=" * 70)
    print("TOTAL OPERATIONS PER PATTERN")
    print("=" * 70)

    print("\nPRUNING ALGORITHM:")
    pruning_total = {}
    for op in fwd_ops:
        pruning_total[op] = fwd_ops.get(op, 0) + pruning_final.get(op, 0)
    for op, count in pruning_total.items():
        print(f"  {op:15s}: {count:6d}")

    print("\nPROPAGATION ALGORITHM:")
    prop_total = {}
    for op in set(list(fwd_ops.keys()) + list(bwd_ops.keys()) + list(prop_final.keys())):
        prop_total[op] = fwd_ops.get(op, 0) + bwd_ops.get(op, 0) + prop_final.get(op, 0)
    for op, count in sorted(prop_total.items()):
        print(f"  {op:15s}: {count:6d}")

    # Compute ratios
    print("\n" + "=" * 70)
    print("OPERATION RATIOS (Propagation / Pruning)")
    print("=" * 70)

    pruning_basic = (pruning_total.get('additions', 0) +
                     pruning_total.get('multiplications', 0) +
                     pruning_total.get('divisions', 0) +
                     pruning_total.get('comparisons', 0))
    pruning_logs = pruning_total.get('logs', 0)

    prop_basic = (prop_total.get('additions', 0) +
                  prop_total.get('multiplications', 0) +
                  prop_total.get('divisions', 0) +
                  prop_total.get('comparisons', 0))
    prop_logs = prop_total.get('logs', 0)

    print(f"\nPruning:")
    print(f"  Basic ops (add/mul/div/cmp): {pruning_basic}")
    print(f"  Log operations:              {pruning_logs}")

    print(f"\nPropagation:")
    print(f"  Basic ops (add/mul/div/cmp): {prop_basic}")
    print(f"  Log operations:              {prop_logs}")

    print(f"\nRatios:")
    print(f"  Basic ops ratio:    {prop_basic}/{pruning_basic} = {prop_basic/pruning_basic:.3f}")
    print(f"  Log ops ratio:      {prop_logs}/{pruning_logs} = {prop_logs/pruning_logs:.3f}")

    # Weighted estimate (log ops cost ~20x basic ops)
    log_cost_factor = 20
    pruning_weighted = pruning_basic + pruning_logs * log_cost_factor
    prop_weighted = prop_basic + prop_logs * log_cost_factor

    print(f"\n  Weighted total (log={log_cost_factor}x):")
    print(f"    Pruning:     {pruning_weighted}")
    print(f"    Propagation: {prop_weighted}")
    print(f"    Ratio:       {prop_weighted/pruning_weighted:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("Propagation requires ~2x the operations of Pruning because:")
    print("  - Forward pass: shared (same operations)")
    print("  - Backward pass: ~same cost as forward pass")
    print("  - Final computation: propagation is simpler, but negligible")
    print(f"\nExpected time ratio: ~{prop_weighted/pruning_weighted:.2f}x")


if __name__ == "__main__":
    main()
