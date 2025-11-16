#!/usr/bin/env python3
"""
Analyze how root clique selection affects memoization efficiency.
For each potential root clique, compute the distribution of subtree sizes
and estimate the expected cache hit rate.
"""

import csv
from collections import defaultdict

def load_tree_structure(csv_file='cache_statistics.csv'):
    """
    Load the tree structure from cache statistics.
    Extract parent-child relationships from the data.
    """
    # Load current statistics to understand the tree structure
    edges = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['direction'] == 'upward':
                # Upward message: child sends to parent
                # clique_edge is the child clique
                edges.append({
                    'child': row['clique_edge'],
                    'subtree_size': int(row['subtree_size']),
                    'complement_size': int(row['complement_size'])
                })

    total_taxa = edges[0]['subtree_size'] + edges[0]['complement_size']
    return edges, total_taxa

def estimate_hit_rate(subtree_size, total_taxa=38):
    """
    Estimate cache hit rate based on subtree size.
    This is based on the empirical relationship observed in the data.

    Empirical observation from 648 patterns, 38 taxa:
    - Size 1: ~99.3% hit rate
    - Size 2: ~97.4% hit rate
    - Size 5: ~77.5% hit rate
    - Size 10: ~71.5% hit rate
    - Size 20: ~5% hit rate
    - Size 36: ~0.15% hit rate

    The probability of pattern match decreases exponentially with size.
    For k independent variables with ~4 states each:
    P(match) ≈ (1/4)^k for random patterns
    But DNA sequences are correlated, so actual hit rate is higher.
    """
    # Empirical model based on observed data
    # Using exponential decay model fit to the data
    if subtree_size == 0:
        return 100.0
    elif subtree_size == 1:
        return 99.3
    elif subtree_size <= 3:
        return 100 - 2.5 * subtree_size
    elif subtree_size <= 10:
        return 100 - 3.0 * subtree_size
    else:
        # Exponential decay for larger subtrees
        return max(0.1, 100.0 * (0.85 ** subtree_size))

def build_tree_from_edgelist(edgelist_file):
    """Build tree structure from edge list file."""
    edges = []
    vertices = set()

    with open(edgelist_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                v1, v2 = parts[0], parts[1]
                edges.append((v1, v2))
                vertices.add(v1)
                vertices.add(v2)

    # Build adjacency list
    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    return adj, list(vertices), edges

def compute_subtree_sizes_with_root(adj, root, observed_vertices):
    """
    Compute subtree sizes for each edge when tree is rooted at 'root'.

    For each internal node, compute how many observed variables (leaves)
    are in its subtree.
    """
    # First, root the tree using BFS from root
    parent = {root: None}
    children = defaultdict(list)
    queue = [root]
    visited = {root}

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)

    # Compute subtree sizes (number of observed variables in subtree)
    subtree_size = {}

    # Post-order traversal to compute subtree sizes
    def compute_size(node):
        if node in observed_vertices and len(children[node]) == 0:
            # This is an observed leaf
            subtree_size[node] = 1
        else:
            size = 0
            if node in observed_vertices:
                size = 1  # This node is observed
            for child in children[node]:
                size += compute_size(child)
            subtree_size[node] = size
        return subtree_size[node]

    compute_size(root)

    # For each non-root node, the message it sends has subtree_size[node] leaves
    edge_subtree_sizes = {}
    for node in visited:
        if parent[node] is not None:
            # Edge from node to parent[node]
            edge_subtree_sizes[(node, parent[node])] = subtree_size[node]

    return edge_subtree_sizes, subtree_size

def evaluate_root_choice(adj, root, observed_vertices, total_patterns=648):
    """
    Evaluate memoization efficiency for a given root choice.

    Returns:
    - Total expected cache hits
    - Average hit rate
    - Distribution of subtree sizes
    """
    edge_sizes, node_sizes = compute_subtree_sizes_with_root(adj, root, observed_vertices)

    total_messages = len(edge_sizes) * total_patterns
    expected_hits = 0
    hit_rates = []
    size_distribution = defaultdict(int)

    for edge, size in edge_sizes.items():
        hit_rate = estimate_hit_rate(size)
        hit_rates.append(hit_rate)
        size_distribution[size] += 1
        # Expected hits for this edge
        expected_hits += hit_rate / 100.0 * total_patterns

    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0

    return {
        'root': root,
        'total_edges': len(edge_sizes),
        'avg_hit_rate': avg_hit_rate,
        'expected_hits': expected_hits,
        'total_messages': total_messages,
        'size_distribution': dict(size_distribution),
        'edge_sizes': edge_sizes
    }

def analyze_all_roots(edgelist_file='data/RAxML_bipartitions.CDS_FcC_partition.edgelist'):
    """
    Analyze memoization efficiency for all possible root choices.
    """
    adj, vertices, edges = build_tree_from_edgelist(edgelist_file)

    # Identify observed vertices (leaves - those starting with capital letters usually)
    # From the data, we know there are 38 taxa
    observed = set()
    internal = set()

    for v in vertices:
        # Heuristic: internal nodes start with 'h_', observed are taxa names
        if v.startswith('h_'):
            internal.add(v)
        else:
            observed.add(v)

    print(f"Tree structure:")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Internal nodes: {len(internal)}")
    print(f"  Observed (taxa): {len(observed)}")
    print(f"  Total edges: {len(edges)}")

    # Evaluate each internal node as potential root
    results = []

    print("\nEvaluating all possible root positions...")
    for root in internal:
        result = evaluate_root_choice(adj, root, observed)
        results.append(result)

    # Sort by average hit rate
    results.sort(key=lambda x: x['avg_hit_rate'], reverse=True)

    return results, observed, internal

def print_analysis(results):
    """Print detailed analysis of root selection."""
    print("\n" + "="*70)
    print("ROOT SELECTION ANALYSIS FOR MEMOIZATION EFFICIENCY")
    print("="*70)

    print("\nTOP 10 ROOT CHOICES (highest average hit rate):")
    print(f"{'Rank':<6} {'Root':<10} {'Avg Hit Rate':<15} {'Expected Hits':<15} {'Edges'}")
    print("-"*60)
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['root']:<10} {r['avg_hit_rate']:>10.2f}%    "
              f"{r['expected_hits']:>12.0f}     {r['total_edges']}")

    print("\nWORST 10 ROOT CHOICES (lowest average hit rate):")
    print(f"{'Rank':<6} {'Root':<10} {'Avg Hit Rate':<15} {'Expected Hits':<15} {'Edges'}")
    print("-"*60)
    for i, r in enumerate(results[-10:], len(results)-9):
        print(f"{i:<6} {r['root']:<10} {r['avg_hit_rate']:>10.2f}%    "
              f"{r['expected_hits']:>12.0f}     {r['total_edges']}")

    # Detailed analysis of best and worst
    best = results[0]
    worst = results[-1]

    print("\n" + "="*70)
    print("DETAILED COMPARISON: BEST vs WORST ROOT")
    print("="*70)

    print(f"\nBEST ROOT: {best['root']}")
    print(f"  Average hit rate: {best['avg_hit_rate']:.2f}%")
    print(f"  Expected cache hits: {best['expected_hits']:.0f} / {best['total_messages']}")
    print(f"  Subtree size distribution:")
    for size in sorted(best['size_distribution'].keys()):
        count = best['size_distribution'][size]
        rate = estimate_hit_rate(size)
        print(f"    Size {size:2d}: {count:2d} edges  (est. hit rate: {rate:.1f}%)")

    print(f"\nWORST ROOT: {worst['root']}")
    print(f"  Average hit rate: {worst['avg_hit_rate']:.2f}%")
    print(f"  Expected cache hits: {worst['expected_hits']:.0f} / {worst['total_messages']}")
    print(f"  Subtree size distribution:")
    for size in sorted(worst['size_distribution'].keys()):
        count = worst['size_distribution'][size]
        rate = estimate_hit_rate(size)
        print(f"    Size {size:2d}: {count:2d} edges  (est. hit rate: {rate:.1f}%)")

    improvement = best['avg_hit_rate'] - worst['avg_hit_rate']
    hits_improvement = best['expected_hits'] - worst['expected_hits']

    print(f"\nIMPROVEMENT:")
    print(f"  Hit rate improvement: {improvement:.2f}%")
    print(f"  Additional cache hits: {hits_improvement:.0f}")
    print(f"  Relative improvement: {100*hits_improvement/worst['expected_hits']:.1f}%")

def compute_tree_center(adj, vertices):
    """
    Find the center of the tree (minimizes maximum distance to any node).
    The center typically maximizes memoization efficiency.
    """
    # Compute eccentricity of each vertex
    def bfs_distances(start):
        dist = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    eccentricity = {}
    for v in vertices:
        distances = bfs_distances(v)
        eccentricity[v] = max(distances.values())

    min_eccentricity = min(eccentricity.values())
    centers = [v for v in vertices if eccentricity[v] == min_eccentricity]

    return centers, eccentricity

def main():
    print("Analyzing root selection for memoization efficiency...")

    results, observed, internal = analyze_all_roots()

    print_analysis(results)

    # Also compute tree center
    adj, vertices, _ = build_tree_from_edgelist('data/RAxML_bipartitions.CDS_FcC_partition.edgelist')
    centers, eccentricity = compute_tree_center(adj, vertices)

    print("\n" + "="*70)
    print("TREE CENTER ANALYSIS")
    print("="*70)
    print(f"Tree center(s): {centers}")
    print(f"Center eccentricity: {eccentricity[centers[0]]}")

    # Find where current root (h_0) and center rank
    current_root_rank = next((i+1 for i, r in enumerate(results) if r['root'] == 'h_0'), None)
    center_ranks = [(c, next((i+1 for i, r in enumerate(results) if r['root'] == c), None))
                    for c in centers if c in internal]

    if current_root_rank:
        print(f"\nCurrent root (h_0) ranks: #{current_root_rank} out of {len(results)}")

    for center, rank in center_ranks:
        if rank:
            print(f"Tree center ({center}) ranks: #{rank} out of {len(results)}")

    # Save detailed results to file
    with open('root_selection_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['root', 'avg_hit_rate', 'expected_hits', 'total_messages',
                        'num_edges', 'small_subtrees_count'])
        for r in results:
            small_count = sum(c for s, c in r['size_distribution'].items() if s <= 3)
            writer.writerow([r['root'], f"{r['avg_hit_rate']:.2f}",
                           f"{r['expected_hits']:.0f}", r['total_messages'],
                           r['total_edges'], small_count])

    print("\nDetailed results saved to: root_selection_analysis.csv")

if __name__ == "__main__":
    main()
