#!/usr/bin/env python3
"""
Generate cache_hit_rate_analysis plots for different root clique choices.
Shows how subtree/complement sizes affect hit rates for upward and downward messages.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    return adj, list(vertices), edges

def root_clique_tree(adj, root):
    """Root the clique tree at given vertex."""
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

    return parent, children

def find_root_clique_edge(adj, root_vertex):
    """Find the edge that corresponds to the root clique."""
    neighbors = adj[root_vertex]
    if not neighbors:
        return None, None

    def count_subtree(node, par):
        count = 1
        for neighbor in adj[node]:
            if neighbor != par:
                count += count_subtree(neighbor, node)
        return count

    max_size = -1
    best_neighbor = None
    for neighbor in neighbors:
        size = count_subtree(neighbor, root_vertex)
        if size > max_size:
            max_size = size
            best_neighbor = neighbor

    return (root_vertex, best_neighbor)

def compute_edge_stats_for_root(adj, root, observed_vertices):
    """
    Compute subtree sizes for each edge when tree is rooted at 'root'.
    Returns list of dicts with upward and downward statistics.
    """
    parent, children = root_clique_tree(adj, root)

    # Compute subtree sizes (number of observed variables in subtree)
    subtree_size = {}

    def compute_size(node):
        if node in observed_vertices and len(children[node]) == 0:
            subtree_size[node] = 1
        else:
            size = 1 if node in observed_vertices else 0
            for child in children[node]:
                size += compute_size(child)
            subtree_size[node] = size
        return subtree_size[node]

    compute_size(root)

    total_observed = len(observed_vertices)

    upward_stats = []
    downward_stats = []

    for node in parent:
        if parent[node] is not None:
            up_size = subtree_size[node]
            down_size = total_observed - up_size

            upward_stats.append({
                'edge': f"{node}-{parent[node]}",
                'subtree_size': up_size,
                'complement_size': down_size,
                'hit_rate': estimate_hit_rate(up_size)
            })

            downward_stats.append({
                'edge': f"{parent[node]}-{node}",
                'subtree_size': up_size,
                'complement_size': down_size,
                'hit_rate': estimate_hit_rate(down_size)
            })

    return upward_stats, downward_stats

def estimate_hit_rate(size):
    """
    Estimate cache hit rate based on subtree/complement size.
    Based on empirical data from 38 taxa, 648 patterns.
    """
    if size == 0:
        return 100.0
    elif size == 1:
        return 99.3
    elif size <= 3:
        return 100 - 2.5 * size
    elif size <= 10:
        return 100 - 3.0 * size
    else:
        return max(0.1, 100.0 * (0.85 ** size))

def plot_cache_analysis(upward_stats, downward_stats, root_clique_edge, filename):
    """
    Create cache_hit_rate_analysis style plot for given root clique.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for upward messages
    up_sizes = [s['subtree_size'] for s in upward_stats]
    up_rates = [s['hit_rate'] for s in upward_stats]

    # Extract data for downward messages
    down_comp_sizes = [s['complement_size'] for s in downward_stats]
    down_rates = [s['hit_rate'] for s in downward_stats]

    # Plot 1: Upward hit rate vs subtree size
    ax1 = axes[0, 0]
    ax1.scatter(up_sizes, up_rates, alpha=0.7, s=50, c='blue', edgecolors='black')
    ax1.set_xlabel('Subtree Size (number of taxa)', fontsize=12)
    ax1.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax1.set_title('Upward Messages: Hit Rate vs Subtree Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add statistics
    mean_up = np.mean(up_rates)
    median_up = np.median(up_rates)
    ax1.text(0.05, 0.15, f'Mean: {mean_up:.1f}%\nMedian: {median_up:.1f}%',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Downward hit rate vs complement size
    ax2 = axes[0, 1]
    ax2.scatter(down_comp_sizes, down_rates, alpha=0.7, s=50, c='gold', edgecolors='black')
    ax2.set_xlabel('Complement Size (number of taxa)', fontsize=12)
    ax2.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax2.set_title('Downward Messages: Hit Rate vs Complement Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Add statistics
    mean_down = np.mean(down_rates)
    median_down = np.median(down_rates)
    ax2.text(0.05, 0.95, f'Mean: {mean_down:.1f}%\nMedian: {median_down:.1f}%',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Distribution of hit rates for upward messages
    ax3 = axes[1, 0]
    bins = np.arange(0, 105, 5)
    ax3.hist(up_rates, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(mean_up, color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {mean_up:.1f}%')
    ax3.set_xlabel('Cache Hit Rate (%)', fontsize=12)
    ax3.set_ylabel('Number of Clique Edges', fontsize=12)
    ax3.set_title('Distribution of Upward Message Hit Rates', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Count edges above thresholds
    above_90 = sum(1 for r in up_rates if r > 90)
    above_70 = sum(1 for r in up_rates if r > 70)
    ax3.text(0.05, 0.85, f'Edges > 90%: {above_90}/{len(up_rates)}\n'
                         f'Edges > 70%: {above_70}/{len(up_rates)}',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 4: Distribution of hit rates for downward messages
    ax4 = axes[1, 1]
    ax4.hist(down_rates, bins=bins, alpha=0.7, color='gold', edgecolor='black')
    ax4.axvline(mean_down, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_down:.1f}%')
    ax4.set_xlabel('Cache Hit Rate (%)', fontsize=12)
    ax4.set_ylabel('Number of Clique Edges', fontsize=12)
    ax4.set_title('Distribution of Downward Message Hit Rates', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Count edges above thresholds
    above_90_down = sum(1 for r in down_rates if r > 90)
    above_70_down = sum(1 for r in down_rates if r > 70)
    ax4.text(0.05, 0.85, f'Edges > 90%: {above_90_down}/{len(down_rates)}\n'
                         f'Edges > 70%: {above_70_down}/{len(down_rates)}',
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='khaki', alpha=0.5))

    # Overall title
    root_edge_str = f"{root_clique_edge[0]}-{root_clique_edge[1]}" if root_clique_edge else "N/A"
    fig.suptitle(f'Cache Hit Rate Analysis\n'
                 f'Root Clique Edge: {root_edge_str}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("Loading tree structure...")
    adj, vertices, edges = build_tree_from_edgelist('data/RAxML_bipartitions.CDS_FcC_partition.edgelist')

    # Identify observed vs internal vertices
    observed = set()
    for v in vertices:
        if not v.startswith('h_'):
            observed.add(v)

    print(f"  Total vertices: {len(vertices)}")
    print(f"  Observed (taxa): {len(observed)}")
    print(f"  Total edges: {len(edges)}")

    # Best root clique (from optimization analysis)
    best_root = 'h_13'
    best_edge = find_root_clique_edge(adj, best_root)
    print(f"\nGenerating analysis for BEST root clique: {best_edge[0]}-{best_edge[1]}")
    up_best, down_best = compute_edge_stats_for_root(adj, best_root, observed)
    plot_cache_analysis(up_best, down_best, best_edge, f'cache_analysis_best_{best_edge[0]}_{best_edge[1]}.png')

    # Worst root clique
    worst_root = 'h_28'
    worst_edge = find_root_clique_edge(adj, worst_root)
    print(f"Generating analysis for WORST root clique: {worst_edge[0]}-{worst_edge[1]}")
    up_worst, down_worst = compute_edge_stats_for_root(adj, worst_root, observed)
    plot_cache_analysis(up_worst, down_worst, worst_edge, f'cache_analysis_worst_{worst_edge[0]}_{worst_edge[1]}.png')

    # Current root clique (h_0)
    current_root = 'h_0'
    current_edge = find_root_clique_edge(adj, current_root)
    print(f"Generating analysis for CURRENT root clique: {current_edge[0]}-{current_edge[1]}")
    up_current, down_current = compute_edge_stats_for_root(adj, current_root, observed)
    plot_cache_analysis(up_current, down_current, current_edge, f'cache_analysis_current_{current_edge[0]}_{current_edge[1]}.png')

    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    configs = [
        ("Best (h_13-h_12)", up_best, down_best),
        ("Worst (h_28-h_27)", up_worst, down_worst),
        ("Current (h_0-h_1)", up_current, down_current)
    ]

    print(f"\n{'Config':<20} {'Up Mean':<10} {'Up >90%':<12} {'Down Mean':<10} {'Down >90%'}")
    print("-"*62)
    for name, up, down in configs:
        up_mean = np.mean([s['hit_rate'] for s in up])
        up_90 = sum(1 for s in up if s['hit_rate'] > 90)
        down_mean = np.mean([s['hit_rate'] for s in down])
        down_90 = sum(1 for s in down if s['hit_rate'] > 90)
        print(f"{name:<20} {up_mean:<10.1f} {up_90:<12} {down_mean:<10.1f} {down_90}")

    print("\nDone!")

if __name__ == "__main__":
    main()
