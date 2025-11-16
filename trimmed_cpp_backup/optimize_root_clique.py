#!/usr/bin/env python3
"""
Optimize root clique selection for maximum memoization efficiency.
Finds the root clique that maximizes the number of edges with cache hit rate > threshold.
Uses ete3 for unrooted tree visualization.
"""

import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def build_tree_from_edgelist(edgelist_file):
    """Build tree structure from edge list file."""
    edges = []
    vertices = set()
    edge_lengths = {}

    with open(edgelist_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                v1, v2 = parts[0], parts[1]
                edges.append((v1, v2))
                vertices.add(v1)
                vertices.add(v2)
                if len(parts) >= 3:
                    edge_lengths[(v1, v2)] = float(parts[2])
                    edge_lengths[(v2, v1)] = float(parts[2])

    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    return adj, list(vertices), edges, edge_lengths

def root_clique_tree(adj, root_clique_vertex):
    """
    Root the clique tree at a clique containing root_clique_vertex.
    Returns parent/children relationships for the clique tree.
    """
    parent = {root_clique_vertex: None}
    children = defaultdict(list)
    queue = [root_clique_vertex]
    visited = {root_clique_vertex}

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
    """
    Find the edge that corresponds to the root clique.
    The root clique is the clique that receives all upward messages first.
    It corresponds to the edge incident to the root vertex that leads to the largest subtree.
    """
    neighbors = adj[root_vertex]
    if not neighbors:
        return None, None

    # Find the neighbor that leads to the largest subtree
    def count_subtree(node, parent):
        count = 1
        for neighbor in adj[node]:
            if neighbor != parent:
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

def compute_subtree_sizes(adj, root, observed_vertices):
    """
    Compute subtree sizes for each edge when tree is rooted at 'root'.
    Returns: edge_sizes dict mapping (child, parent) -> subtree_size
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

    # For each non-root node, the upward message has subtree_size[node] leaves
    # The downward message has (total - subtree_size[node]) leaves
    total_observed = len(observed_vertices)
    edge_sizes = {}

    for node in parent:
        if parent[node] is not None:
            edge_sizes[(node, parent[node])] = {
                'upward_size': subtree_size[node],
                'downward_size': total_observed - subtree_size[node]
            }

    return edge_sizes, parent, children

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

def evaluate_root_choice(adj, root, observed_vertices, alpha=90.0):
    """
    Evaluate a root choice based on number of edges with hit rate > alpha.

    Returns:
    - count of upward edges with hit rate > alpha
    - count of downward edges with hit rate > alpha
    - total count
    - average hit rate
    """
    edge_sizes, parent, children = compute_subtree_sizes(adj, root, observed_vertices)

    upward_above_threshold = 0
    downward_above_threshold = 0
    upward_rates = []
    downward_rates = []

    for edge, sizes in edge_sizes.items():
        up_rate = estimate_hit_rate(sizes['upward_size'])
        down_rate = estimate_hit_rate(sizes['downward_size'])

        upward_rates.append(up_rate)
        downward_rates.append(down_rate)

        if up_rate > alpha:
            upward_above_threshold += 1
        if down_rate > alpha:
            downward_above_threshold += 1

    return {
        'root': root,
        'upward_above_alpha': upward_above_threshold,
        'downward_above_alpha': downward_above_threshold,
        'total_above_alpha': upward_above_threshold + downward_above_threshold,
        'avg_upward_rate': sum(upward_rates) / len(upward_rates) if upward_rates else 0,
        'avg_downward_rate': sum(downward_rates) / len(downward_rates) if downward_rates else 0,
        'total_edges': len(edge_sizes)
    }

def find_optimal_root_clique(edgelist_file, alpha=90.0):
    """
    Find the optimal root clique that maximizes edges with hit rate > alpha.
    """
    adj, vertices, edges, edge_lengths = build_tree_from_edgelist(edgelist_file)

    # Identify observed vs internal vertices
    observed = set()
    internal = set()
    for v in vertices:
        if v.startswith('h_'):
            internal.add(v)
        else:
            observed.add(v)

    print(f"Tree structure:")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Internal nodes: {len(internal)}")
    print(f"  Observed (taxa): {len(observed)}")
    print(f"  Total edges: {len(edges)}")
    print(f"  Threshold alpha: {alpha}%")

    # Evaluate each internal node as potential root clique
    results = []
    for root in internal:
        result = evaluate_root_choice(adj, root, observed, alpha)
        results.append(result)

    # Sort by total edges above threshold, then by avg upward rate
    results.sort(key=lambda x: (x['total_above_alpha'], x['avg_upward_rate']), reverse=True)

    return results, adj, vertices, edges, edge_lengths, observed, internal

def print_optimization_results(results, alpha, adj):
    """Print detailed optimization results."""
    print(f"\n{'='*70}")
    print(f"ROOT CLIQUE OPTIMIZATION (alpha = {alpha}%)")
    print(f"{'='*70}")
    print(f"Note: Each root clique corresponds to an EDGE in the phylogenetic tree.")

    print(f"\nTOP 10 ROOT CLIQUE CHOICES (by root vertex, clique = incident edge):")
    print(f"{'Rank':<6} {'Root Vtx':<10} {'Upward>α':<12} {'Down>α':<12} {'Total>α':<12} {'Avg Up%':<10} {'Avg Down%'}")
    print("-"*70)
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['root']:<10} {r['upward_above_alpha']:<12} "
              f"{r['downward_above_alpha']:<12} {r['total_above_alpha']:<12} "
              f"{r['avg_upward_rate']:<10.1f} {r['avg_downward_rate']:.1f}")

    print(f"\nWORST 5 ROOT CLIQUE CHOICES:")
    print(f"{'Rank':<6} {'Root Vtx':<10} {'Upward>α':<12} {'Down>α':<12} {'Total>α':<12} {'Avg Up%':<10} {'Avg Down%'}")
    print("-"*70)
    for i, r in enumerate(results[-5:], len(results)-4):
        print(f"{i:<6} {r['root']:<10} {r['upward_above_alpha']:<12} "
              f"{r['downward_above_alpha']:<12} {r['total_above_alpha']:<12} "
              f"{r['avg_upward_rate']:<10.1f} {r['avg_downward_rate']:.1f}")

    best = results[0]
    worst = results[-1]

    # Get the actual root clique edges
    best_edge = find_root_clique_edge(adj, best['root'])
    worst_edge = find_root_clique_edge(adj, worst['root'])
    best_edge_str = f"{best_edge[0]}-{best_edge[1]}" if best_edge else "N/A"
    worst_edge_str = f"{worst_edge[0]}-{worst_edge[1]}" if worst_edge else "N/A"

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Best root clique edge: {best_edge_str} (rooted at {best['root']})")
    print(f"    - Upward edges with hit rate > {alpha}%: {best['upward_above_alpha']}/{best['total_edges']}")
    print(f"    - Downward edges with hit rate > {alpha}%: {best['downward_above_alpha']}/{best['total_edges']}")
    print(f"    - Total edges above threshold: {best['total_above_alpha']}/{2*best['total_edges']}")
    print(f"    - Average upward hit rate: {best['avg_upward_rate']:.1f}%")
    print(f"    - Average downward hit rate: {best['avg_downward_rate']:.1f}%")

    print(f"\n  Worst root clique edge: {worst_edge_str} (rooted at {worst['root']})")
    print(f"    - Total edges above threshold: {worst['total_above_alpha']}/{2*worst['total_edges']}")

    improvement = best['total_above_alpha'] - worst['total_above_alpha']
    print(f"\n  Improvement: {improvement} additional edges above {alpha}% threshold")

def plot_unrooted_tree_with_stats(adj, vertices, edges, edge_lengths, observed,
                                   root_clique_vertex, alpha=90.0, filename='unrooted_tree.png'):
    """
    Plot the unrooted phylogenetic tree with edges colored by hit rate.
    Uses a circular/radial layout for unrooted tree.
    Root clique is highlighted as an EDGE (not a node).
    """
    # Compute edge statistics for this root choice
    edge_sizes, parent, children = compute_subtree_sizes(adj, root_clique_vertex, observed)

    # Find the root clique edge
    root_clique_edge = find_root_clique_edge(adj, root_clique_vertex)

    # Create edge stats dictionary
    edge_stats = {}
    for (child, par), sizes in edge_sizes.items():
        up_rate = estimate_hit_rate(sizes['upward_size'])
        down_rate = estimate_hit_rate(sizes['downward_size'])
        # For unrooted view, show combined metric
        avg_rate = (up_rate + down_rate) / 2
        edge_stats[(child, par)] = {
            'upward_rate': up_rate,
            'downward_rate': down_rate,
            'avg_rate': avg_rate,
            'upward_size': sizes['upward_size'],
            'downward_size': sizes['downward_size']
        }
        # Also store reverse direction
        edge_stats[(par, child)] = edge_stats[(child, par)]

    # Use radial layout for unrooted tree
    positions = compute_radial_layout(adj, vertices, root_clique_vertex)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    # Create colormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('efficiency', colors)

    # Draw edges
    drawn_edges = set()
    root_edge_midpoint = None
    for v1, v2 in edges:
        if (v1, v2) not in drawn_edges and (v2, v1) not in drawn_edges:
            x1, y1 = positions[v1]
            x2, y2 = positions[v2]

            # Get stats for this edge
            if (v1, v2) in edge_stats:
                stats = edge_stats[(v1, v2)]
                avg_rate = stats['avg_rate']
                up_rate = stats['upward_rate']
                down_rate = stats['downward_rate']
            else:
                avg_rate = 50.0
                up_rate = 50.0
                down_rate = 50.0

            # Check if this is the root clique edge
            is_root_edge = ((v1, v2) == root_clique_edge or (v2, v1) == root_clique_edge)

            # Color based on average rate
            color = cmap(avg_rate / 100.0)
            width = 2 + avg_rate / 15.0

            if is_root_edge:
                # Draw root clique edge with special highlighting
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=width + 4,
                       solid_capstyle='round', zorder=0, alpha=0.4)
                root_edge_midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                   solid_capstyle='round', zorder=1, alpha=0.8)

            # Add label showing both rates
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Offset label perpendicular to edge
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                offset_x = -dy / length * 0.3
                offset_y = dx / length * 0.3
            else:
                offset_x = 0
                offset_y = 0.3

            label = f"↑{up_rate:.0f}% ↓{down_rate:.0f}%"
            bbox_color = 'yellow' if is_root_edge else 'white'
            ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=5,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor=bbox_color, alpha=0.7))

            drawn_edges.add((v1, v2))

    # Draw nodes
    for node in vertices:
        x, y = positions[node]
        if node.startswith('h_'):
            # Internal node - smaller
            circle = plt.Circle((x, y), 0.15, facecolor='lightgray',
                               edgecolor='black', linewidth=0.5, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=4, zorder=3)
        else:
            # Observed (leaf) node - larger with name
            circle = plt.Circle((x, y), 0.25, facecolor='lightblue',
                               edgecolor='black', linewidth=1, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=6,
                   fontweight='bold', zorder=3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Average Cache Hit Rate (%)', shrink=0.8)

    # Set axis properties
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # Count edges above threshold
    above_threshold = sum(1 for s in edge_stats.values() if s['avg_rate'] > alpha) // 2
    total_edges = len(edges)

    # Format root clique edge as string
    root_edge_str = f"{root_clique_edge[0]}-{root_clique_edge[1]}" if root_clique_edge else "N/A"

    ax.set_title(f'Unrooted Phylogenetic Tree\n'
                f'Root Clique Edge: {root_edge_str} (highlighted in red/yellow)\n'
                f'Edge Color = Average Hit Rate (↑upward ↓downward)\n'
                f'Edges with avg hit rate > {alpha}%: {above_threshold}/{total_edges}',
                fontsize=14, pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=cmap(0.99), edgecolor='black', label='High (>95%)'),
        mpatches.Patch(facecolor=cmap(0.75), edgecolor='black', label='Good (75-95%)'),
        mpatches.Patch(facecolor=cmap(0.50), edgecolor='black', label='Medium (50-75%)'),
        mpatches.Patch(facecolor=cmap(0.25), edgecolor='black', label='Low (25-50%)'),
        mpatches.Patch(facecolor=cmap(0.01), edgecolor='black', label='Very Low (<25%)'),
        mpatches.Patch(facecolor='yellow', edgecolor='red', linewidth=2, label='Root Clique Edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Avg Hit Rate', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def compute_radial_layout(adj, vertices, center):
    """
    Compute a radial (unrooted) layout for the tree.
    Places center at origin, arranges subtrees radially.
    """
    positions = {}
    visited = set()

    def layout_subtree(node, parent_node, angle_start, angle_end, radius):
        """Recursively layout a subtree within an angular sector."""
        positions[node] = (radius * np.cos((angle_start + angle_end) / 2),
                          radius * np.sin((angle_start + angle_end) / 2))
        visited.add(node)

        # Get unvisited neighbors
        neighbors = [n for n in adj[node] if n not in visited]
        if not neighbors:
            return

        # Count subtree sizes for proportional allocation
        subtree_sizes = []
        for neighbor in neighbors:
            size = count_subtree_size(neighbor, node)
            subtree_sizes.append(size)

        total_size = sum(subtree_sizes)
        if total_size == 0:
            total_size = len(neighbors)
            subtree_sizes = [1] * len(neighbors)

        # Allocate angular sectors proportionally
        angle_range = angle_end - angle_start
        current_angle = angle_start

        for i, neighbor in enumerate(neighbors):
            proportion = subtree_sizes[i] / total_size
            neighbor_angle_range = angle_range * proportion
            neighbor_angle_end = current_angle + neighbor_angle_range

            layout_subtree(neighbor, node, current_angle, neighbor_angle_end, radius + 2)
            current_angle = neighbor_angle_end

    def count_subtree_size(node, parent):
        """Count nodes in subtree rooted at node (excluding parent direction)."""
        count = 1
        for neighbor in adj[node]:
            if neighbor != parent:
                count += count_subtree_size(neighbor, node)
        return count

    # Import numpy for trigonometry
    import numpy as np

    # Place center node at origin
    positions[center] = (0, 0)
    visited.add(center)

    # Get neighbors of center
    neighbors = adj[center]
    if not neighbors:
        return positions

    # Count subtree sizes
    subtree_sizes = [count_subtree_size(n, center) for n in neighbors]
    total_size = sum(subtree_sizes)

    # Allocate angular sectors
    current_angle = 0
    for i, neighbor in enumerate(neighbors):
        proportion = subtree_sizes[i] / total_size
        angle_range = 2 * np.pi * proportion
        angle_end = current_angle + angle_range

        layout_subtree(neighbor, center, current_angle, angle_end, 2)
        current_angle = angle_end

    return positions

def main():
    import sys

    alpha = 90.0
    if len(sys.argv) > 1:
        try:
            alpha = float(sys.argv[1])
        except ValueError:
            pass

    print(f"Optimizing root clique selection (alpha = {alpha}%)...")

    results, adj, vertices, edges, edge_lengths, observed, internal = \
        find_optimal_root_clique('data/RAxML_bipartitions.CDS_FcC_partition.edgelist', alpha)

    print_optimization_results(results, alpha, adj)

    # Generate plots for best and worst root choices
    best_root = results[0]['root']
    worst_root = results[-1]['root']
    best_edge = find_root_clique_edge(adj, best_root)
    worst_edge = find_root_clique_edge(adj, worst_root)

    print(f"\nGenerating unrooted tree visualizations...")
    best_edge_str = f"{best_edge[0]}_{best_edge[1]}" if best_edge else "unknown"
    worst_edge_str = f"{worst_edge[0]}_{worst_edge[1]}" if worst_edge else "unknown"

    plot_unrooted_tree_with_stats(adj, vertices, edges, edge_lengths, observed,
                                   best_root, alpha, f'unrooted_tree_best_{best_edge_str}.png')
    plot_unrooted_tree_with_stats(adj, vertices, edges, edge_lengths, observed,
                                   worst_root, alpha, f'unrooted_tree_worst_{worst_edge_str}.png')

    # Also plot for current root (h_0)
    current_root = 'h_0'
    if current_root != best_root and current_root != worst_root:
        current_edge = find_root_clique_edge(adj, current_root)
        current_edge_str = f"{current_edge[0]}_{current_edge[1]}" if current_edge else "unknown"
        plot_unrooted_tree_with_stats(adj, vertices, edges, edge_lengths, observed,
                                       current_root, alpha, f'unrooted_tree_current_{current_edge_str}.png')

    # Save results to CSV with edge information
    with open('root_clique_optimization.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['root_vertex', 'root_clique_edge', 'upward_above_alpha', 'downward_above_alpha',
                        'total_above_alpha', 'avg_upward_rate', 'avg_downward_rate'])
        for r in results:
            edge = find_root_clique_edge(adj, r['root'])
            edge_str = f"{edge[0]}-{edge[1]}" if edge else "N/A"
            writer.writerow([r['root'], edge_str, r['upward_above_alpha'], r['downward_above_alpha'],
                           r['total_above_alpha'], f"{r['avg_upward_rate']:.2f}",
                           f"{r['avg_downward_rate']:.2f}"])

    print(f"\nDetailed results saved to: root_clique_optimization.csv")
    print("\nDone!")

if __name__ == "__main__":
    main()
