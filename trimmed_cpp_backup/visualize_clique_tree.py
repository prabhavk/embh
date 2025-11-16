#!/usr/bin/env python3
"""
Visualize clique tree with nodes colored/sized by memoization efficiency.
"""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from collections import defaultdict

def load_cache_statistics(csv_file='cache_statistics.csv'):
    """Load cache statistics for each edge."""
    upward_stats = {}
    downward_stats = {}
    # Also load as ordered list for index-based matching
    upward_list = []
    downward_list = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clique = row['clique_edge']
            stats = {
                'hit_rate': float(row['hit_rate']),
                'subtree_size': int(row['subtree_size']),
                'complement_size': int(row['complement_size'])
            }
            if row['direction'] == 'upward':
                upward_stats[clique] = stats
                upward_list.append(stats)
            else:  # downward
                downward_stats[clique] = stats
                downward_list.append(stats)

    return upward_stats, upward_list, downward_stats, downward_list

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

def root_tree(adj, root):
    """Root the tree at a given node, return parent/children relationships."""
    parent = {root: None}
    children = defaultdict(list)
    queue = [root]
    visited = {root}
    order = [root]

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)
                order.append(neighbor)

    return parent, children, order

def compute_hierarchical_layout(parent, children, root):
    """
    Compute hierarchical tree layout using Reingold-Tilford-like algorithm.
    Returns positions for each node.
    """
    # Compute depth for each node
    depth = {root: 0}
    queue = [root]
    max_depth = 0

    while queue:
        node = queue.pop(0)
        for child in children[node]:
            depth[child] = depth[node] + 1
            max_depth = max(max_depth, depth[child])
            queue.append(child)

    # Compute x-positions using post-order traversal
    x_pos = {}
    next_x = [0]  # Mutable counter

    def assign_x(node):
        if not children[node]:
            # Leaf node
            x_pos[node] = next_x[0]
            next_x[0] += 1
        else:
            # Internal node - center over children
            for child in children[node]:
                assign_x(child)
            child_xs = [x_pos[c] for c in children[node]]
            x_pos[node] = sum(child_xs) / len(child_xs)

    assign_x(root)

    # Create positions dictionary
    positions = {}
    for node in depth:
        positions[node] = (x_pos[node], -depth[node])

    return positions

def get_clique_name(parent_node, child_node, upward_stats):
    """Get the clique name for an edge, checking both orderings."""
    name1 = f"{parent_node}-{child_node}"
    name2 = f"{child_node}-{parent_node}"

    if name1 in upward_stats:
        return name1
    elif name2 in upward_stats:
        return name2
    else:
        return name1  # Default

def assign_stats_to_edges(parent, children, root, upward_list):
    """
    Assign cache statistics to tree edges based on post-order traversal.
    The CSV outputs stats in post-order, so we match by traversal order.
    """
    edge_stats = {}
    stat_idx = [0]  # Mutable counter

    def traverse_postorder(node):
        for child in children[node]:
            traverse_postorder(child)
            # After visiting all children, this edge (child->node) gets next stat
            if stat_idx[0] < len(upward_list):
                edge_stats[(child, node)] = upward_list[stat_idx[0]]
                stat_idx[0] += 1

    traverse_postorder(root)
    return edge_stats

def plot_tree_with_colors(parent, children, root, stats_dict, direction='upward', filename='clique_tree_colored.png'):
    """
    Plot the SEM tree with edges colored by memoization efficiency.
    Each edge represents a clique, colored by its hit rate.
    """
    positions = compute_hierarchical_layout(parent, children, root)

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    # Create colormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('efficiency', colors)

    # Draw edges with colors based on hit rate
    edge_colors = []
    edge_widths = []

    matched_edges = 0
    unmatched_edges = 0

    for node in parent:
        if parent[node] is not None:
            p = parent[node]
            x1, y1 = positions[p]
            x2, y2 = positions[node]

            # Get clique stats by direct name matching
            clique_name = get_clique_name(p, node, stats_dict)
            if clique_name in stats_dict:
                hit_rate = stats_dict[clique_name]['hit_rate']
                subtree_size = stats_dict[clique_name]['subtree_size']
                matched_edges += 1
            else:
                hit_rate = 50.0
                subtree_size = 0
                unmatched_edges += 1

            # Color and width based on hit rate
            color = cmap(hit_rate / 100.0)
            width = 2 + hit_rate / 20.0  # Width from 2 to 7

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, solid_capstyle='round', zorder=1)

            # Add label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label = f"{hit_rate:.0f}%"
            ax.text(mid_x + 0.1, mid_y, label, fontsize=6, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    print(f"  {direction.capitalize()} - Matched edges: {matched_edges}, Unmatched: {unmatched_edges}")

    # Draw nodes
    for node in positions:
        x, y = positions[node]
        if node.startswith('h_'):
            # Internal node
            circle = plt.Circle((x, y), 0.3, facecolor='lightgray', edgecolor='black', linewidth=1, zorder=2)
        else:
            # Observed (leaf) node
            circle = plt.Circle((x, y), 0.3, facecolor='lightblue', edgecolor='black', linewidth=1, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=5, zorder=3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Cache Hit Rate (%)', shrink=0.8)

    # Set axis properties
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Phylogenetic Tree: Edge Color = Memoization Efficiency\n'
                '(Green = High Hit Rate, Red = Low Hit Rate)\n'
                f'Root: {root}',
                fontsize=14, pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=cmap(0.99), edgecolor='black', label='High (>95%)'),
        mpatches.Patch(facecolor=cmap(0.75), edgecolor='black', label='Good (75-95%)'),
        mpatches.Patch(facecolor=cmap(0.50), edgecolor='black', label='Medium (50-75%)'),
        mpatches.Patch(facecolor=cmap(0.25), edgecolor='black', label='Low (25-50%)'),
        mpatches.Patch(facecolor=cmap(0.01), edgecolor='black', label='Very Low (<25%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Hit Rate', fontsize=9)

    # Update title based on direction
    dir_label = "Upward (Leaves to Root)" if direction == 'upward' else "Downward (Root to Leaves)"
    ax.set_title(f'Phylogenetic Tree: {dir_label} Message Hit Rates\n'
                '(Green = High Hit Rate, Red = Low Hit Rate)\n'
                f'Root: {root}',
                fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_tree_with_sizes(parent, children, root, stats_dict, direction='upward', filename='clique_tree_sized.png'):
    """
    Plot the SEM tree with edge thickness proportional to memoization efficiency.
    Thicker edges = higher hit rate.
    """
    positions = compute_hierarchical_layout(parent, children, root)

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    # Draw edges with thickness based on hit rate
    for node in parent:
        if parent[node] is not None:
            p = parent[node]
            x1, y1 = positions[p]
            x2, y2 = positions[node]

            # Get clique stats by direct name matching
            clique_name = get_clique_name(p, node, stats_dict)
            if clique_name in stats_dict:
                hit_rate = stats_dict[clique_name]['hit_rate']
                subtree_size = stats_dict[clique_name]['subtree_size']
            else:
                hit_rate = 50.0
                subtree_size = 0

            # Width based on hit rate (1 to 12)
            width = 1 + hit_rate / 10.0

            # Color by subtree size
            if subtree_size <= 1:
                color = '#2166ac'  # Dark blue
            elif subtree_size <= 3:
                color = '#4393c3'  # Blue
            elif subtree_size <= 6:
                color = '#92c5de'  # Light blue
            elif subtree_size <= 10:
                color = '#fddbc7'  # Light orange
            elif subtree_size <= 20:
                color = '#f4a582'  # Orange
            else:
                color = '#d6604d'  # Red

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                   solid_capstyle='round', alpha=0.8, zorder=1)

            # Add subtree size label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label = f"s={subtree_size}"
            ax.text(mid_x + 0.1, mid_y, label, fontsize=5, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    # Draw nodes
    for node in positions:
        x, y = positions[node]
        if node.startswith('h_'):
            circle = plt.Circle((x, y), 0.3, facecolor='lightgray', edgecolor='black', linewidth=1, zorder=2)
        else:
            circle = plt.Circle((x, y), 0.3, facecolor='lightblue', edgecolor='black', linewidth=1, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=5, zorder=3)

    # Set axis properties
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Phylogenetic Tree: Edge Thickness = Memoization Efficiency\n'
                '(Thicker = Higher Hit Rate, Color = Subtree Size)\n'
                f'Root: {root}',
                fontsize=14, pad=20)

    # Add legend for colors (subtree size)
    legend_elements = [
        mpatches.Patch(facecolor='#2166ac', edgecolor='black', label='Size 1 (leaf)'),
        mpatches.Patch(facecolor='#4393c3', edgecolor='black', label='Size 2-3'),
        mpatches.Patch(facecolor='#92c5de', edgecolor='black', label='Size 4-6'),
        mpatches.Patch(facecolor='#fddbc7', edgecolor='black', label='Size 7-10'),
        mpatches.Patch(facecolor='#f4a582', edgecolor='black', label='Size 11-20'),
        mpatches.Patch(facecolor='#d6604d', edgecolor='black', label='Size >20'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Subtree Size', fontsize=9)

    # Add thickness reference
    ref_y = min(all_y) - 2
    ref_x_start = min(all_x) + 2
    ref_widths = [2, 5, 8, 11]
    ref_rates = [10, 40, 70, 100]

    for i, (w, rate) in enumerate(zip(ref_widths, ref_rates)):
        x = ref_x_start + i * 3
        ax.plot([x, x + 1.5], [ref_y, ref_y], color='gray', linewidth=w)
        ax.text(x + 0.75, ref_y - 0.5, f'{rate}%', ha='center', va='top', fontsize=8)

    ax.text(ref_x_start + 4.5, ref_y - 1.2, 'Thickness Reference (Hit Rate)', ha='center', fontsize=10)

    # Update title based on direction
    dir_label = "Upward (Leaves to Root)" if direction == 'upward' else "Downward (Root to Leaves)"
    ax.set_title(f'Phylogenetic Tree: {dir_label} Message Hit Rates\n'
                '(Thicker = Higher Hit Rate, Color = Subtree Size)\n'
                f'Root: {root}',
                fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("Loading cache statistics...")
    upward_stats, upward_list, downward_stats, downward_list = load_cache_statistics()
    print(f"Loaded stats for {len(upward_stats)} upward and {len(downward_stats)} downward messages")

    # Show hit rate range
    if upward_list:
        up_rates = [s['hit_rate'] for s in upward_list]
        print(f"  Upward hit rate range: {min(up_rates):.1f}% to {max(up_rates):.1f}%")
    if downward_list:
        down_rates = [s['hit_rate'] for s in downward_list]
        print(f"  Downward hit rate range: {min(down_rates):.1f}% to {max(down_rates):.1f}%")

    print("\nBuilding tree structure...")
    adj, vertices, edges = build_tree_from_edgelist('data/RAxML_bipartitions.CDS_FcC_partition.edgelist')
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {len(edges)}")

    # Root the tree at h_0 (current root)
    root = 'h_0'
    print(f"\nRooting tree at {root}...")
    parent, children, order = root_tree(adj, root)

    # Count edges in rooted tree
    num_edges = sum(1 for n in parent if parent[n] is not None)
    print(f"  Edges in rooted tree: {num_edges}")
    print(f"  Upward stats available: {len(upward_list)}")
    print(f"  Downward stats available: {len(downward_list)}")

    print(f"\nGenerating upward message visualizations...")
    plot_tree_with_colors(parent, children, root, upward_stats, 'upward', 'clique_tree_upward_colored.png')
    plot_tree_with_sizes(parent, children, root, upward_stats, 'upward', 'clique_tree_upward_sized.png')

    print(f"\nGenerating downward message visualizations...")
    plot_tree_with_colors(parent, children, root, downward_stats, 'downward', 'clique_tree_downward_colored.png')
    plot_tree_with_sizes(parent, children, root, downward_stats, 'downward', 'clique_tree_downward_sized.png')

    print("\nDone!")

if __name__ == "__main__":
    main()
