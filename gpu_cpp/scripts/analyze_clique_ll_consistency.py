#!/usr/bin/env python3
"""
Analyze clique log-likelihood consistency test results.

Creates visualization showing:
- LL difference vs distance between cliques (scatter plot)
- Summary statistics by distance
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 7),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_results(filepath):
    """Parse the clique_ll_consistency_test output file."""
    edge_lls = {}
    distance_diff_pairs = []
    stats_by_distance = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_edge_ll = False
    in_distance_diff = False
    in_stats_summary = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Parse edge/clique LL values section (supports both formats)
        if line.startswith('=== Edge LL Values ===') or line.startswith('=== Clique LL Values ==='):
            in_edge_ll = True
            in_distance_diff = False
            in_stats_summary = False
            continue

        if in_edge_ll and (line.startswith('EdgeID') or line.startswith('CliqueID')):
            continue

        if in_edge_ll and line:
            parts = line.split()
            # Handle both old format (EdgeID LL) and new format (CliqueID X_node Y_node LL)
            if len(parts) == 2:
                try:
                    edge_id = int(parts[0])
                    ll_value = float(parts[1])
                    edge_lls[edge_id] = ll_value
                except ValueError:
                    in_edge_ll = False
            elif len(parts) == 4:
                try:
                    clique_id = int(parts[0])
                    ll_value = float(parts[3])
                    edge_lls[clique_id] = ll_value
                except ValueError:
                    in_edge_ll = False

        # Parse distance vs LL difference section
        if line.startswith('=== Distance vs LL Difference ==='):
            in_edge_ll = False
            in_distance_diff = True
            in_stats_summary = False
            continue

        if in_distance_diff and line.startswith('Edge1'):
            continue

        if in_distance_diff and line and not line.startswith('==='):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    e1 = int(parts[0])
                    e2 = int(parts[1])
                    dist = int(parts[2])
                    diff = float(parts[3])
                    distance_diff_pairs.append((dist, diff))
                except ValueError:
                    pass

        # Parse statistics by distance section
        if line.startswith('=== LL Difference Statistics by Distance ==='):
            in_edge_ll = False
            in_distance_diff = False
            in_stats_summary = True
            continue

        if in_stats_summary and line.startswith('Distance'):
            continue

        if in_stats_summary and line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    dist = int(parts[0])
                    count = int(parts[1])
                    mean_diff = float(parts[2])
                    max_diff = float(parts[3])
                    std_diff = float(parts[4])
                    stats_by_distance[dist] = {
                        'count': count,
                        'mean': mean_diff,
                        'max': max_diff,
                        'std': std_diff
                    }
                except ValueError:
                    pass

    return edge_lls, distance_diff_pairs, stats_by_distance


def create_scatter_plot(distance_diff_pairs, output_file):
    """Create scatter plot of LL difference vs clique distance."""
    if not distance_diff_pairs:
        print("No distance-difference pairs found, skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    distances = [p[0] for p in distance_diff_pairs]
    differences = [p[1] for p in distance_diff_pairs]

    # Scatter plot
    ax.scatter(distances, differences, alpha=0.3, s=10, color='steelblue', edgecolors='none')

    ax.set_xlabel('Distance Between Cliques (number of edges)')
    ax.set_ylabel('Absolute Difference in Log-Likelihood')
    ax.set_title('Log-Likelihood Difference vs. Distance Between Cliques')

    # Set y-axis to log scale if differences are small
    if max(differences) > 0 and max(differences) < 1e-6:
        ax.set_yscale('log')
        ax.set_ylabel('Absolute Difference in Log-Likelihood (log scale)')

    # Add statistics text
    mean_diff = np.mean(differences)
    max_diff = np.max(differences)
    std_diff = np.std(differences)

    stats_text = (f'Total pairs: {len(differences)}\n'
                 f'Mean diff: {mean_diff:.4e}\n'
                 f'Max diff: {max_diff:.4e}\n'
                 f'Std diff: {std_diff:.4e}')
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_boxplot_by_distance(distance_diff_pairs, output_file):
    """Create box plot of LL differences grouped by distance."""
    if not distance_diff_pairs:
        print("No distance-difference pairs found, skipping box plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Group differences by distance
    data_by_distance = {}
    for dist, diff in distance_diff_pairs:
        if dist not in data_by_distance:
            data_by_distance[dist] = []
        data_by_distance[dist].append(diff)

    # Prepare data for box plot
    distances = sorted(data_by_distance.keys())
    data = [data_by_distance[d] for d in distances]

    # Create box plot
    bp = ax.boxplot(data, positions=distances, widths=0.6, patch_artist=True)

    # Style the box plot
    for box in bp['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)

    ax.set_xlabel('Distance Between Cliques (number of edges)')
    ax.set_ylabel('Absolute Difference in Log-Likelihood')
    ax.set_title('Distribution of Log-Likelihood Differences by Clique Distance')

    # Set y-axis to log scale if differences are small
    all_diffs = [d for _, d in distance_diff_pairs]
    if max(all_diffs) > 0 and max(all_diffs) < 1e-6:
        ax.set_yscale('log')
        ax.set_ylabel('Absolute Difference in Log-Likelihood (log scale)')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_mean_diff_plot(stats_by_distance, output_file):
    """Create line plot of mean LL difference vs distance."""
    if not stats_by_distance:
        print("No statistics by distance found, skipping mean difference plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    distances = sorted(stats_by_distance.keys())
    means = [stats_by_distance[d]['mean'] for d in distances]
    maxes = [stats_by_distance[d]['max'] for d in distances]
    stds = [stats_by_distance[d]['std'] for d in distances]

    # Plot mean with error bars
    ax.errorbar(distances, means, yerr=stds, fmt='o-', capsize=5,
                color='steelblue', linewidth=2, markersize=8,
                label='Mean ± Std Dev')

    # Plot max as separate line
    ax.plot(distances, maxes, 's--', color='red', linewidth=1.5, markersize=6,
            alpha=0.7, label='Max')

    ax.set_xlabel('Distance Between Cliques (number of edges)')
    ax.set_ylabel('Absolute Difference in Log-Likelihood')
    ax.set_title('Mean Log-Likelihood Difference vs. Clique Distance')
    ax.legend()

    # Set y-axis to log scale if differences are small
    if max(maxes) > 0 and max(maxes) < 1e-6:
        ax.set_yscale('log')
        ax.set_ylabel('Absolute Difference in Log-Likelihood (log scale)')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_edge_ll_distribution_plot(edge_lls, output_file):
    """Create histogram of edge LL values."""
    if not edge_lls:
        print("No edge LL values found, skipping distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    lls = list(edge_lls.values())

    mean_ll = np.mean(lls)
    std_ll = np.std(lls)
    min_ll = np.min(lls)
    max_ll = np.max(lls)
    data_range = max_ll - min_ll

    # Create histogram
    num_bins = 30
    if data_range > 0:
        padding = 0.1 * data_range
        bin_edges = np.linspace(min_ll - padding, max_ll + padding, num_bins + 1)
    else:
        bin_edges = np.linspace(mean_ll - 1e-10, mean_ll + 1e-10, num_bins + 1)

    ax.hist(lls, bins=bin_edges, density=False, alpha=0.7,
           color='forestgreen', edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Log-Likelihood Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Log-Likelihood Values Across All Cliques')

    # Add statistics text
    stats_text = (f'Num cliques: {len(lls)}\n'
                 f'Mean: {mean_ll:.10f}\n'
                 f'Std Dev: {std_ll:.4e}\n'
                 f'Range: {data_range:.4e}')
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_clique_ll_consistency.py <results_file>")
        print("Example: python analyze_clique_ll_consistency.py results/clique_ll_consistency.txt")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)

    # Parse results
    print(f"Parsing results from {results_file}...")
    edge_lls, distance_diff_pairs, stats_by_distance = parse_results(results_file)

    print(f"Found {len(edge_lls)} edge LL values")
    print(f"Found {len(distance_diff_pairs)} distance-difference pairs")
    print(f"Found statistics for {len(stats_by_distance)} distance levels")

    # Calculate overall statistics
    if edge_lls:
        lls = list(edge_lls.values())
        print("\n=== Edge LL Statistics ===")
        print(f"  Mean:     {np.mean(lls):.10f}")
        print(f"  Std Dev:  {np.std(lls):.10e}")
        print(f"  Range:    {np.min(lls):.10f} to {np.max(lls):.10f}")
        print(f"  Max diff: {np.max(lls) - np.min(lls):.10e}")

    if distance_diff_pairs:
        diffs = [d[1] for d in distance_diff_pairs]
        print("\n=== Distance-Difference Pair Statistics ===")
        print(f"  Total pairs: {len(diffs)}")
        print(f"  Mean diff:   {np.mean(diffs):.10e}")
        print(f"  Max diff:    {np.max(diffs):.10e}")
        print(f"  Std diff:    {np.std(diffs):.10e}")

    # Create output directory
    results_path = Path(results_file)
    if results_path.parent.name == "results":
        output_dir = results_path.parent.parent / "figs"
    else:
        output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True)

    base_name = Path(results_file).stem

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Scatter plot of distance vs LL difference
    if distance_diff_pairs:
        create_scatter_plot(
            distance_diff_pairs,
            output_dir / f"{base_name}_scatter.png"
        )

    # Plot 2: Box plot by distance
    if distance_diff_pairs:
        create_boxplot_by_distance(
            distance_diff_pairs,
            output_dir / f"{base_name}_boxplot.png"
        )

    # Plot 3: Mean difference vs distance
    if stats_by_distance:
        create_mean_diff_plot(
            stats_by_distance,
            output_dir / f"{base_name}_mean_diff.png"
        )

    # Plot 4: Edge LL distribution
    if edge_lls:
        create_edge_ll_distribution_plot(
            edge_lls,
            output_dir / f"{base_name}_ll_dist.png"
        )

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
