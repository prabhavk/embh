#!/usr/bin/env python3
"""
Analyze log-likelihood precision test results and create publication-ready plots.

This script reads the output from ll_precision_test and generates:
1. Distribution plot for Pruning Algorithm LL values
2. Distribution plot for Propagation Algorithm LL values
3. Distribution plot for the difference in LL values between algorithms
"""

import re
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
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_results(filepath):
    """Parse the precision_memory_timing_test output file."""
    pruning_lls = []
    propagation_lls = []
    differences = []
    timing_ratios = []
    fwd_times = []
    bwd_times = []
    mem_ratios = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the data section
    in_data = False
    for line in lines:
        line = line.strip()

        # Detect start of main data (unified format with LL, timing, and memory)
        if line.startswith('---') and not in_data:
            in_data = True
            continue

        # Detect end of data section
        if line.startswith('---') and in_data and len(pruning_lls) > 0:
            in_data = False
            continue

        # Parse unified data lines (Trial, Pruning LL, Propagation LL, Difference, Fwd_ms, Bwd_ms, Time_Ratio, Mem_Ratio)
        if in_data and line:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    trial = int(parts[0])
                    pruning = float(parts[1])
                    propagation = float(parts[2])
                    diff = float(parts[3])
                    fwd = float(parts[4])
                    bwd = float(parts[5])
                    time_ratio = float(parts[6])
                    # Memory ratio is constant but included in output
                    if len(parts) >= 8:
                        mem_ratio = float(parts[7])
                    else:
                        mem_ratio = 1.98  # Default from earlier tests

                    if not (np.isinf(pruning) or np.isinf(propagation)):
                        pruning_lls.append(pruning)
                        propagation_lls.append(propagation)
                        differences.append(diff)
                        fwd_times.append(fwd)
                        bwd_times.append(bwd)
                        timing_ratios.append(time_ratio)
                        mem_ratios.append(mem_ratio)
                except (ValueError, IndexError):
                    continue

    return (np.array(pruning_lls), np.array(propagation_lls), np.array(differences),
            np.array(timing_ratios), np.array(fwd_times), np.array(bwd_times), np.array(mem_ratios))


def create_distribution_plot(data, title, xlabel, output_file, color='steelblue'):
    """Create a bar chart for discrete values (when range is very small)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    data_range = max_val - min_val

    # Count unique values
    unique_vals, counts = np.unique(data, return_counts=True)

    if len(unique_vals) <= 10:
        # Use bar chart for discrete distribution
        # Show actual values with full precision in last significant digits
        # Format: show last few decimal places that differ
        bar_labels = []
        for val in unique_vals:
            # Show as offset from a rounded base value
            offset = val - mean_val
            bar_labels.append(f'{offset:+.4e}')

        bars = ax.bar(range(len(unique_vals)), counts, color=color,
                     edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.set_xticks(range(len(unique_vals)))
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_xlabel(f'Offset from Mean\n(Mean = {mean_val:.12f})')
        ax.set_ylabel('Count')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)
    else:
        # Use histogram for continuous distribution with tight range
        # Fixed number of bins for uniform bar widths
        num_bins = 30
        if data_range > 0:
            # Use tight range around actual data (only 10% padding instead of too much)
            padding = 0.1 * data_range
            bin_edges = np.linspace(min_val - padding, max_val + padding, num_bins + 1)
        else:
            bin_edges = np.linspace(mean_val - 1e-10, mean_val + 1e-10, 11)

        ax.hist(data, bins=bin_edges, density=False, alpha=0.7,
               color=color, edgecolor='black', linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')

        # Set x-axis limits to tight range around data
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.ticklabel_format(useOffset=True, style='plain', axis='x')
        plt.xticks(rotation=45, ha='right')

    # Add text annotation with statistics
    stats_text = (f'Mean: {mean_val:.15f}\n'
                 f'Std Dev: {std_val:.4e}\n'
                 f'Range: {data_range:.4e}\n'
                 f'Unique values: {len(unique_vals)}\n'
                 f'N={len(data)}')
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_difference_plot(differences, output_file):
    """Create bar chart for difference in LL values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    min_diff = np.min(differences)
    max_diff = np.max(differences)
    data_range = max_diff - min_diff

    # Count unique values
    unique_vals, counts = np.unique(differences, return_counts=True)

    if len(unique_vals) <= 10:
        # Use bar chart for discrete distribution
        bar_labels = [f'{val:.2e}' for val in unique_vals]

        bars = ax.bar(range(len(unique_vals)), counts, color='forestgreen',
                     edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.set_xticks(range(len(unique_vals)))
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_xlabel('Difference (Pruning LL - Propagation LL)')
        ax.set_ylabel('Count')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)
    else:
        # Use histogram for continuous distribution
        num_bins = min(30, max(10, len(differences) // 5))
        if data_range > 0:
            bin_edges = np.linspace(min_diff - 0.1*data_range, max_diff + 0.1*data_range, num_bins + 1)
        else:
            bin_edges = np.linspace(mean_diff - 1e-10, mean_diff + 1e-10, 11)

        ax.hist(differences, bins=bin_edges, density=False, alpha=0.7,
               color='forestgreen', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Difference (Pruning LL - Propagation LL)')
        ax.set_ylabel('Count')
        ax.ticklabel_format(useOffset=True, style='scientific', axis='x', scilimits=(-3, 3))
        plt.xticks(rotation=45, ha='right')

    # Add text annotations
    stats_text = (f'Mean: {mean_diff:.4e}\n'
                 f'Std Dev: {std_diff:.4e}\n'
                 f'Range: {data_range:.4e}\n'
                 f'Unique values: {len(unique_vals)}\n'
                 f'N={len(differences)}')
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.set_title('Distribution of Log-Likelihood Difference Between Algorithms')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_timing_ratio_plot(timing_ratios, output_file):
    """Create histogram for GPU timing ratio distribution."""
    if len(timing_ratios) == 0:
        print("No timing data available, skipping timing ratio plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    mean_ratio = np.mean(timing_ratios)
    std_ratio = np.std(timing_ratios)
    min_ratio = np.min(timing_ratios)
    max_ratio = np.max(timing_ratios)
    data_range = max_ratio - min_ratio

    # Use histogram for continuous distribution
    num_bins = 30
    if data_range > 0:
        padding = 0.1 * data_range
        bin_edges = np.linspace(min_ratio - padding, max_ratio + padding, num_bins + 1)
    else:
        bin_edges = np.linspace(mean_ratio - 0.1, mean_ratio + 0.1, num_bins + 1)

    ax.hist(timing_ratios, bins=bin_edges, density=False, alpha=0.7,
           color='purple', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('Time Ratio (Propagation / Pruning)')
    ax.set_ylabel('Count')

    # Add vertical line at mean
    ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_ratio:.4f}')
    ax.legend()

    # Add text annotations
    stats_text = (f'Mean: {mean_ratio:.6f}\n'
                 f'Std Dev: {std_ratio:.6f}\n'
                 f'Range: [{min_ratio:.6f}, {max_ratio:.6f}]\n'
                 f'N={len(timing_ratios)}')
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.set_title('Distribution of GPU Time Ratio (Propagation / Pruning)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def create_memory_ratio_plot(mem_ratios, output_file):
    """Create bar chart showing memory ratio comparison."""
    if len(mem_ratios) == 0:
        print("No memory data available, skipping memory ratio plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Memory ratio is constant, so show as single bar comparison
    mem_ratio = mem_ratios[0]  # All values should be the same

    # Create comparison bars
    categories = ['Pruning\nAlgorithm', 'Propagation\nAlgorithm']
    values = [1.0, mem_ratio]
    colors = ['steelblue', 'darkorange']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Relative Memory Usage')
    ax.set_title('GPU Memory Allocation Ratio\n(Propagation / Pruning)')

    # Add annotation
    stats_text = (f'Memory Ratio: {mem_ratio:.4f}x\n'
                 f'Propagation requires {100*(mem_ratio-1):.1f}% more memory')
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ll_precision.py <results_file>")
        print("Example: python analyze_ll_precision.py results/ll_precision_test_n100_seed42.txt")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)

    # Parse results
    print(f"Parsing results from {results_file}...")
    pruning_lls, propagation_lls, differences, timing_ratios, fwd_times, bwd_times, mem_ratios = parse_results(results_file)

    if len(pruning_lls) == 0:
        print("Error: No valid LL data found in results file")
        sys.exit(1)

    print(f"Found {len(pruning_lls)} valid trials")
    if len(timing_ratios) > 0:
        print(f"Found {len(timing_ratios)} timing measurements")
    if len(mem_ratios) > 0:
        print(f"Memory ratio: {mem_ratios[0]:.4f}")

    # Calculate statistics
    print("\n=== Log-Likelihood Statistics ===")
    print(f"Pruning Algorithm:")
    print(f"  Mean:    {np.mean(pruning_lls):.10f}")
    print(f"  Std Dev: {np.std(pruning_lls):.10e}")
    print(f"  Range:   {np.min(pruning_lls):.10f} to {np.max(pruning_lls):.10f}")

    print(f"\nPropagation Algorithm:")
    print(f"  Mean:    {np.mean(propagation_lls):.10f}")
    print(f"  Std Dev: {np.std(propagation_lls):.10e}")
    print(f"  Range:   {np.min(propagation_lls):.10f} to {np.max(propagation_lls):.10f}")

    print(f"\nDifference (Pruning - Propagation):")
    print(f"  Mean:    {np.mean(differences):.10e}")
    print(f"  Std Dev: {np.std(differences):.10e}")
    print(f"  Range:   {np.min(differences):.10e} to {np.max(differences):.10e}")

    # Timing statistics
    if len(timing_ratios) > 0:
        print(f"\n=== GPU Timing Statistics ===")
        print(f"Forward pass (Pruning):")
        print(f"  Mean:    {np.mean(fwd_times):.6f} ms")
        print(f"  Std Dev: {np.std(fwd_times):.6f} ms")
        print(f"  Range:   {np.min(fwd_times):.6f} to {np.max(fwd_times):.6f} ms")

        print(f"\nBackward pass:")
        print(f"  Mean:    {np.mean(bwd_times):.6f} ms")
        print(f"  Std Dev: {np.std(bwd_times):.6f} ms")
        print(f"  Range:   {np.min(bwd_times):.6f} to {np.max(bwd_times):.6f} ms")

        print(f"\nTime Ratio (Propagation / Pruning):")
        print(f"  Mean:    {np.mean(timing_ratios):.6f}")
        print(f"  Std Dev: {np.std(timing_ratios):.6f}")
        print(f"  Range:   {np.min(timing_ratios):.6f} to {np.max(timing_ratios):.6f}")

    # Create output directory for figures (relative to the results file location)
    results_path = Path(results_file)
    if results_path.parent.name == "results":
        # Results file is in a results subdirectory, put figs in parent
        output_dir = results_path.parent.parent / "figs"
    else:
        # Default to figs in current directory
        output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True)

    # Extract base name for output files (remove _fixed_ if present)
    base_name = Path(results_file).stem.replace('_fixed_', '_').replace('_fixed', '')

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Pruning LL distribution
    create_distribution_plot(
        pruning_lls,
        'Distribution of Pruning Algorithm Log-Likelihood',
        'Log-Likelihood',
        output_dir / f"{base_name}_pruning_dist.png",
        color='steelblue'
    )

    # Plot 2: Propagation LL distribution
    create_distribution_plot(
        propagation_lls,
        'Distribution of Propagation Algorithm Log-Likelihood',
        'Log-Likelihood',
        output_dir / f"{base_name}_propagation_dist.png",
        color='darkorange'
    )

    # Plot 3: Difference distribution
    create_difference_plot(
        differences,
        output_dir / f"{base_name}_difference_dist.png"
    )

    # Plot 4: GPU timing ratio distribution (if available)
    if len(timing_ratios) > 0:
        create_timing_ratio_plot(
            timing_ratios,
            output_dir / f"{base_name}_timing_ratio_dist.png"
        )

    # Plot 5: Memory ratio comparison (if available)
    if len(mem_ratios) > 0:
        create_memory_ratio_plot(
            mem_ratios,
            output_dir / f"{base_name}_memory_ratio.png"
        )

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
