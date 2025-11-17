#!/usr/bin/env python3
"""
Compare EM convergence with and without Aitken acceleration.

Creates visualization showing:
- Log-likelihood convergence over iterations
- Rate of convergence comparison
- Lambda values from Aitken acceleration
- Time efficiency comparison
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
    'figure.figsize': (12, 10),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_em_results(filepath):
    """Parse EM convergence results file."""
    results = {
        'aitken_enabled': False,
        'aitken_damping': 0.0,
        'converged': False,
        'total_iterations': 0,
        'total_time': 0.0,
        'final_ll': 0.0,
        'iterations': [],
        'lls': [],
        'aitken_ll_ests': [],
        'times': [],
        'accel_events': [],  # List of (iter, lambda, accel_factor, applied)
        'num_patterns': None  # Extract from filename if present
    }

    # Try to extract num_patterns and root_name from filename
    import re
    match = re.search(r'num_patterns(\d+)', str(filepath))
    if match:
        results['num_patterns'] = int(match.group(1))

    root_match = re.search(r'root_([^_]+)', str(filepath))
    if root_match:
        results['root_name'] = root_match.group(1)
    else:
        results['root_name'] = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_iter_data = False
    in_accel_data = False

    for line in lines:
        line = line.strip()

        # Parse header metadata
        if line.startswith('# Aitken acceleration:'):
            results['aitken_enabled'] = 'enabled' in line
        elif line.startswith('# Aitken damping:'):
            results['aitken_damping'] = float(line.split(':')[1].strip())
        elif line.startswith('# Converged:'):
            results['converged'] = 'yes' in line
        elif line.startswith('# Total iterations:'):
            results['total_iterations'] = int(line.split(':')[1].strip())
        elif line.startswith('# Total time (ms):'):
            results['total_time'] = float(line.split(':')[1].strip())
        elif line.startswith('# Final LL:'):
            results['final_ll'] = float(line.split(':')[1].strip())

        # Parse iteration data section
        if line.startswith('=== Iteration Data ==='):
            in_iter_data = True
            in_accel_data = False
            continue

        if in_iter_data and line.startswith('Iter\t'):
            continue

        if in_iter_data and line and not line.startswith('==='):
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    results['iterations'].append(int(parts[0]))
                    results['lls'].append(float(parts[1]))
                    results['aitken_ll_ests'].append(float(parts[2]))
                    results['times'].append(float(parts[3]))
                except ValueError:
                    pass

        # Parse acceleration events section
        if line.startswith('=== Aitken Acceleration Events ==='):
            in_iter_data = False
            in_accel_data = True
            continue

        if in_accel_data and line.startswith('Iter\t'):
            continue

        if in_accel_data and line:
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    iter_num = int(parts[0])
                    lambda_val = float(parts[1])
                    accel_factor = float(parts[2])
                    applied = parts[3] == 'yes'
                    results['accel_events'].append((iter_num, lambda_val, accel_factor, applied))
                except ValueError:
                    pass

    return results


def plot_convergence_comparison(with_aitken, without_aitken, output_dir, num_patterns=None, root_name=None):
    """Create main convergence comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Log-likelihood vs iteration
    ax = axes[0, 0]
    ax.plot(with_aitken['iterations'], with_aitken['lls'],
            'b-o', linewidth=2, markersize=6, label='With Aitken', alpha=0.8)
    ax.plot(without_aitken['iterations'], without_aitken['lls'],
            'r-s', linewidth=2, markersize=5, label='Without Aitken', alpha=0.8)

    # Mark acceleration events
    for event in with_aitken['accel_events']:
        iter_num, lambda_val, _, applied = event
        if applied:
            # Find LL at this iteration
            idx = with_aitken['iterations'].index(iter_num)
            ll = with_aitken['lls'][idx]
            ax.plot(iter_num, ll, 'g^', markersize=12, markerfacecolor='none',
                   markeredgewidth=2, label='_nolegend_')

    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('EM Convergence: Log-Likelihood')
    ax.legend(loc='lower right')

    # Add text annotation
    info_text = (f'With Aitken: {len(with_aitken["lls"])} iters, {with_aitken["total_time"]:.1f} ms\n'
                f'Without Aitken: {len(without_aitken["lls"])} iters, {without_aitken["total_time"]:.1f} ms\n'
                f'Speedup: {without_aitken["total_time"]/with_aitken["total_time"]:.2f}x')
    ax.text(0.05, 0.35, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')

    # Plot 2: LL difference from final (convergence rate)
    ax = axes[0, 1]
    final_ll = with_aitken['final_ll']

    with_aitken_diff = [ll - final_ll for ll in with_aitken['lls']]
    without_aitken_diff = [ll - final_ll for ll in without_aitken['lls']]

    ax.semilogy(with_aitken['iterations'], [-d for d in with_aitken_diff],
                'b-o', linewidth=2, markersize=6, label='With Aitken', alpha=0.8)
    ax.semilogy(without_aitken['iterations'], [-d for d in without_aitken_diff],
                'r-s', linewidth=2, markersize=5, label='Without Aitken', alpha=0.8)

    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Distance to Optimum (LL* - LL)')
    ax.set_title('Convergence Rate (Log Scale)')
    ax.legend(loc='upper right')

    # Plot 3: Lambda values and acceleration factor
    ax = axes[1, 0]
    if with_aitken['accel_events']:
        iters = [e[0] for e in with_aitken['accel_events']]
        lambdas = [e[1] for e in with_aitken['accel_events']]
        accel_factors = [e[2] for e in with_aitken['accel_events']]

        ax.plot(iters, lambdas, 'g-o', linewidth=2, markersize=10, label='λ (contraction factor)')
        ax2 = ax.twinx()
        ax2.plot(iters, accel_factors, 'purple', linestyle='--', marker='D',
                linewidth=2, markersize=8, label='Acceleration factor')

        ax.set_xlabel('EM Iteration')
        ax.set_ylabel('λ (Contraction Factor)', color='g')
        ax2.set_ylabel('Acceleration Factor', color='purple')
        ax.set_title('Aitken Acceleration Parameters')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Add lambda values as text
        for i, (iter_num, lam) in enumerate(zip(iters, lambdas)):
            ax.annotate(f'{lam:.3f}', xy=(iter_num, lam), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, color='green')
    else:
        ax.text(0.5, 0.5, 'No Aitken acceleration events', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        ax.set_title('Aitken Acceleration Parameters')

    # Plot 4: Iteration time and cumulative time
    ax = axes[1, 1]

    # Cumulative time
    with_cumtime = np.cumsum(with_aitken['times'])
    without_cumtime = np.cumsum(without_aitken['times'])

    ax.plot(with_cumtime, with_aitken['lls'], 'b-o', linewidth=2, markersize=6,
           label='With Aitken', alpha=0.8)
    ax.plot(without_cumtime, without_aitken['lls'], 'r-s', linewidth=2, markersize=5,
           label='Without Aitken', alpha=0.8)

    ax.set_xlabel('Cumulative Time (ms)')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Convergence vs. Wall Clock Time')
    ax.legend(loc='lower right')

    plt.suptitle('EM Convergence: With vs Without Aitken Acceleration', fontsize=18, y=1.02)
    plt.tight_layout()

    # Build filename with root name and pattern count if available
    suffix = ""
    if root_name:
        suffix += f"_root_{root_name}"
    if num_patterns:
        suffix += f"_num_patterns{num_patterns}"
    output_file = output_dir / f'em_convergence_comparison{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_aitken_ll_estimation(with_aitken, output_dir, num_patterns=None, root_name=None):
    """Plot Aitken LL estimation accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))

    iters = with_aitken['iterations']
    lls = with_aitken['lls']
    aitken_ests = with_aitken['aitken_ll_ests']
    final_ll = with_aitken['final_ll']

    ax.plot(iters, lls, 'b-o', linewidth=2, markersize=6, label='Actual LL', alpha=0.8)
    ax.plot(iters, aitken_ests, 'g--s', linewidth=2, markersize=6,
           label='Aitken LL Estimate', alpha=0.8)
    ax.axhline(y=final_ll, color='k', linestyle=':', linewidth=2,
              label=f'Final LL = {final_ll:.6f}')

    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Aitken Log-Likelihood Estimation Accuracy')
    ax.legend(loc='lower right')

    # Add text showing estimation error
    errors = [abs(aitken_ests[i] - final_ll) for i in range(2, len(aitken_ests))]
    if errors:
        info_text = (f'Mean estimation error: {np.mean(errors):.6f}\n'
                    f'Final estimation error: {errors[-1]:.2e}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')

    plt.tight_layout()
    # Build filename with root name and pattern count if available
    suffix = ""
    if root_name:
        suffix += f"_root_{root_name}"
    if num_patterns:
        suffix += f"_num_patterns{num_patterns}"
    output_file = output_dir / f'aitken_ll_estimation{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_ll_comparison_only(with_aitken, without_aitken, output_dir, num_patterns=None, root_name=None):
    """Create a simple direct comparison of LL convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: LL vs Iteration
    ax = axes[0]
    ax.plot(with_aitken['iterations'], with_aitken['lls'],
            'b-o', linewidth=2, markersize=8, label=f'With Aitken ({len(with_aitken["lls"])} iters)')
    ax.plot(without_aitken['iterations'], without_aitken['lls'],
            'r-s', linewidth=2, markersize=6, label=f'Without Aitken ({len(without_aitken["lls"])} iters)')

    # Mark acceleration events with triangles
    for event in with_aitken['accel_events']:
        iter_num, lambda_val, _, applied = event
        if applied and iter_num <= len(with_aitken['lls']):
            idx = iter_num - 1
            if idx < len(with_aitken['lls']):
                ll = with_aitken['lls'][idx]
                ax.plot(iter_num, ll, 'g^', markersize=14, markerfacecolor='none',
                       markeredgewidth=3)

    # Add vertical lines at convergence
    ax.axvline(x=len(with_aitken['lls']), color='blue', linestyle='--', alpha=0.7, linewidth=2,
               label=f'_nolegend_')
    ax.axvline(x=len(without_aitken['lls']), color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'_nolegend_')

    # Add text annotations for convergence iterations
    ymin, ymax = ax.get_ylim()
    ax.text(len(with_aitken['lls']), ymax - (ymax-ymin)*0.05,
            f'{len(with_aitken["lls"])}', color='blue', fontsize=12,
            ha='center', va='top', fontweight='bold')
    ax.text(len(without_aitken['lls']), ymax - (ymax-ymin)*0.05,
            f'{len(without_aitken["lls"])}', color='red', fontsize=12,
            ha='center', va='top', fontweight='bold')

    ax.set_xlabel('EM Iteration', fontsize=14)
    ax.set_ylabel('Log-Likelihood', fontsize=14)
    ax.set_title('EM Convergence Comparison', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)

    # Add summary text
    speedup_iter = len(without_aitken['lls']) / len(with_aitken['lls'])
    speedup_time = without_aitken['total_time'] / with_aitken['total_time']
    info_text = (f'Iteration speedup: {speedup_iter:.2f}x\n'
                f'Time speedup: {speedup_time:.2f}x\n'
                f'Green triangles: Aitken acceleration')
    ax.text(0.05, 0.25, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
           family='monospace')

    # Plot 2: Distance to optimum (log scale)
    ax = axes[1]
    final_ll = max(with_aitken['final_ll'], without_aitken['final_ll'])

    with_diffs = [final_ll - ll for ll in with_aitken['lls']]
    without_diffs = [final_ll - ll for ll in without_aitken['lls']]

    # Filter out negative or zero values for log scale
    with_iters_plot = []
    with_diffs_plot = []
    for i, d in enumerate(with_diffs):
        if d > 0:
            with_iters_plot.append(with_aitken['iterations'][i])
            with_diffs_plot.append(d)

    without_iters_plot = []
    without_diffs_plot = []
    for i, d in enumerate(without_diffs):
        if d > 0:
            without_iters_plot.append(without_aitken['iterations'][i])
            without_diffs_plot.append(d)

    if with_diffs_plot:
        ax.semilogy(with_iters_plot, with_diffs_plot,
                   'b-o', linewidth=2, markersize=8, label='With Aitken', alpha=0.8)
    if without_diffs_plot:
        ax.semilogy(without_iters_plot, without_diffs_plot,
                   'r-s', linewidth=2, markersize=6, label='Without Aitken', alpha=0.8)

    ax.set_xlabel('EM Iteration', fontsize=14)
    ax.set_ylabel('Distance to Optimum (LL* - LL)', fontsize=14)
    ax.set_title('Convergence Rate (Log Scale)', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)

    plt.suptitle('With vs Without Aitken Acceleration', fontsize=18, y=1.02)
    plt.tight_layout()

    # Build filename with root name and pattern count if available
    suffix = ""
    if root_name:
        suffix += f"_root_{root_name}"
    if num_patterns:
        suffix += f"_num_patterns{num_patterns}"
    output_file = output_dir / f'em_convergence_direct_comparison{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_summary(with_aitken, without_aitken):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EM CONVERGENCE COMPARISON SUMMARY")
    print("="*60)

    print("\nWith Aitken Acceleration:")
    print(f"  Total iterations: {len(with_aitken['lls'])}")
    print(f"  Total time: {with_aitken['total_time']:.2f} ms")
    print(f"  Final LL: {with_aitken['final_ll']:.10f}")
    print(f"  Converged: {with_aitken['converged']}")
    if with_aitken['aitken_enabled']:
        print(f"  Damping factor: {with_aitken['aitken_damping']}")

    print("\nWithout Aitken Acceleration:")
    print(f"  Total iterations: {len(without_aitken['lls'])}")
    print(f"  Total time: {without_aitken['total_time']:.2f} ms")
    print(f"  Final LL: {without_aitken['final_ll']:.10f}")
    print(f"  Converged: {without_aitken['converged']}")

    print("\nSpeedup:")
    iter_ratio = len(without_aitken['lls']) / len(with_aitken['lls'])
    time_ratio = without_aitken['total_time'] / with_aitken['total_time']
    print(f"  Iteration reduction: {iter_ratio:.2f}x ({len(without_aitken['lls'])} -> {len(with_aitken['lls'])})")
    print(f"  Time speedup: {time_ratio:.2f}x ({without_aitken['total_time']:.1f} ms -> {with_aitken['total_time']:.1f} ms)")

    print("\nAitken Acceleration Events:")
    if with_aitken['accel_events']:
        print(f"  {'Iter':<6} {'Lambda':<12} {'Accel Factor':<15} {'Applied'}")
        print("  " + "-"*45)
        for event in with_aitken['accel_events']:
            iter_num, lambda_val, accel_factor, applied = event
            print(f"  {iter_num:<6} {lambda_val:<12.6f} {accel_factor:<15.6f} {'Yes' if applied else 'No'}")
    else:
        print("  No acceleration events recorded")

    print("\n" + "="*60)


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_em_convergence.py <with_aitken_file> <without_aitken_file>")
        print("Example: python compare_em_convergence.py results/em_with_aitken.txt results/em_without_aitken.txt")
        sys.exit(1)

    with_aitken_file = sys.argv[1]
    without_aitken_file = sys.argv[2]

    # Parse results
    print(f"Parsing {with_aitken_file}...")
    with_aitken = parse_em_results(with_aitken_file)

    print(f"Parsing {without_aitken_file}...")
    without_aitken = parse_em_results(without_aitken_file)

    # Determine output directory
    results_path = Path(with_aitken_file)
    if results_path.parent.name == "results":
        output_dir = results_path.parent.parent / "figs"
    else:
        output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True)

    # Print summary
    print_summary(with_aitken, without_aitken)

    # Get pattern count and root name (from either result file)
    num_patterns = with_aitken.get('num_patterns') or without_aitken.get('num_patterns')
    root_name = with_aitken.get('root_name') or without_aitken.get('root_name')

    # Build suffix for filenames
    suffix = ""
    if root_name:
        suffix += f"_root_{root_name}"
    if num_patterns:
        suffix += f"_num_patterns{num_patterns}"

    # Create plots
    print("\nGenerating plots...")
    plot_convergence_comparison(with_aitken, without_aitken, output_dir, num_patterns, root_name)
    plot_ll_comparison_only(with_aitken, without_aitken, output_dir, num_patterns, root_name)
    plot_aitken_ll_estimation(with_aitken, output_dir, num_patterns, root_name)

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
