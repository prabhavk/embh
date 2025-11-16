#!/usr/bin/env python3
"""
Visualize EM convergence for EMBH GPU program.
Plots Log-Likelihood (LL) and Expected Complete Data Log-Likelihood (ECDLL) over iterations.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def load_em_results(filename):
    """Load EM results from CSV file."""
    iters = []
    ll_values = []
    ecdll_values = []
    improvements = []
    rates = []
    aitken_dists = []
    gpu_times = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                iters.append(int(parts[0]))
                ll_values.append(float(parts[1]))
                ecdll_values.append(float(parts[2]))
                improvements.append(float(parts[3]))
                rates.append(float(parts[4]))
                aitken_dists.append(float(parts[5]))
                gpu_times.append(float(parts[6]))

    return {
        'iter': np.array(iters),
        'LL': np.array(ll_values),
        'ECDLL': np.array(ecdll_values),
        'improvement': np.array(improvements),
        'rate': np.array(rates),
        'aitken_dist': np.array(aitken_dists),
        'gpu_time': np.array(gpu_times)
    }

def plot_convergence(data, output_prefix='em_convergence'):
    """Create convergence plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: LL and ECDLL over iterations
    ax1 = axes[0, 0]
    ax1.plot(data['iter'], data['LL'], 'b-', linewidth=2, label='Log-Likelihood (LL)', marker='o', markersize=4)
    ax1.plot(data['iter'], data['ECDLL'], 'r--', linewidth=2, label='ECDLL (Q-function)', marker='s', markersize=4)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Log-Likelihood', fontsize=12)
    ax1.set_title('EM Convergence: LL and ECDLL', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Change in LL and ECDLL per iteration
    ax2 = axes[0, 1]
    if len(data['iter']) > 1:
        delta_ll = np.diff(data['LL'])
        delta_ecdll = np.diff(data['ECDLL'])
        iters_diff = data['iter'][1:]

        ax2.plot(iters_diff, delta_ll, 'b-', linewidth=2, label='ΔLL', marker='o', markersize=4)
        ax2.plot(iters_diff, delta_ecdll, 'r--', linewidth=2, label='ΔECDLL', marker='s', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Change per Iteration', fontsize=12)
        ax2.set_title('Change in LL and ECDLL', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence rate
    ax3 = axes[1, 0]
    valid_rates = data['rate'] > 0
    if np.any(valid_rates):
        ax3.plot(data['iter'][valid_rates], data['rate'][valid_rates], 'g-', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=0.9, color='r', linestyle='--', linewidth=1.5, label='Rate threshold (0.9)')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Convergence Rate', fontsize=12)
        ax3.set_title('Aitken Convergence Rate', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, min(1.5, np.max(data['rate'][valid_rates]) * 1.1)])

    # Plot 4: Aitken distance (log scale)
    ax4 = axes[1, 1]
    valid_aitken = data['aitken_dist'] > 0
    if np.any(valid_aitken):
        ax4.semilogy(data['iter'][valid_aitken], data['aitken_dist'][valid_aitken], 'm-', linewidth=2, marker='o', markersize=4)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Aitken Distance (log scale)', fontsize=12)
        ax4.set_title('Aitken Distance to Convergence', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('EM Algorithm Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    print(f"Saved plots to {output_prefix}.png and {output_prefix}.pdf")

    # Show plot
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EM Convergence Summary")
    print("="*60)
    print(f"Total iterations: {len(data['iter'])}")
    print(f"Initial LL: {data['LL'][0]:.6f}")
    print(f"Final LL: {data['LL'][-1]:.6f}")
    print(f"Total LL improvement: {data['LL'][-1] - data['LL'][0]:.6f}")
    print(f"Final ECDLL: {data['ECDLL'][-1]:.6f}")

    if len(data['iter']) > 1:
        print(f"\nAverage LL improvement per iteration: {np.mean(np.diff(data['LL'])):.6f}")
        print(f"Average ECDLL change per iteration: {np.mean(np.diff(data['ECDLL'])):.6f}")

    valid_rates = data['rate'] > 0
    if np.any(valid_rates):
        print(f"\nFinal convergence rate: {data['rate'][-1]:.6f}")
        print(f"Average rate: {np.mean(data['rate'][valid_rates]):.6f}")

    print(f"\nTotal GPU time: {np.sum(data['gpu_time']):.2f} ms")
    print(f"Average GPU time per iteration: {np.mean(data['gpu_time']):.2f} ms")
    print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_em_convergence.py <em_results.txt> [output_name]")
        print("\nExample:")
        print("  python visualize_em_convergence.py ../results/em_results_h_4.txt")
        print("  python visualize_em_convergence.py ../results/em_results_h_4.txt h_4_convergence")
        print("\nFigures will be saved to ../figs/ directory")
        sys.exit(1)

    input_file = sys.argv[1]

    # Determine output prefix - save to ../figs/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, '..', 'figs')

    if len(sys.argv) > 2:
        output_name = sys.argv[2]
    else:
        # Extract base name from input file
        base_name = os.path.basename(input_file).replace('.txt', '')
        output_name = base_name

    output_prefix = os.path.join(figs_dir, output_name)

    print(f"Loading EM results from: {input_file}")
    data = load_em_results(input_file)

    if len(data['iter']) == 0:
        print("Error: No data found in file")
        sys.exit(1)

    print_summary(data)
    plot_convergence(data, output_prefix)

if __name__ == "__main__":
    main()
