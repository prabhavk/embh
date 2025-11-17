#!/usr/bin/env python3
"""
Run EMBH with Aitken acceleration multiple times and collect statistics.

Stores:
- Number of iterations for convergence
- Final log-likelihood values
- Lambda values and acceleration factors from each run
- Mean and standard deviation of all metrics
"""

import subprocess
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime


def run_single_trial(trial_num, edges_file, patterns_file, taxon_file, basecomp_file, output_dir, root_name=None):
    """Run a single EMBH trial with Aitken acceleration."""

    cmd = [
        './embh_clique_tree_gpu',
        '--edges', edges_file,
        '--patterns', patterns_file,
        '--taxon', taxon_file,
        '--basecomp', basecomp_file,
        '--output', f'{output_dir}/trial_{trial_num:03d}.txt'
    ]

    if root_name:
        cmd.extend(['--root', root_name])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"Trial {trial_num} failed!")
        print(result.stderr)
        return None

    # Parse output
    output = result.stdout

    # Extract number of iterations
    iter_match = re.search(r'Total iterations: (\d+)', output)
    iterations = int(iter_match.group(1)) if iter_match else None

    # Extract final LL
    ll_match = re.search(r'Final log-likelihood: ([-\d.]+)', output)
    final_ll = float(ll_match.group(1)) if ll_match else None

    # Extract total time
    time_match = re.search(r'Total time: ([\d.]+) ms', output)
    total_time = float(time_match.group(1)) if time_match else None

    # Extract lambda values and acceleration factors
    lambda_matches = re.findall(r'Aitken: lambda=([\d.]+), accel_factor=([\d.]+)', output)
    lambdas = [float(m[0]) for m in lambda_matches]
    accel_factors = [float(m[1]) for m in lambda_matches]

    return {
        'iterations': iterations,
        'final_ll': final_ll,
        'total_time': total_time,
        'lambdas': lambdas,
        'accel_factors': accel_factors
    }


def run_trials_for_root(num_trials, edges_file, patterns_file, taxon_file, basecomp_file, output_dir, root_name=None):
    """Run all trials for a specific root location."""

    print(f"\n{'='*70}")
    if root_name:
        print(f"Running {num_trials} trials with root = {root_name}")
    else:
        print(f"Running {num_trials} trials with default root")
    print(f"{'='*70}")

    # Storage for results
    all_iterations = []
    all_final_lls = []
    all_times = []
    all_lambdas = []
    all_accel_factors = []

    # Run trials
    for trial in range(1, num_trials + 1):
        print(f"Trial {trial}/{num_trials}...", end=' ', flush=True)

        result = run_single_trial(trial, edges_file, patterns_file, taxon_file, basecomp_file, output_dir, root_name)

        if result:
            all_iterations.append(result['iterations'])
            all_final_lls.append(result['final_ll'])
            all_times.append(result['total_time'])
            all_lambdas.extend(result['lambdas'])
            all_accel_factors.extend(result['accel_factors'])

            print(f"Iterations: {result['iterations']}, LL: {result['final_ll']:.10f}, "
                  f"Time: {result['total_time']:.2f} ms")
        else:
            print("FAILED")

    return {
        'iterations': all_iterations,
        'final_lls': all_final_lls,
        'times': all_times,
        'lambdas': all_lambdas,
        'accel_factors': all_accel_factors
    }


def compute_and_print_stats(results, root_name):
    """Compute and print statistics for a set of results."""

    if not results['iterations']:
        print("No successful trials!")
        return None

    iterations_arr = np.array(results['iterations'])
    final_lls_arr = np.array(results['final_lls'])
    times_arr = np.array(results['times'])
    lambdas_arr = np.array(results['lambdas'])
    accel_factors_arr = np.array(results['accel_factors'])

    print(f"\n--- Root: {root_name} ---")
    print(f"Number of successful trials: {len(results['iterations'])}")

    print("\n  Iterations to Convergence:")
    print(f"    Mean:   {np.mean(iterations_arr):.2f}")
    print(f"    Std:    {np.std(iterations_arr):.2f}")
    print(f"    Min:    {np.min(iterations_arr)}")
    print(f"    Max:    {np.max(iterations_arr)}")

    print("\n  Final Log-Likelihood:")
    print(f"    Mean:   {np.mean(final_lls_arr):.10f}")
    print(f"    Std:    {np.std(final_lls_arr):.10e}")
    print(f"    Range:  {np.max(final_lls_arr) - np.min(final_lls_arr):.10e}")

    print("\n  Total Time (ms):")
    print(f"    Mean:   {np.mean(times_arr):.2f}")
    print(f"    Std:    {np.std(times_arr):.2f}")

    print("\n  Lambda Values (Contraction Factor):")
    print(f"    Mean:   {np.mean(lambdas_arr):.6f}")
    print(f"    Std:    {np.std(lambdas_arr):.6f}")

    print("\n  Acceleration Factors:")
    print(f"    Mean:   {np.mean(accel_factors_arr):.6f}")
    print(f"    Std:    {np.std(accel_factors_arr):.6f}")

    return {
        'root_name': root_name,
        'iterations_mean': np.mean(iterations_arr),
        'iterations_std': np.std(iterations_arr),
        'iterations_min': np.min(iterations_arr),
        'iterations_max': np.max(iterations_arr),
        'final_ll_mean': np.mean(final_lls_arr),
        'final_ll_std': np.std(final_lls_arr),
        'final_ll_range': np.max(final_lls_arr) - np.min(final_lls_arr),
        'time_mean': np.mean(times_arr),
        'time_std': np.std(times_arr),
        'lambda_mean': np.mean(lambdas_arr),
        'lambda_std': np.std(lambdas_arr),
        'accel_mean': np.mean(accel_factors_arr),
        'accel_std': np.std(accel_factors_arr),
        'raw': results
    }


def main():
    if len(sys.argv) < 2:
        num_trials = 100
    else:
        num_trials = int(sys.argv[1])

    # Default to full dataset - use data symlink in gpu_cpp directory
    edges_file = 'data/tree_edges.txt'
    patterns_file = 'data/patterns.pat'
    taxon_file = 'data/patterns.taxon_order'
    basecomp_file = 'data/patterns.basecomp'

    # Root locations to test
    roots = ['h_0', 'h_8', 'h_14']  # Explicit root names

    # Create base output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = Path(f'results/aitken_multi_root_trials_{timestamp}')
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {num_trials} trials of EMBH with Aitken acceleration")
    print(f"Output directory: {base_output_dir}")
    print(f"Edges file: {edges_file}")
    print(f"Patterns file: {patterns_file}")
    print(f"Roots to test: {roots}")

    all_stats = []

    # Run trials for each root
    for root_name in roots:
        root_label = root_name
        trial_output_dir = base_output_dir / f'root_{root_label}'
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        results = run_trials_for_root(num_trials, edges_file, patterns_file, taxon_file, basecomp_file,
                                       trial_output_dir, root_name)

        stats = compute_and_print_stats(results, root_label)
        if stats:
            all_stats.append(stats)

    # Print combined summary
    print("\n" + "="*70)
    print("COMBINED SUMMARY - ALL ROOTS")
    print("="*70)

    for stats in all_stats:
        print(f"\nRoot: {stats['root_name']}")
        print(f"  Iterations: {stats['iterations_mean']:.2f} ± {stats['iterations_std']:.2f} "
              f"(range: {stats['iterations_min']}-{stats['iterations_max']})")
        print(f"  Final LL:   {stats['final_ll_mean']:.10f} ± {stats['final_ll_std']:.2e}")
        print(f"  Time (ms):  {stats['time_mean']:.2f} ± {stats['time_std']:.2f}")
        print(f"  Lambda:     {stats['lambda_mean']:.6f} ± {stats['lambda_std']:.6f}")
        print(f"  Accel:      {stats['accel_mean']:.6f} ± {stats['accel_std']:.6f}")

    # Save summary to file
    summary_file = base_output_dir / 'combined_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"EMBH with Aitken Acceleration - Multi-Root {num_trials} Trial Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Patterns file: {patterns_file}\n")
        f.write(f"Number of trials per root: {num_trials}\n")
        f.write("\n")

        for stats in all_stats:
            f.write(f"=== Root: {stats['root_name']} ===\n")
            f.write(f"Iterations to Convergence:\n")
            f.write(f"  Mean:   {stats['iterations_mean']:.2f}\n")
            f.write(f"  Std:    {stats['iterations_std']:.2f}\n")
            f.write(f"  Min:    {stats['iterations_min']}\n")
            f.write(f"  Max:    {stats['iterations_max']}\n")
            f.write(f"\nFinal Log-Likelihood:\n")
            f.write(f"  Mean:   {stats['final_ll_mean']:.10f}\n")
            f.write(f"  Std:    {stats['final_ll_std']:.10e}\n")
            f.write(f"  Range:  {stats['final_ll_range']:.10e}\n")
            f.write(f"\nTotal Time (ms):\n")
            f.write(f"  Mean:   {stats['time_mean']:.2f}\n")
            f.write(f"  Std:    {stats['time_std']:.2f}\n")
            f.write(f"\nLambda Values (Contraction Factor):\n")
            f.write(f"  Mean:   {stats['lambda_mean']:.6f}\n")
            f.write(f"  Std:    {stats['lambda_std']:.6f}\n")
            f.write(f"\nAcceleration Factors:\n")
            f.write(f"  Mean:   {stats['accel_mean']:.6f}\n")
            f.write(f"  Std:    {stats['accel_std']:.6f}\n")
            f.write("\n" + "-"*50 + "\n\n")

            # Save individual trial data
            data_file = base_output_dir / f'trial_data_root_{stats["root_name"]}.txt'
            with open(data_file, 'w') as df:
                df.write("Trial\tIterations\tFinal_LL\tTime_ms\n")
                raw = stats['raw']
                for i in range(len(raw['iterations'])):
                    df.write(f"{i+1}\t{raw['iterations'][i]}\t{raw['final_lls'][i]:.10f}\t{raw['times'][i]:.2f}\n")

    print(f"\nSummary saved to: {summary_file}")


def main_original():
    """Original main function for single root trials."""
    if len(sys.argv) < 2:
        num_trials = 100
    else:
        num_trials = int(sys.argv[1])

    # Default to full dataset
    patterns_file = '../data/patterns.pat'
    taxon_file = '../data/patterns.taxon_order'
    basecomp_file = '../data/patterns.basecomp'

    # Create output directory for individual trial results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/aitken_trials_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {num_trials} trials of EMBH with Aitken acceleration")
    print(f"Output directory: {output_dir}")
    print(f"Patterns file: {patterns_file}")
    print()

    # Storage for results
    all_iterations = []
    all_final_lls = []
    all_times = []
    all_lambdas = []
    all_accel_factors = []

    # Run trials
    for trial in range(1, num_trials + 1):
        print(f"Trial {trial}/{num_trials}...", end=' ', flush=True)

        result = run_single_trial(trial, patterns_file, taxon_file, basecomp_file, output_dir)

        if result:
            all_iterations.append(result['iterations'])
            all_final_lls.append(result['final_ll'])
            all_times.append(result['total_time'])
            all_lambdas.extend(result['lambdas'])
            all_accel_factors.extend(result['accel_factors'])

            print(f"Iterations: {result['iterations']}, LL: {result['final_ll']:.10f}, "
                  f"Time: {result['total_time']:.2f} ms")
        else:
            print("FAILED")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if not all_iterations:
        print("No successful trials!")
        return

    # Compute statistics
    iterations_arr = np.array(all_iterations)
    final_lls_arr = np.array(all_final_lls)
    times_arr = np.array(all_times)
    lambdas_arr = np.array(all_lambdas)
    accel_factors_arr = np.array(all_accel_factors)

    print(f"\nNumber of successful trials: {len(all_iterations)}")

    print("\n--- Iterations to Convergence ---")
    print(f"  Mean:   {np.mean(iterations_arr):.2f}")
    print(f"  Std:    {np.std(iterations_arr):.2f}")
    print(f"  Min:    {np.min(iterations_arr)}")
    print(f"  Max:    {np.max(iterations_arr)}")
    print(f"  Median: {np.median(iterations_arr):.1f}")

    print("\n--- Final Log-Likelihood ---")
    print(f"  Mean:   {np.mean(final_lls_arr):.10f}")
    print(f"  Std:    {np.std(final_lls_arr):.10e}")
    print(f"  Min:    {np.min(final_lls_arr):.10f}")
    print(f"  Max:    {np.max(final_lls_arr):.10f}")
    print(f"  Range:  {np.max(final_lls_arr) - np.min(final_lls_arr):.10e}")

    print("\n--- Total Time (ms) ---")
    print(f"  Mean:   {np.mean(times_arr):.2f}")
    print(f"  Std:    {np.std(times_arr):.2f}")
    print(f"  Min:    {np.min(times_arr):.2f}")
    print(f"  Max:    {np.max(times_arr):.2f}")

    print("\n--- Lambda Values (Contraction Factor) ---")
    print(f"  Mean:   {np.mean(lambdas_arr):.6f}")
    print(f"  Std:    {np.std(lambdas_arr):.6f}")
    print(f"  Min:    {np.min(lambdas_arr):.6f}")
    print(f"  Max:    {np.max(lambdas_arr):.6f}")

    print("\n--- Acceleration Factors ---")
    print(f"  Mean:   {np.mean(accel_factors_arr):.6f}")
    print(f"  Std:    {np.std(accel_factors_arr):.6f}")
    print(f"  Min:    {np.min(accel_factors_arr):.6f}")
    print(f"  Max:    {np.max(accel_factors_arr):.6f}")

    # Save summary to file
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"EMBH with Aitken Acceleration - {num_trials} Trial Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Patterns file: {patterns_file}\n")
        f.write(f"Number of successful trials: {len(all_iterations)}\n")
        f.write("\n")

        f.write("=== Iterations to Convergence ===\n")
        f.write(f"Mean:   {np.mean(iterations_arr):.2f}\n")
        f.write(f"Std:    {np.std(iterations_arr):.2f}\n")
        f.write(f"Min:    {np.min(iterations_arr)}\n")
        f.write(f"Max:    {np.max(iterations_arr)}\n")
        f.write(f"Median: {np.median(iterations_arr):.1f}\n")
        f.write("\n")

        f.write("=== Final Log-Likelihood ===\n")
        f.write(f"Mean:   {np.mean(final_lls_arr):.10f}\n")
        f.write(f"Std:    {np.std(final_lls_arr):.10e}\n")
        f.write(f"Min:    {np.min(final_lls_arr):.10f}\n")
        f.write(f"Max:    {np.max(final_lls_arr):.10f}\n")
        f.write(f"Range:  {np.max(final_lls_arr) - np.min(final_lls_arr):.10e}\n")
        f.write("\n")

        f.write("=== Total Time (ms) ===\n")
        f.write(f"Mean:   {np.mean(times_arr):.2f}\n")
        f.write(f"Std:    {np.std(times_arr):.2f}\n")
        f.write(f"Min:    {np.min(times_arr):.2f}\n")
        f.write(f"Max:    {np.max(times_arr):.2f}\n")
        f.write("\n")

        f.write("=== Lambda Values (Contraction Factor) ===\n")
        f.write(f"Mean:   {np.mean(lambdas_arr):.6f}\n")
        f.write(f"Std:    {np.std(lambdas_arr):.6f}\n")
        f.write(f"Min:    {np.min(lambdas_arr):.6f}\n")
        f.write(f"Max:    {np.max(lambdas_arr):.6f}\n")
        f.write("\n")

        f.write("=== Acceleration Factors ===\n")
        f.write(f"Mean:   {np.mean(accel_factors_arr):.6f}\n")
        f.write(f"Std:    {np.std(accel_factors_arr):.6f}\n")
        f.write(f"Min:    {np.min(accel_factors_arr):.6f}\n")
        f.write(f"Max:    {np.max(accel_factors_arr):.6f}\n")
        f.write("\n")

        f.write("=== Individual Trial Results ===\n")
        f.write("Trial\tIterations\tFinal_LL\tTime_ms\n")
        for i in range(len(all_iterations)):
            f.write(f"{i+1}\t{all_iterations[i]}\t{all_final_lls[i]:.10f}\t{all_times[i]:.2f}\n")

    print(f"\nSummary saved to: {summary_file}")

    # Also save raw data for further analysis
    data_file = output_dir / 'trial_data.txt'
    with open(data_file, 'w') as f:
        f.write("Trial\tIterations\tFinal_LL\tTime_ms\tNum_Accel_Events\n")
        for i in range(len(all_iterations)):
            f.write(f"{i+1}\t{all_iterations[i]}\t{all_final_lls[i]:.10f}\t{all_times[i]:.2f}\t")
            # Count acceleration events per trial (lambdas per trial)
            num_accels = all_iterations[i] // 3  # Approximate, since we accelerate every 3rd iteration
            f.write(f"{num_accels}\n")

    print(f"Raw data saved to: {data_file}")


if __name__ == "__main__":
    main()
