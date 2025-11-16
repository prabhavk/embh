#!/usr/bin/env python3
"""
Visualize cache hit rate statistics for memoized message passing.
Analyzes how hit rate varies with subtree/complement size for upward/downward messages.
"""

import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file='cache_statistics.csv'):
    """Load cache statistics from CSV file."""
    data = {
        'clique_edge': [],
        'subtree_size': [],
        'complement_size': [],
        'hit_rate': [],
        'direction': []
    }

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['clique_edge'].append(row['clique_edge'])
            data['subtree_size'].append(int(row['subtree_size']))
            data['complement_size'].append(int(row['complement_size']))
            data['hit_rate'].append(float(row['hit_rate']))
            data['direction'].append(row['direction'])

    return data

def filter_by_direction(data, direction):
    """Filter data by direction (upward/downward)."""
    indices = [i for i, d in enumerate(data['direction']) if d == direction]
    return {
        'clique_edge': [data['clique_edge'][i] for i in indices],
        'subtree_size': [data['subtree_size'][i] for i in indices],
        'complement_size': [data['complement_size'][i] for i in indices],
        'hit_rate': [data['hit_rate'][i] for i in indices],
        'direction': [data['direction'][i] for i in indices]
    }

def plot_hit_rate_vs_size(data):
    """Plot hit rate vs subtree/complement size for upward and downward messages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Separate upward and downward messages
    upward = filter_by_direction(data, 'upward')
    downward = filter_by_direction(data, 'downward')

    # Plot 1: Upward hit rate vs subtree size
    ax1 = axes[0, 0]
    ax1.scatter(upward['subtree_size'], upward['hit_rate'], alpha=0.7, s=50, c='blue', edgecolors='black')
    ax1.set_xlabel('Subtree Size (number of taxa)', fontsize=12)
    ax1.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax1.set_title('Upward Messages: Hit Rate vs Subtree Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add trend line for upward
    z = np.polyfit(upward['subtree_size'], upward['hit_rate'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(upward['subtree_size']), max(upward['subtree_size']), 100)
    ax1.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label=f'Quadratic fit')
    ax1.legend()

    # Plot 2: Downward hit rate vs complement size
    ax2 = axes[0, 1]
    ax2.scatter(downward['complement_size'], downward['hit_rate'], alpha=0.7, s=50, c='red', edgecolors='black')
    ax2.set_xlabel('Complement Size (number of taxa)', fontsize=12)
    ax2.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax2.set_title('Downward Messages: Hit Rate vs Complement Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Add trend line for downward
    z = np.polyfit(downward['complement_size'], downward['hit_rate'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(downward['complement_size']), max(downward['complement_size']), 100)
    ax2.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label=f'Quadratic fit')
    ax2.legend()

    # Plot 3: Distribution of hit rates for upward messages
    ax3 = axes[1, 0]
    bins = np.arange(0, 105, 5)
    ax3.hist(upward['hit_rate'], bins=bins, alpha=0.7, color='blue', edgecolor='black')
    mean_up = np.mean(upward['hit_rate'])
    ax3.axvline(mean_up, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_up:.1f}%')
    ax3.set_xlabel('Cache Hit Rate (%)', fontsize=12)
    ax3.set_ylabel('Number of Clique Edges', fontsize=12)
    ax3.set_title('Distribution of Upward Message Hit Rates', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Distribution of hit rates for downward messages
    ax4 = axes[1, 1]
    ax4.hist(downward['hit_rate'], bins=bins, alpha=0.7, color='red', edgecolor='black')
    mean_down = np.mean(downward['hit_rate'])
    ax4.axvline(mean_down, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_down:.1f}%')
    ax4.set_xlabel('Cache Hit Rate (%)', fontsize=12)
    ax4.set_ylabel('Number of Clique Edges', fontsize=12)
    ax4.set_title('Distribution of Downward Message Hit Rates', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cache Memoization Analysis for Clique Tree Message Passing', fontsize=16)
    plt.tight_layout()
    plt.savefig('cache_hit_rate_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: cache_hit_rate_analysis.png")
    plt.close()

def plot_combined_analysis(data):
    """Create a combined analysis plot showing key insights."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    upward = filter_by_direction(data, 'upward')
    downward = filter_by_direction(data, 'downward')

    # Plot 1: Side-by-side comparison using signature size
    ax1 = axes[0]

    # For upward: signature size = subtree_size
    # For downward: signature size = complement_size
    ax1.scatter(upward['subtree_size'], upward['hit_rate'], alpha=0.7, s=60,
                c='blue', edgecolors='black', label='Upward (subtree)', marker='o')
    ax1.scatter(downward['complement_size'], downward['hit_rate'], alpha=0.7, s=60,
                c='red', edgecolors='black', label='Downward (complement)', marker='^')

    ax1.set_xlabel('Signature Size (number of taxa in key)', fontsize=12)
    ax1.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax1.set_title('Hit Rate vs Cache Key Size', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add annotation about the pattern
    ax1.annotate('Small keys = High hit rate\n(more pattern repetition)',
                xy=(3, 95), fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    ax1.annotate('Large keys = Low hit rate\n(few exact matches)',
                xy=(25, 10), fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.5))

    # Plot 2: Summary statistics table
    ax2 = axes[1]
    ax2.axis('off')

    # Create summary data
    up_hit = upward['hit_rate']
    down_hit = downward['hit_rate']
    up_sig = upward['subtree_size']
    down_sig = downward['complement_size']

    summary_data = [
        ['Metric', 'Upward Messages', 'Downward Messages'],
        ['Mean Hit Rate', f'{np.mean(up_hit):.1f}%', f'{np.mean(down_hit):.1f}%'],
        ['Median Hit Rate', f'{np.median(up_hit):.1f}%', f'{np.median(down_hit):.1f}%'],
        ['Min Hit Rate', f'{np.min(up_hit):.1f}%', f'{np.min(down_hit):.1f}%'],
        ['Max Hit Rate', f'{np.max(up_hit):.1f}%', f'{np.max(down_hit):.1f}%'],
        ['Mean Signature Size', f'{np.mean(up_sig):.1f}', f'{np.mean(down_sig):.1f}'],
        ['Number of Edges', str(len(up_hit)), str(len(down_hit))],
        ['Edges with >90% Hit Rate', str(sum(1 for h in up_hit if h > 90)), str(sum(1 for h in down_hit if h > 90))],
        ['Edges with >50% Hit Rate', str(sum(1 for h in up_hit if h > 50)), str(sum(1 for h in down_hit if h > 50))],
    ]

    table = ax2.table(cellText=summary_data[1:],
                      colLabels=summary_data[0],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Color the header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')

    ax2.set_title('Summary Statistics', fontsize=14, pad=20)

    plt.suptitle('Memoization Effectiveness Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('cache_summary_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: cache_summary_analysis.png")
    plt.close()

def print_recommendations(data):
    """Print recommendations based on the analysis."""
    upward = filter_by_direction(data, 'upward')
    downward = filter_by_direction(data, 'downward')

    up_hit = upward['hit_rate']
    up_sub = upward['subtree_size']
    down_hit = downward['hit_rate']
    down_comp = downward['complement_size']

    print("\n" + "="*60)
    print("MEMOIZATION OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    # Analyze upward messages
    high_hit_count = sum(1 for h in up_hit if h > 90)
    low_hit_count = sum(1 for h in up_hit if h < 50)
    max_sub_high_hit = max(s for s, h in zip(up_sub, up_hit) if h > 90)

    print(f"\nUPWARD MESSAGES:")
    print(f"  - Total edges: {len(up_hit)}")
    print(f"  - Edges with >90% hit rate: {high_hit_count} ({100*high_hit_count/len(up_hit):.1f}%)")
    print(f"  - Max subtree size for >90% hit rate: {max_sub_high_hit}")
    print(f"  - Edges with <50% hit rate: {low_hit_count} ({100*low_hit_count/len(up_hit):.1f}%)")

    # Find optimal threshold
    max_subtree = max(up_sub)
    best_threshold = 1
    best_efficiency = 0

    for t in range(1, max_subtree + 1):
        subset_hits = [h for s, h in zip(up_sub, up_hit) if s <= t]
        if len(subset_hits) > 0:
            avg_hit_rate = np.mean(subset_hits)
            coverage = len(subset_hits) / len(up_hit)
            efficiency = avg_hit_rate * coverage
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_threshold = t

    subset_count = sum(1 for s in up_sub if s <= best_threshold)
    subset_hits = [h for s, h in zip(up_sub, up_hit) if s <= best_threshold]

    print(f"\n  RECOMMENDATION: Cache only upward messages with subtree_size <= {best_threshold}")
    print(f"    - This covers {subset_count}/{len(up_hit)} edges ({100*subset_count/len(up_hit):.1f}%)")
    print(f"    - Average hit rate for these: {np.mean(subset_hits):.1f}%")

    # Analyze downward messages
    high_hit_down = sum(1 for h in down_hit if h > 50)

    print(f"\nDOWNWARD MESSAGES:")
    print(f"  - Total edges: {len(down_hit)}")
    print(f"  - Average hit rate: {np.mean(down_hit):.1f}%")
    print(f"  - Edges with >50% hit rate: {high_hit_down}")
    print(f"\n  RECOMMENDATION: DO NOT cache downward messages")
    print(f"    - Very low average hit rate")
    print(f"    - Large signature sizes (complement = {np.mean(down_comp):.1f} taxa on avg)")
    print(f"    - Cache lookup overhead exceeds computation savings")

    print("\n" + "="*60)

def main():
    # Load data
    print("Loading cache statistics...")
    data = load_data('cache_statistics.csv')

    upward_count = sum(1 for d in data['direction'] if d == 'upward')
    downward_count = sum(1 for d in data['direction'] if d == 'downward')

    print(f"Loaded {len(data['direction'])} entries")
    print(f"  - Upward messages: {upward_count}")
    print(f"  - Downward messages: {downward_count}")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_hit_rate_vs_size(data)
    plot_combined_analysis(data)

    # Print recommendations
    print_recommendations(data)

if __name__ == "__main__":
    main()
