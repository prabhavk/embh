#!/usr/bin/env python3
"""
Analyze cache hit rate statistics for memoized message passing.
Text-based analysis (no matplotlib required).
"""

import csv

def load_data(csv_file='cache_statistics.csv'):
    """Load cache statistics from CSV file."""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'clique_edge': row['clique_edge'],
                'subtree_size': int(row['subtree_size']),
                'complement_size': int(row['complement_size']),
                'hit_rate': float(row['hit_rate']),
                'direction': row['direction']
            })
    return data

def analyze_data(data):
    """Analyze cache statistics."""
    upward = [d for d in data if d['direction'] == 'upward']
    downward = [d for d in data if d['direction'] == 'downward']

    print("="*70)
    print("CACHE STATISTICS ANALYSIS")
    print("="*70)

    # Upward messages analysis
    up_hits = [d['hit_rate'] for d in upward]
    up_sizes = [d['subtree_size'] for d in upward]

    print(f"\nUPWARD MESSAGES (leaves to root):")
    print(f"  Total edges: {len(upward)}")
    print(f"  Hit rate statistics:")
    print(f"    Mean:   {sum(up_hits)/len(up_hits):.2f}%")
    print(f"    Min:    {min(up_hits):.2f}%")
    print(f"    Max:    {max(up_hits):.2f}%")
    print(f"    Median: {sorted(up_hits)[len(up_hits)//2]:.2f}%")

    # Group by subtree size
    print(f"\n  Hit rate by subtree size:")
    sizes = sorted(set(up_sizes))
    for size in sizes:
        rates = [d['hit_rate'] for d in upward if d['subtree_size'] == size]
        print(f"    Size {size:2d}: {sum(rates)/len(rates):6.2f}% avg  ({len(rates):2d} edges)")

    # Downward messages analysis
    down_hits = [d['hit_rate'] for d in downward]
    down_sizes = [d['complement_size'] for d in downward]

    print(f"\nDOWNWARD MESSAGES (root to leaves):")
    print(f"  Total edges: {len(downward)}")
    print(f"  Hit rate statistics:")
    print(f"    Mean:   {sum(down_hits)/len(down_hits):.2f}%")
    print(f"    Min:    {min(down_hits):.2f}%")
    print(f"    Max:    {max(down_hits):.2f}%")
    print(f"    Median: {sorted(down_hits)[len(down_hits)//2]:.2f}%")

    # Group by complement size
    print(f"\n  Hit rate by complement size:")
    sizes = sorted(set(down_sizes))
    for size in sizes:
        rates = [d['hit_rate'] for d in downward if d['complement_size'] == size]
        print(f"    Size {size:2d}: {sum(rates)/len(rates):6.2f}% avg  ({len(rates):2d} edges)")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find optimal threshold for upward messages
    best_threshold = 1
    best_weighted_rate = 0
    for threshold in range(1, max(up_sizes) + 1):
        subset = [d for d in upward if d['subtree_size'] <= threshold]
        if subset:
            avg_rate = sum(d['hit_rate'] for d in subset) / len(subset)
            coverage = len(subset) / len(upward)
            weighted = avg_rate * coverage
            if weighted > best_weighted_rate:
                best_weighted_rate = weighted
                best_threshold = threshold

    subset = [d for d in upward if d['subtree_size'] <= best_threshold]
    subset_rate = sum(d['hit_rate'] for d in subset) / len(subset)

    print(f"\nUPWARD MESSAGES:")
    print(f"  RECOMMENDED: Cache messages with subtree_size <= {best_threshold}")
    print(f"    - Covers {len(subset)}/{len(upward)} edges ({100*len(subset)/len(upward):.1f}%)")
    print(f"    - Average hit rate: {subset_rate:.1f}%")

    high_hit_upward = sum(1 for h in up_hits if h > 90)
    print(f"    - Edges with >90% hit rate: {high_hit_upward}/{len(upward)}")

    print(f"\nDOWNWARD MESSAGES:")
    print(f"  RECOMMENDED: DO NOT cache downward messages")
    print(f"    - Low average hit rate: {sum(down_hits)/len(down_hits):.1f}%")
    print(f"    - Large signature sizes: {sum(down_sizes)/len(down_sizes):.1f} taxa on average")
    print(f"    - Overhead exceeds benefit")

    # Show detailed data for inspection
    print("\n" + "="*70)
    print("DETAILED DATA (sorted by hit rate)")
    print("="*70)

    print("\nTOP 10 UPWARD MESSAGES (highest hit rate):")
    upward_sorted = sorted(upward, key=lambda x: x['hit_rate'], reverse=True)
    print(f"{'Clique Edge':<12} {'Subtree':<8} {'Hit Rate':<10}")
    print("-"*30)
    for d in upward_sorted[:10]:
        print(f"{d['clique_edge']:<12} {d['subtree_size']:<8} {d['hit_rate']:.2f}%")

    print("\nBOTTOM 10 UPWARD MESSAGES (lowest hit rate):")
    print(f"{'Clique Edge':<12} {'Subtree':<8} {'Hit Rate':<10}")
    print("-"*30)
    for d in upward_sorted[-10:]:
        print(f"{d['clique_edge']:<12} {d['subtree_size']:<8} {d['hit_rate']:.2f}%")

    print("\nTOP 10 DOWNWARD MESSAGES (highest hit rate):")
    downward_sorted = sorted(downward, key=lambda x: x['hit_rate'], reverse=True)
    print(f"{'Clique Edge':<12} {'Complement':<12} {'Hit Rate':<10}")
    print("-"*34)
    for d in downward_sorted[:10]:
        print(f"{d['clique_edge']:<12} {d['complement_size']:<12} {d['hit_rate']:.2f}%")

    print("\nBOTTOM 10 DOWNWARD MESSAGES (lowest hit rate):")
    print(f"{'Clique Edge':<12} {'Complement':<12} {'Hit Rate':<10}")
    print("-"*34)
    for d in downward_sorted[-10:]:
        print(f"{d['clique_edge']:<12} {d['complement_size']:<12} {d['hit_rate']:.2f}%")

def main():
    print("Loading cache statistics from cache_statistics.csv...")
    data = load_data('cache_statistics.csv')
    print(f"Loaded {len(data)} entries\n")
    analyze_data(data)

if __name__ == "__main__":
    main()
