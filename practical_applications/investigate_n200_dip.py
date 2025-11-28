#!/usr/bin/env python3
"""
Investigate the N=200 Performance Dip
=====================================

The clock-resonant optimizer underperforms at N=200 compared to the original.
This script investigates potential causes:
1. Random variation (test multiple seeds)
2. Clustering artifacts (empty/imbalanced clusters)
3. Parameter sensitivity (n_clusters, n_phases)
4. Resonance field quality
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workbench.processors.sublinear_qik import SublinearQIK
from workbench.processors.sublinear_clock import SublinearClockOptimizer, CLOCK_AVAILABLE
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


def compute_mst_bound(cities: np.ndarray) -> float:
    """Compute MST lower bound."""
    dist_matrix = squareform(pdist(cities))
    mst = minimum_spanning_tree(dist_matrix)
    return mst.sum()


def analyze_clustering(cities: np.ndarray, optimizer: SublinearClockOptimizer) -> Dict:
    """Analyze clustering behavior."""
    from scipy.cluster.vq import kmeans2
    
    n = len(cities)
    k = int(np.ceil(np.sqrt(n)))
    
    centroids, labels = kmeans2(cities, k, minit='points', iter=10)
    
    # Count cluster sizes
    cluster_sizes = []
    empty_clusters = 0
    for i in range(k):
        size = np.sum(labels == i)
        if size == 0:
            empty_clusters += 1
        cluster_sizes.append(size)
    
    return {
        'n_clusters_target': k,
        'n_clusters_actual': k - empty_clusters,
        'empty_clusters': empty_clusters,
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes),
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'imbalance_ratio': max(cluster_sizes) / (min([s for s in cluster_sizes if s > 0]) + 1e-10),
    }


def run_multi_seed_test(n_cities: int = 200, n_seeds: int = 10, n_trials: int = 3) -> List[Dict]:
    """Run benchmark across multiple seeds."""
    results = []
    
    for seed in range(42, 42 + n_seeds):
        np.random.seed(seed)
        cities = np.random.rand(n_cities, 2)
        lb = compute_mst_bound(cities)
        
        # Original
        original = SublinearQIK()
        orig_lengths = []
        for _ in range(n_trials):
            tour, length, _ = original.optimize_tsp(cities)
            orig_lengths.append(length)
        
        # Clock single
        clock_single = SublinearClockOptimizer(use_multi_clock=False, clocks=['golden'])
        clock_lengths = []
        for _ in range(n_trials):
            tour, length, stats = clock_single.optimize_tsp(cities)
            clock_lengths.append(length)
        
        # Clock multi
        clock_multi = SublinearClockOptimizer(use_multi_clock=True, clocks=['golden', 'silver', 'bronze'])
        multi_lengths = []
        multi_stats = None
        for _ in range(n_trials):
            tour, length, stats = clock_multi.optimize_tsp(cities)
            multi_lengths.append(length)
            multi_stats = stats
        
        # Analyze clustering
        cluster_info = analyze_clustering(cities, clock_multi)
        
        results.append({
            'seed': seed,
            'lb': lb,
            'original_best': min(orig_lengths),
            'original_mean': np.mean(orig_lengths),
            'clock_best': min(clock_lengths),
            'clock_mean': np.mean(clock_lengths),
            'multi_best': min(multi_lengths),
            'multi_mean': np.mean(multi_lengths),
            'original_gap': 100 * (min(orig_lengths) - lb) / lb,
            'clock_gap': 100 * (min(clock_lengths) - lb) / lb,
            'multi_gap': 100 * (min(multi_lengths) - lb) / lb,
            'cluster_info': cluster_info,
            'n_phases': multi_stats.n_clock_phases if multi_stats else 0,
            'resonance_strength': multi_stats.resonance_strength if multi_stats else 0,
        })
        
        # Print progress
        winner = 'ORIGINAL' if results[-1]['original_gap'] < results[-1]['multi_gap'] else 'CLOCK'
        print(f"Seed {seed}: Original {results[-1]['original_gap']:.2f}% | "
              f"Clock {results[-1]['clock_gap']:.2f}% | "
              f"Multi {results[-1]['multi_gap']:.2f}% | "
              f"Winner: {winner} | "
              f"Clusters: {cluster_info['n_clusters_actual']}/{cluster_info['n_clusters_target']} | "
              f"Imbalance: {cluster_info['imbalance_ratio']:.2f}")
    
    return results


def analyze_results(results: List[Dict]):
    """Analyze results to find patterns."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Count wins
    original_wins = sum(1 for r in results if r['original_gap'] < r['multi_gap'])
    clock_wins = len(results) - original_wins
    
    print(f"\nWin rate: Original {original_wins}/{len(results)}, Clock {clock_wins}/{len(results)}")
    
    # Average gaps
    avg_original = np.mean([r['original_gap'] for r in results])
    avg_clock = np.mean([r['clock_gap'] for r in results])
    avg_multi = np.mean([r['multi_gap'] for r in results])
    
    print(f"\nAverage gaps:")
    print(f"  Original: {avg_original:.2f}%")
    print(f"  Clock:    {avg_clock:.2f}%")
    print(f"  Multi:    {avg_multi:.2f}%")
    
    # Correlation with clustering
    imbalances = [r['cluster_info']['imbalance_ratio'] for r in results]
    multi_gaps = [r['multi_gap'] for r in results]
    
    corr = np.corrcoef(imbalances, multi_gaps)[0, 1]
    print(f"\nCorrelation (imbalance vs multi_gap): {corr:.3f}")
    
    # Correlation with resonance strength
    resonances = [r['resonance_strength'] for r in results]
    corr_res = np.corrcoef(resonances, multi_gaps)[0, 1]
    print(f"Correlation (resonance vs multi_gap): {corr_res:.3f}")
    
    # Find worst cases for clock
    worst_for_clock = sorted(results, key=lambda r: r['multi_gap'] - r['original_gap'], reverse=True)[:3]
    print(f"\nWorst cases for clock (multi_gap - original_gap):")
    for r in worst_for_clock:
        print(f"  Seed {r['seed']}: diff={r['multi_gap'] - r['original_gap']:.2f}%, "
              f"imbalance={r['cluster_info']['imbalance_ratio']:.2f}, "
              f"empty={r['cluster_info']['empty_clusters']}, "
              f"resonance={r['resonance_strength']:.4f}")
    
    return results


def visualize_problem_instance(seed: int = 42, n_cities: int = 200):
    """Visualize the problematic instance."""
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2)
    
    # Get tours from both methods
    original = SublinearQIK()
    tour_orig, length_orig, _ = original.optimize_tsp(cities)
    
    clock = SublinearClockOptimizer(use_multi_clock=True, clocks=['golden', 'silver', 'bronze'])
    tour_clock, length_clock, stats = clock.optimize_tsp(cities)
    
    # Get clustering
    from scipy.cluster.vq import kmeans2
    k = int(np.ceil(np.sqrt(n_cities)))
    centroids, labels = kmeans2(cities, k, minit='points', iter=10)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Clusters
    ax = axes[0]
    scatter = ax.scatter(cities[:, 0], cities[:, 1], c=labels, cmap='tab20', s=20, alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, linewidths=2)
    ax.set_title(f'Clustering (k={k})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot 2: Original tour
    ax = axes[1]
    ax.scatter(cities[:, 0], cities[:, 1], c='blue', s=20, alpha=0.5)
    tour_cities = cities[tour_orig]
    ax.plot(np.append(tour_cities[:, 0], tour_cities[0, 0]),
            np.append(tour_cities[:, 1], tour_cities[0, 1]),
            'b-', alpha=0.5, linewidth=0.5)
    ax.set_title(f'Original Tour (length={length_orig:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot 3: Clock tour
    ax = axes[2]
    ax.scatter(cities[:, 0], cities[:, 1], c='green', s=20, alpha=0.5)
    tour_cities = cities[tour_clock]
    ax.plot(np.append(tour_cities[:, 0], tour_cities[0, 0]),
            np.append(tour_cities[:, 1], tour_cities[0, 1]),
            'g-', alpha=0.5, linewidth=0.5)
    ax.set_title(f'Clock Tour (length={length_clock:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(f'n200_seed{seed}_comparison.png', dpi=150)
    print(f"\nSaved visualization to n200_seed{seed}_comparison.png")
    
    return fig


def test_parameter_sensitivity(n_cities: int = 200, seed: int = 42):
    """Test sensitivity to n_clusters and n_phases."""
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2)
    lb = compute_mst_bound(cities)
    
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY TEST")
    print("=" * 80)
    
    # Test different cluster counts
    print("\nVarying n_clusters (with fixed n_phases=15):")
    for k in [5, 10, 14, 15, 20, 25]:
        # Manually set k by modifying the optimizer behavior
        class FixedClusterOptimizer(SublinearClockOptimizer):
            def __init__(self, fixed_k, **kwargs):
                super().__init__(**kwargs)
                self.fixed_k = fixed_k
                
            def optimize_tsp(self, cities, n_phases=15, verbose=False):
                # Override k calculation
                original_hierarchical = self.use_hierarchical
                self.use_hierarchical = True
                
                n = len(cities)
                k = self.fixed_k
                
                # Rest of the method...
                import time as t
                t0 = t.time()
                clusters, centroids = self._hierarchical_cluster(cities, k)
                t_cluster = t.time() - t0
                
                # Pre-compute phases
                clock_phases = self._precompute_clock_phases(n_phases)
                
                # Inter-cluster
                t0 = t.time()
                if len(centroids) > 1:
                    inter_tour = self._solve_inter_cluster(centroids, clock_phases)
                else:
                    inter_tour = np.array([0])
                t_inter = t.time() - t0
                
                # Intra-cluster
                t0 = t.time()
                tour = []
                for cluster_idx in inter_tour:
                    cluster_cities = cities[clusters[cluster_idx]]
                    if len(cluster_cities) > 1:
                        cluster_tour = self._solve_intra_cluster(cluster_cities, clock_phases)
                        global_indices = clusters[cluster_idx][cluster_tour]
                    else:
                        global_indices = clusters[cluster_idx]
                    tour.extend(global_indices)
                tour = np.array(tour)
                t_intra = t.time() - t0
                
                length = self._compute_tour_length(tour, cities)
                
                from workbench.processors.sublinear_clock import ClockResonanceStats
                stats = ClockResonanceStats(
                    n_cities=n, n_clusters=len(clusters), n_clock_phases=n_phases,
                    n_clocks_used=len(self.clocks), clock_eval_time=0,
                    clustering_time=t_cluster, inter_cluster_time=t_inter,
                    intra_cluster_time=t_intra, total_time=t_cluster+t_inter+t_intra,
                    theoretical_complexity="", empirical_complexity="",
                    tour_length=length, resonance_strength=0
                )
                
                return tour, length, stats
        
        opt = FixedClusterOptimizer(fixed_k=k, use_multi_clock=True, clocks=['golden', 'silver'])
        _, length, _ = opt.optimize_tsp(cities)
        gap = 100 * (length - lb) / lb
        print(f"  k={k:2d}: length={length:.2f}, gap={gap:.2f}%")
    
    # Test different phase counts
    print("\nVarying n_phases (with default clustering):")
    for n_phases in [5, 10, 15, 20, 30, 50]:
        opt = SublinearClockOptimizer(use_multi_clock=True, clocks=['golden', 'silver'])
        _, length, _ = opt.optimize_tsp(cities, n_phases=n_phases)
        gap = 100 * (length - lb) / lb
        print(f"  n_phases={n_phases:2d}: length={length:.2f}, gap={gap:.2f}%")


def main():
    print("=" * 80)
    print("INVESTIGATING N=200 PERFORMANCE DIP")
    print("=" * 80)
    print(f"Clock downcaster available: {CLOCK_AVAILABLE}")
    
    # Test 1: Multi-seed analysis
    print("\n" + "-" * 40)
    print("TEST 1: Multi-seed analysis (N=200)")
    print("-" * 40)
    results = run_multi_seed_test(n_cities=200, n_seeds=10, n_trials=3)
    analyze_results(results)
    
    # Test 2: Parameter sensitivity
    print("\n" + "-" * 40)
    print("TEST 2: Parameter sensitivity")
    print("-" * 40)
    test_parameter_sensitivity(n_cities=200, seed=42)
    
    # Test 3: Visualization
    print("\n" + "-" * 40)
    print("TEST 3: Visualization")
    print("-" * 40)
    visualize_problem_instance(seed=42, n_cities=200)
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
