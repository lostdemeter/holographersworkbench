#!/usr/bin/env python3
"""
Large-Scale TSP Benchmark for Clock-Resonant Optimizer v2
==========================================================

Tests the optimizer on larger instances (200-2000 cities) to validate
scalability and verify Grok's claim of <8% gap on 1000+ cities.

This benchmark uses random Euclidean instances since embedding
full TSPLIB coordinates for pr1002 (1002 cities) would be impractical.

Expected results (from Grok's analysis):
- N=200: ~6% gap
- N=500: ~7% gap  
- N=1000: <8% gap in <5s
- N=2000: <10% gap in <15s
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workbench.processors.sublinear_clock_v2 import (
    SublinearClockOptimizerV2,
    solve_tsp_clock_v2,
    JAX_AVAILABLE
)


def compute_greedy_tour(cities: np.ndarray) -> tuple:
    """Compute greedy nearest-neighbor tour as baseline."""
    n = len(cities)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: np.linalg.norm(cities[current] - cities[x]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # Compute length
    length = sum(np.linalg.norm(cities[tour[i]] - cities[tour[(i+1) % n]]) 
                 for i in range(n))
    
    return np.array(tour), length


def compute_held_karp_lower_bound(cities: np.ndarray) -> float:
    """
    Estimate lower bound using full MST + minimum matching approximation.
    
    For random Euclidean TSP, the expected optimal is approximately:
    0.7124 * sqrt(n * A) where A is the area.
    
    This is the Beardwood-Halton-Hammersley theorem.
    """
    n = len(cities)
    
    # For 1000x1000 grid, area = 1e6
    # Expected optimal ≈ 0.7124 * sqrt(n * 1e6) = 712.4 * sqrt(n)
    # This is a statistical estimate for random uniform instances
    
    # Use MST as a more concrete lower bound
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist
    
    dist_matrix = cdist(cities, cities)
    mst = minimum_spanning_tree(dist_matrix)
    mst_cost = mst.sum()
    
    # MST is a lower bound for TSP (TSP >= MST)
    # A tighter bound is MST + min edge, but MST alone is valid
    return mst_cost


def compute_bhh_expected(n: int, area: float = 1e6) -> float:
    """
    Beardwood-Halton-Hammersley expected optimal tour length.
    
    For n points uniformly distributed in area A:
    E[L] ≈ 0.7124 * sqrt(n * A)
    """
    return 0.7124 * np.sqrt(n * area)


def benchmark_size(n: int, n_trials: int = 3, seed_base: int = 42):
    """Benchmark clock v2 on random instances of size n."""
    print(f"\n{'='*60}")
    print(f"N = {n} cities")
    print(f"{'='*60}")
    
    results = []
    
    for trial in range(n_trials):
        seed = seed_base + trial * 100
        np.random.seed(seed)
        
        # Generate random Euclidean instance
        cities = np.random.rand(n, 2) * 1000  # 1000x1000 grid
        
        # Greedy baseline
        t0 = time.time()
        greedy_tour, greedy_length = compute_greedy_tour(cities)
        greedy_time = time.time() - t0
        
        # Lower bound estimates
        mst_lb = compute_held_karp_lower_bound(cities)
        bhh_expected = compute_bhh_expected(n)
        
        # Clock v2
        t0 = time.time()
        tour, length, stats = solve_tsp_clock_v2(cities)
        v2_time = time.time() - t0
        
        # Compute gaps (vs BHH expected, which is more realistic)
        greedy_gap = 100 * (greedy_length - bhh_expected) / bhh_expected
        v2_gap = 100 * (length - bhh_expected) / bhh_expected
        improvement = 100 * (greedy_length - length) / greedy_length
        
        results.append({
            'seed': seed,
            'greedy_length': greedy_length,
            'greedy_time': greedy_time,
            'v2_length': length,
            'v2_time': v2_time,
            'mst_lb': mst_lb,
            'bhh_expected': bhh_expected,
            'greedy_gap': greedy_gap,
            'v2_gap': v2_gap,
            'improvement': improvement,
            'resonance': stats.resonance_strength,
            'clusters': stats.n_clusters,
            'dim': stats.instance_dimension
        })
        
        print(f"  Trial {trial+1}: v2={length:.1f} ({v2_gap:.1f}% gap), "
              f"greedy={greedy_length:.1f} ({greedy_gap:.1f}% gap), "
              f"time={v2_time:.2f}s, improve={improvement:.1f}%")
    
    # Summary
    avg_v2_gap = np.mean([r['v2_gap'] for r in results])
    avg_greedy_gap = np.mean([r['greedy_gap'] for r in results])
    avg_time = np.mean([r['v2_time'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    
    print(f"\n  Summary: v2 gap={avg_v2_gap:.2f}%, greedy gap={avg_greedy_gap:.2f}%, "
          f"time={avg_time:.2f}s, improvement={avg_improvement:.1f}%")
    
    return {
        'n': n,
        'avg_v2_gap': avg_v2_gap,
        'avg_greedy_gap': avg_greedy_gap,
        'avg_time': avg_time,
        'avg_improvement': avg_improvement,
        'trials': results
    }


def main():
    print("=" * 70)
    print("LARGE-SCALE TSP BENCHMARK: Clock-Resonant Optimizer v2")
    print("=" * 70)
    print(f"JAX acceleration: {'ENABLED' if JAX_AVAILABLE else 'DISABLED'}")
    
    # Test sizes
    sizes = [100, 200, 500, 1000]
    
    all_results = {}
    for n in sizes:
        all_results[n] = benchmark_size(n, n_trials=3)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'N':>6} {'v2 Gap':>10} {'Greedy Gap':>12} {'Time':>10} {'Improvement':>12}")
    print("-" * 55)
    
    for n, results in all_results.items():
        print(f"{n:>6} {results['avg_v2_gap']:>10.2f}% {results['avg_greedy_gap']:>12.2f}% "
              f"{results['avg_time']:>10.2f}s {results['avg_improvement']:>11.1f}%")
    
    print("\n" + "=" * 70)
    print("Grok's targets:")
    print("  - N=1000: <8% gap in <5s")
    print("  - N=2000: <10% gap in <15s")
    
    # Check if we meet targets
    if 1000 in all_results:
        r = all_results[1000]
        gap_ok = r['avg_v2_gap'] < 8
        time_ok = r['avg_time'] < 5
        print(f"\nN=1000 results: gap={r['avg_v2_gap']:.2f}% {'✓' if gap_ok else '✗'}, "
              f"time={r['avg_time']:.2f}s {'✓' if time_ok else '✗'}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
