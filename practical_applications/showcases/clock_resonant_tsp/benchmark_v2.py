#!/usr/bin/env python3
"""
Benchmark: Clock-Resonant v2 vs v1 vs Original
===============================================

Compares all three optimizer versions to demonstrate v2 improvements.
"""

import numpy as np
import time
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from workbench.processors.sublinear_qik import SublinearQIK
from workbench.processors.sublinear_clock import SublinearClockOptimizer
from workbench.processors.sublinear_clock_v2 import SublinearClockOptimizerV2
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


def compute_mst_bound(cities: np.ndarray) -> float:
    dist_matrix = squareform(pdist(cities))
    mst = minimum_spanning_tree(dist_matrix)
    return mst.sum()


def benchmark(n_cities: int, n_trials: int = 3, seed: int = 42):
    """Run benchmark on a single size."""
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2)
    lb = compute_mst_bound(cities)
    
    results = {'n': n_cities, 'lb': lb}
    
    # Original SublinearQIK
    original = SublinearQIK()
    lengths = []
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        _, length, _ = original.optimize_tsp(cities)
        times.append(time.time() - t0)
        lengths.append(length)
    results['original'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - lb) / lb
    }
    
    # Clock v1
    clock_v1 = SublinearClockOptimizer(use_multi_clock=True)
    lengths = []
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        _, length, _ = clock_v1.optimize_tsp(cities)
        times.append(time.time() - t0)
        lengths.append(length)
    results['clock_v1'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - lb) / lb
    }
    
    # Clock v2 (all features)
    clock_v2 = SublinearClockOptimizerV2(
        use_6d_tensor=True,
        use_gradient_flow=True,
        use_adaptive_dimension=True
    )
    lengths = []
    times = []
    stats_list = []
    for _ in range(n_trials):
        t0 = time.time()
        _, length, stats = clock_v2.optimize_tsp(cities)
        times.append(time.time() - t0)
        lengths.append(length)
        stats_list.append(stats)
    results['clock_v2'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - lb) / lb,
        'instance_dim': stats_list[0].instance_dimension,
        'clusters': stats_list[0].n_clusters,
        'resonance': stats_list[0].resonance_strength
    }
    
    # Clock v2 (no gradient flow - faster)
    clock_v2_fast = SublinearClockOptimizerV2(
        use_6d_tensor=True,
        use_gradient_flow=False,
        use_adaptive_dimension=True
    )
    lengths = []
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        _, length, _ = clock_v2_fast.optimize_tsp(cities)
        times.append(time.time() - t0)
        lengths.append(length)
    results['clock_v2_fast'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - lb) / lb
    }
    
    return results


def main():
    print("=" * 80)
    print("Clock-Resonant Optimizer Benchmark: v2 vs v1 vs Original")
    print("=" * 80)
    
    sizes = [20, 50, 100, 200]
    all_results = []
    
    for n in sizes:
        print(f"\nTesting N={n}...")
        results = benchmark(n, n_trials=3)
        all_results.append(results)
        
        print(f"\n  MST lower bound: {results['lb']:.4f}")
        print(f"\n  {'Method':<20} {'Best':>10} {'Gap %':>10} {'Time':>10}")
        print("  " + "-" * 55)
        
        for method in ['original', 'clock_v1', 'clock_v2', 'clock_v2_fast']:
            r = results[method]
            print(f"  {method:<20} {r['best']:>10.4f} {r['gap']:>10.2f}% {r['time']:>10.4f}s")
            
        if 'instance_dim' in results['clock_v2']:
            print(f"\n  v2 diagnostics: dim={results['clock_v2']['instance_dim']:.3f}, "
                  f"clusters={results['clock_v2']['clusters']}, "
                  f"resonance={results['clock_v2']['resonance']:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Gap % (lower is better)")
    print("=" * 80)
    print(f"\n{'N':>6} {'Original':>12} {'Clock v1':>12} {'Clock v2':>12} {'v2 Fast':>12} {'v2 Improv':>12}")
    print("-" * 70)
    
    for results in all_results:
        n = results['n']
        orig = results['original']['gap']
        v1 = results['clock_v1']['gap']
        v2 = results['clock_v2']['gap']
        v2f = results['clock_v2_fast']['gap']
        improvement = 100 * (orig - v2) / orig
        
        print(f"{n:>6} {orig:>12.2f} {v1:>12.2f} {v2:>12.2f} {v2f:>12.2f} {improvement:>11.1f}%")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
