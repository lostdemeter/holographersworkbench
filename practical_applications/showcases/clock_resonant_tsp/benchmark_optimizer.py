#!/usr/bin/env python3
"""
Benchmark: Clock-Resonant vs Original Sublinear Optimizer
==========================================================

Compares the new clock-resonant optimizer against the original SublinearQIK
to demonstrate the improvement from using exact clock eigenphases.

Usage:
    python benchmark_clock_optimizer.py
    python benchmark_clock_optimizer.py --n-cities 200 --n-trials 10
"""

import numpy as np
import time
import argparse
from typing import Tuple, List, Dict
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from workbench.processors.sublinear_qik import SublinearQIK
from workbench.processors.sublinear_clock import SublinearClockOptimizer, CLOCK_AVAILABLE


def generate_test_instances(n_cities: int, instance_type: str = 'random', seed: int = 42) -> np.ndarray:
    """Generate test TSP instances."""
    np.random.seed(seed)
    
    if instance_type == 'random':
        return np.random.rand(n_cities, 2)
    elif instance_type == 'clustered':
        # 5 clusters
        n_clusters = 5
        centers = np.random.rand(n_clusters, 2)
        cities = []
        for i in range(n_cities):
            center = centers[i % n_clusters]
            city = center + 0.1 * np.random.randn(2)
            cities.append(city)
        return np.array(cities)
    elif instance_type == 'grid':
        side = int(np.ceil(np.sqrt(n_cities)))
        x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        cities = np.column_stack([x.ravel(), y.ravel()])[:n_cities]
        return cities + 0.02 * np.random.randn(n_cities, 2)
    else:
        return np.random.rand(n_cities, 2)


def compute_optimal_lower_bound(cities: np.ndarray) -> float:
    """Compute a lower bound on optimal tour length (MST-based)."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    dist_matrix = squareform(pdist(cities))
    mst = minimum_spanning_tree(dist_matrix)
    return mst.sum()


def benchmark_single(
    cities: np.ndarray,
    n_trials: int = 5
) -> Dict:
    """Run benchmark on a single instance."""
    
    results = {
        'n_cities': len(cities),
        'lower_bound': compute_optimal_lower_bound(cities),
        'original': {'lengths': [], 'times': []},
        'clock': {'lengths': [], 'times': []},
        'clock_multi': {'lengths': [], 'times': []},
    }
    
    # Original SublinearQIK
    original = SublinearQIK(
        use_hierarchical=True,
        use_dimensional_sketch=True,
        use_sparse_resonance=True
    )
    
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, stats = original.optimize_tsp(cities)
        t1 = time.time()
        results['original']['lengths'].append(length)
        results['original']['times'].append(t1 - t0)
        
    # Clock-resonant (single clock)
    clock_single = SublinearClockOptimizer(
        use_hierarchical=True,
        use_multi_clock=False,
        clocks=['golden']
    )
    
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, stats = clock_single.optimize_tsp(cities)
        t1 = time.time()
        results['clock']['lengths'].append(length)
        results['clock']['times'].append(t1 - t0)
        
    # Clock-resonant (multi-clock)
    clock_multi = SublinearClockOptimizer(
        use_hierarchical=True,
        use_multi_clock=True,
        clocks=['golden', 'silver', 'bronze']
    )
    
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, stats = clock_multi.optimize_tsp(cities)
        t1 = time.time()
        results['clock_multi']['lengths'].append(length)
        results['clock_multi']['times'].append(t1 - t0)
        
    return results


def print_results(results: Dict):
    """Print benchmark results."""
    n = results['n_cities']
    lb = results['lower_bound']
    
    print(f"\n{'='*70}")
    print(f"N = {n} cities, MST lower bound = {lb:.4f}")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<25} {'Best Length':>12} {'Mean Length':>12} {'Mean Time':>12} {'Gap %':>10}")
    print("-" * 70)
    
    for method in ['original', 'clock', 'clock_multi']:
        lengths = results[method]['lengths']
        times = results[method]['times']
        
        best = min(lengths)
        mean = np.mean(lengths)
        mean_time = np.mean(times)
        gap = 100 * (best - lb) / lb
        
        method_name = {
            'original': 'SublinearQIK (original)',
            'clock': 'Clock-Resonant (golden)',
            'clock_multi': 'Clock-Resonant (multi)',
        }[method]
        
        print(f"{method_name:<25} {best:>12.4f} {mean:>12.4f} {mean_time:>12.4f}s {gap:>9.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Benchmark clock-resonant optimizer')
    parser.add_argument('--n-cities', type=int, default=100, help='Number of cities')
    parser.add_argument('--n-trials', type=int, default=5, help='Number of trials per method')
    parser.add_argument('--instance-type', type=str, default='random',
                       choices=['random', 'clustered', 'grid'])
    parser.add_argument('--scaling-test', action='store_true',
                       help='Run scaling test with multiple sizes')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Clock-Resonant Optimizer Benchmark")
    print("=" * 70)
    print(f"Clock downcaster available: {CLOCK_AVAILABLE}")
    
    if args.scaling_test:
        # Test multiple sizes
        sizes = [20, 50, 100, 200, 500]
        
        print(f"\nScaling test: {sizes}")
        print(f"Instance type: {args.instance_type}")
        
        all_results = []
        for n in sizes:
            print(f"\nTesting N={n}...")
            cities = generate_test_instances(n, args.instance_type)
            results = benchmark_single(cities, n_trials=3)
            all_results.append(results)
            print_results(results)
            
        # Summary
        print("\n" + "=" * 70)
        print("SCALING SUMMARY")
        print("=" * 70)
        print(f"\n{'N':>6} {'Original':>12} {'Clock':>12} {'Multi-Clock':>12} {'Improvement':>12}")
        print("-" * 60)
        
        for results in all_results:
            n = results['n_cities']
            orig = min(results['original']['lengths'])
            clock = min(results['clock']['lengths'])
            multi = min(results['clock_multi']['lengths'])
            best_clock = min(clock, multi)
            improvement = 100 * (orig - best_clock) / orig
            
            print(f"{n:>6} {orig:>12.4f} {clock:>12.4f} {multi:>12.4f} {improvement:>11.2f}%")
            
    else:
        # Single test
        print(f"\nSingle test: N={args.n_cities}, type={args.instance_type}")
        cities = generate_test_instances(args.n_cities, args.instance_type)
        results = benchmark_single(cities, n_trials=args.n_trials)
        print_results(results)
        
    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
