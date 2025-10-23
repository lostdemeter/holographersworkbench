#!/usr/bin/env python3
"""
Benchmark script to compare TSP optimization performance.

Tests different optimization levels:
- Baseline (no optimizations)
- Fast (all optimizations enabled)
- Individual optimizations

Measures speedup and quality preservation.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workbench.primitives import QuantumFolder


def generate_test_instance(n_cities: int, seed: int = 42) -> np.ndarray:
    """Generate random TSP instance."""
    np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100


def greedy_nearest_neighbor(cities: np.ndarray) -> list:
    """Simple greedy construction."""
    n = len(cities)
    tour = [0]
    unvisited = set(range(1, n))
    
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda c: np.linalg.norm(
            cities[current] - cities[c]
        ))
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour


def tour_length(cities: np.ndarray, tour: list) -> float:
    """Compute tour length."""
    length = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
    return length


def benchmark_configuration(
    cities: np.ndarray,
    initial_tour: list,
    config_name: str,
    use_vectorized: bool,
    use_fast_mds: bool,
    use_adaptive: bool,
    use_early_stop: bool,
    n_runs: int = 3
) -> dict:
    """Benchmark a specific configuration."""
    folder = QuantumFolder()
    
    times = []
    lengths = []
    
    for run in range(n_runs):
        start = time.time()
        tour, length, info = folder.optimize_tour_dimensional_folding_fast(
            cities,
            initial_tour,
            n_restarts=2,
            iterations_per_restart=20,
            use_vectorized=use_vectorized,
            use_fast_mds=use_fast_mds,
            use_adaptive_sampling=use_adaptive,
            use_early_stopping=use_early_stop
        )
        elapsed = time.time() - start
        
        times.append(elapsed)
        lengths.append(length)
    
    return {
        'config': config_name,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'vectorized': use_vectorized,
        'fast_mds': use_fast_mds,
        'adaptive': use_adaptive,
        'early_stop': use_early_stop
    }


def main():
    print("="*80)
    print("TSP OPTIMIZATION BENCHMARK")
    print("="*80)
    
    # Test configurations
    sizes = [20, 40, 60, 80]
    
    for n_cities in sizes:
        print(f"\n{'='*80}")
        print(f"Testing with {n_cities} cities")
        print(f"{'='*80}\n")
        
        # Generate instance
        cities = generate_test_instance(n_cities)
        initial_tour = greedy_nearest_neighbor(cities)
        baseline_length = tour_length(cities, initial_tour)
        
        print(f"Baseline tour length: {baseline_length:.2f}\n")
        
        # Test configurations
        configs = [
            ("Baseline (original)", False, False, False, False),
            ("Vectorized only", True, False, False, False),
            ("Fast MDS only", False, True, False, False),
            ("Adaptive only", False, False, True, False),
            ("Early stop only", False, False, False, True),
            ("All optimizations", True, True, True, True),
        ]
        
        results = []
        for config_name, vec, mds, adp, early in configs:
            result = benchmark_configuration(
                cities, initial_tour, config_name,
                vec, mds, adp, early, n_runs=3
            )
            results.append(result)
        
        # Print results
        print(f"{'Configuration':<25} {'Time (s)':<12} {'Speedup':<10} {'Quality':<12}")
        print("-"*80)
        
        baseline_time = results[0]['avg_time']
        
        for r in results:
            speedup = baseline_time / r['avg_time']
            improvement = (baseline_length - r['avg_length']) / baseline_length * 100
            
            print(f"{r['config']:<25} {r['avg_time']:>8.4f}±{r['std_time']:.4f}  "
                  f"{speedup:>6.2f}×    {improvement:>6.2f}%")
        
        print()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
