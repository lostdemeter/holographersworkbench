#!/usr/bin/env python3
"""
Large-Scale TSPLIB Benchmark for Clock-Resonant Optimizer v2
============================================================

Tests the optimizer on larger TSPLIB instances (200-2000+ cities)
to validate scalability and real-world performance.

This script downloads instances from TSPLIB if not cached locally.

Target: <10% gap on 1000+ cities in <10 seconds
"""

import numpy as np
import time
import sys
import os
import urllib.request
import re
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workbench.processors.sublinear_clock_v2 import SublinearClockOptimizerV2, JAX_AVAILABLE

# TSPLIB instance metadata
# Format: (name, n_cities, optimal_length, url_suffix)
LARGE_INSTANCES = [
    # Medium (100-200)
    ('kroA100', 100, 21282, 'kroA100.tsp'),
    ('kroB100', 100, 22141, 'kroB100.tsp'),
    ('kroC100', 100, 20749, 'kroC100.tsp'),
    ('kroD100', 100, 21294, 'kroD100.tsp'),
    ('kroE100', 100, 22068, 'kroE100.tsp'),
    ('rd100', 100, 7910, 'rd100.tsp'),
    ('eil101', 101, 629, 'eil101.tsp'),
    ('lin105', 105, 14379, 'lin105.tsp'),
    ('pr107', 107, 44303, 'pr107.tsp'),
    ('pr124', 124, 59030, 'pr124.tsp'),
    ('bier127', 127, 118282, 'bier127.tsp'),
    ('ch130', 130, 6110, 'ch130.tsp'),
    ('pr136', 136, 96772, 'pr136.tsp'),
    ('pr144', 144, 58537, 'pr144.tsp'),
    ('ch150', 150, 6528, 'ch150.tsp'),
    ('kroA150', 150, 26524, 'kroA150.tsp'),
    ('kroB150', 150, 26130, 'kroB150.tsp'),
    ('pr152', 152, 73682, 'pr152.tsp'),
    ('u159', 159, 42080, 'u159.tsp'),
    ('rat195', 195, 2323, 'rat195.tsp'),
    ('d198', 198, 15780, 'd198.tsp'),
    ('kroA200', 200, 29368, 'kroA200.tsp'),
    ('kroB200', 200, 29437, 'kroB200.tsp'),
    
    # Large (200-500)
    ('ts225', 225, 126643, 'ts225.tsp'),
    ('tsp225', 225, 3916, 'tsp225.tsp'),
    ('pr226', 226, 80369, 'pr226.tsp'),
    ('gil262', 262, 2378, 'gil262.tsp'),
    ('pr264', 264, 49135, 'pr264.tsp'),
    ('a280', 280, 2579, 'a280.tsp'),
    ('pr299', 299, 48191, 'pr299.tsp'),
    ('lin318', 318, 42029, 'lin318.tsp'),
    ('rd400', 400, 15281, 'rd400.tsp'),
    ('fl417', 417, 11861, 'fl417.tsp'),
    ('pr439', 439, 107217, 'pr439.tsp'),
    ('pcb442', 442, 50778, 'pcb442.tsp'),
    ('d493', 493, 35002, 'd493.tsp'),
    
    # Very Large (500-1000)
    ('att532', 532, 27686, 'att532.tsp'),
    ('ali535', 535, 202339, 'ali535.tsp'),
    ('u574', 574, 36905, 'u574.tsp'),
    ('rat575', 575, 6773, 'rat575.tsp'),
    ('p654', 654, 34643, 'p654.tsp'),
    ('d657', 657, 48912, 'd657.tsp'),
    ('u724', 724, 41910, 'u724.tsp'),
    ('rat783', 783, 8806, 'rat783.tsp'),
    ('pr1002', 1002, 259045, 'pr1002.tsp'),
    
    # Huge (1000+)
    ('u1060', 1060, 224094, 'u1060.tsp'),
    ('vm1084', 1084, 239297, 'vm1084.tsp'),
    ('pcb1173', 1173, 56892, 'pcb1173.tsp'),
    ('d1291', 1291, 50801, 'd1291.tsp'),
    ('rl1304', 1304, 252948, 'rl1304.tsp'),
    ('rl1323', 1323, 270199, 'rl1323.tsp'),
    ('nrw1379', 1379, 56638, 'nrw1379.tsp'),
    ('fl1400', 1400, 20127, 'fl1400.tsp'),
    ('u1432', 1432, 152970, 'u1432.tsp'),
    ('fl1577', 1577, 22249, 'fl1577.tsp'),
    ('d1655', 1655, 62128, 'd1655.tsp'),
    ('vm1748', 1748, 336556, 'vm1748.tsp'),
    ('u1817', 1817, 57201, 'u1817.tsp'),
    ('rl1889', 1889, 316536, 'rl1889.tsp'),
    ('d2103', 2103, 80450, 'd2103.tsp'),
    ('u2152', 2152, 64253, 'u2152.tsp'),
    ('u2319', 2319, 234256, 'u2319.tsp'),
    ('pr2392', 2392, 378032, 'pr2392.tsp'),
]

# Cache directory for downloaded instances
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.tsplib_cache')
TSPLIB_BASE_URL = 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/'


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def download_instance(name: str, url_suffix: str) -> Optional[str]:
    """Download a TSPLIB instance if not cached."""
    ensure_cache_dir()
    cache_path = os.path.join(CACHE_DIR, f'{name}.tsp')
    
    if os.path.exists(cache_path):
        return cache_path
    
    url = TSPLIB_BASE_URL + url_suffix
    try:
        print(f"  Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, cache_path)
        return cache_path
    except Exception as e:
        print(f"  Failed to download {name}: {e}")
        return None


def parse_tsp_file(filepath: str) -> Optional[np.ndarray]:
    """Parse a TSPLIB .tsp file and return city coordinates."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find coordinate section
        coord_section = re.search(
            r'NODE_COORD_SECTION\s*\n(.*?)(?:EOF|$)',
            content,
            re.DOTALL
        )
        
        if not coord_section:
            return None
        
        coords = []
        for line in coord_section.group(1).strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
                except ValueError:
                    continue
        
        if len(coords) == 0:
            return None
        
        return np.array(coords)
    
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def generate_random_instance(n: int, seed: int = 42) -> np.ndarray:
    """Generate a random Euclidean TSP instance."""
    np.random.seed(seed)
    return np.random.rand(n, 2) * 1000


def benchmark_instance(
    name: str,
    cities: np.ndarray,
    optimal: int,
    n_trials: int = 1,
    use_jax: bool = True
) -> Dict:
    """Benchmark the optimizer on a single instance."""
    n = len(cities)
    
    # Create optimizer
    optimizer = SublinearClockOptimizerV2(
        use_hierarchical=True,
        use_6d_tensor=True,
        use_gradient_flow=True,
        use_adaptive_dimension=True,
        use_jax_autodiff=use_jax,
        gradient_flow_steps=3
    )
    
    best_length = float('inf')
    best_time = float('inf')
    total_time = 0
    
    for trial in range(n_trials):
        t0 = time.time()
        tour, length, stats = optimizer.optimize_tsp(cities, verbose=False)
        elapsed = time.time() - t0
        
        total_time += elapsed
        if length < best_length:
            best_length = length
            best_time = elapsed
    
    avg_time = total_time / n_trials
    gap = 100 * (best_length - optimal) / optimal
    
    return {
        'name': name,
        'n_cities': n,
        'optimal': optimal,
        'best_length': best_length,
        'gap': gap,
        'best_time': best_time,
        'avg_time': avg_time,
        'n_trials': n_trials
    }


def run_benchmark(
    max_cities: int = 500,
    n_trials: int = 1,
    use_jax: bool = True,
    include_random: bool = True
):
    """Run the full benchmark suite."""
    print("=" * 80)
    print("LARGE-SCALE TSPLIB BENCHMARK: Clock-Resonant Optimizer v2")
    print("=" * 80)
    print(f"\nJAX autodiff: {'ENABLED' if use_jax and JAX_AVAILABLE else 'DISABLED'}")
    print(f"Max cities: {max_cities}")
    print(f"Trials per instance: {n_trials}")
    print()
    
    results = []
    
    # Filter instances by size
    instances = [(n, o, s, u) for n, c, o, s, u in 
                 [(name, cities, opt, suffix, suffix) 
                  for name, cities, opt, suffix in LARGE_INSTANCES]
                 if c <= max_cities]
    
    # Actually, let me fix this - the tuple unpacking was wrong
    instances = [(name, n_cities, optimal, suffix) 
                 for name, n_cities, optimal, suffix in LARGE_INSTANCES
                 if n_cities <= max_cities]
    
    print(f"Running {len(instances)} TSPLIB instances...\n")
    
    for name, n_cities, optimal, suffix in instances:
        print(f"[{name}] n={n_cities}, optimal={optimal}")
        
        # Try to download/load instance
        filepath = download_instance(name, suffix)
        
        if filepath:
            cities = parse_tsp_file(filepath)
        else:
            cities = None
        
        if cities is None:
            print(f"  Skipping (could not load)")
            continue
        
        if len(cities) != n_cities:
            print(f"  Warning: expected {n_cities} cities, got {len(cities)}")
        
        result = benchmark_instance(name, cities, optimal, n_trials, use_jax)
        results.append(result)
        
        print(f"  Length: {result['best_length']:.0f} ({result['gap']:.2f}% gap)")
        print(f"  Time: {result['best_time']:.3f}s")
    
    # Add random instances for scalability testing
    if include_random:
        print("\nRandom Euclidean instances (for scalability):")
        for n in [100, 200, 500, 1000, 2000]:
            if n > max_cities:
                continue
            
            print(f"[random_{n}] n={n}")
            cities = generate_random_instance(n)
            
            # No known optimal, so we'll just measure time and length
            t0 = time.time()
            optimizer = SublinearClockOptimizerV2(
                use_hierarchical=True,
                use_6d_tensor=True,
                use_gradient_flow=True,
                use_adaptive_dimension=True,
                use_jax_autodiff=use_jax
            )
            tour, length, stats = optimizer.optimize_tsp(cities, verbose=False)
            elapsed = time.time() - t0
            
            print(f"  Length: {length:.0f}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Resonance: {stats.resonance_strength:.4f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Instance':<12} {'N':>6} {'Optimal':>10} {'Found':>10} {'Gap':>8} {'Time':>8}")
    print("-" * 60)
    
    total_gap = 0
    count = 0
    
    for r in results:
        print(f"{r['name']:<12} {r['n_cities']:>6} {r['optimal']:>10} "
              f"{r['best_length']:>10.0f} {r['gap']:>7.2f}% {r['best_time']:>7.3f}s")
        total_gap += r['gap']
        count += 1
    
    if count > 0:
        avg_gap = total_gap / count
        print("-" * 60)
        print(f"{'AVERAGE':<12} {'':<6} {'':<10} {'':<10} {avg_gap:>7.2f}%")
    
    # Performance tiers
    print("\n" + "=" * 80)
    print("PERFORMANCE TIERS")
    print("=" * 80)
    
    tiers = {
        'Excellent (<3%)': [r for r in results if r['gap'] < 3],
        'Good (3-5%)': [r for r in results if 3 <= r['gap'] < 5],
        'Acceptable (5-10%)': [r for r in results if 5 <= r['gap'] < 10],
        'Needs Work (>10%)': [r for r in results if r['gap'] >= 10]
    }
    
    for tier_name, tier_results in tiers.items():
        if tier_results:
            names = ', '.join(r['name'] for r in tier_results)
            print(f"\n{tier_name}: {len(tier_results)} instances")
            print(f"  {names}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Large-scale TSPLIB benchmark')
    parser.add_argument('--max-cities', type=int, default=500,
                        help='Maximum number of cities (default: 500)')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per instance (default: 1)')
    parser.add_argument('--no-jax', action='store_true',
                        help='Disable JAX autodiff')
    parser.add_argument('--no-random', action='store_true',
                        help='Skip random instances')
    
    args = parser.parse_args()
    
    run_benchmark(
        max_cities=args.max_cities,
        n_trials=args.trials,
        use_jax=not args.no_jax,
        include_random=not args.no_random
    )


if __name__ == "__main__":
    main()
