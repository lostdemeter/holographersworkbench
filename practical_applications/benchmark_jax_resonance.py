#!/usr/bin/env python3
"""
Benchmark JAX-Accelerated Resonance Computation
================================================

Tests the speedup from using JAX for resonance field and strength
computation (NOT for gradients - those don't help discrete TSP).

Expected results:
- 2-3× speedup on resonance field computation for N >= 100
- 2× speedup on resonance strength computation for N >= 50
- No change in tour quality (same algorithm, just faster)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workbench.processors.sublinear_clock_v2 import (
    SublinearClockOptimizerV2,
    solve_tsp_clock_v2,
    JAX_AVAILABLE,
    _jax_resonance_strength,
    _jax_batch_resonance_field,
    CLOCK_RATIOS_6D,
    LazyClockOracle
)

if JAX_AVAILABLE:
    import jax.numpy as jnp
    print("JAX available: True")
else:
    print("JAX available: False (using NumPy fallback)")


def benchmark_resonance_field(n_cities: int, n_phases: int = 20, n_trials: int = 5):
    """Benchmark resonance field computation."""
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2)
    
    # Normalize cities
    cities_norm = cities - cities.min(axis=0)
    scale = cities_norm.max()
    if scale > 1e-10:
        cities_norm = cities_norm / scale
    
    # Pre-compute phases
    oracle = LazyClockOracle()
    n_clocks = 6
    clock_names = list(CLOCK_RATIOS_6D.keys())
    all_clock_phases = np.zeros((n_clocks, n_phases))
    for clock_idx, clock_name in enumerate(clock_names):
        for phase_n in range(1, n_phases + 1):
            all_clock_phases[clock_idx, phase_n - 1] = oracle.get_fractional_phase(phase_n, clock_name)
    
    projection_angles = np.array([np.pi * i / n_clocks for i in range(n_clocks)])
    
    # NumPy version
    times_numpy = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result_np = _jax_batch_resonance_field(
            cities_norm, projection_angles, all_clock_phases, n_clocks
        )
        times_numpy.append(time.perf_counter() - t0)
    
    # JAX version (if available)
    if JAX_AVAILABLE:
        cities_jax = jnp.array(cities_norm)
        angles_jax = jnp.array(projection_angles)
        phases_jax = jnp.array(all_clock_phases)
        
        # Warm-up JIT
        _ = _jax_batch_resonance_field(cities_jax, angles_jax, phases_jax, n_clocks)
        
        times_jax = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result_jax = _jax_batch_resonance_field(
                cities_jax, angles_jax, phases_jax, n_clocks
            )
            result_jax.block_until_ready()  # Ensure computation is complete
            times_jax.append(time.perf_counter() - t0)
        
        # Check results match
        diff = np.abs(np.array(result_jax) - result_np).max()
        
        return {
            'numpy_time': np.mean(times_numpy),
            'jax_time': np.mean(times_jax),
            'speedup': np.mean(times_numpy) / np.mean(times_jax),
            'max_diff': diff
        }
    else:
        return {
            'numpy_time': np.mean(times_numpy),
            'jax_time': None,
            'speedup': 1.0,
            'max_diff': 0.0
        }


def benchmark_resonance_strength(n_cities: int, n_phases: int = 20, n_trials: int = 10):
    """Benchmark resonance strength computation."""
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2)
    tour = np.arange(n_cities)
    np.random.shuffle(tour)
    
    tour_cities = cities[tour]
    edges = np.diff(tour_cities, axis=0, append=tour_cities[0:1])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles_norm = (angles / (2 * np.pi)) % 1.0
    
    # Pre-compute phases
    oracle = LazyClockOracle()
    all_phases = []
    for clock_name in CLOCK_RATIOS_6D:
        for phase_n in range(1, min(n_phases + 1, 20)):
            all_phases.append(oracle.get_fractional_phase(phase_n, clock_name))
    all_phases = np.array(all_phases)
    
    # NumPy version
    times_numpy = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result_np = _jax_resonance_strength(angles_norm, all_phases)
        times_numpy.append(time.perf_counter() - t0)
    
    # JAX version (if available)
    if JAX_AVAILABLE:
        angles_jax = jnp.array(angles_norm)
        phases_jax = jnp.array(all_phases)
        
        # Warm-up JIT
        _ = _jax_resonance_strength(angles_jax, phases_jax)
        
        times_jax = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result_jax = float(_jax_resonance_strength(angles_jax, phases_jax))
            times_jax.append(time.perf_counter() - t0)
        
        diff = abs(result_jax - result_np)
        
        return {
            'numpy_time': np.mean(times_numpy),
            'jax_time': np.mean(times_jax),
            'speedup': np.mean(times_numpy) / np.mean(times_jax),
            'max_diff': diff
        }
    else:
        return {
            'numpy_time': np.mean(times_numpy),
            'jax_time': None,
            'speedup': 1.0,
            'max_diff': 0.0
        }


def benchmark_full_optimization(n_cities: int, n_trials: int = 3):
    """Benchmark full TSP optimization."""
    results = []
    
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        cities = np.random.rand(n_cities, 2) * 100
        
        times = []
        lengths = []
        
        for _ in range(n_trials):
            t0 = time.perf_counter()
            tour, length, stats = solve_tsp_clock_v2(cities)
            times.append(time.perf_counter() - t0)
            lengths.append(length)
        
        results.append({
            'seed': seed,
            'avg_time': np.mean(times),
            'avg_length': np.mean(lengths),
            'resonance': stats.resonance_strength
        })
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("JAX-Accelerated Resonance Benchmark")
    print("=" * 70)
    
    # Test resonance field computation
    print("\n1. Resonance Field Computation (6D)")
    print("-" * 50)
    for n in [50, 100, 200, 500, 1000]:
        result = benchmark_resonance_field(n)
        if result['jax_time'] is not None:
            print(f"  N={n:4d}: NumPy={result['numpy_time']*1000:.2f}ms, "
                  f"JAX={result['jax_time']*1000:.2f}ms, "
                  f"Speedup={result['speedup']:.2f}×, "
                  f"MaxDiff={result['max_diff']:.2e}")
        else:
            print(f"  N={n:4d}: NumPy={result['numpy_time']*1000:.2f}ms (JAX not available)")
    
    # Test resonance strength computation
    print("\n2. Resonance Strength Computation")
    print("-" * 50)
    for n in [30, 50, 100, 200, 500]:
        result = benchmark_resonance_strength(n)
        if result['jax_time'] is not None:
            print(f"  N={n:4d}: NumPy={result['numpy_time']*1000:.3f}ms, "
                  f"JAX={result['jax_time']*1000:.3f}ms, "
                  f"Speedup={result['speedup']:.2f}×, "
                  f"MaxDiff={result['max_diff']:.2e}")
        else:
            print(f"  N={n:4d}: NumPy={result['numpy_time']*1000:.3f}ms (JAX not available)")
    
    # Test full optimization
    print("\n3. Full TSP Optimization")
    print("-" * 50)
    for n in [50, 100, 200]:
        print(f"\n  N={n}:")
        results = benchmark_full_optimization(n, n_trials=1)
        for r in results:
            print(f"    Seed {r['seed']}: Length={r['avg_length']:.2f}, "
                  f"Time={r['avg_time']:.3f}s, Resonance={r['resonance']:.4f}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 50)
    if JAX_AVAILABLE:
        print("✓ JAX acceleration enabled for resonance computation")
        print("✓ Gradients NOT used (discrete TSP doesn't benefit)")
        print("✓ Expected speedup: 2-3× on resonance field, 2× on strength")
    else:
        print("✗ JAX not available - using NumPy fallback")
        print("  Install JAX for 2-3× speedup on resonance computation")
    print("=" * 70)
