#!/usr/bin/env python3
"""
Benchmark: Dimensional Downcasting Integration
===============================================

Compares DD-enhanced methods vs pure Workbench methods for:
1. Zeta zero computation (speed + accuracy)
2. TSP optimization (gap + time)
3. Gushurst Crystal predictions

Expected gains:
- Zeta: 20-30% faster on large batches
- TSP: 5-10% better gaps via manifold projection
- Gushurst: 35% faster extrapolation
"""

import numpy as np
import time
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from workbench.core import (
    zetazero_batch,
    is_dd_available,
)

if is_dd_available():
    from workbench.core import (
        ZetaDowncaster,
        DowncastTSP,
        zetazero_batch_dd,
        solve_tsp_downcast,
    )

from workbench.processors.sublinear_clock_v2 import solve_tsp_clock_v2


def benchmark_zeta(n_zeros: int = 50):
    """Compare zeta zero computation methods."""
    print("\n" + "=" * 60)
    print(f"ZETA ZERO BENCHMARK: {n_zeros} zeros")
    print("=" * 60)
    
    # Workbench method
    print("\n1. Workbench (hybrid fractal-Newton):")
    t0 = time.time()
    wb_zeros = zetazero_batch(1, n_zeros)
    wb_time = time.time() - t0
    print(f"   Time: {wb_time:.3f}s ({wb_time/n_zeros*1000:.1f}ms/zero)")
    
    if is_dd_available():
        # DD method
        print("\n2. Dimensional Downcasting:")
        t0 = time.time()
        dd_zeros = zetazero_batch_dd(1, n_zeros)
        dd_time = time.time() - t0
        print(f"   Time: {dd_time:.3f}s ({dd_time/n_zeros*1000:.1f}ms/zero)")
        
        # Compare
        speedup = wb_time / dd_time
        print(f"\n   Speedup: {speedup:.2f}x {'(DD faster)' if speedup > 1 else '(WB faster)'}")
        
        # Accuracy comparison
        print("\n   Accuracy comparison (first 5 zeros):")
        for i in range(min(5, n_zeros)):
            wb_val = float(wb_zeros[i+1])
            dd_val = dd_zeros[i]
            diff = abs(wb_val - dd_val)
            print(f"   n={i+1}: WB={wb_val:.10f}, DD={dd_val:.10f}, diff={diff:.2e}")
    else:
        print("\n   DD not available - skipping comparison")


def benchmark_tsp(sizes: list = [50, 100]):
    """Compare TSP optimization methods."""
    print("\n" + "=" * 60)
    print("TSP BENCHMARK: Clock v2 vs Downcast TSP")
    print("=" * 60)
    
    # BHH expected for gap calculation
    def bhh_expected(n):
        return 0.7124 * np.sqrt(n * 1e6)
    
    for n in sizes:
        print(f"\n--- N = {n} cities ---")
        
        np.random.seed(42)
        cities = np.random.rand(n, 2) * 1000
        
        # Clock v2
        t0 = time.time()
        tour1, length1, stats1 = solve_tsp_clock_v2(cities)
        time1 = time.time() - t0
        gap1 = 100 * (length1 - bhh_expected(n)) / bhh_expected(n)
        print(f"   Clock v2:     length={length1:.1f}, gap={gap1:.1f}%, time={time1:.3f}s")
        
        if is_dd_available():
            # Downcast TSP
            t0 = time.time()
            tour2, length2, stats2 = solve_tsp_downcast(cities)
            time2 = time.time() - t0
            gap2 = 100 * (length2 - bhh_expected(n)) / bhh_expected(n)
            print(f"   Downcast TSP: length={length2:.1f}, gap={gap2:.1f}%, time={time2:.3f}s")
            
            # Improvement
            improvement = 100 * (length1 - length2) / length1
            print(f"   Improvement:  {improvement:.1f}%")
        else:
            print("   DD not available - skipping Downcast TSP")


def benchmark_predictor():
    """Compare predictor accuracy."""
    print("\n" + "=" * 60)
    print("PREDICTOR BENCHMARK: Ramanujan vs Clock-Seeded")
    print("=" * 60)
    
    if not is_dd_available():
        print("   DD not available - skipping")
        return
    
    from dimensional_downcasting.src.predictors import (
        RamanujanPredictor, ClockSeededPredictor
    )
    from mpmath import zetazero as mp_zetazero
    
    ram = RamanujanPredictor()
    clock = ClockSeededPredictor()
    
    test_ns = [10, 50, 100, 200, 500]
    
    print(f"\n{'n':>6} | {'True':>12} | {'Ramanujan':>12} | {'Clock':>12} | {'Ram Err':>10} | {'Clock Err':>10}")
    print("-" * 80)
    
    ram_errors = []
    clock_errors = []
    
    for n in test_ns:
        t_true = float(mp_zetazero(n).imag)
        t_ram = ram.predict(n)
        t_clock = clock.predict(n)
        
        err_ram = abs(t_ram - t_true)
        err_clock = abs(t_clock - t_true)
        
        ram_errors.append(err_ram)
        clock_errors.append(err_clock)
        
        print(f"{n:>6} | {t_true:>12.4f} | {t_ram:>12.4f} | {t_clock:>12.4f} | {err_ram:>10.4f} | {err_clock:>10.4f}")
    
    print("-" * 80)
    print(f"{'AVG':>6} | {'':>12} | {'':>12} | {'':>12} | {np.mean(ram_errors):>10.4f} | {np.mean(clock_errors):>10.4f}")
    
    improvement = 100 * (np.mean(ram_errors) - np.mean(clock_errors)) / np.mean(ram_errors)
    print(f"\nClock-seeded improvement: {improvement:.1f}%")


def main():
    print("=" * 60)
    print("DIMENSIONAL DOWNCASTING INTEGRATION BENCHMARK")
    print("=" * 60)
    print(f"DD Available: {is_dd_available()}")
    
    # Run benchmarks
    benchmark_zeta(n_zeros=20)  # Smaller for speed
    benchmark_tsp(sizes=[50, 100])
    benchmark_predictor()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
