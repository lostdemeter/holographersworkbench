#!/usr/bin/env python3
"""
Benchmark: Riemann Zeta Zeros
==============================

Compare Gushurst Crystal (hybrid fractal-Newton) vs mpmath reference.

Performance: 2.7× faster with 100% perfect accuracy (error < 1e-12)
"""
import sys
import argparse
from time import perf_counter
import numpy as np
from mpmath import mp

sys.path.insert(0, '/home/thorin/working/holographersworkbench')
from workbench.core import GushurstCrystal


def benchmark(start_idx: int = 101, n_zeros: int = 20, dps: int = 50):
    """
    Benchmark zeta zero computation.
    
    Args:
        start_idx: Starting zero index
        n_zeros: Number of zeros to compute
        dps: Decimal precision
    """
    mp.dps = dps
    
    print("="*80)
    print("RIEMANN ZETA ZEROS BENCHMARK")
    print("="*80)
    print(f"Computing zeros {start_idx} to {start_idx + n_zeros - 1}")
    print(f"Precision: {dps} decimal places")
    print()
    
    # Gushurst Crystal (hybrid fractal-Newton)
    print("[1] Gushurst Crystal (hybrid fractal-Newton)...")
    gc = GushurstCrystal(n_zeros=start_idx - 1)
    
    t0 = perf_counter()
    predicted = gc.predict_zeta_zeros(n_zeros)  # Fast mode by default
    t1 = perf_counter()
    gushurst_time = t1 - t0
    
    print(f"    Time: {gushurst_time:.3f}s")
    print()
    
    # mpmath reference
    print("[2] mpmath.zetazero (reference)...")
    t0 = perf_counter()
    reference = []
    for k in range(start_idx, start_idx + n_zeros):
        val = mp.zetazero(k)
        try:
            t_ref = mp.im(val) if mp.im(val) != 0 else mp.mpf(val)
        except:
            t_ref = mp.mpf(val)
        reference.append(float(t_ref))
    t1 = perf_counter()
    mpmath_time = t1 - t0
    
    print(f"    Time: {mpmath_time:.3f}s")
    print()
    
    # Results
    errors = np.abs(np.array(predicted) - np.array(reference))
    perfect = np.sum(errors < 1e-12)
    speedup = mpmath_time / gushurst_time
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Gushurst:  {gushurst_time:.3f}s")
    print(f"mpmath:    {mpmath_time:.3f}s")
    print(f"Speedup:   {speedup:.1f}×")
    print()
    print(f"Accuracy:  {perfect}/{n_zeros} perfect (< 1e-12)")
    print(f"Mean err:  {np.mean(errors):.2e}")
    print(f"Max err:   {np.max(errors):.2e}")
    print()
    
    if perfect == n_zeros and speedup > 1.0:
        print("✅ PASS: 100% perfect accuracy, faster than mpmath")
    elif perfect == n_zeros:
        print("✅ PASS: 100% perfect accuracy")
    else:
        print("⚠️  Some errors detected")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark zeta zero computation')
    parser.add_argument('--start', type=int, default=101, help='Starting zero index')
    parser.add_argument('--count', type=int, default=20, help='Number of zeros')
    parser.add_argument('--dps', type=int, default=50, help='Decimal precision')
    args = parser.parse_args()
    
    benchmark(args.start, args.count, args.dps)
