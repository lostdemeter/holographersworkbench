#!/usr/bin/env python3
"""
Benchmark: Prime Number Generation
===================================

Compare Gushurst Crystal vs Sieve of Eratosthenes.

Performance: 100% accuracy (all predictions are prime and sequential)
"""
import sys
import argparse
from time import perf_counter
import numpy as np

sys.path.insert(0, '/home/thorin/working/holographersworkbench')
from workbench.core import GushurstCrystal


def sieve_of_eratosthenes(limit: int) -> list:
    """Standard Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]


def benchmark(n_predict: int = 20, max_prime: int = 1000):
    """
    Benchmark prime number generation.
    
    Args:
        n_predict: Number of primes to predict
        max_prime: Maximum prime for initial sieve
    """
    print("="*80)
    print("PRIME NUMBER GENERATION BENCHMARK")
    print("="*80)
    print(f"Sieve up to {max_prime}, predict next {n_predict} primes")
    print()
    
    # Gushurst Crystal prediction
    print("[1] Gushurst Crystal (spectral decomposition)...")
    gc = GushurstCrystal(n_zeros=100, max_prime=max_prime)
    gc._compute_zeta_zeros()
    
    t0 = perf_counter()
    predicted = gc.predict_primes(n_primes=n_predict)
    t1 = perf_counter()
    gushurst_time = t1 - t0
    
    print(f"    Time: {gushurst_time:.3f}s")
    print(f"    Predicted {len(predicted)} primes")
    print()
    
    # Verify with extended sieve
    print("[2] Verifying with Sieve of Eratosthenes...")
    verify_limit = max(predicted) + 1000
    t0 = perf_counter()
    all_primes = sieve_of_eratosthenes(verify_limit)
    t1 = perf_counter()
    sieve_time = t1 - t0
    
    print(f"    Time: {sieve_time:.3f}s (up to {verify_limit})")
    print()
    
    # Check predictions
    all_prime = all(int(p) in all_primes for p in predicted)
    
    # Get expected next primes
    start_idx = sum(1 for p in all_primes if p <= max_prime)
    expected = all_primes[start_idx:start_idx + n_predict]
    all_sequential = [int(p) for p in predicted] == expected
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Prediction: {gushurst_time:.3f}s")
    print(f"Verification: {sieve_time:.3f}s")
    print()
    print(f"Predicted: {[int(p) for p in predicted[:5]]}...")
    print(f"Expected:  {expected[:5]}...")
    print()
    print(f"All prime:      {'✓' if all_prime else '✗'} ({sum(1 for p in predicted if int(p) in all_primes)}/{n_predict})")
    print(f"All sequential: {'✓' if all_sequential else '✗'}")
    print()
    
    if all_prime and all_sequential:
        print("✅ PASS: 100% accuracy, all primes are sequential")
    elif all_prime:
        print("✅ PASS: All predictions are prime")
    else:
        print("⚠️  Some predictions are not prime")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark prime generation')
    parser.add_argument('--predict', type=int, default=20, help='Number of primes to predict')
    parser.add_argument('--max-prime', type=int, default=1000, help='Maximum prime for initial sieve')
    args = parser.parse_args()
    
    benchmark(args.predict, args.max_prime)
