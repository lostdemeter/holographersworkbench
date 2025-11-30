#!/usr/bin/env python3
"""
Workbench CLI - Command-line interface for benchmarking and utilities.

Usage:
    python -m workbench bench tsp --n=1000 --pyramid --jax
    python -m workbench bench zeta --start=1 --end=100
    python -m workbench bench tsplib
"""

import argparse
import sys
import time
import numpy as np


def bench_tsp(args):
    """Benchmark TSP solver on random instances."""
    from workbench.processors.sublinear_clock_v2 import (
        SublinearClockOptimizerV2,
        solve_tsp_clock_v2,
        JAX_AVAILABLE
    )
    
    print("=" * 60)
    print("TSP Benchmark: Clock-Resonant Optimizer v2")
    print("=" * 60)
    print(f"JAX acceleration: {'ENABLED' if JAX_AVAILABLE else 'DISABLED'}")
    print(f"Pyramid phases: {'ENABLED' if args.pyramid else 'DISABLED'}")
    print()
    
    # BHH expected for gap calculation
    def bhh_expected(n):
        return 0.7124 * np.sqrt(n * 1e6)
    
    for n in args.sizes:
        print(f"N = {n} cities")
        print("-" * 40)
        
        results = []
        for trial in range(args.trials):
            np.random.seed(args.seed + trial * 100)
            cities = np.random.rand(n, 2) * 1000
            
            optimizer = SublinearClockOptimizerV2(
                use_pyramid=args.pyramid,
                use_gradient_flow=True
            )
            
            t0 = time.time()
            tour, length, stats = optimizer.optimize_tsp(cities)
            elapsed = time.time() - t0
            
            gap = 100 * (length - bhh_expected(n)) / bhh_expected(n)
            results.append({
                'length': length,
                'gap': gap,
                'time': elapsed,
                'resonance': stats.resonance_strength
            })
            
            print(f"  Trial {trial+1}: length={length:.1f}, gap={gap:.1f}%, time={elapsed:.2f}s")
        
        avg_gap = np.mean([r['gap'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        print(f"  Average: gap={avg_gap:.2f}%, time={avg_time:.2f}s")
        print()


def bench_tsplib(args):
    """Benchmark on TSPLIB instances."""
    import subprocess
    import os
    
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'practical_applications',
        'showcases',
        'clock_resonant_tsp',
        'benchmark_tsplib.py'
    )
    
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"Error: {script_path} not found")
        sys.exit(1)


def bench_zeta(args):
    """Benchmark zeta zero computation."""
    from workbench.core import zetazero_batch
    
    print("=" * 60)
    print("Zeta Zero Benchmark: Hybrid Fractal-Newton")
    print("=" * 60)
    print(f"Range: zeros {args.start} to {args.end}")
    print()
    
    t0 = time.time()
    zeros = zetazero_batch(args.start, args.end)
    elapsed = time.time() - t0
    
    n_zeros = args.end - args.start + 1
    print(f"Computed {n_zeros} zeros in {elapsed:.3f}s")
    print(f"Rate: {n_zeros / elapsed:.1f} zeros/sec")
    print()
    print("First 5 zeros:")
    zero_items = list(zeros.items())[:5]
    for n, z in zero_items:
        print(f"  #{n}: {z:.10f}")


def main():
    parser = argparse.ArgumentParser(
        description="Holographer's Workbench CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m workbench bench tsp --n 100 200 500 1000
  python -m workbench bench tsp --n 1000 --pyramid --trials 5
  python -m workbench bench tsplib
  python -m workbench bench zeta --start 1 --end 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Benchmark subcommand
    bench_parser = subparsers.add_parser('bench', help='Run benchmarks')
    bench_subparsers = bench_parser.add_subparsers(dest='bench_type', help='Benchmark type')
    
    # TSP benchmark
    tsp_parser = bench_subparsers.add_parser('tsp', help='TSP benchmark on random instances')
    tsp_parser.add_argument('--n', '--sizes', dest='sizes', type=int, nargs='+',
                           default=[100, 200, 500, 1000],
                           help='Instance sizes to benchmark')
    tsp_parser.add_argument('--trials', type=int, default=3,
                           help='Number of trials per size')
    tsp_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    tsp_parser.add_argument('--pyramid', action='store_true',
                           help='Enable multi-scale pyramid phases')
    tsp_parser.add_argument('--jax', action='store_true',
                           help='Enable JAX acceleration (auto-detected)')
    
    # TSPLIB benchmark
    tsplib_parser = bench_subparsers.add_parser('tsplib', help='TSPLIB benchmark')
    
    # Zeta benchmark
    zeta_parser = bench_subparsers.add_parser('zeta', help='Zeta zero benchmark')
    zeta_parser.add_argument('--start', type=int, default=1,
                            help='First zero index')
    zeta_parser.add_argument('--end', type=int, default=100,
                            help='Last zero index')
    
    args = parser.parse_args()
    
    if args.command == 'bench':
        if args.bench_type == 'tsp':
            bench_tsp(args)
        elif args.bench_type == 'tsplib':
            bench_tsplib(args)
        elif args.bench_type == 'zeta':
            bench_zeta(args)
        else:
            bench_parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
