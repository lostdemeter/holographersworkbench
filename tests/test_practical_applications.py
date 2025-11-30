#!/usr/bin/env python3
"""
Test Practical Applications and Showcases
==========================================

Tests the showcase benchmarks and practical application scripts.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_script(script_path: Path, timeout: int = 60) -> tuple:
    """Run a Python script and return (success, output/error)."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent.parent.parent),  # repo root
            capture_output=True,
            timeout=timeout,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout[:500] if result.stdout else "OK"
        else:
            return False, result.stderr[:500] if result.stderr else result.stdout[:500]
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def test_showcases():
    """Test all showcase benchmarks."""
    print("\n" + "="*70)
    print("TESTING SHOWCASES")
    print("="*70)
    
    showcases_dir = Path(__file__).parent.parent / "practical_applications" / "showcases"
    
    tests = [
        # Clock-Resonant TSP showcases
        ("TSP: benchmark_v2", showcases_dir / "clock_resonant_tsp" / "benchmark_v2.py"),
        ("TSP: benchmark_tsplib", showcases_dir / "clock_resonant_tsp" / "benchmark_tsplib.py"),
        ("TSP: benchmark_optimizer", showcases_dir / "clock_resonant_tsp" / "benchmark_optimizer.py"),
        # Gushurst Crystal showcases
        ("Crystal: benchmark_zeta", showcases_dir / "gushurst_crystal" / "benchmark_zeta_zeros.py"),
        ("Crystal: benchmark_primes", showcases_dir / "gushurst_crystal" / "benchmark_primes.py"),
        # Dimensional Downcasting (may fail if DD not available)
        ("DD: benchmark_integration", showcases_dir / "dimensional_downcasting" / "benchmark_dd_integration.py"),
        # Clock Compiler
        ("Compiler: demo", showcases_dir / "clock_compiler" / "demo_clock_compiler.py"),
    ]
    
    results = []
    for name, script in tests:
        print(f"\n{name}...", end=" ")
        if not script.exists():
            print("✗ NOT FOUND")
            results.append((name, False, "Script not found"))
            continue
            
        success, output = run_script(script, timeout=90)
        if success:
            print("✓ PASS")
            results.append((name, True, None))
        else:
            print("✗ FAIL")
            results.append((name, False, output))
    
    return results


def test_workbench_cli():
    """Test workbench CLI commands."""
    print("\n" + "="*70)
    print("TESTING WORKBENCH CLI")
    print("="*70)
    
    results = []
    
    # Test bench tsp
    print("\nCLI: bench tsp...", end=" ")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "workbench", "bench", "tsp", "--n", "20"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            print("✓ PASS")
            results.append(("bench tsp", True, None))
        else:
            print("✗ FAIL")
            results.append(("bench tsp", False, result.stderr[:300]))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("bench tsp", False, str(e)))
    
    # Test bench zeta
    print("CLI: bench zeta...", end=" ")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "workbench", "bench", "zeta", "--start", "1", "--end", "10"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            print("✓ PASS")
            results.append(("bench zeta", True, None))
        else:
            print("✗ FAIL")
            results.append(("bench zeta", False, result.stderr[:300]))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("bench zeta", False, str(e)))
    
    return results


def test_core_imports():
    """Test that core workbench imports work."""
    print("\n" + "="*70)
    print("TESTING CORE IMPORTS")
    print("="*70)
    
    results = []
    
    imports = [
        ("workbench.core", "from workbench.core import zetazero_batch, GushurstCrystal"),
        ("workbench.processors", "from workbench.processors import solve_tsp_clock_v2, SpectralScorer"),
        ("workbench.primitives", "from workbench.primitives import QuantumFolder, ChaosSeeder"),
        ("workbench.analysis", "from workbench.analysis import PerformanceProfiler, ErrorPatternAnalyzer"),
    ]
    
    for name, import_stmt in imports:
        print(f"\n{name}...", end=" ")
        try:
            exec(import_stmt)
            print("✓ PASS")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ FAIL: {e}")
            results.append((name, False, str(e)))
    
    return results


def test_clock_optimizers():
    """Test clock-resonant optimizers."""
    print("\n" + "="*70)
    print("TESTING CLOCK OPTIMIZERS")
    print("="*70)
    
    import numpy as np
    results = []
    
    # Test SublinearClockOptimizer (v1)
    print("\nSublinearClockOptimizer...", end=" ")
    try:
        from workbench.processors.sublinear_clock import SublinearClockOptimizer
        optimizer = SublinearClockOptimizer()
        cities = np.random.rand(15, 2) * 100
        tour, length, stats = optimizer.optimize_tsp(cities)
        assert len(tour) == len(cities)
        assert length > 0
        print("✓ PASS")
        results.append(("SublinearClockOptimizer", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("SublinearClockOptimizer", False, str(e)))
    
    # Test SublinearClockOptimizerV2
    print("SublinearClockOptimizerV2...", end=" ")
    try:
        from workbench.processors.sublinear_clock_v2 import SublinearClockOptimizerV2, solve_tsp_clock_v2
        cities = np.random.rand(15, 2) * 100
        tour, length, stats = solve_tsp_clock_v2(cities)
        assert len(tour) == len(cities)
        assert length > 0
        assert hasattr(stats, 'resonance_strength')
        print("✓ PASS")
        results.append(("SublinearClockOptimizerV2", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("SublinearClockOptimizerV2", False, str(e)))
    
    # Test ClockResonanceCompiler
    print("ClockResonanceCompiler...", end=" ")
    try:
        from workbench.core.clock_compiler import ClockResonanceCompiler
        compiler = ClockResonanceCompiler(verbose=False)
        from workbench.processors.spectral import SpectralScorer
        analysis = compiler.analyze(SpectralScorer)
        assert hasattr(analysis, 'estimated_difficulty')
        assert hasattr(analysis, 'processor_name')
        print("✓ PASS")
        results.append(("ClockResonanceCompiler", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("ClockResonanceCompiler", False, str(e)))
    
    # Test LazyClockOracle (from sublinear_clock_v2)
    print("LazyClockOracle...", end=" ")
    try:
        from workbench.processors.sublinear_clock_v2 import LazyClockOracle
        oracle = LazyClockOracle(max_n=100, use_12d=True)
        phase = oracle.get_fractional_phase(50, 'golden')
        assert 0 <= phase <= 1
        print("✓ PASS")
        results.append(("LazyClockOracle", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("LazyClockOracle", False, str(e)))
    
    return results


def test_all_processors():
    """Test all workbench processors."""
    print("\n" + "="*70)
    print("TESTING ALL PROCESSORS")
    print("="*70)
    
    import numpy as np
    results = []
    
    processors = [
        ("SpectralScorer", "from workbench.processors.spectral import SpectralScorer; s = SpectralScorer()"),
        ("HolographicEncoder", "from workbench.processors.encoding import HolographicEncoder; from workbench.core import GushurstCrystal; gc = GushurstCrystal(n_zeros=10, max_prime=50); e = HolographicEncoder(gc)"),
        ("FractalPeeler", "from workbench.processors.compression import FractalPeeler; p = FractalPeeler()"),
        ("HolographicCompressor", "from workbench.processors.compression import HolographicCompressor; c = HolographicCompressor()"),
        ("ErgodicJump", "from workbench.processors.ergodic import ErgodicJump; j = ErgodicJump()"),
        ("SublinearOptimizer", "from workbench.processors.optimization import SublinearOptimizer; o = SublinearOptimizer()"),
        ("AdaptiveNonlocalityOptimizer", "from workbench.processors.adaptive_nonlocality import AdaptiveNonlocalityOptimizer; a = AdaptiveNonlocalityOptimizer()"),
        ("SublinearQIK", "from workbench.processors.sublinear_qik import SublinearQIK; q = SublinearQIK()"),
        ("HolographicDepthExtractor", "from workbench.processors.holographic_depth import HolographicDepthExtractor; h = HolographicDepthExtractor()"),
        ("QuantumAutoencoder", "from workbench.processors.quantum_autoencoder import QuantumAutoencoder; q = QuantumAutoencoder()"),
        ("AdditiveErrorStereo", "from workbench.processors.additive_error_stereo import AdditiveErrorStereo; s = AdditiveErrorStereo()"),
    ]
    
    for name, code in processors:
        print(f"\n{name}...", end=" ")
        try:
            exec(code)
            print("✓ PASS")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ FAIL: {e}")
            results.append((name, False, str(e)))
    
    return results


def main():
    """Run all tests."""
    print("="*70)
    print("PRACTICAL APPLICATIONS TEST SUITE")
    print("="*70)
    
    all_results = []
    
    # Core imports
    all_results.extend(test_core_imports())
    
    # Clock optimizers (sublinear_clock, sublinear_clock_v2, clock_compiler)
    all_results.extend(test_clock_optimizers())
    
    # All processors
    all_results.extend(test_all_processors())
    
    # CLI tests
    all_results.extend(test_workbench_cli())
    
    # Showcase tests
    all_results.extend(test_showcases())
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in all_results if success)
    total = len(all_results)
    
    for name, success, error in all_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")
        if error and not success:
            print(f"         Error: {error[:100]}...")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
