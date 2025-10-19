#!/usr/bin/env python3
"""
Comprehensive test suite for Holographer's Workbench
=====================================================

Tests all major modules and functionality.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path (workbench root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        # Import from local modules
        import spectral
        import holographic
        import optimization
        import fractal_peeling
        import holographic_compression
        import fast_zetas
        import time_affinity
        import performance_profiler
        import error_pattern_visualizer
        import formula_code_generator
        import convergence_analyzer
        import utils
        
        # Test specific imports
        from spectral import (
            SpectralScorer, ZetaFiducials
        )
        from holographic import phase_retrieve_hilbert
        from fractal_peeling import FractalPeeler
        from holographic_compression import compress_image
        from fast_zetas import zetazero
        from time_affinity import quick_calibrate
        from performance_profiler import PerformanceProfiler
        from error_pattern_visualizer import ErrorPatternAnalyzer
        from formula_code_generator import FormulaCodeGenerator
        from convergence_analyzer import ConvergenceAnalyzer
        from utils import normalize_signal
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_spectral():
    """Test spectral module."""
    print("\n" + "="*70)
    print("TEST 2: Spectral Module")
    print("="*70)
    
    try:
        from spectral import ZetaFiducials, SpectralScorer
        
        # Test zeta fiducials
        zeros = ZetaFiducials.get_standard(10)
        assert len(zeros) == 10, "Should get 10 zeros"
        assert zeros[0] > 14.0 and zeros[0] < 15.0, "First zero should be ~14.13"
        print(f"✓ ZetaFiducials: {len(zeros)} zeros loaded")
        
        # Test spectral scorer
        scorer = SpectralScorer(frequencies=zeros, damping=0.05)
        candidates = np.arange(100, 200)
        scores = scorer.compute_scores(candidates, shift=0.05, mode='real')
        assert len(scores) == len(candidates), "Scores length mismatch"
        print(f"✓ SpectralScorer: {len(scores)} candidates scored")
        
        return True
    except Exception as e:
        print(f"✗ Spectral test failed: {e}")
        return False


def test_holographic():
    """Test holographic module."""
    print("\n" + "="*70)
    print("TEST 3: Holographic Module")
    print("="*70)
    
    try:
        from holographic import phase_retrieve_hilbert, holographic_refinement
        
        # Test phase retrieval
        signal = np.sin(np.linspace(0, 10*np.pi, 500))
        envelope, phase_var = phase_retrieve_hilbert(signal)
        assert len(envelope) == len(signal), "Envelope length mismatch"
        assert phase_var >= 0, "Phase variance should be non-negative"
        print(f"✓ Phase retrieval: PV={phase_var:.4f}")
        
        # Test holographic refinement
        reference = np.ones_like(signal)
        refined = holographic_refinement(signal, reference, method='hilbert')
        assert len(refined) == len(signal), "Refined length mismatch"
        print(f"✓ Holographic refinement: {len(refined)} samples")
        
        return True
    except Exception as e:
        print(f"✗ Holographic test failed: {e}")
        return False


def test_fractal_peeling():
    """Test fractal peeling module."""
    print("\n" + "="*70)
    print("TEST 4: Fractal Peeling")
    print("="*70)
    
    try:
        from fractal_peeling import FractalPeeler, resfrac_score
        
        # Test resfrac score
        signal = np.sin(np.linspace(0, 4*np.pi, 200))
        rho = resfrac_score(signal, order=3)
        assert 0 <= rho <= 1, "Resfrac should be in [0,1]"
        print(f"✓ Resfrac score: ρ={rho:.4f}")
        
        # Test compression/decompression
        peeler = FractalPeeler(order=3, max_depth=5)
        tree = peeler.compress(signal)
        reconstructed = peeler.decompress(tree)
        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, f"Reconstruction error too large: {error}"
        print(f"✓ Lossless compression: error={error:.2e}")
        
        return True
    except Exception as e:
        print(f"✗ Fractal peeling test failed: {e}")
        return False


def test_holographic_compression():
    """Test holographic image compression."""
    print("\n" + "="*70)
    print("TEST 5: Holographic Compression")
    print("="*70)
    
    try:
        from holographic_compression import compress_image, decompress_image
        
        # Create test image
        size = 64
        y, x = np.ogrid[:size, :size]
        cy, cx = size // 2, size // 2
        theta = np.arctan2(y - cy, x - cx)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        image = (128 + 127 * np.sin(15 * theta) * np.exp(-r / 20)).astype(np.uint8)
        
        # Compress and decompress
        compressed, stats = compress_image(image, harmonic_order=15)
        reconstructed = decompress_image(compressed)
        
        # Verify lossless
        is_lossless = np.array_equal(image, reconstructed)
        assert is_lossless, "Compression should be lossless"
        print(f"✓ Lossless: {is_lossless}, Ratio: {stats.compression_ratio:.2f}x")
        
        return True
    except Exception as e:
        print(f"✗ Holographic compression test failed: {e}")
        return False


def test_fast_zetas():
    """Test fast zeta zeros."""
    print("\n" + "="*70)
    print("TEST 6: Fast Zetas")
    print("="*70)
    
    try:
        from fast_zetas import zetazero, zetazero_batch
        
        # Test single zero
        z = zetazero(10)
        assert 49 < float(z) < 50, "10th zero should be ~49.77"
        print(f"✓ zetazero(10) = {float(z):.6f}")
        
        # Test batch
        zeros = zetazero_batch(1, 20, parallel=False)
        assert len(zeros) == 20, "Should get 20 zeros"
        print(f"✓ Batch: {len(zeros)} zeros computed")
        
        return True
    except Exception as e:
        print(f"✗ Fast zetas test failed: {e}")
        return False


def test_time_affinity():
    """Test time affinity optimization."""
    print("\n" + "="*70)
    print("TEST 7: Time Affinity")
    print("="*70)
    
    try:
        from time_affinity import quick_calibrate
        
        # Simple test algorithm
        def test_algo(x, y):
            penalty = (x - 0.5)**2 + (y - 0.5)**2
            for i in range(int(10 + penalty * 100)):
                _ = np.sum(np.random.rand(10))
            return x + y
        
        # Quick calibration
        result = quick_calibrate(
            test_algo,
            target_time=0.002,
            param_bounds={'x': (0.0, 1.0), 'y': (0.0, 1.0)},
            method='gradient',
            max_iterations=10,
            verbose=False
        )
        
        assert result.best_params is not None, "Should find parameters"
        print(f"✓ Calibration: params={result.best_params}, error={result.time_error:.6f}s")
        
        return True
    except Exception as e:
        print(f"✗ Time affinity test failed: {e}")
        return False


def test_optimization():
    """Test optimization module."""
    print("\n" + "="*70)
    print("TEST 8: Optimization Module")
    print("="*70)
    
    try:
        from optimization import SublinearOptimizer
        
        # Test sublinear optimizer
        candidates = np.arange(1000)
        def score_fn(c): return np.sin(c * 0.01)
        
        optimizer = SublinearOptimizer(use_holographic=False)
        top_k, stats = optimizer.optimize(candidates, score_fn, top_k=50)
        
        assert len(top_k) == 50, "Should return 50 candidates"
        print(f"✓ Sublinear: {stats.n_original}→{stats.n_final}, {stats.complexity_estimate}")
        
        return True
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\n" + "="*70)
    print("TEST 9: Utility Functions")
    print("="*70)
    
    try:
        from utils import normalize_signal, compute_psnr, detect_peaks
        
        # Test normalize
        signal = np.random.randn(100)
        normalized = normalize_signal(signal, method='minmax')
        assert np.min(normalized) >= 0 and np.max(normalized) <= 1, "Should be in [0,1]"
        print(f"✓ Normalize: range=[{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        
        # Test PSNR
        original = np.random.rand(100)
        reconstructed = original + 0.01 * np.random.randn(100)
        psnr = compute_psnr(original, reconstructed)
        assert psnr > 0, "PSNR should be positive"
        print(f"✓ PSNR: {psnr:.2f} dB")
        
        # Test peak detection
        signal = np.sin(np.linspace(0, 10*np.pi, 500))
        peaks = detect_peaks(signal, threshold=0.5)
        assert len(peaks) > 0, "Should find peaks"
        print(f"✓ Peak detection: {len(peaks)} peaks found")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def test_performance_profiler():
    """Test performance profiler."""
    print("\n" + "="*70)
    print("TEST 10: Performance Profiler")
    print("="*70)
    
    try:
        from performance_profiler import PerformanceProfiler, profile
        
        # Test basic profiling
        profiler = PerformanceProfiler(track_memory=False)
        
        def test_func(n):
            return sum(range(n))
        
        result, profile_result = profiler.profile_function(test_func, 1000)
        assert result == sum(range(1000)), "Function result incorrect"
        assert profile_result.execution_time > 0, "Should measure time"
        print(f"✓ Function profiling: {profile_result.execution_time*1000:.3f}ms")
        
        # Test decorator (returns just the result)
        @profile()
        def decorated_func(x):
            return x ** 2
        
        result = decorated_func(5)
        assert result == 25, f"Decorated function result incorrect: got {result}"
        print(f"✓ Decorator profiling works")
        
        return True
    except Exception as e:
        print(f"✗ Performance profiler test failed: {e}")
        return False


def test_error_pattern_visualizer():
    """Test error pattern visualizer."""
    print("\n" + "="*70)
    print("TEST 11: Error Pattern Visualizer")
    print("="*70)
    
    try:
        from error_pattern_visualizer import ErrorPatternAnalyzer
        
        # Create synthetic error with known pattern
        x = np.linspace(0, 10, 100)
        actual = np.sin(x) + 0.1 * x
        predicted = np.sin(x)
        
        analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Test")
        report = analyzer.analyze_all()
        
        assert report.polynomial_pattern is not None, "Should detect polynomial"
        assert len(report.suggestions) > 0, "Should have suggestions"
        print(f"✓ Pattern detection: {len(report.suggestions)} corrections found")
        
        # Test correction application
        if report.suggestions:
            corrected = analyzer.apply_correction(report.suggestions[0])
            assert corrected.rmse < analyzer.rmse, "Should improve RMSE"
            print(f"✓ Correction: RMSE {analyzer.rmse:.6f} → {corrected.rmse:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Error pattern visualizer test failed: {e}")
        return False


def test_formula_code_generator():
    """Test formula code generator."""
    print("\n" + "="*70)
    print("TEST 12: Formula Code Generator")
    print("="*70)
    
    try:
        from formula_code_generator import FormulaCodeGenerator, CodeValidator
        
        # Test code generation
        generator = FormulaCodeGenerator(
            base_formula="x**2",
            name="test_formula"
        )
        generator.add_correction("correction = 0.1 * x")
        
        code = generator.generate_function()
        assert "def test_formula" in code, "Should contain function definition"
        assert "correction" in code, "Should contain correction"
        print(f"✓ Code generation: {len(code)} characters")
        
        # Test validation
        validation = generator.validate_code(code)
        assert validation.is_valid, "Generated code should be valid"
        print(f"✓ Validation: {len(validation.errors)} errors, {len(validation.warnings)} warnings")
        
        return True
    except Exception as e:
        print(f"✗ Formula code generator test failed: {e}")
        return False


def test_convergence_analyzer():
    """Test convergence analyzer."""
    print("\n" + "="*70)
    print("TEST 13: Convergence Analyzer")
    print("="*70)
    
    try:
        from convergence_analyzer import ConvergenceAnalyzer
        
        # Test with exponential convergence
        history = [1.0, 0.5, 0.25, 0.125, 0.0625]
        analyzer = ConvergenceAnalyzer(history, "Error")
        report = analyzer.analyze()
        
        assert report.convergence_rate.model_type == "exponential", "Should detect exponential"
        assert report.convergence_rate.convergence_speed == "fast", "Should be fast"
        print(f"✓ Convergence detection: {report.convergence_rate.model_type}")
        
        # Test prediction
        future_iters, future_metrics = analyzer.predict_future_improvements(5)
        assert len(future_iters) == 5, "Should predict 5 iterations"
        assert len(future_metrics) == 5, "Should predict 5 metrics"
        print(f"✓ Prediction: next value ≈ {future_metrics[0]:.6f}")
        
        # Test stopping recommendation
        assert report.stopping_recommendation is not None, "Should have recommendation"
        print(f"✓ Recommendation: should_stop={report.stopping_recommendation.should_stop}")
        
        return True
    except Exception as e:
        print(f"✗ Convergence analyzer test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("HOLOGRAPHER'S WORKBENCH - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Spectral", test_spectral),
        ("Holographic", test_holographic),
        ("Fractal Peeling", test_fractal_peeling),
        ("Holographic Compression", test_holographic_compression),
        ("Fast Zetas", test_fast_zetas),
        ("Time Affinity", test_time_affinity),
        ("Optimization", test_optimization),
        ("Utils", test_utils),
        ("Performance Profiler", test_performance_profiler),
        ("Error Pattern Visualizer", test_error_pattern_visualizer),
        ("Formula Code Generator", test_formula_code_generator),
        ("Convergence Analyzer", test_convergence_analyzer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
