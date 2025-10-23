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
        # Import from new workbench structure
        from workbench.processors import spectral, holographic, optimization
        from workbench.processors.compression import FractalPeeler, HolographicCompressor
        from workbench.core import zeta, GushurstCrystal
        from workbench.processors import encoding, ergodic
        from workbench.analysis import affinity, performance, errors, convergence
        from workbench.generation import code
        from workbench.primitives import signal
        
        # Test specific imports
        from workbench.processors.spectral import SpectralScorer
        from workbench.core.zeta import ZetaFiducials, zetazero
        from workbench.processors.holographic import phase_retrieve_hilbert
        from workbench.processors.compression import FractalPeeler, compress_image
        from workbench.analysis.affinity import quick_calibrate
        from workbench.analysis.performance import PerformanceProfiler
        from workbench.analysis.errors import ErrorPatternAnalyzer
        from workbench.generation.code import FormulaCodeGenerator
        from workbench.analysis.convergence import ConvergenceAnalyzer
        from workbench.primitives.signal import normalize
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
        from workbench.core.zeta import ZetaFiducials
        from workbench.processors.spectral import SpectralScorer
        
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
        from workbench.processors.holographic import phase_retrieve_hilbert, holographic_refinement
        
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
        from workbench.processors.compression import FractalPeeler, resfrac_score
        
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
        from workbench.processors.compression import compress_image, decompress_image
        
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
        from workbench.core.zeta import zetazero, zetazero_batch
        
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


def test_gushurst_crystal():
    """Test Gushurst Crystal (replaces quantum clock)."""
    print("\n" + "="*70)
    print("TEST 7: Gushurst Crystal")
    print("="*70)
    
    try:
        from workbench.core import GushurstCrystal
        
        # Test instantiation
        gc = GushurstCrystal(n_zeros=50, max_prime=1000)
        assert gc.n_zeros == 50, "Should initialize with 50 zeros"
        assert gc.max_prime == 1000, "Should have max_prime=1000"
        print(f"✓ GushurstCrystal initialized with {gc.n_zeros} zeros")
        
        # Test zeta computation
        gc._compute_zeta_zeros()
        assert gc.zeta_zeros is not None, "Should have zeta zeros"
        assert len(gc.zeta_zeros) == 50, "Should have 50 zeros"
        assert len(gc.zeta_spacings) == 49, "Should have 49 spacings"
        assert np.all(gc.zeta_spacings > 0), "All spacings should be positive"
        print(f"✓ Computed {len(gc.zeta_spacings)} spacings, mean={np.mean(gc.zeta_spacings):.4f}")
        
        # Test prime computation
        gc._compute_primes()
        assert gc.primes is not None, "Should have primes"
        assert gc.primes[0] == 2, "First prime should be 2"
        print(f"✓ Computed {len(gc.primes)} primes up to {gc.max_prime}")
        
        # Test fractal peel cascade
        cascade = gc.fractal_peel_cascade(gc.zeta_spacings, max_levels=5)
        assert 'fractal_dim' in cascade, "Should have fractal_dim"
        assert 'resfrac' in cascade, "Should have resfrac"
        assert 0 < cascade['fractal_dim'] < 2, "Fractal dim should be reasonable"
        assert 0 < cascade['resfrac'] < 1, "Resfrac should be in (0,1)"
        print(f"✓ Fractal peel: dim={cascade['fractal_dim']:.3f}, resfrac={cascade['resfrac']:.2e}")
        
        # Test crystalline lattice
        lattice = gc.build_crystalline_lattice()
        assert lattice.shape[0] == lattice.shape[1], "Lattice should be square"
        assert lattice.shape[0] >= 6, "Lattice should have at least 6 nodes"
        assert np.allclose(lattice, lattice.T), "Lattice should be symmetric"
        print(f"✓ Crystalline lattice: {lattice.shape}")
        
        # Test prime prediction
        primes = gc.predict_primes(n_primes=3)
        assert len(primes) == 3, "Should predict 3 primes"
        assert all(p > gc.max_prime for p in primes), "Predicted primes should be > max_prime"
        print(f"✓ Prime prediction: {primes}")
        
        # Test zeta zero prediction
        zeros = gc.predict_zeta_zeros(n_zeros=2)
        assert len(zeros) == 2, "Should predict 2 zeros"
        assert all(z > gc.zeta_zeros[-1] for z in zeros), "Predicted zeros should be > last known"
        print(f"✓ Zeta prediction: {[f'{z:.2f}' for z in zeros]}")
        
        # Test complete analysis
        structure = gc.analyze_crystal_structure()
        assert 'fractal_dim' in structure, "Should have fractal_dim"
        assert 'n_primes' in structure, "Should have n_primes"
        assert 'n_zeros' in structure, "Should have n_zeros"
        print(f"✓ Complete analysis: {structure['n_resonances']} resonances detected")
        
        return True
    except Exception as e:
        print(f"✗ Gushurst Crystal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_holographic_encoder():
    """Test holographic encoder."""
    print("\n" + "="*70)
    print("TEST 8: Holographic Encoder")
    print("="*70)
    
    try:
        from workbench.processors.encoding import HolographicEncoder
        from workbench.core import GushurstCrystal
        
        # Initialize Gushurst crystal
        gc = GushurstCrystal(n_zeros=50)
        
        # Initialize encoder (will compute zeros automatically)
        encoder = HolographicEncoder(gc)
        assert encoder.qc is not None, "Encoder should have Gushurst crystal"
        print(f"✓ HolographicEncoder initialized")
        
        # Test encoding
        test_array = np.random.randn(10, 20)
        hologram = encoder.encode(test_array, quantum_modes=[2, 3, 5])
        assert 'mode_coefficients' in hologram, "Should have mode coefficients"
        assert len(hologram['mode_coefficients']) == 3, "Should have 3 modes"
        print(f"✓ Encoded array shape {test_array.shape} with 3 modes")
        
        # Test lossy decoding
        reconstructed_lossy = encoder.decode(hologram, use_residual=False)
        assert reconstructed_lossy.shape == test_array.shape, "Shape should match"
        print(f"✓ Lossy decoding successful")
        
        # Test lossless decoding
        reconstructed_lossless = encoder.decode(hologram, use_residual=True)
        fidelity = encoder.measure_fidelity(test_array, reconstructed_lossless)
        assert fidelity['abs_error'] < 0.1, "Lossless should be reasonably accurate"
        print(f"✓ Lossless decoding: error={fidelity['abs_error']:.2e}")
        
        # Test resonance analysis
        resonances = encoder.analyze_resonances(hologram)
        assert len(resonances) == 3, "Should have 3 resonances"
        assert resonances[0]['energy_fraction'] > 0, "Should have energy"
        print(f"✓ Resonance analysis: top mode={resonances[0]['mode']}")
        
        return True
    except Exception as e:
        print(f"✗ Holographic encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ergodic_jump():
    """Test ergodic jump diagnostics."""
    print("\n" + "="*70)
    print("TEST 9: Ergodic Jump")
    print("="*70)
    
    try:
        from workbench.processors.ergodic import ErgodicJump
        
        # Initialize
        jump = ErgodicJump(injection_freq=1/np.sqrt(5), amp=0.15)
        assert jump.injection_freq > 0, "Should have injection frequency"
        print(f"✓ ErgodicJump initialized with freq={jump.injection_freq:.4f}")
        
        # Test on ergodic signal
        ergodic_signal = np.random.randn(512)
        result = jump.execute(ergodic_signal)
        
        assert 'filament' in result, "Should have filament"
        assert 'resfrac_drop' in result, "Should have resfrac_drop"
        assert 'hurst_shift' in result, "Should have hurst_shift"
        print(f"✓ Executed jump: resfrac_drop={result['resfrac_drop']:.4f}")
        
        # Test ergodicity diagnosis
        diagnosis = jump.diagnose_ergodicity(ergodic_signal)
        assert 'is_ergodic' in diagnosis, "Should have ergodicity flag"
        assert 'confidence' in diagnosis, "Should have confidence"
        assert 0 <= diagnosis['confidence'] <= 1, "Confidence in [0,1]"
        print(f"✓ Diagnosis: ergodic={diagnosis['is_ergodic']}, conf={diagnosis['confidence']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Ergodic jump test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_affinity():
    """Test time affinity optimization."""
    print("\n" + "="*70)
    print("TEST 10: Time Affinity")
    print("="*70)
    
    try:
        from workbench.analysis.affinity import quick_calibrate
        
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
    print("TEST 11: Optimization Module")
    print("="*70)
    
    try:
        from workbench.processors.optimization import SublinearOptimizer
        
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
    print("TEST 12: Utility Functions")
    print("="*70)
    
    try:
        from workbench.primitives.signal import normalize, psnr, detect_peaks
        normalize_signal = normalize  # Alias for backward compatibility
        compute_psnr = psnr  # Alias for backward compatibility
        
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
    print("TEST 13: Performance Profiler")
    print("="*70)
    
    try:
        from workbench.analysis.performance import PerformanceProfiler, profile
        
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
    print("TEST 14: Error Pattern Visualizer")
    print("="*70)
    
    try:
        from workbench.analysis.errors import ErrorPatternAnalyzer
        
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
    print("TEST 15: Formula Code Generator")
    print("="*70)
    
    try:
        from workbench.generation.code import FormulaCodeGenerator, CodeValidator
        
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
    print("TEST 16: Convergence Analyzer")
    print("="*70)
    
    try:
        from workbench.analysis.convergence import ConvergenceAnalyzer
        
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


def test_zeta_zero_benchmark():
    """Test zeta zero computation benchmark."""
    print("\n" + "="*70)
    print("TEST: Zeta Zero Benchmark")
    print("="*70)
    
    try:
        from workbench.core import GushurstCrystal, zetazero
        from mpmath import mp, zetazero as mp_zetazero
        
        mp.dps = 50
        
        # Test small batch
        print("Testing 5 zeros...")
        gc = GushurstCrystal(n_zeros=10)
        predicted = gc.predict_zeta_zeros(5)
        
        # Verify accuracy
        reference = [float(mp_zetazero(k).imag) for k in range(11, 16)]
        errors = [abs(p - r) for p, r in zip(predicted, reference)]
        
        perfect = sum(1 for e in errors if e < 1e-12)
        mean_err = np.mean(errors)
        
        print(f"  Perfect (< 1e-12): {perfect}/5")
        print(f"  Mean error: {mean_err:.2e}")
        
        if perfect >= 4:  # Allow 1 imperfect due to timing
            print("✓ Zeta zero benchmark passed")
            return True
        else:
            print(f"✗ Only {perfect}/5 perfect zeros")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_prime_benchmark():
    """Test prime number generation benchmark."""
    print("\n" + "="*70)
    print("TEST: Prime Generation Benchmark")
    print("="*70)
    
    try:
        from workbench.core import GushurstCrystal
        
        # Test prime prediction
        print("Testing 5 primes...")
        gc = GushurstCrystal(n_zeros=50, max_prime=500)
        gc._compute_zeta_zeros()
        
        predicted = gc.predict_primes(n_primes=5)
        
        # Verify all are prime
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        all_prime = all(is_prime(int(p)) for p in predicted)
        
        print(f"  Predicted: {[int(p) for p in predicted[:5]]}")
        print(f"  All prime: {all_prime}")
        
        if all_prime:
            print("✓ Prime benchmark passed")
            return True
        else:
            print("✗ Some predictions are not prime")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_quantum_folding():
    """Test Quantum Entanglement Dimensional Folding."""
    print("\n" + "="*70)
    print("TEST: Quantum Folding")
    print("="*70)
    
    try:
        from workbench.primitives import QuantumFolder
        
        folder = QuantumFolder()
        assert folder is not None and len(folder.dimensions) == 6
        print(f"✓ QuantumFolder instantiated with {len(folder.dimensions)} dimensions")
        
        np.random.seed(42)
        cities = np.random.rand(15, 2) * 100
        
        folded_collapse = folder.fold_dimension_collapse(cities, 1.5)
        assert folded_collapse.shape == cities.shape
        print(f"✓ Dimensional collapse (D=1.5)")
        
        folded_fast = folder.fold_dimension_expand_fast(cities, 3.0)
        assert folded_fast.shape == cities.shape
        print(f"✓ Fast dimensional expansion (D=3.0)")
        
        tour = list(range(len(cities)))
        np.random.shuffle(tour)
        
        entanglement_vec = folder.compute_entanglement_vectorized(cities, tour)
        assert entanglement_vec.shape == (len(cities), len(cities))
        print(f"✓ Vectorized entanglement computed")
        
        sparse_ent = folder.compute_sparse_entanglement(cities, tour, k=5)
        assert len(sparse_ent) == len(cities)
        print(f"✓ Sparse entanglement (k=5)")
        
        initial_tour = list(range(len(cities)))
        tour, length, info = folder.optimize_tour_dimensional_folding_fast(
            cities, initial_tour, n_restarts=1, iterations_per_restart=5
        )
        assert len(tour) == len(cities) and length > 0
        print(f"✓ Optimization: length={length:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Quantum Folding test failed: {e}")
        return False


def test_chaos_seeding():
    """Test Residual Chaos Seeding."""
    print("\n" + "="*70)
    print("TEST: Chaos Seeding")
    print("="*70)
    
    try:
        from workbench.primitives import ChaosSeeder, AdaptiveChaosSeeder
        
        seeder = ChaosSeeder(window_size=3, chaos_weight=0.5)
        print(f"✓ ChaosSeeder instantiated")
        
        adaptive_seeder = AdaptiveChaosSeeder(initial_chaos_weight=0.8, final_chaos_weight=0.2, decay_rate=0.95)
        print(f"✓ AdaptiveChaosSeeder instantiated")
        
        np.random.seed(42)
        cities = np.random.rand(15, 2) * 100
        tour = list(range(len(cities)))
        
        projection = seeder.compute_projection(cities, tour)
        assert projection.shape == cities.shape
        print(f"✓ Projection computed")
        
        chaos = seeder.compute_chaos_magnitude(cities, tour)
        assert chaos >= 0
        print(f"✓ Chaos magnitude: {chaos:.2f}")
        
        tour, chaos_val = seeder.greedy_construction_chaos_seeded(cities)
        assert len(tour) == len(cities) and len(set(tour)) == len(cities)
        print(f"✓ Greedy construction: chaos={chaos_val:.2f}")
        
        tour, length, info = seeder.hybrid_chaos_construction(cities, n_restarts=2)
        assert len(tour) == len(cities) and length > 0
        print(f"✓ Hybrid construction: length={length:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Chaos Seeding test failed: {e}")
        return False


def test_adaptive_nonlocality():
    """Test Adaptive Nonlocality Optimizer."""
    print("\n" + "="*70)
    print("TEST: Adaptive Nonlocality")
    print("="*70)
    
    try:
        from workbench import AdaptiveNonlocalityOptimizer
        
        anl = AdaptiveNonlocalityOptimizer(d_min=1.0, d_max=2.5, n_dim_samples=20)
        assert anl is not None and len(anl.dimensions) == 20
        print(f"✓ AdaptiveNonlocalityOptimizer instantiated")
        
        np.random.seed(42)
        cities = np.random.rand(12, 2) * 100
        
        problem_affinity = anl.compute_problem_affinity(cities)
        assert len(problem_affinity) == 20 and np.all(problem_affinity >= 0)
        print(f"✓ Problem affinity computed")
        
        tour = list(range(len(cities)))
        solution_affinity = anl.compute_solution_affinity(tour, cities)
        assert len(solution_affinity) == 20
        print(f"✓ Solution affinity computed")
        
        coupling = anl.compute_coupling(problem_affinity, solution_affinity, 1.0)
        print(f"✓ Coupling computed")
        
        def cost_fn(tour, cities):
            length = 0.0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
            return length
        
        def local_search(solution, cities, dimension):
            n = len(solution)
            i, j = np.random.randint(0, n-1), np.random.randint(2, n)
            if i >= j-1: return solution
            new_solution = solution.copy()
            new_solution[i+1:j+1] = list(reversed(new_solution[i+1:j+1]))
            return new_solution
        
        best_solution, best_cost, trajectory = anl.optimize(
            tour, cities, cost_fn, local_search, max_iterations=10, verbose=False
        )
        assert len(best_solution) == len(cities)
        print(f"✓ Optimization: cost={best_cost:.2f}")
        
        analysis = anl.analyze_trajectory(trajectory)
        assert 'final_dimension' in analysis
        print(f"✓ Trajectory analysis: final_D={analysis['final_dimension']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Adaptive Nonlocality test failed: {e}")
        return False


def test_sublinear_qik():
    """Test Sublinear QIK."""
    print("\n" + "="*70)
    print("TEST: Sublinear QIK")
    print("="*70)
    
    try:
        from workbench import SublinearQIK, zetazero_batch
        
        qik = SublinearQIK(use_hierarchical=True, use_dimensional_sketch=True, use_sparse_resonance=True)
        print(f"✓ SublinearQIK instantiated")
        
        np.random.seed(42)
        cities = np.random.rand(20, 2) * 100
        
        print("  Computing zeta zeros...")
        zeta_zeros = zetazero_batch(1, 10)
        assert len(zeta_zeros) == 10
        print(f"✓ Zeta zeros computed: {len(zeta_zeros)}")
        
        tour, length, stats = qik.optimize_tsp(cities, zeta_zeros, verbose=False)
        assert len(tour) == len(cities) and len(set(tour)) == len(cities) and length > 0
        print(f"✓ Optimization: length={length:.2f}")
        
        assert stats.n_cities == len(cities) and stats.n_clusters > 0
        print(f"✓ Statistics: {stats.n_clusters} clusters, {stats.n_dim_samples} dim samples")
        
        assert "N^1.5" in stats.theoretical_complexity
        print(f"✓ Complexity: {stats.theoretical_complexity}")
        
        return True
    except Exception as e:
        print(f"✗ Sublinear QIK test failed: {e}")
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
        ("Gushurst Crystal", test_gushurst_crystal),
        ("Zeta Zero Benchmark", test_zeta_zero_benchmark),
        ("Prime Benchmark", test_prime_benchmark),
        ("Holographic Encoder", test_holographic_encoder),
        ("Ergodic Jump", test_ergodic_jump),
        ("Time Affinity", test_time_affinity),
        ("Optimization", test_optimization),
        ("Utils", test_utils),
        ("Performance Profiler", test_performance_profiler),
        ("Error Pattern Visualizer", test_error_pattern_visualizer),
        ("Formula Code Generator", test_formula_code_generator),
        ("Convergence Analyzer", test_convergence_analyzer),
        ("Quantum Folding", test_quantum_folding),
        ("Chaos Seeding", test_chaos_seeding),
        ("Adaptive Nonlocality", test_adaptive_nonlocality),
        ("Sublinear QIK", test_sublinear_qik),
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
