"""
Unit Tests for Dimensional Downcasting Solver
=============================================

Tests the core functionality of the solver and predictors.
"""

import pytest
import numpy as np
from mpmath import mp, zetazero
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import DimensionalDowncaster
from src.predictors import RamanujanPredictor, GeometricPredictor, gue_spacing

mp.dps = 50


class TestRamanujanPredictor:
    """Tests for the Ramanujan predictor."""
    
    def setup_method(self):
        self.predictor = RamanujanPredictor()
    
    def test_first_zero(self):
        """Test prediction for the first zero."""
        t_pred = self.predictor.predict(1)
        t_true = float(zetazero(1).imag)
        assert abs(t_pred - t_true) < 1.0, "First zero prediction should be within 1.0"
    
    def test_accuracy_bound(self):
        """Test that predictions are within the quantum barrier."""
        errors = []
        for n in [10, 50, 100]:
            t_pred = self.predictor.predict(n)
            t_true = float(zetazero(n).imag)
            errors.append(abs(t_pred - t_true))
        
        rms = np.sqrt(np.mean(np.array(errors)**2))
        assert rms < 1.0, f"RMS error {rms} should be < 1.0"
    
    def test_monotonic(self):
        """Test that predictions are monotonically increasing."""
        predictions = [self.predictor.predict(n) for n in range(1, 20)]
        for i in range(len(predictions) - 1):
            assert predictions[i] < predictions[i+1], "Predictions should be monotonic"


class TestGeometricPredictor:
    """Tests for the Geometric predictor."""
    
    def setup_method(self):
        self.predictor = GeometricPredictor()
    
    def test_first_zero(self):
        """Test prediction for the first zero."""
        t_pred = self.predictor.predict(1)
        t_true = float(zetazero(1).imag)
        assert abs(t_pred - t_true) < 1.0, "First zero prediction should be within 1.0"
    
    def test_light_cone_transition(self):
        """Test behavior around the light cone at n=80."""
        # Predictions should be smooth across the transition
        predictions = [self.predictor.predict(n) for n in range(75, 86)]
        diffs = np.diff(predictions)
        
        # All differences should be positive and similar magnitude
        assert all(d > 0 for d in diffs), "Predictions should be monotonic"
        assert np.std(diffs) / np.mean(diffs) < 0.5, "Transition should be smooth"


class TestDimensionalDowncaster:
    """Tests for the main solver."""
    
    def setup_method(self):
        self.solver = DimensionalDowncaster()
    
    def test_first_zero(self):
        """Test solving for the first zero."""
        t_solved = self.solver.solve(1)
        t_true = float(zetazero(1).imag)
        assert abs(t_solved - t_true) < 1e-10, f"Error {abs(t_solved - t_true)} should be < 1e-10"
    
    def test_machine_precision(self):
        """Test that solver achieves machine precision."""
        for n in [10, 50, 100]:
            t_solved = self.solver.solve(n)
            t_true = float(zetazero(n).imag)
            error = abs(t_solved - t_true)
            assert error < 1e-10, f"n={n}: Error {error} should be < 1e-10"
    
    def test_large_zeros(self):
        """Test solving for large zero indices."""
        for n in [500, 1000]:
            t_solved = self.solver.solve(n)
            t_true = float(zetazero(n).imag)
            error = abs(t_solved - t_true)
            assert error < 1e-10, f"n={n}: Error {error} should be < 1e-10"
    
    def test_n_smooth_offset(self):
        """Test the key insight: N_smooth(t_n) ≈ n - 0.5."""
        for n in [10, 50, 100]:
            t_n = float(zetazero(n).imag)
            N_s = self.solver._N_smooth(t_n)
            offset = abs(N_s - (n - 0.5))
            assert offset < 0.5, f"n={n}: N_smooth offset {offset} should be < 0.5"
    
    def test_solve_range(self):
        """Test solving a range of zeros."""
        zeros = self.solver.solve_range(1, 5)
        assert len(zeros) == 5, "Should return 5 zeros"
        
        for i, t in enumerate(zeros, 1):
            t_true = float(zetazero(i).imag)
            assert abs(t - t_true) < 1e-10, f"Zero {i} should be accurate"
    
    def test_verify(self):
        """Test the verify method."""
        result = self.solver.verify(100)
        
        assert 'n' in result
        assert 't_solved' in result
        assert 't_true' in result
        assert 'error' in result
        assert 'Z_at_t' in result
        
        assert result['error'] < 1e-10
        assert result['Z_at_t'] < 1e-10
    
    def test_stats_tracking(self):
        """Test that statistics are tracked."""
        self.solver.stats = {'Z_evals': 0, 'zeros_solved': 0}
        self.solver.solve(100)
        
        assert self.solver.stats['Z_evals'] > 0, "Should track Z evaluations"
        assert self.solver.stats['zeros_solved'] == 1, "Should track zeros solved"


class TestGUESpacing:
    """Tests for the GUE spacing function."""
    
    def test_positive(self):
        """Test that spacing is always positive."""
        for t in [10, 100, 1000]:
            assert gue_spacing(t) > 0, "Spacing should be positive"
    
    def test_increasing(self):
        """Test that spacing increases with t."""
        spacings = [gue_spacing(t) for t in [10, 100, 1000]]
        for i in range(len(spacings) - 1):
            assert spacings[i] < spacings[i+1], "Spacing should increase with t"
    
    def test_approximate_formula(self):
        """Test that spacing matches log(t)/(2π)."""
        for t in [100, 1000]:
            expected = np.log(t) / (2 * np.pi)
            actual = gue_spacing(t)
            # Should be close (within 10%)
            assert abs(actual - expected) / expected < 0.1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        self.solver = DimensionalDowncaster()
    
    def test_small_n(self):
        """Test with small n values."""
        for n in [1, 2, 3]:
            t_solved = self.solver.solve(n)
            t_true = float(zetazero(n).imag)
            assert abs(t_solved - t_true) < 1e-10
    
    def test_predictor_swap(self):
        """Test swapping predictors."""
        geo = GeometricPredictor()
        solver = DimensionalDowncaster(predictor=geo)
        
        t_solved = solver.solve(100)
        t_true = float(zetazero(100).imag)
        assert abs(t_solved - t_true) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
