"""
Accuracy Benchmarks for Dimensional Downcasting
================================================

Comprehensive accuracy tests across different zero ranges.
"""

import pytest
import numpy as np
from mpmath import mp, zetazero
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import DimensionalDowncaster
from src.predictors import RamanujanPredictor, GeometricPredictor

mp.dps = 50


class TestAccuracyBenchmarks:
    """Comprehensive accuracy benchmarks."""
    
    def setup_method(self):
        self.solver = DimensionalDowncaster()
        self.ramanujan = RamanujanPredictor()
        self.geometric = GeometricPredictor()
    
    def test_first_100_zeros(self):
        """Test accuracy on first 100 zeros."""
        errors = []
        
        for n in range(1, 101):
            t_solved = self.solver.solve(n)
            t_true = float(zetazero(n).imag)
            errors.append(abs(t_solved - t_true))
        
        max_error = max(errors)
        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        
        assert max_error < 1e-10, f"Max error {max_error} should be < 1e-10"
        assert rms_error < 1e-10, f"RMS error {rms_error} should be < 1e-10"
    
    def test_improvement_over_predictors(self):
        """Test that dimensional downcasting improves over predictors."""
        test_zeros = [10, 50, 100, 200]
        
        for n in test_zeros:
            t_true = float(zetazero(n).imag)
            
            err_ram = abs(self.ramanujan.predict(n) - t_true)
            err_geo = abs(self.geometric.predict(n) - t_true)
            err_dim = abs(self.solver.solve(n) - t_true)
            
            # Dimensional should be at least 1000x better
            assert err_dim < err_ram / 1000, f"n={n}: Should be 1000x better than Ramanujan"
            assert err_dim < err_geo / 1000, f"n={n}: Should be 1000x better than Geometric"
    
    def test_consistency(self):
        """Test that repeated solves give consistent results."""
        n = 100
        
        results = [self.solver.solve(n) for _ in range(5)]
        
        # All results should be identical
        for r in results[1:]:
            assert r == results[0], "Repeated solves should give identical results"
    
    def test_z_function_at_solution(self):
        """Test that |Z(t)| is small at the solution."""
        for n in [10, 50, 100, 500]:
            result = self.solver.verify(n)
            assert result['Z_at_t'] < 1e-10, f"n={n}: |Z(t)| should be < 1e-10"


class TestScaling:
    """Test scaling behavior with zero index."""
    
    def setup_method(self):
        self.solver = DimensionalDowncaster()
    
    def test_time_scaling(self):
        """Test that time scales reasonably with n."""
        times = []
        n_values = [100, 500, 1000]
        
        for n in n_values:
            start = time.time()
            self.solver.solve(n)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Time should not grow faster than O(n)
        # (Actually should be O(log t) which is even slower growth)
        ratio = times[-1] / times[0]
        n_ratio = n_values[-1] / n_values[0]
        
        assert ratio < n_ratio, "Time should scale sub-linearly"
    
    def test_z_evals_scaling(self):
        """Test that Z evaluations are roughly constant."""
        n_values = [100, 500, 1000]
        z_evals = []
        
        for n in n_values:
            self.solver.stats = {'Z_evals': 0, 'zeros_solved': 0}
            self.solver.solve(n)
            z_evals.append(self.solver.stats['Z_evals'])
        
        # Z evaluations should be roughly constant (within 2x)
        assert max(z_evals) / min(z_evals) < 2, "Z evaluations should be roughly constant"


class TestNSmoothProperty:
    """Test the key N_smooth ≈ n - 0.5 property."""
    
    def setup_method(self):
        self.solver = DimensionalDowncaster()
    
    def test_offset_statistics(self):
        """Test statistics of the N_smooth offset."""
        offsets = []
        
        for n in range(1, 101):
            t_n = float(zetazero(n).imag)
            N_s = self.solver._N_smooth(t_n)
            offset = N_s - (n - 0.5)
            offsets.append(offset)
        
        mean_offset = np.mean(offsets)
        std_offset = np.std(offsets)
        
        # Mean should be close to 0
        assert abs(mean_offset) < 0.2, f"Mean offset {mean_offset} should be < 0.2"
        
        # Std should be small
        assert std_offset < 0.3, f"Std offset {std_offset} should be < 0.3"
    
    def test_offset_bounded(self):
        """Test that offset is bounded."""
        for n in range(1, 101):
            t_n = float(zetazero(n).imag)
            N_s = self.solver._N_smooth(t_n)
            offset = abs(N_s - (n - 0.5))
            
            assert offset < 0.5, f"n={n}: Offset {offset} should be < 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
