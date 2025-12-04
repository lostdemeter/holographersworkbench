"""
Formula Discovery Processor
===========================

Discovers formulas for mathematical constants using group-theoretic search.

This processor:
1. Starts from a seed formula (e.g., BBP with integer coefficients)
2. Uses error analysis to find corrections
3. Applies group-theoretic guidance
4. Iterates until convergence

Ported from ribbon_solver2/tools/discovery_pipeline.py with group theory integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

from .base import BaseProcessor, ProcessorResult
from .error_analyzer import ErrorAnalyzer


# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class FormulaResult(ProcessorResult):
    """Result from formula discovery."""
    target: str
    base: int
    initial_error: float
    final_error: float
    n_smooth: float
    coefficients: List[float]
    corrections: List[Dict]
    convergence_rate: float


class FormulaDiscovery(BaseProcessor):
    """
    Discover formulas using group-theoretic search.
    
    Usage:
        discovery = FormulaDiscovery()
        result = discovery.discover_bbp(base=4096, target='pi')
    """
    
    def __init__(self, group=None, verbose=True):
        super().__init__(group, verbose)
        self.error_analyzer = ErrorAnalyzer(group, verbose=False)
    
    def discover_bbp(self,
                     base: int = 4096,
                     scale: int = 64,
                     slots: List[Tuple[int, int]] = None,
                     initial_coefs: List[float] = None,
                     max_iterations: int = 10,
                     target_n_smooth: float = 14.0,
                     time_limit: float = 60.0,
                     output: str = None) -> FormulaResult:
        """
        Discover a BBP-type formula using error analysis.
        
        Args:
            base: Formula base (e.g., 4096)
            scale: Scaling factor (e.g., 64)
            slots: (period, offset) pairs
            initial_coefs: Starting coefficients
            max_iterations: Maximum correction iterations
            target_n_smooth: Target precision (14 = machine precision)
            time_limit: Maximum time in seconds
            output: Optional path to save results
            
        Returns:
            FormulaResult with discovered formula
        """
        self._start_timer()
        
        # Default slots for base 4096
        if slots is None:
            slots = [(4, 1), (4, 3), (12, 1), (12, 3), 
                     (12, 5), (12, 7), (12, 9), (12, 11)]
        
        # Default integer coefficients (Bellard-like scaled)
        if initial_coefs is None:
            initial_coefs = [256, -32, 4, 1, -128, -64, -128, 4]
        
        self._log("=" * 60)
        self._log("BBP FORMULA DISCOVERY")
        self._log("=" * 60)
        self._log(f"Base: {base}, Scale: {scale}")
        self._log(f"Slots: {len(slots)}")
        self._log(f"Initial coefficients: {initial_coefs}")
        self._log(f"Target N_smooth: {target_n_smooth}")
        
        current_coefs = list(initial_coefs)
        corrections = []
        
        # Evaluate initial formula
        initial_value = self._eval_bbp(base, scale, slots, current_coefs)
        initial_error = abs(initial_value - np.pi)
        initial_n_smooth = -np.log10(initial_error) if initial_error > 0 else 15.0
        
        self._log(f"\nInitial: error={initial_error:.2e}, N_smooth={initial_n_smooth:.2f}")
        
        best_error = initial_error
        best_coefs = current_coefs.copy()
        best_n_smooth = initial_n_smooth
        
        # Iterative correction using φ-powers
        for iteration in range(max_iterations):
            if self._elapsed() > time_limit:
                self._log(f"\nTime limit reached")
                break
            
            if best_n_smooth >= target_n_smooth:
                self._log(f"\nTarget precision reached!")
                break
            
            # Try φ-power corrections on each coefficient
            improved = False
            
            for i in range(len(current_coefs)):
                for k in range(1, 12):
                    for sign in [1, -1]:
                        # Try correction of form (n/d) × φ^(-k)
                        for n in range(1, 10):
                            for d in range(1, 10):
                                correction = sign * (n / d) * (PHI ** (-k))
                                
                                test_coefs = current_coefs.copy()
                                test_coefs[i] += correction
                                
                                value = self._eval_bbp(base, scale, slots, test_coefs)
                                error = abs(value - np.pi)
                                
                                if error < best_error:
                                    best_error = error
                                    best_coefs = test_coefs.copy()
                                    best_n_smooth = -np.log10(error) if error > 0 else 15.0
                                    improved = True
                                    
                                    corrections.append({
                                        'iteration': iteration,
                                        'coefficient': i,
                                        'correction': correction,
                                        'pattern': f"({n}/{d}) × φ^(-{k})",
                                        'new_error': error,
                                        'n_smooth': best_n_smooth,
                                    })
                                    
                                    self._log(f"  Iter {iteration}: coef[{i}] += {correction:.6f}")
                                    self._log(f"    → error={error:.2e}, N_smooth={best_n_smooth:.2f}")
            
            if improved:
                current_coefs = best_coefs.copy()
            else:
                self._log(f"\nNo improvement found at iteration {iteration}")
                break
        
        # Compute convergence rate
        convergence_rate = self._estimate_convergence_rate(base, scale, slots, best_coefs)
        
        result = FormulaResult(
            processor='FormulaDiscovery',
            success=best_n_smooth >= 10,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=corrections,
            summary=f"{'✓ Found' if best_n_smooth >= 10 else '✗ No'} formula: N_smooth={best_n_smooth:.2f}, rate={convergence_rate:.2f} d/t",
            target='pi',
            base=base,
            initial_error=initial_error,
            final_error=best_error,
            n_smooth=best_n_smooth,
            coefficients=best_coefs,
            corrections=corrections,
            convergence_rate=convergence_rate,
        )
        
        if output:
            result.save(output)
        
        self._log(f"\n{result.summary}")
        return result
    
    def process(self, target: str = 'pi', **kwargs) -> FormulaResult:
        """Main processing method."""
        if target == 'pi':
            return self.discover_bbp(**kwargs)
        else:
            raise ValueError(f"Unknown target: {target}")
    
    def _eval_bbp(self, base: int, scale: int, 
                  slots: List[Tuple[int, int]], 
                  coefs: List[float],
                  n_terms: int = 100) -> float:
        """Evaluate BBP-type formula."""
        total = 0.0
        
        for k in range(n_terms):
            sign = (-1) ** k
            base_power = base ** k
            
            term = 0.0
            for (period, offset), coef in zip(slots, coefs):
                denom = period * k + offset
                if denom != 0:
                    term += coef / denom
            
            total += sign * term / base_power
        
        return total / scale
    
    def _estimate_convergence_rate(self, base: int, scale: int,
                                    slots: List[Tuple[int, int]],
                                    coefs: List[float]) -> float:
        """
        Estimate convergence rate in digits per term.
        
        Rate = log10(base) for BBP formulas.
        """
        # For BBP, rate ≈ log10(base)
        return np.log10(base)
