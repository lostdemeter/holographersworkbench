"""
Error Analyzer Processor
========================

Uses ERROR AS SIGNAL to discover mathematical structure.

Key insight from φ-BBP discovery:
- When a formula is "almost" correct, the error has STRUCTURE
- Error structure reveals the mathematical identity we're missing
- φ-BBP was discovered by finding φ^(-k) patterns in BBP errors

This processor:
1. Takes a formula that's "close" to a target
2. Analyzes the error structure
3. Identifies patterns (φ, e, sqrt2, etc.)
4. Proposes corrections

Ported from ribbon_solver2/tools/error_analyzer.py with group theory integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict

from .base import BaseProcessor, ProcessorResult


# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2
E = np.e
SQRT2 = np.sqrt(2)
LN2 = np.log(2)


@dataclass
class ErrorPattern:
    """A detected pattern in error structure."""
    pattern_type: str  # 'phi_power', 'e_power', 'rational', 'arctan'
    parameters: Dict
    fit_quality: float  # R² or similar (0-1)
    correction: float
    description: str
    anchor: str  # Which truth space anchor this relates to


@dataclass
class ErrorResult(ProcessorResult):
    """Result from error analysis."""
    formula_name: str
    base_error: float
    patterns_found: List[Dict]
    best_correction: float
    corrected_error: float
    improvement_factor: float


class ErrorAnalyzer(BaseProcessor):
    """
    Analyze error structure to discover corrections.
    
    Philosophy: Error is not noise - it's signal pointing to missing math.
    
    Usage:
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze_formula(my_formula, target=np.pi)
        result = analyzer.analyze_coefficients([256.02, -32.05, 4.01])
    """
    
    # Constants to check for patterns
    CONSTANTS = {
        'phi': (PHI, 'growth'),
        'phi_inv': (1/PHI, 'growth'),
        'e': (E, 'growth'),
        'e_inv': (1/E, 'growth'),
        'sqrt2': (SQRT2, 'inverse'),
        'sqrt2_inv': (1/SQRT2, 'inverse'),
        'ln2': (LN2, 'growth'),
        'pi': (np.pi, 'pattern'),
    }
    
    def __init__(self, group=None, verbose=True):
        super().__init__(group, verbose)
    
    def analyze_formula(self,
                        formula: Callable[[int], float],
                        formula_name: str = "unknown",
                        target: float = None,
                        output: str = None) -> ErrorResult:
        """
        Analyze error structure of a formula.
        
        Args:
            formula: Function that takes n_terms and returns value
            formula_name: Name for reporting
            target: Target value (default: π)
            output: Optional path to save results
            
        Returns:
            ErrorResult with detected patterns
        """
        self._start_timer()
        
        if target is None:
            target = np.pi
        
        self._log(f"Analyzing error structure of {formula_name}...")
        
        # Compute errors at different term counts
        errors = []
        for n in [5, 10, 20, 50, 100, 200, 500]:
            try:
                value = formula(n)
                error = value - target
                errors.append((n, error))
            except:
                pass
        
        if not errors:
            return ErrorResult(
                processor='ErrorAnalyzer',
                success=False,
                timestamp=self._timestamp(),
                elapsed_time=self._elapsed(),
                findings=[],
                summary="Could not evaluate formula",
                formula_name=formula_name,
                base_error=float('inf'),
                patterns_found=[],
                best_correction=0.0,
                corrected_error=float('inf'),
                improvement_factor=1.0,
            )
        
        # Get base error (at highest term count)
        base_error = abs(errors[-1][1])
        
        # Find patterns in error structure
        patterns = self._find_patterns(errors)
        
        # Apply best correction
        best_correction = 0.0
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.fit_quality)
            best_correction = best_pattern.correction
        
        # Compute corrected error
        corrected_value = formula(500) + best_correction
        corrected_error = abs(corrected_value - target)
        
        improvement = base_error / corrected_error if corrected_error > 0 else float('inf')
        
        result = ErrorResult(
            processor='ErrorAnalyzer',
            success=len(patterns) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=[asdict(p) for p in patterns],
            summary=f"Found {len(patterns)} patterns, {improvement:.1f}× improvement",
            formula_name=formula_name,
            base_error=base_error,
            patterns_found=[asdict(p) for p in patterns],
            best_correction=best_correction,
            corrected_error=corrected_error,
            improvement_factor=improvement,
        )
        
        if output:
            result.save(output)
        
        self._log(f"\n{result.summary}")
        return result
    
    def analyze_coefficients(self,
                             coefficients: List[float],
                             integer_targets: List[int] = None,
                             output: str = None) -> ErrorResult:
        """
        Analyze coefficient deviations for patterns.
        
        This is how φ-BBP was discovered - by finding φ^(-k) patterns
        in the deviations of BBP coefficients from integers.
        
        Args:
            coefficients: Current coefficient values
            integer_targets: Expected integer values (if known)
            output: Optional path to save results
            
        Returns:
            ErrorResult with detected patterns
        """
        self._start_timer()
        
        if integer_targets is None:
            integer_targets = [round(c) for c in coefficients]
        
        self._log(f"Analyzing {len(coefficients)} coefficient deviations...")
        
        patterns = []
        total_correction = 0.0
        
        for i, (coef, target) in enumerate(zip(coefficients, integer_targets)):
            diff = coef - target
            
            if abs(diff) < 1e-10:
                continue
            
            self._log(f"  Coefficient {i}: {coef:.6f} (target: {target})")
            self._log(f"    Deviation: {diff:.6f}")
            
            # Check for φ-power patterns
            phi_patterns = self._find_phi_patterns(diff, i)
            patterns.extend(phi_patterns)
            
            # Check for rational patterns
            rational_patterns = self._find_rational_patterns(diff, i)
            patterns.extend(rational_patterns)
        
        # Sum up corrections
        if patterns:
            total_correction = sum(p.correction for p in patterns)
        
        result = ErrorResult(
            processor='ErrorAnalyzer',
            success=len(patterns) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=[asdict(p) for p in patterns],
            summary=f"Found {len(patterns)} patterns in coefficient deviations",
            formula_name="coefficients",
            base_error=sum(abs(c - t) for c, t in zip(coefficients, integer_targets)),
            patterns_found=[asdict(p) for p in patterns],
            best_correction=total_correction,
            corrected_error=0.0,  # Would need formula to compute
            improvement_factor=1.0,
        )
        
        if output:
            result.save(output)
        
        return result
    
    def process(self, target, **kwargs) -> ErrorResult:
        """Main processing method."""
        if callable(target):
            return self.analyze_formula(target, **kwargs)
        elif isinstance(target, list):
            return self.analyze_coefficients(target, **kwargs)
        else:
            raise ValueError("Target must be a formula (callable) or coefficients (list)")
    
    def _find_patterns(self, errors: List[Tuple[int, float]]) -> List[ErrorPattern]:
        """Find patterns in error sequence."""
        patterns = []
        
        if len(errors) < 2:
            return patterns
        
        # Check for exponential decay
        n_vals = np.array([e[0] for e in errors])
        err_vals = np.array([abs(e[1]) for e in errors])
        
        # Avoid log of zero
        err_vals = np.maximum(err_vals, 1e-20)
        
        # Fit log(error) vs n
        try:
            coeffs = np.polyfit(n_vals, np.log(err_vals), 1)
            decay_rate = -coeffs[0]
            
            if decay_rate > 0:
                # Check which constant this relates to
                for name, (value, anchor) in self.CONSTANTS.items():
                    if abs(decay_rate - np.log(value)) < 0.1:
                        patterns.append(ErrorPattern(
                            pattern_type=f'{name}_decay',
                            parameters={'decay_rate': decay_rate, 'constant': name},
                            fit_quality=0.8,
                            correction=0.0,  # Decay pattern doesn't give direct correction
                            description=f"Error decays like {name}^(-n)",
                            anchor=anchor,
                        ))
        except:
            pass
        
        return patterns
    
    def _find_phi_patterns(self, diff: float, index: int) -> List[ErrorPattern]:
        """Find φ-power patterns in a deviation."""
        patterns = []
        
        # Check φ^(-k) for various k
        for k in range(1, 15):
            phi_power = PHI ** (-k)
            
            # Check if diff ≈ (n/d) × φ^(-k) for small n, d
            for n in range(-20, 21):
                if n == 0:
                    continue
                for d in range(1, 20):
                    candidate = (n / d) * phi_power
                    
                    if abs(diff - candidate) < abs(diff) * 0.01:  # 1% match
                        fit = 1.0 - abs(diff - candidate) / abs(diff)
                        
                        patterns.append(ErrorPattern(
                            pattern_type='phi_power',
                            parameters={'n': n, 'd': d, 'k': k},
                            fit_quality=fit,
                            correction=-candidate,  # Correction is negative of pattern
                            description=f"({n}/{d}) × φ^(-{k})",
                            anchor='growth',
                        ))
                        
                        if fit > 0.99:
                            return patterns  # Found excellent match
        
        return patterns
    
    def _find_rational_patterns(self, diff: float, index: int) -> List[ErrorPattern]:
        """Find rational patterns in a deviation."""
        patterns = []
        
        # Check simple fractions
        for d in range(1, 100):
            n = round(diff * d)
            if n == 0:
                continue
            
            candidate = n / d
            if abs(diff - candidate) < abs(diff) * 0.001:  # 0.1% match
                fit = 1.0 - abs(diff - candidate) / abs(diff)
                
                patterns.append(ErrorPattern(
                    pattern_type='rational',
                    parameters={'n': n, 'd': d},
                    fit_quality=fit,
                    correction=-candidate,
                    description=f"{n}/{d}",
                    anchor='stability',
                ))
                
                if fit > 0.999:
                    return patterns
        
        return patterns
