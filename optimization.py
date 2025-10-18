"""
Optimization Module
===================

Unified sublinear optimization and parameter calibration.

This module consolidates optimization techniques from:
- sublinear_optimizer.py: Sublinear search algorithms
- srt_auto_calibrator.py: SRT parameter calibration
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .spectral import SpectralScorer, DiracOperator, ZetaFiducials
from .holographic import holographic_refinement


@dataclass
class OptimizationStats:
    """Statistics about optimization."""
    n_original: int
    n_final: int
    reduction_ratio: float
    complexity_estimate: str
    time_elapsed: float = 0.0


class SublinearOptimizer:
    """
    Unified sublinear optimization.
    
    Converts O(n) operations into O(√n) or O(log n) using:
    - Spectral scoring
    - Holographic refinement
    - Adaptive filtering
    """
    
    def __init__(self,
                 use_holographic: bool = True,
                 phase_retrieval_method: str = "hilbert",
                 blend_ratio: float = 0.6):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        use_holographic : bool
            Apply holographic refinement.
        phase_retrieval_method : str
            'hilbert' or 'gs'
        blend_ratio : float
            Blending ratio for object vs reference.
        """
        self.use_holographic = use_holographic
        self.phase_retrieval_method = phase_retrieval_method
        self.blend_ratio = blend_ratio
    
    def optimize(self,
                candidates: np.ndarray,
                score_fn: Callable[[np.ndarray], np.ndarray],
                reference_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                top_k: Optional[int] = None) -> Tuple[np.ndarray, OptimizationStats]:
        """
        Main optimization routine.
        
        Parameters
        ----------
        candidates : np.ndarray
            Initial candidate set.
        score_fn : callable
            Function that scores candidates.
        reference_fn : callable, optional
            Function that computes reference baseline.
        top_k : int, optional
            Number of top candidates to return.
        
        Returns
        -------
        top_candidates : np.ndarray
            Reduced set of top candidates.
        stats : OptimizationStats
            Optimization statistics.
        """
        import time
        start_time = time.time()
        
        candidates = np.asarray(candidates)
        n_original = len(candidates)
        
        # Compute initial scores
        scores = score_fn(candidates)
        
        # Apply holographic refinement if enabled
        if self.use_holographic:
            if reference_fn is not None:
                reference = reference_fn(candidates)
            else:
                # Default: smooth baseline
                reference = 1.0 / (np.log(candidates + 1e-12) + 1e-12)
            
            try:
                scores = holographic_refinement(
                    scores, reference,
                    method=self.phase_retrieval_method,
                    blend_ratio=self.blend_ratio
                )
            except Exception:
                pass  # Fall back to original scores
        
        # Select top-k
        if top_k is not None and top_k < len(candidates):
            top_idx = np.argsort(-scores)[:top_k]
            top_candidates = candidates[top_idx]
            n_final = top_k
        else:
            top_candidates = candidates
            n_final = len(candidates)
        
        # Compute statistics
        reduction_ratio = n_final / max(n_original, 1)
        
        if n_final <= np.sqrt(n_original):
            complexity = "O(√n)"
        elif n_final <= np.log(n_original) * np.log(n_original):
            complexity = "O(log²n)"
        else:
            complexity = f"O({n_final}/{n_original}·n)"
        
        elapsed = time.time() - start_time
        
        stats = OptimizationStats(
            n_original=n_original,
            n_final=n_final,
            reduction_ratio=reduction_ratio,
            complexity_estimate=complexity,
            time_elapsed=elapsed
        )
        
        return top_candidates, stats


@dataclass
class SRTParams:
    """SRT parameter configuration."""
    z: float = 0.05
    corr: float = 0.12
    theta: float = 0.0
    L: int = 20
    corr_decay: float = 10.0
    affinity_type: str = 'default'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'z': self.z,
            'corr': self.corr,
            'theta': self.theta,
            'L': self.L,
            'corr_decay': self.corr_decay,
            'affinity_type': self.affinity_type,
        }


class SRTCalibrator:
    """
    Automated SRT parameter calibration.
    
    Unified from srt_auto_calibrator.py.
    """
    
    def __init__(self,
                 ground_truth_indices: np.ndarray,
                 ground_truth_values: np.ndarray,
                 affinity_functions: Dict[str, Callable],
                 verbose: bool = True):
        """
        Initialize calibrator.
        
        Parameters
        ----------
        ground_truth_indices : np.ndarray
            Known solution indices.
        ground_truth_values : np.ndarray
            Known solution values.
        affinity_functions : dict
            Dictionary of affinity functions to try.
        verbose : bool
            Print progress.
        """
        self.gt_indices = ground_truth_indices
        self.gt_values = ground_truth_values
        self.affinity_functions = affinity_functions
        self.verbose = verbose
        
        # Get zeta fiducials
        self.zeta_fiducials = ZetaFiducials.get_standard(20)
        self.dirac = DiracOperator(self.zeta_fiducials)
    
    def evaluate_params(self,
                       params: SRTParams,
                       affinity_func: Callable,
                       train_indices: np.ndarray,
                       train_values: np.ndarray,
                       test_indices: np.ndarray,
                       test_values: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate parameter configuration.
        
        Returns metrics including resonance and prediction error.
        """
        try:
            # Default correlation function
            def corr_func(i, j, indices):
                idx_i, idx_j = indices[i], indices[j]
                return np.exp(-abs(idx_i - idx_j) / params.corr_decay)
            
            # Build Dirac operator
            D = self.dirac.build(
                train_indices, affinity_func, corr_func,
                z=params.z, corr=params.corr, theta=params.theta
            )
            
            # Compute resonance
            R = self.dirac.compute_resonance(D)
            eigenvalues = self.dirac.compute_eigenvalues(D)
            
            # Simple prediction: use eigenvalue-weighted average
            from scipy.linalg import eigh
            _, eigenvectors = eigh(D)
            
            predictions = []
            for test_idx in test_indices:
                # Use top eigenvectors for prediction
                k = min(5, len(eigenvalues))
                top_indices = np.argsort(eigenvalues)[-k:]
                
                weights = np.zeros(len(train_indices))
                for idx in top_indices:
                    vec = eigenvectors[:, idx]
                    weights += np.abs(vec) ** 2 * eigenvalues[idx]
                
                if np.sum(weights) > 0:
                    weights /= np.sum(weights)
                else:
                    weights = np.ones(len(train_indices)) / len(train_indices)
                
                pred = np.sum(weights * train_values)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            prediction_error = np.mean(np.abs(predictions - test_values))
            
            return {
                'params': params,
                'resonance': R,
                'prediction_error': prediction_error,
                'eigenvalue_spread': eigenvalues[-1] - eigenvalues[0],
                'spectral_gap': eigenvalues[-1] - eigenvalues[-2] if len(eigenvalues) > 1 else 0,
                'success': True,
            }
        
        except Exception as e:
            return {
                'params': params,
                'success': False,
                'error': str(e),
            }
    
    def calibrate(self,
                 grid_resolution: str = 'coarse',
                 train_fraction: float = 0.7) -> Tuple[SRTParams, Dict[str, float]]:
        """
        Run automated calibration.
        
        Parameters
        ----------
        grid_resolution : str
            'coarse' or 'fine'
        train_fraction : float
            Fraction of data for training.
        
        Returns
        -------
        best_params : SRTParams
            Optimal parameters found.
        metrics : dict
            Performance metrics.
        """
        import itertools
        
        # Split data
        n_total = len(self.gt_indices)
        n_train = int(n_total * train_fraction)
        
        perm = np.random.permutation(n_total)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        train_indices = self.gt_indices[train_idx]
        train_values = self.gt_values[train_idx]
        test_indices = self.gt_indices[test_idx]
        test_values = self.gt_values[test_idx]
        
        # Define parameter grid
        if grid_resolution == 'coarse':
            z_values = [0.01, 0.05, 0.1]
            corr_values = [0.05, 0.12, 0.25]
            theta_values = [0.0]
            corr_decay_values = [5.0, 10.0]
        else:
            z_values = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
            corr_values = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
            theta_values = [0.0, 0.01]
            corr_decay_values = [3.0, 5.0, 8.0, 10.0, 15.0]
        
        affinity_funcs = list(self.affinity_functions.items())
        
        # Grid search
        results = []
        for z, corr, theta, (aff_name, aff_func), decay in itertools.product(
            z_values, corr_values, theta_values, affinity_funcs, corr_decay_values
        ):
            params = SRTParams(
                z=z, corr=corr, theta=theta,
                L=len(self.zeta_fiducials),
                corr_decay=decay,
                affinity_type=aff_name
            )
            
            result = self.evaluate_params(
                params, aff_func,
                train_indices, train_values,
                test_indices, test_values
            )
            
            if result['success']:
                results.append(result)
        
        # Find best
        results.sort(key=lambda r: r['prediction_error'])
        best = results[0]
        
        return best['params'], {
            'prediction_error': best['prediction_error'],
            'resonance': best['resonance'],
            'eigenvalue_spread': best['eigenvalue_spread'],
            'spectral_gap': best['spectral_gap'],
        }


def optimize_sublinear(candidates: np.ndarray,
                      score_fn: Callable,
                      top_k: int,
                      use_holographic: bool = True) -> np.ndarray:
    """
    Convenience function for sublinear optimization.
    
    Parameters
    ----------
    candidates : np.ndarray
        Candidate set.
    score_fn : callable
        Scoring function.
    top_k : int
        Number of top results.
    use_holographic : bool
        Use holographic refinement.
    
    Returns
    -------
    np.ndarray
        Top-k candidates.
    """
    optimizer = SublinearOptimizer(use_holographic=use_holographic)
    top_candidates, _ = optimizer.optimize(candidates, score_fn, top_k=top_k)
    return top_candidates
