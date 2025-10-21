"""
Spectral Analysis Module
========================

Unified spectral scoring and frequency-domain analysis.

This module consolidates spectral techniques from:
- sublinear_optimizer.py: SpectralScorer
- zeta_chudnovsky.py: compute_spectral_scores
- srt_auto_calibrator.py: Dirac operator construction
"""

import numpy as np
from typing import Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""
    frequencies: np.ndarray
    damping: float = 0.05
    shift: float = 0.0
    use_taper: bool = True
    taper_type: str = "gaussian"  # or "exponential"


class ZetaFiducials:
    """
    Manage Riemann zeta zeros as spectral fiducials.
    
    Unified interface for computing and caching zeta zeros
    across all workbench modules.
    """
    
    _cache = {}
    
    @classmethod
    def compute(cls, n: int, method: str = "fast_zetas") -> np.ndarray:
        """
        Compute first n zeta zeros using fast_zetas.
        
        Parameters
        ----------
        n : int
            Number of zeros to compute.
        method : str
            Method to use. Default 'fast_zetas' (only supported method).
            'auto' is an alias for 'fast_zetas'.
        
        Returns
        -------
        np.ndarray
            Imaginary parts of first n zeta zeros.
        """
        if n in cls._cache:
            return cls._cache[n]
        
        # Handle 'auto' as alias for 'fast_zetas'
        if method == "auto":
            method = "fast_zetas"
        
        if method == "fast_zetas":
            try:
                from fast_zetas import zetazero
                zeros = np.array([float(zetazero(k)) for k in range(1, n + 1)])
                cls._cache[n] = zeros
                return zeros
            except ImportError:
                raise ImportError(
                    "fast_zetas module required for zeta zero computation. "
                    "This workbench uses mathematically-based fast_zetas instead of mpmath."
                )
        
        raise ValueError(f"Unknown method: {method}. Only 'fast_zetas' (or 'auto') is supported.")
    
    @classmethod
    def get_standard(cls, count: int = 20) -> np.ndarray:
        """Get standard set of zeta zeros for general use."""
        return cls.compute(count)


class SpectralScorer:
    """
    Unified spectral scoring using oscillatory patterns.
    
    Consolidates scoring methods from multiple modules.
    """
    
    def __init__(self, 
                 frequencies: Optional[np.ndarray] = None,
                 damping: float = 0.05,
                 use_zeta: bool = False,
                 n_zeta: int = 20):
        """
        Initialize spectral scorer.
        
        Parameters
        ----------
        frequencies : np.ndarray, optional
            Custom frequencies. If None and use_zeta=True, uses zeta zeros.
        damping : float
            Damping factor for high frequencies.
        use_zeta : bool
            Use Riemann zeta zeros as frequencies.
        n_zeta : int
            Number of zeta zeros if use_zeta=True.
        """
        if frequencies is not None:
            self.frequencies = np.asarray(frequencies)
        elif use_zeta:
            self.frequencies = ZetaFiducials.get_standard(n_zeta)
        else:
            # Default: low harmonics
            self.frequencies = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
        
        self.damping = damping
    
    def compute_scores(self, 
                      candidates: np.ndarray,
                      shift: float = 0.0,
                      mode: str = "complex") -> np.ndarray:
        """
        Compute spectral scores for candidates.
        
        Parameters
        ----------
        candidates : np.ndarray
            Candidate values to score.
        shift : float
            Phase shift parameter.
        mode : str
            'complex' (full complex), 'real' (real part only), 
            'magnitude' (absolute value)
        
        Returns
        -------
        np.ndarray
            Spectral scores.
        """
        candidates = np.asarray(candidates, dtype=float)
        log_vals = np.log(candidates + 1e-12)
        
        # Compute oscillatory sum
        osc = np.zeros(len(candidates), dtype=complex)
        
        for gamma in self.frequencies:
            # Damping (Gaussian taper)
            if self.damping > 0:
                taper = np.exp(-0.5 * self.damping**2 * gamma**2)
            else:
                taper = 1.0
            
            # Phase shift
            phase_shift = (0.5 + 1j * gamma) * shift
            
            # Oscillatory term
            phases = np.exp(1j * gamma * log_vals + phase_shift)
            osc += taper * phases / (0.5 + 1j * gamma + 1e-12)
        
        # Return based on mode
        if mode == "complex":
            return osc
        elif mode == "real":
            return np.real(osc)
        elif mode == "magnitude":
            return np.abs(osc)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute_interference(self,
                            candidates: np.ndarray,
                            h: float = 0.05) -> np.ndarray:
        """
        Compute interference pattern scores (zeta_chudnovsky style).
        
        Parameters
        ----------
        candidates : np.ndarray
            Candidate values.
        h : float
            Interference parameter.
        
        Returns
        -------
        np.ndarray
            Interference-based scores.
        """
        log_ns = np.log(candidates + 1e-12)
        osc_plus = np.zeros(len(candidates), dtype=complex)
        osc_minus = np.zeros(len(candidates), dtype=complex)
        
        for gamma in self.frequencies:
            taper = np.exp(-0.5 * h**2 * gamma**2)
            shift_plus = (0.5 + 1j * gamma) * h
            shift_minus = (0.5 + 1j * gamma) * (-h)
            
            phases_plus = np.exp(1j * gamma * log_ns + shift_plus)
            phases_minus = np.exp(1j * gamma * log_ns + shift_minus)
            
            osc_plus += taper * phases_plus / (0.5 + 1j * gamma + 1e-12)
            osc_minus += taper * phases_minus / (0.5 + 1j * gamma + 1e-12)
        
        psi_plus = np.array(candidates) * np.exp(h) - 2 * np.real(osc_plus)
        psi_minus = np.array(candidates) * np.exp(-h) - 2 * np.real(osc_minus)
        
        logn_arr = np.log(candidates + 1e-12)
        scores = (psi_plus - psi_minus) / (2 * h * np.array(candidates) * logn_arr + 1e-12)
        
        return scores


def compute_spectral_scores(candidates: np.ndarray,
                            frequencies: np.ndarray,
                            h: float = 0.05,
                            method: str = "interference") -> np.ndarray:
    """
    Convenience function for computing spectral scores.
    
    Parameters
    ----------
    candidates : np.ndarray
        Values to score.
    frequencies : np.ndarray
        Spectral frequencies (e.g., zeta zeros).
    h : float
        Modulation parameter.
    method : str
        'interference' or 'oscillatory'
    
    Returns
    -------
    np.ndarray
        Spectral scores.
    """
    scorer = SpectralScorer(frequencies=frequencies)
    
    if method == "interference":
        return scorer.compute_interference(candidates, h=h)
    elif method == "oscillatory":
        return scorer.compute_scores(candidates, shift=h, mode="real")
    else:
        raise ValueError(f"Unknown method: {method}")


class DiracOperator:
    """
    Dirac operator construction for SRT-style spectral analysis.
    
    Unified from srt_auto_calibrator.py.
    """
    
    def __init__(self,
                 zeta_fiducials: Optional[np.ndarray] = None,
                 n_fiducials: int = 20):
        """
        Initialize Dirac operator builder.
        
        Parameters
        ----------
        zeta_fiducials : np.ndarray, optional
            Zeta zeros to use. If None, computes standard set.
        n_fiducials : int
            Number of fiducials if not provided.
        """
        if zeta_fiducials is None:
            self.fiducials = ZetaFiducials.get_standard(n_fiducials)
        else:
            self.fiducials = np.asarray(zeta_fiducials)
    
    def build(self,
              indices: np.ndarray,
              affinity_func: Callable[[float], float],
              correlation_func: Callable[[int, int, np.ndarray], float],
              z: float = 0.05,
              corr: float = 0.12,
              theta: float = 0.0) -> np.ndarray:
        """
        Build Dirac operator matrix.
        
        Parameters
        ----------
        indices : np.ndarray
            Element indices.
        affinity_func : callable
            Function mapping index to affinity p ∈ [0,1].
        correlation_func : callable
            Function mapping (i, j, indices) to correlation γ_ij.
        z : float
            Zeta modulation strength.
        corr : float
            Correlation weight.
        theta : float
            Noncommutative deformation.
        
        Returns
        -------
        np.ndarray
            Hermitian Dirac operator matrix.
        """
        n = len(indices)
        L = len(self.fiducials)
        
        D = np.zeros((n, n), dtype=complex)
        
        # Diagonal: affinities
        for i, idx in enumerate(indices):
            D[i, i] = affinity_func(idx)
        
        # Off-diagonal: zeta-modulated correlations
        for i in range(n):
            for j in range(i + 1, n):
                sign = (-1) ** (i + j)
                zeta_idx = (i + j) % L
                zeta_phase = self.fiducials[zeta_idx]
                gamma_ij = corr * correlation_func(i, j, indices)
                delta_ij = sign * gamma_ij * zeta_phase * z
                
                if theta > 0:
                    delta_ij *= (1 + theta * np.random.randn() * 0.1)
                
                D[i, j] = delta_ij
                D[j, i] = np.conj(delta_ij)
        
        return D
    
    def compute_eigenvalues(self, D: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of Dirac operator."""
        from scipy.linalg import eigh
        return eigh(D, eigvals_only=True)
    
    def compute_resonance(self, D: np.ndarray) -> float:
        """
        Compute resonance functional R(S).
        
        Lower values indicate better eigenvalue separation.
        """
        eigenvalues = self.compute_eigenvalues(D)
        
        R = 0.0
        for i in range(len(eigenvalues)):
            for j in range(i + 1, len(eigenvalues)):
                diff = abs(eigenvalues[i] - eigenvalues[j])
                R += 1.0 / (diff + 1e-6)
        
        return R
