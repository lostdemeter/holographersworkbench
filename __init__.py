"""
Holographer's Workbench
=======================

A unified toolkit for holographic signal processing, spectral optimization,
and sublinear algorithms.

Core Modules:
-------------
- spectral: Frequency-domain analysis and zeta-based scoring
- holographic: Phase retrieval, interference patterns, and signal refinement
- optimization: Sublinear algorithms and parameter calibration
- compression: Lossless holographic image compression
- utils: Common utilities and helper functions

Philosophy:
-----------
The workbench unifies techniques from:
1. Holography (interference, phase retrieval)
2. Spectral theory (Fourier, zeta zeros, eigenvalues)
3. Optimization (sublinear search, SRT calibration)
4. Signal processing (envelope detection, filtering)

All tools share common patterns:
- Spectral decomposition
- Phase/amplitude separation
- Interference-based enhancement
- Adaptive parameter tuning
"""

__version__ = "0.1.0"

# Core imports
from .spectral import (
    SpectralScorer,
    ZetaFiducials,
    compute_spectral_scores,
)

from .holographic import (
    PhaseRetrieval,
    phase_retrieve_hilbert,
    phase_retrieve_gs,
    align_phase,
    holographic_refinement,
)

from .optimization import (
    SublinearOptimizer,
    SRTCalibrator,
    optimize_sublinear,
)

from .utils import (
    compute_envelope,
    normalize_signal,
    adaptive_blend,
    compute_psnr,
    detect_peaks,
    smooth_signal,
)

__all__ = [
    # Spectral
    'SpectralScorer',
    'ZetaFiducials',
    'compute_spectral_scores',
    
    # Holographic
    'PhaseRetrieval',
    'phase_retrieve_hilbert',
    'phase_retrieve_gs',
    'align_phase',
    'holographic_refinement',
    
    # Optimization
    'SublinearOptimizer',
    'SRTCalibrator',
    'optimize_sublinear',
    
    # Utils
    'compute_envelope',
    'normalize_signal',
    'adaptive_blend',
    'compute_psnr',
    'detect_peaks',
    'smooth_signal',
]
