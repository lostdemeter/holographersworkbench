"""
workbench.primitives
~~~~~~~~~~~~~~~~~~~~

Layer 1: Pure stateless utility functions for signal processing, frequency analysis,
and phase manipulation.

This layer contains no classes, only pure functions with no side effects.
"""

from .signal import (
    compute_envelope,
    normalize,
    adaptive_blend,
    psnr,
    detect_peaks,
    smooth,
    compute_correlation,
    sliding_window,
)

from .frequency import (
    compute_fft,
    compute_power_spectrum,
    compute_ifft,
)

from .phase import (
    align,
    retrieve_hilbert,
    retrieve_gs,
)

from .kernels import (
    exponential_decay_kernel,
    gaussian_kernel,
)

from .quantum_folding import QuantumFolder
from .chaos_seeding import ChaosSeeder, AdaptiveChaosSeeder

__all__ = [
    # Signal processing
    'compute_envelope',
    'normalize',
    'adaptive_blend',
    'psnr',
    'detect_peaks',
    'smooth',
    'compute_correlation',
    'sliding_window',
    
    # Frequency analysis
    'compute_fft',
    'compute_power_spectrum',
    'compute_ifft',
    
    # Phase manipulation
    'align',
    'retrieve_hilbert',
    'retrieve_gs',
    
    # Kernels
    'exponential_decay_kernel',
    'gaussian_kernel',
    
    # Optimization
    'QuantumFolder',
    'ChaosSeeder',
    'AdaptiveChaosSeeder',
]
