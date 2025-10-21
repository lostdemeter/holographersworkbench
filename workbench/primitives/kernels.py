"""
workbench.primitives.kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure kernel functions for signal processing and weighting.

Key Functions:
    - exponential_decay_kernel: Exponential decay weighting
    - gaussian_kernel: Gaussian weighting

Example:
    >>> from workbench.primitives import kernels
    >>> distances = np.arange(100)
    >>> weights = kernels.gaussian_kernel(distances, sigma=10.0)

Dependencies:
    - numpy
"""

import numpy as np


def exponential_decay_kernel(distance: np.ndarray,
                             decay_rate: float = 10.0) -> np.ndarray:
    """
    Compute exponential decay kernel.
    
    Parameters
    ----------
    distance : np.ndarray
        Distance values.
    decay_rate : float
        Decay rate parameter.
    
    Returns
    -------
    np.ndarray
        Kernel values.
    """
    return np.exp(-np.abs(distance) / decay_rate)


def gaussian_kernel(distance: np.ndarray,
                   sigma: float = 1.0) -> np.ndarray:
    """
    Compute Gaussian kernel.
    
    Parameters
    ----------
    distance : np.ndarray
        Distance values.
    sigma : float
        Standard deviation.
    
    Returns
    -------
    np.ndarray
        Kernel values.
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)
