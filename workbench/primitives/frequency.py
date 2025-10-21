"""
workbench.primitives.frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure frequency-domain functions: FFT, power spectrum, and frequency analysis.

Key Functions:
    - compute_fft: Compute Fast Fourier Transform
    - compute_ifft: Compute inverse FFT
    - compute_power_spectrum: Compute power spectral density

Example:
    >>> from workbench.primitives import frequency
    >>> signal = np.random.randn(1000)
    >>> fft_result = frequency.compute_fft(signal)
    >>> power = frequency.compute_power_spectrum(signal)

Dependencies:
    - numpy
"""

import numpy as np
from typing import Tuple


def compute_fft(signal: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute Fast Fourier Transform of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (real or complex).
    normalize : bool
        If True, normalize by signal length.
    
    Returns
    -------
    np.ndarray
        Complex FFT coefficients.
    """
    signal = np.asarray(signal)
    fft_result = np.fft.fft(signal)
    
    if normalize:
        fft_result = fft_result / len(signal)
    
    return fft_result


def compute_ifft(fft_coeffs: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute inverse Fast Fourier Transform.
    
    Parameters
    ----------
    fft_coeffs : np.ndarray
        FFT coefficients (complex).
    normalize : bool
        If True, normalize by length.
    
    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    fft_coeffs = np.asarray(fft_coeffs)
    signal = np.fft.ifft(fft_coeffs)
    
    if normalize:
        signal = signal * len(fft_coeffs)
    
    return signal


def compute_power_spectrum(signal: np.ndarray, 
                           normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    normalize : bool
        If True, normalize power spectrum.
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency bins.
    power : np.ndarray
        Power spectral density.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = len(signal)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    
    # Compute power (magnitude squared)
    power = np.abs(fft_result) ** 2
    
    # Normalize if requested
    if normalize:
        power = power / n
    
    # Frequency bins
    frequencies = np.fft.fftfreq(n)
    
    return frequencies, power


def compute_frequency_bins(n_samples: int, 
                          sample_rate: float = 1.0) -> np.ndarray:
    """
    Compute frequency bins for FFT.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    sample_rate : float
        Sampling rate (Hz).
    
    Returns
    -------
    np.ndarray
        Frequency bins in Hz.
    """
    return np.fft.fftfreq(n_samples, d=1.0/sample_rate)
