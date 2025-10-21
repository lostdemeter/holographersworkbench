"""
workbench.primitives.phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure phase manipulation functions: phase retrieval, alignment, and unwrapping.

Key Functions:
    - retrieve_hilbert: Extract phase using Hilbert transform
    - retrieve_gs: Gerchberg-Saxton phase retrieval
    - align: Align phase between two signals

Example:
    >>> from workbench.primitives import phase
    >>> signal = np.random.randn(1000)
    >>> envelope, phase_var = phase.retrieve_hilbert(signal)
    >>> theta, aligned = phase.align(signal1, signal2)

Dependencies:
    - numpy
"""

import numpy as np
from typing import Tuple


def retrieve_hilbert(signal: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract envelope and phase variance using Hilbert transform.
    
    Parameters
    ----------
    signal : np.ndarray
        Real-valued signal.
    
    Returns
    -------
    envelope : np.ndarray
        Amplitude envelope of the signal.
    phase_variance : float
        Variance of the instantaneous phase.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = len(signal)
    
    if n == 0:
        return np.array([]), 0.0
    
    # Compute analytic signal via FFT
    fft_sig = np.fft.fft(signal)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    
    analytic = np.fft.ifft(fft_sig * h)
    envelope = np.abs(analytic)
    
    # Phase variance (stability metric)
    phase = np.angle(analytic)
    phase_diff = np.diff(phase)
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_variance = float(np.var(phase_diff))
    
    return envelope, phase_variance


def retrieve_gs(intensity_meas: np.ndarray,
                target_amp: np.ndarray,
                n_iter: int = 30,
                tol: float = 1e-4) -> np.ndarray:
    """
    Gerchberg-Saxton algorithm for 1D phase retrieval.
    
    Parameters
    ----------
    intensity_meas : np.ndarray
        Measured Fourier-domain magnitudes |F{u}|.
    target_amp : np.ndarray
        Desired time-domain amplitude |u|.
    n_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    np.ndarray
        Complex refined field u.
    """
    intensity_meas = np.asarray(intensity_meas, dtype=float).ravel()
    target_amp = np.asarray(target_amp, dtype=float).ravel()
    n = min(intensity_meas.size, target_amp.size)
    
    if n == 0:
        return target_amp.astype(complex)
    
    I = intensity_meas[:n]
    A = target_amp[:n]
    
    # Initialize with random phase
    rng = np.random.default_rng(0)
    u = A * np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    
    for _ in range(max(1, int(n_iter))):
        # Fourier domain: enforce magnitude
        U = np.fft.fft(u)
        Up = I * np.exp(1j * np.angle(U))
        
        # Time domain: enforce amplitude
        up = np.fft.ifft(Up)
        u_new = A * np.exp(1j * np.angle(up))
        
        # Check convergence
        if np.mean((np.abs(np.fft.fft(u_new)) - I) ** 2) < float(tol):
            u = u_new
            break
        u = u_new
    
    return u


def align(object_signal: np.ndarray,
          reference_signal: np.ndarray,
          n_angles: int = 360) -> Tuple[float, np.ndarray]:
    """
    Align phase/polarization between object and reference signals.
    
    Parameters
    ----------
    object_signal : np.ndarray
        Object signal to align.
    reference_signal : np.ndarray
        Reference signal (baseline).
    n_angles : int
        Number of angles to try.
    
    Returns
    -------
    theta : float
        Optimal rotation angle.
    aligned : np.ndarray
        Phase-aligned object signal.
    """
    obj = np.asarray(object_signal, dtype=float).ravel()
    ref = np.asarray(reference_signal, dtype=float).ravel()
    n = min(len(obj), len(ref))
    
    if n == 0:
        return 0.0, obj
    
    obj = obj[:n]
    ref = ref[:n]
    
    # Find optimal rotation via correlation
    angles = np.linspace(0, 2 * np.pi, n_angles)
    correlations = []
    
    for theta in angles:
        rotated = obj * np.cos(theta) + np.roll(obj, 1) * np.sin(theta)
        corr = np.corrcoef(rotated, ref)[0, 1]
        correlations.append(corr if np.isfinite(corr) else -1)
    
    best_idx = np.argmax(correlations)
    best_theta = angles[best_idx]
    
    # Apply best rotation
    aligned = obj * np.cos(best_theta) + np.roll(obj, 1) * np.sin(best_theta)
    
    return float(best_theta), aligned
