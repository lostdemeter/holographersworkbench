"""
workbench.primitives.signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure signal processing functions: normalization, envelope detection, smoothing,
peak detection, and quality metrics.

Key Functions:
    - normalize: Normalize signal to [0,1] or standardize
    - compute_envelope: Extract signal envelope
    - smooth: Smooth signal using various methods
    - detect_peaks: Find local maxima in signal
    - psnr: Compute peak signal-to-noise ratio
    - adaptive_blend: Blend two signals based on quality metric

Example:
    >>> from workbench.primitives import signal
    >>> data = np.random.randn(1000)
    >>> normalized = signal.normalize(data, method='minmax')
    >>> smoothed = signal.smooth(normalized, window_size=5)

Dependencies:
    - numpy, scipy
"""

import numpy as np
from typing import Optional


def compute_envelope(signal: np.ndarray, method: str = "hilbert") -> np.ndarray:
    """
    Compute envelope of a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    method : str
        'hilbert' or 'abs'
    
    Returns
    -------
    np.ndarray
        Signal envelope.
    """
    if method == "hilbert":
        # Use Hilbert transform via FFT
        signal = np.asarray(signal, dtype=float).ravel()
        n = len(signal)
        
        if n == 0:
            return np.array([])
        
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
        return envelope
        
    elif method == "abs":
        return np.abs(signal)
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize(signal: np.ndarray, 
              method: str = "minmax",
              epsilon: float = 1e-12) -> np.ndarray:
    """
    Normalize signal to [0, 1] or standardize.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    method : str
        'minmax', 'zscore', or 'max'
    epsilon : float
        Small value to avoid division by zero.
    
    Returns
    -------
    np.ndarray
        Normalized signal.
    """
    signal = np.asarray(signal, dtype=float)
    
    if method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + epsilon)
    
    elif method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + epsilon)
    
    elif method == "max":
        max_val = np.max(np.abs(signal))
        return signal / (max_val + epsilon)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def adaptive_blend(object_signal: np.ndarray,
                  reference_signal: np.ndarray,
                  phase_variance: float,
                  base_ratio: float = 0.6) -> np.ndarray:
    """
    Adaptively blend object and reference based on phase variance.
    
    Parameters
    ----------
    object_signal : np.ndarray
        Object signal.
    reference_signal : np.ndarray
        Reference signal.
    phase_variance : float
        Phase variance (measure of signal quality).
    base_ratio : float
        Base blending ratio for object.
    
    Returns
    -------
    np.ndarray
        Blended signal.
    """
    # Adjust blend ratio based on phase variance
    if phase_variance < 0.05:
        # High quality: trust object more
        ratio = min(base_ratio + 0.2, 0.9)
    elif phase_variance < 0.12:
        # Medium quality: use base ratio
        ratio = base_ratio
    else:
        # Low quality: trust reference more
        ratio = max(base_ratio - 0.2, 0.2)
    
    # Normalize signals
    obj_norm = normalize(object_signal, method="max")
    ref_norm = normalize(reference_signal, method="max")
    
    # Blend
    blended = ratio * obj_norm + (1 - ratio) * ref_norm
    
    return blended


def compute_correlation(signal1: np.ndarray,
                       signal2: np.ndarray,
                       method: str = "pearson") -> float:
    """
    Compute correlation between two signals.
    
    Parameters
    ----------
    signal1, signal2 : np.ndarray
        Input signals.
    method : str
        'pearson', 'spearman', or 'cosine'
    
    Returns
    -------
    float
        Correlation coefficient.
    """
    s1 = np.asarray(signal1, dtype=float).ravel()
    s2 = np.asarray(signal2, dtype=float).ravel()
    n = min(len(s1), len(s2))
    
    if n == 0:
        return 0.0
    
    s1 = s1[:n]
    s2 = s2[:n]
    
    if method == "pearson":
        corr = np.corrcoef(s1, s2)[0, 1]
        return corr if np.isfinite(corr) else 0.0
    
    elif method == "cosine":
        dot = np.dot(s1, s2)
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)
        return dot / (norm1 * norm2 + 1e-12)
    
    elif method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(s1, s2)
        return corr if np.isfinite(corr) else 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")


def psnr(original: np.ndarray,
         reconstructed: np.ndarray,
         max_value: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Parameters
    ----------
    original : np.ndarray
        Original signal.
    reconstructed : np.ndarray
        Reconstructed signal.
    max_value : float
        Maximum possible value.
    
    Returns
    -------
    float
        PSNR in dB (inf if perfect match).
    """
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    
    if mse <= 1e-12:
        return float('inf')
    
    return 20 * np.log10(max_value / np.sqrt(mse))


def sliding_window(signal: np.ndarray,
                  window_size: int,
                  stride: int = 1) -> np.ndarray:
    """
    Create sliding windows over signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Size of each window.
    stride : int
        Stride between windows.
    
    Returns
    -------
    np.ndarray
        Array of windows, shape (n_windows, window_size).
    """
    signal = np.asarray(signal).ravel()
    n = len(signal)
    
    if window_size > n:
        return signal.reshape(1, -1)
    
    n_windows = (n - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start = i * stride
        windows[i] = signal[start:start + window_size]
    
    return windows


def detect_peaks(signal: np.ndarray,
                threshold: Optional[float] = None,
                min_distance: int = 1) -> np.ndarray:
    """
    Detect peaks in signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    threshold : float, optional
        Minimum peak height. If None, uses mean + std.
    min_distance : int
        Minimum distance between peaks.
    
    Returns
    -------
    np.ndarray
        Indices of peaks.
    """
    signal = np.asarray(signal).ravel()
    
    if threshold is None:
        threshold = np.mean(signal) + np.std(signal)
    
    # Find local maxima
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(i)
    
    # Enforce minimum distance
    if min_distance > 1 and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
    
    return np.array(peaks)


def smooth(signal: np.ndarray,
           window_size: int = 5,
           method: str = "moving_average") -> np.ndarray:
    """
    Smooth signal using various methods.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Size of smoothing window.
    method : str
        'moving_average', 'gaussian', or 'median'
    
    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    
    if method == "moving_average":
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')
    
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 4.0
        return gaussian_filter1d(signal, sigma)
    
    elif method == "median":
        from scipy.ndimage import median_filter
        return median_filter(signal, size=window_size)
    
    else:
        raise ValueError(f"Unknown method: {method}")
