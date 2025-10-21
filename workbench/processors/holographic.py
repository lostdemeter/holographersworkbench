"""
Holographic Signal Processing Module
====================================

Unified phase retrieval, interference patterns, and holographic refinement.

This module consolidates holographic techniques from:
- sublinear_optimizer.py: Phase retrieval methods
- holo_lossless.py: 4-phase shifting holography
- zeta_chudnovsky.py: Holographic refinement
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhaseRetrievalConfig:
    """Configuration for phase retrieval."""
    method: str = "hilbert"  # 'hilbert' or 'gs'
    n_iter: int = 30
    tol: float = 1e-4
    blend_ratio: float = 0.6


class PhaseRetrieval:
    """
    Unified phase retrieval interface.
    
    Supports multiple methods:
    - Hilbert transform (fast, approximate)
    - Gerchberg-Saxton (slower, more accurate)
    """
    
    def __init__(self, method: str = "hilbert", **kwargs):
        """
        Initialize phase retrieval.
        
        Parameters
        ----------
        method : str
            'hilbert' or 'gs'
        **kwargs : dict
            Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
    
    def retrieve(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Retrieve phase/envelope from signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Real-valued signal.
        
        Returns
        -------
        envelope : np.ndarray
            Amplitude envelope.
        phase_variance : float
            Measure of phase stability.
        """
        if self.method == "hilbert":
            return phase_retrieve_hilbert(signal)
        elif self.method == "gs":
            # GS needs additional parameters
            intensity = self.kwargs.get('intensity', None)
            target_amp = self.kwargs.get('target_amp', None)
            if intensity is None or target_amp is None:
                raise ValueError("GS method requires 'intensity' and 'target_amp'")
            
            u_refined = phase_retrieve_gs(
                intensity, target_amp,
                n_iter=self.kwargs.get('n_iter', 30),
                tol=self.kwargs.get('tol', 1e-4)
            )
            
            # Extract envelope from refined field
            envelope = np.abs(u_refined)
            phase = np.angle(u_refined)
            phase_diff = np.diff(phase)
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            phase_variance = float(np.var(phase_diff))
            
            return envelope, phase_variance
        else:
            raise ValueError(f"Unknown method: {self.method}")


def phase_retrieve_hilbert(signal: np.ndarray) -> Tuple[np.ndarray, float]:
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


def phase_retrieve_gs(intensity_meas: np.ndarray,
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


def align_phase(object_signal: np.ndarray,
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


def holographic_refinement(scores: np.ndarray,
                           reference: np.ndarray,
                           method: str = "hilbert",
                           blend_ratio: float = 0.6,
                           phase_threshold: float = 0.12,
                           damping_factor: float = 0.85) -> np.ndarray:
    """
    Apply holographic refinement to scores using phase retrieval.
    
    This is the core technique for extracting signal from noise.
    
    Parameters
    ----------
    scores : np.ndarray
        Initial scores (object signal).
    reference : np.ndarray
        Reference baseline signal.
    method : str
        Phase retrieval method ('hilbert' or 'gs').
    blend_ratio : float
        Weight for object vs reference (0-1).
    phase_threshold : float
        Phase variance threshold for stability gate.
    damping_factor : float
        Damping applied if phase variance exceeds threshold.
    
    Returns
    -------
    np.ndarray
        Refined scores.
    """
    obj = np.asarray(scores, dtype=float).ravel()
    ref = np.asarray(reference, dtype=float).ravel()
    n = min(len(obj), len(ref))
    
    if n == 0:
        return obj
    
    obj = obj[:n]
    ref = ref[:n]
    
    # Phase alignment
    _, obj_aligned = align_phase(obj, ref)
    
    if method == "gs":
        # Gerchberg-Saxton refinement
        intensity_meas = np.abs(np.fft.fft(obj_aligned))
        target_amp = ref / (np.max(ref) + 1e-12)
        u_refined = phase_retrieve_gs(intensity_meas, target_amp, n_iter=30, tol=1e-4)
        refined_real = np.real(u_refined)
        
        # Extract envelope
        env, phase_var = phase_retrieve_hilbert(refined_real)
        env_norm = env / (np.max(env) + 1e-12)
        ref_norm = target_amp
        
        # Blend
        refined_scores = blend_ratio * refined_real * env_norm + (1 - blend_ratio) * ref_norm
        
        # Stability gate
        if phase_var > phase_threshold:
            refined_scores *= damping_factor
    
    else:
        # Hilbert envelope method (default, faster)
        env, phase_var = phase_retrieve_hilbert(obj_aligned)
        env_norm = env / (np.max(env) + 1e-12)
        ref_norm = ref / (np.max(ref) + 1e-12)
        
        # Blend
        refined_scores = blend_ratio * obj_aligned * env_norm + (1 - blend_ratio) * ref_norm
        
        # Stability gate
        if phase_var > phase_threshold:
            refined_scores *= damping_factor
    
    return refined_scores


class FourPhaseShifting:
    """
    4-phase shifting holography for lossless encoding.
    
    Unified from holo_lossless.py.
    """
    
    def __init__(self, kx: float = 0.3, ky: float = 0.3):
        """
        Initialize 4-phase shifting.
        
        Parameters
        ----------
        kx, ky : float
            Reference wave tilt (cycles/pixel).
        """
        self.kx = kx
        self.ky = ky
        self.phases = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image using 4-phase shifting.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (H, W) or (H, W, 3).
        
        Returns
        -------
        np.ndarray
            Stack of 4 phase-shifted holograms.
        """
        image = np.asarray(image, dtype=np.float32)
        
        if image.ndim == 2:
            height, width = image.shape
            is_color = False
        elif image.ndim == 3 and image.shape[2] == 3:
            height, width = image.shape[:2]
            is_color = True
        else:
            raise ValueError(f"Image must be (H,W) or (H,W,3), got {image.shape}")
        
        # Reference wave
        x = np.arange(width, dtype=float)
        y = np.arange(height, dtype=float)
        X, Y = np.meshgrid(x, y)
        R = np.exp(1j * 2 * np.pi * (self.kx * X + self.ky * Y))
        
        scale_factor = 4.0
        
        if not is_color:
            # Grayscale
            O = np.sqrt(np.clip(image, 0, 1)) * np.exp(1j * 0)
            frames = []
            for phi in self.phases:
                R_phi = R * np.exp(1j * phi)
                I = np.abs(O + R_phi) ** 2
                frames.append(np.clip(I / scale_factor, 0.0, 1.0))
            return np.stack(frames, axis=0)
        
        else:
            # RGB
            img_rgb = np.clip(image, 0, 1).astype(np.float32)
            stacks = []
            for c in range(3):
                Oc = np.sqrt(np.clip(img_rgb[:, :, c], 0, 1)) * np.exp(1j * 0)
                frames = []
                for phi in self.phases:
                    R_phi = R * np.exp(1j * phi)
                    I = np.abs(Oc + R_phi) ** 2
                    frames.append(np.clip(I / scale_factor, 0.0, 1.0))
                stacks.append(np.stack(frames, axis=0))
            return np.stack(stacks, axis=0)
    
    def decode(self, holograms: np.ndarray, scale: float = 4.0) -> np.ndarray:
        """
        Decode 4-phase shifted holograms.
        
        Parameters
        ----------
        holograms : np.ndarray
            Stack of holograms (4, H, W) or (3, 4, H, W).
        scale : float
            Scale factor used during encoding.
        
        Returns
        -------
        np.ndarray
            Reconstructed image.
        """
        I = holograms * scale
        
        if I.ndim == 3 and I.shape[0] == 4:
            # Grayscale
            I0, I90, I180, I270 = I[0], I[1], I[2], I[3]
            ReC = (I0 - I180) / 4.0
            ImC = (I270 - I90) / 4.0
            C = ReC + 1j * ImC
            recon = np.abs(C) ** 2
        
        elif I.ndim == 4 and I.shape[0] == 3 and I.shape[1] == 4:
            # RGB
            chans = []
            for c in range(3):
                I0, I90, I180, I270 = I[c, 0], I[c, 1], I[c, 2], I[c, 3]
                ReC = (I0 - I180) / 4.0
                ImC = (I270 - I90) / 4.0
                C = ReC + 1j * ImC
                chan = np.abs(C) ** 2
                chans.append(chan)
            recon = np.stack(chans, axis=2)
        
        else:
            raise ValueError(f"Invalid hologram shape: {I.shape}")
        
        return np.clip(recon, 0, 1)
