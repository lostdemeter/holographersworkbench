"""
Recursive Fractal Peeling Module
=================================

Implementation of the Recursive Fractal Peeling algorithm for lossless
data compression based on iterative pattern extraction and residual analysis.

Mathematical formulation based on:
- Resfrac invariant: ρ(x) = σ(r) / σ(x)
- Autoregressive prediction with least squares
- Recursive tree structure for compression
- Lossless reconstruction guarantee

Key Components:
- CompressionTree: Tree data structure (LEAF/NODE)
- resfrac_score: Compute structure measure
- extract_pattern: AR model fitting
- compress: Recursive fractal peeling (Φ)
- decompress: Reconstruction (Ψ)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PredictorModel:
    """
    Autoregressive predictor model.
    
    Attributes
    ----------
    coefficients : np.ndarray
        AR coefficients β ∈ ℝᵏ
    mean : float
        Normalization mean μ
    std : float
        Normalization std σ
    initial_values : np.ndarray
        Initial k values for reconstruction
    order : int
        Model order k
    """
    coefficients: np.ndarray
    mean: float
    std: float
    initial_values: np.ndarray
    order: int
    
    def size_bytes(self) -> int:
        """Estimate storage size in bytes."""
        # coefficients + mean + std + initial_values + order
        return (len(self.coefficients) * 8 + 8 + 8 + 
                len(self.initial_values) * 8 + 4)


@dataclass
class CompressionNode:
    """
    Internal node in compression tree.
    
    Attributes
    ----------
    model : PredictorModel
        Predictor model M
    residual_tree : Union[CompressionNode, CompressionLeaf]
        Subtree for residuals
    resfrac : float
        Original resfrac score ρ
    residual_resfrac : float
        Residual resfrac score ρᵣ
    improvement : float
        Improvement Δρ = ρ - ρᵣ
    depth : int
        Depth in tree
    """
    model: PredictorModel
    residual_tree: Union['CompressionNode', 'CompressionLeaf']
    resfrac: float
    residual_resfrac: float
    improvement: float
    depth: int = 0
    
    def size_bytes(self) -> int:
        """Compute total storage size."""
        return (self.model.size_bytes() + 
                self.residual_tree.size_bytes() + 
                5 * 8)  # 5 floats/ints


@dataclass
class CompressionLeaf:
    """
    Leaf node in compression tree (raw data).
    
    Attributes
    ----------
    data : np.ndarray
        Raw data x ∈ ℝⁿ
    resfrac : float
        Resfrac score ρ
    depth : int
        Depth in tree
    """
    data: np.ndarray
    resfrac: float
    depth: int = 0
    
    def size_bytes(self) -> int:
        """Compute storage size."""
        return len(self.data) * 8 + 8 + 4  # data + resfrac + depth


# Type alias for compression tree
CompressionTree = Union[CompressionNode, CompressionLeaf]


# ============================================================================
# Core Functions
# ============================================================================

def resfrac_score(data: np.ndarray, 
                  order: int = 3,
                  epsilon: float = 1e-12) -> float:
    """
    Compute resfrac invariant score.
    
    The resfrac score measures predictability:
    ρ(x) = σ(r) / σ(x)
    
    where r = x - P(x) is the residual from an AR(k) predictor.
    
    Parameters
    ----------
    data : np.ndarray
        Input data sequence x ∈ ℝⁿ
    order : int
        Autoregressive model order k
    epsilon : float
        Small value to avoid division by zero
    
    Returns
    -------
    float
        Resfrac score ρ ∈ [0, 1]
        - ρ ≈ 0: highly predictable (structure)
        - ρ ≈ 1: unpredictable (randomness)
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    
    if n < order + 1:
        # Too short for AR model
        return 1.0
    
    # Compute standard deviation of original data
    sigma_x = np.std(data)
    if sigma_x < epsilon:
        # Constant signal
        return 0.0
    
    # Fit AR model and compute residuals
    try:
        _, residuals = _fit_ar_model(data, order)
        sigma_r = np.std(residuals)
        
        # Resfrac score
        rho = sigma_r / (sigma_x + epsilon)
        return float(np.clip(rho, 0.0, 1.0))
    
    except (np.linalg.LinAlgError, ValueError):
        # Model fitting failed
        return 1.0


def extract_pattern(data: np.ndarray,
                   order: int = 3) -> Tuple[PredictorModel, np.ndarray]:
    """
    Extract fractal pattern using autoregressive model.
    
    Fits AR(k) model: x̂ᵢ = β₀·xᵢ₋₁ + β₁·xᵢ₋₂ + ... + βₖ₋₁·xᵢ₋ₖ
    
    Parameters
    ----------
    data : np.ndarray
        Input data x ∈ ℝⁿ
    order : int
        Model order k
    
    Returns
    -------
    model : PredictorModel
        Fitted predictor model M
    residuals : np.ndarray
        Residuals r = x - P(x)
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    
    if n < order + 1:
        raise ValueError(f"Data length {n} too short for order {order}")
    
    # Normalize data
    mean = np.mean(data)
    std = np.std(data)
    epsilon = 1e-12
    
    if std < epsilon:
        # Constant signal - return trivial model
        model = PredictorModel(
            coefficients=np.zeros(order),
            mean=mean,
            std=epsilon,
            initial_values=data[:order].copy(),
            order=order
        )
        residuals = np.zeros(n - order)
        return model, residuals
    
    data_norm = (data - mean) / std
    
    # Fit AR model
    coefficients, residuals_norm = _fit_ar_model(data_norm, order)
    
    # Denormalize residuals
    residuals = residuals_norm * std
    
    # Create model
    model = PredictorModel(
        coefficients=coefficients,
        mean=mean,
        std=std,
        initial_values=data[:order].copy(),
        order=order
    )
    
    return model, residuals


def _fit_ar_model(data_norm: np.ndarray, 
                  order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit autoregressive model using least squares.
    
    Constructs design matrix X and solves: β* = argmin_β ‖y - Xβ‖²
    
    Parameters
    ----------
    data_norm : np.ndarray
        Normalized data
    order : int
        Model order k
    
    Returns
    -------
    coefficients : np.ndarray
        AR coefficients β ∈ ℝᵏ
    residuals : np.ndarray
        Normalized residuals
    """
    n = len(data_norm)
    
    # Construct design matrix X ∈ ℝ^((n-k) × k)
    # X[i, :] = [x_{i+k-1}, x_{i+k-2}, ..., x_i]
    X = np.zeros((n - order, order))
    for i in range(n - order):
        for j in range(order):
            X[i, j] = data_norm[i + order - 1 - j]
    
    # Target vector y = [x_k, x_{k+1}, ..., x_{n-1}]
    y = data_norm[order:]
    
    # Solve least squares: β* = (X^T X)^{-1} X^T y
    try:
        coefficients, residuals_vec, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        coefficients = np.linalg.pinv(X) @ y
    
    # Compute residuals
    predictions = X @ coefficients
    residuals = y - predictions
    
    return coefficients, residuals


def compress(data: np.ndarray,
            order: int = 3,
            noise_threshold: float = 0.95,
            improvement_threshold: float = 0.01,
            max_depth: int = 10,
            current_depth: int = 0) -> CompressionTree:
    """
    Recursive fractal peeling compression algorithm (Φ).
    
    Recursively decomposes data into predictor models and residuals,
    building a compression tree. Guarantees lossless reconstruction.
    
    Algorithm:
    1. Compute ρ(x)
    2. If ρ > θ_noise or d ≥ d_max: return LEAF
    3. Extract pattern (M, r)
    4. Compute ρ(r) and Δρ
    5. If Δρ < ε: return LEAF (no improvement)
    6. Recurse on residuals: T_r = Φ(r, d+1)
    7. Return NODE(M, T_r, ρ, ρ_r, Δρ)
    
    Parameters
    ----------
    data : np.ndarray
        Input data x ∈ ℝⁿ
    order : int
        AR model order k
    noise_threshold : float
        Entropy barrier θ_noise (typically 0.95)
    improvement_threshold : float
        Minimum improvement ε (typically 0.01)
    max_depth : int
        Maximum recursion depth d_max
    current_depth : int
        Current depth (internal)
    
    Returns
    -------
    CompressionTree
        Compression tree T ∈ 𝒯
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    
    # Step 1: Compute resfrac score
    rho = resfrac_score(data, order=order)
    
    # Step 2: Check termination conditions
    if rho > noise_threshold:
        # Entropy barrier - data is too random
        return CompressionLeaf(data=data, resfrac=rho, depth=current_depth)
    
    if current_depth >= max_depth:
        # Depth limit reached
        return CompressionLeaf(data=data, resfrac=rho, depth=current_depth)
    
    if n < order + 1:
        # Too short for AR model
        return CompressionLeaf(data=data, resfrac=rho, depth=current_depth)
    
    # Step 3: Extract pattern
    try:
        model, residuals = extract_pattern(data, order=order)
    except (ValueError, np.linalg.LinAlgError):
        # Pattern extraction failed
        return CompressionLeaf(data=data, resfrac=rho, depth=current_depth)
    
    # Step 4: Evaluate improvement
    rho_r = resfrac_score(residuals, order=order)
    delta_rho = rho - rho_r
    
    if delta_rho < improvement_threshold:
        # No significant improvement
        return CompressionLeaf(data=data, resfrac=rho, depth=current_depth)
    
    # Step 5: Recurse on residuals
    residual_tree = compress(
        residuals,
        order=order,
        noise_threshold=noise_threshold,
        improvement_threshold=improvement_threshold,
        max_depth=max_depth,
        current_depth=current_depth + 1
    )
    
    # Step 6: Return internal node
    return CompressionNode(
        model=model,
        residual_tree=residual_tree,
        resfrac=rho,
        residual_resfrac=rho_r,
        improvement=delta_rho,
        depth=current_depth
    )


def decompress(tree: CompressionTree) -> np.ndarray:
    """
    Reconstruct data from compression tree (Ψ).
    
    Recursively reconstructs data from predictor models and residuals.
    Guarantees: Ψ(Φ(x)) = x (lossless).
    
    Algorithm:
    - LEAF: return raw data
    - NODE: 
        1. Recursively reconstruct residuals: r = Ψ(T_r)
        2. Apply AR model to predict: x̂ = M(x₀:ₖ)
        3. Combine: x = [x₀:ₖ, x̂ₖ:ₙ + r]
    
    Parameters
    ----------
    tree : CompressionTree
        Compression tree T ∈ 𝒯
    
    Returns
    -------
    np.ndarray
        Reconstructed data x ∈ ℝⁿ
    """
    if isinstance(tree, CompressionLeaf):
        # Case 1: Leaf node - return raw data
        return tree.data.copy()
    
    elif isinstance(tree, CompressionNode):
        # Case 2: Internal node - reconstruct from model + residuals
        model = tree.model
        
        # Recursively reconstruct residuals
        residuals = decompress(tree.residual_tree)
        
        # Reconstruct predictions
        predictions = _apply_ar_model(model, len(residuals))
        
        # Combine: initial values + (predictions + residuals)
        reconstructed = np.concatenate([
            model.initial_values,
            predictions + residuals
        ])
        
        return reconstructed
    
    else:
        raise TypeError(f"Unknown tree type: {type(tree)}")


def _apply_ar_model(model: PredictorModel, n_predict: int) -> np.ndarray:
    """
    Apply AR model to generate predictions.
    
    Parameters
    ----------
    model : PredictorModel
        Fitted AR model
    n_predict : int
        Number of predictions to generate
    
    Returns
    -------
    np.ndarray
        Predictions (denormalized)
    """
    order = model.order
    coefficients = model.coefficients
    mean = model.mean
    std = model.std
    
    # Initialize with normalized initial values
    data_norm = (model.initial_values - mean) / std
    data_norm = list(data_norm)
    
    # Generate predictions
    predictions_norm = []
    for i in range(n_predict):
        # Predict next value: x̂ᵢ = Σⱼ βⱼ·xᵢ₋ⱼ₋₁
        pred = 0.0
        for j in range(order):
            if i + order - j - 1 < len(data_norm):
                pred += coefficients[j] * data_norm[-(j+1)]
        
        predictions_norm.append(pred)
        data_norm.append(pred)
    
    # Denormalize
    predictions = np.array(predictions_norm) * std + mean
    
    return predictions


# ============================================================================
# Utilities and Metrics
# ============================================================================

def compression_ratio(tree: CompressionTree, original_size: int) -> float:
    """
    Compute compression ratio.
    
    R(x) = n / S(T)
    
    Parameters
    ----------
    tree : CompressionTree
        Compression tree
    original_size : int
        Original data length n
    
    Returns
    -------
    float
        Compression ratio (>1 means compression achieved)
    """
    compressed_size = tree.size_bytes() / 8  # Convert bytes to float64 elements
    return original_size / compressed_size


def tree_statistics(tree: CompressionTree) -> Dict[str, Any]:
    """
    Compute statistics about compression tree.
    
    Parameters
    ----------
    tree : CompressionTree
        Compression tree
    
    Returns
    -------
    dict
        Statistics including depth, nodes, leaves, etc.
    """
    stats = {
        'max_depth': 0,
        'num_nodes': 0,
        'num_leaves': 0,
        'total_size_bytes': tree.size_bytes(),
        'resfrac_scores': [],
        'improvements': []
    }
    
    def traverse(node: CompressionTree, depth: int = 0):
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        if isinstance(node, CompressionLeaf):
            stats['num_leaves'] += 1
            stats['resfrac_scores'].append(node.resfrac)
        
        elif isinstance(node, CompressionNode):
            stats['num_nodes'] += 1
            stats['resfrac_scores'].append(node.resfrac)
            stats['improvements'].append(node.improvement)
            traverse(node.residual_tree, depth + 1)
    
    traverse(tree)
    
    # Compute summary statistics
    if stats['resfrac_scores']:
        stats['mean_resfrac'] = float(np.mean(stats['resfrac_scores']))
        stats['min_resfrac'] = float(np.min(stats['resfrac_scores']))
        stats['max_resfrac'] = float(np.max(stats['resfrac_scores']))
    
    if stats['improvements']:
        stats['mean_improvement'] = float(np.mean(stats['improvements']))
        stats['total_improvement'] = float(np.sum(stats['improvements']))
    
    return stats


def visualize_tree(tree: CompressionTree, indent: int = 0) -> str:
    """
    Create text visualization of compression tree.
    
    Parameters
    ----------
    tree : CompressionTree
        Compression tree
    indent : int
        Indentation level
    
    Returns
    -------
    str
        Tree visualization
    """
    prefix = "  " * indent
    
    if isinstance(tree, CompressionLeaf):
        return f"{prefix}LEAF(n={len(tree.data)}, ρ={tree.resfrac:.4f})\n"
    
    elif isinstance(tree, CompressionNode):
        result = f"{prefix}NODE(ρ={tree.resfrac:.4f}, ρᵣ={tree.residual_resfrac:.4f}, Δρ={tree.improvement:.4f})\n"
        result += f"{prefix}  Model: order={tree.model.order}, size={tree.model.size_bytes()}B\n"
        result += visualize_tree(tree.residual_tree, indent + 1)
        return result
    
    return ""


class FractalPeeler:
    """
    High-level interface for recursive fractal peeling.
    
    Parameters
    ----------
    order : int
        Autoregressive model order k (default: 3)
    noise_threshold : float
        Entropy barrier θ_noise (default: 0.95)
    improvement_threshold : float
        Minimum improvement ε (default: 0.01)
    max_depth : int
        Maximum recursion depth (default: 10)
    
    Examples
    --------
    >>> peeler = FractalPeeler(order=3)
    >>> tree = peeler.compress(data)
    >>> reconstructed = peeler.decompress(tree)
    >>> ratio = peeler.compression_ratio(tree, len(data))
    """
    
    def __init__(self,
                 order: int = 3,
                 noise_threshold: float = 0.95,
                 improvement_threshold: float = 0.01,
                 max_depth: int = 10):
        self.order = order
        self.noise_threshold = noise_threshold
        self.improvement_threshold = improvement_threshold
        self.max_depth = max_depth
    
    def compress(self, data: np.ndarray) -> CompressionTree:
        """Compress data into tree structure."""
        return compress(
            data,
            order=self.order,
            noise_threshold=self.noise_threshold,
            improvement_threshold=self.improvement_threshold,
            max_depth=self.max_depth
        )
    
    def decompress(self, tree: CompressionTree) -> np.ndarray:
        """Reconstruct data from tree."""
        return decompress(tree)
    
    def compress_decompress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compress and decompress, returning reconstructed data and statistics.
        
        Returns
        -------
        reconstructed : np.ndarray
            Reconstructed data
        stats : dict
            Compression statistics
        """
        tree = self.compress(data)
        reconstructed = self.decompress(tree)
        
        stats = tree_statistics(tree)
        stats['compression_ratio'] = compression_ratio(tree, len(data))
        stats['reconstruction_error'] = float(np.max(np.abs(data - reconstructed)))
        
        return reconstructed, stats
    
    def compression_ratio(self, tree: CompressionTree, original_size: int) -> float:
        """Compute compression ratio."""
        return compression_ratio(tree, original_size)
    
    def tree_stats(self, tree: CompressionTree) -> Dict[str, Any]:
        """Get tree statistics."""
        return tree_statistics(tree)
    
    def visualize(self, tree: CompressionTree) -> str:
        """Visualize tree structure."""
        return visualize_tree(tree)


# ============================================================================
# Holographic Compression (merged from holographic_compression.py)
# ============================================================================

import numpy as np
from typing import Tuple, Dict, Any
import struct
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics from compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    harmonic_order: int
    phase_symmetry_score: float
    residual_entropy: float


class HolographicCompressor:
    """
    Lossless holographic image compressor exploiting 15th order harmonics.

    This compressor:
    1. Transforms image to frequency domain
    2. Extracts and encodes dominant 15th order harmonics
    3. Exploits phase symmetry for compression
    4. Encodes residuals for lossless reconstruction

    Attributes:
        harmonic_order (int): Primary harmonic order to exploit (default: 15)
        c (float): Universal reference constant (speed of light)
        phase_quantization_bits (int): Bits for phase quantization
    """

    def __init__(self, harmonic_order: int = 15, phase_quantization_bits: int = 8):
        """
        Initialize holographic compressor.

        Args:
            harmonic_order: Primary harmonic order to exploit
            phase_quantization_bits: Bits for phase quantization (higher = better quality)
        """
        self.harmonic_order = harmonic_order
        self.c = 299792458.0  # Speed of light in m/s (universal reference)
        self.phase_quantization_bits = phase_quantization_bits
        self.phase_levels = 2 ** phase_quantization_bits

    def _extract_harmonic_ring(self, fft_data: np.ndarray, order: int) -> np.ndarray:
        """
        Extract a specific harmonic order ring from FFT data.

        Args:
            fft_data: 2D FFT of image
            order: Harmonic order to extract

        Returns:
            Complex values at the harmonic ring
        """
        h, w = fft_data.shape
        cy, cx = h // 2, w // 2

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Calculate distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Extract ring (with tolerance for discrete grid)
        mask = (dist >= order - 0.5) & (dist < order + 0.5)

        return fft_data[mask]

    def _reconstruct_from_ring(self, ring_values: np.ndarray, 
                                ring_positions: np.ndarray, 
                                shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct FFT from harmonic ring values.

        Args:
            ring_values: Complex values at ring positions
            ring_positions: Boolean mask of ring positions
            shape: Output shape

        Returns:
            Reconstructed FFT array
        """
        fft_recon = np.zeros(shape, dtype=np.complex128)
        fft_recon[ring_positions] = ring_values
        return fft_recon

    def _quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Quantize phase values to reduce entropy.

        Args:
            phase: Phase array in radians [-π, π]

        Returns:
            Quantized phase indices
        """
        # Normalize to [0, 1]
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Quantize
        phase_quantized = np.round(phase_norm * (self.phase_levels - 1)).astype(np.uint16)

        return phase_quantized

    def _dequantize_phase(self, phase_indices: np.ndarray) -> np.ndarray:
        """
        Dequantize phase indices back to radians.

        Args:
            phase_indices: Quantized phase indices

        Returns:
            Phase in radians
        """
        phase_norm = phase_indices / (self.phase_levels - 1)
        phase = phase_norm * 2 * np.pi - np.pi
        return phase

    def _encode_residuals(self, residuals: np.ndarray) -> bytes:
        """
        Encode residuals for lossless reconstruction.

        Args:
            residuals: Residual array (int16)

        Returns:
            Compressed residual bytes
        """
        import zlib
        # Residuals are already int16
        residual_bytes = residuals.astype(np.int16).tobytes()
        compressed = zlib.compress(residual_bytes, level=9)
        return compressed

    def _decode_residuals(self, compressed_bytes: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode residuals from compressed bytes.

        Args:
            compressed_bytes: Compressed residual data
            shape: Output shape

        Returns:
            Residual array (int16)
        """
        import zlib
        decompressed = zlib.decompress(compressed_bytes)
        residuals = np.frombuffer(decompressed, dtype=np.int16).reshape(shape)
        return residuals

    def compress(self, image: np.ndarray) -> Tuple[bytes, CompressionStats]:
        """
        Compress image using holographic encoding.

        Args:
            image: Input image (grayscale, values 0-255)

        Returns:
            Tuple of (compressed_bytes, compression_stats)
        """
        # Normalize image
        img_norm = image.astype(np.float64) / 255.0
        h, w = img_norm.shape

        # FFT
        fft_data = np.fft.fftshift(np.fft.fft2(img_norm))

        # Extract 15th order harmonics
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ring_mask = (dist >= self.harmonic_order - 0.5) & (dist < self.harmonic_order + 0.5)

        ring_values = fft_data[ring_mask]

        # Separate magnitude and phase
        magnitude = np.abs(ring_values)
        phase = np.angle(ring_values)

        # Quantize phase (exploit symmetry)
        phase_quantized = self._quantize_phase(phase)

        # Calculate phase symmetry score
        phase_hist = np.bincount(phase_quantized, minlength=self.phase_levels)
        phase_symmetry = np.max(phase_hist) / len(phase_quantized) if len(phase_quantized) > 0 else 0

        # Reconstruct from quantized harmonics
        phase_dequant = self._dequantize_phase(phase_quantized)
        ring_recon = magnitude * np.exp(1j * phase_dequant)

        fft_recon = np.zeros_like(fft_data)
        fft_recon[ring_mask] = ring_recon

        # Inverse FFT to get prediction
        img_pred = np.real(np.fft.ifft2(np.fft.ifftshift(fft_recon)))

        # Compute residuals (store difference in original uint8 space for lossless)
        img_pred_uint8 = np.clip(img_pred * 255.0, 0, 255).astype(np.uint8)
        residuals_uint8 = image.astype(np.int16) - img_pred_uint8.astype(np.int16)

        # Calculate residual entropy
        residual_entropy = float(np.std(residuals_uint8))

        # Encode components
        # Header: shape, harmonic_order, phase_bits
        header = struct.pack('IIII', h, w, self.harmonic_order, self.phase_quantization_bits)

        # Magnitude (float32)
        magnitude_bytes = magnitude.astype(np.float32).tobytes()

        # Phase (quantized)
        phase_bytes = phase_quantized.tobytes()

        # Residuals (compressed)
        residual_bytes = self._encode_residuals(residuals_uint8)

        # Combine with length prefixes
        compressed = header
        compressed += struct.pack('I', len(magnitude_bytes)) + magnitude_bytes
        compressed += struct.pack('I', len(phase_bytes)) + phase_bytes
        compressed += struct.pack('I', len(residual_bytes)) + residual_bytes

        # Calculate stats
        original_size = image.size
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            harmonic_order=self.harmonic_order,
            phase_symmetry_score=phase_symmetry,
            residual_entropy=residual_entropy
        )

        return compressed, stats

    def decompress(self, compressed_bytes: bytes) -> np.ndarray:
        """
        Decompress holographically encoded image.

        Args:
            compressed_bytes: Compressed image data

        Returns:
            Reconstructed image (grayscale, values 0-255)
        """
        # Parse header
        offset = 0
        h, w, harmonic_order, phase_bits = struct.unpack('IIII', compressed_bytes[offset:offset+16])
        offset += 16

        # Parse magnitude
        mag_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        magnitude = np.frombuffer(compressed_bytes[offset:offset+mag_len], dtype=np.float32)
        offset += mag_len

        # Parse phase
        phase_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        phase_quantized = np.frombuffer(compressed_bytes[offset:offset+phase_len], dtype=np.uint16)
        offset += phase_len

        # Parse residuals
        residual_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        residual_bytes = compressed_bytes[offset:offset+residual_len]
        residuals = self._decode_residuals(residual_bytes, (h, w))

        # Reconstruct FFT
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ring_mask = (dist >= harmonic_order - 0.5) & (dist < harmonic_order + 0.5)

        # Dequantize phase
        phase_levels = 2 ** phase_bits
        phase = self._dequantize_phase(phase_quantized)

        # Reconstruct ring
        ring_recon = magnitude * np.exp(1j * phase)

        fft_recon = np.zeros((h, w), dtype=np.complex128)
        fft_recon[ring_mask] = ring_recon

        # Inverse FFT
        img_pred = np.real(np.fft.ifft2(np.fft.ifftshift(fft_recon)))
        img_pred_uint8 = np.clip(img_pred * 255.0, 0, 255).astype(np.uint8)

        # Add residuals (int16 + uint8 = int16, then clip to uint8)
        img_recon = np.clip(img_pred_uint8.astype(np.int16) + residuals, 0, 255).astype(np.uint8)

        return img_recon


def compress_image(image: np.ndarray, harmonic_order: int = 15, 
                   phase_bits: int = 8) -> Tuple[bytes, CompressionStats]:
    """
    Convenience function to compress an image.

    Args:
        image: Input grayscale image (0-255)
        harmonic_order: Harmonic order to exploit (default: 15)
        phase_bits: Phase quantization bits (default: 8)

    Returns:
        Tuple of (compressed_bytes, compression_stats)
    """
    compressor = HolographicCompressor(harmonic_order, phase_bits)
    return compressor.compress(image)


def decompress_image(compressed_bytes: bytes) -> np.ndarray:
    """
    Convenience function to decompress an image.

    Args:
        compressed_bytes: Compressed image data

    Returns:
        Reconstructed grayscale image (0-255)
    """
    compressor = HolographicCompressor()
    return compressor.decompress(compressed_bytes)


# Example usage
if __name__ == "__main__":
    # Create a test image with 15th order structure
    size = 128
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Generate 15th order pattern
    theta = np.arctan2(y - cy, x - cx)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    test_image = (128 + 127 * np.sin(15 * theta) * np.exp(-r / 30)).astype(np.uint8)

    # Compress
    compressed, stats = compress_image(test_image)

    print(f"Holographic Compression Results:")
    print(f"  Original size: {stats.original_size} bytes")
    print(f"  Compressed size: {stats.compressed_size} bytes")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Harmonic order: {stats.harmonic_order}")
    print(f"  Phase symmetry: {stats.phase_symmetry_score:.3f}")
    print(f"  Residual entropy: {stats.residual_entropy:.2f}")

    # Decompress
    reconstructed = decompress_image(compressed)

    # Verify lossless
    is_lossless = np.array_equal(test_image, reconstructed)
    print(f"  Lossless: {is_lossless}")

    if not is_lossless:
        max_error = np.max(np.abs(test_image.astype(int) - reconstructed.astype(int)))
        print(f"  Max error: {max_error}")
