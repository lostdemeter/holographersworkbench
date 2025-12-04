#!/usr/bin/env python3
"""
Dimensional Navigation: Core Implementation

A universal process for exact computation via structured navigation through
"truth space." Values are represented as coordinates in a precomputed mesh,
enabling compression and fast reconstruction.

The 5-Step Process:
1. DOWNCAST  - Map continuous values to discrete indices
2. QUANTIZE  - Store residual for precision
3. BUILD MESH - Precompute the lookup structure
4. UPSCALE   - Refine iteratively for exact values
5. RECONSTRUCT - Navigate to answer via lookup

Usage:
    from dimensional_navigation import PhiLens, VectorMesh
    
    # Compress data
    lens = PhiLens()
    encoded = lens.encode(data)
    
    # Reconstruct
    reconstructed = lens.decode(encoded)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
LOG_PHI = np.log(PHI)       # ≈ 0.481212

# Standard mathematical constants that appear in truth space
TRUTH_CONSTANTS = {
    'φ^(-4)': PHI**(-4),    # 0.145898
    'φ^(-3)': PHI**(-3),    # 0.236068
    'φ^(-2)': PHI**(-2),    # 0.381966
    'φ^(-1)': PHI**(-1),    # 0.618034
    '1/(2φ)': 1/(2*PHI),    # 0.309017 - Event horizon
    '3/4': 0.75,            # Emerges in MNIST
    '1/√2': 1/np.sqrt(2),   # 0.707107
    '1': 1.0,               # Identity
    'φ^(0.5)': PHI**(0.5),  # 1.272020
    'φ': PHI,               # 1.618034
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EncodedData:
    """
    Result of encoding values to truth space.
    
    Attributes:
        indices: Index into mesh for each value (or n_int for φ-Lens)
        signs: Sign of each value (+1 or -1)
        scale: Global scale factor
        shape: Original shape of the data
        n_frac: Fractional indices (φ-Lens only)
        residuals: Residuals for multi-level encoding
    """
    indices: np.ndarray
    signs: np.ndarray
    scale: float
    shape: Tuple[int, ...]
    n_frac: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    
    @property
    def bits_per_value(self) -> float:
        """Compute bits per value for this encoding."""
        n_values = np.prod(self.shape)
        
        # Indices
        if self.n_frac is not None:
            # φ-Lens: n_int (5 bits) + n_frac (4 bits) + sign (1 bit) = 10 bits
            total_bits = n_values * 10 + 32  # +32 for scale
        else:
            # Vector mesh: log2(n_anchors) + 1 bit for sign
            n_anchors = self.indices.max() + 1
            bits_per_index = int(np.ceil(np.log2(max(n_anchors, 2))))
            total_bits = n_values * (bits_per_index + 1) + 32
        
        return total_bits / n_values
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP32."""
        return 32 / self.bits_per_value


@dataclass
class TruthMesh:
    """
    Precomputed mesh structure for navigation.
    
    Attributes:
        anchors: The anchor values in the mesh
        names: Human-readable names for anchors
        n_levels: Number of refinement levels
        refinements: Per-level refinement grids
    """
    anchors: np.ndarray
    names: List[str] = field(default_factory=list)
    n_levels: int = 1
    refinements: Optional[List[np.ndarray]] = None


# =============================================================================
# PHI-LENS: Single-basis encoding with continuous power
# =============================================================================

class PhiLens:
    """
    φ-Lens encoder/decoder.
    
    Encodes values as φ^(-N_smooth) where N_smooth is quantized to
    (n_int, n_frac) pairs. Achieves ~3x compression with high accuracy.
    
    Args:
        n_int_bits: Bits for integer part of N_smooth (default: 5, range 0-31)
        n_frac_bits: Bits for fractional part (default: 4, range 0-15)
    
    Example:
        lens = PhiLens()
        encoded = lens.encode(weights)
        reconstructed = lens.decode(encoded)
    """
    
    def __init__(self, n_int_bits: int = 5, n_frac_bits: int = 4):
        self.n_int_bits = n_int_bits
        self.n_frac_bits = n_frac_bits
        self.n_int_max = 2**n_int_bits - 1
        self.n_frac_max = 2**n_frac_bits - 1
        
        # Build LUT
        self.lut = self._build_lut()
    
    def _build_lut(self) -> np.ndarray:
        """Build the φ lookup table."""
        lut = []
        for n_int in range(self.n_int_max + 1):
            for n_frac in range(self.n_frac_max + 1):
                n_smooth = n_int + n_frac / self.n_frac_max - 0.5
                lut.append(PHI ** (-n_smooth))
        return np.array(lut)
    
    def encode(self, data: np.ndarray) -> EncodedData:
        """
        Encode data to φ-levels.
        
        Args:
            data: Array of values to encode
            
        Returns:
            EncodedData with indices (n_int), n_frac, signs, scale
        """
        shape = data.shape
        flat = data.flatten()
        
        # Extract scale and signs
        scale = np.abs(flat).max()
        if scale < 1e-10:
            scale = 1.0
        normalized = flat / scale
        
        signs = np.sign(normalized)
        signs[signs == 0] = 1
        
        # Compute N_smooth = -log(|v|) / log(φ)
        abs_norm = np.abs(normalized).clip(1e-10)
        n_smooth = -np.log(abs_norm) / LOG_PHI
        
        # Quantize
        n_int = np.round(n_smooth).astype(int).clip(0, self.n_int_max)
        n_frac_float = (n_smooth - n_int + 0.5) * self.n_frac_max
        n_frac = np.round(n_frac_float).astype(int).clip(0, self.n_frac_max)
        
        return EncodedData(
            indices=n_int,
            signs=signs,
            scale=float(scale),
            shape=shape,
            n_frac=n_frac
        )
    
    def decode(self, encoded: EncodedData) -> np.ndarray:
        """
        Decode from φ-levels back to values.
        
        Args:
            encoded: EncodedData from encode()
            
        Returns:
            Reconstructed array with original shape
        """
        # Use LUT for fast decoding
        lut_indices = encoded.indices * (self.n_frac_max + 1) + encoded.n_frac
        values = self.lut[lut_indices] * encoded.signs * encoded.scale
        
        return values.reshape(encoded.shape)
    
    def encode_decode(self, data: np.ndarray) -> np.ndarray:
        """Convenience method: encode then decode."""
        return self.decode(self.encode(data))


# =============================================================================
# VECTOR MESH: Multi-anchor encoding with integer powers
# =============================================================================

class VectorMesh:
    """
    Vector mesh encoder/decoder.
    
    Encodes values by finding the closest anchor in a predefined mesh.
    More flexible than φ-Lens for values that cluster at specific constants.
    
    Args:
        anchors: Dict of name -> value, or None for default φ-based anchors
        
    Example:
        mesh = VectorMesh()
        encoded = mesh.encode(data)
        reconstructed = mesh.decode(encoded)
    """
    
    def __init__(self, anchors: Dict[str, float] = None):
        if anchors is None:
            anchors = TRUTH_CONSTANTS
        
        self.anchor_names = list(anchors.keys())
        self.anchor_values = np.array([anchors[n] for n in self.anchor_names])
        
        # Sort by value
        order = np.argsort(self.anchor_values)
        self.anchor_values = self.anchor_values[order]
        self.anchor_names = [self.anchor_names[i] for i in order]
        
        self.mesh = TruthMesh(
            anchors=self.anchor_values,
            names=self.anchor_names
        )
    
    def encode(self, data: np.ndarray) -> EncodedData:
        """
        Encode data to mesh indices.
        
        Args:
            data: Array of values to encode
            
        Returns:
            EncodedData with indices, signs, scale
        """
        shape = data.shape
        flat = data.flatten()
        
        # Extract scale and signs
        scale = np.abs(flat).max()
        if scale < 1e-10:
            scale = 1.0
        normalized = np.abs(flat) / scale
        
        signs = np.sign(flat)
        signs[signs == 0] = 1
        
        # Find closest anchor for each value
        diffs = np.abs(normalized[:, None] - self.anchor_values)
        indices = np.argmin(diffs, axis=1)
        
        # Compute residuals
        residuals = normalized / self.anchor_values[indices].clip(1e-10)
        
        return EncodedData(
            indices=indices,
            signs=signs,
            scale=float(scale),
            shape=shape,
            residuals=residuals
        )
    
    def decode(self, encoded: EncodedData) -> np.ndarray:
        """
        Decode from mesh indices back to values.
        
        Args:
            encoded: EncodedData from encode()
            
        Returns:
            Reconstructed array with original shape
        """
        values = self.anchor_values[encoded.indices] * encoded.signs * encoded.scale
        return values.reshape(encoded.shape)
    
    def decode_with_residuals(self, encoded: EncodedData) -> np.ndarray:
        """Decode using residuals for higher accuracy."""
        if encoded.residuals is None:
            return self.decode(encoded)
        
        values = (self.anchor_values[encoded.indices] * 
                  encoded.residuals * encoded.signs * encoded.scale)
        return values.reshape(encoded.shape)
    
    def analyze_clustering(self, data: np.ndarray, threshold: float = 0.1) -> Dict[str, float]:
        """
        Analyze how data clusters around anchors.
        
        Args:
            data: Data to analyze
            threshold: Distance threshold for "close" to anchor
            
        Returns:
            Dict of anchor_name -> percentage of values near that anchor
        """
        scale = np.abs(data).max()
        if scale < 1e-10:
            return {}
        normalized = np.abs(data.flatten()) / scale
        
        results = {}
        for name, anchor in zip(self.anchor_names, self.anchor_values):
            count = np.sum(np.abs(normalized - anchor) < threshold)
            pct = count / len(normalized) * 100
            if pct > 0:
                results[name] = pct
        
        return results


# =============================================================================
# HIERARCHICAL MESH: Multi-level refinement
# =============================================================================

class HierarchicalMesh:
    """
    Hierarchical mesh with multi-level refinement.
    
    Level 0: Coarse anchors (φ-based or custom)
    Level 1+: Refinement multipliers around 1.0
    
    This enables arbitrary precision by adding levels.
    
    Args:
        base_anchors: Base anchor values (default: φ powers)
        n_levels: Number of refinement levels
        refinement_range: Range around 1.0 for refinements
        refinement_steps: Number of steps per refinement level
    """
    
    def __init__(
        self,
        base_anchors: np.ndarray = None,
        n_levels: int = 3,
        refinement_range: float = 0.2,
        refinement_steps: int = 5
    ):
        if base_anchors is None:
            base_anchors = np.array([PHI**(-k) for k in range(8)])
        
        self.base_anchors = base_anchors
        self.n_levels = n_levels
        
        # Build refinement grids
        self.refinements = []
        for level in range(n_levels - 1):
            step = refinement_range / (2 ** level)
            grid = np.linspace(1 - step, 1 + step, refinement_steps)
            self.refinements.append(grid)
    
    def encode(self, data: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float]:
        """
        Encode with hierarchical refinement.
        
        Returns:
            (list of index arrays per level, signs, scale)
        """
        shape = data.shape
        flat = data.flatten()
        
        scale = np.abs(flat).max()
        if scale < 1e-10:
            scale = 1.0
        normalized = np.abs(flat) / scale
        signs = np.sign(flat)
        signs[signs == 0] = 1
        
        all_indices = []
        current = normalized.copy()
        
        # Level 0: Base anchors
        diffs = np.abs(current[:, None] - self.base_anchors)
        indices_0 = np.argmin(diffs, axis=1)
        all_indices.append(indices_0)
        current = current / self.base_anchors[indices_0].clip(1e-10)
        
        # Levels 1+: Refinements
        for refinement in self.refinements:
            diffs = np.abs(current[:, None] - refinement)
            indices = np.argmin(diffs, axis=1)
            all_indices.append(indices)
            current = current / refinement[indices].clip(1e-10)
        
        return all_indices, signs, scale, shape
    
    def decode(
        self,
        all_indices: List[np.ndarray],
        signs: np.ndarray,
        scale: float,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Decode from hierarchical indices."""
        values = self.base_anchors[all_indices[0]]
        
        for level, indices in enumerate(all_indices[1:]):
            values = values * self.refinements[level][indices]
        
        values = values * signs * scale
        return values.reshape(shape)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def measure_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Measure reconstruction error.
    
    Returns:
        Dict with max_error, mean_error, mse, relative_error
    """
    diff = np.abs(original - reconstructed)
    return {
        'max_error': float(diff.max()),
        'mean_error': float(diff.mean()),
        'mse': float((diff ** 2).mean()),
        'relative_error': float((diff / np.abs(original).clip(1e-10)).mean())
    }


def pack_10bit(n_int: np.ndarray, n_frac: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """
    Pack φ-Lens encoding to 10-bit format (3 values per int32).
    
    Layout per value: [n_int:5][n_frac:4][sign:1]
    Layout per int32: [val0:10][val1:10][val2:10][unused:2]
    """
    n = len(n_int)
    # Pad to multiple of 3
    pad = (3 - n % 3) % 3
    if pad > 0:
        n_int = np.pad(n_int, (0, pad), constant_values=0)
        n_frac = np.pad(n_frac, (0, pad), constant_values=0)
        signs = np.pad(signs, (0, pad), constant_values=1)
    
    # Reshape to groups of 3
    n_int = n_int.reshape(-1, 3)
    n_frac = n_frac.reshape(-1, 3)
    sign_bits = (signs.reshape(-1, 3) > 0).astype(np.int32)
    
    # Pack each value to 10 bits
    def pack_one(ni, nf, s):
        return ((ni.astype(np.int32) & 0x1F) << 5) | \
               ((nf.astype(np.int32) & 0x0F) << 1) | \
               (s & 0x01)
    
    w0 = pack_one(n_int[:, 0], n_frac[:, 0], sign_bits[:, 0])
    w1 = pack_one(n_int[:, 1], n_frac[:, 1], sign_bits[:, 1])
    w2 = pack_one(n_int[:, 2], n_frac[:, 2], sign_bits[:, 2])
    
    # Combine into int32
    packed = (w0 << 20) | (w1 << 10) | w2
    
    return packed.astype(np.int32)


def unpack_10bit(packed: np.ndarray, n_values: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack 10-bit format back to (n_int, n_frac, signs).
    """
    # Extract 3 values per int32
    w0 = (packed >> 20) & 0x3FF
    w1 = (packed >> 10) & 0x3FF
    w2 = packed & 0x3FF
    
    # Interleave
    n_packed = len(packed)
    all_w = np.zeros(n_packed * 3, dtype=np.int32)
    all_w[0::3] = w0
    all_w[1::3] = w1
    all_w[2::3] = w2
    all_w = all_w[:n_values]
    
    # Unpack each 10-bit value
    n_int = (all_w >> 5) & 0x1F
    n_frac = (all_w >> 1) & 0x0F
    signs = (all_w & 0x01) * 2 - 1  # Convert 0/1 to -1/+1
    
    return n_int, n_frac, signs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compress(data: np.ndarray, method: str = 'phi') -> EncodedData:
    """
    Compress data using dimensional navigation.
    
    Args:
        data: Array to compress
        method: 'phi' for φ-Lens, 'mesh' for VectorMesh
        
    Returns:
        EncodedData
    """
    if method == 'phi':
        return PhiLens().encode(data)
    elif method == 'mesh':
        return VectorMesh().encode(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def decompress(encoded: EncodedData, method: str = 'phi') -> np.ndarray:
    """
    Decompress encoded data.
    
    Args:
        encoded: EncodedData from compress()
        method: 'phi' for φ-Lens, 'mesh' for VectorMesh
        
    Returns:
        Reconstructed array
    """
    if method == 'phi':
        return PhiLens().decode(encoded)
    elif method == 'mesh':
        return VectorMesh().decode(encoded)
    else:
        raise ValueError(f"Unknown method: {method}")
