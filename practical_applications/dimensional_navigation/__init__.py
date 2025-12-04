"""
Dimensional Navigation
======================

A universal process for exact computation via structured navigation through
"truth space." Values are represented as coordinates in a precomputed mesh,
enabling compression and fast reconstruction.

Quick Start
-----------

    from dimensional_navigation import PhiLens, VectorMesh
    
    # Method 1: φ-Lens (best for neural network weights)
    lens = PhiLens()
    encoded = lens.encode(weights)
    reconstructed = lens.decode(encoded)
    print(f"Compression: {encoded.compression_ratio:.1f}x")
    
    # Method 2: Vector Mesh (best for data with known constants)
    mesh = VectorMesh()
    encoded = mesh.encode(data)
    reconstructed = mesh.decode(encoded)

The Process
-----------

1. DOWNCAST  - Map continuous values to discrete indices
2. QUANTIZE  - Store residual for precision  
3. BUILD MESH - Precompute the lookup structure
4. UPSCALE   - Refine iteratively for exact values
5. RECONSTRUCT - Navigate to answer via lookup

Key Insight
-----------

Values cluster at mathematically significant constants (φ, 1/√2, 3/4, etc.).
By storing indices into a precomputed mesh instead of raw values, we achieve:

- 3x compression (32-bit → 10-bit)
- Fast reconstruction (lookup instead of compute)
- Exact representation for values at mesh points

Classes
-------

- PhiLens: Single-basis encoding using φ^(-N_smooth)
- VectorMesh: Multi-anchor encoding with closest-match
- HierarchicalMesh: Multi-level refinement for arbitrary precision
- EncodedData: Container for encoded values
- TruthMesh: Precomputed mesh structure

Functions
---------

- compress(data, method): Convenience function to compress
- decompress(encoded, method): Convenience function to decompress
- measure_error(original, reconstructed): Compute error metrics
- pack_10bit / unpack_10bit: Binary packing for storage

Constants
---------

- PHI: Golden ratio ≈ 1.618034
- TRUTH_CONSTANTS: Standard mathematical constants in truth space
"""

from .core import (
    # Constants
    PHI,
    LOG_PHI,
    TRUTH_CONSTANTS,
    
    # Data structures
    EncodedData,
    TruthMesh,
    
    # Main classes
    PhiLens,
    VectorMesh,
    HierarchicalMesh,
    
    # Utility functions
    measure_error,
    pack_10bit,
    unpack_10bit,
    
    # Convenience functions
    compress,
    decompress,
)

__version__ = "1.0.0"
__all__ = [
    'PHI',
    'LOG_PHI', 
    'TRUTH_CONSTANTS',
    'EncodedData',
    'TruthMesh',
    'PhiLens',
    'VectorMesh',
    'HierarchicalMesh',
    'measure_error',
    'pack_10bit',
    'unpack_10bit',
    'compress',
    'decompress',
]
