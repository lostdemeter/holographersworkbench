#!/usr/bin/env python3
"""
Dimensional Navigation: Examples

Run this file to see dimensional navigation in action:
    python examples.py
"""

import numpy as np
from core import (
    PhiLens, VectorMesh, HierarchicalMesh,
    measure_error, pack_10bit, unpack_10bit,
    PHI, TRUTH_CONSTANTS
)


def example_phi_lens():
    """Basic φ-Lens compression example."""
    print("=" * 70)
    print("EXAMPLE 1: φ-Lens Compression")
    print("=" * 70)
    print()
    
    # Create test data (simulating neural network weights)
    np.random.seed(42)
    weights = np.random.randn(100, 100) * 0.1
    
    print(f"Original data: {weights.shape}")
    print(f"Range: [{weights.min():.4f}, {weights.max():.4f}]")
    print()
    
    # Encode
    lens = PhiLens()
    encoded = lens.encode(weights)
    
    print(f"Encoded:")
    print(f"  n_int range: [{encoded.indices.min()}, {encoded.indices.max()}]")
    print(f"  n_frac range: [{encoded.n_frac.min()}, {encoded.n_frac.max()}]")
    print(f"  scale: {encoded.scale:.6f}")
    print(f"  bits per value: {encoded.bits_per_value:.2f}")
    print(f"  compression ratio: {encoded.compression_ratio:.2f}x")
    print()
    
    # Decode
    reconstructed = lens.decode(encoded)
    errors = measure_error(weights, reconstructed)
    
    print(f"Reconstruction error:")
    print(f"  max: {errors['max_error']:.6f}")
    print(f"  mean: {errors['mean_error']:.6f}")
    print(f"  mse: {errors['mse']:.8f}")
    print()


def example_vector_mesh():
    """Vector mesh with custom anchors."""
    print("=" * 70)
    print("EXAMPLE 2: Vector Mesh with Custom Anchors")
    print("=" * 70)
    print()
    
    # Data that clusters at specific values
    np.random.seed(42)
    data = np.concatenate([
        np.random.randn(100) * 0.05 + 0.75,   # Cluster at 3/4
        np.random.randn(100) * 0.05 + 0.618,  # Cluster at 1/φ
        np.random.randn(100) * 0.05 + 0.382,  # Cluster at φ^(-2)
    ])
    
    print(f"Data with 3 clusters: {len(data)} values")
    print()
    
    # Encode with default anchors
    mesh = VectorMesh()
    encoded = mesh.encode(data)
    
    # Analyze clustering
    clustering = mesh.analyze_clustering(data)
    print("Clustering analysis:")
    for name, pct in sorted(clustering.items(), key=lambda x: -x[1]):
        print(f"  {name}: {pct:.1f}%")
    print()
    
    # Decode
    reconstructed = mesh.decode(encoded)
    errors = measure_error(data, reconstructed)
    
    print(f"Reconstruction error (without residuals):")
    print(f"  max: {errors['max_error']:.6f}")
    print()
    
    # Decode with residuals for higher accuracy
    reconstructed_precise = mesh.decode_with_residuals(encoded)
    errors_precise = measure_error(data, reconstructed_precise)
    
    print(f"Reconstruction error (with residuals):")
    print(f"  max: {errors_precise['max_error']:.6f}")
    print()


def example_matrix_solve():
    """Solve a matrix equation using dimensional navigation."""
    print("=" * 70)
    print("EXAMPLE 3: Matrix-Vector Multiplication")
    print("=" * 70)
    print()
    
    # Create a 4x4 matrix
    np.random.seed(42)
    A = np.random.randn(4, 4) * 0.5
    x = np.array([1.0, 2.0, 3.0, 4.0])
    
    print("Matrix A:")
    print(A)
    print()
    print(f"Vector x: {x}")
    print()
    
    # True result
    y_true = A @ x
    print(f"True y = A @ x: {y_true}")
    print()
    
    # Encode matrix
    lens = PhiLens()
    encoded = lens.encode(A)
    
    print(f"Matrix encoded: {encoded.compression_ratio:.2f}x compression")
    print()
    
    # Reconstruct and compute
    A_reconstructed = lens.decode(encoded)
    y_reconstructed = A_reconstructed @ x
    
    print(f"Reconstructed y: {y_reconstructed}")
    print(f"Error: {np.abs(y_true - y_reconstructed).max():.6f}")
    print()


def example_hierarchical():
    """Multi-level refinement for higher precision."""
    print("=" * 70)
    print("EXAMPLE 4: Hierarchical Mesh (Multi-Level)")
    print("=" * 70)
    print()
    
    # Create test data
    np.random.seed(42)
    data = np.random.randn(1000) * 0.1
    
    print(f"Data: {len(data)} values")
    print()
    
    # Compare different numbers of levels
    for n_levels in [1, 2, 3]:
        mesh = HierarchicalMesh(n_levels=n_levels)
        indices, signs, scale, shape = mesh.encode(data)
        reconstructed = mesh.decode(indices, signs, scale, shape)
        errors = measure_error(data, reconstructed)
        
        print(f"Levels: {n_levels}")
        print(f"  Max error: {errors['max_error']:.6f}")
        print(f"  Mean error: {errors['mean_error']:.6f}")
    print()


def example_binary_packing():
    """Pack to 10-bit binary format."""
    print("=" * 70)
    print("EXAMPLE 5: Binary Packing (10-bit)")
    print("=" * 70)
    print()
    
    # Encode some data
    np.random.seed(42)
    data = np.random.randn(12) * 0.1  # 12 values = 4 packed int32s
    
    lens = PhiLens()
    encoded = lens.encode(data)
    
    print(f"Original: {len(data)} values × 32 bits = {len(data) * 32} bits")
    print()
    
    # Pack to 10-bit
    packed = pack_10bit(encoded.indices, encoded.n_frac, encoded.signs)
    
    print(f"Packed: {len(packed)} int32s × 32 bits = {len(packed) * 32} bits")
    print(f"(stores {len(packed) * 3} values at 10.67 bits each)")
    print()
    
    # Unpack
    n_int, n_frac, signs = unpack_10bit(packed, len(data))
    
    # Verify
    print("Verification:")
    print(f"  n_int match: {np.array_equal(n_int, encoded.indices)}")
    print(f"  n_frac match: {np.array_equal(n_frac, encoded.n_frac)}")
    print(f"  signs match: {np.array_equal(signs, encoded.signs)}")
    print()


def example_truth_constants():
    """Show the truth space constants."""
    print("=" * 70)
    print("EXAMPLE 6: Truth Space Constants")
    print("=" * 70)
    print()
    
    print("Standard constants in truth space:")
    print()
    for name, value in sorted(TRUTH_CONSTANTS.items(), key=lambda x: x[1]):
        # Show relationship to φ
        if value > 0:
            phi_power = -np.log(value) / np.log(PHI)
            print(f"  {name:>10} = {value:.6f}  (≈ φ^({-phi_power:.2f}))")
    print()
    
    print("Key insight: These constants appear naturally in trained neural")
    print("networks and many other domains. Values cluster at these points.")
    print()


def main():
    """Run all examples."""
    example_phi_lens()
    example_vector_mesh()
    example_matrix_solve()
    example_hierarchical()
    example_binary_packing()
    example_truth_constants()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Dimensional Navigation enables:")
    print("  ✓ 3x compression of floating-point data")
    print("  ✓ Fast reconstruction via lookup tables")
    print("  ✓ Exact representation at mesh points")
    print("  ✓ Arbitrary precision via hierarchical refinement")
    print()
    print("The key insight: values cluster at mathematical constants.")
    print("Store coordinates, not values. Navigate, don't compute.")


if __name__ == "__main__":
    main()
