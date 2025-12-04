#!/usr/bin/env python3
"""
Test structural compression on actual discovered truth space points.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from structural_compression import FibonacciRecurrenceEncoder, GoldenQuantizer, StructuralCompressor
from truth_structure_discovery import TruthStructureDiscovery, DiscoveryConfig
from visualizations.truth_space_explorer import create_mathematical_validity_fn


def main():
    print("=" * 70)
    print("TESTING STRUCTURAL COMPRESSION ON REAL TRUTH SPACE DATA")
    print("=" * 70)
    
    # Generate real truth space points using discovery
    print("\nGenerating truth space points with 'golden' constraint...")
    
    validity_fn = create_mathematical_validity_fn("golden")
    config = DiscoveryConfig(
        n_boundary_samples=5000,
        n_interior_samples=2500,
    )
    
    discovery = TruthStructureDiscovery(config)
    discovery.explore_boundary(validity_fn)
    discovery.sample_interior(validity_fn)
    
    all_points = discovery.boundary_points + discovery.interior_points
    print(f"Generated {len(all_points)} valid points")
    
    # Analyze Fibonacci structure in real data
    print("\n" + "=" * 50)
    print("FIBONACCI STRUCTURE ANALYSIS")
    print("=" * 50)
    
    fib_errors = []
    for point in all_points[:1000]:
        sorted_vals = np.sort(point)[::-1]
        if len(sorted_vals) >= 3 and sorted_vals[2] > 0.01:
            error = abs(sorted_vals[0] - sorted_vals[1] - sorted_vals[2])
            fib_errors.append(error)
    
    print(f"Mean Fibonacci error: {np.mean(fib_errors):.4f}")
    print(f"Median Fibonacci error: {np.median(fib_errors):.4f}")
    print(f"Points with error < 0.05: {np.sum(np.array(fib_errors) < 0.05) / len(fib_errors):.1%}")
    print(f"Points with error < 0.10: {np.sum(np.array(fib_errors) < 0.10) / len(fib_errors):.1%}")
    
    # Test compression
    print("\n" + "=" * 50)
    print("COMPRESSION TEST")
    print("=" * 50)
    
    # Use more tolerant encoder
    encoder = FibonacciRecurrenceEncoder(tolerance=0.15, correction_bits=8)
    
    test_points = all_points[:1000]
    compressed, stats = encoder.compress(test_points)
    
    print(f"\nOriginal size:     {stats.original_bits:,} bits ({stats.original_bits//8:,} bytes)")
    print(f"Compressed size:   {stats.compressed_bits:,} bits ({stats.compressed_bits//8:,} bytes)")
    print(f"Compression ratio: {stats.ratio:.2f}x")
    print(f"Reconstruction error: {stats.reconstruction_error:.6f}")
    print(f"Fibonacci encoding: {stats.fibonacci_fraction:.1%} of points")
    
    # Verify reconstruction
    decompressed = encoder.decompress(compressed)
    
    print("\n" + "=" * 50)
    print("RECONSTRUCTION SAMPLES")
    print("=" * 50)
    
    errors = []
    for i in range(len(test_points)):
        error = np.linalg.norm(test_points[i] - decompressed[i])
        errors.append(error)
    
    print(f"\nReconstruction error distribution:")
    print(f"  Mean:   {np.mean(errors):.6f}")
    print(f"  Median: {np.median(errors):.6f}")
    print(f"  Max:    {np.max(errors):.6f}")
    print(f"  Min:    {np.min(errors):.6f}")
    
    # Show some examples
    for i in [0, 100, 500]:
        print(f"\nPoint {i}:")
        print(f"  Original:      [{', '.join(f'{v:.4f}' for v in test_points[i])}]")
        print(f"  Reconstructed: [{', '.join(f'{v:.4f}' for v in decompressed[i])}]")
        print(f"  Error: {errors[i]:.6f}")
    
    # Test φ-quantization
    print("\n" + "=" * 50)
    print("φ-QUANTIZATION TEST")
    print("=" * 50)
    
    quantizer = GoldenQuantizer(max_power=12)
    print(f"Quantization levels: {len(quantizer.levels)}")
    print(f"Levels: {[f'{l:.4f}' for l in quantizer.levels]}")
    
    # Quantize and measure error
    quant_errors = []
    for point in test_points[:100]:
        encoded = quantizer.encode_point(point)
        decoded = quantizer.decode_point(encoded)
        error = np.linalg.norm(point - decoded)
        quant_errors.append(error)
    
    print(f"\nφ-quantization error:")
    print(f"  Mean:   {np.mean(quant_errors):.6f}")
    print(f"  Bits per point: 24 (vs 192 for full float)")
    print(f"  Compression ratio: {192/24:.1f}x")
    
    # High-level structural compressor
    print("\n" + "=" * 50)
    print("STRUCTURAL COMPRESSOR")
    print("=" * 50)
    
    compressor = StructuralCompressor()
    compressed, info = compressor.compress(test_points)
    
    print(f"\nDetected structure: {info['structure']}")
    if 'stats' in info:
        print(f"Compression ratio: {info['stats'].ratio:.2f}x")
    elif 'ratio' in info:
        print(f"Compression ratio: {info['ratio']:.2f}x")


if __name__ == "__main__":
    main()
