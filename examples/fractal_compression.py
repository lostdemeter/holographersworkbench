"""
Example: Recursive Fractal Peeling
===================================

Demonstrates the Recursive Fractal Peeling algorithm for lossless
data compression and structure analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from workbench.processors.compression import (
    FractalPeeler,
    resfrac_score
)


def example_1_basic_compression():
    """Example 1: Basic compression and decompression."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Compression")
    print("="*70)
    
    # Create a structured signal
    t = np.linspace(0, 10, 500)
    signal = np.sin(2*np.pi*t) + 0.5*np.sin(6*np.pi*t)
    
    print(f"Original signal: n = {len(signal)}")
    print(f"Resfrac score: ρ = {resfrac_score(signal):.4f}")
    
    # Compress
    peeler = FractalPeeler(order=4, max_depth=5)
    tree = peeler.compress(signal)
    
    # Decompress
    reconstructed = peeler.decompress(tree)
    
    # Verify lossless
    error = np.max(np.abs(signal - reconstructed))
    print(f"\nReconstruction error: {error:.2e}")
    print(f"Lossless: {error < 1e-10}")
    
    # Statistics
    stats = peeler.tree_stats(tree)
    ratio = peeler.compression_ratio(tree, len(signal))
    print(f"\nCompression ratio: {ratio:.2f}x")
    print(f"Tree depth: {stats['max_depth']}")
    print(f"Nodes: {stats['num_nodes']}, Leaves: {stats['num_leaves']}")


def example_2_resfrac_analysis():
    """Example 2: Analyze predictability with resfrac scores."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Resfrac Score Analysis")
    print("="*70)
    
    n = 300
    signals = {
        'Sine wave': np.sin(np.linspace(0, 8*np.pi, n)),
        'Polynomial': np.polyval([0.001, -0.1, 2, 10], np.arange(n)),
        'Random noise': np.random.randn(n),
        'Sine + noise': np.sin(np.linspace(0, 8*np.pi, n)) + 0.3*np.random.randn(n)
    }
    
    print("\nResfrac scores (ρ ∈ [0,1], lower = more predictable):")
    print("-" * 60)
    
    for name, sig in signals.items():
        rho = resfrac_score(sig, order=3)
        if rho < 0.3:
            category = "highly structured"
        elif rho < 0.7:
            category = "mixed"
        else:
            category = "random"
        print(f"{name:<20} ρ = {rho:.4f}  ({category})")


def example_3_compression_comparison():
    """Example 3: Compare compression across signal types."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Compression Performance Comparison")
    print("="*70)
    
    n = 400
    peeler = FractalPeeler(order=3, max_depth=8)
    
    signals = {
        'Sine': np.sin(np.linspace(0, 10*np.pi, n)),
        'Chirp': np.sin(np.linspace(0, 20*np.pi, n) * np.linspace(1, 3, n)),
        'Exponential': np.exp(np.linspace(0, 2, n)),
        'Polynomial': np.polyval([0.001, -0.1, 2, 10], np.arange(n)),
        'Random': np.random.randn(n)
    }
    
    print(f"\n{'Signal':<15} {'ρ':<8} {'Depth':<8} {'Nodes':<8} {'Ratio':<10}")
    print("-" * 60)
    
    for name, sig in signals.items():
        rho = resfrac_score(sig)
        tree = peeler.compress(sig)
        stats = peeler.tree_stats(tree)
        ratio = peeler.compression_ratio(tree, len(sig))
        
        print(f"{name:<15} {rho:<8.4f} {stats['max_depth']:<8} "
              f"{stats['num_nodes']:<8} {ratio:<10.2f}x")


def example_4_tree_visualization():
    """Example 4: Visualize compression tree structure."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Tree Visualization")
    print("="*70)
    
    # Create a signal with clear structure
    t = np.linspace(0, 5, 200)
    signal = np.sin(4*np.pi*t) + 0.2*np.random.randn(len(t))
    
    # Compress with limited depth for visualization
    peeler = FractalPeeler(order=3, max_depth=3)
    tree = peeler.compress(signal)
    
    print("\nCompression Tree Structure:")
    print(peeler.visualize(tree))
    
    print("Legend:")
    print("  NODE: Internal node with AR model + residuals")
    print("  LEAF: Terminal node with raw data")
    print("  ρ:    Resfrac score (predictability)")
    print("  ρᵣ:   Residual resfrac score")
    print("  Δρ:   Improvement from pattern extraction")


def example_5_parameter_tuning():
    """Example 5: Effect of AR model order."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Parameter Tuning (AR Order)")
    print("="*70)
    
    signal = np.sin(np.linspace(0, 12*np.pi, 500)) + 0.1*np.random.randn(500)
    
    print(f"\nSignal: n = {len(signal)}, ρ = {resfrac_score(signal):.4f}")
    print(f"\n{'Order':<10} {'Depth':<10} {'Nodes':<10} {'Ratio':<10}")
    print("-" * 50)
    
    for order in [2, 3, 5, 8]:
        peeler = FractalPeeler(order=order, max_depth=8)
        tree = peeler.compress(signal)
        stats = peeler.tree_stats(tree)
        ratio = peeler.compression_ratio(tree, len(signal))
        
        print(f"{order:<10} {stats['max_depth']:<10} "
              f"{stats['num_nodes']:<10} {ratio:<10.2f}x")
    
    print("\nInsight: Higher order captures more complex patterns")
    print("         but increases model size. Optimal depends on signal.")


def example_6_integration_pipeline():
    """Example 6: Integration with workbench pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Integration with Workbench Pipeline")
    print("="*70)
    
    # Import other workbench modules
    try:
        from spectral import ZetaFiducials, SpectralScorer
        from holographic import phase_retrieve_hilbert
        
        # Create complex signal
        t = np.linspace(0, 15, 800)
        signal = (np.sin(2*np.pi*t) + 0.5*np.sin(6*np.pi*t) + 
                  0.3*np.sin(10*np.pi*t) + 0.2*np.random.randn(len(t)))
        
        print("\nPipeline: Spectral → Holographic → Fractal Peeling")
        print("-" * 60)
        
        # Step 1: Spectral scoring
        zeros = ZetaFiducials.get_standard(10)
        scorer = SpectralScorer(zeros)
        indices = np.arange(len(signal))
        scores = scorer.compute_scores(indices, mode='real')
        print(f"1. Spectral scoring:   mean = {np.mean(scores):.4f}")
        
        # Step 2: Phase retrieval
        envelope, phase_var = phase_retrieve_hilbert(signal)
        print(f"2. Phase retrieval:    PV = {phase_var:.4f}")
        
        # Step 3: Fractal peeling
        peeler = FractalPeeler(order=4, max_depth=6)
        tree = peeler.compress(signal)
        ratio = peeler.compression_ratio(tree, len(signal))
        print(f"3. Fractal peeling:    {ratio:.2f}x compression")
        
        # Verify lossless
        reconstructed = peeler.decompress(tree)
        error = np.max(np.abs(signal - reconstructed))
        print(f"4. Reconstruction:     error = {error:.2e}")
        
        print("-" * 60)
        print("✓ Pipeline complete: Lossless compression with structure analysis")
        
    except ImportError as e:
        print(f"\nSkipping integration example (missing modules): {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("RECURSIVE FRACTAL PEELING EXAMPLES")
    print("="*70)
    
    example_1_basic_compression()
    example_2_resfrac_analysis()
    example_3_compression_comparison()
    example_4_tree_visualization()
    example_5_parameter_tuning()
    example_6_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
