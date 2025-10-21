#!/usr/bin/env python3
"""
Holographer's Workbench Examples
=================================

Demonstrates the unified workbench API with practical examples.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from workbench import (
    SpectralScorer,
    ZetaFiducials,
    SublinearOptimizer,
    holographic_refinement,
    phase_retrieve_hilbert,
)
from workbench.primitives.signal import normalize as normalize_signal, psnr as compute_psnr


def example_1_spectral_scoring():
    """Example 1: Unified spectral scoring."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Spectral Scoring with Zeta Zeros")
    print("=" * 70)
    
    # Get zeta zeros (cached automatically)
    zeros = ZetaFiducials.get_standard(20)
    print(f"Loaded {len(zeros)} zeta zeros")
    print(f"First 5: {zeros[:5]}")
    
    # Create candidates
    candidates = np.arange(100, 1000)
    print(f"\nScoring {len(candidates)} candidates...")
    
    # Score using spectral method
    scorer = SpectralScorer(frequencies=zeros, damping=0.05)
    scores = scorer.compute_scores(candidates, shift=0.05, mode="real")
    
    # Find top 10
    top_idx = np.argsort(-scores)[:10]
    top_candidates = candidates[top_idx]
    
    print(f"Top 10 candidates: {top_candidates}")
    print("=" * 70)


def example_2_phase_retrieval():
    """Example 2: Phase retrieval and envelope extraction."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Phase Retrieval")
    print("=" * 70)
    
    # Create noisy signal with peaks
    n = 1000
    x = np.linspace(0, 10, n)
    signal = np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.random.randn(n)
    
    print(f"Signal length: {n}")
    print(f"Signal range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
    
    # Extract envelope using Hilbert transform
    envelope, phase_var = phase_retrieve_hilbert(signal)
    
    print(f"\nEnvelope extracted:")
    print(f"  Envelope range: [{np.min(envelope):.3f}, {np.max(envelope):.3f}]")
    print(f"  Phase variance: {phase_var:.6f}")
    
    if phase_var < 0.05:
        print("  ✓ High quality signal (low phase variance)")
    elif phase_var < 0.12:
        print("  ○ Medium quality signal")
    else:
        print("  ✗ Low quality signal (high phase variance)")
    
    print("=" * 70)


def example_3_holographic_refinement():
    """Example 3: Holographic refinement of noisy scores."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Holographic Refinement")
    print("=" * 70)
    
    # Create noisy scores
    n = 500
    candidates = np.arange(n)
    
    # True signal: peaks at specific locations
    true_peaks = [100, 200, 300, 400]
    true_signal = np.zeros(n)
    for peak in true_peaks:
        true_signal += 10 * np.exp(-((candidates - peak) / 20) ** 2)
    
    # Add noise
    noisy_scores = true_signal + np.random.randn(n) * 2
    
    # Define smooth reference
    reference = 1.0 / (1.0 + 0.001 * np.abs(candidates - n/2))
    
    print(f"Candidates: {n}")
    print(f"True peaks at: {true_peaks}")
    print(f"Noise level: 2.0")
    
    # Refine using holographic method
    refined = holographic_refinement(
        noisy_scores,
        reference,
        method="hilbert",
        blend_ratio=0.6
    )
    
    # Find peaks in refined signal
    refined_norm = normalize_signal(refined, method="minmax")
    detected_peaks = []
    for i in range(1, n-1):
        if refined_norm[i] > refined_norm[i-1] and refined_norm[i] > refined_norm[i+1]:
            if refined_norm[i] > 0.7:  # Threshold
                detected_peaks.append(i)
    
    # Check accuracy
    found = 0
    for true_peak in true_peaks:
        if any(abs(p - true_peak) < 30 for p in detected_peaks):
            found += 1
    
    print(f"\nRefinement results:")
    print(f"  Detected {len(detected_peaks)} peaks")
    print(f"  Found {found}/{len(true_peaks)} true peaks")
    print(f"  Accuracy: {found/len(true_peaks)*100:.1f}%")
    
    # Compare PSNR
    psnr_before = compute_psnr(true_signal, noisy_scores, max_value=np.max(true_signal))
    psnr_after = compute_psnr(true_signal, refined, max_value=np.max(true_signal))
    
    print(f"\nPSNR improvement:")
    print(f"  Before: {psnr_before:.2f} dB")
    print(f"  After:  {psnr_after:.2f} dB")
    print(f"  Gain:   {psnr_after - psnr_before:.2f} dB")
    
    print("=" * 70)


def example_4_sublinear_optimization():
    """Example 4: Sublinear optimization with holographic refinement."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Sublinear Optimization")
    print("=" * 70)
    
    # Large candidate set
    n = 10000
    candidates = np.arange(n)
    
    print(f"Problem size: {n:,} candidates")
    print(f"Goal: Find top 100 using O(√n) operations")
    
    # Define expensive scoring function
    def expensive_score(cands):
        # Simulate expensive computation
        return np.sin(cands * 0.01) + 0.5 * np.cos(cands * 0.03)
    
    # Optimize
    optimizer = SublinearOptimizer(
        use_holographic=True,
        phase_retrieval_method="hilbert",
        blend_ratio=0.6
    )
    
    top_100, stats = optimizer.optimize(
        candidates,
        expensive_score,
        top_k=100
    )
    
    print(f"\nOptimization results:")
    print(f"  Original size: {stats.n_original:,}")
    print(f"  Final size: {stats.n_final}")
    print(f"  Reduction: {stats.reduction_ratio:.2%}")
    print(f"  Complexity: {stats.complexity_estimate}")
    print(f"  Time: {stats.time_elapsed:.4f}s")
    
    print(f"\nTop 10 candidates: {top_100[:10]}")
    
    print("=" * 70)


def example_5_complete_workflow():
    """Example 5: Complete workflow combining all techniques."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 70)
    
    print("Task: Find best 50 candidates from 5000 using all techniques")
    print()
    
    # Setup
    n = 5000
    candidates = np.arange(n)
    
    # Step 1: Spectral scoring
    print("Step 1: Spectral scoring with zeta zeros...")
    zeros = ZetaFiducials.get_standard(15)
    scorer = SpectralScorer(frequencies=zeros, damping=0.05)
    spectral_scores = scorer.compute_scores(candidates, shift=0.05, mode="real")
    print(f"  ✓ Computed spectral scores")
    
    # Step 2: Define reference baseline
    print("\nStep 2: Computing reference baseline...")
    reference = 1.0 / (np.log(candidates + 2) + 1e-12)
    reference = normalize_signal(reference, method="max")
    print(f"  ✓ Reference baseline ready")
    
    # Step 3: Holographic refinement
    print("\nStep 3: Holographic refinement...")
    refined_scores = holographic_refinement(
        spectral_scores,
        reference,
        method="hilbert",
        blend_ratio=0.6
    )
    print(f"  ✓ Scores refined using phase retrieval")
    
    # Step 4: Sublinear optimization
    print("\nStep 4: Sublinear optimization...")
    optimizer = SublinearOptimizer(use_holographic=False)  # Already refined
    
    def score_fn(cands):
        return refined_scores[cands]
    
    top_50, stats = optimizer.optimize(
        candidates,
        score_fn,
        top_k=50
    )
    
    print(f"  ✓ Optimized to top 50")
    
    # Results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Original candidates: {stats.n_original:,}")
    print(f"Final candidates: {stats.n_final}")
    print(f"Reduction: {(1 - stats.reduction_ratio) * 100:.1f}%")
    print(f"Complexity: {stats.complexity_estimate}")
    print(f"Time: {stats.time_elapsed:.4f}s")
    print()
    print(f"Top 10 results: {top_50[:10]}")
    print()
    print("Techniques used:")
    print("  ✓ Spectral scoring (zeta zeros)")
    print("  ✓ Phase retrieval (Hilbert transform)")
    print("  ✓ Holographic refinement (interference)")
    print("  ✓ Sublinear optimization (O(√n))")
    
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("HOLOGRAPHER'S WORKBENCH - UNIFIED API EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating the unified toolkit for holographic signal processing,")
    print("spectral optimization, and sublinear algorithms.")
    
    try:
        example_1_spectral_scoring()
        example_2_phase_retrieval()
        example_3_holographic_refinement()
        example_4_sublinear_optimization()
        example_5_complete_workflow()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nThe workbench provides:")
        print("  • Unified spectral scoring")
        print("  • Phase retrieval methods")
        print("  • Holographic refinement")
        print("  • Sublinear optimization")
        print("  • Reusable utilities")
        print()
        print("See workbench/README.md for full documentation.")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
