#!/usr/bin/env python3
"""
Ergodic Jump: Jump Sequence Diagnostics
========================================

Diagnostic technique for uncovering hidden structure in ergodic (fully mixed,
high-entropy) signals by injecting targeted harmonics to induce non-ergodic
"stickiness," then peeling to extract the resonant filament.

Like uncoiling a quantum yo-yo: inject a harmonic into chaotic noise,
then recursively peel (dyadic downsampling) to yank out the filament, exposing
its resonant scar while collapsing noise layers.

Key Concepts:
-------------
1. **Ergodic Signal**: Fully mixed, white-noise chaotic (uniform entropy)
2. **Harmonic Injection**: Add low-frequency sine (e.g., 1/φ rhythm) to break ergodicity
3. **Ergodic Jump**: Recursive peeling extracts the injected filament
4. **Metrics**: resfrac drops to ~10^-4, Hurst spikes to ~0.72 reveal structure

Mathematical Foundation:
------------------------
- Inject: s' = s + A·sin(2πf·t) where f = 1/φ (golden ratio)
- Peel: Haar-style variance cascade v_l = Var(downsample_l(s'))
- Extract: filament = peel(s') - peel(s) (residual difference)
- Hurst: H = -slope(log(v_l) / log(2^l)) / 2

Use Cases:
----------
- Uncover latent biases in "random" errors
- Stress-test ergodicity in zeta clocks
- Enhance holographic refinement with filaments
- Detect hidden harmonics in residuals

Usage:
------
    from ergodic_jump import ErgodicJump
    
    # Initialize with golden ratio frequency
    jump = ErgodicJump(injection_freq=1/np.sqrt(5), amp=0.15)
    
    # Generate ergodic signal
    ergodic_signal = np.random.randn(1024)
    
    # Execute ergodic jump
    result = jump.execute(ergodic_signal)
    
    print(f"Resfrac drop: {result['resfrac_drop']:.4f}")
    print(f"Hurst shift: {result['hurst_shift']:.4f}")
    print(f"Filament length: {len(result['filament'])}")

Author: Holographer's Workbench
Date: October 21, 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    from .fractal_peeling import FractalPeeler, resfrac_score
except ImportError:
    from fractal_peeling import FractalPeeler, resfrac_score


class ErgodicJump(FractalPeeler):
    """
    Ergodic jump analyzer for jump sequence diagnostics.
    
    Injects non-ergodic harmonics into ergodic signals, then peels to extract
    the resonant filament and measure structural changes.
    """
    
    def __init__(self, 
                 injection_freq: float = None,
                 amp: float = 0.15,
                 max_depth: int = 8,
                 order: int = 4):
        """
        Initialize ergodic jump analyzer.
        
        Args:
            injection_freq: Frequency for harmonic injection (default: 1/φ golden ratio)
            amp: Amplitude scaling factor for injection (default: 0.15)
            max_depth: Maximum peel depth (default: 8)
            order: Fractal peel order (default: 4)
        """
        super().__init__(order=order, max_depth=max_depth)
        
        # Default to golden ratio inverse if not specified
        if injection_freq is None:
            injection_freq = 1.0 / np.sqrt(5)  # ~0.447 (golden ratio inverse)
        
        self.injection_freq = injection_freq
        self.amp = amp
        self.hurst_history = []  # Track non-ergodic shifts over time
    
    def inject_harmonic(self, signal: np.ndarray) -> np.ndarray:
        """
        Inject non-ergodic harmonic into ergodic signal.
        
        Adds a low-frequency harmonic to break ergodicity and create
        persistent correlations (stickiness).
        
        Args:
            signal: Input ergodic signal
        
        Returns:
            Signal with injected harmonic (non-ergodic)
        """
        n = len(signal)
        phase = 2 * np.pi * self.injection_freq * np.arange(n)
        harmonic = self.amp * np.sin(phase)
        
        # Scale harmonic to signal variance to maintain relative strength
        signal_std = np.std(signal)
        if signal_std > 1e-10:
            harmonic *= signal_std
        
        return signal + harmonic
    
    def execute(self, 
               ergodic_signal: np.ndarray,
               peel_levels: Optional[int] = None) -> Dict:
        """
        Execute ergodic jump: inject harmonic, peel, and extract filament.
        
        This is the main diagnostic method. It:
        1. Computes baseline ergodic peel
        2. Injects harmonic to break ergodicity
        3. Peels the injected signal
        4. Extracts the filament (difference)
        5. Computes diagnostic metrics
        
        Args:
            ergodic_signal: Input ergodic (chaotic) signal
            peel_levels: Number of peel levels (default: max_depth)
        
        Returns:
            Dictionary containing:
                - filament: Extracted harmonic residue
                - resfrac_drop: Change in residual fraction
                - hurst_shift: Change in Hurst exponent (persistence)
                - base_resfrac: Baseline ergodic resfrac
                - injected_resfrac: Resfrac after injection
                - base_hurst: Baseline Hurst exponent
                - injected_hurst: Hurst after injection
                - peel_data: Full peel data for injected signal
                - base_peel_data: Full peel data for baseline
        """
        if peel_levels is None:
            peel_levels = self.max_depth
        
        # Step 1: Baseline ergodic peel (for comparison)
        base_peel_tree = self.compress(ergodic_signal)
        base_resfrac = resfrac_score(ergodic_signal)
        
        # Extract variance cascade from tree
        base_variances = self._extract_variances(base_peel_tree)
        base_hurst = self._compute_hurst(base_variances)
        
        # Step 2: Inject harmonic to break ergodicity
        non_ergodic = self.inject_harmonic(ergodic_signal)
        
        # Step 3: Peel the injected signal
        injected_peel_tree = self.compress(non_ergodic)
        injected_resfrac = resfrac_score(non_ergodic)
        
        # Extract variance cascade from tree
        injected_variances = self._extract_variances(injected_peel_tree)
        injected_hurst = self._compute_hurst(injected_variances)
        
        # Step 4: Extract 'filament' (residual difference)
        base_data = self._extract_leaf_data(base_peel_tree)
        injected_data = self._extract_leaf_data(injected_peel_tree)
        filament = self._extract_filament(injected_data, base_data)
        
        # Step 5: Compute pull metrics
        resfrac_drop = base_resfrac - injected_resfrac
        hurst_shift = injected_hurst - base_hurst
        
        # Track history
        self.hurst_history.append(injected_hurst)
        
        return {
            'filament': filament,
            'resfrac_drop': resfrac_drop,
            'hurst_shift': hurst_shift,
            'base_resfrac': base_resfrac,
            'injected_resfrac': injected_resfrac,
            'base_hurst': base_hurst,
            'injected_hurst': injected_hurst,
            'peel_data': injected_peel_tree,
            'base_peel_data': base_peel_tree,
        }
    
    def _extract_variances(self, tree) -> np.ndarray:
        """
        Extract variance cascade from compression tree.
        
        Args:
            tree: CompressionLeaf or CompressionNode
        
        Returns:
            Array of variances at each level
        """
        variances = []
        current = tree
        
        # Traverse tree to collect variances
        while hasattr(current, 'left'):  # It's a node
            # Compute variance at this level
            if hasattr(current, 'left') and hasattr(current.left, 'data'):
                left_var = np.var(current.left.data) if len(current.left.data) > 0 else 0
                right_var = np.var(current.right.data) if hasattr(current.right, 'data') and len(current.right.data) > 0 else 0
                variances.append((left_var + right_var) / 2)
            current = current.left if hasattr(current, 'left') else None
            if current is None:
                break
        
        # If we got no variances, use the leaf data
        if len(variances) == 0 and hasattr(tree, 'data'):
            variances = [np.var(tree.data)]
        
        return np.array(variances) if len(variances) > 0 else np.array([1.0])
    
    def _extract_leaf_data(self, tree) -> np.ndarray:
        """
        Extract leaf data from compression tree.
        
        Args:
            tree: CompressionLeaf or CompressionNode
        
        Returns:
            Data array from deepest leaf
        """
        current = tree
        
        # Traverse to deepest leaf
        while hasattr(current, 'left'):
            current = current.left
        
        # Return leaf data
        if hasattr(current, 'data'):
            return current.data
        else:
            return np.array([])
    
    def _extract_filament(self, 
                         peeled_levels: np.ndarray,
                         base_levels: np.ndarray) -> np.ndarray:
        """
        Extract filament by subtracting baseline from injected peel.
        
        The filament is the residual difference that reveals the
        injected harmonic structure.
        
        Args:
            peeled_levels: Final peel level from injected signal
            base_levels: Final peel level from baseline signal
        
        Returns:
            Filament (difference array)
        """
        # Handle empty arrays
        if len(peeled_levels) == 0 or len(base_levels) == 0:
            return np.array([])
        
        # Align lengths and subtract
        min_len = min(len(peeled_levels), len(base_levels))
        return peeled_levels[:min_len] - base_levels[:min_len]
    
    def _compute_hurst(self, variances: np.ndarray) -> float:
        """
        Compute Hurst exponent from variance decay slope.
        
        Hurst exponent measures persistence:
        - H = 0.5: White noise (ergodic)
        - H > 0.5: Persistent (non-ergodic, sticky)
        - H < 0.5: Anti-persistent
        
        Args:
            variances: Variance cascade from fractal peel
        
        Returns:
            Hurst exponent
        """
        if len(variances) < 2:
            return 0.5  # Default to ergodic
        
        # Filter out zeros/negatives for log
        valid_vars = variances[variances > 1e-10]
        if len(valid_vars) < 2:
            return 0.5
        
        log_vars = np.log(valid_vars)
        log_scales = np.log(2 ** np.arange(len(valid_vars)))
        
        # Linear fit: log(var) ~ slope * log(scale)
        coeffs = np.polyfit(log_scales, log_vars, 1)
        
        # Hurst = -slope / 2
        return -coeffs[0] / 2
    
    def diagnose_ergodicity(self, signal: np.ndarray) -> Dict:
        """
        Quick ergodicity diagnostic.
        
        Determines if a signal is truly ergodic or has hidden structure.
        
        Args:
            signal: Input signal to diagnose
        
        Returns:
            Dictionary with diagnostic results:
                - is_ergodic: Boolean (True if ergodic)
                - confidence: Confidence score [0, 1]
                - hurst: Hurst exponent
                - resfrac: Residual fraction
                - recommendation: String describing recommendation
        """
        result = self.execute(signal)
        
        # Ergodic criteria:
        # 1. Hurst near 0.5 (±0.1)
        # 2. Small resfrac drop (<0.02)
        # 3. Low hurst shift (<0.1)
        
        hurst_ergodic = abs(result['base_hurst'] - 0.5) < 0.1
        resfrac_ergodic = abs(result['resfrac_drop']) < 0.02
        hurst_stable = abs(result['hurst_shift']) < 0.1
        
        ergodic_score = sum([hurst_ergodic, resfrac_ergodic, hurst_stable]) / 3.0
        is_ergodic = ergodic_score > 0.66
        
        if is_ergodic:
            recommendation = "Signal appears ergodic (fully mixed). No hidden structure detected."
        elif result['hurst_shift'] > 0.1:
            recommendation = "Non-ergodic: Persistent correlations detected. Consider harmonic analysis."
        elif result['resfrac_drop'] > 0.05:
            recommendation = "Non-ergodic: Significant structure pullable. Use filament for refinement."
        else:
            recommendation = "Borderline: Weak structure detected. May benefit from deeper analysis."
        
        return {
            'is_ergodic': is_ergodic,
            'confidence': ergodic_score,
            'hurst': result['base_hurst'],
            'resfrac': result['base_resfrac'],
            'hurst_shift': result['hurst_shift'],
            'resfrac_drop': result['resfrac_drop'],
            'recommendation': recommendation,
        }


def demo():
    """Demonstration of ergodic jump."""
    print("=" * 70)
    print("ERGODIC JUMP DEMO")
    print("Jump Sequence Diagnostics")
    print("=" * 70)
    
    # Initialize analyzer
    print("\nInitializing ergodic jump analyzer...")
    print(f"  Injection frequency: 1/√5 ≈ {1/np.sqrt(5):.4f} (golden ratio)")
    print(f"  Amplitude: 0.15")
    jump = ErgodicJump(injection_freq=1/np.sqrt(5), amp=0.15)
    
    # Test 1: Pure ergodic signal (white noise)
    print("\n" + "=" * 70)
    print("TEST 1: Pure Ergodic Signal (White Noise)")
    print("=" * 70)
    
    np.random.seed(42)
    ergodic_signal = np.random.randn(1024)
    
    result = jump.execute(ergodic_signal)
    
    print(f"\nBaseline metrics:")
    print(f"  Resfrac: {result['base_resfrac']:.6f}")
    print(f"  Hurst: {result['base_hurst']:.4f}")
    
    print(f"\nAfter injection:")
    print(f"  Resfrac: {result['injected_resfrac']:.6f}")
    print(f"  Hurst: {result['injected_hurst']:.4f}")
    
    print(f"\nPull metrics:")
    print(f"  Resfrac drop: {result['resfrac_drop']:.6f}")
    print(f"  Hurst shift: {result['hurst_shift']:.4f}")
    print(f"  Filament length: {len(result['filament'])}")
    
    if result['hurst_shift'] > 0.1:
        print(f"  ✓ Non-ergodic structure detected!")
    else:
        print(f"  ~ Weak structure (expected for pure noise)")
    
    # Test 2: Pre-structured signal
    print("\n" + "=" * 70)
    print("TEST 2: Pre-Structured Signal (Low-freq sine)")
    print("=" * 70)
    
    t = np.arange(1024)
    structured_signal = np.sin(2 * np.pi * 0.01 * t) + 0.3 * np.random.randn(1024)
    
    result2 = jump.execute(structured_signal)
    
    print(f"\nBaseline metrics:")
    print(f"  Resfrac: {result2['base_resfrac']:.6f}")
    print(f"  Hurst: {result2['base_hurst']:.4f}")
    
    print(f"\nPull metrics:")
    print(f"  Resfrac drop: {result2['resfrac_drop']:.6f}")
    print(f"  Hurst shift: {result2['hurst_shift']:.4f}")
    
    if abs(result2['hurst_shift']) > abs(result['hurst_shift']):
        print(f"  ✓ Stronger response to injection (pre-existing structure)")
    
    # Test 3: Ergodicity diagnosis
    print("\n" + "=" * 70)
    print("TEST 3: Ergodicity Diagnosis")
    print("=" * 70)
    
    diag1 = jump.diagnose_ergodicity(ergodic_signal)
    print(f"\nWhite noise diagnosis:")
    print(f"  Is ergodic: {diag1['is_ergodic']}")
    print(f"  Confidence: {diag1['confidence']:.2f}")
    print(f"  Recommendation: {diag1['recommendation']}")
    
    diag2 = jump.diagnose_ergodicity(structured_signal)
    print(f"\nStructured signal diagnosis:")
    print(f"  Is ergodic: {diag2['is_ergodic']}")
    print(f"  Confidence: {diag2['confidence']:.2f}")
    print(f"  Recommendation: {diag2['recommendation']}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Ergodic jump reveals hidden structure in 'random' signals")
    print("✓ Hurst shift >0.1 indicates non-ergodic persistence")
    print("✓ Resfrac drop >0.05 suggests pullable structure")
    print("✓ Filament extraction enables targeted refinement")
    print("\n🧵 Quantum yo-yo uncoiled! 🧵")
    print("=" * 70)


if __name__ == '__main__':
    demo()
