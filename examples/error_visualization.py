"""
Example: Using the Error Pattern Visualizer
============================================

Demonstrates automatic error pattern discovery and correction suggestions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from workbench.analysis.errors import ErrorPatternAnalyzer


# ============================================================================
# Example 1: Basic Error Analysis
# ============================================================================

def example_1_basic_analysis():
    """Analyze error with known pattern (sin + linear trend)."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Error Analysis")
    print("=" * 70)
    
    # Create synthetic data with known error pattern
    x = np.linspace(0, 10, 100)
    actual = np.sin(x) + 0.1 * x  # Sin + linear trend
    predicted = np.sin(x)  # Missing linear term
    
    # Analyze errors
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Sin + Linear")
    report = analyzer.analyze_all()
    
    # Print summary
    report.print_summary()
    
    # Show top suggestion
    if report.suggestions:
        print("\nTop correction code:")
        print(report.suggestions[0].code_snippet)
    
    print()


# ============================================================================
# Example 2: Spectral Pattern Detection
# ============================================================================

def example_2_spectral_patterns():
    """Detect spectral harmonics in error."""
    print("=" * 70)
    print("EXAMPLE 2: Spectral Pattern Detection")
    print("=" * 70)
    
    # Create data with multiple harmonics
    x = np.linspace(0, 10, 200)
    actual = np.sin(2 * np.pi * 1.0 * x) + 0.5 * np.sin(2 * np.pi * 3.0 * x)
    predicted = np.sin(2 * np.pi * 1.0 * x)  # Missing 3rd harmonic
    
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Multiple Harmonics")
    
    # Analyze spectral patterns
    spectral = analyzer.analyze_spectral(n_harmonics=5)
    print(f"Detected {len(spectral.frequencies)} harmonics")
    print(f"Explained variance: {spectral.explained_variance:.1%}")
    print(f"Dominant frequencies: {spectral.frequencies[:3]}")
    
    print()


# ============================================================================
# Example 3: Apply Corrections
# ============================================================================

def example_3_apply_corrections():
    """Apply corrections and measure improvement."""
    print("=" * 70)
    print("EXAMPLE 3: Apply Corrections")
    print("=" * 70)
    
    # Create data with polynomial error
    x = np.linspace(1, 10, 100)
    actual = x ** 2 + 10
    predicted = x ** 2  # Missing constant offset
    
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Polynomial Error")
    
    print(f"Original RMSE: {analyzer.rmse:.6f}")
    
    # Get suggestions
    suggestions = analyzer.suggest_corrections(top_k=1)
    
    if suggestions:
        # Apply best correction
        corrected_analyzer = analyzer.apply_correction(suggestions[0])
        print(f"Corrected RMSE: {corrected_analyzer.rmse:.6f}")
        improvement = (1 - corrected_analyzer.rmse / analyzer.rmse) * 100
        print(f"Improvement: {improvement:.1f}%")
    
    print()


# ============================================================================
# Example 4: Recursive Refinement
# ============================================================================

def example_4_recursive_refinement():
    """Recursively apply corrections until convergence."""
    print("=" * 70)
    print("EXAMPLE 4: Recursive Refinement")
    print("=" * 70)
    
    # Create data with multiple error patterns
    x = np.linspace(0, 10, 150)
    actual = np.sin(x) + 0.05 * x ** 2 + 0.2 * np.sin(2 * np.pi * 2.0 * x)
    predicted = np.sin(x)  # Missing polynomial and harmonic
    
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Complex Error")
    
    # Apply recursive refinement
    history = analyzer.recursive_refinement(max_depth=5, improvement_threshold=0.01)
    
    print(f"Initial RMSE: {history.initial_rmse:.6f}")
    print(f"Final RMSE:   {history.final_rmse:.6f}")
    print(f"Improvement:  {history.improvement:.1%}")
    print(f"Depth:        {history.depth} corrections")
    
    print("\nApplied corrections:")
    for i, corr in enumerate(history.corrections_applied, 1):
        print(f"  {i}. {corr.description}")
        print(f"     RMSE after: {history.rmse_history[i]:.6f}")
    
    print()


# ============================================================================
# Example 5: Scale-Dependent Errors
# ============================================================================

def example_5_scale_dependence():
    """Detect scale-dependent error patterns."""
    print("=" * 70)
    print("EXAMPLE 5: Scale-Dependent Errors")
    print("=" * 70)
    
    # Create data with scale-dependent error
    x = np.linspace(1, 100, 200)
    actual = np.log(x) + 0.1 * x ** 0.5  # Log + sqrt
    predicted = np.log(x)  # Missing sqrt term
    
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Scale-Dependent")
    
    # Analyze scale dependence
    scale = analyzer.analyze_scale_dependence(n_bins=10)
    
    if scale.scale_function:
        model = scale.scale_params.get('model', 'unknown')
        r2 = scale.scale_params.get('r2', 0.0)
        print(f"Detected scale pattern: {model}")
        print(f"R²: {r2:.4f}")
    else:
        print("No significant scale dependence detected")
    
    print()


# ============================================================================
# Example 6: Code Generation
# ============================================================================

def example_6_code_generation():
    """Generate correction code snippets."""
    print("=" * 70)
    print("EXAMPLE 6: Code Generation")
    print("=" * 70)
    
    # Simple example
    x = np.linspace(0, 5, 50)
    actual = 2 * x + 5
    predicted = 2 * x  # Missing constant
    
    analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Linear Offset")
    report = analyzer.analyze_all()
    
    print("Generated correction code:\n")
    for i, sug in enumerate(report.suggestions, 1):
        print(f"Correction {i}: {sug.description}")
        print("-" * 50)
        print(sug.code_snippet)
        print()
    
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ERROR PATTERN VISUALIZER EXAMPLES" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_1_basic_analysis()
    example_2_spectral_patterns()
    example_3_apply_corrections()
    example_4_recursive_refinement()
    example_5_scale_dependence()
    example_6_code_generation()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
