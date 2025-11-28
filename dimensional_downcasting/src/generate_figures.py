#!/usr/bin/env python3
"""
Generate Figures for the Paper
==============================

Creates publication-quality figures demonstrating the dimensional
downcasting approach for Riemann zeta zeros.

Run with: python src/generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpmath import mp, zetazero, siegelz, siegeltheta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import DimensionalDowncaster
from src.predictors import RamanujanPredictor, GeometricPredictor, gue_spacing

mp.dps = 50

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def figure_n_smooth_offset():
    """
    Figure 1: The N_smooth offset discovery.
    
    Shows that N_smooth(t_n) ≈ n - 0.5 for all zeros.
    """
    print("Generating Figure 1: N_smooth offset...")
    
    solver = DimensionalDowncaster()
    
    # Compute for many zeros
    n_values = list(range(1, 101))
    offsets = []
    
    for n in n_values:
        t_n = float(zetazero(n).imag)
        N_s = solver._N_smooth(t_n)
        offset = N_s - (n - 0.5)
        offsets.append(offset)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: N_smooth vs n
    ax1.scatter(n_values, [solver._N_smooth(float(zetazero(n).imag)) for n in n_values], 
                s=20, alpha=0.7, label=r'$N_{\mathrm{smooth}}(t_n)$')
    ax1.plot(n_values, [n - 0.5 for n in n_values], 'r--', lw=2, label=r'$n - 0.5$')
    ax1.set_xlabel('Zero index $n$')
    ax1.set_ylabel(r'$N_{\mathrm{smooth}}(t_n)$')
    ax1.set_title(r'Smooth Counting Function at Zeros')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Offset from n - 0.5
    ax2.scatter(n_values, offsets, s=20, alpha=0.7, c='green')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.axhline(y=np.mean(offsets), color='blue', linestyle=':', lw=2, 
                label=f'Mean = {np.mean(offsets):.4f}')
    ax2.fill_between(n_values, 
                     [np.mean(offsets) - np.std(offsets)] * len(n_values),
                     [np.mean(offsets) + np.std(offsets)] * len(n_values),
                     alpha=0.2, color='blue', label=f'±1σ = {np.std(offsets):.4f}')
    ax2.set_xlabel('Zero index $n$')
    ax2.set_ylabel(r'$N_{\mathrm{smooth}}(t_n) - (n - 0.5)$')
    ax2.set_title('Offset from $n - 0.5$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'n_smooth_offset.png'))
    plt.close()
    print("  Saved: figures/n_smooth_offset.png")


def figure_accuracy_comparison():
    """
    Figure 2: Accuracy comparison of different methods.
    """
    print("Generating Figure 2: Accuracy comparison...")
    
    ram = RamanujanPredictor()
    geo = GeometricPredictor()
    dim = DimensionalDowncaster()
    
    n_values = list(range(10, 201, 10))
    
    errors_ram = []
    errors_geo = []
    errors_dim = []
    
    for n in n_values:
        t_true = float(zetazero(n).imag)
        errors_ram.append(abs(ram.predict(n) - t_true))
        errors_geo.append(abs(geo.predict(n) - t_true))
        errors_dim.append(abs(dim.solve(n) - t_true))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(n_values, errors_ram, 'o-', label='Ramanujan Predictor', markersize=6)
    ax.semilogy(n_values, errors_geo, 's-', label='Geometric Predictor', markersize=6)
    ax.semilogy(n_values, [max(e, 1e-16) for e in errors_dim], '^-', 
                label='Dimensional Downcasting', markersize=6)
    
    # Reference lines
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='Quantum barrier (0.33)')
    ax.axhline(y=1e-6, color='gray', linestyle=':', alpha=0.5, label='HDR accuracy (10⁻⁶)')
    ax.axhline(y=1e-14, color='red', linestyle='--', alpha=0.5, label='Machine precision (10⁻¹⁴)')
    
    ax.set_xlabel('Zero index $n$')
    ax.set_ylabel('Absolute error')
    ax.set_title('Accuracy Comparison of Zero-Finding Methods')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-17, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_comparison.png'))
    plt.close()
    print("  Saved: figures/accuracy_comparison.png")


def figure_hardy_z_function():
    """
    Figure 3: The Hardy Z-function near a zero.
    """
    print("Generating Figure 3: Hardy Z-function...")
    
    n = 100
    t_zero = float(zetazero(n).imag)
    spacing = gue_spacing(t_zero)
    
    # Sample Z(t) around the zero
    t_range = np.linspace(t_zero - 2*spacing, t_zero + 2*spacing, 500)
    Z_values = [float(siegelz(t)) for t in t_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_range, Z_values, 'b-', lw=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=t_zero, color='red', linestyle='--', lw=2, label=f'Zero at $t_{{100}}$ = {t_zero:.6f}')
    
    # Mark the sign change
    ax.fill_between(t_range, Z_values, 0, where=[z > 0 for z in Z_values], 
                    alpha=0.3, color='green', label='Z(t) > 0')
    ax.fill_between(t_range, Z_values, 0, where=[z < 0 for z in Z_values], 
                    alpha=0.3, color='red', label='Z(t) < 0')
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$Z(t)$')
    ax.set_title(f'Hardy Z-function near the {n}th zero')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hardy_z_function.png'))
    plt.close()
    print("  Saved: figures/hardy_z_function.png")


def figure_gue_spacing():
    """
    Figure 4: GUE spacing vs actual spacing.
    """
    print("Generating Figure 4: GUE spacing...")
    
    # Compute actual spacings
    n_zeros = 200
    zeros = [float(zetazero(n).imag) for n in range(1, n_zeros + 1)]
    spacings = np.diff(zeros)
    
    # Compute GUE predictions
    gue_predictions = [gue_spacing(zeros[i]) for i in range(len(spacings))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Spacings
    ax1.scatter(range(1, len(spacings)+1), spacings, s=10, alpha=0.5, label='Actual spacing')
    ax1.plot(range(1, len(spacings)+1), gue_predictions, 'r-', lw=2, label='GUE prediction')
    ax1.set_xlabel('Zero index $n$')
    ax1.set_ylabel('Spacing $t_{n+1} - t_n$')
    ax1.set_title('Zero Spacings vs GUE Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Normalized spacings
    normalized = spacings / np.array(gue_predictions)
    ax2.hist(normalized, bins=30, density=True, alpha=0.7, edgecolor='black')
    ax2.axvline(x=1, color='red', linestyle='--', lw=2, label='Expected mean')
    ax2.axvline(x=np.mean(normalized), color='blue', linestyle=':', lw=2, 
                label=f'Actual mean = {np.mean(normalized):.3f}')
    ax2.set_xlabel('Normalized spacing')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Normalized Spacings')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'gue_spacing.png'))
    plt.close()
    print("  Saved: figures/gue_spacing.png")


def figure_convergence():
    """
    Figure 5: Convergence of the algorithm.
    """
    print("Generating Figure 5: Convergence...")
    
    n = 100
    t_true = float(zetazero(n).imag)
    
    solver = DimensionalDowncaster()
    t_guess = solver.predictor.predict(n)
    spacing = gue_spacing(t_guess)
    
    # Track convergence during bisection
    a = t_guess - 3 * spacing
    b = t_guess + 3 * spacing
    
    # Find the correct bracket first
    n_samples = 30
    t_samples = np.linspace(a, b, n_samples)
    Z_samples = [solver._hardy_Z(t) for t in t_samples]
    
    target_N = n - 0.5
    best_bracket = None
    best_diff = float('inf')
    
    for i in range(len(Z_samples) - 1):
        if Z_samples[i] * Z_samples[i+1] < 0:
            t_mid = (t_samples[i] + t_samples[i+1]) / 2
            N_mid = solver._N_smooth(t_mid)
            diff = abs(N_mid - target_N)
            if diff < best_diff:
                best_diff = diff
                best_bracket = (t_samples[i], t_samples[i+1])
    
    a, b = best_bracket
    
    # Track bisection
    errors = []
    widths = []
    
    Z_a = solver._hardy_Z(a)
    for iteration in range(60):
        mid = (a + b) / 2
        errors.append(abs(mid - t_true))
        widths.append(b - a)
        
        Z_mid = solver._hardy_Z(mid)
        if Z_a * Z_mid < 0:
            b = mid
        else:
            a = mid
            Z_a = Z_mid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Error vs iteration
    ax1.semilogy(range(len(errors)), errors, 'b-', lw=2)
    ax1.axhline(y=1e-14, color='red', linestyle='--', label='Machine precision')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Absolute error')
    ax1.set_title('Convergence of Bisection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Bracket width vs iteration
    ax2.semilogy(range(len(widths)), widths, 'g-', lw=2)
    ax2.axhline(y=1e-14, color='red', linestyle='--', label='Machine precision')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Bracket width')
    ax2.set_title('Bracket Width Reduction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'convergence.png'))
    plt.close()
    print("  Saved: figures/convergence.png")


def figure_algorithm_schematic():
    """
    Figure 6: Algorithm schematic showing the key steps.
    """
    print("Generating Figure 6: Algorithm schematic...")
    
    n = 100
    t_true = float(zetazero(n).imag)
    
    solver = DimensionalDowncaster()
    t_guess = solver.predictor.predict(n)
    spacing = gue_spacing(t_guess)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Initial guess
    ax1 = axes[0, 0]
    ax1.axvline(x=t_true, color='green', linestyle='-', lw=3, label=f'True zero')
    ax1.axvline(x=t_guess, color='blue', linestyle='--', lw=2, label=f'Initial guess')
    ax1.axvspan(t_guess - 3*spacing, t_guess + 3*spacing, alpha=0.2, color='blue', 
                label='Search bracket')
    ax1.set_xlim(t_guess - 4*spacing, t_guess + 4*spacing)
    ax1.set_xlabel('$t$')
    ax1.set_title('Step 1: Initial Guess & Bracket')
    ax1.legend(loc='upper right')
    ax1.set_yticks([])
    
    # Panel 2: Sign changes
    ax2 = axes[0, 1]
    a = t_guess - 3 * spacing
    b = t_guess + 3 * spacing
    t_range = np.linspace(a, b, 200)
    Z_values = [float(siegelz(t)) for t in t_range]
    
    ax2.plot(t_range, Z_values, 'b-', lw=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=t_true, color='green', linestyle='-', lw=2, label='True zero')
    
    # Mark sign changes
    for i in range(len(Z_values) - 1):
        if Z_values[i] * Z_values[i+1] < 0:
            ax2.axvline(x=t_range[i], color='red', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$Z(t)$')
    ax2.set_title('Step 2: Find Sign Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: N_smooth selection
    ax3 = axes[1, 0]
    
    # Find sign changes
    n_samples = 30
    t_samples = np.linspace(a, b, n_samples)
    Z_samples = [solver._hardy_Z(t) for t in t_samples]
    
    sign_changes = []
    for i in range(len(Z_samples) - 1):
        if Z_samples[i] * Z_samples[i+1] < 0:
            t_mid = (t_samples[i] + t_samples[i+1]) / 2
            N_mid = solver._N_smooth(t_mid)
            sign_changes.append((t_mid, N_mid))
    
    target_N = n - 0.5
    
    for t_mid, N_mid in sign_changes:
        diff = abs(N_mid - target_N)
        color = 'green' if diff == min(abs(sc[1] - target_N) for sc in sign_changes) else 'red'
        ax3.scatter([t_mid], [N_mid], s=100, c=color, zorder=5)
    
    ax3.axhline(y=target_N, color='blue', linestyle='--', lw=2, label=f'Target: $n - 0.5$ = {target_N}')
    ax3.set_xlabel('$t$')
    ax3.set_ylabel(r'$N_{\mathrm{smooth}}(t)$')
    ax3.set_title('Step 3: Select by $N_{smooth} \\approx n - 0.5$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Final refinement
    ax4 = axes[1, 1]
    
    # Narrow view around the zero
    t_narrow = np.linspace(t_true - 0.1, t_true + 0.1, 100)
    Z_narrow = [float(siegelz(t)) for t in t_narrow]
    
    ax4.plot(t_narrow, Z_narrow, 'b-', lw=2)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.axvline(x=t_true, color='green', linestyle='-', lw=2, label=f'Zero: {t_true:.10f}')
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('$Z(t)$')
    ax4.set_title('Step 4: Refine to Machine Precision')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'algorithm_schematic.png'))
    plt.close()
    print("  Saved: figures/algorithm_schematic.png")


def figure_scaling_analysis():
    """
    Figure 7: Time scaling with zero index.
    """
    print("Generating Figure 7: Scaling analysis...")
    
    import time
    
    solver = DimensionalDowncaster()
    
    n_values = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    times = []
    z_evals = []
    
    for n in n_values:
        solver.stats = {'Z_evals': 0, 'zeros_solved': 0}
        
        start = time.time()
        solver.solve(n)
        elapsed = time.time() - start
        
        times.append(elapsed)
        z_evals.append(solver.stats['Z_evals'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Time vs n
    ax1.loglog(n_values, times, 'bo-', markersize=8, lw=2)
    ax1.set_xlabel('Zero index $n$')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time vs Zero Index')
    ax1.grid(True, alpha=0.3)
    
    # Right: Z evaluations vs n
    ax2.semilogx(n_values, z_evals, 'go-', markersize=8, lw=2)
    ax2.axhline(y=np.mean(z_evals), color='red', linestyle='--', 
                label=f'Mean = {np.mean(z_evals):.0f}')
    ax2.set_xlabel('Zero index $n$')
    ax2.set_ylabel('Z-function evaluations')
    ax2.set_title('Z Evaluations vs Zero Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'scaling_analysis.png'))
    plt.close()
    print("  Saved: figures/scaling_analysis.png")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("GENERATING FIGURES FOR PAPER")
    print("=" * 60)
    
    figure_n_smooth_offset()
    figure_accuracy_comparison()
    figure_hardy_z_function()
    figure_gue_spacing()
    figure_convergence()
    figure_algorithm_schematic()
    figure_scaling_analysis()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
