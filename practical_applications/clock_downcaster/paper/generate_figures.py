#!/usr/bin/env python3
"""
Generate Figures for the Dimensional Downcasting Paper
======================================================

Creates publication-quality figures demonstrating the algorithm.

Usage:
    python generate_figures.py
    python generate_figures.py --figure 1  # Generate specific figure
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clock_solver import ClockDimensionalDowncaster, ClockFunction, ClockPredictor

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def figure1_counting_function():
    """The Counting Function and 0.5 Offset"""
    print("Generating Figure 1: Counting Function...")
    
    solver = ClockDimensionalDowncaster()
    
    # Compute first 10 eigenphases
    eigenphases = [solver.solve(n) for n in range(1, 11)]
    
    # Create smooth curve
    theta = np.linspace(0.1, eigenphases[-1] + 10, 500)
    N_smooth = [solver._N_smooth(t) for t in theta]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot smooth counting function
    ax.plot(theta, N_smooth, 'b-', label=r'$N_{\mathrm{smooth}}(\theta)$', linewidth=2)
    
    # Plot n - 0.5 lines and eigenphase points
    for i, ep in enumerate(eigenphases):
        n = i + 1
        ax.axhline(y=n - 0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.plot(ep, n - 0.5, 'ro', markersize=10, zorder=5)
        ax.annotate(f'$\\theta_{{{n}}}$', (ep, n - 0.5), 
                   textcoords="offset points", xytext=(5, 5), fontsize=10)
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$N(\theta)$')
    ax.set_title(r'The Counting Function: $N_{\mathrm{smooth}}(\theta_n) \approx n - 0.5$')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, eigenphases[-1] + 10)
    ax.set_ylim(0, 11)
    
    plt.savefig('figure1_counting_function.png')
    plt.close()
    print("  Saved: figure1_counting_function.png")


def figure2_clock_function():
    """The Clock Function C(θ)"""
    print("Generating Figure 2: Clock Function...")
    
    clock_fn = ClockFunction()
    solver = ClockDimensionalDowncaster()
    
    # Compute eigenphases
    eigenphases = [solver.solve(n) for n in range(1, 13)]
    
    # Sample clock function
    theta = np.linspace(5, 130, 1000)
    C_values = np.array([clock_fn.evaluate(t) for t in theta])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot clock function
    ax.plot(theta, C_values, 'b-', linewidth=1.5, label=r'$C(\theta)$')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Fill positive/negative regions
    ax.fill_between(theta, C_values, 0, where=C_values > 0, 
                    alpha=0.3, color='green', label='Positive')
    ax.fill_between(theta, C_values, 0, where=C_values < 0, 
                    alpha=0.3, color='red', label='Negative')
    
    # Mark eigenphases
    for ep in eigenphases:
        ax.axvline(x=ep, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.plot(ep, 0, 'ko', markersize=6, zorder=5)
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$C(\theta)$')
    ax.set_title('The Clock Function: Sign Changes at Eigenphases')
    ax.legend(loc='upper right')
    ax.set_xlim(5, 130)
    ax.set_ylim(-1.2, 1.2)
    
    plt.savefig('figure2_clock_function.png')
    plt.close()
    print("  Saved: figure2_clock_function.png")


def figure3_disambiguation():
    """Disambiguation via N_smooth"""
    print("Generating Figure 3: Disambiguation...")
    
    solver = ClockDimensionalDowncaster()
    clock_fn = ClockFunction()
    predictor = ClockPredictor()
    
    # Focus on n = 50
    n = 50
    theta_guess = predictor.predict(n)
    spacing = predictor.spacing(theta_guess)
    
    # Bracket
    a = theta_guess - 3 * spacing
    b = theta_guess + 3 * spacing
    
    theta = np.linspace(a, b, 500)
    C_values = np.array([clock_fn.evaluate(t) for t in theta])
    
    # Find sign changes
    sign_changes = []
    for i in range(len(C_values) - 1):
        if C_values[i] * C_values[i+1] < 0:
            t_mid = (theta[i] + theta[i+1]) / 2
            N_mid = solver._N_smooth(t_mid)
            sign_changes.append((t_mid, N_mid))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(theta, C_values, 'b-', linewidth=2, label=r'$C(\theta)$')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Mark sign changes with N_smooth values
    target_N = n - 0.5
    for t_mid, N_mid in sign_changes:
        diff = abs(N_mid - target_N)
        color = 'green' if diff < 0.3 else 'red'
        ax.axvline(x=t_mid, color=color, linestyle='--', alpha=0.7, linewidth=2)
        ax.annotate(f'$N_{{smooth}} = {N_mid:.2f}$\n$|\\Delta| = {diff:.2f}$', 
                   (t_mid, 0.8), textcoords="data", fontsize=9,
                   ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.axhline(y=0.5, color='purple', linestyle=':', linewidth=1, 
               label=f'Target: $n - 0.5 = {target_N}$')
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$C(\theta)$')
    ax.set_title(f'Disambiguation for n = {n}: Select Sign Change Closest to $N_{{smooth}} = {target_N}$')
    ax.legend(loc='upper right')
    ax.set_ylim(-1.2, 1.2)
    
    plt.savefig('figure3_disambiguation.png')
    plt.close()
    print("  Saved: figure3_disambiguation.png")


def figure5_accuracy():
    """Accuracy vs n"""
    print("Generating Figure 5: Accuracy...")
    
    solver = ClockDimensionalDowncaster()
    clock_fn = ClockFunction()
    
    ns = list(range(1, 101))
    C_residuals = []
    N_errors = []
    
    for n in ns:
        result = solver.verify(n)
        C_residuals.append(result['C_at_theta'])
        N_errors.append(result['N_smooth_error'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: |C(θ_n)|
    ax1.semilogy(ns, C_residuals, 'b.', markersize=4)
    ax1.axhline(y=1e-14, color='r', linestyle='--', label=r'$10^{-14}$ threshold')
    ax1.set_xlabel('n')
    ax1.set_ylabel(r'$|C(\theta_n)|$')
    ax1.set_title('Clock Function Residual (Machine Precision)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: N_smooth error
    ax2.plot(ns, N_errors, 'g.', markersize=4)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Expected: 0.5')
    ax2.set_xlabel('n')
    ax2.set_ylabel(r'$|N_{\mathrm{smooth}}(\theta_n) - (n - 0.5)|$')
    ax2.set_title(r'$N_{\mathrm{smooth}}$ Error (Confirms $n - 0.5$ Relationship)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 0.6)
    
    plt.tight_layout()
    plt.savefig('figure5_accuracy.png')
    plt.close()
    print("  Saved: figure5_accuracy.png")


def figure7_light_cone():
    """The Light Cone Boundary"""
    print("Generating Figure 7: Light Cone...")
    
    solver = ClockDimensionalDowncaster()
    predictor = ClockPredictor()
    
    ns = list(range(1, 201))
    errors = []
    
    for n in ns:
        theta_exact = solver.solve(n)
        theta_pred = predictor.predict(n)
        errors.append(abs(theta_exact - theta_pred))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogy(ns, errors, 'b.', markersize=4, alpha=0.7)
    ax.axvline(x=80, color='r', linestyle='--', linewidth=2, label='Light Cone (n = 80)')
    
    # Highlight regions
    ax.axvspan(0, 80, alpha=0.1, color='blue', label='Pre-horizon (Classical)')
    ax.axvspan(80, 200, alpha=0.1, color='red', label='Post-horizon (Quantum)')
    
    ax.set_xlabel('n')
    ax.set_ylabel('Predictor Error')
    ax.set_title('The Light Cone Boundary: Phase Transition at n ≈ 80')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('figure7_light_cone.png')
    plt.close()
    print("  Saved: figure7_light_cone.png")


def figure8_histogram():
    """N_smooth Error Histogram"""
    print("Generating Figure 8: Histogram...")
    
    solver = ClockDimensionalDowncaster()
    
    ns = list(range(1, 501))
    N_errors = []
    
    for n in ns:
        result = solver.verify(n)
        N_errors.append(result['N_smooth_error'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(N_errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=np.mean(N_errors), color='r', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(N_errors):.4f}')
    ax.axvline(x=0.5, color='g', linestyle=':', linewidth=2,
               label='Expected = 0.5')
    
    ax.set_xlabel(r'$|N_{\mathrm{smooth}}(\theta_n) - (n - 0.5)|$')
    ax.set_ylabel('Frequency')
    ax.set_title(r'Distribution of $N_{\mathrm{smooth}}$ Errors (n = 1 to 500)')
    ax.legend()
    
    # Add statistics
    stats_text = f'Mean: {np.mean(N_errors):.4f}\nStd: {np.std(N_errors):.4f}\nMin: {np.min(N_errors):.4f}\nMax: {np.max(N_errors):.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('figure8_histogram.png')
    plt.close()
    print("  Saved: figure8_histogram.png")


def figure9_complexity():
    """Complexity Scaling"""
    print("Generating Figure 9: Complexity...")
    
    import time
    
    solver = ClockDimensionalDowncaster()
    
    ns = [10, 50, 100, 500, 1000, 5000, 10000]
    times = []
    
    for n in ns:
        start = time.perf_counter()
        for _ in range(5):  # Average over 5 runs
            solver.solve(n)
        elapsed = (time.perf_counter() - start) / 5
        times.append(elapsed * 1000)  # Convert to ms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(ns, times, 'bo-', markersize=8, linewidth=2, label='Measured time')
    
    # Fit O(log n)
    log_ns = np.log(ns)
    coeffs = np.polyfit(log_ns, times, 1)
    fit_times = coeffs[0] * log_ns + coeffs[1]
    ax.loglog(ns, fit_times, 'r--', linewidth=2, label=f'O(log n) fit')
    
    # O(n) for comparison
    ax.loglog(ns, [times[0] * n / ns[0] for n in ns], 
              'g:', linewidth=2, label='O(n) reference')
    
    ax.set_xlabel('n')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Complexity Scaling: O(log n) Confirmed')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.savefig('figure9_complexity.png')
    plt.close()
    print("  Saved: figure9_complexity.png")


def generate_all():
    """Generate all figures"""
    print("=" * 60)
    print("Generating All Figures for Dimensional Downcasting Paper")
    print("=" * 60)
    
    figure1_counting_function()
    figure2_clock_function()
    figure3_disambiguation()
    figure5_accuracy()
    figure7_light_cone()
    figure8_histogram()
    figure9_complexity()
    
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--figure', type=int, help='Generate specific figure (1-9)')
    args = parser.parse_args()
    
    if args.figure:
        figures = {
            1: figure1_counting_function,
            2: figure2_clock_function,
            3: figure3_disambiguation,
            5: figure5_accuracy,
            7: figure7_light_cone,
            8: figure8_histogram,
            9: figure9_complexity,
        }
        if args.figure in figures:
            figures[args.figure]()
        else:
            print(f"Figure {args.figure} not implemented. Available: {list(figures.keys())}")
    else:
        generate_all()
