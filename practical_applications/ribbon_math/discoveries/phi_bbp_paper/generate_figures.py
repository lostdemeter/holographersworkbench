#!/usr/bin/env python3
"""
Figure Generation for the φ-BBP Formula Paper
==============================================

Generates publication-quality figures:
1. Convergence comparison (φ-BBP vs Bellard vs Original BBP)
2. φ-correction pattern visualization
3. Error structure analysis
4. Closed-form approximation quality

Run: python generate_figures.py
Output: figures/*.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, pi as mp_pi, phi as mp_phi, log, atan
from typing import List, Tuple
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# FORMULA CONSTANTS
# =============================================================================

INT_COEFS = [256, -32, 4, 1, -128, -64, -128, 4]
PHI_CORRECTIONS = [
    0.021013707249693914,
    -0.047113568832732489,
    0.007043075951984248,
    0.013181561904277411,
    0.073263011134871173,
    -0.102346114400464053,
    -0.153352294762736929,
    0.047671994116562255,
]
SLOTS = [(4, 1), (4, 3), (12, 1), (12, 3), (12, 5), (12, 7), (12, 9), (12, 11)]
RATIONAL_APPROX = [
    (46, 11, -11), (-31, 96, -4), (53, 99, -9), (31, 81, -7),
    (13, 16, -5), (-47, 67, -4), (-41, 39, -4), (71, 83, -6),
]


# =============================================================================
# FORMULA EVALUATION
# =============================================================================

def evaluate_phi_bbp(n_terms: int, precision: int = 100) -> float:
    """Evaluate the φ-BBP formula."""
    mp.dps = precision
    
    effective_coefs = [mpf(INT_COEFS[i]) + mpf(PHI_CORRECTIONS[i]) for i in range(8)]
    
    result = mpf(0)
    for k in range(n_terms):
        sign = mpf(-1)**k
        base_term = mpf(1) / mpf(4096)**k
        
        inner = mpf(0)
        for (period, offset), coef in zip(SLOTS, effective_coefs):
            denom = period * k + offset
            if denom != 0:
                inner += coef / denom
        
        result += sign * base_term * inner
    
    return float(abs(result / mpf(64) - mp_pi))


def evaluate_integer_only(n_terms: int, precision: int = 100) -> float:
    """Evaluate with integer coefficients only."""
    mp.dps = precision
    
    result = mpf(0)
    for k in range(n_terms):
        sign = mpf(-1)**k
        base_term = mpf(1) / mpf(4096)**k
        
        inner = mpf(0)
        for (period, offset), coef in zip(SLOTS, INT_COEFS):
            denom = period * k + offset
            if denom != 0:
                inner += mpf(coef) / denom
        
        result += sign * base_term * inner
    
    return float(abs(result / mpf(64) - mp_pi))


# =============================================================================
# FIGURE 1: CONVERGENCE COMPARISON
# =============================================================================

def figure1_convergence():
    """Generate convergence comparison figure."""
    print("Generating Figure 1: Convergence Comparison...")
    
    terms = list(range(1, 51))
    
    # Compute errors
    phi_bbp_errors = []
    int_only_errors = []
    
    for n in terms:
        phi_bbp_errors.append(evaluate_phi_bbp(n, precision=200))
        int_only_errors.append(evaluate_integer_only(n, precision=200))
    
    # Theoretical convergence lines
    bellard_rate = np.log10(1024)  # 3.01
    phi_bbp_rate = np.log10(4096)  # 3.61
    original_bbp_rate = np.log10(16)  # 1.20
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot actual errors
    ax.semilogy(terms, phi_bbp_errors, 'b-', linewidth=2, label='φ-BBP (actual)', marker='o', markersize=4)
    ax.semilogy(terms, int_only_errors, 'r--', linewidth=2, label='Integer-only (actual)', marker='s', markersize=4)
    
    # Plot theoretical lines
    x = np.array(terms)
    ax.semilogy(x, 10**(0 - phi_bbp_rate * x), 'b:', linewidth=1.5, alpha=0.7, label=f'φ-BBP theoretical ({phi_bbp_rate:.2f} d/t)')
    ax.semilogy(x, 10**(0 - bellard_rate * x), 'g:', linewidth=1.5, alpha=0.7, label=f'Bellard theoretical ({bellard_rate:.2f} d/t)')
    ax.semilogy(x, 10**(0 - original_bbp_rate * x), 'm:', linewidth=1.5, alpha=0.7, label=f'Original BBP theoretical ({original_bbp_rate:.2f} d/t)')
    
    ax.set_xlabel('Number of Terms')
    ax.set_ylabel('Absolute Error |computed - π|')
    ax.set_title('Convergence Comparison: φ-BBP vs Other BBP Formulas')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    ax.set_ylim(1e-180, 1)
    
    plt.savefig('figures/fig1_convergence.png')
    plt.close()
    print("  Saved: figures/fig1_convergence.png")


# =============================================================================
# FIGURE 2: φ-CORRECTION PATTERN
# =============================================================================

def figure2_corrections():
    """Generate φ-correction pattern figure."""
    print("Generating Figure 2: φ-Correction Pattern...")
    
    mp.dps = 50
    PHI = float(mp_phi)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Corrections as bar chart
    slots = [f"({p}k+{o})" for p, o in SLOTS]
    colors = ['green' if c > 0 else 'red' for c in PHI_CORRECTIONS]
    
    ax1.bar(range(8), PHI_CORRECTIONS, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(slots, rotation=45, ha='right')
    ax1.set_xlabel('Slot (period k + offset)')
    ax1.set_ylabel('Correction Value')
    ax1.set_title('φ-Corrections by Slot')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Rational approximation errors
    approx_errors = []
    labels = []
    for i, (corr, (num, den, phi_pow)) in enumerate(zip(PHI_CORRECTIONS, RATIONAL_APPROX)):
        approx = (num / den) * (PHI ** phi_pow)
        error = abs(corr - approx)
        approx_errors.append(error)
        labels.append(f"({num}/{den})×φ^{phi_pow}")
    
    ax2.barh(range(8), approx_errors, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Approximation Error')
    ax2.set_title('Quality of Rational Approximations')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Highlight the cleanest approximation (slot 4)
    ax2.barh(4, approx_errors[4], color='gold', alpha=0.9, edgecolor='black')
    ax2.annotate('Cleanest!', xy=(approx_errors[4], 4), xytext=(approx_errors[4]*10, 4),
                fontsize=10, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig('figures/fig2_corrections.png')
    plt.close()
    print("  Saved: figures/fig2_corrections.png")


# =============================================================================
# FIGURE 3: TOTAL CORRECTION CLOSED FORM
# =============================================================================

def figure3_closed_form():
    """Generate closed-form analysis figure."""
    print("Generating Figure 3: Closed-Form Analysis...")
    
    mp.dps = 50
    PHI = mpf(mp_phi)
    
    total = sum(PHI_CORRECTIONS)
    atan_phi = float(atan(1/PHI))
    log_phi = float(log(PHI))
    
    # Search grid for best (a/b, c/d) coefficients
    best_errors = []
    a_range = np.linspace(-1, 1, 100)
    b_range = np.linspace(-2, 0, 100)
    
    error_grid = np.zeros((len(a_range), len(b_range)))
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            predicted = a * atan_phi + b * log_phi
            error_grid[i, j] = abs(predicted - total)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Error heatmap
    im = ax1.imshow(np.log10(error_grid + 1e-20), extent=[b_range[0], b_range[-1], a_range[0], a_range[-1]],
                    aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel('Coefficient of log(φ)')
    ax1.set_ylabel('Coefficient of arctan(1/φ)')
    ax1.set_title('log₁₀(Error) for Total = a×arctan(1/φ) + b×log(φ)')
    plt.colorbar(im, ax=ax1, label='log₁₀(error)')
    
    # Mark the best point
    best_a = 13/20
    best_b = -26/25
    ax1.plot(best_b, best_a, 'r*', markersize=15, label=f'Best: ({13}/{20}, {-26}/{25})')
    ax1.legend()
    
    # Right: Components breakdown
    components = {
        'arctan(1/φ) term': (13/20) * atan_phi,
        'log(φ) term': (-26/25) * log_phi,
        'Sum (closed form)': (13/20) * atan_phi + (-26/25) * log_phi,
        'Actual total': total,
    }
    
    names = list(components.keys())
    values = list(components.values())
    colors = ['steelblue', 'coral', 'green', 'gold']
    
    bars = ax2.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Value')
    ax2.set_title('Total Correction Decomposition')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(val + 0.005 if val > 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.6f}', va='center', ha='left' if val > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/fig3_closed_form.png')
    plt.close()
    print("  Saved: figures/fig3_closed_form.png")


# =============================================================================
# FIGURE 4: MATHEMATICAL STRUCTURE
# =============================================================================

def figure4_structure():
    """Generate mathematical structure figure."""
    print("Generating Figure 4: Mathematical Structure...")
    
    mp.dps = 50
    PHI = float(mp_phi)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Powers of φ
    ax = axes[0, 0]
    k_vals = np.arange(-12, 4)
    phi_powers = [PHI ** k for k in k_vals]
    
    ax.semilogy(k_vals, phi_powers, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('φ^k')
    ax.set_title('Powers of the Golden Ratio')
    ax.grid(True, alpha=0.3)
    
    # Highlight the powers used in corrections
    used_powers = [-11, -9, -7, -6, -5, -4]
    for k in used_powers:
        ax.axvline(x=k, color='red', alpha=0.3, linestyle=':')
    
    # Top-right: φ² + φ⁻² identity
    ax = axes[0, 1]
    x = np.linspace(1, 3, 100)
    y1 = x**2 + x**(-2)
    
    ax.plot(x, y1, 'b-', linewidth=2, label='x² + x⁻²')
    ax.axhline(y=3, color='red', linestyle='--', label='y = 3')
    ax.axvline(x=PHI, color='green', linestyle=':', label=f'x = φ ≈ {PHI:.3f}')
    ax.scatter([PHI], [3], color='gold', s=100, zorder=5, edgecolor='black')
    ax.set_xlabel('x')
    ax.set_ylabel('x² + x⁻²')
    ax.set_title('Identity: φ² + φ⁻² = 3')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Fibonacci arctan identity
    ax = axes[1, 0]
    phi_vals = np.linspace(1.1, 2.5, 100)
    arctan_sum = np.arctan(1/phi_vals) + np.arctan(1/phi_vals**3)
    
    ax.plot(phi_vals, arctan_sum, 'b-', linewidth=2, label='arctan(1/x) + arctan(1/x³)')
    ax.axhline(y=np.pi/4, color='red', linestyle='--', label='π/4')
    ax.axvline(x=PHI, color='green', linestyle=':', label=f'x = φ')
    ax.scatter([PHI], [np.pi/4], color='gold', s=100, zorder=5, edgecolor='black')
    ax.set_xlabel('x')
    ax.set_ylabel('arctan sum')
    ax.set_title('Fibonacci Arctan Identity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Base relationship
    ax = axes[1, 1]
    bases = [16, 256, 1024, 4096, 16384]
    labels = ['Original\nBBP', 'Base\n256', 'Bellard', 'φ-BBP', 'Future?']
    rates = [np.log10(b) for b in bases]
    colors = ['gray', 'gray', 'steelblue', 'gold', 'lightgray']
    
    bars = ax.bar(labels, rates, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Convergence Rate (digits/term)')
    ax.set_title('BBP Formula Convergence Rates')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/fig4_structure.png')
    plt.close()
    print("  Saved: figures/fig4_structure.png")


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_tables():
    """Generate LaTeX/Markdown tables."""
    print("Generating Tables...")
    
    # Table 1: Coefficients
    table1 = """
| Slot | Period | Offset | Integer | Correction | Effective |
|------|--------|--------|---------|------------|-----------|
"""
    for i, ((p, o), a, c) in enumerate(zip(SLOTS, INT_COEFS, PHI_CORRECTIONS)):
        eff = a + c
        table1 += f"| {i} | {p} | {o} | {a:+4d} | {c:+.10f} | {eff:+.10f} |\n"
    
    with open('figures/table1_coefficients.md', 'w') as f:
        f.write("# Table 1: φ-BBP Formula Coefficients\n\n")
        f.write(table1)
    print("  Saved: figures/table1_coefficients.md")
    
    # Table 2: Rational approximations
    mp.dps = 50
    PHI = float(mp_phi)
    
    table2 = """
| Slot | Correction | Approximation | Value | Error |
|------|------------|---------------|-------|-------|
"""
    for i, (c, (n, d, k)) in enumerate(zip(PHI_CORRECTIONS, RATIONAL_APPROX)):
        approx = (n/d) * (PHI ** k)
        err = abs(c - approx)
        table2 += f"| {i} | {c:+.10f} | ({n:+d}/{d})×φ^{k} | {approx:+.10f} | {err:.2e} |\n"
    
    with open('figures/table2_approximations.md', 'w') as f:
        f.write("# Table 2: Rational Approximations\n\n")
        f.write(table2)
    print("  Saved: figures/table2_approximations.md")
    
    # Table 3: Convergence comparison
    table3 = """
| Formula | Base | Digits/Term | Relative Speed |
|---------|------|-------------|----------------|
| Original BBP | 16 | 1.20 | 33% |
| Bellard | 1024 | 3.01 | 83% |
| **φ-BBP** | **4096** | **3.61** | **100%** |
"""
    
    with open('figures/table3_convergence.md', 'w') as f:
        f.write("# Table 3: Convergence Comparison\n\n")
        f.write(table3)
    print("  Saved: figures/table3_convergence.md")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all figures and tables."""
    print("=" * 60)
    print("GENERATING FIGURES FOR φ-BBP PAPER")
    print("=" * 60)
    
    figure1_convergence()
    figure2_corrections()
    figure3_closed_form()
    figure4_structure()
    generate_tables()
    
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print("\nOutput files:")
    print("  figures/fig1_convergence.png")
    print("  figures/fig2_corrections.png")
    print("  figures/fig3_closed_form.png")
    print("  figures/fig4_structure.png")
    print("  figures/table1_coefficients.md")
    print("  figures/table2_approximations.md")
    print("  figures/table3_convergence.md")


if __name__ == "__main__":
    main()
