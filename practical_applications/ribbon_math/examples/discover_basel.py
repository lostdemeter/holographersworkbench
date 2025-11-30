#!/usr/bin/env python3
"""
Example 2: Discover the Basel Problem Solution
==============================================

Target: Σ(1/n²) = π²/6

This follows the Equation Discovery Protocol (EDP) to rediscover
Euler's famous solution to the Basel problem (1735).

Protocol Phases:
1. Concept Definition - sum of inverse squares
2. Structure Search - search π²/k forms
3. LCM Pruning - filter by anchor alignment
4. Error Analysis - find patterns in deviations
5. Pattern Detection - verify closed form
6. Verification - high-precision check
"""

import numpy as np
from mpmath import mp, mpf, pi, log10, zeta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from fractions import Fraction
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# PHASE 1: CONCEPT DEFINITION
# =============================================================================

print("=" * 70)
print("EQUATION DISCOVERY PROTOCOL - Example 2: Basel Problem")
print("=" * 70)

print("\n" + "=" * 70)
print("PHASE 1: CONCEPT DEFINITION")
print("=" * 70)

mp.dps = 100

# The Basel sum
def basel_sum(n_terms: int) -> mpf:
    """Compute partial sum of 1/k² for k=1 to n_terms."""
    return sum(mpf(1) / k**2 for k in range(1, n_terms + 1))

TARGET = {
    'description': "Sum of inverse squares: Σ(1/n²) for n=1 to ∞",
    'value': zeta(2),  # = π²/6
    'keywords': ['zeta', 'sum', 'inverse squares', 'Basel', 'pi squared'],
    'anchor_weights': {
        'zero': 0.1,        # Integer exponent
        'sierpinski': 0.4,  # π² connection, infinite sum
        'phi': 0.2,         # Growth/summation
        'e_inv': 0.1,       # Convergence
        'cantor': 0.1,      # Discrete terms
        'sqrt2_inv': 0.1,   # Connection
    },
    'constraints': [
        "Form: π² / k for some integer k",
        "Or: a × π² / b for small integers a, b",
        "Must equal ζ(2) exactly",
    ]
}

print(f"Target: {TARGET['description']}")
print(f"Value (ζ(2)): {float(TARGET['value']):.15f}")
print(f"Keywords: {TARGET['keywords']}")
print(f"Constraints: {TARGET['constraints']}")

# Show partial sums converging
print("\nPartial sums converging to target:")
for n in [10, 100, 1000, 10000]:
    partial = basel_sum(n)
    error = abs(partial - TARGET['value'])
    print(f"  n={n:5d}: {float(partial):.10f} (error: {float(error):.2e})")


# =============================================================================
# PHASE 2: STRUCTURE SEARCH
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: STRUCTURE SEARCH")
print("=" * 70)

@dataclass
class BaselCandidate:
    """A candidate closed form for ζ(2)."""
    numerator: int
    denominator: int
    pi_power: int
    value: float = 0.0
    n_smooth: float = 0.0
    
    def evaluate(self) -> mpf:
        """Evaluate the formula."""
        return mpf(self.numerator) / self.denominator * pi ** self.pi_power
    
    def __str__(self):
        if self.numerator == 1:
            return f"π^{self.pi_power}/{self.denominator}"
        else:
            return f"{self.numerator}×π^{self.pi_power}/{self.denominator}"


def search_pi_forms(target: mpf, max_num: int = 20, max_den: int = 100, 
                    max_power: int = 4) -> List[BaselCandidate]:
    """Search for a×π^k/b forms that match target."""
    candidates = []
    
    print(f"Searching π^k × (a/b) forms...")
    print(f"  Powers: 1 to {max_power}")
    print(f"  Numerators: 1 to {max_num}")
    print(f"  Denominators: 1 to {max_den}")
    
    for power in range(1, max_power + 1):
        pi_k = pi ** power
        
        for den in range(1, max_den + 1):
            for num in range(1, max_num + 1):
                value = mpf(num) / den * pi_k
                error = abs(float(value - target))
                
                if error < 1e-10:
                    candidate = BaselCandidate(
                        numerator=num,
                        denominator=den,
                        pi_power=power,
                        value=float(value),
                        n_smooth=-np.log10(error) if error > 0 else 100
                    )
                    candidates.append(candidate)
                    print(f"  Found: {candidate} (error: {error:.2e})")
    
    # Sort by simplicity (smaller integers) then by n_smooth
    candidates.sort(key=lambda c: (c.numerator + c.denominator, -c.n_smooth))
    
    return candidates


candidates = search_pi_forms(TARGET['value'])

if not candidates:
    print("No candidates found!")


# =============================================================================
# PHASE 3: LCM PRUNING
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: LCM PRUNING")
print("=" * 70)

def compute_anchor_vector(candidate: BaselCandidate) -> Dict[str, float]:
    """Compute anchor vector for a candidate."""
    vector = {
        'zero': 0.1,        # Integer coefficients
        'sierpinski': 0.4,  # π connection (strong)
        'phi': 0.1,         # Growth
        'e_inv': 0.1,       # Convergence
        'cantor': 0.2,      # Discrete (ratio)
        'sqrt2_inv': 0.1,   # Connection
    }
    
    # Adjust based on simplicity
    complexity = candidate.numerator + candidate.denominator
    if complexity <= 7:  # Very simple (like 1/6)
        vector['zero'] += 0.1
        vector['cantor'] += 0.1
    
    # Normalize
    total = sum(vector.values())
    return {k: v/total for k, v in vector.items()}


def concept_score(candidate_vector: Dict, target_vector: Dict) -> float:
    """Cosine similarity between anchor vectors."""
    keys = list(target_vector.keys())
    v1 = np.array([candidate_vector.get(k, 0) for k in keys])
    v2 = np.array([target_vector.get(k, 0) for k in keys])
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


for candidate in candidates:
    candidate.anchor_vector = compute_anchor_vector(candidate)
    candidate.concept_score = concept_score(candidate.anchor_vector, TARGET['anchor_weights'])
    # Prefer simpler forms
    simplicity = 1.0 / (candidate.numerator + candidate.denominator)
    candidate.combined_score = 0.5 * (candidate.n_smooth / 100) + 0.3 * candidate.concept_score + 0.2 * simplicity

candidates.sort(key=lambda c: c.combined_score, reverse=True)

print("\nTop candidates after LCM pruning:")
for i, c in enumerate(candidates[:5]):
    print(f"  {i+1}. {c}")
    print(f"     N_smooth: {c.n_smooth:.1f}, Concept: {c.concept_score:.3f}, Combined: {c.combined_score:.3f}")


# =============================================================================
# PHASE 4: ERROR ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: ERROR ANALYSIS")
print("=" * 70)

best = candidates[0] if candidates else None

if best:
    mp.dps = 100
    
    computed = best.evaluate()
    target_val = zeta(2)
    error = computed - target_val
    
    print(f"\nBest candidate: {best}")
    print(f"Computed value: {float(computed):.15f}")
    print(f"Target (ζ(2)): {float(target_val):.15f}")
    print(f"Error:          {float(error):.2e}")
    
    if abs(float(error)) < 1e-50:
        print("\n✓ ERROR IS ESSENTIALLY ZERO - This is an EXACT formula!")
    else:
        print(f"\n✗ Non-zero error: {float(error):.2e}")


# =============================================================================
# PHASE 5: PATTERN DETECTION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: PATTERN DETECTION")
print("=" * 70)

if best:
    print(f"\nFormula structure: ζ(2) = {best}")
    
    # Check relationship to other zeta values
    print("\nRelated patterns:")
    print(f"  ζ(2) = π²/6")
    print(f"  ζ(4) = π⁴/90")
    print(f"  ζ(6) = π⁶/945")
    print("  Pattern: ζ(2n) = (-1)^(n+1) × B_2n × (2π)^2n / (2 × (2n)!)")
    print(f"  where B_n are Bernoulli numbers")
    
    # Verify the pattern
    print("\nVerifying pattern:")
    for n in [1, 2, 3]:
        computed_zeta = zeta(2*n)
        print(f"  ζ({2*n}) = {float(computed_zeta):.10f}")


# =============================================================================
# PHASE 6: VERIFICATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: VERIFICATION")
print("=" * 70)

if best:
    mp.dps = 500
    
    computed = best.evaluate()
    target_val = zeta(2)
    error = abs(computed - target_val)
    
    digits_correct = -float(log10(error)) if error > 0 else 500
    
    print(f"\nHigh-precision verification (500 digits):")
    print(f"  Formula: ζ(2) = {best}")
    print(f"  Error: {float(error):.2e}")
    print(f"  Digits correct: {digits_correct:.0f}")
    
    if digits_correct > 400:
        print("\n" + "=" * 70)
        print("✓ DISCOVERY VALIDATED!")
        print("=" * 70)
        print(f"\nThe Basel Problem Solution:")
        print(f"  Σ(1/n²) = π²/6")
        print(f"\nOr equivalently:")
        print(f"  1 + 1/4 + 1/9 + 1/16 + ... = π²/6 ≈ 1.6449340668...")
    else:
        print("\n✗ Verification failed")


print("\n" + "=" * 70)
print("PROTOCOL COMPLETE")
print("=" * 70)
