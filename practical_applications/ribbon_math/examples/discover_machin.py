#!/usr/bin/env python3
"""
Example 1: Discover Machin's Formula for π
==========================================

Target: π/4 = 4×arctan(1/5) - arctan(1/239)

This follows the Equation Discovery Protocol (EDP) to rediscover
Machin's famous formula from 1706.

Protocol Phases:
1. Concept Definition - arctan-based π formula
2. Structure Search - search arctan(1/n) combinations
3. LCM Pruning - filter by anchor alignment
4. Error Analysis - find patterns in deviations
5. Pattern Detection - verify closed form
6. Verification - high-precision check
"""

import numpy as np
from mpmath import mp, mpf, pi, atan, log10
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
print("EQUATION DISCOVERY PROTOCOL - Example 1: Machin's Formula")
print("=" * 70)

print("\n" + "=" * 70)
print("PHASE 1: CONCEPT DEFINITION")
print("=" * 70)

TARGET = {
    'description': "Arctan-based formula for π/4",
    'value': mp.pi / 4,
    'keywords': ['pi', 'arctan', 'inverse tangent', 'Machin'],
    'anchor_weights': {
        'zero': 0.1,        # Integer coefficients
        'sierpinski': 0.3,  # π itself
        'phi': 0.1,         # Growth (less relevant here)
        'e_inv': 0.1,       # Series convergence
        'cantor': 0.3,      # Discrete terms (1/5, 1/239)
        'sqrt2_inv': 0.1,   # Connections
    },
    'constraints': [
        "Form: Σ aᵢ × arctan(1/nᵢ)",
        "Coefficients aᵢ are small integers",
        "Arguments nᵢ are positive integers",
        "Maximum 3 terms",
    ]
}

print(f"Target: {TARGET['description']}")
print(f"Value: {float(TARGET['value']):.15f}")
print(f"Keywords: {TARGET['keywords']}")
print(f"Anchor weights: {TARGET['anchor_weights']}")
print(f"Constraints: {TARGET['constraints']}")


# =============================================================================
# PHASE 2: STRUCTURE SEARCH
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: STRUCTURE SEARCH")
print("=" * 70)

mp.dps = 50  # 50 decimal places for search

@dataclass
class ArctanCandidate:
    """A candidate arctan formula."""
    terms: List[Tuple[int, int]]  # [(coefficient, argument), ...]
    value: float = 0.0
    n_smooth: float = 0.0
    
    def evaluate(self) -> mpf:
        """Evaluate the formula."""
        result = mpf(0)
        for coef, arg in self.terms:
            result += coef * atan(mpf(1) / arg)
        return result
    
    def __str__(self):
        parts = []
        for coef, arg in self.terms:
            sign = '+' if coef > 0 else ''
            parts.append(f"{sign}{coef}×arctan(1/{arg})")
        return ' '.join(parts)


def search_arctan_formulas(target: mpf, max_terms: int = 2, 
                           max_coef: int = 10, max_arg: int = 300) -> List[ArctanCandidate]:
    """Search for arctan formulas that approximate target."""
    candidates = []
    
    # Two-term search: a×arctan(1/m) + b×arctan(1/n)
    print(f"Searching {max_coef*2}×{max_arg}×{max_coef*2}×{max_arg} = {(max_coef*2)**2 * max_arg**2:,} combinations...")
    
    # Precompute arctan values
    arctan_values = {n: atan(mpf(1)/n) for n in range(1, max_arg + 1)}
    
    best_error = float('inf')
    checked = 0
    
    for m in range(2, max_arg + 1):  # Start from 2 to avoid trivial arctan(1) = π/4
        for n in range(m + 1, max_arg + 1):  # n > m to get distinct arguments
            for a in range(-max_coef, max_coef + 1):
                if a == 0:
                    continue
                for b in range(-max_coef, max_coef + 1):
                    if b == 0:
                        continue
                    
                    value = a * arctan_values[m] + b * arctan_values[n]
                    error = abs(float(value - target))
                    
                    if error < 1e-10:  # Good candidate
                        candidate = ArctanCandidate(
                            terms=[(a, m), (b, n)],
                            value=float(value),
                            n_smooth=-np.log10(error) if error > 0 else 50
                        )
                        candidates.append(candidate)
                        
                        if error < best_error:
                            best_error = error
                            print(f"  Found: {candidate} (error: {error:.2e})")
                    
                    checked += 1
    
    print(f"Checked {checked:,} combinations")
    print(f"Found {len(candidates)} candidates with error < 1e-10")
    
    # Sort by n_smooth (higher = better)
    candidates.sort(key=lambda c: c.n_smooth, reverse=True)
    
    return candidates[:20]  # Top 20


# Run search
candidates = search_arctan_formulas(TARGET['value'], max_coef=5, max_arg=250)

if not candidates:
    print("No candidates found! Expanding search...")
    candidates = search_arctan_formulas(TARGET['value'], max_coef=10, max_arg=300)


# =============================================================================
# PHASE 3: LCM PRUNING
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: LCM PRUNING")
print("=" * 70)

def compute_anchor_vector(candidate: ArctanCandidate) -> Dict[str, float]:
    """Compute anchor vector for a candidate."""
    # For arctan formulas:
    # - sierpinski: π connection
    # - cantor: discrete arguments
    # - zero: integer coefficients
    
    vector = {
        'zero': 0.2,        # Integer coefficients always present
        'sierpinski': 0.3,  # π connection
        'phi': 0.0,
        'e_inv': 0.1,       # Convergence
        'cantor': 0.3,      # Discrete arguments
        'sqrt2_inv': 0.1,   # Connection (equals sign)
    }
    
    # Adjust based on argument sizes (smaller = simpler = more cantor-like)
    avg_arg = np.mean([arg for _, arg in candidate.terms])
    if avg_arg < 10:
        vector['cantor'] += 0.1
    elif avg_arg > 100:
        vector['cantor'] -= 0.1
        vector['sierpinski'] += 0.1
    
    # Normalize
    total = sum(vector.values())
    return {k: v/total for k, v in vector.items()}


def concept_score(candidate_vector: Dict, target_vector: Dict) -> float:
    """Cosine similarity between anchor vectors."""
    keys = list(target_vector.keys())
    v1 = np.array([candidate_vector.get(k, 0) for k in keys])
    v2 = np.array([target_vector.get(k, 0) for k in keys])
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Score candidates
for candidate in candidates:
    candidate.anchor_vector = compute_anchor_vector(candidate)
    candidate.concept_score = concept_score(candidate.anchor_vector, TARGET['anchor_weights'])
    candidate.combined_score = 0.7 * (candidate.n_smooth / 50) + 0.3 * candidate.concept_score

# Sort by combined score
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
    target_val = mp.pi / 4
    error = computed - target_val
    
    print(f"\nBest candidate: {best}")
    print(f"Computed value: {computed}")
    print(f"Target (π/4):   {target_val}")
    print(f"Error:          {float(error):.2e}")
    
    # For Machin-type formulas, error should be exactly 0
    if abs(float(error)) < 1e-50:
        print("\n✓ ERROR IS ESSENTIALLY ZERO - This is an EXACT formula!")
    else:
        print(f"\n✗ Non-zero error detected: {float(error):.2e}")


# =============================================================================
# PHASE 5: PATTERN DETECTION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: PATTERN DETECTION")
print("=" * 70)

if best:
    # For Machin formulas, the pattern IS the formula itself
    print(f"\nFormula structure: {best}")
    
    # Check if this matches known Machin-type formulas
    known_machin = [
        ([(4, 5), (-1, 239)], "Machin (1706)"),
        ([(1, 2), (1, 3)], "Euler"),
        ([(2, 2), (-1, 7)], "Hermann"),
        ([(2, 3), (1, 7)], "Hutton"),
    ]
    
    for terms, name in known_machin:
        if set(best.terms) == set(terms) or set(best.terms) == set([(-t[0], t[1]) for t in terms]):
            print(f"\n✓ MATCHES KNOWN FORMULA: {name}")
            break
    else:
        print("\n? This may be a new Machin-type formula!")


# =============================================================================
# PHASE 6: VERIFICATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: VERIFICATION")
print("=" * 70)

if best:
    mp.dps = 500  # High precision
    
    computed = best.evaluate()
    target_val = mp.pi / 4
    error = abs(computed - target_val)
    
    digits_correct = -float(log10(error)) if error > 0 else 500
    
    print(f"\nHigh-precision verification (500 digits):")
    print(f"  Formula: {best}")
    print(f"  Error: {float(error):.2e}")
    print(f"  Digits correct: {digits_correct:.0f}")
    
    if digits_correct > 400:
        print("\n" + "=" * 70)
        print("✓ DISCOVERY VALIDATED!")
        print("=" * 70)
        print(f"\nπ/4 = {best}")
        print(f"\nOr equivalently:")
        parts = []
        for coef, arg in best.terms:
            parts.append(f"{coef}×arctan(1/{arg})")
        print(f"π = 4×({' + '.join(parts).replace('+ -', '- ')})")
    else:
        print("\n✗ Verification failed - not enough precision")


print("\n" + "=" * 70)
print("PROTOCOL COMPLETE")
print("=" * 70)
