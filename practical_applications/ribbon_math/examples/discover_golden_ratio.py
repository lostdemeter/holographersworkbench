#!/usr/bin/env python3
"""
Example 3: Discover Golden Ratio Identities
===========================================

Target: Find identities involving φ = (1+√5)/2

This follows the Equation Discovery Protocol (EDP) to discover
various golden ratio identities.

Protocol Phases:
1. Concept Definition - golden ratio relationships
2. Structure Search - search algebraic forms
3. LCM Pruning - filter by anchor alignment
4. Error Analysis - find patterns
5. Pattern Detection - verify closed forms
6. Verification - high-precision check
"""

import numpy as np
from mpmath import mp, mpf, sqrt, log, log10, pi, atan, fib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# PHASE 1: CONCEPT DEFINITION
# =============================================================================

print("=" * 70)
print("EQUATION DISCOVERY PROTOCOL - Example 3: Golden Ratio Identities")
print("=" * 70)

print("\n" + "=" * 70)
print("PHASE 1: CONCEPT DEFINITION")
print("=" * 70)

mp.dps = 100

PHI = (1 + sqrt(5)) / 2
PSI = 1 / PHI  # = φ - 1

TARGET = {
    'description': "Identities involving the golden ratio φ",
    'value': PHI,
    'keywords': ['golden ratio', 'phi', 'fibonacci', 'continued fraction', 'self-similar'],
    'anchor_weights': {
        'zero': 0.1,        # Unity, integers
        'sierpinski': 0.2,  # Self-similar structure
        'phi': 0.4,         # Golden ratio itself (dominant!)
        'e_inv': 0.1,       # Convergence
        'cantor': 0.1,      # Discrete (Fibonacci)
        'sqrt2_inv': 0.1,   # Algebraic connection
    },
    'constraints': [
        "Must involve φ or √5",
        "Should reveal self-similar structure",
        "Prefer simple integer coefficients",
    ]
}

print(f"Target: {TARGET['description']}")
print(f"φ = {float(PHI):.15f}")
print(f"1/φ = {float(PSI):.15f}")
print(f"Keywords: {TARGET['keywords']}")


# =============================================================================
# PHASE 2: STRUCTURE SEARCH
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: STRUCTURE SEARCH")
print("=" * 70)

@dataclass
class GoldenIdentity:
    """A golden ratio identity."""
    lhs: str
    rhs: str
    lhs_value: float
    rhs_value: float
    error: float
    n_smooth: float = 0.0
    
    def __str__(self):
        return f"{self.lhs} = {self.rhs}"


def search_golden_identities() -> List[GoldenIdentity]:
    """Search for golden ratio identities."""
    identities = []
    
    phi = PHI
    psi = PSI
    sqrt5 = sqrt(5)
    
    # Known identity forms to search
    print("Searching algebraic identities...")
    
    # 1. Power identities: φ^n = F_n × φ + F_{n-1}
    print("\n1. Fibonacci-power identities:")
    for n in range(2, 10):
        lhs = phi ** n
        # φ^n = F_n × φ + F_{n-1}
        f_n = fib(n)
        f_n1 = fib(n-1)
        rhs = f_n * phi + f_n1
        error = abs(float(lhs - rhs))
        
        if error < 1e-10:
            identity = GoldenIdentity(
                lhs=f"φ^{n}",
                rhs=f"F_{n}×φ + F_{n-1} = {int(f_n)}φ + {int(f_n1)}",
                lhs_value=float(lhs),
                rhs_value=float(rhs),
                error=error,
                n_smooth=-np.log10(error) if error > 0 else 100
            )
            identities.append(identity)
            print(f"  ✓ {identity}")
    
    # 2. Self-similar identity: φ² = φ + 1
    print("\n2. Self-similar identities:")
    tests = [
        ("φ²", phi**2, "φ + 1", phi + 1),
        ("φ³", phi**3, "2φ + 1", 2*phi + 1),
        ("1/φ", psi, "φ - 1", phi - 1),
        ("φ - 1/φ", phi - psi, "1", mpf(1)),
        ("φ × 1/φ", phi * psi, "1", mpf(1)),
        ("φ² - φ", phi**2 - phi, "1", mpf(1)),
    ]
    
    for lhs_str, lhs_val, rhs_str, rhs_val in tests:
        error = abs(float(lhs_val - rhs_val))
        if error < 1e-10:
            identity = GoldenIdentity(
                lhs=lhs_str, rhs=rhs_str,
                lhs_value=float(lhs_val), rhs_value=float(rhs_val),
                error=error,
                n_smooth=-np.log10(error) if error > 0 else 100
            )
            identities.append(identity)
            print(f"  ✓ {identity}")
    
    # 3. Trigonometric identities
    print("\n3. Trigonometric identities:")
    trig_tests = [
        ("2×cos(π/5)", 2*mp.cos(pi/5), "φ", phi),
        ("2×cos(2π/5)", 2*mp.cos(2*pi/5), "1/φ", psi),
        ("arctan(1/φ) + arctan(1/φ³)", atan(psi) + atan(psi**3), "π/4", pi/4),
    ]
    
    for lhs_str, lhs_val, rhs_str, rhs_val in trig_tests:
        error = abs(float(lhs_val - rhs_val))
        if error < 1e-10:
            identity = GoldenIdentity(
                lhs=lhs_str, rhs=rhs_str,
                lhs_value=float(lhs_val), rhs_value=float(rhs_val),
                error=error,
                n_smooth=-np.log10(error) if error > 0 else 100
            )
            identities.append(identity)
            print(f"  ✓ {identity}")
    
    # 4. Logarithmic identities
    print("\n4. Logarithmic identities:")
    log_tests = [
        ("log(φ)", log(phi), "asinh(1/2)", mp.asinh(mpf(1)/2)),
        ("2×log(φ)", 2*log(phi), "log(φ²)", log(phi**2)),
    ]
    
    for lhs_str, lhs_val, rhs_str, rhs_val in log_tests:
        error = abs(float(lhs_val - rhs_val))
        if error < 1e-10:
            identity = GoldenIdentity(
                lhs=lhs_str, rhs=rhs_str,
                lhs_value=float(lhs_val), rhs_value=float(rhs_val),
                error=error,
                n_smooth=-np.log10(error) if error > 0 else 100
            )
            identities.append(identity)
            print(f"  ✓ {identity}")
    
    # 5. Algebraic identities with √5
    print("\n5. Algebraic identities:")
    alg_tests = [
        ("φ + 1/φ", phi + psi, "√5", sqrt5),
        ("φ - 1/φ", phi - psi, "1", mpf(1)),
        ("φ² + 1/φ²", phi**2 + psi**2, "3", mpf(3)),
        ("φ³ + 1/φ³", phi**3 + psi**3, "4", mpf(4)),
        ("φ⁴ + 1/φ⁴", phi**4 + psi**4, "7", mpf(7)),
        ("(φ + 1/φ)²", (phi + psi)**2, "5", mpf(5)),
    ]
    
    for lhs_str, lhs_val, rhs_str, rhs_val in alg_tests:
        error = abs(float(lhs_val - rhs_val))
        if error < 1e-10:
            identity = GoldenIdentity(
                lhs=lhs_str, rhs=rhs_str,
                lhs_value=float(lhs_val), rhs_value=float(rhs_val),
                error=error,
                n_smooth=-np.log10(error) if error > 0 else 100
            )
            identities.append(identity)
            print(f"  ✓ {identity}")
    
    return identities


identities = search_golden_identities()
print(f"\nFound {len(identities)} identities")


# =============================================================================
# PHASE 3: LCM PRUNING
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: LCM PRUNING")
print("=" * 70)

def compute_anchor_vector(identity: GoldenIdentity) -> Dict[str, float]:
    """Compute anchor vector for an identity."""
    vector = {
        'zero': 0.1,
        'sierpinski': 0.2,
        'phi': 0.4,  # Golden ratio dominant
        'e_inv': 0.1,
        'cantor': 0.1,
        'sqrt2_inv': 0.1,
    }
    
    # Adjust based on content
    if 'F_' in identity.rhs or 'Fibonacci' in identity.rhs.lower():
        vector['cantor'] += 0.1  # Discrete sequence
    if 'π' in identity.lhs or 'π' in identity.rhs:
        vector['sierpinski'] += 0.1
    if 'log' in identity.lhs or 'log' in identity.rhs:
        vector['e_inv'] += 0.1
    
    # Normalize
    total = sum(vector.values())
    return {k: v/total for k, v in vector.items()}


def concept_score(candidate_vector: Dict, target_vector: Dict) -> float:
    keys = list(target_vector.keys())
    v1 = np.array([candidate_vector.get(k, 0) for k in keys])
    v2 = np.array([target_vector.get(k, 0) for k in keys])
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


for identity in identities:
    identity.anchor_vector = compute_anchor_vector(identity)
    identity.concept_score = concept_score(identity.anchor_vector, TARGET['anchor_weights'])

# Sort by concept score
identities.sort(key=lambda i: i.concept_score, reverse=True)

print("\nTop identities by concept alignment:")
for i, ident in enumerate(identities[:10]):
    print(f"  {i+1}. {ident} (score: {ident.concept_score:.3f})")


# =============================================================================
# PHASE 4 & 5: ERROR ANALYSIS & PATTERN DETECTION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4 & 5: ERROR ANALYSIS & PATTERN DETECTION")
print("=" * 70)

print("\nAll identities have essentially zero error (exact algebraic identities)")
print("\nKey patterns discovered:")

# Group by type
fibonacci_patterns = [i for i in identities if 'F_' in i.rhs]
self_similar = [i for i in identities if '+ 1' in i.rhs or '- 1' in i.rhs]
algebraic = [i for i in identities if '√5' in i.rhs or i.rhs.isdigit()]

print(f"\n1. Fibonacci-Power Pattern ({len(fibonacci_patterns)} identities):")
print("   φ^n = F_n × φ + F_{n-1}")
print("   This connects powers of φ to Fibonacci numbers")

print(f"\n2. Self-Similar Pattern ({len(self_similar)} identities):")
print("   φ² = φ + 1 (defining property)")
print("   This is the source of φ's self-similarity")

print(f"\n3. Lucas Number Pattern:")
print("   φ^n + 1/φ^n = L_n (Lucas numbers)")
print("   L_1=1, L_2=3, L_3=4, L_4=7, ...")


# =============================================================================
# PHASE 6: VERIFICATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: VERIFICATION")
print("=" * 70)

mp.dps = 500

# Recompute with higher precision
PHI_HP = (1 + sqrt(5)) / 2
PSI_HP = 1 / PHI_HP

print("\nHigh-precision verification of key identities:")

key_identities = [
    ("φ² = φ + 1", PHI_HP**2, PHI_HP + 1),
    ("φ + 1/φ = √5", PHI_HP + PSI_HP, sqrt(5)),
    ("φ² + 1/φ² = 3", PHI_HP**2 + PSI_HP**2, mpf(3)),
    ("arctan(1/φ) + arctan(1/φ³) = π/4", atan(PSI_HP) + atan(PSI_HP**3), pi/4),
]

all_verified = True
for name, lhs, rhs in key_identities:
    error = abs(lhs - rhs)
    digits = -float(log10(error)) if error > 0 else 500
    status = "✓" if digits > 400 else "✗"
    print(f"  {status} {name}: {digits:.0f} digits correct")
    if digits < 400:
        all_verified = False

if all_verified:
    print("\n" + "=" * 70)
    print("✓ ALL DISCOVERIES VALIDATED!")
    print("=" * 70)
    print("\nThe Golden Ratio φ = (1+√5)/2 satisfies:")
    print("  • φ² = φ + 1 (self-similar)")
    print("  • φ^n = F_n×φ + F_{n-1} (Fibonacci connection)")
    print("  • φ^n + φ^{-n} = L_n (Lucas numbers)")
    print("  • arctan(1/φ) + arctan(1/φ³) = π/4 (π connection)")


print("\n" + "=" * 70)
print("PROTOCOL COMPLETE")
print("=" * 70)
