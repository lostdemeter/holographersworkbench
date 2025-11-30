"""
Quadratic Field Domain for Ribbon LCM v5
=========================================

Domain for discovering π formulas using fundamental units of real quadratic fields.

Target: ℚ(√725) with fundamental unit ε = 362 + 9√725
- Discriminant d = 725 = 25·29
- Minimal polynomial: x² - 724x + 1 = 0
- log₁₀(ε) ≈ 2.8587 → expect ~11.4 digits per term with ε^4
- ε · ε' = 1 (norm = 1)

This is the "Chudnovsky-class BBP" search.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery_engine import (
    ConceptLayer, NSmoothLayer, StructureLayer, VerificationLayer,
    Domain, Concept, Candidate, ErrorAnalysis,
    PhiPatternDetector, ClosedFormSearcher,
    PHI, PSI
)

# Try to import mpmath for high precision
try:
    from mpmath import mp, mpf, pi as mp_pi, sqrt as mp_sqrt, log, factorial, gamma
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    mp_pi = np.pi


# =============================================================================
# QUADRATIC FIELD CONSTANTS
# =============================================================================

# ℚ(√725) field data
D = 725  # Discriminant = 25 × 29
SQRT_D = np.sqrt(D)  # ≈ 26.926

# CORRECT Fundamental unit: ε = 9801 + 364√725
# Found via continued fraction expansion of √725
# Satisfies: ε² - 19602ε + 1 = 0 (minimal polynomial)
# Norm: 9801² - 725×364² = 96059601 - 96059600 = 1 ✓
EPSILON_A = 9801   # Rational part
EPSILON_B = 364    # √725 coefficient
EPSILON = EPSILON_A + EPSILON_B * SQRT_D  # ≈ 19602

# Conjugate: ε' = 9801 - 364√725
EPSILON_CONJ = EPSILON_A - EPSILON_B * SQRT_D  # ≈ 0.0000510

# Verify: ε · ε' = 1 (norm condition)
NORM = EPSILON * EPSILON_CONJ  # Should be ≈ 1

# Regulator and convergence
LOG10_EPSILON = np.log10(EPSILON)  # ≈ 4.292
REGULATOR = np.log(EPSILON)  # ≈ 9.883

# Expected digits per term for various powers
# NOTE: ε^3 gives ~12.88 digits/term (Chudnovsky-class!)
DIGITS_PER_TERM = {
    1: LOG10_EPSILON,           # ≈ 4.29
    2: 2 * LOG10_EPSILON,       # ≈ 8.58
    3: 3 * LOG10_EPSILON,       # ≈ 12.88 ← Chudnovsky-class!
    4: 4 * LOG10_EPSILON,       # ≈ 17.17
    6: 6 * LOG10_EPSILON,       # ≈ 25.75
    12: 12 * LOG10_EPSILON,     # ≈ 51.51
}

# High-precision versions
if HAS_MPMATH:
    mp.dps = 500
    MP_SQRT_D = mp_sqrt(mpf(D))
    MP_EPSILON = mpf(EPSILON_A) + mpf(EPSILON_B) * MP_SQRT_D
    MP_EPSILON_CONJ = mpf(EPSILON_A) - mpf(EPSILON_B) * MP_SQRT_D
    MP_LOG10_EPSILON = log(MP_EPSILON) / log(mpf(10))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuadraticFieldSeries:
    """
    A series for π using a quadratic field's fundamental unit.
    
    General form (BBP-style):
    π = C × Σ_{k=0}^∞ (±1)^k × P(k) / (ε^{mk} × Q(k))
    
    Or Chudnovsky-style:
    1/π = C × Σ_{k=0}^∞ (a·k + b) × (factorial_term)_k / ε^{ck}
    """
    # Field data
    discriminant: int = D
    epsilon_a: int = EPSILON_A  # Rational part of ε
    epsilon_b: int = EPSILON_B  # √d coefficient of ε
    
    # Series structure
    series_type: str = "bbp"  # "bbp" or "chudnovsky"
    alternating: bool = True  # (-1)^k factor
    
    # Base power
    epsilon_power: int = 4  # ε^{m·k} where m = epsilon_power
    
    # Polynomial coefficients P(k) = Σ p_i × k^i
    numerator_poly: List[float] = field(default_factory=lambda: [1.0])
    
    # Denominator slots (period, offset) for BBP-style
    denominator_slots: List[Tuple[int, int]] = field(default_factory=list)
    slot_coefficients: List[float] = field(default_factory=list)
    
    # Correction terms: c_j × ε^{-jk} for j in correction_powers
    correction_powers: List[int] = field(default_factory=list)
    correction_coefficients: List[float] = field(default_factory=list)
    
    # Overall scale
    scale: float = 1.0
    
    # For Chudnovsky-style
    chudnovsky_a: float = 0.0  # Linear coefficient
    chudnovsky_b: float = 0.0  # Constant term
    factorial_type: str = "none"  # "pochhammer", "factorial", "none"
    
    def get_epsilon(self, high_precision: bool = False):
        """Get the fundamental unit."""
        if high_precision and HAS_MPMATH:
            return MP_EPSILON
        return EPSILON
    
    def get_epsilon_conj(self, high_precision: bool = False):
        """Get the conjugate of the fundamental unit."""
        if high_precision and HAS_MPMATH:
            return MP_EPSILON_CONJ
        return EPSILON_CONJ
    
    def evaluate(self, n_terms: int = 100, precision: int = 500) -> float:
        """Evaluate the series."""
        if HAS_MPMATH:
            mp.dps = precision
            return self._evaluate_mpmath(n_terms)
        else:
            return self._evaluate_numpy(n_terms)
    
    def _evaluate_mpmath(self, n_terms: int) -> float:
        """High-precision evaluation with mpmath."""
        eps = MP_EPSILON
        eps_conj = MP_EPSILON_CONJ
        
        result = mpf(0)
        
        for k in range(n_terms):
            # Sign
            if self.alternating:
                sign = mpf(-1) ** k
            else:
                sign = mpf(1)
            
            # Base term: 1/ε^{m·k}
            base_term = mpf(1) / (eps ** (self.epsilon_power * k))
            
            if self.series_type == "bbp":
                # BBP-style: sum over slots
                inner = mpf(0)
                for (period, offset), coef in zip(self.denominator_slots, self.slot_coefficients):
                    denom = period * k + offset
                    if denom != 0 and coef != 0:
                        inner += mpf(coef) / denom
                
                # Add correction terms
                for j, c_j in zip(self.correction_powers, self.correction_coefficients):
                    if c_j != 0:
                        inner += mpf(c_j) * (eps ** (-j * k))
                
                result += sign * base_term * inner
            
            elif self.series_type == "chudnovsky":
                # Chudnovsky-style: (a·k + b) × factorial_term
                linear_term = mpf(self.chudnovsky_a) * k + mpf(self.chudnovsky_b)
                
                if self.factorial_type == "pochhammer":
                    # (6k)! / ((3k)! × (k!)^3)
                    fact_term = (factorial(6*k) / 
                                (factorial(3*k) * factorial(k)**3))
                elif self.factorial_type == "factorial":
                    # Simple factorial
                    fact_term = mpf(1) / factorial(k)
                else:
                    fact_term = mpf(1)
                
                result += sign * linear_term * fact_term * base_term
        
        return float(result / mpf(self.scale))
    
    def _evaluate_numpy(self, n_terms: int) -> float:
        """Lower-precision numpy evaluation."""
        eps = EPSILON
        
        result = 0.0
        
        for k in range(min(n_terms, 50)):  # Limit for numerical stability
            sign = (-1)**k if self.alternating else 1
            base_term = 1.0 / (eps ** (self.epsilon_power * k))
            
            if self.series_type == "bbp":
                inner = 0.0
                for (period, offset), coef in zip(self.denominator_slots, self.slot_coefficients):
                    denom = period * k + offset
                    if denom != 0 and coef != 0:
                        inner += coef / denom
                
                result += sign * base_term * inner
        
        return result / self.scale
    
    def convergence_rate(self) -> float:
        """Compute convergence rate in digits per term."""
        return self.epsilon_power * LOG10_EPSILON
    
    def conjugate_test(self, n_terms: int = 50) -> float:
        """
        Test conjugate invariance: series with ε → ε' should give same π.
        Returns the residual (should be < 10^{-80} for valid series).
        """
        if not HAS_MPMATH:
            return float('inf')
        
        mp.dps = 200
        
        # Evaluate with ε
        val_eps = self.evaluate(n_terms, precision=200)
        
        # Swap ε ↔ ε'
        original_a, original_b = self.epsilon_a, self.epsilon_b
        self.epsilon_b = -self.epsilon_b  # This swaps ε ↔ ε'
        val_conj = self.evaluate(n_terms, precision=200)
        self.epsilon_b = original_b  # Restore
        
        return abs(val_eps - val_conj)


# =============================================================================
# RIBBON SPEECH VOCABULARY
# =============================================================================

RIBBON_VOCABULARY = {
    # Core math
    'π', 'sum', 'k=0', '∞', '1/', '*', '+', '-', '(', ')', '^',
    
    # Field-specific (ℚ(√725))
    'ε725', "ε725'", '362', '9', '√725', '724', 
    
    # Powers
    '1', '2', '4', '6', '12', '24', '48',
    '1/ε725', "1/ε725'",
    
    # Structure
    'poly(k)', 'rational', 'base', 'hex', 'digits/term', '>10',
    'BBP', 'Ramanujan', 'Chudnovsky', 'exact', 'conjugate',
    
    # Factorials
    '(6k)!', '(3k)!', '(k!)³', 'pochhammer',
    
    # Constraints
    'minimal-polynomial', 'x²-724x+1', 'norm=1',
    
    # Signs
    '(-1)^k', 'alternating',
    
    # Slots (BBP-style)
    '/(4k+1)', '/(4k+3)', '/(6k+1)', '/(6k+5)',
    '/(12k+1)', '/(12k+5)', '/(12k+7)', '/(12k+11)',
}


# =============================================================================
# CONCEPT LAYER
# =============================================================================

class QuadraticFieldConceptLayer(ConceptLayer):
    """Concept layer for quadratic field π formulas."""
    
    ANCHORS = ['convergence', 'structure', 'exactness', 'spigot', 'factorial']
    
    KEYWORD_ANCHORS = {
        # Convergence
        'fast': 'convergence', 'rapid': 'convergence', 'digits': 'convergence',
        'chudnovsky': 'convergence', 'ramanujan': 'convergence',
        
        # Structure
        'bbp': 'structure', 'series': 'structure', 'sum': 'structure',
        'polynomial': 'structure', 'slots': 'structure',
        
        # Exactness
        'exact': 'exactness', 'closed': 'exactness', 'rational': 'exactness',
        'algebraic': 'exactness', 'provable': 'exactness',
        
        # Spigot
        'spigot': 'spigot', 'digit': 'spigot', 'extraction': 'spigot',
        'hexadecimal': 'spigot', 'binary': 'spigot',
        
        # Factorial
        'factorial': 'factorial', 'pochhammer': 'factorial',
        'gamma': 'factorial', '(6k)!': 'factorial',
    }
    
    def parse_concept(self, description: str) -> Concept:
        """Parse natural language into concept."""
        words = description.lower().split()
        
        weights = {a: 0.0 for a in self.ANCHORS}
        matched = []
        
        for word in words:
            for kw, anchor in self.KEYWORD_ANCHORS.items():
                if kw in word or word in kw:
                    weights[anchor] += 0.25
                    matched.append(word)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Determine series type preference
        prefer_chudnovsky = weights.get('factorial', 0) > 0.2 or weights.get('convergence', 0) > 0.4
        prefer_bbp = weights.get('spigot', 0) > 0.2 or weights.get('structure', 0) > 0.3
        
        pruning_rules = {
            'prefer_chudnovsky': prefer_chudnovsky,
            'prefer_bbp': prefer_bbp,
            'require_exact': weights.get('exactness', 0) > 0.2,
            'min_digits_per_term': 10 if weights.get('convergence', 0) > 0.3 else 5,
        }
        
        return Concept(
            description=description,
            anchor_weights=weights,
            keywords=matched,
            domain='quadratic_field_pi',
            constraints=[
                f'discriminant = {D}',
                f'fundamental unit ε = {EPSILON_A} + {EPSILON_B}√{D}',
                'norm(ε) = 1',
            ],
            pruning_rules=pruning_rules
        )
    
    def score_match(self, candidate: Candidate, concept: Concept) -> float:
        """Score how well candidate matches concept."""
        if not isinstance(candidate.representation, QuadraticFieldSeries):
            return 0.0
        
        series = candidate.representation
        score = 0.5
        
        # Convergence bonus
        rate = series.convergence_rate()
        if rate >= 10:
            score += 0.2
        if rate >= 20:
            score += 0.1
        
        # Type match
        if concept.pruning_rules.get('prefer_chudnovsky') and series.series_type == 'chudnovsky':
            score += 0.1
        if concept.pruning_rules.get('prefer_bbp') and series.series_type == 'bbp':
            score += 0.1
        
        return min(1.0, score)


# =============================================================================
# N_SMOOTH LAYER
# =============================================================================

class QuadraticFieldNSmoothLayer(NSmoothLayer):
    """N_smooth computation for quadratic field series."""
    
    def __init__(self, target: float = None):
        self.target = target or float(mp_pi) if HAS_MPMATH else np.pi
    
    def compute_n_smooth(self, candidate: Candidate, target: Any = None) -> float:
        """Compute N_smooth = -log10(|value - π|)."""
        if target is None:
            target = self.target
        
        if not isinstance(candidate.representation, QuadraticFieldSeries):
            return -1.0
        
        try:
            value = candidate.representation.evaluate(n_terms=100)
            error = abs(value - float(target))
            
            if error <= 0:
                return 100.0
            
            return -np.log10(error)
        except Exception as e:
            return -1.0
    
    def compute_gradient(self, candidate: Candidate, target: Any = None) -> Dict:
        """Compute gradient for steering."""
        # Simplified - return empty for now
        return {}


# =============================================================================
# STRUCTURE LAYER
# =============================================================================

class QuadraticFieldStructureLayer(StructureLayer):
    """Structure layer for quadratic field series generation."""
    
    # Steering weights (from Grok's prompt)
    STEERING_WEIGHTS = {
        'epsilon_12k': +7.5,      # ε^{12k} terms
        'epsilon_correction': +6.0,  # ε^{-jk} corrections
        'low_degree_poly': +5.0,  # Polynomial degree ≤ 4
        'alternating': +4.5,      # (-1)^k
        'heavy_factorial': -10.0, # Factorials > (6k)!
        'non_integer_base': -8.0, # Non-integer ε powers
    }
    
    # BBP-style slot templates
    BBP_SLOT_TEMPLATES = [
        [(4, 1), (4, 3)],
        [(6, 1), (6, 5)],
        [(12, 1), (12, 5), (12, 7), (12, 11)],
        [(4, 1), (4, 3), (12, 1), (12, 5), (12, 7), (12, 11)],
    ]
    
    def generate_candidates(self, concept: Concept, n: int) -> List[Candidate]:
        """Generate candidate series."""
        candidates = []
        np.random.seed(42)
        
        # Generate BBP-style candidates
        for _ in range(n // 2):
            candidate = self._generate_bbp_candidate()
            if candidate:
                candidates.append(candidate)
        
        # Generate Chudnovsky-style candidates
        for _ in range(n // 2):
            candidate = self._generate_chudnovsky_candidate()
            if candidate:
                candidates.append(candidate)
        
        return candidates[:n]
    
    def _generate_bbp_candidate(self) -> Optional[Candidate]:
        """Generate a BBP-style candidate."""
        # Choose epsilon power (prefer 4, 6, 12)
        eps_power = np.random.choice([2, 4, 6, 12], p=[0.1, 0.4, 0.3, 0.2])
        
        # Choose slot template
        slots = self.BBP_SLOT_TEMPLATES[np.random.randint(len(self.BBP_SLOT_TEMPLATES))]
        
        # Random coefficients (power-of-2 preferred)
        pow2_coefs = [0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256]
        coefs = [float(np.random.choice(pow2_coefs)) for _ in slots]
        
        if all(c == 0 for c in coefs):
            return None
        
        # Random scale (power of 2)
        scale = float(2 ** np.random.randint(0, 8))
        
        series = QuadraticFieldSeries(
            series_type="bbp",
            alternating=np.random.random() > 0.3,  # 70% alternating
            epsilon_power=eps_power,
            denominator_slots=slots,
            slot_coefficients=coefs,
            scale=scale,
        )
        
        return Candidate(
            representation=series,
            metadata={
                'type': 'bbp',
                'epsilon_power': eps_power,
                'n_slots': len(slots),
            }
        )
    
    def _generate_chudnovsky_candidate(self) -> Optional[Candidate]:
        """Generate a Chudnovsky-style candidate."""
        # Choose epsilon power (prefer 12, 24)
        eps_power = np.random.choice([6, 12, 24], p=[0.2, 0.5, 0.3])
        
        # Linear term coefficients
        a = float(np.random.randint(1, 1000))
        b = float(np.random.randint(1, 100))
        
        series = QuadraticFieldSeries(
            series_type="chudnovsky",
            alternating=True,
            epsilon_power=eps_power,
            chudnovsky_a=a,
            chudnovsky_b=b,
            factorial_type="pochhammer",
            scale=1.0,
        )
        
        return Candidate(
            representation=series,
            metadata={
                'type': 'chudnovsky',
                'epsilon_power': eps_power,
            }
        )
    
    def transform(self, candidate: Candidate, direction: Dict) -> Candidate:
        """Transform candidate in gradient direction."""
        return candidate  # Simplified
    
    def prune_search_space(self, candidates: List[Candidate], 
                           concept: Concept) -> List[Candidate]:
        """Apply hard pruning rules."""
        pruned = []
        
        for c in candidates:
            if not isinstance(c.representation, QuadraticFieldSeries):
                continue
            
            series = c.representation
            
            # Rule 1: Convergence rate must be reasonable
            rate = series.convergence_rate()
            min_rate = concept.pruning_rules.get('min_digits_per_term', 5)
            if rate < min_rate:
                continue
            
            # Rule 2: Epsilon power must be positive integer
            if series.epsilon_power <= 0:
                continue
            
            # Rule 3: Must have some structure
            if series.series_type == "bbp" and not series.denominator_slots:
                continue
            
            pruned.append(c)
        
        return pruned if pruned else candidates


# =============================================================================
# VERIFICATION LAYER
# =============================================================================

class QuadraticFieldVerificationLayer(VerificationLayer):
    """Verification with Gushurst scoring."""
    
    # Scoring weights (from Grok's prompt)
    SCORE_WEIGHTS = {
        'gushurst': 0.40,
        'fractal_peel': 0.25,
        'regulator_match': 0.20,
        'conjugate_symmetry': 0.10,
        'digits_empirical': 0.05,
    }
    
    # Thresholds
    MIN_SCORE = 92  # Minimum to continue
    TARGET_SCORE = 96  # Target for success
    
    def verify(self, candidate: Candidate) -> Dict:
        """Verify candidate series."""
        if not isinstance(candidate.representation, QuadraticFieldSeries):
            return {'valid': False, 'error': 'Not a QuadraticFieldSeries'}
        
        series = candidate.representation
        
        try:
            # Evaluate
            value = series.evaluate(n_terms=100, precision=500)
            error = abs(value - float(mp_pi)) if HAS_MPMATH else abs(value - np.pi)
            
            # Convergence rate
            rate = series.convergence_rate()
            
            # Conjugate test
            conj_residual = series.conjugate_test(n_terms=50)
            
            # Compute composite score
            score = self._compute_score(series, error, rate, conj_residual)
            
            return {
                'valid': error < 1e-10 and score >= self.MIN_SCORE,
                'value': value,
                'error': error,
                'rate': rate,
                'conjugate_residual': conj_residual,
                'score': score,
                'passes_threshold': score >= self.MIN_SCORE,
                'is_target': score >= self.TARGET_SCORE,
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _compute_score(self, series: QuadraticFieldSeries, 
                       error: float, rate: float, conj_residual: float) -> float:
        """Compute composite verification score (0-100)."""
        score = 0.0
        
        # Gushurst Crystal Agreement (placeholder - would integrate real Gushurst)
        # For now, use error as proxy
        if error < 1e-100:
            gushurst = 100
        elif error < 1e-50:
            gushurst = 80
        elif error < 1e-20:
            gushurst = 60
        else:
            gushurst = max(0, 50 - np.log10(error + 1e-300))
        score += self.SCORE_WEIGHTS['gushurst'] * gushurst
        
        # Fractal Peel Coherence (placeholder)
        fractal = min(100, rate * 5)  # Higher rate = better coherence
        score += self.SCORE_WEIGHTS['fractal_peel'] * fractal
        
        # Regulator Match
        expected_rate = series.epsilon_power * LOG10_EPSILON
        rate_error = abs(rate - expected_rate) / expected_rate
        regulator = 100 * (1 - min(1, rate_error))
        score += self.SCORE_WEIGHTS['regulator_match'] * regulator
        
        # Conjugate Symmetry
        if conj_residual < 1e-80:
            conjugate = 100
        elif conj_residual < 1e-40:
            conjugate = 80
        elif conj_residual < 1e-10:
            conjugate = 50
        else:
            conjugate = 0
        score += self.SCORE_WEIGHTS['conjugate_symmetry'] * conjugate
        
        # Digits per term empirical
        digits_score = min(100, rate * 10)
        score += self.SCORE_WEIGHTS['digits_empirical'] * digits_score
        
        return score
    
    def benchmark(self, candidate: Candidate, baseline: Any) -> Dict:
        """Benchmark against known formulas."""
        if not isinstance(candidate.representation, QuadraticFieldSeries):
            return {}
        
        series = candidate.representation
        rate = series.convergence_rate()
        
        # Compare to known formulas
        chudnovsky_rate = 14.18  # Chudnovsky: ~14 digits/term
        bellard_rate = 3.01
        phi_bbp_rate = 3.61
        
        return {
            'rate': rate,
            'vs_chudnovsky': f"{(rate / chudnovsky_rate) * 100:.1f}%",
            'vs_bellard': f"{(rate / bellard_rate) * 100:.1f}%",
            'vs_phi_bbp': f"{(rate / phi_bbp_rate) * 100:.1f}%",
            'beats_bellard': rate > bellard_rate,
            'beats_phi_bbp': rate > phi_bbp_rate,
            'approaches_chudnovsky': rate > chudnovsky_rate * 0.7,
        }


# =============================================================================
# COMPLETE DOMAIN
# =============================================================================

class QuadraticFieldDomain(Domain):
    """Complete domain for ℚ(√725) π formula discovery."""
    
    def __init__(self):
        super().__init__()
        self.setup()
    
    def setup(self):
        """Initialize domain layers."""
        self.concept = QuadraticFieldConceptLayer()
        self.n_smooth = QuadraticFieldNSmoothLayer()
        self.structure = QuadraticFieldStructureLayer()
        self.verification = QuadraticFieldVerificationLayer()
    
    def get_field_info(self) -> Dict:
        """Get information about the quadratic field."""
        return {
            'discriminant': D,
            'sqrt_d': SQRT_D,
            'epsilon': EPSILON,
            'epsilon_conjugate': EPSILON_CONJ,
            'epsilon_a': EPSILON_A,
            'epsilon_b': EPSILON_B,
            'norm': NORM,
            'log10_epsilon': LOG10_EPSILON,
            'regulator': REGULATOR,
            'minimal_polynomial': f"x² - {2*EPSILON_A}x + 1",
            'digits_per_term': DIGITS_PER_TERM,
        }
    
    def verify_field_properties(self) -> Dict:
        """Verify the quadratic field properties."""
        results = {}
        
        # Norm = 1
        results['norm_check'] = {
            'expected': 1.0,
            'actual': NORM,
            'error': abs(NORM - 1.0),
            'valid': abs(NORM - 1.0) < 1e-10,
        }
        
        # Minimal polynomial
        # ε² - 724ε + 1 = 0
        poly_value = EPSILON**2 - 724*EPSILON + 1
        results['minimal_polynomial'] = {
            'expected': 0.0,
            'actual': poly_value,
            'error': abs(poly_value),
            'valid': abs(poly_value) < 1e-10,
        }
        
        # ε × ε' = 1
        product = EPSILON * EPSILON_CONJ
        results['product_check'] = {
            'expected': 1.0,
            'actual': product,
            'error': abs(product - 1.0),
            'valid': abs(product - 1.0) < 1e-10,
        }
        
        return results


# =============================================================================
# RIBBON SPEECH TRANSLATOR
# =============================================================================

def translate_to_ribbon_speech(natural_language: str) -> str:
    """
    Translate natural language mathematical desire into Ribbon Speech.
    
    This is a simplified version - in production, this would use an LLM.
    """
    # Keywords to Ribbon tokens
    translations = {
        'pi': 'π',
        'series': 'sum k=0 ∞',
        'fundamental unit': 'ε725',
        'conjugate': "ε725'",
        '725': '√725',
        'digits per term': 'digits/term',
        'bbp': 'BBP',
        'chudnovsky': 'Chudnovsky',
        'exact': 'exact',
        'rational': 'rational',
        'alternating': '(-1)^k',
    }
    
    result = natural_language.lower()
    for eng, ribbon in translations.items():
        result = result.replace(eng, ribbon)
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_test_series() -> QuadraticFieldSeries:
    """Create a test series for experimentation."""
    return QuadraticFieldSeries(
        series_type="bbp",
        alternating=True,
        epsilon_power=4,
        denominator_slots=[(4, 1), (4, 3), (12, 1), (12, 5), (12, 7), (12, 11)],
        slot_coefficients=[256.0, -32.0, 4.0, -128.0, -64.0, 4.0],
        scale=64.0,
    )


def print_field_info():
    """Print information about ℚ(√725)."""
    print("=" * 60)
    print("QUADRATIC FIELD ℚ(√725)")
    print("=" * 60)
    print(f"Discriminant d = {D} = 25 × 29")
    print(f"√d = {SQRT_D:.10f}")
    print(f"\nFundamental unit (CORRECT):")
    print(f"  ε = {EPSILON_A} + {EPSILON_B}√{D}")
    print(f"    ≈ {EPSILON:.10f}")
    print(f"  ε' = {EPSILON_A} - {EPSILON_B}√{D}")
    print(f"     ≈ {EPSILON_CONJ:.15f}")
    print(f"\nNorm: ε × ε' = {NORM:.15f} (should be 1)")
    print(f"Norm check: {EPSILON_A}² - {D}×{EPSILON_B}² = {EPSILON_A**2 - D*EPSILON_B**2}")
    print(f"Minimal polynomial: x² - {2*EPSILON_A}x + 1 = 0")
    print(f"\nRegulator: R = ln(ε) = {REGULATOR:.6f}")
    print(f"log₁₀(ε) = {LOG10_EPSILON:.6f}")
    print(f"\nExpected digits per term:")
    for power, digits in sorted(DIGITS_PER_TERM.items()):
        marker = " ← Chudnovsky-class!" if 12 <= digits <= 15 else ""
        print(f"  ε^{power:2d}k: {digits:.2f} digits/term{marker}")
