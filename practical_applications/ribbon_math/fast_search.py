"""
Fast Value Search Module
========================

Addresses the key bottleneck: searching for specific values quickly.

This module provides:
1. Vectorized PSLQ-style integer relation detection
2. Fast pattern matching against known mathematical constants
3. Cached computation of common bases and powers
4. Parallel search across multiple structures

Design Philosophy:
- Pre-compute everything possible
- Vectorize all inner loops
- Cache aggressively
- Fail fast on impossible cases
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import time

# Try to import high-precision library
try:
    from mpmath import mp, mpf, pi as mp_pi, sqrt as mp_sqrt, log, pslq, identify
    HAS_MPMATH = True
    mp.dps = 100  # Default precision
except ImportError:
    HAS_MPMATH = False
    print("Warning: mpmath not available. Using numpy (lower precision).")


# =============================================================================
# MATHEMATICAL CONSTANTS CACHE
# =============================================================================

@dataclass
class ConstantsCache:
    """Pre-computed mathematical constants for fast lookup."""
    
    # Fundamental constants
    pi: float = np.pi
    e: float = np.e
    phi: float = (1 + np.sqrt(5)) / 2
    sqrt2: float = np.sqrt(2)
    sqrt3: float = np.sqrt(3)
    sqrt5: float = np.sqrt(5)
    
    # Derived constants
    log_phi: float = field(default_factory=lambda: np.log((1 + np.sqrt(5)) / 2))
    arctan_phi: float = field(default_factory=lambda: np.arctan(2 / (1 + np.sqrt(5))))
    
    # Powers of phi (pre-computed for speed)
    phi_powers: np.ndarray = field(default_factory=lambda: np.array([
        ((1 + np.sqrt(5)) / 2) ** k for k in range(-20, 21)
    ]))
    
    # Common integer sequences
    fibonacci: np.ndarray = field(default_factory=lambda: np.array([
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181
    ]))
    lucas: np.ndarray = field(default_factory=lambda: np.array([
        2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364
    ]))
    primes: np.ndarray = field(default_factory=lambda: np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71
    ]))
    
    def __post_init__(self):
        # Ensure derived values are computed
        self.log_phi = np.log(self.phi)
        self.arctan_phi = np.arctan(1/self.phi)
        self.phi_powers = np.array([self.phi ** k for k in range(-20, 21)])


# Global constants cache
CONSTANTS = ConstantsCache()


# =============================================================================
# FAST PATTERN MATCHING
# =============================================================================

class FastPatternMatcher:
    """
    Fast pattern matching against known mathematical structures.
    
    Uses vectorized operations for speed.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Pre-build lookup tables for common patterns."""
        # Rational approximations: n/d for small n, d (reduced range for speed)
        max_int = 30  # Reduced from 100
        self.rationals = {}
        for d in range(1, max_int + 1):
            for n in range(-max_int, max_int + 1):
                val = n / d
                bucket = round(val * 1000)
                if bucket not in self.rationals:
                    self.rationals[bucket] = []
                self.rationals[bucket].append((n, d, val))
        
        # φ-patterns: (n/d) × φ^k (reduced range for speed)
        self.phi_patterns = []
        for k in range(-10, 6):  # Reduced from (-15, 10)
            phi_k = CONSTANTS.phi ** k
            for d in range(1, 20):  # Reduced from 50
                for n in range(-20, 21):  # Reduced from 50
                    if n == 0:
                        continue
                    val = (n / d) * phi_k
                    self.phi_patterns.append((n, d, k, val))
        
        # Sort by value for binary search
        self.phi_patterns.sort(key=lambda x: x[3])
        self.phi_values = np.array([p[3] for p in self.phi_patterns])
    
    def find_rational(self, value: float) -> Optional[Tuple[int, int, float]]:
        """Find rational approximation n/d for value."""
        bucket = round(value * 1000)
        
        best = None
        best_error = self.tolerance
        
        # Check nearby buckets
        for b in range(bucket - 2, bucket + 3):
            if b in self.rationals:
                for n, d, v in self.rationals[b]:
                    error = abs(value - v)
                    if error < best_error:
                        best_error = error
                        best = (n, d, error)
        
        return best
    
    def find_phi_pattern(self, value: float) -> Optional[Dict]:
        """Find (n/d) × φ^k approximation using binary search."""
        # Binary search for closest value
        idx = np.searchsorted(self.phi_values, value)
        
        best = None
        best_error = self.tolerance
        
        # Check nearby indices
        for i in range(max(0, idx - 5), min(len(self.phi_patterns), idx + 5)):
            n, d, k, v = self.phi_patterns[i]
            error = abs(value - v)
            if error < best_error:
                best_error = error
                best = {
                    'numerator': n,
                    'denominator': d,
                    'phi_power': k,
                    'value': v,
                    'error': error,
                    'expression': f"({n}/{d}) × φ^{k}"
                }
        
        return best
    
    def find_all_patterns(self, value: float) -> List[Dict]:
        """Find all matching patterns for a value."""
        patterns = []
        
        # Rational
        rat = self.find_rational(value)
        if rat:
            n, d, err = rat
            patterns.append({
                'type': 'rational',
                'expression': f"{n}/{d}",
                'value': n/d,
                'error': err
            })
        
        # φ-pattern
        phi = self.find_phi_pattern(value)
        if phi:
            phi['type'] = 'phi'
            patterns.append(phi)
        
        # arctan(1/φ) multiple
        arctan_ratio = value / CONSTANTS.arctan_phi
        rat_arctan = self.find_rational(arctan_ratio)
        if rat_arctan:
            n, d, _ = rat_arctan
            approx = (n/d) * CONSTANTS.arctan_phi
            error = abs(value - approx)
            if error < self.tolerance:
                patterns.append({
                    'type': 'arctan_phi',
                    'expression': f"({n}/{d}) × arctan(1/φ)",
                    'value': approx,
                    'error': error
                })
        
        # log(φ) multiple
        log_ratio = value / CONSTANTS.log_phi
        rat_log = self.find_rational(log_ratio)
        if rat_log:
            n, d, _ = rat_log
            approx = (n/d) * CONSTANTS.log_phi
            error = abs(value - approx)
            if error < self.tolerance:
                patterns.append({
                    'type': 'log_phi',
                    'expression': f"({n}/{d}) × log(φ)",
                    'value': approx,
                    'error': error
                })
        
        return sorted(patterns, key=lambda x: x['error'])


# =============================================================================
# FAST PSLQ WRAPPER
# =============================================================================

class FastPSLQ:
    """
    Fast PSLQ wrapper with caching and early termination.
    
    Key optimizations:
    1. Cache results for repeated queries
    2. Detect minimal polynomial hits (coefficient[0] = 0)
    3. Progressive precision (start low, increase if needed)
    4. Timeout support
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.stats = {'hits': 0, 'misses': 0, 'timeouts': 0}
    
    def _cache_key(self, values: Tuple[float, ...], precision: int) -> str:
        """Generate cache key from values."""
        return f"{hash(values)}_{precision}"
    
    def find_relation(self, values: List[float], 
                      target_idx: int = -1,
                      max_coef: int = 10**8,
                      precision: int = 50,
                      timeout: float = 5.0) -> Optional[List[int]]:
        """
        Find integer relation among values.
        
        Args:
            values: List of values to find relation for
            target_idx: Index of target value (check if coefficient is non-zero)
            max_coef: Maximum coefficient magnitude
            precision: Decimal precision for PSLQ
            timeout: Maximum time in seconds
            
        Returns:
            List of integer coefficients, or None if not found
        """
        if not HAS_MPMATH:
            return None
        
        # Check cache
        key = self._cache_key(tuple(values), precision)
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key]
        
        self.stats['misses'] += 1
        
        # Set precision
        old_dps = mp.dps
        mp.dps = precision
        
        try:
            # Convert to mpf
            mp_values = [mpf(str(v)) for v in values]
            
            # Run PSLQ with timeout
            start = time.time()
            result = pslq(mp_values, tol=10**(-precision//2), maxcoeff=max_coef)
            
            if time.time() - start > timeout:
                self.stats['timeouts'] += 1
                return None
            
            if result is None:
                self._cache_result(key, None)
                return None
            
            # Check if target coefficient is non-zero
            if target_idx >= 0 and result[target_idx] == 0:
                # PSLQ found minimal polynomial, not target relation
                self._cache_result(key, None)
                return None
            
            self._cache_result(key, result)
            return result
            
        except Exception as e:
            return None
        finally:
            mp.dps = old_dps
    
    def _cache_result(self, key: str, result: Optional[List[int]]):
        """Cache a result, evicting old entries if needed."""
        if len(self.cache) >= self.cache_size:
            # Simple FIFO eviction
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = result
    
    def find_algebraic(self, value: float, 
                       basis: List[float],
                       precision: int = 50) -> Optional[Dict]:
        """
        Express value in terms of basis elements.
        
        Args:
            value: Value to express
            basis: List of basis elements (e.g., [1, √29])
            precision: Decimal precision
            
        Returns:
            Dict with coefficients, or None
        """
        # Construct vector: [value, basis[0], basis[1], ...]
        vec = [value] + basis
        
        result = self.find_relation(vec, target_idx=0, precision=precision)
        
        if result is None or result[0] == 0:
            return None
        
        # Extract coefficients: value = -Σ(result[i] * basis[i-1]) / result[0]
        coefficients = [-result[i] / result[0] for i in range(1, len(result))]
        
        return {
            'coefficients': coefficients,
            'relation': result,
            'expression': ' + '.join(f"{c}×b{i}" for i, c in enumerate(coefficients) if c != 0)
        }


# =============================================================================
# FAST SERIES EVALUATOR
# =============================================================================

class FastSeriesEvaluator:
    """
    Vectorized series evaluation for BBP-type formulas.
    
    Pre-computes power tables for fast evaluation.
    """
    
    def __init__(self, base: float, n_terms: int = 100):
        self.base = base
        self.n_terms = n_terms
        self._build_power_table()
    
    def _build_power_table(self):
        """Pre-compute powers of 1/base."""
        # Use log-space to avoid overflow for large bases
        log_base = np.log(float(self.base))
        self.powers = np.exp(-log_base * np.arange(self.n_terms))
        self.signs = np.array([(-1) ** k for k in range(self.n_terms)])
        self.k_values = np.arange(self.n_terms)
    
    def evaluate_bbp(self, slots: List[Tuple[int, int]], 
                     coefficients: List[float],
                     scale: float = 1.0,
                     alternating: bool = True) -> float:
        """
        Evaluate BBP-type series:
        
        result = (1/scale) × Σ (±1)^k × (1/base)^k × Σ coef[i] / (period[i]*k + offset[i])
        
        Args:
            slots: List of (period, offset) tuples
            coefficients: Coefficient for each slot
            scale: Overall scale factor
            alternating: Use (-1)^k signs
            
        Returns:
            Series value
        """
        result = 0.0
        signs = self.signs if alternating else np.ones(self.n_terms)
        
        for k in range(self.n_terms):
            inner = 0.0
            for (period, offset), coef in zip(slots, coefficients):
                denom = period * k + offset
                if denom != 0 and coef != 0:
                    inner += coef / denom
            result += signs[k] * self.powers[k] * inner
        
        return result / scale
    
    def evaluate_bbp_vectorized(self, slots: List[Tuple[int, int]], 
                                 coefficients: np.ndarray,
                                 scale: float = 1.0,
                                 alternating: bool = True) -> float:
        """
        Fully vectorized BBP evaluation.
        
        Args:
            slots: List of (period, offset) tuples
            coefficients: Coefficient array
            scale: Overall scale factor
            alternating: Use (-1)^k signs
            
        Returns:
            Series value
        """
        n_slots = len(slots)
        
        # Build denominator matrix: denom[k, i] = period[i] * k + offset[i]
        periods = np.array([s[0] for s in slots])
        offsets = np.array([s[1] for s in slots])
        
        # k_values: (n_terms,), periods: (n_slots,)
        # denom: (n_terms, n_slots)
        denom = np.outer(self.k_values, periods) + offsets
        
        # Avoid division by zero
        denom = np.where(denom == 0, np.inf, denom)
        
        # inner[k] = Σ coef[i] / denom[k, i]
        inner = np.sum(coefficients / denom, axis=1)
        
        # result = Σ sign[k] * power[k] * inner[k]
        signs = self.signs if alternating else np.ones(self.n_terms)
        result = np.sum(signs * self.powers * inner)
        
        return result / scale


# =============================================================================
# UNIFIED FAST SEARCH
# =============================================================================

class FastSearch:
    """
    Unified fast search interface.
    
    Combines pattern matching, PSLQ, and series evaluation
    for rapid mathematical discovery.
    """
    
    def __init__(self, precision: int = 50):
        self.pattern_matcher = FastPatternMatcher()
        self.pslq = FastPSLQ()
        self.precision = precision
        
        # Pre-built evaluators for common bases
        self.evaluators = {
            16: FastSeriesEvaluator(16),
            64: FastSeriesEvaluator(64),
            1024: FastSeriesEvaluator(1024),
            4096: FastSeriesEvaluator(4096),
        }
    
    def get_evaluator(self, base: float) -> FastSeriesEvaluator:
        """Get or create evaluator for a base."""
        if base not in self.evaluators:
            self.evaluators[base] = FastSeriesEvaluator(base)
        return self.evaluators[base]
    
    def identify_value(self, value: float) -> List[Dict]:
        """
        Identify a value using all available methods.
        
        Returns list of possible identifications, sorted by confidence.
        """
        results = []
        
        # 1. Pattern matching (fastest)
        patterns = self.pattern_matcher.find_all_patterns(value)
        for p in patterns:
            p['method'] = 'pattern'
            p['confidence'] = 1.0 - min(1.0, p['error'] * 1e6)
            results.append(p)
        
        # 2. mpmath identify (if available)
        if HAS_MPMATH:
            try:
                mp.dps = self.precision
                identified = identify(mpf(str(value)))
                if identified:
                    results.append({
                        'type': 'mpmath_identify',
                        'expression': identified,
                        'method': 'mpmath',
                        'confidence': 0.9,
                        'error': 0.0  # mpmath doesn't give error
                    })
            except:
                pass
        
        # Sort by confidence
        return sorted(results, key=lambda x: -x.get('confidence', 0))
    
    def search_bbp_coefficients(self, target: float,
                                 base: float,
                                 slots: List[Tuple[int, int]],
                                 scale: float = 1.0,
                                 max_coef: int = 512) -> Optional[Dict]:
        """
        Search for integer BBP coefficients that give target value.
        
        Uses vectorized evaluation for speed.
        
        Args:
            target: Target value (e.g., π)
            base: Series base
            slots: List of (period, offset) tuples
            scale: Overall scale factor
            max_coef: Maximum coefficient magnitude
            
        Returns:
            Dict with coefficients and error, or None
        """
        evaluator = self.get_evaluator(base)
        n_slots = len(slots)
        
        best_result = None
        best_error = float('inf')
        
        # For small slot counts, try exhaustive search
        if n_slots <= 4 and max_coef <= 32:
            # Exhaustive search
            from itertools import product
            
            for coefs in product(range(-max_coef, max_coef + 1), repeat=n_slots):
                if all(c == 0 for c in coefs):
                    continue
                
                coef_array = np.array(coefs, dtype=float)
                value = evaluator.evaluate_bbp_vectorized(slots, coef_array, scale)
                error = abs(value - target)
                
                if error < best_error:
                    best_error = error
                    best_result = {
                        'coefficients': list(coefs),
                        'value': value,
                        'error': error,
                        'digits': -np.log10(error) if error > 0 else 50
                    }
        else:
            # Use optimization for larger searches
            from scipy.optimize import minimize
            
            def objective(coefs):
                value = evaluator.evaluate_bbp_vectorized(slots, coefs, scale)
                return (value - target) ** 2
            
            # Try multiple starting points
            for _ in range(10):
                x0 = np.random.randint(-max_coef//4, max_coef//4 + 1, n_slots).astype(float)
                result = minimize(objective, x0, method='Nelder-Mead')
                
                # Round to integers
                int_coefs = np.round(result.x).astype(int)
                value = evaluator.evaluate_bbp_vectorized(slots, int_coefs.astype(float), scale)
                error = abs(value - target)
                
                if error < best_error:
                    best_error = error
                    best_result = {
                        'coefficients': int_coefs.tolist(),
                        'value': value,
                        'error': error,
                        'digits': -np.log10(error) if error > 0 else 50
                    }
        
        return best_result
    
    def stats(self) -> Dict:
        """Return search statistics."""
        return {
            'pslq_cache_hits': self.pslq.stats['hits'],
            'pslq_cache_misses': self.pslq.stats['misses'],
            'pslq_timeouts': self.pslq.stats['timeouts'],
            'evaluators_cached': len(self.evaluators)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global fast search instance
_fast_search = None

def get_fast_search() -> FastSearch:
    """Get global FastSearch instance."""
    global _fast_search
    if _fast_search is None:
        _fast_search = FastSearch()
    return _fast_search


def identify(value: float) -> List[Dict]:
    """Quick identification of a value."""
    return get_fast_search().identify_value(value)


def find_bbp(target: float, base: float, slots: List[Tuple[int, int]], 
             scale: float = 1.0, max_coef: int = 512) -> Optional[Dict]:
    """Quick BBP coefficient search."""
    return get_fast_search().search_bbp_coefficients(target, base, slots, scale, max_coef)


# =============================================================================
# RIBBON LCM INTEGRATION
# =============================================================================

class RibbonSearcher:
    """
    Fast search integrated with Ribbon LCM vocabulary.
    
    Expresses search results in Ribbon Speech using anchor concepts.
    """
    
    def __init__(self):
        self.fast_search = FastSearch()
        
        # Ribbon LCM anchor values
        self.anchors = {
            "zero": 0.0,
            "sierpinski": np.log(3)/np.log(2),  # ~1.585
            "phi": (1 + np.sqrt(5)) / 2,         # ~1.618
            "e_inv": 1 / np.e,                   # ~0.368
            "cantor": np.log(2)/np.log(3),       # ~0.631
            "sqrt2_inv": 1 / np.sqrt(2),         # ~0.707
        }
        
        # Anchor semantic meanings
        self.anchor_meanings = {
            "zero": "origin/potential",
            "sierpinski": "pattern/fractal",
            "phi": "growth/harmony",
            "e_inv": "decay/change",
            "cantor": "discrete/boundary",
            "sqrt2_inv": "bridge/connection",
        }
    
    def identify_with_anchors(self, value: float) -> Dict:
        """
        Identify a value and express it in terms of Ribbon LCM anchors.
        
        Returns both the mathematical identification and the semantic meaning.
        """
        result = {
            'value': value,
            'patterns': [],
            'nearest_anchor': None,
            'ribbon_speech': None,
        }
        
        # Find nearest anchor
        min_dist = float('inf')
        for name, anchor_val in self.anchors.items():
            dist = abs(value - anchor_val)
            if dist < min_dist:
                min_dist = dist
                result['nearest_anchor'] = {
                    'name': name,
                    'value': anchor_val,
                    'distance': dist,
                    'meaning': self.anchor_meanings[name]
                }
        
        # Check if value IS an anchor (within tolerance)
        if min_dist < 1e-10:
            anchor = result['nearest_anchor']
            result['ribbon_speech'] = f"This is {anchor['name']} ({anchor['meaning']})"
            result['is_anchor'] = True
            return result
        
        # Get mathematical patterns
        result['patterns'] = self.fast_search.identify_value(value)
        
        # Check for anchor ratios
        for name1, val1 in self.anchors.items():
            if val1 == 0:
                continue
            for name2, val2 in self.anchors.items():
                if val2 == 0:
                    continue
                # Check value = val1 / val2
                if abs(value - val1/val2) < 1e-10:
                    result['anchor_ratio'] = f"{name1}/{name2}"
                    result['ribbon_speech'] = f"{name1} divided by {name2}"
                # Check value = val1 * val2
                if abs(value - val1*val2) < 1e-10:
                    result['anchor_product'] = f"{name1}×{name2}"
                    result['ribbon_speech'] = f"{name1} times {name2}"
        
        # Generate Ribbon Speech from patterns
        if not result.get('ribbon_speech') and result['patterns']:
            p = result['patterns'][0]
            anchor = result['nearest_anchor']
            result['ribbon_speech'] = (
                f"Near {anchor['name']} ({anchor['meaning']}), "
                f"expressed as {p.get('expression', 'unknown pattern')}"
            )
        
        return result
    
    def search_equation(self, target_name: str, target_value: float) -> Dict:
        """
        Search for a value and express the result as a Ribbon equation.
        """
        result = self.identify_with_anchors(target_value)
        result['equation_name'] = target_name
        
        # Build anchor vector (6D)
        anchor_order = ["zero", "sierpinski", "phi", "e_inv", "cantor", "sqrt2_inv"]
        anchor_vector = np.zeros(6)
        
        nearest = result.get('nearest_anchor', {})
        if nearest:
            idx = anchor_order.index(nearest['name'])
            anchor_vector[idx] = 1.0 - min(1.0, nearest['distance'])
        
        result['anchor_vector'] = anchor_vector.tolist()
        
        return result


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RIBBON LCM FAST SEARCH")
    print("=" * 60)
    
    rs = RibbonSearcher()
    
    # Test 1: Identify anchor values
    print("\n1. ANCHOR IDENTIFICATION")
    print("-" * 40)
    
    for name, val in rs.anchors.items():
        result = rs.identify_with_anchors(val)
        print(f"{name:12} = {val:.10f}")
        print(f"  → {result['ribbon_speech']}")
    
    # Test 2: Identify derived values
    print("\n2. DERIVED VALUES")
    print("-" * 40)
    
    derived = [
        ("1/φ", 1/rs.anchors['phi']),
        ("φ²", rs.anchors['phi']**2),
        ("log(φ)", np.log(rs.anchors['phi'])),
        ("arctan(1/φ)", np.arctan(1/rs.anchors['phi'])),
        ("π", np.pi),
        ("e", np.e),
    ]
    
    for name, val in derived:
        result = rs.identify_with_anchors(val)
        print(f"\n{name} = {val:.10f}")
        print(f"  Nearest anchor: {result['nearest_anchor']['name']}")
        print(f"  → {result['ribbon_speech']}")
    
    # Test 3: Speed (just anchor lookup, not full pattern matching)
    print("\n3. SPEED TEST")
    print("-" * 40)
    
    import time
    
    # Fast: just anchor distance
    start = time.time()
    for _ in range(10000):
        val = 0.618033988749895
        min_dist = min(abs(val - v) for v in rs.anchors.values())
    elapsed = time.time() - start
    print(f"  Anchor lookup: {10000/elapsed:.0f}/sec")
    
    # Medium: pattern matching without mpmath
    start = time.time()
    for _ in range(100):
        rs.fast_search.pattern_matcher.find_phi_pattern(0.618)
    elapsed = time.time() - start
    print(f"  Pattern match: {100/elapsed:.0f}/sec")
    
    print("\n" + "=" * 60)
