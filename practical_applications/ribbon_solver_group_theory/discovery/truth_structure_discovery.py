"""
Truth Structure Discovery Engine

Discovers the mathematical rules governing truth space by:
1. Exploring the space with minimal constraints
2. Analyzing boundary patterns
3. Finding mathematical relationships between coordinates
4. Formulating constraint equations

This is analogous to the Ribbon LCM discovery engine but for truth space geometry.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Set
from collections import defaultdict
import itertools


# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PHI_INV = 1 / PHI
SQRT2 = np.sqrt(2)
SQRT2_INV = 1 / SQRT2
E_INV = 1 / np.e


@dataclass
class DiscoveredRelation:
    """A discovered mathematical relationship."""
    name: str
    description: str
    equation: str  # Human-readable equation
    test_fn: Callable[[np.ndarray], float]  # Returns residual (0 = perfect match)
    confidence: float  # 0-1, based on how many points satisfy it
    examples: List[np.ndarray] = field(default_factory=list)


@dataclass 
class DiscoveryConfig:
    """Configuration for truth structure discovery."""
    # Exploration parameters
    n_boundary_samples: int = 10000
    n_interior_samples: int = 5000
    
    # Analysis parameters
    residual_threshold: float = 0.01  # Max residual to consider a relation valid
    min_confidence: float = 0.8  # Minimum confidence to report a relation
    
    # Search parameters
    max_polynomial_degree: int = 3
    test_golden_ratios: bool = True
    test_symmetries: bool = True
    test_sum_rules: bool = True


class TruthStructureDiscovery:
    """
    Discovers mathematical structure in truth space.
    
    Similar to how Ribbon LCM discovers patterns in BBP formulas,
    this discovers geometric constraints in truth space.
    """
    
    # Anchor names for the 6D truth space
    ANCHORS = ['identity', 'pattern', 'structure', 'unity', 'ground', 'inverse']
    
    def __init__(self, config: DiscoveryConfig = None):
        self.config = config or DiscoveryConfig()
        self.boundary_points: List[np.ndarray] = []
        self.interior_points: List[np.ndarray] = []
        self.discovered_relations: List[DiscoveredRelation] = []
        
        # Candidate relations to test
        self._candidate_relations: List[Tuple[str, str, Callable]] = []
        self._build_candidate_relations()
    
    def _build_candidate_relations(self):
        """Build list of candidate mathematical relations to test."""
        
        # Sum rules: subsets of coordinates sum to specific values
        self._add_sum_rules()
        
        # Ratio rules: ratios between coordinates equal constants
        self._add_ratio_rules()
        
        # Product rules: products of coordinates equal constants
        self._add_product_rules()
        
        # Polynomial rules: polynomial combinations equal zero
        self._add_polynomial_rules()
        
        # Symmetry rules: coordinate permutations preserve validity
        self._add_symmetry_rules()
        
        # Golden ratio rules
        if self.config.test_golden_ratios:
            self._add_golden_rules()
    
    def _add_sum_rules(self):
        """Add candidate sum rules."""
        # Pairs summing to specific values
        for i, j in itertools.combinations(range(6), 2):
            for target in [0.5, 1/3, 1/PHI, PHI_INV, 0.25]:
                name = f"sum_{self.ANCHORS[i]}_{self.ANCHORS[j]}={target:.4f}"
                desc = f"{self.ANCHORS[i]} + {self.ANCHORS[j]} = {target:.4f}"
                
                def make_test(ii, jj, t):
                    return lambda p: abs(p[ii] + p[jj] - t)
                
                self._candidate_relations.append((name, desc, make_test(i, j, target)))
        
        # Triplets summing to specific values
        for i, j, k in itertools.combinations(range(6), 3):
            for target in [0.5, 2/3, 1/PHI]:
                name = f"sum3_{i}_{j}_{k}={target:.4f}"
                desc = f"{self.ANCHORS[i]} + {self.ANCHORS[j]} + {self.ANCHORS[k]} = {target:.4f}"
                
                def make_test(ii, jj, kk, t):
                    return lambda p: abs(p[ii] + p[jj] + p[kk] - t)
                
                self._candidate_relations.append((name, desc, make_test(i, j, k, target)))
    
    def _add_ratio_rules(self):
        """Add candidate ratio rules."""
        for i, j in itertools.combinations(range(6), 2):
            for ratio in [PHI, PHI_INV, 2.0, 0.5, SQRT2, SQRT2_INV]:
                name = f"ratio_{self.ANCHORS[i]}_{self.ANCHORS[j]}={ratio:.4f}"
                desc = f"{self.ANCHORS[i]} / {self.ANCHORS[j]} = {ratio:.4f}"
                
                def make_test(ii, jj, r):
                    def test(p):
                        if abs(p[jj]) < 1e-10:
                            return float('inf')
                        return abs(p[ii] / p[jj] - r)
                    return test
                
                self._candidate_relations.append((name, desc, make_test(i, j, ratio)))
    
    def _add_product_rules(self):
        """Add candidate product rules."""
        # Pairs
        for i, j in itertools.combinations(range(6), 2):
            for target in [1/6, 1/PHI**2, 0.1, 0.05]:
                name = f"prod_{self.ANCHORS[i]}_{self.ANCHORS[j]}={target:.4f}"
                desc = f"{self.ANCHORS[i]} × {self.ANCHORS[j]} = {target:.4f}"
                
                def make_test(ii, jj, t):
                    return lambda p: abs(p[ii] * p[jj] - t)
                
                self._candidate_relations.append((name, desc, make_test(i, j, target)))
    
    def _add_polynomial_rules(self):
        """Add polynomial constraint rules."""
        # Quadratic: a*x^2 + b*y^2 = c
        for i, j in itertools.combinations(range(6), 2):
            # Ellipse-like constraints
            for a, b, c in [(1, 1, 0.1), (PHI, 1, 0.1), (1, PHI, 0.1)]:
                name = f"quad_{i}_{j}_{a:.2f}_{b:.2f}_{c:.2f}"
                desc = f"{a:.2f}×{self.ANCHORS[i]}² + {b:.2f}×{self.ANCHORS[j]}² = {c:.2f}"
                
                def make_test(ii, jj, aa, bb, cc):
                    return lambda p: abs(aa * p[ii]**2 + bb * p[jj]**2 - cc)
                
                self._candidate_relations.append((name, desc, make_test(i, j, a, b, c)))
    
    def _add_symmetry_rules(self):
        """Add symmetry-based rules."""
        # Test if swapping certain coordinates preserves validity
        # This is tested differently - by checking if symmetric points are also valid
        pass
    
    def _add_golden_rules(self):
        """Add golden ratio specific rules."""
        # Fibonacci-like recurrence in coordinates
        for i in range(4):
            name = f"fib_recurrence_{i}"
            desc = f"{self.ANCHORS[i+2]} = {self.ANCHORS[i+1]} + {self.ANCHORS[i]} (Fibonacci-like)"
            
            def make_test(ii):
                return lambda p: abs(p[ii+2] - p[ii+1] - p[ii])
            
            self._candidate_relations.append((name, desc, make_test(i)))
        
        # Golden spiral: each coordinate is phi times the previous
        name = "golden_spiral"
        desc = "Each coordinate is φ times the previous"
        
        def golden_spiral_test(p):
            residuals = []
            for i in range(5):
                if abs(p[i]) > 1e-10:
                    residuals.append(abs(p[i+1] / p[i] - PHI_INV))
            return np.mean(residuals) if residuals else float('inf')
        
        self._candidate_relations.append((name, desc, golden_spiral_test))
    
    def explore_boundary(self, validity_fn: Callable[[np.ndarray], bool],
                        n_samples: int = None) -> List[np.ndarray]:
        """
        Find boundary points of the valid region.
        
        Uses binary search from interior to exterior to find boundary.
        """
        n_samples = n_samples or self.config.n_boundary_samples
        
        # Start from center of simplex
        center = np.ones(6) / 6
        
        boundary_points = []
        
        # Generate random directions on the simplex
        for _ in range(n_samples):
            # Random direction (tangent to simplex)
            direction = np.random.randn(6)
            direction = direction - np.mean(direction)  # Project to simplex tangent
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Binary search for boundary
            low, high = 0.0, 1.0
            
            # First find a point outside valid region
            while high < 10.0:
                test_point = center + high * direction
                test_point = np.clip(test_point, 0, 1)
                test_point = test_point / np.sum(test_point)  # Normalize to simplex
                
                if not validity_fn(test_point):
                    break
                high *= 2
            
            if high >= 10.0:
                continue  # No boundary found in this direction
            
            # Binary search
            for _ in range(20):  # ~20 iterations gives ~1e-6 precision
                mid = (low + high) / 2
                test_point = center + mid * direction
                test_point = np.clip(test_point, 0, 1)
                test_point = test_point / np.sum(test_point)
                
                if validity_fn(test_point):
                    low = mid
                else:
                    high = mid
            
            # Record boundary point
            boundary_point = center + low * direction
            boundary_point = np.clip(boundary_point, 0, 1)
            boundary_point = boundary_point / np.sum(boundary_point)
            boundary_points.append(boundary_point)
        
        self.boundary_points = boundary_points
        return boundary_points
    
    def sample_interior(self, validity_fn: Callable[[np.ndarray], bool],
                       n_samples: int = None) -> List[np.ndarray]:
        """Sample valid interior points."""
        n_samples = n_samples or self.config.n_interior_samples
        
        interior_points = []
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(interior_points) < n_samples and attempts < max_attempts:
            # Random point on simplex
            point = np.random.exponential(1, 6)
            point = point / np.sum(point)
            
            if validity_fn(point):
                interior_points.append(point)
            
            attempts += 1
        
        self.interior_points = interior_points
        return interior_points
    
    def test_relation(self, relation: Tuple[str, str, Callable],
                     points: List[np.ndarray]) -> Optional[DiscoveredRelation]:
        """Test if a relation holds for the given points."""
        name, desc, test_fn = relation
        
        residuals = []
        satisfying_points = []
        
        for point in points:
            try:
                residual = test_fn(point)
                if np.isfinite(residual):
                    residuals.append(residual)
                    if residual < self.config.residual_threshold:
                        satisfying_points.append(point)
            except:
                pass
        
        if not residuals:
            return None
        
        confidence = len(satisfying_points) / len(residuals)
        
        if confidence >= self.config.min_confidence:
            return DiscoveredRelation(
                name=name,
                description=desc,
                equation=desc,
                test_fn=test_fn,
                confidence=confidence,
                examples=satisfying_points[:5]
            )
        
        return None
    
    def discover(self, validity_fn: Callable[[np.ndarray], bool],
                use_boundary: bool = True,
                use_interior: bool = True,
                verbose: bool = True) -> List[DiscoveredRelation]:
        """
        Main discovery process.
        
        1. Explore boundary and interior
        2. Test all candidate relations
        3. Return discovered relations sorted by confidence
        """
        if verbose:
            print("=" * 60)
            print("TRUTH STRUCTURE DISCOVERY")
            print("=" * 60)
        
        # Collect points to analyze
        points = []
        
        if use_boundary:
            if verbose:
                print(f"\nExploring boundary ({self.config.n_boundary_samples} samples)...")
            boundary = self.explore_boundary(validity_fn)
            points.extend(boundary)
            if verbose:
                print(f"  Found {len(boundary)} boundary points")
        
        if use_interior:
            if verbose:
                print(f"\nSampling interior ({self.config.n_interior_samples} samples)...")
            interior = self.sample_interior(validity_fn)
            points.extend(interior)
            if verbose:
                print(f"  Found {len(interior)} interior points")
        
        if not points:
            print("No points found!")
            return []
        
        # Test all candidate relations
        if verbose:
            print(f"\nTesting {len(self._candidate_relations)} candidate relations...")
        
        discovered = []
        for relation in self._candidate_relations:
            result = self.test_relation(relation, points)
            if result:
                discovered.append(result)
        
        # Sort by confidence
        discovered.sort(key=lambda r: r.confidence, reverse=True)
        
        self.discovered_relations = discovered
        
        if verbose:
            print(f"\nDiscovered {len(discovered)} relations:")
            for i, rel in enumerate(discovered[:10]):  # Top 10
                print(f"  {i+1}. [{rel.confidence:.1%}] {rel.description}")
        
        return discovered
    
    def analyze_coordinate_distributions(self, points: List[np.ndarray]) -> Dict:
        """Analyze statistical properties of coordinates."""
        if not points:
            return {}
        
        points_array = np.array(points)
        
        analysis = {
            'means': np.mean(points_array, axis=0),
            'stds': np.std(points_array, axis=0),
            'mins': np.min(points_array, axis=0),
            'maxs': np.max(points_array, axis=0),
            'correlations': np.corrcoef(points_array.T),
        }
        
        # Check for special values
        for i, anchor in enumerate(self.ANCHORS):
            mean = analysis['means'][i]
            
            # Check if mean is close to special values
            special_values = {
                '1/6': 1/6,
                '1/φ': PHI_INV,
                '1/φ²': PHI_INV**2,
                '1/√2': SQRT2_INV,
                '1/e': E_INV,
            }
            
            for name, val in special_values.items():
                if abs(mean - val) < 0.01:
                    print(f"  {anchor} mean ≈ {name} ({mean:.4f} vs {val:.4f})")
        
        return analysis
    
    def find_symmetry_groups(self, validity_fn: Callable[[np.ndarray], bool],
                            n_tests: int = 1000) -> List[Tuple[int, ...]]:
        """
        Find coordinate permutations that preserve validity.
        
        Returns list of permutation tuples that are symmetries.
        """
        symmetries = []
        
        # Test all 720 permutations of 6 elements (6!)
        # But that's expensive, so we test common ones first
        
        # Pair swaps
        for i, j in itertools.combinations(range(6), 2):
            perm = list(range(6))
            perm[i], perm[j] = perm[j], perm[i]
            
            if self._test_permutation_symmetry(validity_fn, tuple(perm), n_tests):
                symmetries.append(tuple(perm))
        
        # Cyclic permutations
        for shift in range(1, 6):
            perm = tuple((i + shift) % 6 for i in range(6))
            if self._test_permutation_symmetry(validity_fn, perm, n_tests):
                symmetries.append(perm)
        
        return symmetries
    
    def _test_permutation_symmetry(self, validity_fn: Callable[[np.ndarray], bool],
                                   perm: Tuple[int, ...], n_tests: int) -> bool:
        """Test if a permutation preserves validity."""
        matches = 0
        
        for _ in range(n_tests):
            # Random point on simplex
            point = np.random.exponential(1, 6)
            point = point / np.sum(point)
            
            # Permuted point
            permuted = point[list(perm)]
            
            # Check if validity is preserved
            orig_valid = validity_fn(point)
            perm_valid = validity_fn(permuted)
            
            if orig_valid == perm_valid:
                matches += 1
        
        return matches / n_tests > 0.95
    
    def generate_validity_function(self) -> Callable[[np.ndarray], bool]:
        """
        Generate a validity function from discovered relations.
        
        This creates a composite validity test based on all discovered relations.
        """
        if not self.discovered_relations:
            # Default: just simplex constraint
            def default_validity(point):
                return (np.all(point >= 0) and 
                       np.all(point <= 1) and 
                       abs(np.sum(point) - 1) < 0.01)
            return default_validity
        
        relations = self.discovered_relations
        threshold = self.config.residual_threshold
        
        def composite_validity(point: np.ndarray) -> bool:
            # Basic simplex check
            if np.any(point < 0) or np.any(point > 1):
                return False
            if abs(np.sum(point) - 1) > 0.01:
                return False
            
            # Check discovered relations
            for rel in relations:
                try:
                    residual = rel.test_fn(point)
                    if residual > threshold * 2:  # Allow some slack
                        return False
                except:
                    pass
            
            return True
        
        return composite_validity
    
    def report(self) -> str:
        """Generate a human-readable report of discoveries."""
        lines = [
            "=" * 60,
            "TRUTH STRUCTURE DISCOVERY REPORT",
            "=" * 60,
            "",
            f"Boundary points analyzed: {len(self.boundary_points)}",
            f"Interior points analyzed: {len(self.interior_points)}",
            f"Relations discovered: {len(self.discovered_relations)}",
            "",
        ]
        
        if self.discovered_relations:
            lines.append("TOP DISCOVERED RELATIONS:")
            lines.append("-" * 40)
            
            for i, rel in enumerate(self.discovered_relations[:15]):
                lines.append(f"{i+1}. [{rel.confidence:.1%}] {rel.description}")
            
            lines.append("")
            lines.append("INTERPRETATION:")
            lines.append("-" * 40)
            
            # Group by type
            sum_rules = [r for r in self.discovered_relations if 'sum' in r.name]
            ratio_rules = [r for r in self.discovered_relations if 'ratio' in r.name]
            golden_rules = [r for r in self.discovered_relations if 'golden' in r.name or 'fib' in r.name]
            
            if sum_rules:
                lines.append(f"  Sum constraints: {len(sum_rules)}")
            if ratio_rules:
                lines.append(f"  Ratio constraints: {len(ratio_rules)}")
            if golden_rules:
                lines.append(f"  Golden ratio patterns: {len(golden_rules)}")
        
        return "\n".join(lines)


class ErrorAsSignalAnalyzer:
    """
    Applies the "Error as Signal" paradigm from Ribbon LCM v4.
    
    Instead of looking for exact relationships, we analyze the
    DEVIATIONS from simple patterns to find hidden structure.
    """
    
    ANCHORS = ['identity', 'pattern', 'structure', 'unity', 'ground', 'inverse']
    
    def __init__(self):
        self.deviations: Dict[str, List[float]] = defaultdict(list)
        self.patterns_found: List[Dict] = []
    
    def analyze_deviations(self, points: List[np.ndarray]) -> Dict:
        """
        Analyze deviations from expected patterns.
        
        Key insight: If coordinates should sum to 1/6 each (uniform),
        the deviations from 1/6 may contain structure.
        """
        if not points:
            return {}
        
        points_array = np.array(points)
        expected_uniform = 1/6
        
        results = {
            'uniform_deviations': {},
            'ratio_deviations': {},
            'golden_deviations': {},
        }
        
        # Deviation from uniform distribution
        for i, anchor in enumerate(self.ANCHORS):
            deviations = points_array[:, i] - expected_uniform
            results['uniform_deviations'][anchor] = {
                'mean': float(np.mean(deviations)),
                'std': float(np.std(deviations)),
                'skew': float(self._skewness(deviations)),
            }
        
        # Check if deviations follow φ^(-k) pattern (like BBP discovery)
        for i, anchor in enumerate(self.ANCHORS):
            deviations = points_array[:, i] - expected_uniform
            mean_dev = np.mean(np.abs(deviations))
            
            # Test against powers of φ
            for k in range(1, 12):
                phi_power = PHI ** (-k)
                if abs(mean_dev - phi_power) < 0.01:
                    results['golden_deviations'][anchor] = {
                        'power': k,
                        'expected': phi_power,
                        'actual': mean_dev,
                        'error': abs(mean_dev - phi_power)
                    }
                    break
        
        # Analyze pairwise deviation ratios
        for i in range(6):
            for j in range(i + 1, 6):
                dev_i = points_array[:, i] - expected_uniform
                dev_j = points_array[:, j] - expected_uniform
                
                # Avoid division by zero
                mask = np.abs(dev_j) > 0.01
                if np.sum(mask) > 100:
                    ratios = dev_i[mask] / dev_j[mask]
                    mean_ratio = np.mean(ratios)
                    
                    # Check if ratio is close to φ or 1/φ
                    for target, name in [(PHI, 'φ'), (PHI_INV, '1/φ'), (2.0, '2'), (0.5, '1/2')]:
                        if abs(mean_ratio - target) < 0.1:
                            key = f"{self.ANCHORS[i]}/{self.ANCHORS[j]}"
                            results['ratio_deviations'][key] = {
                                'ratio': mean_ratio,
                                'target': target,
                                'name': name,
                                'error': abs(mean_ratio - target)
                            }
        
        return results
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def find_hidden_structure(self, points: List[np.ndarray]) -> List[Dict]:
        """
        Look for hidden mathematical structure in the point distribution.
        
        This is the core of the "error as signal" approach.
        """
        if not points:
            return []
        
        points_array = np.array(points)
        structures = []
        
        # 1. Check for Fibonacci-like recurrence in sorted coordinates
        for point in points_array[:100]:  # Sample
            sorted_coords = np.sort(point)[::-1]
            
            # Check if each coord is sum of next two (Fibonacci)
            for i in range(len(sorted_coords) - 2):
                if abs(sorted_coords[i] - sorted_coords[i+1] - sorted_coords[i+2]) < 0.02:
                    structures.append({
                        'type': 'fibonacci_recurrence',
                        'position': i,
                        'values': sorted_coords[i:i+3].tolist()
                    })
        
        # 2. Check for golden spiral in coordinate magnitudes
        for point in points_array[:100]:
            sorted_coords = np.sort(point)[::-1]
            ratios = []
            for i in range(len(sorted_coords) - 1):
                if sorted_coords[i+1] > 0.01:
                    ratios.append(sorted_coords[i] / sorted_coords[i+1])
            
            if ratios:
                mean_ratio = np.mean(ratios)
                if abs(mean_ratio - PHI) < 0.1:
                    structures.append({
                        'type': 'golden_spiral',
                        'mean_ratio': mean_ratio,
                        'error': abs(mean_ratio - PHI)
                    })
        
        # 3. Check for self-similarity (Sierpiński-like)
        # Points at different scales should have similar distribution
        center = np.mean(points_array, axis=0)
        distances = np.linalg.norm(points_array - center, axis=1)
        
        # Split into shells
        shells = np.percentile(distances, [25, 50, 75])
        shell_distributions = []
        
        for i in range(3):
            if i == 0:
                mask = distances < shells[0]
            elif i == 1:
                mask = (distances >= shells[0]) & (distances < shells[1])
            else:
                mask = distances >= shells[1]
            
            if np.sum(mask) > 10:
                shell_points = points_array[mask]
                shell_mean = np.mean(shell_points, axis=0)
                shell_distributions.append(shell_mean)
        
        if len(shell_distributions) >= 2:
            # Check if shell distributions are similar (self-similar)
            similarity = np.corrcoef(shell_distributions[0], shell_distributions[-1])[0, 1]
            if similarity > 0.9:
                structures.append({
                    'type': 'self_similar',
                    'correlation': similarity,
                    'shells': len(shell_distributions)
                })
        
        # Count unique structure types
        type_counts = defaultdict(int)
        for s in structures:
            type_counts[s['type']] += 1
        
        self.patterns_found = [
            {'type': t, 'count': c, 'fraction': c / len(structures) if structures else 0}
            for t, c in type_counts.items()
        ]
        
        return structures
    
    def report(self, points: List[np.ndarray]) -> str:
        """Generate a report of error-as-signal analysis."""
        deviations = self.analyze_deviations(points)
        structures = self.find_hidden_structure(points)
        
        lines = [
            "=" * 60,
            "ERROR-AS-SIGNAL ANALYSIS",
            "(Inspired by Ribbon LCM v4 φ-BBP Discovery)",
            "=" * 60,
            "",
        ]
        
        # Uniform deviations
        lines.append("DEVIATIONS FROM UNIFORM (1/6):")
        lines.append("-" * 40)
        for anchor, stats in deviations.get('uniform_deviations', {}).items():
            lines.append(f"  {anchor}: mean={stats['mean']:+.4f}, "
                        f"std={stats['std']:.4f}, skew={stats['skew']:+.2f}")
        
        # Golden ratio patterns
        if deviations.get('golden_deviations'):
            lines.append("")
            lines.append("GOLDEN RATIO PATTERNS (φ^-k):")
            lines.append("-" * 40)
            for anchor, info in deviations['golden_deviations'].items():
                lines.append(f"  {anchor}: deviation ≈ φ^(-{info['power']}) = {info['expected']:.4f}")
        
        # Ratio patterns
        if deviations.get('ratio_deviations'):
            lines.append("")
            lines.append("DEVIATION RATIO PATTERNS:")
            lines.append("-" * 40)
            for pair, info in deviations['ratio_deviations'].items():
                lines.append(f"  {pair} ≈ {info['name']} ({info['ratio']:.3f})")
        
        # Hidden structures
        if self.patterns_found:
            lines.append("")
            lines.append("HIDDEN STRUCTURES FOUND:")
            lines.append("-" * 40)
            for p in self.patterns_found:
                lines.append(f"  {p['type']}: {p['count']} instances ({p['fraction']:.1%})")
        
        return "\n".join(lines)


def demo_discovery():
    """Demonstrate the discovery engine."""
    from ..visualizations.truth_space_explorer import create_mathematical_validity_fn
    
    print("Truth Structure Discovery Demo")
    print("=" * 60)
    
    # Use a more restrictive validity function to have interesting structure
    validity_fn = create_mathematical_validity_fn("balanced")
    
    config = DiscoveryConfig(
        n_boundary_samples=5000,
        n_interior_samples=2000,
        residual_threshold=0.02,
        min_confidence=0.7,
    )
    
    discovery = TruthStructureDiscovery(config)
    
    # Run discovery
    relations = discovery.discover(validity_fn, verbose=True)
    
    # Print report
    print("\n" + discovery.report())
    
    # Analyze distributions
    print("\nCoordinate Analysis:")
    print("-" * 40)
    all_points = discovery.boundary_points + discovery.interior_points
    discovery.analyze_coordinate_distributions(all_points)
    
    return discovery


if __name__ == "__main__":
    demo_discovery()
