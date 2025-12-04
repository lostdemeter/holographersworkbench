"""
Group-Theoretic Discovery Engine
=================================

Automated mathematical discovery using group theory.

This module provides the discovery interface, reimagining ribbon_solver3's
functionality through group-theoretic operations:

Original ribbon_solver3          →  Group Theory Version
─────────────────────────────────────────────────────────
optimize_file()                  →  find_symmetries() 
find_formula()                   →  find_orbit()
find_identities()                →  find_fixed_points()
analyze()                        →  classify()

Key Concepts:
    - **Symmetry**: A transformation that leaves a truth invariant
    - **Orbit**: All truths reachable from a starting point
    - **Fixed Point**: A truth that equals its transformation (identity)
    - **Conjugacy Class**: Equivalent truths under group action

Philosophy:
    SYMMETRY IS TRUTH - Mathematical identities are symmetries of truth space.
    Finding identities = finding fixed points of group actions.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
import json
import re

from .core import (
    TruthGroup, GroupElement, AnchorVector, Subgroup, 
    Anchor, ANCHOR_NAMES, ANCHOR_VALUES
)


# =============================================================================
# DISCOVERY RESULT
# =============================================================================

@dataclass
class DiscoveryResult:
    """Result from any group-theoretic discovery operation."""
    success: bool
    type: str  # 'symmetry', 'orbit', 'fixed_point', 'conjugacy'
    source: str
    timestamp: str
    findings: List[Dict]
    summary: str
    group_info: Dict = None
    output_file: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
        self.output_file = path
    
    def __repr__(self):
        return f"DiscoveryResult(success={self.success}, type='{self.type}', findings={len(self.findings)})"


# =============================================================================
# GROUP DISCOVERY ENGINE
# =============================================================================

class GroupDiscovery:
    """
    Main discovery interface using group theory.
    
    This class provides all discovery operations through the lens of
    group-theoretic concepts.
    
    Usage:
        discovery = GroupDiscovery()
        
        # Find symmetries of an expression
        result = discovery.find_symmetries("sin²(x) + cos²(x)")
        
        # Find the orbit of a formula
        result = discovery.find_orbit("exp(x)", max_depth=3)
        
        # Optimize via conjugation (find simpler equivalent)
        result = discovery.optimize_via_conjugation("arctan(tan(x))")
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.group = TruthGroup()
        self._known_identities = self._load_known_identities()
    
    def _load_known_identities(self) -> List[Dict]:
        """Load known mathematical identities as group elements."""
        return [
            # Pythagorean family
            {'lhs': 'sin²(x) + cos²(x)', 'rhs': '1', 'category': 'trigonometric'},
            {'lhs': 'cosh²(x) - sinh²(x)', 'rhs': '1', 'category': 'hyperbolic'},
            {'lhs': 'tanh²(x) + sech²(x)', 'rhs': '1', 'category': 'hyperbolic'},
            
            # Inverse compositions
            {'lhs': 'arctan(tan(x))', 'rhs': 'x', 'category': 'inverse'},
            {'lhs': 'arcsin(sin(x))', 'rhs': 'x', 'category': 'inverse'},
            {'lhs': 'exp(log(x))', 'rhs': 'x', 'category': 'inverse'},
            {'lhs': 'log(exp(x))', 'rhs': 'x', 'category': 'inverse'},
            
            # Golden ratio
            {'lhs': 'φ² - φ - 1', 'rhs': '0', 'category': 'golden'},
            {'lhs': '1/φ', 'rhs': 'φ - 1', 'category': 'golden'},
            
            # Exponential
            {'lhs': 'exp(x)·exp(-x)', 'rhs': '1', 'category': 'exponential'},
            {'lhs': 'exp(a)·exp(b)', 'rhs': 'exp(a+b)', 'category': 'exponential'},
            
            # Euler's identity
            {'lhs': 'e^(iπ) + 1', 'rhs': '0', 'category': 'fundamental'},
        ]
    
    def find_symmetries(self, expression: str, output: str = None) -> DiscoveryResult:
        """
        Find symmetries of an expression.
        
        A symmetry is a group transformation that leaves the expression
        invariant. These correspond to applicable identities.
        
        Args:
            expression: Mathematical expression to analyze
            output: Optional path to save JSON report
            
        Returns:
            DiscoveryResult with found symmetries
        """
        if self.verbose:
            print(f"Finding symmetries of: {expression}")
        
        g = self.group.element(expression)
        symmetries = self.group.find_symmetries(expression)
        
        # Also check against known identities
        applicable_identities = []
        for identity in self._known_identities:
            if self._pattern_matches(expression, identity['lhs']):
                applicable_identities.append({
                    'identity': f"{identity['lhs']} = {identity['rhs']}",
                    'category': identity['category'],
                    'simplifies_to': identity['rhs'],
                    'verified': self._verify_identity(expression, identity['rhs']),
                })
        
        findings = []
        
        # Add generator symmetries
        for sym in symmetries:
            findings.append({
                'type': 'generator_symmetry',
                'generator': sym.name,
                'anchor': sym.position.dominant_anchor()[0].name,
                'description': f"Expression is invariant under {sym.name} transformation",
            })
        
        # Add applicable identities
        findings.extend(applicable_identities)
        
        result = DiscoveryResult(
            success=len(findings) > 0,
            type='symmetry',
            source=expression,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            findings=findings,
            summary=f"Found {len(symmetries)} generator symmetries, {len(applicable_identities)} applicable identities",
            group_info={
                'position': g.position.to_dict(),
                'category': self.group.classify(g),
                'distance_from_origin': g.position.norm(),
            }
        )
        
        if output:
            result.save(output)
        
        if self.verbose:
            print(f"\n{result.summary}")
            for f in findings[:5]:
                if 'identity' in f:
                    print(f"  • {f['identity']}")
                else:
                    print(f"  • {f['description']}")
        
        return result
    
    def find_orbit(self, expression: str, max_depth: int = 3, 
                   output: str = None) -> DiscoveryResult:
        """
        Find the orbit of an expression under group action.
        
        The orbit is the set of all expressions reachable by applying
        group transformations. This discovers related truths.
        
        Args:
            expression: Starting expression
            max_depth: Maximum transformation depth
            output: Optional path to save JSON report
            
        Returns:
            DiscoveryResult with orbit elements
        """
        if self.verbose:
            print(f"Computing orbit of: {expression}")
        
        g = self.group.element(expression)
        orbit = self.group.orbit(g, max_depth=max_depth)
        
        findings = []
        for i, elem in enumerate(orbit):
            findings.append({
                'index': i,
                'position': elem.position.to_dict(),
                'distance_from_start': elem.distance_to(g),
                'category': self.group.classify(elem),
                'dominant_anchor': elem.position.dominant_anchor()[0].name,
            })
        
        # Find elements that might be simpler (closer to origin)
        simpler = [f for f in findings if f['distance_from_start'] > 0 
                   and f['position']['identity'] > g.position.weights[0]]
        
        result = DiscoveryResult(
            success=len(orbit) > 1,
            type='orbit',
            source=expression,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            findings=findings,
            summary=f"Orbit contains {len(orbit)} elements, {len(simpler)} potentially simpler",
            group_info={
                'start_position': g.position.to_dict(),
                'orbit_radius': max(f['distance_from_start'] for f in findings),
            }
        )
        
        if output:
            result.save(output)
        
        if self.verbose:
            print(f"\n{result.summary}")
        
        return result
    
    def optimize_via_conjugation(self, expression: str, 
                                  output: str = None) -> DiscoveryResult:
        """
        Optimize an expression by finding conjugate elements.
        
        Conjugation g → h·g·h⁻¹ preserves structure but may yield
        simpler representations. This is the group-theoretic analog
        of algebraic simplification.
        
        Args:
            expression: Expression to optimize
            output: Optional path to save JSON report
            
        Returns:
            DiscoveryResult with optimization recommendations
        """
        if self.verbose:
            print(f"Optimizing via conjugation: {expression}")
        
        g = self.group.element(expression)
        findings = []
        
        # Try conjugating by each generator
        for anchor, gen in self.group.generators.items():
            conjugated = g.conjugate(gen)
            
            # Check if conjugation simplifies (moves toward identity)
            if conjugated.position.weights[0] > g.position.weights[0]:
                findings.append({
                    'type': 'conjugation',
                    'conjugate_by': gen.name,
                    'original_identity_weight': float(g.position.weights[0]),
                    'new_identity_weight': float(conjugated.position.weights[0]),
                    'improvement': float(conjugated.position.weights[0] - g.position.weights[0]),
                    'description': f"Conjugating by {gen.name} moves toward identity",
                })
        
        # Check for direct pattern matches
        for identity in self._known_identities:
            if self._pattern_matches(expression, identity['lhs']):
                findings.append({
                    'type': 'direct_simplification',
                    'original': expression,
                    'simplified': identity['rhs'],
                    'category': identity['category'],
                    'verified': self._verify_identity(expression, identity['rhs']),
                })
        
        # Sort by improvement
        findings.sort(key=lambda f: f.get('improvement', 0), reverse=True)
        
        result = DiscoveryResult(
            success=len(findings) > 0,
            type='optimization',
            source=expression,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            findings=findings,
            summary=f"Found {len(findings)} optimization opportunities",
            group_info={
                'original_position': g.position.to_dict(),
                'category': self.group.classify(g),
            }
        )
        
        if output:
            result.save(output)
        
        if self.verbose:
            print(f"\n{result.summary}")
            for f in findings[:3]:
                if f['type'] == 'direct_simplification':
                    print(f"  • {f['original']} → {f['simplified']}")
                else:
                    print(f"  • Conjugate by {f['conjugate_by']}: +{f['improvement']:.3f} toward identity")
        
        return result
    
    def find_formula(self, target: str = "pi", base: int = 4096,
                     time_limit: float = 30.0, output: str = None) -> DiscoveryResult:
        """
        Discover a formula using group-theoretic search.
        
        This reimplements ribbon_solver3's formula discovery using
        orbit exploration in truth space.
        
        Args:
            target: Target constant ('pi', 'e', 'phi')
            base: Base for BBP-type formulas
            time_limit: Maximum search time
            output: Optional path to save JSON report
            
        Returns:
            DiscoveryResult with discovered formula
        """
        if self.verbose:
            print(f"Searching for {target} formula via group orbit...")
        
        t0 = time.time()
        
        # Start from a seed element in the appropriate subgroup
        if target == "pi":
            # π is in the trigonometric subgroup
            seed = self.group.generator(Anchor.PATTERN)
            target_value = np.pi
        elif target == "e":
            seed = self.group.generator(Anchor.GROWTH)
            target_value = np.e
        elif target == "phi":
            seed = self.group.generator(Anchor.GROWTH)
            target_value = (1 + np.sqrt(5)) / 2
        else:
            seed = self.group.identity()
            target_value = 1.0
        
        # Explore orbit looking for formulas
        best_error = float('inf')
        best_formula = None
        iterations = 0
        
        # Use golden ratio for orbit exploration (connects to φ-corrections)
        phi = ANCHOR_VALUES[Anchor.GROWTH]
        
        while time.time() - t0 < time_limit:
            iterations += 1
            
            # Generate candidate by combining generators with φ-weighted coefficients
            coeffs = [256, -32, 4, 1, -128, -64, -128, 4]  # BBP seed
            
            # Apply group-theoretic corrections
            for i in range(len(coeffs)):
                # Use orbit position to guide corrections
                k = (iterations + i) % 12 + 1
                correction = (1 / (i + 1)) * (phi ** (-k))
                coeffs[i] += correction * ((-1) ** i)
            
            # Evaluate BBP-type formula
            value = self._eval_bbp(base, coeffs)
            error = abs(value - target_value)
            
            if error < best_error:
                best_error = error
                best_formula = {
                    'coefficients': coeffs.copy(),
                    'error': error,
                    'n_smooth': -np.log10(error) if error > 0 else 15.0,
                    'iteration': iterations,
                }
            
            if error < 1e-14:
                break
        
        n_smooth = best_formula['n_smooth'] if best_formula else 0
        
        findings = [{
            'name': f'{target}-formula-group',
            'coefficients': best_formula['coefficients'] if best_formula else [],
            'error': float(best_error),
            'n_smooth': n_smooth,
            'iterations': iterations,
            'time': time.time() - t0,
        }]
        
        result = DiscoveryResult(
            success=n_smooth > 10,
            type='formula',
            source=f'{target}_base_{base}',
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            findings=findings,
            summary=f"{'✓ Found' if n_smooth > 10 else '✗ No'} formula: N_smooth={n_smooth:.2f}",
            group_info={
                'subgroup': 'trigonometric' if target == 'pi' else 'exponential',
                'orbit_depth': iterations,
            }
        )
        
        if output:
            result.save(output)
        
        if self.verbose:
            print(f"\n{result.summary}")
        
        return result
    
    def analyze(self, expression: str) -> Dict:
        """
        Quick group-theoretic analysis of an expression.
        
        Args:
            expression: Mathematical expression to analyze
            
        Returns:
            Dict with analysis results
        """
        g = self.group.element(expression)
        symmetries = self.group.find_symmetries(expression)
        
        # Check for applicable identities
        optimizations = []
        for identity in self._known_identities:
            if self._pattern_matches(expression, identity['lhs']):
                optimizations.append({
                    'simplified': identity['rhs'],
                    'category': identity['category'],
                    'verified': self._verify_identity(expression, identity['rhs']),
                })
        
        return {
            'expression': expression,
            'position': g.position.to_dict(),
            'category': self.group.classify(g),
            'dominant_anchor': g.position.dominant_anchor()[0].name,
            'distance_from_origin': g.position.norm(),
            'symmetries': [s.name for s in symmetries],
            'optimizations': optimizations,
        }
    
    def _pattern_matches(self, expr: str, pattern: str) -> bool:
        """Check if expression matches a pattern (simplified)."""
        # Normalize both
        expr_norm = expr.lower().replace(' ', '').replace('**2', '²')
        pattern_norm = pattern.lower().replace(' ', '').replace('**2', '²')
        
        # Direct match
        if expr_norm == pattern_norm:
            return True
        
        # Check for key substrings
        key_parts = pattern_norm.replace('(x)', '').replace('(', '').replace(')', '').split('+')
        return all(part.strip() in expr_norm for part in key_parts if part.strip())
    
    def _verify_identity(self, lhs: str, rhs: str) -> bool:
        """Verify identity numerically."""
        try:
            x = 0.5
            context = {
                'x': x, 'np': np, 'pi': np.pi, 'e': np.e,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'arctan': np.arctan, 'arcsin': np.arcsin, 'arccos': np.arccos,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'φ': (1 + np.sqrt(5)) / 2, 'phi': (1 + np.sqrt(5)) / 2,
            }
            
            # Normalize expressions for eval
            lhs_eval = self._normalize_for_eval(lhs)
            rhs_eval = self._normalize_for_eval(rhs)
            
            lhs_val = eval(lhs_eval, {"__builtins__": {}}, context)
            rhs_val = eval(rhs_eval, {"__builtins__": {}}, context)
            
            return abs(lhs_val - rhs_val) < 1e-10
        except Exception:
            return False
    
    def _normalize_for_eval(self, expr: str) -> str:
        """Normalize expression for safe eval."""
        result = expr
        # Handle multiplication symbols first
        result = result.replace('·', '*')
        result = result.replace('×', '*')
        
        # Handle function notation like sin²(x) -> (sin(x))**2
        # Must do this BEFORE replacing bare ²
        for func in ['sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan']:  # longer names first
            # Pattern: func²(arg) -> (func(arg))**2
            pattern = func + '²('
            while pattern in result:
                start = result.find(pattern)
                # Find matching closing paren
                depth = 1
                i = start + len(pattern)
                while i < len(result) and depth > 0:
                    if result[i] == '(':
                        depth += 1
                    elif result[i] == ')':
                        depth -= 1
                    i += 1
                # Extract the argument
                arg = result[start + len(pattern):i-1]
                # Replace with (func(arg))**2
                old = result[start:i]
                new = f'({func}({arg}))**2'
                result = result.replace(old, new, 1)
        
        # Handle remaining superscript 2 (e.g., x²)
        result = result.replace('²', '**2')
        
        return result
    
    def _eval_bbp(self, base: int, coeffs: List[float], n_terms: int = 50) -> float:
        """Evaluate BBP-type formula."""
        scale = 64
        slots = [(4, 1), (4, 3), (12, 1), (12, 3), (12, 5), (12, 7), (12, 9), (12, 11)]
        
        total = 0.0
        for k in range(n_terms):
            sign = (-1) ** k
            base_power = base ** k
            
            term = 0.0
            for (period, offset), coef in zip(slots, coeffs):
                denom = period * k + offset
                if denom != 0:
                    term += coef / denom
            
            total += sign * term / base_power
        
        return total / scale


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_discovery = None

def _get_discovery(verbose: bool = False) -> GroupDiscovery:
    global _discovery
    if _discovery is None:
        _discovery = GroupDiscovery(verbose=verbose)
    return _discovery


def discover_symmetries(expression: str, output: str = None, 
                        verbose: bool = True) -> DiscoveryResult:
    """Find symmetries of an expression."""
    return GroupDiscovery(verbose=verbose).find_symmetries(expression, output)


def find_orbit(expression: str, max_depth: int = 3, 
               output: str = None, verbose: bool = True) -> DiscoveryResult:
    """Find the orbit of an expression under group action."""
    return GroupDiscovery(verbose=verbose).find_orbit(expression, max_depth, output)


def optimize_via_conjugation(expression: str, output: str = None,
                              verbose: bool = True) -> DiscoveryResult:
    """Optimize an expression via group conjugation."""
    return GroupDiscovery(verbose=verbose).optimize_via_conjugation(expression, output)


def find_formula(target: str = "pi", base: int = 4096,
                 time_limit: float = 30.0, output: str = None,
                 verbose: bool = True) -> DiscoveryResult:
    """Discover a formula using group-theoretic search."""
    return GroupDiscovery(verbose=verbose).find_formula(target, base, time_limit, output)


def analyze(expression: str) -> Dict:
    """Quick group-theoretic analysis of an expression."""
    return _get_discovery().analyze(expression)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="ribbon_solver_group_theory",
        description="Ribbon Solver - Group Theory Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find symmetries of an expression
  python -m ribbon_solver_group_theory --symmetries "sin²(x) + cos²(x)"
  
  # Find orbit of an expression
  python -m ribbon_solver_group_theory --orbit "exp(x)" --depth 3
  
  # Optimize via conjugation
  python -m ribbon_solver_group_theory --optimize "arctan(tan(x))"
  
  # Find a formula
  python -m ribbon_solver_group_theory --formula pi --base 4096
        """
    )
    
    parser.add_argument('--symmetries', '-s', type=str, help='Find symmetries of expression')
    parser.add_argument('--orbit', '-r', type=str, help='Find orbit of expression')
    parser.add_argument('--optimize', '-p', type=str, help='Optimize expression via conjugation')
    parser.add_argument('--formula', '-f', type=str, choices=['pi', 'e', 'phi'], help='Find formula for constant')
    parser.add_argument('--analyze', '-a', type=str, help='Quick analysis of expression')
    parser.add_argument('--depth', '-d', type=int, default=3, help='Orbit depth')
    parser.add_argument('--base', '-b', type=int, default=4096, help='Base for formulas')
    parser.add_argument('--time-limit', '-t', type=float, default=30.0, help='Time limit in seconds')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    discovery = GroupDiscovery(verbose=verbose)
    
    if args.symmetries:
        result = discovery.find_symmetries(args.symmetries, args.output)
    elif args.orbit:
        result = discovery.find_orbit(args.orbit, args.depth, args.output)
    elif args.optimize:
        result = discovery.optimize_via_conjugation(args.optimize, args.output)
    elif args.formula:
        result = discovery.find_formula(args.formula, args.base, args.time_limit, args.output)
    elif args.analyze:
        analysis = discovery.analyze(args.analyze)
        print(json.dumps(analysis, indent=2))
        return
    else:
        parser.print_help()
        return
    
    if args.output:
        print(f"\nResults saved to {args.output}")
    
    import sys
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
