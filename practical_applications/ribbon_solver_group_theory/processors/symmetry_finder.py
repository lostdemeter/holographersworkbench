"""
Symmetry Finder Processor
=========================

Finds symmetries (fixed points) of mathematical expressions.

In group theory, a symmetry is a transformation that leaves something invariant.
For mathematical expressions, symmetries correspond to identities.

This processor:
1. Takes an expression
2. Maps it to a group element
3. Finds which generators leave it invariant
4. Reports the symmetries as applicable identities
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .base import BaseProcessor, ProcessorResult


@dataclass
class Symmetry:
    """A symmetry of an expression."""
    expression: str
    generator: str  # Which generator leaves it invariant
    anchor: str  # Which anchor this relates to
    description: str
    simplifies_to: Optional[str] = None
    verified: bool = False


@dataclass
class SymmetryResult(ProcessorResult):
    """Result from symmetry finding."""
    expression: str
    symmetries_found: int
    simplifications: List[Dict]
    group_info: Dict


class SymmetryFinder(BaseProcessor):
    """
    Find symmetries of mathematical expressions.
    
    Usage:
        finder = SymmetryFinder()
        result = finder.find_symmetries("sin²(x) + cos²(x)")
    """
    
    # Known simplifications by category
    SIMPLIFICATIONS = {
        'inverse': [
            ('arctan(tan(x))', 'x'),
            ('arcsin(sin(x))', 'x'),
            ('arccos(cos(x))', 'x'),
            ('exp(log(x))', 'x'),
            ('log(exp(x))', 'x'),
            ('sqrt(x)**2', 'x'),
        ],
        'pythagorean': [
            ('sin²(x) + cos²(x)', '1'),
            ('cos²(x) + sin²(x)', '1'),
            ('cosh²(x) - sinh²(x)', '1'),
            ('1 - sin²(x)', 'cos²(x)'),
            ('1 - cos²(x)', 'sin²(x)'),
        ],
        'exponential': [
            ('exp(x)·exp(-x)', '1'),
            ('exp(a)·exp(b)', 'exp(a+b)'),
        ],
        'golden': [
            ('φ² - φ - 1', '0'),
            ('1/φ', 'φ - 1'),
            ('φ + 1/φ', '√5'),
        ],
    }
    
    def __init__(self, group=None, verbose=True):
        super().__init__(group, verbose)
    
    def find_symmetries(self, expression: str, 
                        output: str = None) -> SymmetryResult:
        """
        Find symmetries of an expression.
        
        Args:
            expression: Mathematical expression to analyze
            output: Optional path to save results
            
        Returns:
            SymmetryResult with found symmetries
        """
        self._start_timer()
        self._log(f"Finding symmetries of: {expression}")
        
        # Map to group element
        g = self.group.element(expression)
        
        # Find generator symmetries
        symmetries = []
        for anchor, gen in self.group.generators.items():
            conjugated = g.conjugate(gen)
            
            if self.group.are_equivalent(g, conjugated, tolerance=0.05):
                symmetries.append(Symmetry(
                    expression=expression,
                    generator=gen.name,
                    anchor=anchor.name,
                    description=f"Invariant under {gen.name} transformation",
                ))
        
        # Check for known simplifications
        simplifications = []
        for category, patterns in self.SIMPLIFICATIONS.items():
            for lhs, rhs in patterns:
                if self._pattern_matches(expression, lhs):
                    verified = self._verify_numerically(expression, rhs)
                    
                    simplifications.append({
                        'original': expression,
                        'simplified': rhs,
                        'category': category,
                        'verified': verified,
                    })
                    
                    symmetries.append(Symmetry(
                        expression=expression,
                        generator=category,
                        anchor=self._category_to_anchor(category),
                        description=f"{lhs} = {rhs}",
                        simplifies_to=rhs,
                        verified=verified,
                    ))
        
        result = SymmetryResult(
            processor='SymmetryFinder',
            success=len(symmetries) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=[asdict(s) for s in symmetries],
            summary=f"Found {len(symmetries)} symmetries, {len(simplifications)} simplifications",
            expression=expression,
            symmetries_found=len(symmetries),
            simplifications=simplifications,
            group_info={
                'position': g.position.to_dict(),
                'category': self.group.classify(g),
                'dominant_anchor': g.position.dominant_anchor()[0].name,
            }
        )
        
        if output:
            result.save(output)
        
        self._log(f"\n{result.summary}")
        for s in simplifications:
            status = "✓" if s['verified'] else "?"
            self._log(f"  {status} {s['original']} → {s['simplified']}")
        
        return result
    
    def process(self, expression: str, **kwargs) -> SymmetryResult:
        """Main processing method."""
        return self.find_symmetries(expression, **kwargs)
    
    def _pattern_matches(self, expr: str, pattern: str) -> bool:
        """Check if expression matches a pattern."""
        # Normalize both
        expr_norm = expr.lower().replace(' ', '').replace('**2', '²')
        pattern_norm = pattern.lower().replace(' ', '').replace('**2', '²')
        
        if expr_norm == pattern_norm:
            return True
        
        # Check for key substrings
        key_parts = pattern_norm.replace('(x)', '').replace('(', '').replace(')', '').split('+')
        return all(part.strip() in expr_norm for part in key_parts if part.strip())
    
    def _category_to_anchor(self, category: str) -> str:
        """Map category to anchor name."""
        mapping = {
            'inverse': 'INVERSE',
            'pythagorean': 'UNITY',
            'exponential': 'GROWTH',
            'golden': 'GROWTH',
            'algebraic': 'STABILITY',
        }
        return mapping.get(category, 'IDENTITY')
