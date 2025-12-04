"""
Identity Miner Processor
========================

Discovers new mathematical identities using group-theoretic search.

This processor:
1. Searches identity space (not coefficient space)
2. Uses group structure to guide search
3. Verifies candidates numerically
4. Reports discovered identities

Key insight: Valid formulas exist because of underlying identities.
Search for identities, not random coefficients.

Ported from ribbon_solver2/tools/identity_miner.py with group theory integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import itertools

from .base import BaseProcessor, ProcessorResult


@dataclass
class Identity:
    """A mathematical identity."""
    name: str
    lhs: str  # Left-hand side
    rhs: str  # Right-hand side (target)
    category: str  # 'machin', 'pythagorean', 'exponential', 'golden'
    parameters: Dict
    n_smooth: float  # -log10(error)
    verified: bool
    convergence_rate: float = None
    
    def __str__(self):
        return f"{self.lhs} = {self.rhs}"


@dataclass
class IdentityResult(ProcessorResult):
    """Result from identity mining."""
    target: str
    identities_found: int
    best_identity: Optional[Dict]
    all_identities: List[Dict]


class IdentityMiner(BaseProcessor):
    """
    Mine for mathematical identities using group-theoretic search.
    
    Usage:
        miner = IdentityMiner()
        result = miner.mine_machin_like(max_terms=3)
        result = miner.mine_pythagorean()
    """
    
    # Known identity templates
    TEMPLATES = {
        'machin': {
            'form': 'π/4 = Σ a_i × arctan(1/n_i)',
            'target': np.pi / 4,
            'examples': [
                {'name': 'Machin', 'terms': [(4, 5), (-1, 239)]},
                {'name': 'Euler', 'terms': [(1, 2), (1, 3)]},
                {'name': 'Hutton', 'terms': [(2, 3), (1, 7)]},
            ]
        },
        'pythagorean': {
            'form': 'sin²(x) + cos²(x) = 1',
            'target': 1.0,
            'examples': [
                {'name': 'Pythagorean', 'lhs': 'sin²(x) + cos²(x)', 'rhs': '1'},
                {'name': 'Hyperbolic', 'lhs': 'cosh²(x) - sinh²(x)', 'rhs': '1'},
            ]
        },
        'golden': {
            'form': 'φ² - φ - 1 = 0',
            'target': 0.0,
            'examples': [
                {'name': 'Golden defining', 'lhs': 'φ² - φ - 1', 'rhs': '0'},
                {'name': 'Golden reciprocal', 'lhs': '1/φ', 'rhs': 'φ - 1'},
            ]
        }
    }
    
    def __init__(self, group=None, verbose=True):
        super().__init__(group, verbose)
        self.discoveries = []
    
    def mine_machin_like(self, 
                         max_terms: int = 3,
                         max_coef: int = 10,
                         max_base: int = 300,
                         time_limit: float = 10.0,
                         output: str = None) -> IdentityResult:
        """
        Mine for Machin-like arctan identities: π/4 = Σ a_i × arctan(1/n_i)
        
        Args:
            max_terms: Maximum number of arctan terms
            max_coef: Maximum coefficient magnitude
            max_base: Maximum arctan argument
            time_limit: Maximum search time in seconds
            output: Optional path to save results
            
        Returns:
            IdentityResult with discovered identities
        """
        self._start_timer()
        self._log(f"Mining Machin-like identities (max_terms={max_terms})...")
        
        target = np.pi / 4
        identities = []
        
        # First verify known identities
        for known in self.TEMPLATES['machin']['examples']:
            value = sum(a * np.arctan(1/n) for a, n in known['terms'])
            error = abs(value - target)
            n_smooth = -np.log10(error) if error > 0 else 15.0
            
            identity = Identity(
                name=known['name'],
                lhs=' + '.join(f"{a}×arctan(1/{n})" for a, n in known['terms']),
                rhs='π/4',
                category='machin',
                parameters={'terms': known['terms']},
                n_smooth=n_smooth,
                verified=n_smooth > 10,
                convergence_rate=self._estimate_convergence(known['terms']),
            )
            identities.append(identity)
        
        # Search for new identities
        start_time = time.time()
        
        # 2-term search
        if max_terms >= 2:
            for a1 in range(-max_coef, max_coef + 1):
                if a1 == 0:
                    continue
                for n1 in range(2, min(50, max_base)):
                    if time.time() - start_time > time_limit:
                        break
                    
                    # What do we need from second term?
                    needed = target - a1 * np.arctan(1/n1)
                    
                    for a2 in range(-max_coef, max_coef + 1):
                        if a2 == 0:
                            continue
                        
                        # What arctan argument would give this?
                        if abs(needed / a2) < np.pi/2:
                            try:
                                n2_approx = 1 / np.tan(needed / a2)
                                n2 = round(n2_approx)
                                
                                if 2 <= n2 <= max_base and n2 != n1:
                                    value = a1 * np.arctan(1/n1) + a2 * np.arctan(1/n2)
                                    error = abs(value - target)
                                    n_smooth = -np.log10(error) if error > 0 else 15.0
                                    
                                    if n_smooth > 10:  # Machine precision
                                        terms = [(a1, n1), (a2, n2)]
                                        name = f"Machin-{n1}-{n2}"
                                        
                                        # Check if already found
                                        if not any(i.name == name for i in identities):
                                            identity = Identity(
                                                name=name,
                                                lhs=f"{a1}×arctan(1/{n1}) + {a2}×arctan(1/{n2})",
                                                rhs='π/4',
                                                category='machin',
                                                parameters={'terms': terms},
                                                n_smooth=n_smooth,
                                                verified=True,
                                                convergence_rate=self._estimate_convergence(terms),
                                            )
                                            identities.append(identity)
                                            self._log(f"  Found: {identity}")
                            except:
                                pass
        
        # Sort by convergence rate
        identities.sort(key=lambda i: i.convergence_rate or 0, reverse=True)
        
        best = asdict(identities[0]) if identities else None
        
        result = IdentityResult(
            processor='IdentityMiner',
            success=len(identities) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=[asdict(i) for i in identities],
            summary=f"Found {len(identities)} Machin-like identities",
            target='π/4',
            identities_found=len(identities),
            best_identity=best,
            all_identities=[asdict(i) for i in identities],
        )
        
        if output:
            result.save(output)
        
        self._log(f"\n{result.summary}")
        return result
    
    def mine_pythagorean(self, output: str = None) -> IdentityResult:
        """
        Find Pythagorean-type identities.
        
        These are identities of the form f(x)² + g(x)² = 1 or similar.
        """
        self._start_timer()
        self._log("Mining Pythagorean identities...")
        
        identities = []
        
        # Known Pythagorean identities
        pythagorean_forms = [
            ('sin²(x) + cos²(x)', '1', 'Pythagorean'),
            ('cosh²(x) - sinh²(x)', '1', 'Hyperbolic Pythagorean'),
            ('sec²(x) - tan²(x)', '1', 'Secant-Tangent'),
            ('csc²(x) - cot²(x)', '1', 'Cosecant-Cotangent'),
        ]
        
        for lhs, rhs, name in pythagorean_forms:
            verified = self._verify_numerically(lhs, rhs)
            
            # Get group-theoretic info
            g = self.group.element(lhs)
            
            identity = Identity(
                name=name,
                lhs=lhs,
                rhs=rhs,
                category='pythagorean',
                parameters={
                    'group_position': g.position.to_dict(),
                    'dominant_anchor': g.position.dominant_anchor()[0].name,
                },
                n_smooth=15.0 if verified else 0.0,
                verified=verified,
            )
            identities.append(identity)
        
        result = IdentityResult(
            processor='IdentityMiner',
            success=len(identities) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=[asdict(i) for i in identities],
            summary=f"Found {len(identities)} Pythagorean identities",
            target='1',
            identities_found=len(identities),
            best_identity=asdict(identities[0]) if identities else None,
            all_identities=[asdict(i) for i in identities],
        )
        
        if output:
            result.save(output)
        
        return result
    
    def process(self, identity_type: str = 'machin', 
                output: str = None, **kwargs) -> IdentityResult:
        """
        Main processing method.
        
        Args:
            identity_type: Type of identity to mine ('machin', 'pythagorean')
            output: Optional output path
            **kwargs: Additional arguments for specific miners
        """
        if identity_type == 'machin':
            return self.mine_machin_like(output=output, **kwargs)
        elif identity_type == 'pythagorean':
            return self.mine_pythagorean(output=output)
        else:
            raise ValueError(f"Unknown identity type: {identity_type}")
    
    def _estimate_convergence(self, terms: List[Tuple[int, int]]) -> float:
        """
        Estimate convergence rate of arctan identity.
        
        Larger arguments → faster convergence.
        """
        if not terms:
            return 0.0
        
        # Convergence is limited by smallest argument
        min_arg = min(n for _, n in terms)
        return np.log10(min_arg)
