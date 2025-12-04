"""
Truth Space Navigator Agent
============================

Navigates the 6D truth space using group operations to explore
mathematical relationships and find related truths.

The navigator:
1. Maps expressions to group elements (positions in truth space)
2. Applies group operations to explore nearby truths
3. Evaluates coordinates for interesting properties
4. Reports findings with Ribbon Speech descriptions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_agent import BaseAgent, AgentResult
from .ribbon_speech import RibbonSpeechTranslator


@dataclass
class TruthLocation:
    """A location in truth space."""
    expression: str
    position: Dict[str, float]  # Anchor weights
    dominant_anchor: str
    distance_from_origin: float
    ribbon_speech: str
    category: str


@dataclass
class NavigationResult:
    """Result of truth space navigation."""
    start: TruthLocation
    visited: List[TruthLocation]
    interesting_finds: List[Dict[str, Any]]
    path_length: float
    summary: str


class TruthSpaceNavigator(BaseAgent):
    """
    Navigates truth space using group operations.
    
    Uses the TruthGroup from core to:
    - Map expressions to positions
    - Apply transformations (composition, conjugation)
    - Find nearby truths
    - Identify interesting relationships
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Import core components
        from core import TruthGroup, GroupElement
        self.group = TruthGroup()
        self.ribbon_translator = RibbonSpeechTranslator(
            verbose=False, 
            use_llm=kwargs.get('use_llm', True)
        )
        
        # Known interesting locations
        self.landmarks = {
            'identity': {'expression': 'x', 'description': 'The identity element'},
            'unity': {'expression': '1', 'description': 'Multiplicative unity'},
            'zero': {'expression': '0', 'description': 'Additive identity'},
            'golden': {'expression': 'phi', 'description': 'Golden ratio'},
            'euler': {'expression': 'e', 'description': "Euler's number"},
            'pi': {'expression': 'pi', 'description': 'Circle constant'},
        }
    
    def _default_system_prompt(self) -> str:
        return """You are a truth space navigator. You explore mathematical 
relationships by moving through a 6-dimensional space where each dimension
represents a mathematical concept:

1. Identity - things that equal themselves
2. Stability - fixed points, constants
3. Inverse - reciprocal relationships
4. Unity - things that combine to 1
5. Pattern - recursive, cyclic structures
6. Growth - exponential, golden relationships

Describe your navigation findings clearly and mathematically."""
    
    def process(self, expression: str) -> AgentResult:
        """
        Navigate truth space starting from an expression.
        
        Args:
            expression: Starting mathematical expression
            
        Returns:
            AgentResult with navigation findings
        """
        self._start_timer()
        self._log(f"Navigating from: {expression}")
        
        # Map starting expression to truth space
        start_location = self._map_to_location(expression)
        
        # Explore neighborhood
        visited = [start_location]
        interesting = []
        
        # Apply group operations to explore
        explorations = self._explore_neighborhood(expression)
        
        for exp_result in explorations:
            location = self._map_to_location(exp_result['expression'])
            visited.append(location)
            
            # Simplifications are always interesting
            is_simplification = exp_result.get('is_simplification', False)
            
            # Check if interesting
            if is_simplification or self._is_interesting(location, start_location):
                interesting.append({
                    'expression': exp_result['expression'],
                    'operation': exp_result['operation'],
                    'location': location,
                    'reason': exp_result.get('reason', 'Related by group operation'),
                })
        
        # Compute path length
        path_length = sum(
            self._distance(visited[i], visited[i+1]) 
            for i in range(len(visited)-1)
        ) if len(visited) > 1 else 0.0
        
        # Generate summary
        summary = self._generate_summary(start_location, visited, interesting)
        
        nav_result = NavigationResult(
            start=start_location,
            visited=visited,
            interesting_finds=interesting,
            path_length=path_length,
            summary=summary
        )
        
        self._log(f"Found {len(interesting)} interesting locations")
        
        return self._make_result(
            success=True,
            data={
                'start': {
                    'expression': start_location.expression,
                    'position': start_location.position,
                    'dominant_anchor': start_location.dominant_anchor,
                    'ribbon_speech': start_location.ribbon_speech,
                    'category': start_location.category,
                },
                'visited_count': len(visited),
                'interesting_finds': [
                    {
                        'expression': f['expression'],
                        'operation': f['operation'],
                        'ribbon_speech': f['location'].ribbon_speech,
                        'reason': f['reason'],
                    }
                    for f in interesting
                ],
                'path_length': path_length,
                'summary': summary,
            }
        )
    
    def explore_orbit(self, expression: str, depth: int = 2) -> AgentResult:
        """
        Explore the orbit of an expression under group action.
        
        Args:
            expression: Starting expression
            depth: How many levels of transformations to apply
            
        Returns:
            AgentResult with orbit exploration
        """
        self._start_timer()
        self._log(f"Exploring orbit of: {expression} (depth={depth})")
        
        # Get group element
        g = self.group.element(expression)
        
        # Find orbit
        orbit = self.group.find_orbit(g, max_depth=depth)
        
        # Map orbit to locations
        orbit_locations = []
        for element in orbit[:20]:  # Limit for performance
            loc = TruthLocation(
                expression=element.name,
                position=element.position.to_dict(),
                dominant_anchor=element.position.dominant_anchor()[0].name,
                distance_from_origin=element.position.norm(),
                ribbon_speech="",  # Skip for performance
                category=self.group.classify(element)
            )
            orbit_locations.append(loc)
        
        return self._make_result(
            success=True,
            data={
                'expression': expression,
                'orbit_size': len(orbit),
                'orbit_sample': [
                    {
                        'expression': loc.expression,
                        'dominant_anchor': loc.dominant_anchor,
                        'category': loc.category,
                    }
                    for loc in orbit_locations
                ],
            }
        )
    
    def find_path(self, start: str, end: str) -> AgentResult:
        """
        Find a path between two expressions in truth space.
        
        Args:
            start: Starting expression
            end: Target expression
            
        Returns:
            AgentResult with path information
        """
        self._start_timer()
        self._log(f"Finding path: {start} → {end}")
        
        g_start = self.group.element(start)
        g_end = self.group.element(end)
        
        # Compute distance
        distance = g_start.distance_to(g_end)
        
        # The "path" is the group element that transforms start to end
        # In a Lie group, this is g_end ⊕ g_start⁻¹
        transform = g_end.compose(g_start.inverse())
        
        return self._make_result(
            success=True,
            data={
                'start': start,
                'end': end,
                'distance': distance,
                'transform': {
                    'name': transform.name,
                    'position': transform.position.to_dict(),
                },
                'interpretation': f"Distance {distance:.3f} in truth space (proof length)",
            }
        )
    
    def _map_to_location(self, expression: str) -> TruthLocation:
        """Map an expression to a truth space location."""
        g = self.group.element(expression)
        
        # Get ribbon speech
        ribbon_result = self.ribbon_translator.process(expression)
        ribbon_speech = ribbon_result.data.get('speech', '')
        
        return TruthLocation(
            expression=expression,
            position=g.position.to_dict(),
            dominant_anchor=g.position.dominant_anchor()[0].name,
            distance_from_origin=g.position.norm(),
            ribbon_speech=ribbon_speech,
            category=self.group.classify(g)
        )
    
    def _explore_neighborhood(self, expression: str) -> List[Dict]:
        """Explore the neighborhood of an expression."""
        results = []
        g = self.group.element(expression)
        
        # Apply each generator
        for anchor, gen in self.group.generators.items():
            # Composition
            composed = g.compose(gen)
            results.append({
                'expression': f"{expression} ⊕ {anchor.name}",
                'operation': f'compose with {anchor.name}',
                'element': composed,
            })
            
            # Conjugation
            conjugated = g.conjugate(gen)
            results.append({
                'expression': f"{anchor.name} ⊕ {expression} ⊕ {anchor.name}⁻¹",
                'operation': f'conjugate by {anchor.name}',
                'element': conjugated,
            })
        
        # Check for known simplifications
        simplifications = self._check_simplifications(expression)
        for simp in simplifications:
            results.append({
                'expression': simp['simplified'],
                'operation': 'simplification',
                'reason': simp['reason'],
                'is_simplification': True,  # Mark for filtering
            })
        
        return results
    
    def _check_simplifications(self, expression: str) -> List[Dict]:
        """Check if expression has known simplifications."""
        simplifications = []
        
        expr_lower = expression.lower().replace(' ', '')
        
        # Check common patterns
        # phi * phi or phi*phi or φ*φ
        if ('phi*phi' in expr_lower or 'phi**2' in expr_lower or 
            'φ*φ' in expr_lower or 'φ**2' in expr_lower or
            ('phi' in expr_lower and '*' in expression and expression.count('phi') >= 2)):
            simplifications.append({
                'simplified': 'phi + 1',
                'reason': 'φ² = φ + 1 (golden ratio identity)',
            })
            simplifications.append({
                'simplified': '≈ 2.618',
                'reason': 'φ² ≈ 2.618 (numerical value)',
            })
        
        if 'sin' in expr_lower and 'cos' in expr_lower and '+' in expression:
            if '**2' in expression or '²' in expression:
                simplifications.append({
                    'simplified': '1',
                    'reason': 'sin²(x) + cos²(x) = 1 (Pythagorean identity)',
                })
        
        if 'arctan' in expr_lower and 'tan' in expr_lower:
            simplifications.append({
                'simplified': 'x',
                'reason': 'arctan(tan(x)) = x (inverse composition)',
            })
        
        return simplifications
    
    def _is_interesting(self, location: TruthLocation, 
                        reference: TruthLocation) -> bool:
        """Check if a location is interesting relative to reference."""
        # Different dominant anchor is interesting
        if location.dominant_anchor != reference.dominant_anchor:
            return True
        
        # Different category is interesting
        if location.category != reference.category:
            return True
        
        # Very close to origin (simplification) is interesting
        if location.distance_from_origin < 0.3:
            return True
        
        # Much closer than reference is interesting
        if location.distance_from_origin < reference.distance_from_origin * 0.5:
            return True
        
        # Simplifications are always interesting
        if 'simplification' in location.expression.lower() or '≈' in location.expression:
            return True
        
        return False
    
    def _distance(self, loc1: TruthLocation, loc2: TruthLocation) -> float:
        """Compute distance between two locations."""
        pos1 = np.array(list(loc1.position.values()))
        pos2 = np.array(list(loc2.position.values()))
        return np.linalg.norm(pos1 - pos2)
    
    def _generate_summary(self, start: TruthLocation, 
                          visited: List[TruthLocation],
                          interesting: List[Dict]) -> str:
        """Generate a summary of the navigation."""
        lines = [
            f"Starting from '{start.expression}' ({start.category}, dominant: {start.dominant_anchor})",
            f"Visited {len(visited)} locations in truth space",
        ]
        
        if interesting:
            lines.append(f"Found {len(interesting)} interesting relationships:")
            for find in interesting[:3]:  # Top 3
                lines.append(f"  • {find['expression']}: {find['reason']}")
        else:
            lines.append("No particularly interesting relationships found nearby")
        
        return '\n'.join(lines)
