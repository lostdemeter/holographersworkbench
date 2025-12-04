"""
Ribbon Constraint Learner

Learns validity constraints by observing the ribbon solver's actual behavior.
This bridges the gap between theoretical truth space and practical solver constraints.

The approach:
1. Run the ribbon solver on many queries
2. Record which truth space positions are visited
3. Analyze patterns in valid vs invalid transitions
4. Formulate constraint equations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Set
from pathlib import Path
import json


@dataclass
class SolverObservation:
    """A single observation from the ribbon solver."""
    query: str
    positions: List[np.ndarray]  # Sequence of truth space positions
    anchors_visited: List[str]
    success: bool
    confidence: float


@dataclass
class TransitionRule:
    """A learned rule about valid transitions."""
    from_region: str  # Description of source region
    to_region: str    # Description of target region
    probability: float
    examples: int
    constraint: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None


class RibbonConstraintLearner:
    """
    Learns truth space constraints from ribbon solver behavior.
    
    This is a data-driven approach: instead of assuming mathematical
    structure, we observe what the solver actually does and learn
    the implicit rules.
    """
    
    ANCHORS = ['identity', 'pattern', 'structure', 'unity', 'ground', 'inverse']
    
    def __init__(self):
        self.observations: List[SolverObservation] = []
        self.valid_positions: Set[Tuple] = set()
        self.invalid_positions: Set[Tuple] = set()
        self.transitions: Dict[Tuple[str, str], int] = {}
        self.learned_rules: List[TransitionRule] = []
        
        # Grid for discretizing continuous positions
        self.grid_resolution = 20
    
    def _discretize(self, position: np.ndarray) -> Tuple:
        """Convert continuous position to discrete grid cell."""
        discrete = tuple(int(p * self.grid_resolution) for p in position)
        return discrete
    
    def _position_to_region(self, position: np.ndarray) -> str:
        """Describe which region a position is in."""
        # Find dominant anchor
        max_idx = np.argmax(position)
        dominant = self.ANCHORS[max_idx]
        
        # Check for balance
        sorted_pos = np.sort(position)[::-1]
        if sorted_pos[0] - sorted_pos[1] < 0.1:
            return f"balanced_{dominant}"
        elif sorted_pos[0] > 0.4:
            return f"strong_{dominant}"
        else:
            return f"weak_{dominant}"
    
    def observe_solver_run(self, query: str, positions: List[np.ndarray],
                          anchors: List[str], success: bool, confidence: float):
        """Record an observation from a solver run."""
        obs = SolverObservation(
            query=query,
            positions=[np.array(p) for p in positions],
            anchors_visited=anchors,
            success=success,
            confidence=confidence
        )
        self.observations.append(obs)
        
        # Record valid positions
        for pos in positions:
            self.valid_positions.add(self._discretize(pos))
        
        # Record transitions
        for i in range(len(positions) - 1):
            from_region = self._position_to_region(positions[i])
            to_region = self._position_to_region(positions[i + 1])
            key = (from_region, to_region)
            self.transitions[key] = self.transitions.get(key, 0) + 1
    
    def observe_invalid_position(self, position: np.ndarray):
        """Record a position that was rejected by the solver."""
        self.invalid_positions.add(self._discretize(position))
    
    def learn_transition_rules(self) -> List[TransitionRule]:
        """Learn rules about valid transitions from observations."""
        rules = []
        
        # Count total transitions from each region
        from_counts = {}
        for (from_r, to_r), count in self.transitions.items():
            from_counts[from_r] = from_counts.get(from_r, 0) + count
        
        # Calculate transition probabilities
        for (from_r, to_r), count in self.transitions.items():
            prob = count / from_counts[from_r]
            
            if prob > 0.1:  # Only report significant transitions
                rule = TransitionRule(
                    from_region=from_r,
                    to_region=to_r,
                    probability=prob,
                    examples=count
                )
                rules.append(rule)
        
        # Sort by probability
        rules.sort(key=lambda r: r.probability, reverse=True)
        self.learned_rules = rules
        
        return rules
    
    def learn_boundary_constraints(self) -> List[Dict]:
        """
        Learn constraints that define the boundary of valid space.
        
        Analyzes the difference between valid and invalid positions.
        """
        if not self.valid_positions or not self.invalid_positions:
            return []
        
        # Convert to arrays for analysis
        valid_array = np.array([list(p) for p in self.valid_positions]) / self.grid_resolution
        invalid_array = np.array([list(p) for p in self.invalid_positions]) / self.grid_resolution
        
        constraints = []
        
        # Analyze each dimension
        for i, anchor in enumerate(self.ANCHORS):
            valid_vals = valid_array[:, i]
            invalid_vals = invalid_array[:, i]
            
            # Check for max/min constraints
            valid_max = np.max(valid_vals)
            valid_min = np.min(valid_vals)
            
            if valid_max < 0.9:  # There's an upper bound
                constraints.append({
                    'type': 'max',
                    'anchor': anchor,
                    'value': valid_max,
                    'description': f"{anchor} ≤ {valid_max:.3f}"
                })
            
            if valid_min > 0.1:  # There's a lower bound
                constraints.append({
                    'type': 'min',
                    'anchor': anchor,
                    'value': valid_min,
                    'description': f"{anchor} ≥ {valid_min:.3f}"
                })
        
        # Analyze pairwise relationships
        for i in range(6):
            for j in range(i + 1, 6):
                valid_sum = valid_array[:, i] + valid_array[:, j]
                
                if np.std(valid_sum) < 0.05:  # Sum is nearly constant
                    mean_sum = np.mean(valid_sum)
                    constraints.append({
                        'type': 'sum',
                        'anchors': [self.ANCHORS[i], self.ANCHORS[j]],
                        'value': mean_sum,
                        'description': f"{self.ANCHORS[i]} + {self.ANCHORS[j]} ≈ {mean_sum:.3f}"
                    })
        
        return constraints
    
    def generate_validity_function(self) -> Callable[[np.ndarray], bool]:
        """Generate a validity function from learned constraints."""
        constraints = self.learn_boundary_constraints()
        
        def learned_validity(position: np.ndarray) -> bool:
            # Basic simplex check
            if np.any(position < 0) or np.any(position > 1):
                return False
            if abs(np.sum(position) - 1) > 0.05:
                return False
            
            # Check learned constraints
            for c in constraints:
                if c['type'] == 'max':
                    idx = self.ANCHORS.index(c['anchor'])
                    if position[idx] > c['value'] + 0.05:
                        return False
                elif c['type'] == 'min':
                    idx = self.ANCHORS.index(c['anchor'])
                    if position[idx] < c['value'] - 0.05:
                        return False
                elif c['type'] == 'sum':
                    idx1 = self.ANCHORS.index(c['anchors'][0])
                    idx2 = self.ANCHORS.index(c['anchors'][1])
                    if abs(position[idx1] + position[idx2] - c['value']) > 0.1:
                        return False
            
            return True
        
        return learned_validity
    
    def save(self, filepath: str):
        """Save learned constraints to file."""
        data = {
            'n_observations': len(self.observations),
            'n_valid_positions': len(self.valid_positions),
            'n_invalid_positions': len(self.invalid_positions),
            'transitions': {f"{k[0]}->{k[1]}": v for k, v in self.transitions.items()},
            'rules': [
                {
                    'from': r.from_region,
                    'to': r.to_region,
                    'probability': r.probability,
                    'examples': r.examples
                }
                for r in self.learned_rules
            ],
            'constraints': self.learn_boundary_constraints()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved learned constraints to {filepath}")
    
    def load(self, filepath: str):
        """Load learned constraints from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct transitions
        self.transitions = {}
        for key, val in data.get('transitions', {}).items():
            parts = key.split('->')
            if len(parts) == 2:
                self.transitions[(parts[0], parts[1])] = val
        
        # Reconstruct rules
        self.learned_rules = [
            TransitionRule(
                from_region=r['from'],
                to_region=r['to'],
                probability=r['probability'],
                examples=r['examples']
            )
            for r in data.get('rules', [])
        ]
        
        print(f"Loaded {len(self.learned_rules)} rules from {filepath}")
    
    def report(self) -> str:
        """Generate a report of learned constraints."""
        lines = [
            "=" * 60,
            "RIBBON CONSTRAINT LEARNER REPORT",
            "=" * 60,
            "",
            f"Observations: {len(self.observations)}",
            f"Valid positions: {len(self.valid_positions)}",
            f"Invalid positions: {len(self.invalid_positions)}",
            f"Unique transitions: {len(self.transitions)}",
            "",
        ]
        
        if self.learned_rules:
            lines.append("LEARNED TRANSITION RULES:")
            lines.append("-" * 40)
            for rule in self.learned_rules[:10]:
                lines.append(f"  {rule.from_region} → {rule.to_region}: "
                           f"{rule.probability:.1%} ({rule.examples} examples)")
        
        constraints = self.learn_boundary_constraints()
        if constraints:
            lines.append("")
            lines.append("LEARNED BOUNDARY CONSTRAINTS:")
            lines.append("-" * 40)
            for c in constraints:
                lines.append(f"  {c['description']}")
        
        return "\n".join(lines)


def integrate_with_solver():
    """
    Example of how to integrate with the actual ribbon solver.
    
    This would be called during solver runs to collect observations.
    """
    learner = RibbonConstraintLearner()
    
    # Example: simulate some solver observations
    # In practice, this would hook into the actual solver
    
    # Simulated observation 1: query about patterns
    positions = [
        np.array([0.2, 0.3, 0.1, 0.2, 0.1, 0.1]),  # Start
        np.array([0.15, 0.35, 0.15, 0.15, 0.1, 0.1]),  # Move toward pattern
        np.array([0.1, 0.4, 0.15, 0.15, 0.1, 0.1]),  # Stronger pattern
    ]
    learner.observe_solver_run(
        query="find pattern in data",
        positions=positions,
        anchors=['identity', 'pattern', 'pattern'],
        success=True,
        confidence=0.85
    )
    
    # Simulated observation 2: query about structure
    positions = [
        np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.1]),
        np.array([0.15, 0.1, 0.35, 0.2, 0.1, 0.1]),
        np.array([0.1, 0.1, 0.4, 0.2, 0.1, 0.1]),
    ]
    learner.observe_solver_run(
        query="analyze structure",
        positions=positions,
        anchors=['identity', 'structure', 'structure'],
        success=True,
        confidence=0.9
    )
    
    # Learn rules
    rules = learner.learn_transition_rules()
    
    print(learner.report())
    
    return learner


if __name__ == "__main__":
    integrate_with_solver()
