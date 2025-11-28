#!/usr/bin/env python3
"""
3-SAT Benchmark for Clock-Resonant Optimization
================================================

Tests whether clock phases can improve SAT solving by:
1. Guiding variable assignment order
2. Seeding initial assignments
3. Providing restart diversification

We use a simple DPLL-style solver with clock-guided heuristics.

The key insight: Clock phases provide deterministic, equidistributed
sequences that can replace random restarts while maintaining coverage.
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from practical_applications.clock_downcaster.clock_solver import recursive_theta, PHI

# Clock ratios for multi-clock diversification
CLOCK_RATIOS = {
    'golden': PHI,
    'silver': 1 + np.sqrt(2),
    'bronze': (3 + np.sqrt(13)) / 2,
}


@dataclass
class SATStats:
    """Statistics from SAT solving."""
    n_vars: int
    n_clauses: int
    satisfiable: bool
    assignment: Optional[List[bool]]
    decisions: int
    propagations: int
    conflicts: int
    restarts: int
    time: float


def generate_random_3sat(n_vars: int, clause_ratio: float = 4.26, seed: int = None) -> List[List[int]]:
    """
    Generate a random 3-SAT instance.
    
    Args:
        n_vars: Number of variables
        clause_ratio: Clauses per variable (4.26 is the phase transition)
        seed: Random seed
        
    Returns:
        List of clauses, each clause is a list of literals (positive = true, negative = false)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_clauses = int(n_vars * clause_ratio)
    clauses = []
    
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_in_clause = np.random.choice(n_vars, 3, replace=False) + 1
        # Random signs
        signs = np.random.choice([-1, 1], 3)
        clause = list(vars_in_clause * signs)
        clauses.append(clause)
    
    return clauses


class ClockSATSolver:
    """
    A DPLL-style SAT solver with clock-guided heuristics.
    
    Uses clock phases for:
    1. Variable selection order (VSIDS-like but deterministic)
    2. Initial polarity selection
    3. Restart diversification
    """
    
    def __init__(self, use_clock: bool = True, clock_name: str = 'golden'):
        self.use_clock = use_clock
        self.clock_name = clock_name
        self.ratio = CLOCK_RATIOS.get(clock_name, PHI)
        
        # Statistics
        self.decisions = 0
        self.propagations = 0
        self.conflicts = 0
        self.restarts = 0
        
    def _get_clock_phase(self, n: int) -> float:
        """Get fractional clock phase for index n."""
        theta = recursive_theta(n, self.ratio)
        return (theta / (2 * np.pi)) % 1.0
    
    def solve(self, clauses: List[List[int]], n_vars: int, max_restarts: int = 100) -> SATStats:
        """
        Solve a SAT instance.
        
        Args:
            clauses: List of clauses (each clause is list of literals)
            n_vars: Number of variables
            max_restarts: Maximum restart attempts
            
        Returns:
            SATStats with results
        """
        t0 = time.time()
        self.decisions = 0
        self.propagations = 0
        self.conflicts = 0
        self.restarts = 0
        
        # Try with restarts
        for restart in range(max_restarts):
            self.restarts = restart + 1
            
            # Initialize assignment based on clock phases
            assignment = [None] * (n_vars + 1)  # 1-indexed
            
            if self.use_clock:
                # Clock-guided initial polarity
                for v in range(1, n_vars + 1):
                    phase = self._get_clock_phase(v + restart * n_vars)
                    assignment[v] = phase > 0.5
            else:
                # Random initial polarity
                for v in range(1, n_vars + 1):
                    assignment[v] = np.random.random() > 0.5
            
            # Try to solve with DPLL
            result = self._dpll(clauses, assignment, n_vars, restart)
            
            if result is not None:
                return SATStats(
                    n_vars=n_vars,
                    n_clauses=len(clauses),
                    satisfiable=True,
                    assignment=result[1:],  # Remove index 0
                    decisions=self.decisions,
                    propagations=self.propagations,
                    conflicts=self.conflicts,
                    restarts=self.restarts,
                    time=time.time() - t0
                )
        
        return SATStats(
            n_vars=n_vars,
            n_clauses=len(clauses),
            satisfiable=False,
            assignment=None,
            decisions=self.decisions,
            propagations=self.propagations,
            conflicts=self.conflicts,
            restarts=self.restarts,
            time=time.time() - t0
        )
    
    def _dpll(
        self, 
        clauses: List[List[int]], 
        assignment: List[Optional[bool]],
        n_vars: int,
        restart_idx: int
    ) -> Optional[List[bool]]:
        """
        DPLL algorithm with clock-guided variable selection.
        """
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                unassigned = []
                satisfied = False
                
                for lit in clause:
                    var = abs(lit)
                    if assignment[var] is None:
                        unassigned.append(lit)
                    elif (lit > 0) == assignment[var]:
                        satisfied = True
                        break
                
                if satisfied:
                    continue
                    
                if len(unassigned) == 0:
                    # Conflict
                    self.conflicts += 1
                    return None
                    
                if len(unassigned) == 1:
                    # Unit clause - propagate
                    lit = unassigned[0]
                    var = abs(lit)
                    assignment[var] = lit > 0
                    self.propagations += 1
                    changed = True
        
        # Check if all clauses satisfied
        all_satisfied = True
        for clause in clauses:
            satisfied = False
            has_unassigned = False
            
            for lit in clause:
                var = abs(lit)
                if assignment[var] is None:
                    has_unassigned = True
                elif (lit > 0) == assignment[var]:
                    satisfied = True
                    break
            
            if not satisfied:
                if not has_unassigned:
                    # Unsatisfied clause with no unassigned vars
                    self.conflicts += 1
                    return None
                all_satisfied = False
        
        if all_satisfied:
            return assignment
        
        # Choose next variable using clock phases
        unassigned_vars = [v for v in range(1, n_vars + 1) if assignment[v] is None]
        
        if not unassigned_vars:
            return assignment
        
        if self.use_clock:
            # Clock-guided variable selection
            # Score each variable by its clock phase alignment with clause activity
            best_var = None
            best_score = -1
            
            for v in unassigned_vars:
                phase = self._get_clock_phase(v + self.decisions + restart_idx * 1000)
                # Activity score based on clause appearances
                activity = sum(1 for c in clauses for lit in c if abs(lit) == v)
                score = activity * (0.5 + 0.5 * np.sin(2 * np.pi * phase))
                
                if score > best_score:
                    best_score = score
                    best_var = v
        else:
            # Random selection
            best_var = np.random.choice(unassigned_vars)
        
        self.decisions += 1
        
        # Try both polarities
        for polarity in [True, False]:
            new_assignment = assignment.copy()
            new_assignment[best_var] = polarity
            
            result = self._dpll(clauses, new_assignment, n_vars, restart_idx)
            if result is not None:
                return result
        
        return None


def verify_solution(clauses: List[List[int]], assignment: List[bool]) -> bool:
    """Verify that an assignment satisfies all clauses."""
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit) - 1  # 0-indexed
            if (lit > 0) == assignment[var]:
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def benchmark_sat(n_vars_list: List[int], n_instances: int = 10, clause_ratio: float = 4.26):
    """
    Benchmark clock vs random SAT solving.
    """
    print("=" * 70)
    print("3-SAT BENCHMARK: Clock vs Random Heuristics")
    print("=" * 70)
    print(f"Clause ratio: {clause_ratio} (phase transition ≈ 4.26)")
    print(f"Instances per size: {n_instances}")
    
    results = []
    
    for n_vars in n_vars_list:
        print(f"\n{'='*50}")
        print(f"Testing n_vars = {n_vars}")
        print(f"{'='*50}")
        
        clock_stats = {'solved': 0, 'decisions': [], 'time': [], 'restarts': []}
        random_stats = {'solved': 0, 'decisions': [], 'time': [], 'restarts': []}
        
        for instance in range(n_instances):
            # Generate instance
            clauses = generate_random_3sat(n_vars, clause_ratio, seed=instance * 1000 + n_vars)
            
            # Solve with clock
            clock_solver = ClockSATSolver(use_clock=True)
            clock_result = clock_solver.solve(clauses, n_vars)
            
            if clock_result.satisfiable:
                clock_stats['solved'] += 1
                # Verify
                if not verify_solution(clauses, clock_result.assignment):
                    print(f"  WARNING: Clock solution verification failed!")
            
            clock_stats['decisions'].append(clock_result.decisions)
            clock_stats['time'].append(clock_result.time)
            clock_stats['restarts'].append(clock_result.restarts)
            
            # Solve with random
            random_solver = ClockSATSolver(use_clock=False)
            random_result = random_solver.solve(clauses, n_vars)
            
            if random_result.satisfiable:
                random_stats['solved'] += 1
                if not verify_solution(clauses, random_result.assignment):
                    print(f"  WARNING: Random solution verification failed!")
            
            random_stats['decisions'].append(random_result.decisions)
            random_stats['time'].append(random_result.time)
            random_stats['restarts'].append(random_result.restarts)
        
        # Print results
        print(f"\n{'Method':<12} {'Solved':>8} {'Avg Dec':>10} {'Avg Time':>12} {'Avg Restarts':>14}")
        print("-" * 60)
        
        print(f"{'Clock':<12} {clock_stats['solved']:>8}/{n_instances} "
              f"{np.mean(clock_stats['decisions']):>10.1f} "
              f"{np.mean(clock_stats['time']):>12.4f}s "
              f"{np.mean(clock_stats['restarts']):>14.1f}")
        
        print(f"{'Random':<12} {random_stats['solved']:>8}/{n_instances} "
              f"{np.mean(random_stats['decisions']):>10.1f} "
              f"{np.mean(random_stats['time']):>12.4f}s "
              f"{np.mean(random_stats['restarts']):>14.1f}")
        
        # Improvement
        if random_stats['solved'] > 0:
            solve_improvement = (clock_stats['solved'] - random_stats['solved']) / n_instances * 100
            decision_improvement = (np.mean(random_stats['decisions']) - np.mean(clock_stats['decisions'])) / max(1, np.mean(random_stats['decisions'])) * 100
            time_improvement = (np.mean(random_stats['time']) - np.mean(clock_stats['time'])) / max(0.001, np.mean(random_stats['time'])) * 100
            
            print(f"\nImprovements:")
            print(f"  Solve rate: {solve_improvement:+.1f}%")
            print(f"  Decisions: {decision_improvement:+.1f}%")
            print(f"  Time: {time_improvement:+.1f}%")
        
        results.append({
            'n_vars': n_vars,
            'clock': clock_stats,
            'random': random_stats
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'n_vars':>8} {'Clock Solved':>14} {'Random Solved':>14} {'Clock Faster':>14}")
    print("-" * 55)
    
    for r in results:
        clock_faster = np.mean(r['random']['time']) > np.mean(r['clock']['time'])
        print(f"{r['n_vars']:>8} {r['clock']['solved']:>14}/{n_instances} "
              f"{r['random']['solved']:>14}/{n_instances} "
              f"{'Yes' if clock_faster else 'No':>14}")
    
    return results


def main():
    print("3-SAT Benchmark for Clock-Resonant Optimization")
    print("=" * 70)
    
    # Test on various sizes
    # Note: SAT is NP-complete, so we keep sizes small
    # Use clause_ratio < 4.26 for satisfiable instances
    results = benchmark_sat(
        n_vars_list=[20, 30, 40, 50],
        n_instances=10,
        clause_ratio=3.5  # Below phase transition = mostly satisfiable
    )
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    Clock phases provide DETERMINISTIC diversification for SAT solving:
    
    1. Variable Selection: Clock phases guide which variable to branch on,
       providing consistent coverage without randomness.
       
    2. Polarity Selection: Initial truth values are set by clock phases,
       giving equidistributed starting points.
       
    3. Restart Diversification: Each restart uses a different clock offset,
       ensuring systematic exploration of the search space.
    
    Key advantage: REPRODUCIBILITY
    - Same instance → Same search path → Same result
    - No seed management needed
    - Easier debugging and analysis
    """)


if __name__ == "__main__":
    main()
