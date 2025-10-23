"""
Adaptive Non-Locality Optimizer
Solution-Problem Dimensional Coupling via Hausdorff Resonance

Based on: "Adaptive Non-Locality in Optimization: Solution-Problem 
Dimensional Coupling via Hausdorff Resonance" (October 2025)

Mathematical Framework:
- Problem affinity A_P(D): How strongly problem structure resonates at dimension D
- Solution affinity A_S(D; σ): How coherent solution appears at dimension D
- Coupling landscape C(D; σ, T): Geometric mean emphasizing alignment
- Dimensional sampling: P(D; σ, T) ∝ C(D; σ, T)

Key Innovation: Optimization self-organizes through dimensional space by
following gradients in the solution-problem coupling landscape.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


@dataclass
class AffinityMetrics:
    """Affinity measurements at a specific dimension"""
    dimension: float
    problem_affinity: float
    solution_affinity: float
    coupling_strength: float
    

@dataclass
class DimensionalTrajectory:
    """Track dimensional evolution over optimization"""
    iterations: List[int]
    dimensions: List[float]
    temperatures: List[float]
    coupling_strengths: List[float]
    solution_costs: List[float]
    phase: List[str]  # 'exploration', 'coupling', 'exploitation'


class AdaptiveNonlocalityOptimizer:
    """
    Adaptive non-local optimization through dimensional coupling.
    
    Allows optimization to navigate through Hausdorff dimensional space,
    guided by solution-problem affinity resonance.
    
    Parameters
    ----------
    d_min : float
        Minimum Hausdorff dimension (default: 1.0)
    d_max : float
        Maximum Hausdorff dimension (default: 2.5)
    n_dim_samples : int
        Number of dimensional samples (default: 30)
    t_initial : float
        Initial temperature (default: 2.0)
    t_final : float
        Final temperature (default: 0.5)
    tau : float
        Temperature decay constant (default: 50.0)
    epsilon : float
        Ergodicity parameter (default: 0.01)
    """
    
    def __init__(
        self,
        d_min: float = 1.0,
        d_max: float = 2.5,
        n_dim_samples: int = 30,
        t_initial: float = 2.0,
        t_final: float = 0.5,
        tau: float = 50.0,
        epsilon: float = 0.01
    ):
        self.d_min = d_min
        self.d_max = d_max
        self.n_dim_samples = n_dim_samples
        self.t_initial = t_initial
        self.t_final = t_final
        self.tau = tau
        self.epsilon = epsilon
        
        # Dimensional grid
        self.dimensions = np.linspace(d_min, d_max, n_dim_samples)
        
        # Problem affinity (computed once)
        self.problem_affinity_cache = None
        
    def compute_temperature(self, iteration: int, max_iterations: int) -> float:
        """Compute temperature at iteration t"""
        t_normalized = iteration / max_iterations
        return self.t_initial * np.exp(-3 * t_normalized) + self.t_final
        
    def scale_distance(self, distance: float, dimension: float) -> float:
        """Scale distance by Hausdorff dimension: d_D(x,y) = d(x,y)^D"""
        return np.power(distance, dimension)
        
    def compute_problem_affinity(
        self,
        points: np.ndarray,
        structure_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute problem affinity A_P(D) across all dimensions.
        
        Measures how strongly problem structure resonates at each dimension.
        Uses fractal dimension matching via box-counting.
        
        Parameters
        ----------
        points : np.ndarray, shape (n_points, n_features)
            Problem instance points (e.g., city coordinates)
        structure_weights : np.ndarray, optional
            Problem-specific structure weights
            
        Returns
        -------
        affinities : np.ndarray, shape (n_dim_samples,)
            Problem affinity at each dimension
        """
        if self.problem_affinity_cache is not None:
            return self.problem_affinity_cache
            
        # Compute box-counting dimension
        d_box = self._estimate_box_counting_dimension(points)
        
        # Affinity peaks near intrinsic dimension
        affinities = np.exp(-0.5 * (self.dimensions - d_box)**2)
        
        # Add graph clustering component
        if len(points) > 2:
            clustering = self._compute_clustering_affinity(points)
            affinities = 0.7 * affinities + 0.3 * clustering
            
        self.problem_affinity_cache = affinities
        return affinities
        
    def _estimate_box_counting_dimension(self, points: np.ndarray) -> float:
        """Estimate fractal dimension via box-counting"""
        if len(points) < 4:
            return 2.0  # Default for small instances
            
        # Normalize points to [0, 1]
        points_norm = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-10)
        
        # Box sizes
        epsilons = np.logspace(-2, 0, 10)
        counts = []
        
        for eps in epsilons:
            # Count occupied boxes
            boxes = np.floor(points_norm / eps).astype(int)
            unique_boxes = len(np.unique(boxes, axis=0))
            counts.append(unique_boxes)
            
        # Fit power law: N(ε) ~ ε^(-D)
        log_eps = np.log(epsilons)
        log_counts = np.log(counts)
        
        # Linear regression
        coeffs = np.polyfit(log_eps, log_counts, 1)
        d_box = -coeffs[0]
        
        # Clamp to reasonable range
        return np.clip(d_box, 1.0, 3.0)
        
    def _compute_clustering_affinity(self, points: np.ndarray) -> np.ndarray:
        """Compute graph clustering coefficient across dimensions"""
        affinities = np.zeros(self.n_dim_samples)
        
        # Compute distance matrix
        dist_matrix = squareform(pdist(points))
        
        for i, D in enumerate(self.dimensions):
            # Scale distances by dimension
            scaled_dist = np.power(dist_matrix + 1e-10, D)
            
            # Compute clustering via k-nearest neighbors
            k = min(5, len(points) - 1)
            clustering = 0.0
            
            for node in range(len(points)):
                # Find k nearest neighbors
                neighbors = np.argsort(scaled_dist[node])[1:k+1]
                
                # Count edges between neighbors
                edges = 0
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 < n2 and scaled_dist[n1, n2] < np.median(scaled_dist):
                            edges += 1
                            
                max_edges = k * (k - 1) / 2
                if max_edges > 0:
                    clustering += edges / max_edges
                    
            affinities[i] = clustering / len(points)
            
        # Normalize
        return affinities / (affinities.max() + 1e-10)
        
    def compute_solution_affinity(
        self,
        solution: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Compute solution affinity A_S(D; σ) across all dimensions.
        
        Measures how coherent the solution appears at each dimension.
        
        Parameters
        ----------
        solution : np.ndarray, shape (n_points,)
            Current solution (e.g., tour order)
        points : np.ndarray, shape (n_points, n_features)
            Problem points
            
        Returns
        -------
        affinities : np.ndarray, shape (n_dim_samples,)
            Solution affinity at each dimension
        """
        affinities = np.zeros(self.n_dim_samples)
        
        for i, D in enumerate(self.dimensions):
            # Compute path smoothness
            smoothness = self._compute_smoothness(solution, points, D)
            
            # Compute edge uniformity
            uniformity = self._compute_uniformity(solution, points, D)
            
            # Combine metrics
            affinities[i] = 0.6 * smoothness + 0.4 * uniformity
            
        return affinities
        
    def _compute_smoothness(
        self,
        solution: np.ndarray,
        points: np.ndarray,
        dimension: float
    ) -> float:
        """Compute path smoothness (low curvature = high smoothness)"""
        if len(solution) < 3:
            return 1.0
            
        # Get ordered points
        ordered_points = points[solution]
        
        # Compute vectors between consecutive points
        vectors = np.diff(ordered_points, axis=0)
        
        # Scale by dimension
        norms = np.linalg.norm(vectors, axis=1)
        scaled_norms = np.power(norms + 1e-10, dimension)
        vectors_scaled = vectors / (scaled_norms[:, np.newaxis] + 1e-10)
        
        # Compute curvature (1 - cos(angle))
        curvatures = []
        for i in range(len(vectors_scaled) - 1):
            cos_angle = np.dot(vectors_scaled[i], vectors_scaled[i+1])
            curvature = 1 - cos_angle
            curvatures.append(curvature)
            
        mean_curvature = np.mean(curvatures) if curvatures else 0.0
        
        # Smoothness = 1 / (1 + curvature)
        return 1.0 / (1.0 + mean_curvature)
        
    def _compute_uniformity(
        self,
        solution: np.ndarray,
        points: np.ndarray,
        dimension: float
    ) -> float:
        """Compute edge length uniformity"""
        if len(solution) < 2:
            return 1.0
            
        # Get ordered points
        ordered_points = points[solution]
        
        # Compute edge lengths
        edges = np.diff(ordered_points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        
        # Scale by dimension
        scaled_lengths = np.power(lengths + 1e-10, dimension)
        
        # Uniformity = 1 / (1 + CV)
        mean_len = np.mean(scaled_lengths)
        std_len = np.std(scaled_lengths)
        cv = std_len / (mean_len + 1e-10)
        
        return 1.0 / (1.0 + cv)
        
    def compute_coupling(
        self,
        problem_affinity: np.ndarray,
        solution_affinity: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """
        Compute coupling landscape C(D; σ, T).
        
        Geometric mean emphasizes alignment between problem and solution.
        """
        # Geometric mean with temperature modulation
        coupling = np.power(
            problem_affinity * solution_affinity + self.epsilon,
            1.0 / (2.0 * temperature)
        )
        
        return coupling
        
    def sample_dimension(
        self,
        coupling: np.ndarray,
        temperature: float
    ) -> Tuple[float, int]:
        """
        Sample dimension from coupling distribution P(D; σ, T).
        
        Returns
        -------
        dimension : float
            Sampled dimension
        index : int
            Index in dimension grid
        """
        # Normalize to probability distribution
        prob = coupling / (coupling.sum() + 1e-10)
        
        # Sample
        idx = np.random.choice(len(self.dimensions), p=prob)
        
        return self.dimensions[idx], idx
        
    def optimize(
        self,
        initial_solution: np.ndarray,
        points: np.ndarray,
        cost_function: Callable,
        local_search: Callable,
        max_iterations: int = 200,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, DimensionalTrajectory]:
        """
        Adaptive non-local optimization.
        
        Parameters
        ----------
        initial_solution : np.ndarray
            Initial solution (e.g., tour order)
        points : np.ndarray
            Problem points
        cost_function : callable
            Function to compute solution cost
        local_search : callable
            Local search operator: (solution, points, dimension) -> new_solution
        max_iterations : int
            Maximum iterations
        verbose : bool
            Print progress
            
        Returns
        -------
        best_solution : np.ndarray
            Best solution found
        best_cost : float
            Cost of best solution
        trajectory : DimensionalTrajectory
            Dimensional evolution history
        """
        # Initialize
        solution = initial_solution.copy()
        cost = cost_function(solution, points)
        best_solution = solution.copy()
        best_cost = cost
        
        # Compute problem affinity once
        problem_affinity = self.compute_problem_affinity(points)
        
        # Track trajectory
        trajectory = DimensionalTrajectory(
            iterations=[],
            dimensions=[],
            temperatures=[],
            coupling_strengths=[],
            solution_costs=[],
            phase=[]
        )
        
        for iteration in range(max_iterations):
            # Compute temperature
            temperature = self.compute_temperature(iteration, max_iterations)
            
            # Compute solution affinity
            solution_affinity = self.compute_solution_affinity(solution, points)
            
            # Compute coupling
            coupling = self.compute_coupling(
                problem_affinity,
                solution_affinity,
                temperature
            )
            
            # Sample dimension
            dimension, dim_idx = self.sample_dimension(coupling, temperature)
            
            # Determine phase
            if temperature > 1.5:
                phase = 'exploration'
            elif temperature > 0.8:
                phase = 'coupling'
            else:
                phase = 'exploitation'
                
            # Apply local search in sampled dimension
            new_solution = local_search(solution, points, dimension)
            new_cost = cost_function(new_solution, points)
            
            # Accept if better
            if new_cost < cost:
                solution = new_solution
                cost = new_cost
                
                if cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = cost
                    
            # Record trajectory
            trajectory.iterations.append(iteration)
            trajectory.dimensions.append(dimension)
            trajectory.temperatures.append(temperature)
            trajectory.coupling_strengths.append(coupling[dim_idx])
            trajectory.solution_costs.append(cost)
            trajectory.phase.append(phase)
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: D={dimension:.3f}, T={temperature:.3f}, "
                      f"Cost={cost:.2f}, Phase={phase}")
                      
        return best_solution, best_cost, trajectory
        
    def analyze_trajectory(self, trajectory: DimensionalTrajectory) -> Dict:
        """Analyze dimensional trajectory for insights"""
        phases = ['exploration', 'coupling', 'exploitation']
        phase_stats = {}
        
        for phase in phases:
            mask = np.array(trajectory.phase) == phase
            if mask.any():
                phase_stats[phase] = {
                    'mean_dimension': np.mean(np.array(trajectory.dimensions)[mask]),
                    'std_dimension': np.std(np.array(trajectory.dimensions)[mask]),
                    'mean_coupling': np.mean(np.array(trajectory.coupling_strengths)[mask]),
                    'iterations': int(mask.sum())
                }
                
        return {
            'phase_statistics': phase_stats,
            'final_dimension': trajectory.dimensions[-1],
            'dimensional_entropy': entropy(np.histogram(trajectory.dimensions, bins=10)[0] + 1),
            'cost_improvement': trajectory.solution_costs[0] - trajectory.solution_costs[-1],
            'convergence_iteration': self._find_convergence(trajectory.solution_costs)
        }
        
    def _find_convergence(self, costs: List[float], window: int = 20) -> int:
        """Find iteration where cost converged"""
        if len(costs) < window:
            return len(costs)
            
        for i in range(len(costs) - window):
            if np.std(costs[i:i+window]) < 0.01 * np.mean(costs[i:i+window]):
                return i
                
        return len(costs)
