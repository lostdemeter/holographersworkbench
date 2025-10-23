"""
Residual Chaos Seeding (RCS)

A principled framework for combinatorial optimization that exploits the duality
between solution space and residual space. By formalizing the geometric tension
in candidate solutions as a "chaos" magnitude, we provide a novel heuristic that
guides search toward regions of low geometric incoherence.

Mathematical Foundation:
- Projection P(T) smooths tour via Gaussian kernel
- Residual R(T) = C - P(T) captures geometric incoherence
- Chaos magnitude χ(T) = ||R(T)|| quantifies solution quality
- Chaos-weighted distance guides construction

Reference: Residual_Chaos_Seeding_Mathematical_Framework.docx
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional, Callable


class ChaosSeeder:
    """
    Residual Chaos Seeding for combinatorial optimization.
    
    This class implements the RCS framework that uses residual analysis
    to identify and exploit geometric incoherence in solution spaces.
    """
    
    def __init__(
        self,
        window_size: int = 3,
        chaos_weight: float = 0.5,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize the chaos seeder.
        
        Args:
            window_size: Smoothing window for projection (w in paper)
            chaos_weight: Weight for chaos in distance metric (α in paper)
            max_iterations: Maximum iterations for iterative refinement
            convergence_threshold: Convergence threshold for chaos magnitude
        """
        self.window_size = window_size
        self.chaos_weight = chaos_weight
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def compute_projection(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> np.ndarray:
        """
        Compute smooth projection P(T) of tour using Gaussian smoothing.
        
        P(T)ᵢ = Σⱼ w(i,j) · cⱼ / Σⱼ w(i,j)
        where w(i,j) = exp(-|i-j|² / 2σ²)
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Projected coordinates of shape (n_cities, 2)
        """
        n_cities = len(tour)
        
        # Reorder cities according to tour
        ordered_cities = cities[tour]
        
        # Apply Gaussian smoothing to each coordinate
        sigma = self.window_size / 2.0
        projected_x = gaussian_filter1d(
            ordered_cities[:, 0],
            sigma=sigma,
            mode='wrap'
        )
        projected_y = gaussian_filter1d(
            ordered_cities[:, 1],
            sigma=sigma,
            mode='wrap'
        )
        
        projected = np.column_stack([projected_x, projected_y])
        
        return projected
    
    def compute_residual(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> np.ndarray:
        """
        Compute residual R(T) = C - P(T).
        
        The residual captures what the smooth projection fails to explain,
        representing geometric incoherence in the tour.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Residual vectors of shape (n_cities, 2)
        """
        ordered_cities = cities[tour]
        projected = self.compute_projection(cities, tour)
        residual = ordered_cities - projected
        
        return residual
    
    def compute_chaos_magnitude(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> float:
        """
        Compute chaos magnitude χ(T) = ||R(T)||.
        
        The chaos magnitude quantifies geometric incoherence. Lower chaos
        indicates better alignment with smooth geometric structure.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Chaos magnitude (lower is better)
        """
        residual = self.compute_residual(cities, tour)
        chaos = np.linalg.norm(residual)
        
        return chaos
    
    def compute_chaos_weighted_distance(
        self,
        cities: np.ndarray,
        partial_tour: List[int],
        candidate: int
    ) -> float:
        """
        Compute chaos-weighted distance for adding candidate to partial tour.
        
        d_chaos(i,j) = d_geo(i,j) · (1 + α · χ(T ∪ {j}))
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            partial_tour: Current partial tour
            candidate: Candidate city to add
            
        Returns:
            Chaos-weighted distance
        """
        # Geometric distance
        if len(partial_tour) == 0:
            return 0.0
        
        last_city = partial_tour[-1]
        geo_distance = np.linalg.norm(cities[last_city] - cities[candidate])
        
        # Estimate chaos impact (use partial tour + candidate)
        test_tour = partial_tour + [candidate]
        if len(test_tour) >= 3:  # Need at least 3 cities for meaningful chaos
            chaos = self.compute_chaos_magnitude(cities, test_tour)
            chaos_factor = 1.0 + self.chaos_weight * chaos
        else:
            chaos_factor = 1.0
        
        return geo_distance * chaos_factor
    
    def greedy_construction_chaos_seeded(
        self,
        cities: np.ndarray,
        start_city: Optional[int] = None
    ) -> Tuple[List[int], float]:
        """
        Construct tour greedily using chaos-weighted distances.
        
        At each step, select the unvisited city with minimum chaos-weighted
        distance from the current city.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            start_city: Starting city (default: 0)
            
        Returns:
            Tuple of (tour, chaos_magnitude)
        """
        n_cities = len(cities)
        start_city = start_city or 0
        
        tour = [start_city]
        unvisited = set(range(n_cities)) - {start_city}
        
        while unvisited:
            # Find city with minimum chaos-weighted distance
            best_city = None
            best_distance = float('inf')
            
            for candidate in unvisited:
                distance = self.compute_chaos_weighted_distance(
                    cities, tour, candidate
                )
                if distance < best_distance:
                    best_distance = distance
                    best_city = candidate
            
            tour.append(best_city)
            unvisited.remove(best_city)
        
        # Compute final chaos
        chaos = self.compute_chaos_magnitude(cities, tour)
        
        return tour, chaos
    
    def optimize_tour_chaos_minimization(
        self,
        cities: np.ndarray,
        initial_tour: List[int]
    ) -> Tuple[List[int], float, dict]:
        """
        Optimize tour by iteratively minimizing chaos magnitude.
        
        Algorithm:
        1. Compute chaos χ(T)
        2. Try all 2-opt swaps
        3. Select swap that minimizes χ(T')
        4. Repeat until convergence
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            initial_tour: Initial tour to improve
            
        Returns:
            Tuple of (best_tour, best_chaos, info_dict)
        """
        current_tour = initial_tour.copy()
        current_chaos = self.compute_chaos_magnitude(cities, current_tour)
        
        info = {
            'iterations': 0,
            'chaos_history': [current_chaos],
            'length_history': [self._tour_length(cities, current_tour)],
            'improvements': 0
        }
        
        for iteration in range(self.max_iterations):
            improved = False
            best_swap = None
            best_chaos = current_chaos
            
            n_cities = len(current_tour)
            
            # Try all 2-opt swaps
            for i in range(n_cities):
                for j in range(i + 2, n_cities):
                    # Apply swap
                    new_tour = current_tour.copy()
                    new_tour[i:j] = reversed(new_tour[i:j])
                    
                    # Compute chaos
                    new_chaos = self.compute_chaos_magnitude(cities, new_tour)
                    
                    if new_chaos < best_chaos:
                        best_chaos = new_chaos
                        best_swap = (i, j)
                        improved = True
            
            # Apply best swap
            if improved:
                i, j = best_swap
                current_tour[i:j] = reversed(current_tour[i:j])
                current_chaos = best_chaos
                info['improvements'] += 1
            
            info['iterations'] += 1
            info['chaos_history'].append(current_chaos)
            info['length_history'].append(self._tour_length(cities, current_tour))
            
            # Check convergence
            if not improved or abs(info['chaos_history'][-1] - info['chaos_history'][-2]) < self.convergence_threshold:
                break
        
        return current_tour, current_chaos, info
    
    def hybrid_chaos_construction(
        self,
        cities: np.ndarray,
        n_restarts: int = 5
    ) -> Tuple[List[int], float, dict]:
        """
        Hybrid approach: chaos-seeded construction + chaos minimization.
        
        Combines the strengths of both methods:
        1. Chaos-seeded greedy construction for good initial tour
        2. Chaos minimization for refinement
        3. Multiple restarts from different starting cities
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            n_restarts: Number of random restarts
            
        Returns:
            Tuple of (best_tour, best_length, info_dict)
        """
        n_cities = len(cities)
        best_tour = None
        best_length = float('inf')
        
        info = {
            'restarts': [],
            'best_chaos': float('inf'),
            'construction_method': 'chaos_seeded'
        }
        
        for restart in range(n_restarts):
            # Chaos-seeded construction
            start_city = restart % n_cities
            tour, chaos = self.greedy_construction_chaos_seeded(cities, start_city)
            
            # Chaos minimization
            tour, chaos, opt_info = self.optimize_tour_chaos_minimization(cities, tour)
            
            # Compute length
            length = self._tour_length(cities, tour)
            
            restart_info = {
                'start_city': start_city,
                'initial_chaos': chaos,
                'final_chaos': opt_info['chaos_history'][-1],
                'final_length': length,
                'iterations': opt_info['iterations']
            }
            info['restarts'].append(restart_info)
            
            # Update best
            if length < best_length:
                best_tour = tour
                best_length = length
                info['best_chaos'] = chaos
        
        return best_tour, best_length, info
    
    def compute_chaos_spectrum(
        self,
        cities: np.ndarray,
        tour: List[int],
        window_sizes: Optional[List[int]] = None
    ) -> dict:
        """
        Compute multi-scale chaos spectrum.
        
        Analyzes chaos at multiple smoothing scales to capture both
        local and global geometric incoherence.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            window_sizes: List of window sizes for multi-scale analysis
            
        Returns:
            Dictionary with chaos magnitudes at each scale
        """
        window_sizes = window_sizes or [1, 2, 3, 5, 7, 10]
        
        spectrum = {}
        original_window = self.window_size
        
        for w in window_sizes:
            self.window_size = w
            chaos = self.compute_chaos_magnitude(cities, tour)
            spectrum[f'chaos_w{w}'] = chaos
        
        # Restore original window size
        self.window_size = original_window
        
        return spectrum
    
    def visualize_residual_field(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residual field for visualization.
        
        Returns the original positions, projected positions, and residual
        vectors for plotting.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Tuple of (ordered_cities, projected, residual)
        """
        ordered_cities = cities[tour]
        projected = self.compute_projection(cities, tour)
        residual = self.compute_residual(cities, tour)
        
        return ordered_cities, projected, residual
    
    def _tour_length(self, cities: np.ndarray, tour: List[int]) -> float:
        """Compute total tour length."""
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
        return length


class AdaptiveChaosSeeder(ChaosSeeder):
    """
    Adaptive chaos seeding with dynamic parameter adjustment.
    
    Extends ChaosSeeder with adaptive chaos weighting that decays
    during optimization (exploration → exploitation).
    """
    
    def __init__(
        self,
        window_size: int = 3,
        initial_chaos_weight: float = 1.0,
        final_chaos_weight: float = 0.1,
        decay_rate: float = 0.1,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize adaptive chaos seeder.
        
        Args:
            window_size: Smoothing window for projection
            initial_chaos_weight: Initial chaos weight (exploration)
            final_chaos_weight: Final chaos weight (exploitation)
            decay_rate: Exponential decay rate (τ in paper)
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold
        """
        super().__init__(
            window_size=window_size,
            chaos_weight=initial_chaos_weight,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        self.initial_chaos_weight = initial_chaos_weight
        self.final_chaos_weight = final_chaos_weight
        self.decay_rate = decay_rate
    
    def adaptive_chaos_weight(self, iteration: int) -> float:
        """
        Compute adaptive chaos weight with exponential decay.
        
        α(k) = α_final + (α_initial - α_final) · exp(-k/τ)
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Chaos weight for current iteration
        """
        decay = np.exp(-iteration / self.decay_rate)
        weight = self.final_chaos_weight + \
                 (self.initial_chaos_weight - self.final_chaos_weight) * decay
        return weight
    
    def optimize_tour_adaptive_chaos(
        self,
        cities: np.ndarray,
        initial_tour: List[int]
    ) -> Tuple[List[int], float, dict]:
        """
        Optimize tour with adaptive chaos weighting.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            initial_tour: Initial tour to improve
            
        Returns:
            Tuple of (best_tour, best_length, info_dict)
        """
        current_tour = initial_tour.copy()
        
        info = {
            'iterations': 0,
            'chaos_history': [],
            'length_history': [],
            'weight_history': [],
            'improvements': 0
        }
        
        for iteration in range(self.max_iterations):
            # Update chaos weight
            self.chaos_weight = self.adaptive_chaos_weight(iteration)
            info['weight_history'].append(self.chaos_weight)
            
            # Optimize with current weight
            current_tour, chaos, opt_info = self.optimize_tour_chaos_minimization(
                cities, current_tour
            )
            
            info['chaos_history'].extend(opt_info['chaos_history'])
            info['length_history'].extend(opt_info['length_history'])
            info['improvements'] += opt_info['improvements']
            info['iterations'] += opt_info['iterations']
            
            # Check convergence
            if opt_info['improvements'] == 0:
                break
        
        final_length = self._tour_length(cities, current_tour)
        
        return current_tour, final_length, info
