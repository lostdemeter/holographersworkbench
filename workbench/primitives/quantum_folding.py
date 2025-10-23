"""
Quantum Entanglement Dimensional Folding

A quantum-inspired optimization framework based on dimensional folding and
entanglement exploitation. By projecting solution spaces into multiple
Hausdorff dimensions, we reveal hidden correlations that guide search
toward high-quality solutions.

Mathematical Foundation:
- Dimensional folding operator F_D projects solutions between dimensions
- Entanglement metric E(i,j) combines geometric and topological correlations
- Measurement collapse M_D extracts improved solutions from folded space

Reference: Quantum_Entanglement_Dimensional_Folding.docx
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from typing import List, Tuple, Optional


class QuantumFolder:
    """
    Quantum-inspired dimensional folding for optimization problems.
    
    This class implements the dimensional folding framework that projects
    solution spaces into multiple Hausdorff dimensions to reveal hidden
    structure and guide optimization.
    """
    
    def __init__(
        self,
        dimensions: Optional[List[float]] = None,
        noise_scale: float = 0.1,
        perturbation: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the quantum folder.
        
        Args:
            dimensions: List of target dimensions for folding (default: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
            noise_scale: Scale factor for noise injection in expansion (α in paper)
            perturbation: Perturbation factor for collapse (ε in paper)
            random_state: Random seed for reproducibility
        """
        self.dimensions = dimensions or [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.noise_scale = noise_scale
        self.perturbation = perturbation
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fold_dimension_collapse(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Collapse cities to lower dimension (D < 2).
        
        Uses PCA to project to 1D, then adds small perturbation to maintain 2D structure.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            target_dim: Target Hausdorff dimension (< 2)
            
        Returns:
            Folded cities in 2D space with reduced effective dimension
        """
        n_cities = len(cities)
        
        # Project to 1D using PCA
        pca = PCA(n_components=1, random_state=self.random_state)
        collapsed = pca.fit_transform(cities)
        
        # Add small perturbation to maintain 2D structure
        t = np.linspace(0, 1, n_cities)
        folded = np.column_stack([
            collapsed.flatten(),
            t * self.perturbation * target_dim
        ])
        
        return folded
    
    def fold_dimension_expand(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Expand cities to higher dimension (D > 2).
        
        Injects noise into distance matrix scaled by target dimension,
        then uses MDS to embed back into 2D with expanded structure.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            target_dim: Target Hausdorff dimension (> 2)
            
        Returns:
            Folded cities in 2D space with expanded effective dimension
        """
        # Compute distance matrix
        dist_matrix = squareform(pdist(cities))
        
        # Inject noise scaled by dimension
        noise_scale = (target_dim - 2) * self.noise_scale
        noisy_dist = dist_matrix + np.random.randn(*dist_matrix.shape) * noise_scale
        
        # Ensure valid distance matrix
        noisy_dist = np.maximum(noisy_dist, 0)
        np.fill_diagonal(noisy_dist, 0)
        noisy_dist = (noisy_dist + noisy_dist.T) / 2  # Symmetrize
        
        # Embed back to 2D using MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=self.random_state)
        folded = mds.fit_transform(noisy_dist)
        
        return folded
    
    def fold_dimension(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Fold cities to target Hausdorff dimension.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            target_dim: Target Hausdorff dimension
            
        Returns:
            Folded cities in 2D space
        """
        if target_dim < 2.0:
            return self.fold_dimension_collapse(cities, target_dim)
        elif target_dim > 2.0:
            return self.fold_dimension_expand(cities, target_dim)
        else:
            return cities.copy()
    
    def compute_entanglement(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> np.ndarray:
        """
        Compute entanglement matrix combining geometric and topological correlations.
        
        E(i,j) = (1 / (1 + d_geo(i,j))) * (1 / (1 + d_topo(i,j)))
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Symmetric entanglement matrix of shape (n_cities, n_cities)
        """
        n_cities = len(cities)
        dist_matrix = squareform(pdist(cities))
        entanglement_matrix = np.zeros((n_cities, n_cities))
        
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                # Geometric correlation (inverse distance)
                geo_corr = 1.0 / (1.0 + dist_matrix[i, j])
                
                # Topological correlation (tour proximity)
                tour_i = tour.index(i)
                tour_j = tour.index(j)
                tour_dist = min(
                    abs(tour_i - tour_j),
                    n_cities - abs(tour_i - tour_j)
                )
                topo_corr = 1.0 / (1.0 + tour_dist)
                
                # Combined entanglement
                entanglement = geo_corr * topo_corr
                entanglement_matrix[i, j] = entanglement
                entanglement_matrix[j, i] = entanglement
        
        return entanglement_matrix
    
    def compute_entanglement_score(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> float:
        """
        Compute scalar entanglement score for a tour.
        
        χ(T) = Σᵢⱼ E(i,j) / n²
        
        Higher entanglement indicates better alignment between geometric
        and topological structure.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            tour: List of city indices representing tour order
            
        Returns:
            Entanglement score (higher is better)
        """
        entanglement_matrix = self.compute_entanglement(cities, tour)
        n_cities = len(cities)
        return np.sum(entanglement_matrix) / (n_cities ** 2)
    
    def measure_collapse(
        self,
        original_cities: np.ndarray,
        folded_cities: np.ndarray,
        tour: List[int],
        n_candidates: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Measurement collapse: identify promising edge swaps in folded space.
        
        Finds pairs of cities that are close in folded space but far in
        tour order, suggesting beneficial 2-opt swaps.
        
        Args:
            original_cities: Original city coordinates
            folded_cities: Folded city coordinates
            tour: Current tour
            n_candidates: Number of swap candidates to return
            
        Returns:
            List of (i, j) tuples representing promising edge swaps
        """
        n_cities = len(original_cities)
        
        # Compute distances in folded space
        folded_dist = squareform(pdist(folded_cities))
        
        # Find pairs that are close in folded space but far in tour
        candidates = []
        for i in range(n_cities):
            for j in range(i + 2, n_cities):  # Skip adjacent cities
                if j == (i + 1) % n_cities or i == (j + 1) % n_cities:
                    continue
                
                # Score: close in folded space, far in tour
                tour_i = tour.index(i)
                tour_j = tour.index(j)
                tour_dist = min(
                    abs(tour_i - tour_j),
                    n_cities - abs(tour_i - tour_j)
                )
                
                # Prioritize: small folded distance, large tour distance
                score = tour_dist / (1.0 + folded_dist[i, j])
                candidates.append((score, i, j))
        
        # Return top candidates
        candidates.sort(reverse=True)
        return [(i, j) for _, i, j in candidates[:n_candidates]]
    
    def optimize_tour_dimensional_folding(
        self,
        cities: np.ndarray,
        initial_tour: List[int],
        n_restarts: int = 3,
        iterations_per_restart: int = 30
    ) -> Tuple[List[int], float, dict]:
        """
        Optimize TSP tour using dimensional folding.
        
        Algorithm:
        1. For each restart:
           a. For each dimension D:
              - Fold cities to dimension D
              - Identify promising swaps via measurement collapse
              - Apply best swap
           b. Apply 2-opt local search
        2. Return best tour found
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
            initial_tour: Initial tour to improve
            n_restarts: Number of random restarts
            iterations_per_restart: Iterations per restart
            
        Returns:
            Tuple of (best_tour, best_length, info_dict)
        """
        best_tour = initial_tour.copy()
        best_length = self._tour_length(cities, best_tour)
        
        info = {
            'iterations': 0,
            'improvements': 0,
            'entanglement_scores': [],
            'dimension_improvements': {d: 0 for d in self.dimensions}
        }
        
        for restart in range(n_restarts):
            current_tour = initial_tour.copy() if restart == 0 else self._random_tour(len(cities))
            current_length = self._tour_length(cities, current_tour)
            
            for iteration in range(iterations_per_restart):
                improved = False
                
                # Try each dimension
                for target_dim in self.dimensions:
                    # Fold to target dimension
                    folded_cities = self.fold_dimension(cities, target_dim)
                    
                    # Get swap candidates from measurement collapse
                    candidates = self.measure_collapse(
                        cities, folded_cities, current_tour, n_candidates=5
                    )
                    
                    # Try each candidate swap
                    for i, j in candidates:
                        new_tour = self._apply_2opt_swap(current_tour, i, j)
                        new_length = self._tour_length(cities, new_tour)
                        
                        if new_length < current_length:
                            current_tour = new_tour
                            current_length = new_length
                            improved = True
                            info['improvements'] += 1
                            info['dimension_improvements'][target_dim] += 1
                            break
                    
                    if improved:
                        break
                
                info['iterations'] += 1
                
                # Track entanglement
                entanglement = self.compute_entanglement_score(cities, current_tour)
                info['entanglement_scores'].append(entanglement)
                
                if not improved:
                    break
            
            # Apply 2-opt local search
            current_tour, current_length = self._two_opt_local_search(cities, current_tour)
            
            # Update best
            if current_length < best_length:
                best_tour = current_tour
                best_length = current_length
        
        return best_tour, best_length, info
    
    def _tour_length(self, cities: np.ndarray, tour: List[int]) -> float:
        """Compute total tour length."""
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
        return length
    
    def _apply_2opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Apply 2-opt swap between positions i and j."""
        new_tour = tour.copy()
        # Reverse segment between i and j
        tour_i = tour.index(i)
        tour_j = tour.index(j)
        if tour_i > tour_j:
            tour_i, tour_j = tour_j, tour_i
        new_tour[tour_i:tour_j+1] = reversed(new_tour[tour_i:tour_j+1])
        return new_tour
    
    def _two_opt_local_search(
        self,
        cities: np.ndarray,
        tour: List[int]
    ) -> Tuple[List[int], float]:
        """Apply 2-opt local search until no improvement."""
        n_cities = len(tour)
        improved = True
        current_tour = tour.copy()
        current_length = self._tour_length(cities, current_tour)
        
        while improved:
            improved = False
            for i in range(n_cities):
                for j in range(i + 2, n_cities):
                    # Try swap
                    new_tour = current_tour.copy()
                    new_tour[i:j] = reversed(new_tour[i:j])
                    new_length = self._tour_length(cities, new_tour)
                    
                    if new_length < current_length:
                        current_tour = new_tour
                        current_length = new_length
                        improved = True
                        break
                if improved:
                    break
        
        return current_tour, current_length
    
    def _random_tour(self, n_cities: int) -> List[int]:
        """Generate random tour."""
        tour = list(range(n_cities))
        np.random.shuffle(tour)
        return tour
