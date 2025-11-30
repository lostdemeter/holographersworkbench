"""
Clock-Resonant Quantum Entanglement Dimensional Folding

An upgraded version of QuantumFolder that uses clock eigenphases instead of
random numbers. This provides:

1. REPRODUCIBILITY: Same input always gives same output
2. EQUIDISTRIBUTION: Clock phases cover the space uniformly
3. DETERMINISTIC DIVERSIFICATION: No seed management needed

The clock phases replace:
- Random noise in dimension expansion
- Random dimension sampling
- Random perturbations in collapse

Based on the Clock Resonance Compiler framework.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Try to import sklearn, but make it optional
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, smacof
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


def recursive_theta(n: int, ratio: float = PHI) -> float:
    """
    Real recursive clock phase using the doubling rule.
    
    θ(n) = θ(n//2) + δ ± arctan(tan(θ(n//2)))
    """
    if n <= 0:
        return 0.0
    prev = recursive_theta(n // 2, ratio)
    bit = n % 2
    delta = 2 * np.pi * ratio
    tan_prev = np.tan(prev % np.pi - np.pi/2 + 1e-10)
    if bit:
        return prev + delta + np.arctan(tan_prev)
    else:
        return prev + delta - np.arctan(tan_prev)


class ClockOracleMixin:
    """
    Mixin that provides clock oracle access to any class.
    
    This is a standalone version to avoid import chain issues.
    """
    
    _clock_counter: int = 0
    
    def get_clock_phase(self, n: int = None, clock: str = 'golden') -> float:
        """Get a clock phase. If n is None, uses auto-incrementing counter."""
        if n is None:
            self._clock_counter += 1
            n = self._clock_counter
        theta = recursive_theta(n, PHI)
        return (theta / (2 * np.pi)) % 1.0
    
    def reset_clock_counter(self):
        """Reset the auto-incrementing counter."""
        self._clock_counter = 0


@dataclass
class ClockFoldingStats:
    """Statistics from clock-resonant folding operations."""
    n_cities: int
    dimensions_tried: List[float]
    best_dimension: float
    clock_phases_used: int
    improvement: float
    time: float


class ClockAdaptiveDimensionalSampler(ClockOracleMixin):
    """
    Clock-resonant adaptive sampler that uses clock phases for dimension selection.
    
    Instead of random sampling, uses clock phases to deterministically
    select dimensions while still maintaining good coverage.
    """
    
    def __init__(self, dimensions: List[float]):
        """Initialize with clock oracle."""
        super().__init__()
        self.dimensions = dimensions
        self.success_counts = {d: 1 for d in dimensions}  # Laplace smoothing
        self.total_attempts = {d: 1 for d in dimensions}
        self._sample_counter = 0
    
    def sample_dimensions(self, n_samples: int = 3) -> List[float]:
        """
        Sample dimensions using clock phases weighted by success rate.
        
        Uses clock phases to provide deterministic but well-distributed selection.
        """
        success_rates = {
            d: self.success_counts[d] / self.total_attempts[d]
            for d in self.dimensions
        }
        
        # Sort dimensions by success rate
        sorted_dims = sorted(self.dimensions, key=lambda d: success_rates[d], reverse=True)
        
        # Use clock phases to select from top candidates
        n_samples = min(n_samples, len(self.dimensions))
        selected = []
        
        for i in range(n_samples):
            # Get clock phase for this selection
            phase = self.get_clock_phase(self._sample_counter + i)
            self._sample_counter += 1
            
            # Use phase to select from remaining dimensions
            remaining = [d for d in sorted_dims if d not in selected]
            if remaining:
                # Weight by success rate, modulated by clock phase
                weights = np.array([success_rates[d] for d in remaining])
                weights = weights * (0.5 + 0.5 * np.sin(2 * np.pi * phase + np.arange(len(weights))))
                weights = np.maximum(weights, 0.01)  # Ensure positive
                weights /= weights.sum()
                
                # Select based on cumulative weights
                cumsum = np.cumsum(weights)
                idx = np.searchsorted(cumsum, phase)
                idx = min(idx, len(remaining) - 1)
                selected.append(remaining[idx])
        
        return selected
    
    def update(self, dimension: float, success: bool):
        """Update statistics after trying a dimension."""
        self.total_attempts[dimension] += 1
        if success:
            self.success_counts[dimension] += 1


class ClockQuantumFolder(ClockOracleMixin):
    """
    Clock-resonant quantum-inspired dimensional folding.
    
    This class replaces all random operations in QuantumFolder with
    deterministic clock phases, providing reproducibility while
    maintaining the optimization quality.
    
    Key changes from QuantumFolder:
    - Noise injection uses clock_randn() instead of np.random.randn()
    - Dimension sampling uses ClockAdaptiveDimensionalSampler
    - Perturbations use clock phases
    """
    
    def __init__(
        self,
        dimensions: Optional[List[float]] = None,
        noise_scale: float = 0.1,
        perturbation: float = 0.1
    ):
        """
        Initialize the clock-resonant quantum folder.
        
        Args:
            dimensions: List of target dimensions for folding
            noise_scale: Scale factor for noise injection (α)
            perturbation: Perturbation factor for collapse (ε)
        """
        super().__init__()
        self.dimensions = dimensions or [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.noise_scale = noise_scale
        self.perturbation = perturbation
        
        # Clock-resonant dimension sampler
        self.dimension_sampler = ClockAdaptiveDimensionalSampler(self.dimensions)
        
        # Caches
        self._dist_matrix_cache = None
        self._noise_counter = 0
    
    def _get_distance_matrix(self, cities: np.ndarray) -> np.ndarray:
        """Get cached distance matrix or compute if not cached."""
        if self._dist_matrix_cache is None or self._dist_matrix_cache.shape[0] != len(cities):
            self._dist_matrix_cache = squareform(pdist(cities))
        return self._dist_matrix_cache
    
    def _classical_mds(self, dist_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Classical MDS (Torgerson scaling) without sklearn.
        
        Embeds distance matrix into n_components dimensions.
        """
        n = dist_matrix.shape[0]
        
        # Double centering
        D_sq = dist_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top n_components
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        
        # Handle negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Compute embedding
        embedding = eigenvectors * np.sqrt(eigenvalues)
        
        return embedding
    
    def _clock_noise_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate a noise matrix using clock phases.
        
        Uses Box-Muller transform on clock phases to generate
        Gaussian-like noise that is deterministic.
        """
        n_elements = shape[0] * shape[1]
        noise = np.zeros(n_elements)
        
        for i in range(0, n_elements, 2):
            # Box-Muller transform using clock phases
            u1 = self.get_clock_phase(self._noise_counter)
            u2 = self.get_clock_phase(self._noise_counter + 1)
            self._noise_counter += 2
            
            # Avoid log(0)
            u1 = max(u1, 1e-10)
            
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            
            noise[i] = z1
            if i + 1 < n_elements:
                noise[i + 1] = z2
        
        return noise.reshape(shape)
    
    def fold_dimension_collapse(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Collapse cities to lower dimension (D < 2) using clock phases.
        
        Uses clock phases for the perturbation instead of linspace.
        """
        n_cities = len(cities)
        
        # Project to 1D using PCA or simple projection
        if HAS_SKLEARN:
            pca = PCA(n_components=1)
            collapsed = pca.fit_transform(cities)
        else:
            # Simple projection onto first principal component
            centered = cities - cities.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            collapsed = centered @ Vt[0:1].T
        
        # Use clock phases for perturbation
        perturbations = np.array([
            self.get_clock_phase(i) for i in range(n_cities)
        ])
        
        folded = np.column_stack([
            collapsed.flatten(),
            perturbations * self.perturbation * target_dim
        ])
        
        return folded
    
    def fold_dimension_expand(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Expand cities to higher dimension (D > 2) using clock noise.
        
        Injects clock-based noise into distance matrix scaled by target dimension,
        then uses MDS to embed back into 2D with expanded structure.
        """
        # Compute distance matrix
        dist_matrix = squareform(pdist(cities))
        
        # Inject CLOCK noise scaled by dimension
        noise_scale = (target_dim - 2) * self.noise_scale
        clock_noise = self._clock_noise_matrix(dist_matrix.shape)
        noisy_dist = dist_matrix + clock_noise * noise_scale
        
        # Ensure valid distance matrix
        noisy_dist = np.maximum(noisy_dist, 0)
        np.fill_diagonal(noisy_dist, 0)
        noisy_dist = (noisy_dist + noisy_dist.T) / 2  # Symmetrize
        
        # Embed back to 2D using MDS or classical MDS
        if HAS_SKLEARN:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            folded = mds.fit_transform(noisy_dist)
        else:
            # Classical MDS without sklearn
            folded = self._classical_mds(noisy_dist, 2)
        
        return folded
    
    def fold_dimension_expand_fast(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Fast MDS expansion with clock noise and early stopping.
        """
        # Compute distance matrix
        dist_matrix = squareform(pdist(cities))
        
        # Inject CLOCK noise scaled by dimension
        noise_scale = (target_dim - 2) * self.noise_scale
        clock_noise = self._clock_noise_matrix(dist_matrix.shape)
        noisy_dist = dist_matrix + clock_noise * noise_scale
        
        # Ensure valid distance matrix
        noisy_dist = np.maximum(noisy_dist, 0)
        np.fill_diagonal(noisy_dist, 0)
        noisy_dist = (noisy_dist + noisy_dist.T) / 2  # Symmetrize
        
        # Fast SMACOF with early stopping or classical MDS
        if HAS_SKLEARN:
            folded, stress = smacof(
                noisy_dist,
                n_components=2,
                max_iter=50,
                eps=1e-3,
                random_state=42
            )
        else:
            folded = self._classical_mds(noisy_dist, 2)
        
        return folded
    
    def fold_dimension(self, cities: np.ndarray, target_dim: float) -> np.ndarray:
        """
        Fold cities to target Hausdorff dimension using clock phases.
        """
        if target_dim < 2.0:
            return self.fold_dimension_collapse(cities, target_dim)
        elif target_dim > 2.0:
            return self.fold_dimension_expand_fast(cities, target_dim)
        else:
            return cities.copy()
    
    def _build_position_map(self, tour: np.ndarray) -> np.ndarray:
        """Build position mapping for O(1) lookups."""
        n = len(tour)
        position_map = np.zeros(n, dtype=np.int32)
        for pos, city in enumerate(tour):
            position_map[city] = pos
        return position_map
    
    def compute_entanglement_vectorized(
        self,
        cities: np.ndarray,
        tour: np.ndarray,
        position_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Vectorized entanglement computation.
        
        E(i,j) = (1 / (1 + d_geo(i,j))) * (1 / (1 + d_topo(i,j)))
        """
        n_cities = len(cities)
        dist_matrix = self._get_distance_matrix(cities)
        
        # Build position map if not provided
        if position_map is None:
            position_map = self._build_position_map(tour)
        
        # Geometric correlation
        geo_corr = 1.0 / (1.0 + dist_matrix)
        
        # Topological correlation (vectorized)
        positions = position_map[np.arange(n_cities)]
        pos_i = positions[:, np.newaxis]
        pos_j = positions[np.newaxis, :]
        tour_dist = np.minimum(
            np.abs(pos_i - pos_j),
            n_cities - np.abs(pos_i - pos_j)
        )
        topo_corr = 1.0 / (1.0 + tour_dist)
        
        # Combined entanglement
        entanglement = geo_corr * topo_corr
        np.fill_diagonal(entanglement, 0)
        
        return entanglement
    
    def measure_collapse(
        self,
        cities: np.ndarray,
        tour: np.ndarray,
        folded_cities: np.ndarray,
        n_candidates: int = 5
    ) -> List[np.ndarray]:
        """
        Measurement collapse: extract candidate solutions from folded space.
        
        Uses clock phases for candidate generation instead of random sampling.
        """
        n_cities = len(cities)
        position_map = self._build_position_map(tour)
        
        # Compute entanglement in folded space
        entanglement = self.compute_entanglement_vectorized(folded_cities, tour, position_map)
        
        candidates = []
        
        for c in range(n_candidates):
            # Use clock phase to select starting point
            start_phase = self.get_clock_phase(c * 100)
            start_city = int(start_phase * n_cities) % n_cities
            
            # Build tour greedily using entanglement
            new_tour = [start_city]
            visited = {start_city}
            
            current = start_city
            for _ in range(n_cities - 1):
                # Find best unvisited city by entanglement
                best_city = None
                best_score = -1
                
                for city in range(n_cities):
                    if city not in visited:
                        score = entanglement[current, city]
                        if score > best_score:
                            best_score = score
                            best_city = city
                
                if best_city is not None:
                    new_tour.append(best_city)
                    visited.add(best_city)
                    current = best_city
            
            candidates.append(np.array(new_tour))
        
        return candidates
    
    def _compute_tour_length(self, tour: np.ndarray, cities: np.ndarray) -> float:
        """Compute total tour length."""
        dist_matrix = self._get_distance_matrix(cities)
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += dist_matrix[tour[i], tour[j]]
        return length
    
    def _two_opt(self, tour: np.ndarray, cities: np.ndarray) -> np.ndarray:
        """2-opt local search refinement."""
        tour = tour.copy()
        n = len(tour)
        dist_matrix = self._get_distance_matrix(cities)
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    
                    i1, i2 = tour[i], tour[i + 1]
                    j1, j2 = tour[j], tour[(j + 1) % n]
                    
                    old_dist = dist_matrix[i1, i2] + dist_matrix[j1, j2]
                    new_dist = dist_matrix[i1, j1] + dist_matrix[i2, j2]
                    
                    if new_dist < old_dist - 1e-10:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        
        return tour
    
    def optimize_tour(
        self,
        cities: np.ndarray,
        initial_tour: Optional[np.ndarray] = None,
        n_iterations: int = 10,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, ClockFoldingStats]:
        """
        Optimize a TSP tour using clock-resonant dimensional folding.
        
        Args:
            cities: Array of shape (n_cities, 2)
            initial_tour: Starting tour (default: nearest neighbor)
            n_iterations: Number of folding iterations
            verbose: Print progress
            
        Returns:
            (best_tour, best_length, stats)
        """
        import time
        t0 = time.time()
        
        n_cities = len(cities)
        
        # Initialize tour
        if initial_tour is None:
            # Nearest neighbor construction using clock phase for start
            start_phase = self.get_clock_phase(0)
            start = int(start_phase * n_cities) % n_cities
            
            tour = [start]
            visited = {start}
            dist_matrix = self._get_distance_matrix(cities)
            
            current = start
            for _ in range(n_cities - 1):
                best_next = None
                best_dist = float('inf')
                for city in range(n_cities):
                    if city not in visited and dist_matrix[current, city] < best_dist:
                        best_dist = dist_matrix[current, city]
                        best_next = city
                if best_next is not None:
                    tour.append(best_next)
                    visited.add(best_next)
                    current = best_next
            
            initial_tour = np.array(tour)
        
        best_tour = initial_tour.copy()
        best_length = self._compute_tour_length(best_tour, cities)
        initial_length = best_length
        
        dimensions_tried = []
        best_dimension = 2.0
        
        for iteration in range(n_iterations):
            # Sample dimensions using clock-resonant sampler
            dims_to_try = self.dimension_sampler.sample_dimensions(n_samples=3)
            
            for target_dim in dims_to_try:
                dimensions_tried.append(target_dim)
                
                # Fold to target dimension
                folded = self.fold_dimension(cities, target_dim)
                
                # Generate candidates via measurement collapse
                candidates = self.measure_collapse(cities, best_tour, folded, n_candidates=3)
                
                # Evaluate candidates
                for candidate in candidates:
                    # Refine with 2-opt
                    refined = self._two_opt(candidate, cities)
                    length = self._compute_tour_length(refined, cities)
                    
                    if length < best_length:
                        best_tour = refined
                        best_length = length
                        best_dimension = target_dim
                        self.dimension_sampler.update(target_dim, True)
                        
                        if verbose:
                            print(f"  Iter {iteration}, dim={target_dim:.1f}: "
                                  f"length={length:.4f} (improved)")
                    else:
                        self.dimension_sampler.update(target_dim, False)
        
        improvement = (initial_length - best_length) / initial_length * 100
        
        stats = ClockFoldingStats(
            n_cities=n_cities,
            dimensions_tried=dimensions_tried,
            best_dimension=best_dimension,
            clock_phases_used=self._clock_counter,
            improvement=improvement,
            time=time.time() - t0
        )
        
        return best_tour, best_length, stats


# Convenience function
def solve_tsp_clock_folding(
    cities: np.ndarray,
    n_iterations: int = 10,
    verbose: bool = False
) -> Tuple[np.ndarray, float, ClockFoldingStats]:
    """
    Solve TSP using clock-resonant quantum folding.
    
    This is the recommended entry point.
    """
    folder = ClockQuantumFolder()
    return folder.optimize_tour(cities, n_iterations=n_iterations, verbose=verbose)


if __name__ == "__main__":
    print("Clock-Resonant Quantum Folding Demo")
    print("=" * 60)
    
    np.random.seed(42)
    
    for n in [20, 50, 100]:
        cities = np.random.rand(n, 2)
        
        # Test clock-resonant folder
        folder = ClockQuantumFolder()
        tour, length, stats = folder.optimize_tour(cities, n_iterations=5, verbose=False)
        
        print(f"\nN={n}:")
        print(f"  Tour length: {length:.4f}")
        print(f"  Best dimension: {stats.best_dimension:.1f}")
        print(f"  Clock phases used: {stats.clock_phases_used}")
        print(f"  Improvement: {stats.improvement:.2f}%")
        print(f"  Time: {stats.time:.4f}s")
        
        # Verify reproducibility
        folder2 = ClockQuantumFolder()
        tour2, length2, _ = folder2.optimize_tour(cities, n_iterations=5, verbose=False)
        
        print(f"  Reproducible: {np.array_equal(tour, tour2)}")
    
    print("\n" + "=" * 60)
    print("Key advantage: REPRODUCIBILITY")
    print("Same input → Same output (always)")
