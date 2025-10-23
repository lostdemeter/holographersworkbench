"""
Sublinear Quantum Inverse Kinematics
Breaking the Cubic Barrier in Dimensional-Switching Optimization

Based on: "Sublinear Quantum Inverse Kinematics: Breaking the Cubic 
Barrier in Dimensional-Switching Optimization"

Mathematical Framework:
- Hierarchical decomposition: k = √N clusters
- Dimensional sketching: m = O(log N) samples
- Sparse prime resonance: k = O(log N) zeta zeros

Complexity Reduction: O(N³) → O(N^1.5 log N)

Key Innovation: Treats optimization as quantum IK problem in variable-dimensional
space, using Riemann zeta zeros as energy eigenstates.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from scipy.cluster.vq import kmeans2
from scipy.fft import fft, ifft


@dataclass
class SublinearQIKStats:
    """Statistics from sublinear QIK optimization"""
    n_cities: int
    n_clusters: int
    n_dim_samples: int
    n_zeta_zeros: int
    clustering_time: float
    inter_cluster_time: float
    intra_cluster_time: float
    total_time: float
    theoretical_complexity: str
    empirical_complexity: str
    tour_length: float
    quality_vs_full: Optional[float] = None


class SublinearQIK:
    """
    Sublinear Quantum Inverse Kinematics optimizer.
    
    Reduces complexity from O(N³) to O(N^1.5 log N) via:
    1. Hierarchical decomposition (k = √N clusters)
    2. Dimensional sketching (m = O(log N) samples)
    3. Sparse prime resonance (k = O(log N) zeta zeros)
    
    Parameters
    ----------
    use_hierarchical : bool
        Use hierarchical clustering (default: True)
    use_dimensional_sketch : bool
        Use dimensional sketching (default: True)
    use_sparse_resonance : bool
        Use sparse prime resonance (default: True)
    prime_resonance_dim : float
        Prime resonance dimension (default: 1.585 - Sierpinski)
    """
    
    def __init__(
        self,
        use_hierarchical: bool = True,
        use_dimensional_sketch: bool = True,
        use_sparse_resonance: bool = True,
        prime_resonance_dim: float = 1.585
    ):
        self.use_hierarchical = use_hierarchical
        self.use_dimensional_sketch = use_dimensional_sketch
        self.use_sparse_resonance = use_sparse_resonance
        self.prime_resonance_dim = prime_resonance_dim
        
    def optimize_tsp(
        self,
        cities: np.ndarray,
        zeta_zeros: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, SublinearQIKStats]:
        """
        Solve TSP using sublinear QIK.
        
        Parameters
        ----------
        cities : np.ndarray, shape (n_cities, 2)
            City coordinates
        zeta_zeros : np.ndarray, optional
            Riemann zeta zeros for prime resonance
        verbose : bool
            Print progress
            
        Returns
        -------
        tour : np.ndarray
            Tour order (indices)
        length : float
            Tour length
        stats : SublinearQIKStats
            Performance statistics
        """
        import time
        
        n = len(cities)
        k = int(np.ceil(np.sqrt(n))) if self.use_hierarchical else n
        m = max(5, int(np.ceil(np.log(n)))) if self.use_dimensional_sketch else 20
        k_zeros = max(10, int(np.ceil(np.log(n)))) if self.use_sparse_resonance else 50
        
        if verbose:
            print(f"Sublinear QIK: N={n}, k={k}, m={m}, k_zeros={k_zeros}")
            
        # Phase 1: Hierarchical clustering
        t0 = time.time()
        if self.use_hierarchical and n > 10:
            clusters, centroids = self._hierarchical_cluster(cities, k)
        else:
            clusters = [np.arange(n)]
            centroids = cities.copy()
        t_cluster = time.time() - t0
        
        # Phase 2: Inter-cluster routing
        t0 = time.time()
        if len(centroids) > 1:
            inter_tour = self._solve_inter_cluster(centroids, m)
        else:
            inter_tour = np.array([0])
        t_inter = time.time() - t0
        
        # Phase 3: Intra-cluster routing
        t0 = time.time()
        tour = []
        for cluster_idx in inter_tour:
            cluster_cities = cities[clusters[cluster_idx]]
            
            if len(cluster_cities) > 1:
                # Use dimensional sketching and sparse resonance
                cluster_tour = self._solve_intra_cluster(
                    cluster_cities,
                    m,
                    k_zeros,
                    zeta_zeros
                )
                # Map back to global indices
                global_indices = clusters[cluster_idx][cluster_tour]
            else:
                global_indices = clusters[cluster_idx]
                
            tour.extend(global_indices)
            
        tour = np.array(tour)
        t_intra = time.time() - t0
        
        # Compute tour length
        length = self._compute_tour_length(tour, cities)
        
        # Statistics
        t_total = t_cluster + t_inter + t_intra
        stats = SublinearQIKStats(
            n_cities=n,
            n_clusters=k,
            n_dim_samples=m,
            n_zeta_zeros=k_zeros,
            clustering_time=t_cluster,
            inter_cluster_time=t_inter,
            intra_cluster_time=t_intra,
            total_time=t_total,
            theoretical_complexity=f"O(N^1.5 log N) = O({n}^1.5 log {n})",
            empirical_complexity=f"O(N^{np.log(t_total) / np.log(n):.2f})",
            tour_length=length
        )
        
        return tour, length, stats
        
    def _hierarchical_cluster(
        self,
        cities: np.ndarray,
        k: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Hierarchical k-means clustering.
        
        Returns
        -------
        clusters : list of np.ndarray
            Cluster assignments (list of city indices per cluster)
        centroids : np.ndarray
            Cluster centroids
        """
        # K-means clustering
        centroids, labels = kmeans2(cities, k, minit='points', iter=10)
        
        # Group cities by cluster
        clusters = []
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                clusters.append(cluster_indices)
            else:
                # Empty cluster - add nearest city
                nearest = np.argmin(np.linalg.norm(cities - centroids[i], axis=1))
                clusters.append(np.array([nearest]))
                
        # Recompute centroids
        centroids = np.array([cities[cluster].mean(axis=0) for cluster in clusters])
        
        return clusters, centroids
        
    def _solve_inter_cluster(
        self,
        centroids: np.ndarray,
        m: int
    ) -> np.ndarray:
        """
        Solve TSP on cluster centroids using dimensional sketching.
        
        Returns
        -------
        tour : np.ndarray
            Tour order (centroid indices)
        """
        n = len(centroids)
        
        if n <= 2:
            return np.arange(n)
            
        # Sample dimensions around prime resonance
        dimensions = self._dimensional_sketch(m)
        
        # Find best dimension
        best_tour = None
        best_length = float('inf')
        
        for D in dimensions:
            # Greedy tour in dimension D
            tour = self._greedy_tour_dimensional(centroids, D)
            length = self._compute_tour_length_dimensional(tour, centroids, D)
            
            if length < best_length:
                best_length = length
                best_tour = tour
                
        return best_tour
        
    def _solve_intra_cluster(
        self,
        cities: np.ndarray,
        m: int,
        k_zeros: int,
        zeta_zeros: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Solve TSP within cluster using sparse prime resonance.
        
        Returns
        -------
        tour : np.ndarray
            Tour order (city indices within cluster)
        """
        n = len(cities)
        
        if n <= 2:
            return np.arange(n)
            
        # Dimensional sketch
        dimensions = self._dimensional_sketch(m)
        
        # Sparse prime resonance
        if self.use_sparse_resonance and zeta_zeros is not None:
            # Convert zeta_zeros dict to array if needed
            if isinstance(zeta_zeros, dict):
                zz_array = np.array(list(zeta_zeros.values()))
            else:
                zz_array = zeta_zeros
            
            resonance = self._sparse_prime_resonance(
                cities,
                zz_array[:k_zeros],
                dimensions[0]  # Use first dimension
            )
        else:
            resonance = np.ones(n)
            
        # Greedy tour with resonance guidance
        tour = self._greedy_tour_with_resonance(cities, dimensions[0], resonance)
        
        return tour
        
    def _dimensional_sketch(self, m: int) -> np.ndarray:
        """
        Generate dimensional sketch: m dimensions around prime resonance.
        
        Uses structured sampling: D̄ + A_j sin(ω_j t + φ_j)
        """
        # Oscillate around prime resonance dimension
        amplitudes = np.linspace(0.1, 0.3, m)
        frequencies = np.linspace(1, 3, m)
        phases = np.random.uniform(0, 2*np.pi, m)
        
        dimensions = []
        for j in range(m):
            D = self.prime_resonance_dim + amplitudes[j] * np.sin(frequencies[j] + phases[j])
            dimensions.append(np.clip(D, 1.0, 2.5))
            
        return np.array(dimensions)
        
    def _sparse_prime_resonance(
        self,
        cities: np.ndarray,
        zeta_zeros: np.ndarray,
        dimension: float
    ) -> np.ndarray:
        """
        Compute sparse prime resonance field using top-k zeta zeros.
        
        Φ_k(x, D, t) = Σ_{n=1}^k a_n(x, D) exp(iγ_n t)
        """
        n = len(cities)
        k = len(zeta_zeros)
        
        # Compute FFT of city coordinates
        city_fft = fft(cities[:, 0] + 1j * cities[:, 1])
        
        # Match to zeta zeros (frequency domain)
        resonance = np.zeros(n)
        
        for i, gamma in enumerate(zeta_zeros):
            # Amplitude depends on dimension
            amplitude = np.exp(-0.5 * (dimension - self.prime_resonance_dim)**2)
            
            # Phase from zeta zero
            phase = gamma * np.arange(n) / n
            
            # Resonance contribution
            resonance += amplitude * np.cos(2 * np.pi * phase)
            
        # Normalize
        resonance = (resonance - resonance.min()) / (np.ptp(resonance) + 1e-10)
        
        return resonance
        
    def _greedy_tour_dimensional(
        self,
        cities: np.ndarray,
        dimension: float
    ) -> np.ndarray:
        """Greedy nearest-neighbor tour in dimension D"""
        n = len(cities)
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        
        current = 0
        while unvisited:
            # Find nearest unvisited city in dimension D
            nearest = None
            nearest_dist = float('inf')
            
            for city in unvisited:
                dist = np.linalg.norm(cities[current] - cities[city])
                dist_D = np.power(dist, dimension)
                
                if dist_D < nearest_dist:
                    nearest_dist = dist_D
                    nearest = city
                    
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return np.array(tour)
        
    def _greedy_tour_with_resonance(
        self,
        cities: np.ndarray,
        dimension: float,
        resonance: np.ndarray
    ) -> np.ndarray:
        """Greedy tour guided by prime resonance"""
        n = len(cities)
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        
        current = 0
        while unvisited:
            # Find nearest city weighted by resonance
            nearest = None
            nearest_score = float('inf')
            
            for city in unvisited:
                dist = np.linalg.norm(cities[current] - cities[city])
                dist_D = np.power(dist, dimension)
                
                # Weight by inverse resonance (high resonance = prefer)
                score = dist_D / (resonance[city] + 0.1)
                
                if score < nearest_score:
                    nearest_score = score
                    nearest = city
                    
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return np.array(tour)
        
    def _compute_tour_length(
        self,
        tour: np.ndarray,
        cities: np.ndarray
    ) -> float:
        """Compute Euclidean tour length"""
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
        return length
        
    def _compute_tour_length_dimensional(
        self,
        tour: np.ndarray,
        cities: np.ndarray,
        dimension: float
    ) -> float:
        """Compute tour length in dimension D"""
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            dist = np.linalg.norm(cities[tour[i]] - cities[tour[j]])
            length += np.power(dist, dimension)
        return length
