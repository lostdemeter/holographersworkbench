#!/usr/bin/env python3
"""
Demo: Adaptive Nonlocality + Sublinear QIK on TSP

Combines both novel optimization frameworks:
1. Adaptive dimensional coupling (Hausdorff resonance)
2. Sublinear complexity (hierarchical + sketching + prime resonance)

This demonstrates how the two papers complement each other:
- Adaptive Nonlocality: Finds optimal dimensional space for search
- Sublinear QIK: Achieves O(N^1.5 log N) complexity via hierarchical decomposition
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workbench import (
    AdaptiveNonlocalityOptimizer,
    SublinearQIK,
    zetazero_batch,
    GushurstCrystal
)


def generate_tsp_instance(n_cities: int, instance_type: str = 'random', seed: int = 42) -> np.ndarray:
    """Generate TSP instance"""
    np.random.seed(seed)
    
    if instance_type == 'random':
        return np.random.rand(n_cities, 2) * 100
    
    elif instance_type == 'clustered':
        # Create clusters
        n_clusters = max(3, n_cities // 10)
        cluster_centers = np.random.rand(n_clusters, 2) * 100
        cities = []
        for i in range(n_cities):
            cluster = cluster_centers[i % n_clusters]
            noise = np.random.randn(2) * 5
            cities.append(cluster + noise)
        return np.array(cities)
    
    elif instance_type == 'prime_structured':
        # Hidden prime structure (resonates with zeta zeros)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        cities = []
        for i in range(n_cities):
            p = primes[i % len(primes)]
            r = np.sqrt(p) * 5
            theta = 2 * np.pi * p / 50
            x = r * np.cos(theta) + np.random.normal(0, 2)
            y = r * np.sin(theta) + np.random.normal(0, 2)
            cities.append([x, y])
        return np.array(cities)
    
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")


def greedy_nearest_neighbor(cities: np.ndarray) -> np.ndarray:
    """Simple greedy construction"""
    n = len(cities)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda c: np.linalg.norm(
            cities[current] - cities[c]
        ))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return np.array(tour)


def tour_length(cities: np.ndarray, tour: np.ndarray) -> float:
    """Compute tour length"""
    length = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
    return length


def dimensional_local_search(solution: np.ndarray, cities: np.ndarray, dimension: float) -> np.ndarray:
    """
    Local search operator for Adaptive Nonlocality.
    
    Performs 2-opt swaps in the specified Hausdorff dimension.
    """
    n = len(solution)
    best_solution = solution.copy()
    best_length = tour_length(cities, best_solution)
    
    # Try a few random 2-opt swaps
    for _ in range(5):
        i = np.random.randint(0, n-2)
        j = np.random.randint(i+2, n)
        
        # Create new tour with 2-opt swap
        new_solution = solution.copy()
        new_solution[i+1:j+1] = list(reversed(new_solution[i+1:j+1]))
        
        # Compute length in dimension D
        new_length = 0.0
        for k in range(n):
            l = (k + 1) % n
            dist = np.linalg.norm(cities[new_solution[k]] - cities[new_solution[l]])
            new_length += np.power(dist, dimension)
        
        if new_length < best_length:
            best_solution = new_solution
            best_length = new_length
    
    return best_solution


def visualize_results(cities: np.ndarray, tours: dict, trajectory=None):
    """Visualize TSP solutions and dimensional trajectory"""
    n_methods = len(tours)
    
    if trajectory is not None:
        fig = plt.figure(figsize=(15, 10))
        
        # TSP solutions
        for idx, (method, tour) in enumerate(tours.items()):
            ax = plt.subplot(2, n_methods, idx + 1)
            
            # Plot tour
            tour_coords = cities[tour]
            tour_coords = np.vstack([tour_coords, tour_coords[0]])
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', alpha=0.6)
            ax.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
            ax.scatter(cities[tour[0], 0], cities[tour[0], 1], c='green', s=100, marker='*', zorder=6)
            
            length = tour_length(cities, tour)
            ax.set_title(f'{method}\nLength: {length:.2f}')
            ax.set_aspect('equal')
        
        # Dimensional trajectory
        ax = plt.subplot(2, 1, 2)
        iterations = trajectory.iterations
        dimensions = trajectory.dimensions
        phases = trajectory.phase
        
        # Color by phase
        colors = {'exploration': 'red', 'coupling': 'orange', 'exploitation': 'green'}
        for phase in ['exploration', 'coupling', 'exploitation']:
            mask = np.array(phases) == phase
            if mask.any():
                ax.scatter(np.array(iterations)[mask], np.array(dimensions)[mask],
                          c=colors[phase], label=phase, alpha=0.6, s=20)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hausdorff Dimension')
        ax.set_title('Dimensional Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method, tour) in zip(axes, tours.items()):
            tour_coords = cities[tour]
            tour_coords = np.vstack([tour_coords, tour_coords[0]])
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', alpha=0.6)
            ax.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
            ax.scatter(cities[tour[0], 0], cities[tour[0], 1], c='green', s=100, marker='*', zorder=6)
            
            length = tour_length(cities, tour)
            ax.set_title(f'{method}\nLength: {length:.2f}')
            ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def main():
    print("="*80)
    print("ADAPTIVE NONLOCALITY + SUBLINEAR QIK DEMO")
    print("="*80)
    
    # Configuration
    n_cities = 50
    instance_type = 'prime_structured'  # 'random', 'clustered', 'prime_structured'
    seed = 42
    
    print(f"\nGenerating {instance_type} TSP instance with {n_cities} cities...")
    cities = generate_tsp_instance(n_cities, instance_type, seed)
    
    # Baseline: Greedy nearest neighbor
    print("\n1. Baseline (Greedy Nearest Neighbor)...")
    t0 = time.time()
    baseline_tour = greedy_nearest_neighbor(cities)
    baseline_length = tour_length(cities, baseline_tour)
    baseline_time = time.time() - t0
    print(f"   Length: {baseline_length:.2f}")
    print(f"   Time: {baseline_time:.4f}s")
    
    # Method 1: Sublinear QIK
    print("\n2. Sublinear QIK (Hierarchical + Prime Resonance)...")
    
    # Get zeta zeros for prime resonance
    print("   Computing zeta zeros...")
    gc = GushurstCrystal(n_zeros=20)
    zeta_zeros = zetazero_batch(1, 20)
    
    qik = SublinearQIK(
        use_hierarchical=True,
        use_dimensional_sketch=True,
        use_sparse_resonance=True
    )
    
    t0 = time.time()
    qik_tour, qik_length, qik_stats = qik.optimize_tsp(cities, zeta_zeros, verbose=True)
    qik_time = time.time() - t0
    
    print(f"\n   Results:")
    print(f"   Length: {qik_length:.2f}")
    print(f"   Improvement: {(baseline_length - qik_length) / baseline_length * 100:.2f}%")
    print(f"   Time: {qik_time:.4f}s")
    print(f"   Complexity: {qik_stats.theoretical_complexity}")
    print(f"   Empirical: {qik_stats.empirical_complexity}")
    
    # Method 2: Adaptive Nonlocality
    print("\n3. Adaptive Nonlocality (Dimensional Coupling)...")
    
    anl = AdaptiveNonlocalityOptimizer(
        d_min=1.0,
        d_max=2.5,
        n_dim_samples=30,
        t_initial=2.0,
        t_final=0.5
    )
    
    def cost_fn(tour, cities):
        return tour_length(cities, tour)
    
    t0 = time.time()
    anl_tour, anl_length, trajectory = anl.optimize(
        baseline_tour,
        cities,
        cost_fn,
        dimensional_local_search,
        max_iterations=100,
        verbose=True
    )
    anl_time = time.time() - t0
    
    print(f"\n   Results:")
    print(f"   Length: {anl_length:.2f}")
    print(f"   Improvement: {(baseline_length - anl_length) / baseline_length * 100:.2f}%")
    print(f"   Time: {anl_time:.4f}s")
    
    # Analyze trajectory
    analysis = anl.analyze_trajectory(trajectory)
    print(f"\n   Trajectory Analysis:")
    print(f"   Final dimension: {analysis['final_dimension']:.3f}")
    print(f"   Dimensional entropy: {analysis['dimensional_entropy']:.3f}")
    print(f"   Convergence iteration: {analysis['convergence_iteration']}")
    
    for phase, stats in analysis['phase_statistics'].items():
        print(f"   {phase.capitalize()}: D={stats['mean_dimension']:.3f}±{stats['std_dimension']:.3f}, "
              f"{stats['iterations']} iterations")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'Length':<12} {'Improvement':<15} {'Time (s)':<10}")
    print("-"*80)
    print(f"{'Baseline (Greedy)':<30} {baseline_length:<12.2f} {'-':<15} {baseline_time:<10.4f}")
    print(f"{'Sublinear QIK':<30} {qik_length:<12.2f} {(baseline_length - qik_length) / baseline_length * 100:<14.2f}% {qik_time:<10.4f}")
    print(f"{'Adaptive Nonlocality':<30} {anl_length:<12.2f} {(baseline_length - anl_length) / baseline_length * 100:<14.2f}% {anl_time:<10.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    tours = {
        'Baseline': baseline_tour,
        'Sublinear QIK': qik_tour,
        'Adaptive Nonlocality': anl_tour
    }
    
    fig = visualize_results(cities, tours, trajectory)
    plt.savefig('adaptive_nonlocality_qik_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: adaptive_nonlocality_qik_demo.png")
    
    plt.show()


if __name__ == '__main__':
    main()
