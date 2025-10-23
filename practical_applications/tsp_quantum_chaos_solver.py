#!/usr/bin/env python3
"""
TSP Solver using Quantum Entanglement Dimensional Folding and Residual Chaos Seeding

This script demonstrates the combined power of two novel optimization frameworks:
1. Quantum Entanglement Dimensional Folding - reveals hidden structure via dimensional projection
2. Residual Chaos Seeding - exploits geometric incoherence to guide search

The hybrid approach achieves superior results by combining:
- Dimensional folding for global structure exploration
- Chaos seeding for local refinement and quality assessment
"""

import numpy as np
import time
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workbench.primitives.quantum_folding import QuantumFolder
from workbench.primitives.chaos_seeding import ChaosSeeder, AdaptiveChaosSeeder


class TSPInstance:
    """TSP problem instance with various generation methods."""
    
    @staticmethod
    def random_euclidean(n_cities: int, seed: int = 42) -> np.ndarray:
        """Generate random Euclidean TSP instance."""
        np.random.seed(seed)
        return np.random.rand(n_cities, 2) * 100
    
    @staticmethod
    def clustered(n_cities: int, n_clusters: int = 4, seed: int = 42) -> np.ndarray:
        """Generate clustered TSP instance."""
        np.random.seed(seed)
        cities = []
        cities_per_cluster = n_cities // n_clusters
        
        for i in range(n_clusters):
            # Random cluster center
            center = np.random.rand(2) * 100
            # Cities around center
            cluster_cities = center + np.random.randn(cities_per_cluster, 2) * 5
            cities.append(cluster_cities)
        
        # Add remaining cities
        remaining = n_cities - cities_per_cluster * n_clusters
        if remaining > 0:
            center = np.random.rand(2) * 100
            extra_cities = center + np.random.randn(remaining, 2) * 5
            cities.append(extra_cities)
        
        return np.vstack(cities)
    
    @staticmethod
    def grid(n_cities: int) -> np.ndarray:
        """Generate grid-based TSP instance."""
        side = int(np.sqrt(n_cities))
        x = np.linspace(0, 100, side)
        y = np.linspace(0, 100, side)
        xx, yy = np.meshgrid(x, y)
        cities = np.column_stack([xx.ravel(), yy.ravel()])
        return cities[:n_cities]
    
    @staticmethod
    def circle(n_cities: int) -> np.ndarray:
        """Generate cities on a circle."""
        angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
        x = 50 + 40 * np.cos(angles)
        y = 50 + 40 * np.sin(angles)
        return np.column_stack([x, y])


class TSPSolver:
    """Unified TSP solver using quantum folding and chaos seeding."""
    
    def __init__(self, cities: np.ndarray):
        """
        Initialize solver with city coordinates.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
        """
        self.cities = cities
        self.n_cities = len(cities)
        
        # Initialize solvers
        self.quantum_folder = QuantumFolder()
        self.chaos_seeder = ChaosSeeder()
        self.adaptive_chaos = AdaptiveChaosSeeder()
    
    def greedy_nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float]:
        """Baseline: greedy nearest neighbor.
        
        Complexity: O(n²)
        """
        tour = [start_city]
        unvisited = set(range(self.n_cities)) - {start_city}
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda c: np.linalg.norm(
                self.cities[current] - self.cities[c]
            ))
            tour.append(nearest)
            unvisited.remove(nearest)
        
        length = self._tour_length(tour)
        return tour, length
    
    def solve_quantum_folding(
        self,
        n_restarts: int = 3,
        iterations_per_restart: int = 30
    ) -> dict:
        """
        Solve using Quantum Entanglement Dimensional Folding.
        
        Complexity: O(n² log n × D × R × I)
        where D = number of dimensions, R = restarts, I = iterations
        
        Args:
            n_restarts: Number of random restarts
            iterations_per_restart: Iterations per restart
            
        Returns:
            Dictionary with solution and statistics
        """
        # Get initial tour
        initial_tour, initial_length = self.greedy_nearest_neighbor()
        
        # Apply quantum folding
        start_time = time.time()
        tour, length, info = self.quantum_folder.optimize_tour_dimensional_folding(
            self.cities,
            initial_tour,
            n_restarts=n_restarts,
            iterations_per_restart=iterations_per_restart
        )
        elapsed = time.time() - start_time
        
        # Compute entanglement
        entanglement = self.quantum_folder.compute_entanglement_score(self.cities, tour)
        
        return {
            'method': 'Quantum Folding',
            'complexity': 'O(n² log n × D × R × I)',
            'tour': tour,
            'length': length,
            'initial_length': initial_length,
            'improvement': (initial_length - length) / initial_length * 100,
            'entanglement': entanglement,
            'time': elapsed,
            'info': info
        }
    
    def solve_chaos_seeding(self, n_restarts: int = 5) -> dict:
        """
        Solve using Residual Chaos Seeding.
        
        Complexity: O(n² × R × K)
        where R = restarts, K = max iterations
        
        Args:
            n_restarts: Number of random restarts
            
        Returns:
            Dictionary with solution and statistics
        """
        start_time = time.time()
        tour, length, info = self.chaos_seeder.hybrid_chaos_construction(
            self.cities,
            n_restarts=n_restarts
        )
        elapsed = time.time() - start_time
        
        # Get baseline for comparison
        _, baseline_length = self.greedy_nearest_neighbor()
        
        return {
            'method': 'Chaos Seeding',
            'complexity': 'O(n² × R × K)',
            'tour': tour,
            'length': length,
            'baseline_length': baseline_length,
            'improvement': (baseline_length - length) / baseline_length * 100,
            'chaos': info['best_chaos'],
            'time': elapsed,
            'info': info
        }
    
    def solve_adaptive_chaos(self) -> dict:
        """
        Solve using Adaptive Chaos Seeding.
        
        Complexity: O(n² × K)
        where K = max iterations with adaptive weighting
        
        Returns:
            Dictionary with solution and statistics
        """
        # Get initial tour
        initial_tour, initial_length = self.greedy_nearest_neighbor()
        
        start_time = time.time()
        tour, length, info = self.adaptive_chaos.optimize_tour_adaptive_chaos(
            self.cities,
            initial_tour
        )
        elapsed = time.time() - start_time
        
        return {
            'method': 'Adaptive Chaos',
            'complexity': 'O(n² × K)',
            'tour': tour,
            'length': length,
            'initial_length': initial_length,
            'improvement': (initial_length - length) / initial_length * 100,
            'time': elapsed,
            'info': info
        }
    
    def solve_hybrid(
        self,
        n_restarts: int = 3,
        qf_iterations: int = 20
    ) -> dict:
        """
        Hybrid approach: Chaos Seeding → Quantum Folding.
        
        Uses chaos seeding for initial construction, then quantum folding
        for global optimization.
        
        Complexity: O(n² × R × K) + O(n² log n × D × I)
        Combined chaos seeding and quantum folding
        
        Args:
            n_restarts: Number of restarts for chaos seeding
            qf_iterations: Iterations for quantum folding
            
        Returns:
            Dictionary with solution and statistics
        """
        start_time = time.time()
        
        # Phase 1: Chaos-seeded construction
        chaos_tour, chaos_length, chaos_info = self.chaos_seeder.hybrid_chaos_construction(
            self.cities,
            n_restarts=n_restarts
        )
        
        # Phase 2: Quantum folding refinement
        qf_tour, qf_length, qf_info = self.quantum_folder.optimize_tour_dimensional_folding(
            self.cities,
            chaos_tour,
            n_restarts=1,
            iterations_per_restart=qf_iterations
        )
        
        elapsed = time.time() - start_time
        
        # Get baseline
        _, baseline_length = self.greedy_nearest_neighbor()
        
        # Compute final metrics
        entanglement = self.quantum_folder.compute_entanglement_score(self.cities, qf_tour)
        chaos = self.chaos_seeder.compute_chaos_magnitude(self.cities, qf_tour)
        
        return {
            'method': 'Hybrid (Chaos + Quantum)',
            'complexity': 'O(n² × R × K + n² log n × D × I)',
            'tour': qf_tour,
            'length': qf_length,
            'baseline_length': baseline_length,
            'chaos_length': chaos_length,
            'improvement': (baseline_length - qf_length) / baseline_length * 100,
            'entanglement': entanglement,
            'chaos': chaos,
            'time': elapsed,
            'info': {
                'chaos_info': chaos_info,
                'qf_info': qf_info
            }
        }
    
    def compare_all_methods(self) -> dict:
        """
        Compare all solution methods.
        
        Returns:
            Dictionary with results from all methods
        """
        print(f"Solving TSP with {self.n_cities} cities...\n")
        
        results = {}
        
        # Baseline
        print("Running baseline (greedy nearest neighbor)...")
        start_time = time.time()
        tour, length = self.greedy_nearest_neighbor()
        baseline_time = time.time() - start_time
        results['baseline'] = {
            'method': 'Greedy Nearest Neighbor',
            'complexity': 'O(n²)',
            'tour': tour,
            'length': length,
            'time': baseline_time
        }
        print(f"  Complexity: O(n²)")
        print(f"  Length: {length:.2f}")
        print(f"  Time: {baseline_time:.4f}s\n")
        
        # Quantum Folding
        print("Running Quantum Entanglement Dimensional Folding...")
        results['quantum'] = self.solve_quantum_folding(n_restarts=2, iterations_per_restart=20)
        print(f"  Complexity: {results['quantum']['complexity']}")
        print(f"  Length: {results['quantum']['length']:.2f}")
        print(f"  Improvement: {results['quantum']['improvement']:.2f}%")
        print(f"  Entanglement: {results['quantum']['entanglement']:.4f}")
        print(f"  Time: {results['quantum']['time']:.4f}s")
        print(f"  Speedup vs baseline: {baseline_time / results['quantum']['time']:.2f}×\n")
        
        # Chaos Seeding
        print("Running Residual Chaos Seeding...")
        results['chaos'] = self.solve_chaos_seeding(n_restarts=3)
        print(f"  Complexity: {results['chaos']['complexity']}")
        print(f"  Length: {results['chaos']['length']:.2f}")
        print(f"  Improvement: {results['chaos']['improvement']:.2f}%")
        print(f"  Chaos: {results['chaos']['chaos']:.4f}")
        print(f"  Time: {results['chaos']['time']:.4f}s")
        print(f"  Speedup vs baseline: {baseline_time / results['chaos']['time']:.2f}×\n")
        
        # Adaptive Chaos
        print("Running Adaptive Chaos Seeding...")
        results['adaptive'] = self.solve_adaptive_chaos()
        print(f"  Complexity: {results['adaptive']['complexity']}")
        print(f"  Length: {results['adaptive']['length']:.2f}")
        print(f"  Improvement: {results['adaptive']['improvement']:.2f}%")
        print(f"  Time: {results['adaptive']['time']:.4f}s")
        print(f"  Speedup vs baseline: {baseline_time / results['adaptive']['time']:.2f}×\n")
        
        # Hybrid
        print("Running Hybrid (Chaos + Quantum)...")
        results['hybrid'] = self.solve_hybrid(n_restarts=2, qf_iterations=15)
        print(f"  Complexity: {results['hybrid']['complexity']}")
        print(f"  Length: {results['hybrid']['length']:.2f}")
        print(f"  Improvement: {results['hybrid']['improvement']:.2f}%")
        print(f"  Entanglement: {results['hybrid']['entanglement']:.4f}")
        print(f"  Chaos: {results['hybrid']['chaos']:.4f}")
        print(f"  Time: {results['hybrid']['time']:.4f}s")
        print(f"  Speedup vs baseline: {baseline_time / results['hybrid']['time']:.2f}×\n")
        
        # Print timing comparison table
        self._print_timing_comparison(results)
        
        return results
    
    def visualize_solution(
        self,
        tour: List[int],
        title: str = "TSP Solution",
        save_path: Optional[str] = None
    ):
        """
        Visualize a tour solution.
        
        Args:
            tour: List of city indices
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot cities
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100, zorder=3)
        
        # Plot tour
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            ax.plot(
                [self.cities[tour[i], 0], self.cities[tour[j], 0]],
                [self.cities[tour[i], 1], self.cities[tour[j], 1]],
                'b-', linewidth=2, alpha=0.6
            )
        
        # Mark start city
        ax.scatter(
            self.cities[tour[0], 0],
            self.cities[tour[0], 1],
            c='green', s=200, marker='*', zorder=4, label='Start'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return fig
    
    def visualize_comparison(
        self,
        results: dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize comparison of all methods.
        
        Args:
            results: Results dictionary from compare_all_methods()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        methods = ['baseline', 'quantum', 'chaos', 'adaptive', 'hybrid']
        
        for idx, method in enumerate(methods):
            if method not in results:
                continue
            
            ax = axes[idx]
            result = results[method]
            tour = result['tour']
            
            # Plot cities
            ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50, zorder=3)
            
            # Plot tour
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                ax.plot(
                    [self.cities[tour[i], 0], self.cities[tour[j], 0]],
                    [self.cities[tour[i], 1], self.cities[tour[j], 1]],
                    'b-', linewidth=1.5, alpha=0.6
                )
            
            # Mark start
            ax.scatter(
                self.cities[tour[0], 0],
                self.cities[tour[0], 1],
                c='green', s=150, marker='*', zorder=4
            )
            
            title = f"{result['method']}\nLength: {result['length']:.2f}"
            if 'improvement' in result:
                title += f" ({result['improvement']:.1f}% improvement)"
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Summary plot
        ax = axes[5]
        method_names = [results[m]['method'] for m in methods if m in results]
        lengths = [results[m]['length'] for m in methods if m in results]
        times = [results[m]['time'] for m in methods if m in results]
        
        x = np.arange(len(method_names))
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - 0.2, lengths, 0.4, label='Tour Length', alpha=0.7)
        bars2 = ax2.bar(x + 0.2, times, 0.4, label='Time (s)', alpha=0.7, color='orange')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Tour Length', color='blue')
        ax2.set_ylabel('Time (s)', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
        ax.set_title('Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        return fig
    
    def _print_timing_comparison(self, results: dict):
        """Print detailed timing comparison table."""
        print("="*100)
        print("COMPLEXITY AND TIMING ANALYSIS")
        print("="*100)
        
        # Table header
        print(f"{'Method':<30} {'Complexity':<35} {'Time (s)':<12} {'Speedup':<10}")
        print("-"*100)
        
        baseline_time = results['baseline']['time']
        
        # Baseline
        print(f"{'Greedy Nearest Neighbor':<30} {'O(n²)':<35} {baseline_time:>10.4f}s   {'1.00×':<10}")
        
        # Other methods
        methods = [
            ('quantum', 'Quantum Folding'),
            ('chaos', 'Chaos Seeding'),
            ('adaptive', 'Adaptive Chaos'),
            ('hybrid', 'Hybrid (Chaos + Quantum)')
        ]
        
        for key, name in methods:
            if key in results:
                r = results[key]
                speedup = baseline_time / r['time'] if r['time'] > 0 else 0
                print(f"{name:<30} {r['complexity']:<35} {r['time']:>10.4f}s   {speedup:>5.2f}×")
        
        print("-"*100)
        
        # Find fastest and best quality
        method_keys = ['baseline', 'quantum', 'chaos', 'adaptive', 'hybrid']
        fastest = min([k for k in method_keys if k in results], 
                     key=lambda k: results[k]['time'])
        best_quality = min([k for k in method_keys if k in results], 
                          key=lambda k: results[k]['length'])
        
        print(f"\nFastest method: {results[fastest]['method']} ({results[fastest]['time']:.4f}s)")
        print(f"Best quality: {results[best_quality]['method']} (length: {results[best_quality]['length']:.2f})")
        
        # Time breakdown for hybrid
        if 'hybrid' in results and 'info' in results['hybrid']:
            print(f"\nHybrid method breakdown:")
            print(f"  Phase 1 (Chaos Seeding): Construction + optimization")
            print(f"  Phase 2 (Quantum Folding): Dimensional exploration + refinement")
            print(f"  Total time: {results['hybrid']['time']:.4f}s")
        
        print()
    
    def _tour_length(self, tour: List[int]) -> float:
        """Compute total tour length."""
        length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += np.linalg.norm(self.cities[tour[i]] - self.cities[tour[j]])
        return length


def main():
    parser = argparse.ArgumentParser(
        description='TSP Solver using Quantum Folding and Chaos Seeding'
    )
    parser.add_argument(
        '--n-cities', type=int, default=30,
        help='Number of cities (default: 30)'
    )
    parser.add_argument(
        '--instance-type', type=str, default='random',
        choices=['random', 'clustered', 'grid', 'circle'],
        help='Type of TSP instance (default: random)'
    )
    parser.add_argument(
        '--method', type=str, default='all',
        choices=['all', 'quantum', 'chaos', 'adaptive', 'hybrid'],
        help='Solution method (default: all)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save plots to files'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Generate TSP instance
    print(f"Generating {args.instance_type} TSP instance with {args.n_cities} cities...")
    if args.instance_type == 'random':
        cities = TSPInstance.random_euclidean(args.n_cities, args.seed)
    elif args.instance_type == 'clustered':
        cities = TSPInstance.clustered(args.n_cities, n_clusters=4, seed=args.seed)
    elif args.instance_type == 'grid':
        cities = TSPInstance.grid(args.n_cities)
    elif args.instance_type == 'circle':
        cities = TSPInstance.circle(args.n_cities)
    
    # Create solver
    solver = TSPSolver(cities)
    
    # Solve
    if args.method == 'all':
        results = solver.compare_all_methods()
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        best_method = min(
            [k for k in results.keys() if k != 'baseline'],
            key=lambda k: results[k]['length']
        )
        print(f"Best method: {results[best_method]['method']}")
        print(f"Best length: {results[best_method]['length']:.2f}")
        print(f"Improvement over baseline: {results[best_method]['improvement']:.2f}%")
        
        if args.visualize:
            solver.visualize_comparison(
                results,
                save_path='tsp_comparison.png' if args.save_plots else None
            )
            if not args.save_plots:
                plt.show()
    
    else:
        # Run single method
        if args.method == 'quantum':
            result = solver.solve_quantum_folding()
        elif args.method == 'chaos':
            result = solver.solve_chaos_seeding()
        elif args.method == 'adaptive':
            result = solver.solve_adaptive_chaos()
        elif args.method == 'hybrid':
            result = solver.solve_hybrid()
        
        print(f"\n{result['method']} Results:")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Tour length: {result['length']:.2f}")
        if 'improvement' in result:
            print(f"  Improvement: {result['improvement']:.2f}%")
        print(f"  Time: {result['time']:.4f}s")
        
        if args.visualize:
            solver.visualize_solution(
                result['tour'],
                title=f"{result['method']} - Length: {result['length']:.2f}",
                save_path=f"tsp_{args.method}.png" if args.save_plots else None
            )
            if not args.save_plots:
                plt.show()


if __name__ == '__main__':
    main()
