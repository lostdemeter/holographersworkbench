"""
Test functions for new optimization tools to be merged into test_workbench.py

Tests for:
1. Quantum Folding (QuantumFolder)
2. Chaos Seeding (ChaosSeeder, AdaptiveChaosSeeder)
3. Adaptive Nonlocality (AdaptiveNonlocalityOptimizer)
4. Sublinear QIK (SublinearQIK)

Add these functions to test_workbench.py and include them in run_all_tests()
"""

import numpy as np
import time


def test_quantum_folding():
    """Test Quantum Entanglement Dimensional Folding."""
    print("\n" + "="*70)
    print("TEST: Quantum Folding")
    print("="*70)
    
    try:
        from workbench.primitives import QuantumFolder
        
        # Test instantiation
        folder = QuantumFolder()
        assert folder is not None, "QuantumFolder instantiation failed"
        assert len(folder.dimensions) == 6, "Should have 6 default dimensions"
        print(f"✓ QuantumFolder instantiated with {len(folder.dimensions)} dimensions")
        
        # Test dimensional folding
        np.random.seed(42)
        cities = np.random.rand(15, 2) * 100
        
        # Test collapse (D < 2)
        folded_collapse = folder.fold_dimension_collapse(cities, 1.5)
        assert folded_collapse.shape == cities.shape, "Collapsed shape mismatch"
        print(f"✓ Dimensional collapse (D=1.5): {folded_collapse.shape}")
        
        # Test expansion (D > 2)
        folded_expand = folder.fold_dimension_expand(cities, 3.0)
        assert folded_expand.shape == cities.shape, "Expanded shape mismatch"
        print(f"✓ Dimensional expansion (D=3.0): {folded_expand.shape}")
        
        # Test fast expansion
        folded_fast = folder.fold_dimension_expand_fast(cities, 3.0)
        assert folded_fast.shape == cities.shape, "Fast expanded shape mismatch"
        print(f"✓ Fast dimensional expansion (D=3.0): {folded_fast.shape}")
        
        # Test entanglement computation
        tour = list(range(len(cities)))
        np.random.shuffle(tour)
        
        entanglement = folder.compute_entanglement(cities, tour)
        assert entanglement.shape == (len(cities), len(cities)), "Entanglement shape mismatch"
        print(f"✓ Entanglement matrix: {entanglement.shape}")
        
        # Test vectorized entanglement
        entanglement_vec = folder.compute_entanglement_vectorized(cities, tour)
        assert entanglement_vec.shape == (len(cities), len(cities)), "Vectorized entanglement shape mismatch"
        print(f"✓ Vectorized entanglement: {entanglement_vec.shape}")
        
        # Test sparse entanglement
        sparse_ent = folder.compute_sparse_entanglement(cities, tour, k=5)
        assert len(sparse_ent) == len(cities), "Sparse entanglement size mismatch"
        print(f"✓ Sparse entanglement (k=5): {len(sparse_ent)} cities")
        
        # Test optimization (small instance)
        initial_tour = list(range(len(cities)))
        t0 = time.time()
        tour, length, info = folder.optimize_tour_dimensional_folding_fast(
            cities, initial_tour, n_restarts=1, iterations_per_restart=5
        )
        elapsed = time.time() - t0
        
        assert len(tour) == len(cities), "Tour length mismatch"
        assert length > 0, "Tour length should be positive"
        print(f"✓ Optimization: length={length:.2f}, time={elapsed:.3f}s")
        
        return True
    except Exception as e:
        print(f"✗ Quantum Folding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chaos_seeding():
    """Test Residual Chaos Seeding."""
    print("\n" + "="*70)
    print("TEST: Chaos Seeding")
    print("="*70)
    
    try:
        from workbench.primitives import ChaosSeeder, AdaptiveChaosSeeder
        
        # Test ChaosSeeder instantiation
        seeder = ChaosSeeder(window_size=3, chaos_weight=0.5)
        assert seeder is not None, "ChaosSeeder instantiation failed"
        print(f"✓ ChaosSeeder instantiated (window={seeder.window_size})")
        
        # Test AdaptiveChaosSeeder instantiation
        adaptive_seeder = AdaptiveChaosSeeder(
            initial_weight=0.8,
            final_weight=0.2,
            decay_rate=0.95
        )
        assert adaptive_seeder is not None, "AdaptiveChaosSeeder instantiation failed"
        print(f"✓ AdaptiveChaosSeeder instantiated")
        
        # Test on small TSP instance
        np.random.seed(42)
        cities = np.random.rand(15, 2) * 100
        tour = list(range(len(cities)))
        
        # Test projection
        projection = seeder.compute_projection(cities, tour)
        assert projection.shape == cities.shape, "Projection shape mismatch"
        print(f"✓ Projection computed: {projection.shape}")
        
        # Test residual
        residual = seeder.compute_residual(cities, tour)
        assert residual.shape == cities.shape, "Residual shape mismatch"
        print(f"✓ Residual computed: {residual.shape}")
        
        # Test chaos magnitude
        chaos = seeder.compute_chaos_magnitude(cities, tour)
        assert chaos >= 0, "Chaos magnitude should be non-negative"
        print(f"✓ Chaos magnitude: {chaos:.2f}")
        
        # Test greedy construction
        t0 = time.time()
        tour, chaos_val = seeder.greedy_construction_chaos_seeded(cities)
        elapsed = time.time() - t0
        
        assert len(tour) == len(cities), "Tour length mismatch"
        assert len(set(tour)) == len(cities), "Tour should visit all cities"
        print(f"✓ Greedy construction: chaos={chaos_val:.2f}, time={elapsed:.3f}s")
        
        # Test hybrid chaos construction
        t0 = time.time()
        tour, length, info = seeder.hybrid_chaos_construction(cities, n_restarts=2)
        elapsed = time.time() - t0
        
        assert len(tour) == len(cities), "Tour length mismatch"
        assert length > 0, "Tour length should be positive"
        print(f"✓ Hybrid construction: length={length:.2f}, time={elapsed:.3f}s")
        
        # Test adaptive chaos
        initial_tour = list(range(len(cities)))
        t0 = time.time()
        tour, length, info = adaptive_seeder.optimize_tour_adaptive_chaos(
            cities, initial_tour, max_iterations=10
        )
        elapsed = time.time() - t0
        
        assert len(tour) == len(cities), "Tour length mismatch"
        print(f"✓ Adaptive chaos: length={length:.2f}, time={elapsed:.3f}s")
        
        return True
    except Exception as e:
        print(f"✗ Chaos Seeding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_nonlocality():
    """Test Adaptive Nonlocality Optimizer."""
    print("\n" + "="*70)
    print("TEST: Adaptive Nonlocality")
    print("="*70)
    
    try:
        from workbench import AdaptiveNonlocalityOptimizer
        
        # Test instantiation
        anl = AdaptiveNonlocalityOptimizer(
            d_min=1.0,
            d_max=2.5,
            n_dim_samples=20,
            t_initial=2.0,
            t_final=0.5
        )
        assert anl is not None, "AdaptiveNonlocalityOptimizer instantiation failed"
        assert len(anl.dimensions) == 20, "Should have 20 dimensional samples"
        print(f"✓ AdaptiveNonlocalityOptimizer instantiated ({len(anl.dimensions)} dimensions)")
        
        # Test on small TSP instance
        np.random.seed(42)
        cities = np.random.rand(12, 2) * 100
        
        # Test problem affinity
        problem_affinity = anl.compute_problem_affinity(cities)
        assert len(problem_affinity) == 20, "Problem affinity length mismatch"
        assert np.all(problem_affinity >= 0), "Problem affinity should be non-negative"
        print(f"✓ Problem affinity computed: max={problem_affinity.max():.3f}")
        
        # Test solution affinity
        tour = list(range(len(cities)))
        solution_affinity = anl.compute_solution_affinity(tour, cities)
        assert len(solution_affinity) == 20, "Solution affinity length mismatch"
        assert np.all(solution_affinity >= 0), "Solution affinity should be non-negative"
        print(f"✓ Solution affinity computed: max={solution_affinity.max():.3f}")
        
        # Test coupling
        temperature = 1.0
        coupling = anl.compute_coupling(problem_affinity, solution_affinity, temperature)
        assert len(coupling) == 20, "Coupling length mismatch"
        print(f"✓ Coupling computed: max={coupling.max():.3f}")
        
        # Test dimension sampling
        dimension, idx = anl.sample_dimension(coupling, temperature)
        assert 1.0 <= dimension <= 2.5, "Sampled dimension out of range"
        print(f"✓ Dimension sampled: D={dimension:.3f}")
        
        # Test temperature schedule
        temp_0 = anl.compute_temperature(0, 100)
        temp_50 = anl.compute_temperature(50, 100)
        temp_100 = anl.compute_temperature(100, 100)
        assert temp_0 > temp_50 > temp_100, "Temperature should decrease"
        print(f"✓ Temperature schedule: {temp_0:.2f} → {temp_50:.2f} → {temp_100:.2f}")
        
        # Test optimization (very short run)
        def cost_fn(tour, cities):
            length = 0.0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                length += np.linalg.norm(cities[tour[i]] - cities[tour[j]])
            return length
        
        def local_search(solution, cities, dimension):
            # Simple 2-opt
            n = len(solution)
            i = np.random.randint(0, n-1)
            j = np.random.randint(i+2, n)
            new_solution = solution.copy()
            new_solution[i+1:j+1] = reversed(new_solution[i+1:j+1])
            return new_solution
        
        t0 = time.time()
        best_solution, best_cost, trajectory = anl.optimize(
            tour, cities, cost_fn, local_search, max_iterations=10, verbose=False
        )
        elapsed = time.time() - t0
        
        assert len(best_solution) == len(cities), "Solution length mismatch"
        assert len(trajectory.iterations) == 10, "Trajectory length mismatch"
        print(f"✓ Optimization: cost={best_cost:.2f}, time={elapsed:.3f}s")
        
        # Test trajectory analysis
        analysis = anl.analyze_trajectory(trajectory)
        assert 'phase_statistics' in analysis, "Missing phase statistics"
        assert 'final_dimension' in analysis, "Missing final dimension"
        print(f"✓ Trajectory analysis: final_D={analysis['final_dimension']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Adaptive Nonlocality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sublinear_qik():
    """Test Sublinear QIK."""
    print("\n" + "="*70)
    print("TEST: Sublinear QIK")
    print("="*70)
    
    try:
        from workbench import SublinearQIK, zetazero_batch
        
        # Test instantiation
        qik = SublinearQIK(
            use_hierarchical=True,
            use_dimensional_sketch=True,
            use_sparse_resonance=True
        )
        assert qik is not None, "SublinearQIK instantiation failed"
        print(f"✓ SublinearQIK instantiated")
        
        # Test on small TSP instance
        np.random.seed(42)
        cities = np.random.rand(20, 2) * 100
        
        # Get zeta zeros
        print("  Computing zeta zeros...")
        zeta_zeros = zetazero_batch(list(range(1, 11)))
        assert len(zeta_zeros) == 10, "Should get 10 zeta zeros"
        print(f"✓ Zeta zeros computed: {len(zeta_zeros)}")
        
        # Test optimization
        t0 = time.time()
        tour, length, stats = qik.optimize_tsp(cities, zeta_zeros, verbose=False)
        elapsed = time.time() - t0
        
        assert len(tour) == len(cities), "Tour length mismatch"
        assert len(set(tour)) == len(cities), "Tour should visit all cities"
        assert length > 0, "Tour length should be positive"
        print(f"✓ Optimization: length={length:.2f}, time={elapsed:.3f}s")
        
        # Test statistics
        assert stats.n_cities == len(cities), "Stats n_cities mismatch"
        assert stats.n_clusters > 0, "Should have clusters"
        assert stats.total_time > 0, "Total time should be positive"
        print(f"✓ Statistics: {stats.n_clusters} clusters, {stats.n_dim_samples} dim samples")
        
        # Test complexity
        assert "N^1.5" in stats.theoretical_complexity, "Should mention N^1.5 complexity"
        print(f"✓ Complexity: {stats.theoretical_complexity}")
        print(f"✓ Empirical: {stats.empirical_complexity}")
        
        # Test without hierarchical (should still work)
        qik_simple = SublinearQIK(use_hierarchical=False)
        tour2, length2, stats2 = qik_simple.optimize_tsp(cities[:10], zeta_zeros, verbose=False)
        assert len(tour2) == 10, "Simple tour length mismatch"
        print(f"✓ Non-hierarchical mode: length={length2:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Sublinear QIK test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Instructions for merging:
"""
To merge these tests into test_workbench.py:

1. Copy the four test functions above into test_workbench.py

2. In the run_all_tests() function, add these tests to the test_functions list:

    test_functions = [
        ("Imports", test_imports),
        ("Spectral", test_spectral),
        ("Holographic", test_holographic),
        ("Optimization", test_optimization),
        ("Compression", test_compression),
        ("Encoding", test_encoding),
        ("Ergodic", test_ergodic),
        ("Analysis", test_analysis),
        ("Generation", test_generation),
        ("Quantum Folding", test_quantum_folding),           # NEW
        ("Chaos Seeding", test_chaos_seeding),               # NEW
        ("Adaptive Nonlocality", test_adaptive_nonlocality), # NEW
        ("Sublinear QIK", test_sublinear_qik),              # NEW
    ]

3. Run the tests:
   python tests/test_workbench.py
"""
