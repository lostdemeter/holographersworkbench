#!/usr/bin/env python3
"""
Clock Resonance Compiler Demo
=============================

Demonstrates the Clock Resonance Compiler by:
1. Analyzing real workbench processors
2. Compiling one to use clock phases
3. Benchmarking the improvement

This is the first step towards a fully clock-resonant workbench.
"""

import numpy as np
import time
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from workbench.core.clock_compiler import (
    ClockResonanceCompiler,
    ClockOracleMixin,
    make_clock_resonant
)


def analyze_all_processors():
    """Analyze all major processors in the workbench."""
    print("=" * 70)
    print("ANALYZING WORKBENCH PROCESSORS")
    print("=" * 70)
    
    compiler = ClockResonanceCompiler(verbose=True)
    
    # Import processors to analyze
    processors = []
    
    try:
        from workbench.processors.spectral import SpectralScorer
        processors.append(('SpectralScorer', SpectralScorer))
    except ImportError as e:
        print(f"Could not import SpectralScorer: {e}")
    
    try:
        from workbench.processors.adaptive_nonlocality import AdaptiveNonlocalityOptimizer
        processors.append(('AdaptiveNonlocalityOptimizer', AdaptiveNonlocalityOptimizer))
    except ImportError as e:
        print(f"Could not import AdaptiveNonlocalityOptimizer: {e}")
    
    try:
        from workbench.primitives.quantum_folding import QuantumFolder
        processors.append(('QuantumFolder', QuantumFolder))
    except ImportError as e:
        print(f"Could not import QuantumFolder: {e}")
    
    try:
        from workbench.primitives.chaos_seeding import ChaosSeeder
        processors.append(('ChaosSeeder', ChaosSeeder))
    except ImportError as e:
        print(f"Could not import ChaosSeeder: {e}")
    
    # Analyze each
    analyses = []
    for name, cls in processors:
        print(f"\n{'-'*50}")
        print(f"Analyzing: {name}")
        print(f"{'-'*50}")
        
        analysis = compiler.analyze(cls)
        analyses.append((name, cls, analysis))
        print(analysis)
    
    return analyses


def demo_compile_quantum_folder():
    """
    Demo: Compile QuantumFolder to use clock phases.
    
    QuantumFolder uses dimensional projections - a natural fit for clock phases.
    """
    print("\n" + "=" * 70)
    print("DEMO: Compiling QuantumFolder")
    print("=" * 70)
    
    try:
        from workbench.primitives.quantum_folding import QuantumFolder
    except ImportError:
        print("QuantumFolder not available")
        return
    
    compiler = ClockResonanceCompiler()
    
    # Analyze
    analysis = compiler.analyze(QuantumFolder)
    print(f"\nAnalysis:\n{analysis}")
    
    # Compile
    ClockQuantumFolder = compiler.compile(QuantumFolder)
    print(f"\nCompiled: {ClockQuantumFolder.__name__}")
    
    # Validate
    valid = compiler.validate(QuantumFolder, ClockQuantumFolder)
    print(f"Validation: {'PASSED' if valid else 'FAILED'}")
    
    # Show that compiled version has clock methods
    compiled = ClockQuantumFolder()
    print(f"\nCompiled version has clock methods:")
    print(f"  get_clock_phase(1) = {compiled.get_clock_phase(1):.6f}")
    print(f"  get_clock_phase(2) = {compiled.get_clock_phase(2):.6f}")
    print(f"  clock_random(3) = {compiled.clock_random(3)}")
    
    print("\nNote: Full integration requires updating QuantumFolder internals")
    print("      to call self.clock_random() instead of np.random.rand()")


def demo_create_clock_processor():
    """
    Demo: Create a new processor using ClockOracleMixin.
    
    Shows how to build clock-native processors from scratch.
    """
    print("\n" + "=" * 70)
    print("DEMO: Creating a Clock-Native Processor")
    print("=" * 70)
    
    class ClockGreedyTSP(ClockOracleMixin):
        """
        A TSP solver that uses clock phases for construction.
        
        This is a clock-native processor - built from the ground up
        to use clock eigenphases instead of random numbers.
        """
        
        def __init__(self):
            super().__init__()
            self.name = "ClockGreedyTSP"
            
        def solve(self, cities: np.ndarray) -> tuple:
            """
            Solve TSP using clock-guided greedy construction.
            
            Uses clock phases to:
            1. Select starting city
            2. Guide angular preferences during construction
            """
            n = len(cities)
            self.reset_clock_counter()
            
            # Select starting city using clock phase
            start_phase = self.get_clock_phase()
            start = int(start_phase * n) % n
            
            # Greedy construction with clock-guided angular bias
            unvisited = set(range(n))
            tour = [start]
            unvisited.remove(start)
            current = start
            
            while unvisited:
                # Get clock phase for angular preference
                target_angle = 2 * np.pi * self.get_clock_phase()
                
                best_city = None
                best_score = float('inf')
                
                for city in unvisited:
                    dist = np.linalg.norm(cities[current] - cities[city])
                    
                    # Angular bias from clock phase
                    angle = np.arctan2(
                        cities[city, 1] - cities[current, 1],
                        cities[city, 0] - cities[current, 0]
                    )
                    angle_diff = abs(angle - target_angle)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    
                    bias = np.exp(-angle_diff / 0.5)
                    score = dist / (1 + bias)
                    
                    if score < best_score:
                        best_score = score
                        best_city = city
                
                tour.append(best_city)
                unvisited.remove(best_city)
                current = best_city
            
            # Compute tour length
            length = sum(
                np.linalg.norm(cities[tour[i]] - cities[tour[(i+1) % n]])
                for i in range(n)
            )
            
            return np.array(tour), length
    
    # Test the clock-native processor
    print("\nTesting ClockGreedyTSP...")
    
    solver = ClockGreedyTSP()
    
    np.random.seed(42)
    cities = np.random.rand(50, 2)
    
    tour, length = solver.solve(cities)
    
    print(f"  Tour length: {length:.4f}")
    print(f"  Clock phases used: {solver._clock_counter}")
    
    # Compare with random greedy
    class RandomGreedyTSP:
        def solve(self, cities):
            n = len(cities)
            start = np.random.randint(n)
            
            unvisited = set(range(n))
            tour = [start]
            unvisited.remove(start)
            current = start
            
            while unvisited:
                target_angle = 2 * np.pi * np.random.random()
                
                best_city = None
                best_score = float('inf')
                
                for city in unvisited:
                    dist = np.linalg.norm(cities[current] - cities[city])
                    angle = np.arctan2(
                        cities[city, 1] - cities[current, 1],
                        cities[city, 0] - cities[current, 0]
                    )
                    angle_diff = abs(angle - target_angle)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    bias = np.exp(-angle_diff / 0.5)
                    score = dist / (1 + bias)
                    
                    if score < best_score:
                        best_score = score
                        best_city = city
                
                tour.append(best_city)
                unvisited.remove(best_city)
                current = best_city
            
            length = sum(
                np.linalg.norm(cities[tour[i]] - cities[tour[(i+1) % n]])
                for i in range(n)
            )
            return np.array(tour), length
    
    # Run multiple trials
    print("\nComparing Clock vs Random (10 trials each):")
    
    clock_lengths = []
    random_lengths = []
    
    for trial in range(10):
        np.random.seed(trial)
        cities = np.random.rand(50, 2)
        
        clock_solver = ClockGreedyTSP()
        _, clock_len = clock_solver.solve(cities)
        clock_lengths.append(clock_len)
        
        random_solver = RandomGreedyTSP()
        _, random_len = random_solver.solve(cities)
        random_lengths.append(random_len)
    
    print(f"  Clock mean:  {np.mean(clock_lengths):.4f} ± {np.std(clock_lengths):.4f}")
    print(f"  Random mean: {np.mean(random_lengths):.4f} ± {np.std(random_lengths):.4f}")
    print(f"  Improvement: {100 * (np.mean(random_lengths) - np.mean(clock_lengths)) / np.mean(random_lengths):.2f}%")
    
    # Key advantage: reproducibility
    print("\nKey advantage - Reproducibility:")
    solver1 = ClockGreedyTSP()
    solver2 = ClockGreedyTSP()
    
    np.random.seed(42)
    cities = np.random.rand(30, 2)
    
    _, len1 = solver1.solve(cities)
    _, len2 = solver2.solve(cities)
    
    print(f"  Run 1: {len1:.6f}")
    print(f"  Run 2: {len2:.6f}")
    print(f"  Identical: {len1 == len2}")


def main():
    print("=" * 70)
    print("CLOCK RESONANCE COMPILER DEMO")
    print("=" * 70)
    
    # Analyze all processors
    analyses = analyze_all_processors()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Processors Ready for Clock Upgrade")
    print("=" * 70)
    
    print(f"\n{'Processor':<35} {'Difficulty':<12} {'Random Sources':>15}")
    print("-" * 65)
    
    for name, cls, analysis in analyses:
        print(f"{name:<35} {analysis.estimated_difficulty:<12} {len(analysis.random_sources):>15}")
    
    # Demo: Compile QuantumFolder
    demo_compile_quantum_folder()
    
    # Demo: Create clock-native processor
    demo_create_clock_processor()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
    1. Upgrade easy processors first:
       - SpectralScorer (phase initialization)
       - QuantumAutoencoder (latent space)
       - AdaptiveNonlocalityOptimizer (dimension sampling)
    
    2. Create clock-native versions of key algorithms:
       - ClockSpectralScorer
       - ClockQuantumFolder
       - ClockChaosSeeder
    
    3. Add to workbench exports:
       from workbench import ClockSpectralScorer
       
    4. Document the upgrade path for users
    """)


if __name__ == "__main__":
    main()
