"""
Example: Using the Performance Profiler
========================================

Demonstrates the Performance Profiler module with various profiling patterns.
"""

import numpy as np
import time
from performance_profiler import (
    PerformanceProfiler,
    profile,
    compare_profiles,
    estimate_complexity,
    format_time,
    format_memory
)


# ============================================================================
# Example Functions to Profile
# ============================================================================

def slow_function(n):
    """Simulate a slow computation."""
    result = 0
    for i in range(n):
        result += np.sin(i) * np.cos(i)
    return result


def fast_function(n):
    """Optimized version using vectorization."""
    x = np.arange(n)
    return np.sum(np.sin(x) * np.cos(x))


def iterative_refinement(x, target=0.0):
    """Simulate one iteration of refinement."""
    return x - 0.1 * (x - target)


def batch_processor(data, batch_size=10):
    """Process data in batches."""
    result = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        result.append(np.mean(batch))
    return result


# ============================================================================
# Example 1: Basic Function Profiling
# ============================================================================

def example_1_basic_profiling():
    """Profile individual functions."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Function Profiling")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    
    # Profile slow function
    result1, profile1 = profiler.profile_function(slow_function, 1000, name="slow_function")
    print(f"Slow function: {format_time(profile1.execution_time)}")
    
    # Profile fast function
    result2, profile2 = profiler.profile_function(fast_function, 1000, name="fast_function")
    print(f"Fast function: {format_time(profile2.execution_time)}")
    
    # Compare
    comparison = compare_profiles(profile1, profile2)
    print(f"\nSpeedup: {comparison['speedup']:.2f}x")
    print(f"Time saved: {format_time(comparison['time_saved_seconds'])}")
    
    print()


# ============================================================================
# Example 2: Component Profiling
# ============================================================================

def example_2_component_profiling():
    """Profile multiple components of an algorithm."""
    print("=" * 70)
    print("EXAMPLE 2: Component Profiling")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    
    # Define components
    components = {
        "initialization": (lambda: np.random.randn(1000), ()),
        "computation": (slow_function, (500,)),
        "post_processing": (lambda x: np.sort(x), (np.random.randn(1000),)),
    }
    
    # Profile all components
    results = profiler.profile_components(components)
    
    # Generate report
    print(profiler.generate_report())
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks(threshold=0.2)
    print("\nBottleneck Analysis:")
    for rec in bottlenecks.recommendations:
        print(f"  - {rec}")
    
    print()


# ============================================================================
# Example 3: Iteration Profiling
# ============================================================================

def example_3_iteration_profiling():
    """Profile iterative algorithm."""
    print("=" * 70)
    print("EXAMPLE 3: Iteration Profiling")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    
    # Profile iterations
    iter_profile = profiler.profile_iterations(
        iterative_refinement,
        iterations=10,
        x=1.0,
        target=0.0
    )
    
    print(f"Total iterations: {iter_profile.total_iterations}")
    print(f"Total time: {format_time(iter_profile.total_time)}")
    print(f"Avg time per iteration: {format_time(iter_profile.avg_time_per_iteration)}")
    print(f"Std time per iteration: {format_time(iter_profile.std_time_per_iteration)}")
    
    if iter_profile.convergence_detected:
        print(f"Convergence detected at iteration {iter_profile.convergence_iteration}")
    else:
        print("No convergence detected")
    
    print()


# ============================================================================
# Example 4: Batch Profiling
# ============================================================================

def example_4_batch_profiling():
    """Profile batch processing at different sizes."""
    print("=" * 70)
    print("EXAMPLE 4: Batch Profiling")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    
    # Create test data
    data = np.random.randn(10000)
    
    # Profile different batch sizes
    batch_profile = profiler.profile_batch(
        batch_processor,
        batch_sizes=[10, 50, 100, 500, 1000],
        data=data
    )
    
    print(f"Batch sizes tested: {batch_profile.batch_sizes}")
    print(f"Scaling factor: {batch_profile.scaling_factor:.2f}")
    print(f"Estimated complexity: {estimate_complexity(batch_profile)}")
    print(f"Optimal batch size: {batch_profile.optimal_batch_size}")
    
    print("\nPer-batch timing:")
    for size, time_val in zip(batch_profile.batch_sizes, batch_profile.batch_times):
        print(f"  Batch size {size:4d}: {format_time(time_val)}")
    
    print()


# ============================================================================
# Example 5: Context Manager
# ============================================================================

def example_5_context_manager():
    """Profile code blocks using context manager."""
    print("=" * 70)
    print("EXAMPLE 5: Context Manager Profiling")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    
    with profiler.profile_block("data_generation"):
        data = np.random.randn(5000)
    
    with profiler.profile_block("computation"):
        result = np.fft.fft(data)
    
    with profiler.profile_block("post_processing"):
        magnitude = np.abs(result)
        sorted_mag = np.sort(magnitude)
    
    print(profiler.generate_report())
    print()


# ============================================================================
# Example 6: Decorator Usage
# ============================================================================

profiler_global = PerformanceProfiler()

@profile(name="decorated_function", profiler=profiler_global)
def decorated_computation(n):
    """Function decorated with profiler."""
    return np.sum(np.random.randn(n) ** 2)


def example_6_decorator():
    """Profile using decorator."""
    print("=" * 70)
    print("EXAMPLE 6: Decorator Profiling")
    print("=" * 70)
    
    # Call decorated function multiple times
    for i in range(3):
        result = decorated_computation(1000)
    
    print(profiler_global.generate_report())
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PERFORMANCE PROFILER EXAMPLES" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_1_basic_profiling()
    example_2_component_profiling()
    example_3_iteration_profiling()
    example_4_batch_profiling()
    example_5_context_manager()
    example_6_decorator()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
