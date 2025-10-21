"""
Performance Profiler for Holographic Algorithms
================================================

Provides tools for profiling execution time, memory usage, and identifying
bottlenecks in holographic signal processing algorithms.

This module is part of the Holographer's Workbench toolkit.
"""

import time
import tracemalloc
import functools
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple, Any
import numpy as np


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class ProfileResult:
    """Single component profile result.
    
    Attributes
    ----------
    component_name : str
        Name of the profiled component.
    execution_time : float
        Total execution time in seconds.
    memory_delta : float
        Memory usage change in MB.
    call_count : int
        Number of times the component was called.
    time_per_call : float
        Average time per call in seconds.
    memory_per_call : float
        Average memory per call in MB.
    timestamp : float
        Unix timestamp when profiled.
    """
    component_name: str
    execution_time: float  # seconds
    memory_delta: float    # MB
    call_count: int
    time_per_call: float   # seconds
    memory_per_call: float # MB
    timestamp: float       # when profiled
    
    def relative_time(self, total_time: float) -> float:
        """Compute fraction of total time.
        
        Parameters
        ----------
        total_time : float
            Total time across all components.
            
        Returns
        -------
        float
            Fraction of total time (0.0 to 1.0).
        """
        return self.execution_time / total_time if total_time > 0 else 0.0


@dataclass
class IterationProfile:
    """Profile of iterative algorithm.
    
    Attributes
    ----------
    component_name : str
        Name of the profiled component.
    total_iterations : int
        Total number of iterations profiled.
    iteration_times : List[float]
        Time for each iteration in seconds.
    total_time : float
        Total time across all iterations.
    avg_time_per_iteration : float
        Average time per iteration.
    std_time_per_iteration : float
        Standard deviation of iteration times.
    convergence_detected : bool
        Whether convergence was detected.
    convergence_iteration : int, optional
        Iteration number where convergence occurred.
    """
    component_name: str
    total_iterations: int
    iteration_times: List[float]
    total_time: float
    avg_time_per_iteration: float
    std_time_per_iteration: float
    convergence_detected: bool
    convergence_iteration: Optional[int] = None


@dataclass
class BatchProfile:
    """Profile of batch processing.
    
    Attributes
    ----------
    component_name : str
        Name of the profiled component.
    batch_sizes : List[int]
        List of batch sizes tested.
    batch_times : List[float]
        Time for each batch size.
    times_per_item : List[float]
        Time per item for each batch size.
    scaling_factor : float
        Estimated complexity exponent (O(n^scaling_factor)).
    optimal_batch_size : int, optional
        Batch size with best time per item.
    """
    component_name: str
    batch_sizes: List[int]
    batch_times: List[float]
    times_per_item: List[float]
    scaling_factor: float
    optimal_batch_size: Optional[int] = None


@dataclass
class BottleneckReport:
    """Bottleneck analysis report.
    
    Attributes
    ----------
    bottlenecks : List[ProfileResult]
        List of identified bottleneck components.
    total_time : float
        Total execution time across all components.
    bottleneck_fraction : float
        Fraction of time spent in bottlenecks.
    recommendations : List[str]
        List of optimization recommendations.
    """
    bottlenecks: List[ProfileResult]
    total_time: float
    bottleneck_fraction: float
    recommendations: List[str]


# ============================================================================
# Main Profiler Class
# ============================================================================

class PerformanceProfiler:
    """General-purpose performance profiler for holographic algorithms.
    
    Tracks execution time, memory usage, and identifies bottlenecks.
    
    Parameters
    ----------
    track_memory : bool, optional
        Whether to track memory usage (adds overhead). Default: True.
    warmup_runs : int, optional
        Number of warmup runs before profiling. Default: 1.
    
    Attributes
    ----------
    results : List[ProfileResult]
        Accumulated profile results.
    
    Examples
    --------
    >>> profiler = PerformanceProfiler()
    >>> result, profile = profiler.profile_function(my_func, arg1, arg2, name="my_component")
    >>> print(f"Time: {profile.execution_time:.6f}s")
    """
    
    def __init__(self, track_memory: bool = True, warmup_runs: int = 1):
        self.track_memory = track_memory
        self.warmup_runs = max(0, warmup_runs)
        self.results: List[ProfileResult] = []
        self._memory_tracking_active = False
    
    def profile_function(self, 
                        func: Callable, 
                        *args, 
                        name: Optional[str] = None,
                        **kwargs) -> Tuple[Any, ProfileResult]:
        """Profile a single function execution.
        
        Parameters
        ----------
        func : Callable
            Function to profile.
        *args
            Positional arguments to func.
        name : str, optional
            Optional name for the component. Defaults to func.__name__.
        **kwargs
            Keyword arguments to func.
            
        Returns
        -------
        result : Any
            Return value from func.
        profile : ProfileResult
            Profile result with timing and memory info.
        """
        component_name = name or getattr(func, '__name__', 'unknown')
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                _ = func(*args, **kwargs)
            except Exception:
                break
        
        # Start memory tracking
        memory_start = 0.0
        if self.track_memory and not self._memory_tracking_active:
            try:
                tracemalloc.start()
                self._memory_tracking_active = True
                memory_start, _ = tracemalloc.get_traced_memory()
            except Exception:
                self.track_memory = False
        elif self.track_memory and self._memory_tracking_active:
            try:
                memory_start, _ = tracemalloc.get_traced_memory()
            except Exception:
                memory_start = 0.0
        
        # Profile execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        
        # Get memory delta
        memory_delta = 0.0
        if self.track_memory and self._memory_tracking_active:
            try:
                memory_end, _ = tracemalloc.get_traced_memory()
                memory_delta = (memory_end - memory_start) / (1024 * 1024)
            except Exception:
                memory_delta = 0.0
        
        # Create profile result
        profile = ProfileResult(
            component_name=component_name,
            execution_time=elapsed_time,
            memory_delta=memory_delta,
            call_count=1,
            time_per_call=elapsed_time,
            memory_per_call=memory_delta,
            timestamp=time.time()
        )
        
        self.results.append(profile)
        return result, profile
    
    def profile_components(self, 
                          components: Dict[str, Tuple],
                          shared_args: Optional[Dict] = None) -> List[ProfileResult]:
        """Profile multiple components of an algorithm.
        
        Parameters
        ----------
        components : Dict[str, Tuple]
            Dict of {name: (func, args, kwargs)} or {name: (func, args)}.
        shared_args : Dict, optional
            Optional dict of args shared across components.
            
        Returns
        -------
        List[ProfileResult]
            List of profile results for each component.
        """
        shared_args = shared_args or {}
        results = []
        
        for name, component_spec in components.items():
            if len(component_spec) == 2:
                func, args = component_spec
                kwargs = {}
            elif len(component_spec) == 3:
                func, args, kwargs = component_spec
            else:
                warnings.warn(f"Invalid component spec for '{name}', skipping")
                continue
            
            # Merge shared args
            merged_kwargs = {**shared_args, **kwargs}
            
            # Profile the component
            _, profile = self.profile_function(func, *args, name=name, **merged_kwargs)
            results.append(profile)
        
        return results
    
    def profile_iterations(self,
                          func: Callable,
                          iterations: int,
                          *args,
                          convergence_threshold: float = 1e-6,
                          **kwargs) -> IterationProfile:
        """Profile an iterative algorithm, tracking each iteration.
        
        Parameters
        ----------
        func : Callable
            Function that performs one iteration.
        iterations : int
            Number of iterations to profile.
        *args, **kwargs
            Arguments to func.
        convergence_threshold : float
            Threshold for detecting convergence (relative change in time).
            
        Returns
        -------
        IterationProfile
            Profile with per-iteration breakdown.
        """
        component_name = getattr(func, '__name__', 'iterations')
        iteration_times = []
        
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                _ = func(*args, **kwargs)
            except Exception:
                break
        
        # Profile each iteration
        for i in range(iterations):
            start_time = time.perf_counter()
            _ = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            iteration_times.append(elapsed)
        
        # Analyze convergence
        convergence_detected = False
        convergence_iteration = None
        
        if len(iteration_times) > 3:
            for i in range(2, len(iteration_times)):
                recent_avg = np.mean(iteration_times[i-2:i+1])
                if recent_avg > 0:
                    rel_change = abs(iteration_times[i] - recent_avg) / recent_avg
                    if rel_change < convergence_threshold:
                        convergence_detected = True
                        convergence_iteration = i
                        break
        
        total_time = sum(iteration_times)
        avg_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)
        
        return IterationProfile(
            component_name=component_name,
            total_iterations=iterations,
            iteration_times=iteration_times,
            total_time=total_time,
            avg_time_per_iteration=avg_time,
            std_time_per_iteration=std_time,
            convergence_detected=convergence_detected,
            convergence_iteration=convergence_iteration
        )
    
    def profile_batch(self,
                     func: Callable,
                     batch_sizes: List[int],
                     *args,
                     **kwargs) -> BatchProfile:
        """Profile batch processing at different sizes.
        
        Parameters
        ----------
        func : Callable
            Function to profile (should accept batch_size parameter).
        batch_sizes : List[int]
            List of batch sizes to test.
        *args, **kwargs
            Arguments to func.
            
        Returns
        -------
        BatchProfile
            Profile with scaling analysis.
        """
        component_name = getattr(func, '__name__', 'batch')
        batch_times = []
        times_per_item = []
        
        for batch_size in batch_sizes:
            # Warmup
            for _ in range(self.warmup_runs):
                try:
                    _ = func(*args, batch_size=batch_size, **kwargs)
                except Exception:
                    break
            
            # Profile
            start_time = time.perf_counter()
            _ = func(*args, batch_size=batch_size, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            batch_times.append(elapsed)
            times_per_item.append(elapsed / batch_size if batch_size > 0 else 0.0)
        
        # Estimate scaling factor using log-log fit
        scaling_factor = 1.0
        if len(batch_sizes) >= 2:
            try:
                log_sizes = np.log(batch_sizes)
                log_times = np.log(batch_times)
                coeffs = np.polyfit(log_sizes, log_times, 1)
                scaling_factor = coeffs[0]
            except Exception:
                scaling_factor = 1.0
        
        # Find optimal batch size
        optimal_batch_size = None
        if times_per_item:
            min_idx = np.argmin(times_per_item)
            optimal_batch_size = batch_sizes[min_idx]
        
        return BatchProfile(
            component_name=component_name,
            batch_sizes=batch_sizes,
            batch_times=batch_times,
            times_per_item=times_per_item,
            scaling_factor=scaling_factor,
            optimal_batch_size=optimal_batch_size
        )
    
    def identify_bottlenecks(self, threshold: float = 0.1) -> BottleneckReport:
        """Identify bottlenecks from profiled results.
        
        Parameters
        ----------
        threshold : float
            Minimum fraction of total time to be considered bottleneck.
            
        Returns
        -------
        BottleneckReport
            Bottleneck analysis with recommendations.
        """
        if not self.results:
            return BottleneckReport(
                bottlenecks=[],
                total_time=0.0,
                bottleneck_fraction=0.0,
                recommendations=["No profiling data available"]
            )
        
        total_time = sum(r.execution_time for r in self.results)
        
        bottlenecks = [
            r for r in self.results 
            if r.relative_time(total_time) > threshold
        ]
        bottlenecks.sort(key=lambda r: r.execution_time, reverse=True)
        
        recommendations = []
        for b in bottlenecks:
            pct = b.relative_time(total_time) * 100
            recommendations.append(
                f"{b.component_name}: {pct:.1f}% of total time - consider optimization"
            )
        
        if not bottlenecks:
            recommendations.append("No significant bottlenecks detected")
        
        bottleneck_time = sum(b.execution_time for b in bottlenecks)
        bottleneck_fraction = bottleneck_time / total_time if total_time > 0 else 0.0
        
        return BottleneckReport(
            bottlenecks=bottlenecks,
            total_time=total_time,
            bottleneck_fraction=bottleneck_fraction,
            recommendations=recommendations
        )
    
    def generate_report(self, format: str = "text") -> Any:
        """Generate a profiling report.
        
        Parameters
        ----------
        format : str
            "text", "dict", or "dataframe".
            
        Returns
        -------
        str or dict or DataFrame
            Formatted report.
        """
        if not self.results:
            return "No profiling data available"
        
        total_time = sum(r.execution_time for r in self.results)
        
        if format == "text":
            lines = []
            lines.append("=" * 70)
            lines.append("PERFORMANCE PROFILING REPORT")
            lines.append("=" * 70)
            lines.append(f"Total execution time: {total_time:.6f}s")
            lines.append(f"Number of components: {len(self.results)}")
            lines.append("")
            lines.append(f"{'Component':<30} {'Time (s)':<12} {'%':<8} {'Memory (MB)':<12}")
            lines.append("-" * 70)
            
            for r in sorted(self.results, key=lambda x: x.execution_time, reverse=True):
                pct = r.relative_time(total_time) * 100
                lines.append(
                    f"{r.component_name:<30} {r.execution_time:<12.6f} "
                    f"{pct:<8.2f} {r.memory_delta:<12.3f}"
                )
            
            lines.append("=" * 70)
            return "\n".join(lines)
        
        elif format == "dict":
            return {
                "total_time": total_time,
                "num_components": len(self.results),
                "components": [
                    {
                        "name": r.component_name,
                        "time": r.execution_time,
                        "percentage": r.relative_time(total_time) * 100,
                        "memory_mb": r.memory_delta,
                        "calls": r.call_count
                    }
                    for r in self.results
                ]
            }
        
        elif format == "dataframe":
            try:
                import pandas as pd
                data = [
                    {
                        "Component": r.component_name,
                        "Time (s)": r.execution_time,
                        "Percentage": r.relative_time(total_time) * 100,
                        "Memory (MB)": r.memory_delta,
                        "Calls": r.call_count
                    }
                    for r in self.results
                ]
                return pd.DataFrame(data)
            except ImportError:
                return "pandas not available, use format='text' or 'dict'"
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def profile_block(self, name: str) -> 'ProfileContext':
        """Create a context manager for profiling code blocks.
        
        Parameters
        ----------
        name : str
            Name for the profiled block.
            
        Returns
        -------
        ProfileContext
            Context manager.
            
        Examples
        --------
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile_block("initialization"):
        ...     setup_data()
        """
        return ProfileContext(self, name)
    
    def clear(self):
        """Clear all profiling results."""
        self.results.clear()
        if self._memory_tracking_active:
            try:
                tracemalloc.stop()
                self._memory_tracking_active = False
            except Exception:
                pass
    
    def __del__(self):
        """Cleanup memory tracking on deletion."""
        if self._memory_tracking_active:
            try:
                tracemalloc.stop()
            except Exception:
                pass


# ============================================================================
# Decorator and Context Manager
# ============================================================================

def profile(name: Optional[str] = None, 
           profiler: Optional[PerformanceProfiler] = None):
    """Decorator to automatically profile a function.
    
    Parameters
    ----------
    name : str, optional
        Name for the profiled component.
    profiler : PerformanceProfiler, optional
        Profiler instance to use. If None, creates a new one.
    
    Examples
    --------
    >>> @profile(name="my_component")
    ... def my_function(x, y):
    ...     return x + y
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal profiler
            if profiler is None:
                profiler = PerformanceProfiler()
            result, _ = profiler.profile_function(
                func, *args, name=name or func.__name__, **kwargs
            )
            return result
        wrapper._profiler = profiler
        return wrapper
    return decorator


class ProfileContext:
    """Context manager for profiling code blocks.
    
    Parameters
    ----------
    profiler : PerformanceProfiler
        Profiler instance.
    name : str
        Name for the profiled block.
    
    Examples
    --------
    >>> profiler = PerformanceProfiler()
    >>> with profiler.profile_block("initialization"):
    ...     setup_data()
    """
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        # Start memory tracking
        if self.profiler.track_memory and not self.profiler._memory_tracking_active:
            try:
                tracemalloc.start()
                self.profiler._memory_tracking_active = True
                self.start_memory, _ = tracemalloc.get_traced_memory()
            except Exception:
                self.start_memory = 0.0
        elif self.profiler.track_memory and self.profiler._memory_tracking_active:
            try:
                self.start_memory, _ = tracemalloc.get_traced_memory()
            except Exception:
                self.start_memory = 0.0
        else:
            self.start_memory = 0.0
        
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        
        # Get memory delta
        memory_delta = 0.0
        if self.profiler.track_memory and self.profiler._memory_tracking_active:
            try:
                end_memory, _ = tracemalloc.get_traced_memory()
                memory_delta = (end_memory - self.start_memory) / (1024 * 1024)
            except Exception:
                memory_delta = 0.0
        
        # Create profile result
        profile_result = ProfileResult(
            component_name=self.name,
            execution_time=elapsed_time,
            memory_delta=memory_delta,
            call_count=1,
            time_per_call=elapsed_time,
            memory_per_call=memory_delta,
            timestamp=time.time()
        )
        
        self.profiler.results.append(profile_result)
        return False


# ============================================================================
# Utility Functions
# ============================================================================

def compare_profiles(profile1: ProfileResult, 
                    profile2: ProfileResult) -> Dict[str, float]:
    """Compare two profile results (e.g., before/after optimization).
    
    Parameters
    ----------
    profile1 : ProfileResult
        First profile (e.g., before optimization).
    profile2 : ProfileResult
        Second profile (e.g., after optimization).
        
    Returns
    -------
    dict
        Comparison metrics including speedup and memory improvement.
        
    Examples
    --------
    >>> _, before = profiler.profile_function(slow_func, args)
    >>> # ... optimize function ...
    >>> _, after = profiler.profile_function(fast_func, args)
    >>> comparison = compare_profiles(before, after)
    >>> print(f"Speedup: {comparison['speedup']:.2f}x")
    """
    speedup = profile1.execution_time / profile2.execution_time if profile2.execution_time > 0 else float('inf')
    time_saved = profile1.execution_time - profile2.execution_time
    memory_saved = profile1.memory_delta - profile2.memory_delta
    
    return {
        "speedup": speedup,
        "time_saved_seconds": time_saved,
        "time_improvement_percent": (time_saved / profile1.execution_time * 100) if profile1.execution_time > 0 else 0.0,
        "memory_saved_mb": memory_saved,
        "memory_improvement_percent": (memory_saved / profile1.memory_delta * 100) if profile1.memory_delta > 0 else 0.0,
    }


def estimate_complexity(batch_profile: BatchProfile) -> str:
    """Estimate algorithmic complexity from batch profile.
    
    Parameters
    ----------
    batch_profile : BatchProfile
        Batch profiling results.
        
    Returns
    -------
    str
        Estimated complexity (e.g., "O(n)", "O(n log n)", "O(n^2)").
        
    Examples
    --------
    >>> batch_profile = profiler.profile_batch(my_func, [10, 100, 1000])
    >>> complexity = estimate_complexity(batch_profile)
    >>> print(f"Estimated complexity: {complexity}")
    """
    factor = batch_profile.scaling_factor
    
    if factor < 0.5:
        return "O(1)" 
    elif factor < 0.8:
        return "O(log n)"
    elif factor < 1.2:
        return "O(n)"
    elif factor < 1.5:
        return "O(n log n)"
    elif factor < 2.2:
        return "O(n^2)"
    elif factor < 3.2:
        return "O(n^3)"
    else:
        return f"O(n^{factor:.1f})"


def format_time(seconds: float) -> str:
    """Format time in human-readable format.
    
    Parameters
    ----------
    seconds : float
        Time in seconds.
        
    Returns
    -------
    str
        Formatted time string.
        
    Examples
    --------
    >>> format_time(0.000123)
    '123.0 μs'
    >>> format_time(1.5)
    '1.50 s'
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_memory(megabytes: float) -> str:
    """Format memory in human-readable format.
    
    Parameters
    ----------
    megabytes : float
        Memory in megabytes.
        
    Returns
    -------
    str
        Formatted memory string.
        
    Examples
    --------
    >>> format_memory(0.5)
    '512.0 KB'
    >>> format_memory(1024)
    '1.00 GB'
    """
    if megabytes < 0.001:
        return f"{megabytes * 1024 * 1024:.1f} B"
    elif megabytes < 1:
        return f"{megabytes * 1024:.1f} KB"
    elif megabytes < 1024:
        return f"{megabytes:.2f} MB"
    else:
        return f"{megabytes / 1024:.2f} GB"
