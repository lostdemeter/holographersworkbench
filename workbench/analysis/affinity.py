"""
Time-Affinity Optimization Module
==================================

Optimizes algorithm parameters by using walltime as a fitness signal.

Theory:
-------
When algorithm parameters are "correct" or "resonant" with the problem structure,
the algorithm typically does less work and completes faster. By setting a target
time value and manipulating parameters until the algorithm approaches that time,
we can discover optimal parameter configurations.

Key Insight: Correct solutions → Less work → Faster execution

This works because:
1. Resonant parameters align with problem structure
2. Aligned algorithms explore fewer dead ends
3. Fewer iterations/operations → lower walltime
4. Walltime becomes a proxy for solution quality

Author: Holographer's Workbench
"""

import numpy as np
import time
from typing import Callable, Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import warnings


@dataclass
class TimeAffinityResult:
    """Results from time-affinity optimization."""
    best_params: Dict[str, float]
    best_time: float
    target_time: float
    time_error: float
    iterations: int
    time_history: List[float] = field(default_factory=list)
    param_history: List[Dict[str, float]] = field(default_factory=list)
    convergence_rate: float = 0.0
    
    def __str__(self):
        return (f"TimeAffinityResult(\n"
                f"  best_params={self.best_params},\n"
                f"  best_time={self.best_time:.6f}s,\n"
                f"  target_time={self.target_time:.6f}s,\n"
                f"  time_error={self.time_error:.6f}s,\n"
                f"  iterations={self.iterations}\n"
                f")")


class TimeAffinityOptimizer:
    """
    Optimize algorithm parameters using walltime as fitness signal.
    
    The optimizer manipulates parameters to drive algorithm execution time
    toward a target value, discovering resonant parameter configurations.
    
    Parameters
    ----------
    target_time : float
        Target execution time in seconds
    param_bounds : Dict[str, Tuple[float, float]]
        Parameter bounds {name: (min, max)}
    tolerance : float
        Acceptable time error (default: 0.01s = 10ms)
    max_iterations : int
        Maximum optimization iterations
    learning_rate : float
        Parameter update step size
    momentum : float
        Momentum for parameter updates (0-1)
    
    Examples
    --------
    >>> def my_algorithm(x, y):
    ...     # Some computation
    ...     result = expensive_operation(x, y)
    ...     return result
    >>> 
    >>> optimizer = TimeAffinityOptimizer(
    ...     target_time=0.1,  # Target 100ms
    ...     param_bounds={'x': (0.0, 1.0), 'y': (0.0, 10.0)}
    ... )
    >>> result = optimizer.optimize(my_algorithm)
    >>> print(f"Optimal params: {result.best_params}")
    """
    
    def __init__(self,
                 target_time: float,
                 param_bounds: Dict[str, Tuple[float, float]],
                 tolerance: float = 0.01,
                 max_iterations: int = 100,
                 learning_rate: float = 0.1,
                 momentum: float = 0.5,
                 warmup_runs: int = 3):
        self.target_time = target_time
        self.param_bounds = param_bounds
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.warmup_runs = warmup_runs
        
        # Initialize parameters at midpoint
        self.current_params = {
            name: (bounds[0] + bounds[1]) / 2
            for name, bounds in param_bounds.items()
        }
        
        # Momentum tracking
        self.velocity = {name: 0.0 for name in param_bounds.keys()}
        
        # History
        self.time_history = []
        self.param_history = []
    
    def _clip_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to bounds."""
        clipped = {}
        for name, value in params.items():
            min_val, max_val = self.param_bounds[name]
            clipped[name] = np.clip(value, min_val, max_val)
        return clipped
    
    def _measure_time(self, 
                     algorithm: Callable,
                     params: Dict[str, float],
                     *args, **kwargs) -> float:
        """
        Measure algorithm execution time with given parameters.
        
        Uses median of multiple runs to reduce noise.
        """
        times = []
        for _ in range(self.warmup_runs):
            start = time.perf_counter()
            try:
                algorithm(*args, **params, **kwargs)
            except Exception as e:
                warnings.warn(f"Algorithm failed with params {params}: {e}")
                return float('inf')
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Return median to reduce noise
        return float(np.median(times))
    
    def _compute_gradient(self,
                         algorithm: Callable,
                         params: Dict[str, float],
                         current_time: float,
                         *args, **kwargs) -> Dict[str, float]:
        """
        Estimate gradient of time w.r.t. parameters using finite differences.
        """
        gradient = {}
        epsilon = 1e-4
        
        for name in params.keys():
            # Perturb parameter
            params_plus = params.copy()
            params_plus[name] += epsilon
            params_plus = self._clip_params(params_plus)
            
            # Measure time with perturbation
            time_plus = self._measure_time(algorithm, params_plus, *args, **kwargs)
            
            # Finite difference
            gradient[name] = (time_plus - current_time) / epsilon
        
        return gradient
    
    def optimize(self,
                algorithm: Callable,
                *args,
                verbose: bool = True,
                **kwargs) -> TimeAffinityResult:
        """
        Optimize parameters to match target execution time.
        
        Parameters
        ----------
        algorithm : Callable
            Function to optimize. Should accept parameters as keyword arguments.
        *args, **kwargs
            Additional arguments passed to algorithm
        verbose : bool
            Print progress
        
        Returns
        -------
        TimeAffinityResult
            Optimization results with best parameters
        """
        best_params = self.current_params.copy()
        best_time = float('inf')
        best_error = float('inf')
        
        if verbose:
            print(f"Time-Affinity Optimization")
            print(f"Target time: {self.target_time:.6f}s")
            print(f"Tolerance: {self.tolerance:.6f}s")
            print("=" * 70)
        
        for iteration in range(self.max_iterations):
            # Measure current time
            current_time = self._measure_time(
                algorithm, self.current_params, *args, **kwargs
            )
            
            # Track history
            self.time_history.append(current_time)
            self.param_history.append(self.current_params.copy())
            
            # Compute error
            time_error = abs(current_time - self.target_time)
            
            # Update best
            if time_error < best_error:
                best_error = time_error
                best_time = current_time
                best_params = self.current_params.copy()
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: time={current_time:.6f}s, "
                      f"error={time_error:.6f}s, params={self.current_params}")
            
            # Check convergence
            if time_error < self.tolerance:
                if verbose:
                    print(f"\n✓ Converged at iteration {iteration}")
                break
            
            # Compute gradient
            gradient = self._compute_gradient(
                algorithm, self.current_params, current_time, *args, **kwargs
            )
            
            # Update parameters with momentum
            # Move opposite to gradient to minimize |time - target|
            time_diff = current_time - self.target_time
            for name in self.current_params.keys():
                # Gradient descent on time error
                grad_direction = np.sign(time_diff) * gradient[name]
                
                # Update velocity with momentum
                self.velocity[name] = (
                    self.momentum * self.velocity[name] -
                    self.learning_rate * grad_direction
                )
                
                # Update parameter
                self.current_params[name] += self.velocity[name]
            
            # Clip to bounds
            self.current_params = self._clip_params(self.current_params)
        
        # Compute convergence rate
        if len(self.time_history) > 1:
            errors = [abs(t - self.target_time) for t in self.time_history]
            convergence_rate = (errors[0] - errors[-1]) / len(errors)
        else:
            convergence_rate = 0.0
        
        result = TimeAffinityResult(
            best_params=best_params,
            best_time=best_time,
            target_time=self.target_time,
            time_error=best_error,
            iterations=len(self.time_history),
            time_history=self.time_history,
            param_history=self.param_history,
            convergence_rate=convergence_rate
        )
        
        if verbose:
            print("=" * 70)
            print(result)
        
        return result


class GridSearchTimeAffinity:
    """
    Grid search variant for time-affinity optimization.
    
    Exhaustively searches parameter space to find configuration
    closest to target time. Slower but more robust than gradient-based.
    
    Parameters
    ----------
    target_time : float
        Target execution time
    param_grids : Dict[str, np.ndarray]
        Parameter grids {name: array of values}
    warmup_runs : int
        Runs per configuration for timing
    
    Examples
    --------
    >>> optimizer = GridSearchTimeAffinity(
    ...     target_time=0.1,
    ...     param_grids={
    ...         'x': np.linspace(0, 1, 20),
    ...         'y': np.linspace(0, 10, 20)
    ...     }
    ... )
    >>> result = optimizer.optimize(my_algorithm)
    """
    
    def __init__(self,
                 target_time: float,
                 param_grids: Dict[str, np.ndarray],
                 warmup_runs: int = 3):
        self.target_time = target_time
        self.param_grids = param_grids
        self.warmup_runs = warmup_runs
    
    def optimize(self,
                algorithm: Callable,
                *args,
                verbose: bool = True,
                **kwargs) -> TimeAffinityResult:
        """
        Grid search for optimal parameters.
        
        Parameters
        ----------
        algorithm : Callable
            Function to optimize
        *args, **kwargs
            Additional arguments
        verbose : bool
            Print progress
        
        Returns
        -------
        TimeAffinityResult
            Best parameters found
        """
        param_names = list(self.param_grids.keys())
        param_values = list(self.param_grids.values())
        
        # Generate all combinations
        import itertools
        combinations = list(itertools.product(*param_values))
        total = len(combinations)
        
        if verbose:
            print(f"Grid Search Time-Affinity Optimization")
            print(f"Target time: {self.target_time:.6f}s")
            print(f"Grid size: {total} combinations")
            print("=" * 70)
        
        best_params = None
        best_time = None
        best_error = float('inf')
        time_history = []
        param_history = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Measure time
            times = []
            for _ in range(self.warmup_runs):
                start = time.perf_counter()
                try:
                    algorithm(*args, **params, **kwargs)
                except Exception:
                    times.append(float('inf'))
                    break
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            measured_time = float(np.median(times))
            time_error = abs(measured_time - self.target_time)
            
            time_history.append(measured_time)
            param_history.append(params)
            
            if time_error < best_error:
                best_error = time_error
                best_time = measured_time
                best_params = params
            
            if verbose and (i + 1) % max(1, total // 10) == 0:
                progress = (i + 1) / total * 100
                print(f"Progress: {progress:.0f}% ({i+1}/{total}), "
                      f"best_error={best_error:.6f}s")
        
        result = TimeAffinityResult(
            best_params=best_params,
            best_time=best_time,
            target_time=self.target_time,
            time_error=best_error,
            iterations=total,
            time_history=time_history,
            param_history=param_history
        )
        
        if verbose:
            print("=" * 70)
            print(result)
        
        return result


def quick_calibrate(algorithm: Callable,
                   target_time: float,
                   param_bounds: Dict[str, Tuple[float, float]],
                   method: str = 'gradient',
                   **kwargs) -> TimeAffinityResult:
    """
    Quick time-affinity calibration with sensible defaults.
    
    Parameters
    ----------
    algorithm : Callable
        Function to calibrate
    target_time : float
        Target execution time
    param_bounds : Dict[str, Tuple[float, float]]
        Parameter bounds
    method : str
        'gradient' or 'grid'
    **kwargs
        Additional arguments for optimizer
    
    Returns
    -------
    TimeAffinityResult
        Calibration results
    
    Examples
    --------
    >>> result = quick_calibrate(
    ...     my_algorithm,
    ...     target_time=0.1,
    ...     param_bounds={'x': (0, 1), 'y': (0, 10)}
    ... )
    """
    if method == 'gradient':
        # Extract optimizer kwargs vs optimize kwargs
        opt_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['tolerance', 'max_iterations', 'learning_rate', 'momentum', 'warmup_runs']}
        run_kwargs = {k: v for k, v in kwargs.items() if k not in opt_kwargs}
        
        optimizer = TimeAffinityOptimizer(
            target_time=target_time,
            param_bounds=param_bounds,
            **opt_kwargs
        )
        return optimizer.optimize(algorithm, **run_kwargs)
    
    elif method == 'grid':
        # Generate grids
        param_grids = {
            name: np.linspace(bounds[0], bounds[1], kwargs.get('grid_points', 20))
            for name, bounds in param_bounds.items()
        }
        optimizer = GridSearchTimeAffinity(
            target_time=target_time,
            param_grids=param_grids
        )
        return optimizer.optimize(algorithm)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_spectral_alpha(signal: np.ndarray) -> Tuple[float, float]:
    """
    Estimate 1/f^α spectral exponent from a signal.
    
    The spectral exponent α characterizes the power-law decay of the
    power spectrum: P(f) ∝ 1/f^α
    
    - α ≈ 0: White noise (flat spectrum)
    - α ≈ 1: Pink noise (1/f)
    - α ≈ 2: Brown noise (1/f²)
    - α > 2: Highly correlated
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal to analyze
        
    Returns
    -------
    alpha : float
        Spectral exponent
    r_squared : float
        Goodness of fit (0-1)
        
    Examples
    --------
    >>> phases = [recursive_theta(i) / (2*np.pi) % 1 for i in range(1, 1001)]
    >>> alpha, r2 = estimate_spectral_alpha(phases)
    >>> print(f"Alpha: {alpha:.2f}, R²: {r2:.3f}")
    """
    from scipy import signal as sig
    
    # Compute power spectrum
    freqs, psd = sig.welch(signal, nperseg=min(256, len(signal) // 4))
    
    # Avoid DC component and very high frequencies
    mask = (freqs > freqs[1]) & (freqs < freqs[-1] * 0.8)
    freqs = freqs[mask]
    psd = psd[mask]
    
    if len(freqs) < 5:
        return 1.0, 0.0
    
    # Log-log linear regression
    log_f = np.log10(freqs)
    log_psd = np.log10(psd + 1e-20)
    
    # Linear fit: log(P) = -α * log(f) + c
    coeffs = np.polyfit(log_f, log_psd, 1)
    alpha = -coeffs[0]
    
    # R² goodness of fit
    predicted = np.polyval(coeffs, log_f)
    ss_res = np.sum((log_psd - predicted) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    return alpha, r_squared


def estimate_fractal_dimension_from_alpha(alpha: float) -> float:
    """
    Estimate fractal dimension from spectral exponent.
    
    Uses the relationship: D = (5 - α) / 2 for 1D signals
    
    - α = 1 → D = 2.0 (space-filling)
    - α = 2 → D = 1.5 (Brownian)
    - α = 3 → D = 1.0 (smooth)
    """
    return np.clip((5 - alpha) / 2, 1.0, 2.0)


def compute_adaptive_n_phases(n: int, alpha: float = None, signal: np.ndarray = None) -> int:
    """
    Compute adaptive number of phases based on 1/f^α spectral analysis.
    
    Higher α (more correlated) → fewer phases needed
    Lower α (more random) → more phases needed
    
    Parameters
    ----------
    n : int
        Problem size
    alpha : float, optional
        Pre-computed spectral exponent
    signal : np.ndarray, optional
        Signal to analyze for alpha
        
    Returns
    -------
    n_phases : int
        Recommended number of phases
    """
    if alpha is None and signal is not None:
        alpha, _ = estimate_spectral_alpha(signal)
    elif alpha is None:
        alpha = 1.5  # Default: pink noise assumption
    
    # Base: 8 * log2(N)
    base_phases = 8 * np.log2(max(2, n))
    
    # Scale by alpha: higher α → fewer phases
    # α = 1.0 → scale = 1.0
    # α = 1.5 → scale = 0.85
    # α = 2.0 → scale = 0.7
    scale = 1.0 - 0.15 * (alpha - 1.0)
    scale = np.clip(scale, 0.5, 1.5)
    
    n_phases = int(np.ceil(base_phases * scale))
    return max(10, min(n_phases, 100))  # Clamp to [10, 100]
