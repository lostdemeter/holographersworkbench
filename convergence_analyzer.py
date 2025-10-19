"""
Convergence Analyzer for Iterative Optimization
================================================

Analyzes convergence patterns, detects diminishing returns, and recommends
optimal stopping points for iterative optimization processes.

This module completes the optimization toolkit:
1. Performance Profiler → Identify bottlenecks
2. Error Pattern Visualizer → Discover corrections
3. Formula Code Generator → Generate production code
4. Convergence Analyzer → Decide when to stop
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class ConvergenceRate:
    """Convergence rate analysis.
    
    Attributes
    ----------
    model_type : str
        "exponential", "power_law", "linear", or "logarithmic".
    parameters : Dict[str, float]
        Model parameters.
    r_squared : float
        Goodness of fit.
    convergence_speed : str
        "fast", "moderate", or "slow".
    model_function : Callable
        Function to predict future values.
    """
    model_type: str
    parameters: Dict[str, float]
    r_squared: float
    convergence_speed: str
    model_function: Callable
    
    def predict(self, iterations: np.ndarray) -> np.ndarray:
        """Predict metric values at given iterations."""
        return self.model_function(iterations)
    
    def describe(self) -> str:
        """Human-readable description."""
        speed_desc = {
            "fast": "converging rapidly (exponential decay)",
            "moderate": "converging steadily (power law decay)",
            "slow": "converging slowly (linear/logarithmic)"
        }
        return f"{self.model_type.replace('_', ' ').title()} convergence, {speed_desc[self.convergence_speed]}"


@dataclass
class DiminishingReturnsPoint:
    """Point where diminishing returns begin."""
    iteration: int
    metric_value: float
    improvement_rate: float
    threshold: float
    
    def describe(self) -> str:
        """Human-readable description."""
        return (f"Diminishing returns at iteration {self.iteration} "
                f"(rate: {self.improvement_rate:.2%}, threshold: {self.threshold:.2%})")


@dataclass
class StoppingRecommendation:
    """Recommendation for when to stop."""
    recommended_iteration: int
    current_iteration: int
    should_stop: bool
    reason: str
    confidence: float
    current_metric: float
    predicted_next_metric: float
    predicted_improvement: float
    cost_benefit_ratio: Optional[float] = None
    
    def describe(self) -> str:
        """Human-readable description."""
        action = "STOP" if self.should_stop else "CONTINUE"
        return (f"[{action}] {self.reason}\n"
                f"  Current: iteration {self.current_iteration}, metric = {self.current_metric:.6f}\n"
                f"  Predicted next: metric = {self.predicted_next_metric:.6f} "
                f"(improvement: {self.predicted_improvement:.2%})\n"
                f"  Confidence: {self.confidence:.0%}")


@dataclass
class OscillationPattern:
    """Oscillation pattern detection."""
    is_oscillating: bool
    frequency: Optional[float]
    amplitude: Optional[float]
    trend: str
    
    def describe(self) -> str:
        """Human-readable description."""
        if not self.is_oscillating:
            return "No oscillation detected"
        return (f"Oscillating with frequency {self.frequency:.2f} and amplitude {self.amplitude:.6f}, "
                f"trend: {self.trend}")


@dataclass
class ConvergenceReport:
    """Complete convergence analysis report."""
    metric_name: str
    n_iterations: int
    current_metric: float
    initial_metric: float
    total_improvement: float
    convergence_rate: ConvergenceRate
    diminishing_returns: Optional[DiminishingReturnsPoint]
    stopping_recommendation: StoppingRecommendation
    oscillation: OscillationPattern
    is_stagnant: bool
    is_converged: bool
    predicted_iterations: np.ndarray
    predicted_metrics: np.ndarray
    total_cost: Optional[float] = None
    cost_benefit_ratios: Optional[np.ndarray] = None
    
    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 70)
        print(f"CONVERGENCE ANALYSIS: {self.metric_name}")
        print("=" * 70)
        print(f"Iterations: {self.n_iterations}")
        print(f"Initial: {self.initial_metric:.6f}")
        print(f"Current: {self.current_metric:.6f}")
        print(f"Total improvement: {self.total_improvement:.2%}\n")
        
        print(f"Convergence: {self.convergence_rate.describe()}")
        print(f"R² = {self.convergence_rate.r_squared:.4f}\n")
        
        if self.diminishing_returns:
            print(self.diminishing_returns.describe())
        
        print(f"\nOscillation: {self.oscillation.describe()}")
        print(f"Stagnant: {'Yes' if self.is_stagnant else 'No'}")
        print(f"Converged: {'Yes' if self.is_converged else 'No'}\n")
        
        print("RECOMMENDATION")
        print("-" * 70)
        print(self.stopping_recommendation.describe())
        
        if self.total_cost is not None:
            print(f"\nTotal cost: {self.total_cost:.2f}")
        print("=" * 70)


# ============================================================================
# Convergence Model Fitting
# ============================================================================

class ConvergenceModelFitter:
    """Fit convergence models to metric history."""
    
    @staticmethod
    def fit_exponential(iterations: np.ndarray, metrics: np.ndarray) -> Tuple[Optional[Callable], Dict, float]:
        """Fit exponential decay: y = a * exp(-b * x) + c."""
        def model(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            p0 = [metrics[0] - metrics[-1], 0.1, metrics[-1]]
            params, _ = curve_fit(model, iterations, metrics, p0=p0, maxfev=10000)
            
            predicted = model(iterations, *params)
            ss_res = np.sum((metrics - predicted)**2)
            ss_tot = np.sum((metrics - np.mean(metrics))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return lambda x: model(x, *params), {'a': params[0], 'b': params[1], 'c': params[2]}, r_squared
        except:
            return None, {}, -np.inf
    
    @staticmethod
    def fit_power_law(iterations: np.ndarray, metrics: np.ndarray) -> Tuple[Optional[Callable], Dict, float]:
        """Fit power law decay: y = a * x^(-b) + c."""
        def model(x, a, b, c):
            return a * (x + 1) ** (-b) + c
        
        try:
            p0 = [metrics[0] - metrics[-1], 0.5, metrics[-1]]
            params, _ = curve_fit(model, iterations, metrics, p0=p0, maxfev=10000)
            
            predicted = model(iterations, *params)
            ss_res = np.sum((metrics - predicted)**2)
            ss_tot = np.sum((metrics - np.mean(metrics))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return lambda x: model(x, *params), {'a': params[0], 'b': params[1], 'c': params[2]}, r_squared
        except:
            return None, {}, -np.inf
    
    @staticmethod
    def fit_linear(iterations: np.ndarray, metrics: np.ndarray) -> Tuple[Callable, Dict, float]:
        """Fit linear decay: y = a * x + b."""
        params = np.polyfit(iterations, metrics, 1)
        predicted = np.polyval(params, iterations)
        ss_res = np.sum((metrics - predicted)**2)
        ss_tot = np.sum((metrics - np.mean(metrics))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return lambda x: np.polyval(params, x), {'a': params[0], 'b': params[1]}, r_squared
    
    @staticmethod
    def fit_logarithmic(iterations: np.ndarray, metrics: np.ndarray) -> Tuple[Callable, Dict, float]:
        """Fit logarithmic decay: y = a * log(x + 1) + b."""
        log_iterations = np.log(iterations + 1)
        params = np.polyfit(log_iterations, metrics, 1)
        predicted = np.polyval(params, log_iterations)
        ss_res = np.sum((metrics - predicted)**2)
        ss_tot = np.sum((metrics - np.mean(metrics))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return lambda x: params[0] * np.log(x + 1) + params[1], {'a': params[0], 'b': params[1]}, r_squared
    
    @staticmethod
    def fit_best_model(iterations: np.ndarray, metrics: np.ndarray) -> ConvergenceRate:
        """Fit all models and return the best one."""
        models = {
            'exponential': ConvergenceModelFitter.fit_exponential,
            'power_law': ConvergenceModelFitter.fit_power_law,
            'linear': ConvergenceModelFitter.fit_linear,
            'logarithmic': ConvergenceModelFitter.fit_logarithmic,
        }
        
        best_model = None
        best_r2 = -np.inf
        best_type = None
        best_params = None
        best_func = None
        
        for model_type, fit_func in models.items():
            func, params, r2 = fit_func(iterations, metrics)
            if func is not None and r2 > best_r2:
                best_r2 = r2
                best_type = model_type
                best_params = params
                best_func = func
        
        # Determine convergence speed
        if best_type == 'exponential':
            speed = 'fast'
        elif best_type == 'power_law':
            speed = 'moderate'
        else:
            speed = 'slow'
        
        return ConvergenceRate(
            model_type=best_type,
            parameters=best_params,
            r_squared=best_r2,
            convergence_speed=speed,
            model_function=best_func
        )


# ============================================================================
# Main Analyzer Class
# ============================================================================

class ConvergenceAnalyzer:
    """Analyze convergence of iterative optimization processes.
    
    Parameters
    ----------
    metric_history : List[float]
        History of metric values.
    metric_name : str
        Name of the metric.
    lower_is_better : bool
        Whether lower values are better.
    iteration_costs : List[float], optional
        Cost of each iteration.
    target_metric : float, optional
        Target metric value.
    
    Examples
    --------
    >>> analyzer = ConvergenceAnalyzer([0.5, 0.3, 0.2, 0.15], "RMSE")
    >>> report = analyzer.analyze()
    >>> report.print_summary()
    """
    
    def __init__(
        self,
        metric_history: List[float],
        metric_name: str = "RMSE",
        lower_is_better: bool = True,
        iteration_costs: List[float] = None,
        target_metric: float = None
    ):
        self.metric_history = np.array(metric_history, dtype=float)
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.iteration_costs = iteration_costs
        self.target_metric = target_metric
        
        self.n_iterations = len(metric_history)
        self.iterations = np.arange(self.n_iterations)
        
        self.convergence_rate = None
        self.diminishing_returns_threshold = None
        self.optimal_stopping_point = None
    
    def analyze(self) -> ConvergenceReport:
        """Run complete convergence analysis."""
        # Fit convergence model
        self.convergence_rate = self.detect_convergence_rate()
        
        # Detect diminishing returns
        self.diminishing_returns_threshold = self.detect_diminishing_returns()
        
        # Predict future
        future_iters, future_metrics = self.predict_future_improvements(n_future=10)
        
        # Detect patterns
        is_stagnant = self.detect_stagnation()
        oscillation = self.detect_oscillation()
        
        # Stopping recommendation
        stopping_rec = self.recommend_stopping_point()
        
        # Check convergence
        is_converged = stopping_rec.should_stop
        
        # Total improvement
        if self.lower_is_better:
            total_improvement = (self.metric_history[0] - self.metric_history[-1]) / self.metric_history[0]
        else:
            total_improvement = (self.metric_history[-1] - self.metric_history[0]) / self.metric_history[0]
        
        # Cost analysis
        total_cost = sum(self.iteration_costs) if self.iteration_costs else None
        cost_benefit = self.compute_cost_benefit_ratio()
        
        return ConvergenceReport(
            metric_name=self.metric_name,
            n_iterations=self.n_iterations,
            current_metric=self.metric_history[-1],
            initial_metric=self.metric_history[0],
            total_improvement=total_improvement,
            convergence_rate=self.convergence_rate,
            diminishing_returns=self.diminishing_returns_threshold,
            stopping_recommendation=stopping_rec,
            oscillation=oscillation,
            is_stagnant=is_stagnant,
            is_converged=is_converged,
            predicted_iterations=future_iters,
            predicted_metrics=future_metrics,
            total_cost=total_cost,
            cost_benefit_ratios=cost_benefit
        )
    
    def detect_convergence_rate(self) -> ConvergenceRate:
        """Detect convergence rate pattern."""
        return ConvergenceModelFitter.fit_best_model(self.iterations, self.metric_history)
    
    def predict_future_improvements(self, n_future: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future metric values."""
        if self.convergence_rate is None:
            self.convergence_rate = self.detect_convergence_rate()
        
        future_iters = np.arange(self.n_iterations, self.n_iterations + n_future)
        future_metrics = self.convergence_rate.predict(future_iters)
        
        return future_iters, future_metrics
    
    def detect_diminishing_returns(self, threshold: float = 0.01) -> Optional[DiminishingReturnsPoint]:
        """Detect when improvements become negligible."""
        improvements = self.compute_improvement_rates()
        
        if len(improvements) == 0:
            return None
        
        # Find first point below threshold
        below_threshold = np.where(improvements < threshold)[0]
        
        if len(below_threshold) > 0:
            idx = below_threshold[0]
            return DiminishingReturnsPoint(
                iteration=idx + 1,
                metric_value=self.metric_history[idx + 1],
                improvement_rate=improvements[idx],
                threshold=threshold
            )
        
        return None
    
    def recommend_stopping_point(
        self,
        improvement_threshold: float = 0.01,
        cost_benefit_ratio: float = None
    ) -> StoppingRecommendation:
        """Recommend optimal stopping point."""
        current_iter = self.n_iterations - 1
        current_metric = self.metric_history[-1]
        
        # Predict next iteration
        if self.convergence_rate is None:
            self.convergence_rate = self.detect_convergence_rate()
        
        next_iter = self.n_iterations
        predicted_next = self.convergence_rate.predict(np.array([next_iter]))[0]
        
        # Calculate predicted improvement
        if self.lower_is_better:
            predicted_improvement = (current_metric - predicted_next) / current_metric
        else:
            predicted_improvement = (predicted_next - current_metric) / current_metric
        
        # Decide if should stop
        should_stop = False
        reason = ""
        confidence = 0.0
        
        # Check improvement threshold
        if predicted_improvement < improvement_threshold:
            should_stop = True
            reason = f"Predicted improvement ({predicted_improvement:.2%}) below threshold ({improvement_threshold:.2%})"
            confidence = 0.85
        
        # Check stagnation
        elif self.detect_stagnation():
            should_stop = True
            reason = "Optimization has stagnated"
            confidence = 0.90
        
        # Check target reached
        elif self.target_metric is not None:
            if self.lower_is_better and current_metric <= self.target_metric:
                should_stop = True
                reason = f"Target metric ({self.target_metric:.6f}) reached"
                confidence = 1.0
            elif not self.lower_is_better and current_metric >= self.target_metric:
                should_stop = True
                reason = f"Target metric ({self.target_metric:.6f}) reached"
                confidence = 1.0
        
        if not should_stop:
            reason = f"Predicted improvement ({predicted_improvement:.2%}) above threshold"
            confidence = 0.70
        
        # Cost-benefit analysis
        cb_ratio = None
        if self.iteration_costs is not None:
            cb_ratios = self.compute_cost_benefit_ratio()
            if cb_ratios is not None and len(cb_ratios) > 0:
                cb_ratio = cb_ratios[-1]
        
        return StoppingRecommendation(
            recommended_iteration=current_iter if should_stop else current_iter + 1,
            current_iteration=current_iter,
            should_stop=should_stop,
            reason=reason,
            confidence=confidence,
            current_metric=current_metric,
            predicted_next_metric=predicted_next,
            predicted_improvement=predicted_improvement,
            cost_benefit_ratio=cb_ratio
        )
    
    def detect_stagnation(self, window_size: int = 3, tolerance: float = 1e-6) -> bool:
        """Detect if optimization has stagnated."""
        if self.n_iterations < window_size + 1:
            return False
        
        recent = self.metric_history[-window_size:]
        max_change = np.max(np.abs(np.diff(recent)))
        
        return max_change < tolerance
    
    def detect_oscillation(self) -> OscillationPattern:
        """Detect if metric is oscillating."""
        if self.n_iterations < 4:
            return OscillationPattern(False, None, None, "insufficient_data")
        
        # Simple oscillation detection: check for sign changes in differences
        diffs = np.diff(self.metric_history)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        is_oscillating = sign_changes > len(diffs) / 2
        
        if is_oscillating:
            amplitude = np.std(diffs)
            frequency = sign_changes / len(diffs)
            
            # Determine trend
            if abs(self.metric_history[-1] - self.metric_history[0]) < np.std(self.metric_history) * 0.1:
                trend = "stable"
            elif (self.lower_is_better and self.metric_history[-1] < self.metric_history[0]) or \
                 (not self.lower_is_better and self.metric_history[-1] > self.metric_history[0]):
                trend = "converging"
            else:
                trend = "diverging"
            
            return OscillationPattern(True, frequency, amplitude, trend)
        
        return OscillationPattern(False, None, None, "converging")
    
    def compute_improvement_rates(self) -> np.ndarray:
        """Compute relative improvement rate at each iteration."""
        if len(self.metric_history) < 2:
            return np.array([])
        
        improvements = np.diff(self.metric_history)
        if not self.lower_is_better:
            improvements = -improvements
        
        # Relative improvement
        with np.errstate(divide='ignore', invalid='ignore'):
            relative = improvements / np.abs(self.metric_history[:-1])
            relative[~np.isfinite(relative)] = 0.0
        
        return relative
    
    def compute_cost_benefit_ratio(self) -> Optional[np.ndarray]:
        """Compute cost per unit improvement."""
        if self.iteration_costs is None:
            return None
        
        improvements = self.compute_improvement_rates()
        if len(improvements) == 0:
            return None
        
        costs = np.array(self.iteration_costs[1:])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = costs / np.abs(improvements)
            ratios[~np.isfinite(ratios)] = np.inf
        
        return ratios
    
    def estimate_iterations_to_target(self, target: float = None) -> Optional[int]:
        """Estimate iterations needed to reach target."""
        if target is None:
            target = self.target_metric
        
        if target is None or self.convergence_rate is None:
            return None
        
        # Binary search for iteration where model reaches target
        for i in range(1, 1000):
            predicted = self.convergence_rate.predict(np.array([self.n_iterations + i]))[0]
            
            if self.lower_is_better and predicted <= target:
                return i
            elif not self.lower_is_better and predicted >= target:
                return i
        
        return None  # Target not reachable


# ============================================================================
# Visualization Class
# ============================================================================

class ConvergenceVisualizer:
    """Visualization tools for convergence analysis.
    
    Parameters
    ----------
    analyzer : ConvergenceAnalyzer
        Analyzer instance to visualize.
    """
    
    def __init__(self, analyzer: ConvergenceAnalyzer):
        self.analyzer = analyzer
    
    def plot_convergence_curve(self, ax=None, show_prediction=True):
        """Plot metric history with fitted model and predictions."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available. Install it to use plotting features.")
            return
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual
        ax.plot(self.analyzer.iterations, self.analyzer.metric_history, 
                'o-', label='Actual', linewidth=2, markersize=8)
        
        # Plot fitted model
        if self.analyzer.convergence_rate:
            fitted = self.analyzer.convergence_rate.predict(self.analyzer.iterations)
            ax.plot(self.analyzer.iterations, fitted, 
                    '--', label=f'Fitted ({self.analyzer.convergence_rate.model_type})', 
                    linewidth=2, alpha=0.7)
        
        # Plot predictions
        if show_prediction and self.analyzer.convergence_rate:
            future_iters, future_metrics = self.analyzer.predict_future_improvements(10)
            ax.plot(future_iters, future_metrics, 
                    ':', label='Predicted', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(self.analyzer.metric_name, fontsize=12)
        ax.set_title(f'{self.analyzer.metric_name} Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_improvement_rates(self, ax=None):
        """Plot improvement rate at each iteration."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available.")
            return
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        improvements = self.analyzer.compute_improvement_rates()
        
        if len(improvements) > 0:
            ax.plot(self.analyzer.iterations[1:], improvements * 100, 
                    'o-', linewidth=2, markersize=8)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Improvement Rate (%)', fontsize=12)
            ax.set_title('Improvement Rate per Iteration', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_dashboard(self, figsize=(14, 10)):
        """Create comprehensive convergence dashboard."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        self.plot_convergence_curve(axes[0])
        self.plot_improvement_rates(axes[1])
        
        plt.tight_layout()
        plt.show()
        
        return fig


# ============================================================================
# Utility Functions
# ============================================================================

def analyze_convergence_stability(metric_history: np.ndarray, window_size: int = 3) -> float:
    """Measure stability of convergence."""
    if len(metric_history) < window_size + 1:
        return np.inf
    
    recent_improvements = np.abs(np.diff(metric_history[-window_size-1:]))
    if np.mean(recent_improvements) == 0:
        return 0.0
    
    cv = np.std(recent_improvements) / np.mean(recent_improvements)
    return cv


def estimate_asymptotic_value(convergence_rate: ConvergenceRate) -> Optional[float]:
    """Estimate asymptotic value."""
    if convergence_rate.model_type in ['exponential', 'power_law']:
        return convergence_rate.parameters.get('c', None)
    return None


def compute_efficiency_score(
    metric_history: np.ndarray,
    iteration_costs: Optional[List[float]] = None
) -> float:
    """Compute overall efficiency score."""
    total_improvement = abs(metric_history[-1] - metric_history[0])
    
    if iteration_costs is None:
        total_cost = len(metric_history)
    else:
        total_cost = sum(iteration_costs)
    
    if total_cost == 0:
        return 0.0
    
    return total_improvement / total_cost
