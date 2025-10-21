"""
Error Pattern Visualizer for Holographic Algorithms
====================================================

Automatically discovers correction patterns in error signals and suggests
improvements to formulas and approximations.

This module emerged from optimizing fast_zetas.py, where manual error analysis
was time-consuming. It automates the discovery of spectral, polynomial,
autocorrelation, and scale-dependent patterns in error signals.
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

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
class SpectralPattern:
    """Spectral analysis results from FFT.
    
    Attributes
    ----------
    frequencies : np.ndarray
        Dominant frequencies detected.
    amplitudes : np.ndarray
        Amplitudes at those frequencies.
    phases : np.ndarray
        Phases at those frequencies.
    power_spectrum : np.ndarray
        Full power spectrum.
    frequency_axis : np.ndarray
        Frequency axis for spectrum.
    explained_variance : float
        Fraction of error variance explained by harmonics.
    """
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    power_spectrum: np.ndarray
    frequency_axis: np.ndarray
    explained_variance: float
    
    def to_correction_code(self, var_name: str = "x") -> str:
        """Generate Python code for this correction.
        
        Parameters
        ----------
        var_name : str
            Variable name to use in generated code.
            
        Returns
        -------
        str
            Executable Python code.
        """
        lines = ["# Spectral correction (harmonic terms)"]
        lines.append(f"correction = np.zeros_like({var_name})")
        
        for i, (freq, amp, phase) in enumerate(zip(self.frequencies, self.amplitudes, self.phases)):
            lines.append(
                f"correction += {amp:.6f} * np.sin(2 * np.pi * {freq:.6f} * {var_name} + {phase:.6f})  "
                f"# Harmonic {i+1}"
            )
        
        lines.append(f"predicted_corrected = predicted + correction")
        return "\n".join(lines)


@dataclass
class PolynomialPattern:
    """Polynomial trend analysis.
    
    Attributes
    ----------
    degree : int
        Degree of best-fit polynomial.
    coefficients : np.ndarray
        Polynomial coefficients (highest degree first).
    r_squared : float
        R² goodness of fit.
    residual_std : float
        Standard deviation of residuals.
    """
    degree: int
    coefficients: np.ndarray
    r_squared: float
    residual_std: float
    
    def to_correction_code(self, var_name: str = "x") -> str:
        """Generate Python code for this correction.
        
        Parameters
        ----------
        var_name : str
            Variable name to use in generated code.
            
        Returns
        -------
        str
            Executable Python code.
        """
        lines = ["# Polynomial correction"]
        
        # Build polynomial string
        terms = []
        for i, coeff in enumerate(self.coefficients):
            power = len(self.coefficients) - i - 1
            if abs(coeff) < 1e-10:
                continue
            if power == 0:
                terms.append(f"{coeff:.6f}")
            elif power == 1:
                terms.append(f"{coeff:.6f} * {var_name}")
            else:
                terms.append(f"{coeff:.6f} * {var_name}**{power}")
        
        poly_str = " + ".join(terms).replace("+ -", "- ")
        lines.append(f"correction = {poly_str}")
        lines.append(f"predicted_corrected = predicted + correction")
        return "\n".join(lines)


@dataclass
class AutocorrPattern:
    """Autocorrelation analysis.
    
    Attributes
    ----------
    lags : np.ndarray
        Lag values.
    autocorr : np.ndarray
        Autocorrelation at each lag.
    significant_lags : List[int]
        Lags with significant correlation.
    ar_order : Optional[int]
        Suggested AR model order.
    ar_coefficients : Optional[np.ndarray]
        AR model coefficients if fitted.
    """
    lags: np.ndarray
    autocorr: np.ndarray
    significant_lags: List[int]
    ar_order: Optional[int]
    ar_coefficients: Optional[np.ndarray]
    
    def to_correction_code(self, var_name: str = "x") -> str:
        """Generate Python code for this correction.
        
        Parameters
        ----------
        var_name : str
            Variable name to use in generated code.
            
        Returns
        -------
        str
            Executable Python code.
        """
        lines = ["# Autocorrelation correction (AR model)"]
        
        if self.ar_coefficients is not None and self.ar_order is not None:
            lines.append(f"# AR({self.ar_order}) model")
            lines.append(f"correction = np.zeros_like(predicted)")
            lines.append(f"for i in range({self.ar_order}, len(predicted)):")
            
            terms = []
            for j, coeff in enumerate(self.ar_coefficients):
                terms.append(f"{coeff:.6f} * error[i-{j+1}]")
            
            ar_expr = " + ".join(terms)
            lines.append(f"    correction[i] = {ar_expr}")
            lines.append(f"predicted_corrected = predicted + correction")
        else:
            lines.append("# No significant autocorrelation detected")
            lines.append("predicted_corrected = predicted")
        
        return "\n".join(lines)


@dataclass
class ScalePattern:
    """Scale-dependent error analysis.
    
    Attributes
    ----------
    bins : np.ndarray
        X-value bin centers.
    mean_errors : np.ndarray
        Mean error per bin.
    std_errors : np.ndarray
        Standard deviation per bin.
    scale_function : Optional[Callable]
        Function modeling scale dependence.
    scale_params : Dict
        Parameters for scale function.
    """
    bins: np.ndarray
    mean_errors: np.ndarray
    std_errors: np.ndarray
    scale_function: Optional[Callable]
    scale_params: Dict
    
    def to_correction_code(self, var_name: str = "x") -> str:
        """Generate Python code for this correction.
        
        Parameters
        ----------
        var_name : str
            Variable name to use in generated code.
            
        Returns
        -------
        str
            Executable Python code.
        """
        lines = ["# Scale-dependent correction"]
        
        if self.scale_function is not None and 'model' in self.scale_params:
            model = self.scale_params['model']
            params = self.scale_params['params']
            
            if model == 'power_law':
                a, b = params
                lines.append(f"correction = {a:.6f} * {var_name}**{b:.6f}")
            elif model == 'exponential':
                a, b = params
                lines.append(f"correction = {a:.6f} * np.exp({b:.6f} * {var_name})")
            elif model == 'logarithmic':
                a, b = params
                lines.append(f"correction = {a:.6f} * np.log({var_name} + 1) + {b:.6f}")
            else:
                lines.append("# Unknown scale model")
                lines.append("correction = np.zeros_like(predicted)")
            
            lines.append(f"predicted_corrected = predicted + correction")
        else:
            lines.append("# No scale dependence detected")
            lines.append("predicted_corrected = predicted")
        
        return "\n".join(lines)


@dataclass
class CorrectionSuggestion:
    """Suggested correction term.
    
    Attributes
    ----------
    pattern_type : str
        Type of pattern ("spectral", "polynomial", "autocorr", "scale").
    description : str
        Human-readable description.
    code_snippet : str
        Python code to apply correction.
    improvement_estimate : float
        Estimated RMSE reduction (0-1).
    priority : int
        Priority (1=highest).
    pattern_data : object
        Original pattern object.
    """
    pattern_type: str
    description: str
    code_snippet: str
    improvement_estimate: float
    priority: int
    pattern_data: object
    
    def apply(self, x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Apply this correction to error signal.
        
        Parameters
        ----------
        x : np.ndarray
            X-values.
        error : np.ndarray
            Current error signal.
            
        Returns
        -------
        np.ndarray
            Correction values.
        """
        correction = np.zeros_like(error)
        
        if self.pattern_type == "spectral" and isinstance(self.pattern_data, SpectralPattern):
            pattern = self.pattern_data
            for freq, amp, phase in zip(pattern.frequencies, pattern.amplitudes, pattern.phases):
                correction += amp * np.sin(2 * np.pi * freq * x + phase)
        
        elif self.pattern_type == "polynomial" and isinstance(self.pattern_data, PolynomialPattern):
            pattern = self.pattern_data
            correction = np.polyval(pattern.coefficients, x)
        
        elif self.pattern_type == "scale" and isinstance(self.pattern_data, ScalePattern):
            pattern = self.pattern_data
            if pattern.scale_function is not None and 'params' in pattern.scale_params:
                correction = pattern.scale_function(x, *pattern.scale_params['params'])
        
        return correction


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis report.
    
    Attributes
    ----------
    name : str
        Name of this analysis.
    initial_rmse : float
        Initial root mean squared error.
    initial_mae : float
        Initial mean absolute error.
    initial_max_error : float
        Initial maximum absolute error.
    spectral_pattern : Optional[SpectralPattern]
        Detected spectral pattern.
    polynomial_pattern : Optional[PolynomialPattern]
        Detected polynomial pattern.
    autocorr_pattern : Optional[AutocorrPattern]
        Detected autocorrelation pattern.
    scale_pattern : Optional[ScalePattern]
        Detected scale pattern.
    suggestions : List[CorrectionSuggestion]
        Prioritized correction suggestions.
    """
    name: str
    initial_rmse: float
    initial_mae: float
    initial_max_error: float
    spectral_pattern: Optional[SpectralPattern]
    polynomial_pattern: Optional[PolynomialPattern]
    autocorr_pattern: Optional[AutocorrPattern]
    scale_pattern: Optional[ScalePattern]
    suggestions: List[CorrectionSuggestion]
    
    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 70)
        print(f"ERROR ANALYSIS REPORT: {self.name}")
        print("=" * 70)
        print(f"Initial RMSE: {self.initial_rmse:.6f}")
        print(f"Initial MAE:  {self.initial_mae:.6f}")
        print(f"Initial Max:  {self.initial_max_error:.6f}")
        print()
        
        if self.spectral_pattern:
            print(f"✓ Spectral patterns detected: {len(self.spectral_pattern.frequencies)} harmonics")
            print(f"  Explained variance: {self.spectral_pattern.explained_variance:.1%}")
        
        if self.polynomial_pattern:
            print(f"✓ Polynomial trend detected: degree {self.polynomial_pattern.degree}")
            print(f"  R²: {self.polynomial_pattern.r_squared:.4f}")
        
        if self.autocorr_pattern and self.autocorr_pattern.ar_order:
            print(f"✓ Autocorrelation detected: AR({self.autocorr_pattern.ar_order})")
        
        if self.scale_pattern and self.scale_pattern.scale_function:
            model = self.scale_pattern.scale_params.get('model', 'unknown')
            print(f"✓ Scale dependence detected: {model}")
        
        print()
        print(f"Top {len(self.suggestions)} correction suggestions:")
        for i, sug in enumerate(self.suggestions[:5], 1):
            print(f"  {i}. {sug.description}")
            print(f"     Estimated improvement: {sug.improvement_estimate:.1%}")
        print("=" * 70)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "initial_rmse": float(self.initial_rmse),
            "initial_mae": float(self.initial_mae),
            "initial_max_error": float(self.initial_max_error),
            "has_spectral": self.spectral_pattern is not None,
            "has_polynomial": self.polynomial_pattern is not None,
            "has_autocorr": self.autocorr_pattern is not None,
            "has_scale": self.scale_pattern is not None,
            "num_suggestions": len(self.suggestions)
        }


@dataclass
class RefinementHistory:
    """History of recursive refinement.
    
    Attributes
    ----------
    initial_rmse : float
        RMSE before any corrections.
    final_rmse : float
        RMSE after all corrections.
    improvement : float
        Fraction of RMSE improved (0-1).
    corrections_applied : List[CorrectionSuggestion]
        List of applied corrections.
    rmse_history : List[float]
        RMSE after each correction.
    depth : int
        Number of refinement layers.
    """
    initial_rmse: float
    final_rmse: float
    improvement: float
    corrections_applied: List[CorrectionSuggestion]
    rmse_history: List[float]
    depth: int
    
    def plot_convergence(self):
        """Plot RMSE convergence."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available. Install it to use plotting features.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(self.rmse_history)), self.rmse_history, 'o-', linewidth=2)
        ax.set_xlabel('Refinement Step')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Recursive Refinement Convergence ({self.improvement:.1%} improvement)')
        ax.grid(True, alpha=0.3)
        
        # Annotate corrections
        for i, corr in enumerate(self.corrections_applied):
            ax.annotate(
                corr.pattern_type,
                xy=(i+1, self.rmse_history[i+1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# Pattern Detection Functions
# ============================================================================


def detect_spectral_patterns(error: np.ndarray, x_values: np.ndarray, n_harmonics: int = 10):
    """Detect periodic patterns using FFT.
    
    Algorithm:
    1. Compute FFT of error signal
    2. Find peaks in power spectrum
    3. Extract top n_harmonics frequencies
    4. Compute amplitudes and phases
    5. Estimate explained variance
    
    Parameters
    ----------
    error : np.ndarray
        Error signal.
    x_values : np.ndarray
        X-axis values.
    n_harmonics : int
        Number of harmonics to extract.
        
    Returns
    -------
    SpectralPattern
        Detected spectral pattern.
    """
    
    # FFT
    fft_result = np.fft.fft(error)
    power = np.abs(fft_result) ** 2
    
    # Compute frequency axis
    dx = np.mean(np.diff(x_values)) if len(x_values) > 1 else 1.0
    freqs = np.fft.fftfreq(len(error), d=dx)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power_pos = power[pos_mask]
    fft_pos = fft_result[pos_mask]
    
    # Find peaks
    peaks, properties = signal.find_peaks(power_pos, height=np.max(power_pos) * 0.01)
    
    if len(peaks) == 0:
        # No significant peaks
        return SpectralPattern(
            frequencies=np.array([]),
            amplitudes=np.array([]),
            phases=np.array([]),
            power_spectrum=power_pos,
            frequency_axis=freqs_pos,
            explained_variance=0.0
        )
    
    # Sort by power
    peak_powers = power_pos[peaks]
    sorted_indices = np.argsort(-peak_powers)[:n_harmonics]
    
    dominant_freqs = freqs_pos[peaks[sorted_indices]]
    dominant_amps = np.sqrt(power_pos[peaks[sorted_indices]]) * 2 / len(error)
    phases = np.angle(fft_pos[peaks[sorted_indices]])
    
    # Estimate explained variance
    reconstructed = np.zeros_like(error)
    for freq, amp, phase in zip(dominant_freqs, dominant_amps, phases):
        reconstructed += amp * np.sin(2 * np.pi * freq * x_values + phase)
    
    explained_var = max(0.0, 1 - np.var(error - reconstructed) / np.var(error))
    
    return SpectralPattern(
        frequencies=dominant_freqs,
        amplitudes=dominant_amps,
        phases=phases,
        power_spectrum=power_pos,
        frequency_axis=freqs_pos,
        explained_variance=explained_var
    )


def detect_polynomial_trend(error: np.ndarray, x_values: np.ndarray, max_degree: int = 5):
    """Detect polynomial trends using least squares.
    
    Algorithm:
    1. Fit polynomials of degree 1 to max_degree
    2. Select best degree using BIC-like criterion
    3. Return best-fit polynomial
    
    Parameters
    ----------
    error : np.ndarray
        Error signal.
    x_values : np.ndarray
        X-axis values.
    max_degree : int
        Maximum polynomial degree to try.
        
    Returns
    -------
    PolynomialPattern
        Detected polynomial pattern.
    """
    
    best_degree = 1
    best_score = -np.inf
    best_coeffs = None
    best_r2 = 0.0
    
    for degree in range(1, max_degree + 1):
        try:
            coeffs = np.polyfit(x_values, error, degree)
            predicted = np.polyval(coeffs, x_values)
            
            # R²
            ss_res = np.sum((error - predicted) ** 2)
            ss_tot = np.sum((error - np.mean(error)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # BIC-like penalty
            n = len(error)
            bic_penalty = (degree + 1) * np.log(n) / n
            adjusted_score = r2 - bic_penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_degree = degree
                best_coeffs = coeffs
                best_r2 = r2
        except:
            continue
    
    if best_coeffs is None:
        best_coeffs = np.array([0.0])
        best_degree = 0
        best_r2 = 0.0
    
    predicted = np.polyval(best_coeffs, x_values)
    residual_std = np.std(error - predicted)
    
    return PolynomialPattern(
        degree=best_degree,
        coefficients=best_coeffs,
        r_squared=best_r2,
        residual_std=residual_std
    )


def detect_autocorrelation(error: np.ndarray, max_lag: int = 50):
    """Detect autocorrelation structure.
    
    Algorithm:
    1. Compute autocorrelation function
    2. Find significant lags (> 2/sqrt(n))
    3. Suggest AR model order
    4. Fit AR model if significant
    
    Parameters
    ----------
    error : np.ndarray
        Error signal.
    max_lag : int
        Maximum lag to compute.
        
    Returns
    -------
    AutocorrPattern
        Detected autocorrelation pattern.
    """
    
    # Compute autocorrelation
    autocorr_full = signal.correlate(error, error, mode='full')
    autocorr_full = autocorr_full[len(autocorr_full)//2:]  # Keep only positive lags
    autocorr_full = autocorr_full / autocorr_full[0]  # Normalize
    
    max_lag = min(max_lag, len(autocorr_full) - 1)
    lags = np.arange(max_lag + 1)
    autocorr = autocorr_full[:max_lag + 1]
    
    # Significance threshold (95% confidence)
    threshold = 2 / np.sqrt(len(error))
    significant_lags = lags[np.abs(autocorr) > threshold]
    significant_lags = significant_lags[significant_lags > 0]  # Exclude lag 0
    
    # Suggest AR order (first significant lag)
    ar_order = int(significant_lags[0]) if len(significant_lags) > 0 else None
    
    # Fit AR model if significant
    ar_coeffs = None
    if ar_order is not None and ar_order <= max_lag and ar_order > 0:
        try:
            # Use Yule-Walker equations
            r = autocorr[1:ar_order+1]
            R = toeplitz(autocorr[:ar_order])
            ar_coeffs = np.linalg.solve(R, r)
        except:
            ar_coeffs = None
    
    return AutocorrPattern(
        lags=lags,
        autocorr=autocorr,
        significant_lags=list(significant_lags[:10]),  # Top 10
        ar_order=ar_order,
        ar_coefficients=ar_coeffs
    )


def detect_scale_dependence(error: np.ndarray, x_values: np.ndarray, n_bins: int = 10):
    """Detect if error depends on x-value scale.
    
    Algorithm:
    1. Bin x-values into n_bins
    2. Compute mean and std error per bin
    3. Fit scale function (power law, exponential, logarithmic)
    4. Return best-fit model
    
    Parameters
    ----------
    error : np.ndarray
        Error signal.
    x_values : np.ndarray
        X-axis values.
    n_bins : int
        Number of bins.
        
    Returns
    -------
    ScalePattern
        Detected scale pattern.
    """
    
    # Bin x-values
    bins = np.linspace(np.min(x_values), np.max(x_values), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    mean_errors = np.zeros(n_bins)
    std_errors = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (x_values >= bins[i]) & (x_values < bins[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (x_values >= bins[i]) & (x_values <= bins[i+1])
        
        if np.sum(mask) > 0:
            mean_errors[i] = np.mean(error[mask])
            std_errors[i] = np.std(error[mask])
    
    # Try different scale functions
    def power_law(x, a, b):
        return a * np.power(np.abs(x) + 1e-10, b)
    
    def exponential(x, a, b):
        return a * np.exp(b * x)
    
    def logarithmic(x, a, b):
        return a * np.log(np.abs(x) + 1) + b
    
    models = {
        'power_law': power_law,
        'exponential': exponential,
        'logarithmic': logarithmic
    }
    
    best_model = None
    best_params = None
    best_r2 = -np.inf
    
    for name, func in models.items():
        try:
            # Initial guess
            p0 = [1.0, 0.1] if name != 'logarithmic' else [1.0, 0.0]
            params, _ = curve_fit(func, bin_centers, mean_errors, p0=p0, maxfev=10000)
            predicted = func(bin_centers, *params)
            
            # R²
            ss_res = np.sum((mean_errors - predicted) ** 2)
            ss_tot = np.sum((mean_errors - np.mean(mean_errors)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            if r2 > best_r2 and r2 > 0.1:  # Require at least 10% explained
                best_r2 = r2
                best_model = name
                best_params = params
        except:
            continue
    
    scale_function = models[best_model] if best_model else None
    scale_params = {'model': best_model, 'params': best_params, 'r2': best_r2} if best_model else {}
    
    return ScalePattern(
        bins=bin_centers,
        mean_errors=mean_errors,
        std_errors=std_errors,
        scale_function=scale_function,
        scale_params=scale_params
    )


# Utility functions
def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(actual - predicted))


def compute_max_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute maximum absolute error."""
    return np.max(np.abs(actual - predicted))


def prioritize_corrections(suggestions):
    """Sort corrections by priority and improvement estimate.
    
    Priority order:
    1. Polynomial (systematic bias)
    2. Spectral (periodic patterns)
    3. Scale (scale-dependent)
    4. Autocorr (recursive patterns)
    """
    priority_map = {
        'polynomial': 1,
        'spectral': 2,
        'scale': 3,
        'autocorr': 4
    }
    
    return sorted(
        suggestions,
        key=lambda s: (priority_map.get(s.pattern_type, 99), -s.improvement_estimate)
    )


# ============================================================================
# Main Analyzer Class
# ============================================================================


class ErrorPatternAnalyzer:
    """Analyze error signals to discover correction patterns.
    
    Automatically detects:
    - Spectral patterns (periodic corrections)
    - Polynomial trends (systematic bias)
    - Autocorrelation structure (recursive patterns)
    - Scale-dependent errors
    
    Suggests correction terms with code snippets.
    
    Parameters
    ----------
    actual : np.ndarray
        Ground truth values.
    predicted : np.ndarray
        Predicted/approximated values.
    x_values : np.ndarray, optional
        X-axis values. If None, uses indices.
    name : str
        Name for this analysis.
        
    Attributes
    ----------
    error : np.ndarray
        Computed error (actual - predicted).
    rmse : float
        Root mean squared error.
    mae : float
        Mean absolute error.
    max_error : float
        Maximum absolute error.
    """
    
    def __init__(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        x_values: Optional[np.ndarray] = None,
        name: str = "Error Analysis"
    ):
        self.actual = np.asarray(actual, dtype=float)
        self.predicted = np.asarray(predicted, dtype=float)
        self.error = self.actual - self.predicted
        self.x_values = x_values if x_values is not None else np.arange(len(actual))
        self.name = name
        
        # Computed patterns
        self.spectral_patterns = None
        self.polynomial_patterns = None
        self.autocorr_patterns = None
        self.scale_patterns = None
        
        # Metrics
        self.rmse = compute_rmse(self.actual, self.predicted)
        self.mae = compute_mae(self.actual, self.predicted)
        self.max_error = compute_max_error(self.actual, self.predicted)
    
    def analyze_all(self, max_corrections: int = 5):
        """Run all pattern detection methods.
        
        Parameters
        ----------
        max_corrections : int
            Maximum number of correction terms to suggest.
            
        Returns
        -------
        ErrorAnalysisReport
            Complete analysis report.
        """
        
        # Run all analyses
        self.spectral_patterns = self.analyze_spectral()
        self.polynomial_patterns = self.analyze_polynomial()
        self.autocorr_patterns = self.analyze_autocorrelation()
        self.scale_patterns = self.analyze_scale_dependence()
        
        # Generate suggestions
        suggestions = []
        
        # Polynomial suggestions
        if self.polynomial_patterns and self.polynomial_patterns.r_squared > 0.1:
            improvement = min(self.polynomial_patterns.r_squared, 0.99)
            suggestions.append(CorrectionSuggestion(
                pattern_type="polynomial",
                description=f"Polynomial trend (degree {self.polynomial_patterns.degree})",
                code_snippet=self.polynomial_patterns.to_correction_code(),
                improvement_estimate=improvement,
                priority=1,
                pattern_data=self.polynomial_patterns
            ))
        
        # Spectral suggestions
        if self.spectral_patterns and self.spectral_patterns.explained_variance > 0.05:
            n_harmonics = len(self.spectral_patterns.frequencies)
            improvement = min(self.spectral_patterns.explained_variance, 0.99)
            suggestions.append(CorrectionSuggestion(
                pattern_type="spectral",
                description=f"Spectral harmonics ({n_harmonics} frequencies)",
                code_snippet=self.spectral_patterns.to_correction_code(),
                improvement_estimate=improvement,
                priority=2,
                pattern_data=self.spectral_patterns
            ))
        
        # Scale suggestions
        if self.scale_patterns and self.scale_patterns.scale_function:
            model = self.scale_patterns.scale_params.get('model', 'unknown')
            r2 = self.scale_patterns.scale_params.get('r2', 0.0)
            improvement = min(r2, 0.99)
            suggestions.append(CorrectionSuggestion(
                pattern_type="scale",
                description=f"Scale-dependent error ({model})",
                code_snippet=self.scale_patterns.to_correction_code(),
                improvement_estimate=improvement,
                priority=3,
                pattern_data=self.scale_patterns
            ))
        
        # Autocorr suggestions
        if self.autocorr_patterns and self.autocorr_patterns.ar_order:
            improvement = 0.3  # Conservative estimate
            suggestions.append(CorrectionSuggestion(
                pattern_type="autocorr",
                description=f"Autocorrelation (AR({self.autocorr_patterns.ar_order}))",
                code_snippet=self.autocorr_patterns.to_correction_code(),
                improvement_estimate=improvement,
                priority=4,
                pattern_data=self.autocorr_patterns
            ))
        
        # Prioritize and limit
        suggestions = prioritize_corrections(suggestions)[:max_corrections]
        
        return ErrorAnalysisReport(
            name=self.name,
            initial_rmse=self.rmse,
            initial_mae=self.mae,
            initial_max_error=self.max_error,
            spectral_pattern=self.spectral_patterns,
            polynomial_pattern=self.polynomial_patterns,
            autocorr_pattern=self.autocorr_patterns,
            scale_pattern=self.scale_patterns,
            suggestions=suggestions
        )
    
    def analyze_spectral(self, n_harmonics: int = 10):
        """Detect periodic patterns using FFT.
        
        Parameters
        ----------
        n_harmonics : int
            Number of harmonics to extract.
            
        Returns
        -------
        SpectralPattern
            Detected spectral pattern.
        """
        return detect_spectral_patterns(self.error, self.x_values, n_harmonics)
    
    def analyze_polynomial(self, max_degree: int = 5):
        """Detect polynomial trends in error.
        
        Parameters
        ----------
        max_degree : int
            Maximum polynomial degree.
            
        Returns
        -------
        PolynomialPattern
            Detected polynomial pattern.
        """
        return detect_polynomial_trend(self.error, self.x_values, max_degree)
    
    def analyze_autocorrelation(self, max_lag: int = 50):
        """Detect autocorrelation structure.
        
        Parameters
        ----------
        max_lag : int
            Maximum lag to analyze.
            
        Returns
        -------
        AutocorrPattern
            Detected autocorrelation pattern.
        """
        return detect_autocorrelation(self.error, max_lag)
    
    def analyze_scale_dependence(self, n_bins: int = 10):
        """Detect if error depends on x-value scale.
        
        Parameters
        ----------
        n_bins : int
            Number of bins for analysis.
            
        Returns
        -------
        ScalePattern
            Detected scale pattern.
        """
        return detect_scale_dependence(self.error, self.x_values, n_bins)
    
    def suggest_corrections(self, top_k: int = 3):
        """Suggest top-k correction terms with code snippets.
        
        Parameters
        ----------
        top_k : int
            Number of suggestions to return.
            
        Returns
        -------
        List[CorrectionSuggestion]
            Top correction suggestions.
        """
        report = self.analyze_all(max_corrections=top_k)
        return report.suggestions
    
    def apply_correction(self, correction):
        """Apply a correction and return new analyzer for residuals.
        
        Parameters
        ----------
        correction : CorrectionSuggestion
            Correction to apply.
            
        Returns
        -------
        ErrorPatternAnalyzer
            New analyzer with updated predictions.
        """
        # Apply correction
        correction_values = correction.apply(self.x_values, self.error)
        new_predicted = self.predicted + correction_values
        
        # Create new analyzer
        return ErrorPatternAnalyzer(
            actual=self.actual,
            predicted=new_predicted,
            x_values=self.x_values,
            name=f"{self.name} + {correction.pattern_type}"
        )
    
    def recursive_refinement(
        self, 
        max_depth: int = 5, 
        improvement_threshold: float = 0.01
    ):
        """Recursively apply corrections until no improvement.
        
        Parameters
        ----------
        max_depth : int
            Maximum refinement depth.
        improvement_threshold : float
            Minimum relative improvement to continue (0-1).
            
        Returns
        -------
        RefinementHistory
            History of refinement process.
        """
        
        initial_rmse = self.rmse
        current_analyzer = self
        corrections_applied = []
        rmse_history = [initial_rmse]
        
        for depth in range(max_depth):
            # Get best correction
            suggestions = current_analyzer.suggest_corrections(top_k=1)
            
            if len(suggestions) == 0:
                break
            
            best_correction = suggestions[0]
            
            # Apply correction
            new_analyzer = current_analyzer.apply_correction(best_correction)
            new_rmse = new_analyzer.rmse
            
            # Check improvement
            relative_improvement = (current_analyzer.rmse - new_rmse) / current_analyzer.rmse
            
            if relative_improvement < improvement_threshold:
                break
            
            # Accept correction
            corrections_applied.append(best_correction)
            rmse_history.append(new_rmse)
            current_analyzer = new_analyzer
        
        final_rmse = rmse_history[-1]
        improvement = (initial_rmse - final_rmse) / initial_rmse if initial_rmse > 0 else 0.0
        
        return RefinementHistory(
            initial_rmse=initial_rmse,
            final_rmse=final_rmse,
            improvement=improvement,
            corrections_applied=corrections_applied,
            rmse_history=rmse_history,
            depth=len(corrections_applied)
        )


# ============================================================================
# Visualization Class
# ============================================================================


class ErrorVisualizer:
    """Visualization tools for error patterns.
    
    Parameters
    ----------
    analyzer : ErrorPatternAnalyzer
        Analyzer instance to visualize.
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def plot_error_signal(self, ax=None):
        """Plot error vs x-values.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.analyzer.x_values, self.analyzer.error, 'o-', alpha=0.6, markersize=3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Error (Actual - Predicted)')
        ax.set_title(f'Error Signal: {self.analyzer.name}')
        ax.grid(True, alpha=0.3)
        
        # Add RMSE annotation
        ax.text(
            0.02, 0.98,
            f'RMSE: {self.analyzer.rmse:.6f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    def plot_spectral_analysis(self, ax=None):
        """Plot power spectrum with dominant frequencies highlighted.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        if self.analyzer.spectral_patterns is None:
            self.analyzer.analyze_spectral()
        
        pattern = self.analyzer.spectral_patterns
        
        # Plot power spectrum
        ax.semilogy(pattern.frequency_axis, pattern.power_spectrum, alpha=0.6)
        
        # Highlight dominant frequencies
        if len(pattern.frequencies) > 0:
            for freq in pattern.frequencies:
                idx = np.argmin(np.abs(pattern.frequency_axis - freq))
                ax.plot(freq, pattern.power_spectrum[idx], 'ro', markersize=8)
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title(f'Spectral Analysis (Explained: {pattern.explained_variance:.1%})')
        ax.grid(True, alpha=0.3)
    
    def plot_autocorrelation(self, ax=None):
        """Plot autocorrelation function.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        if self.analyzer.autocorr_patterns is None:
            self.analyzer.analyze_autocorrelation()
        
        pattern = self.analyzer.autocorr_patterns
        
        # Plot autocorrelation
        ax.stem(pattern.lags, pattern.autocorr, basefmt=' ', use_line_collection=True)
        
        # Significance threshold
        threshold = 2 / np.sqrt(len(self.analyzer.error))
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='95% confidence')
        ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
        
        # Highlight significant lags
        if pattern.significant_lags:
            for lag in pattern.significant_lags[:5]:
                if lag < len(pattern.lags):
                    ax.plot(lag, pattern.autocorr[lag], 'ro', markersize=8)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Autocorrelation Analysis (AR order: {pattern.ar_order})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_scale_dependence(self, ax=None):
        """Plot error vs scale (binned).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        if self.analyzer.scale_patterns is None:
            self.analyzer.analyze_scale_dependence()
        
        pattern = self.analyzer.scale_patterns
        
        # Plot binned errors with error bars
        ax.errorbar(
            pattern.bins,
            pattern.mean_errors,
            yerr=pattern.std_errors,
            fmt='o',
            capsize=5,
            label='Binned data'
        )
        
        # Plot fitted model if available
        if pattern.scale_function is not None:
            x_fit = np.linspace(pattern.bins[0], pattern.bins[-1], 100)
            y_fit = pattern.scale_function(x_fit, *pattern.scale_params['params'])
            model_name = pattern.scale_params.get('model', 'unknown')
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: {model_name}')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('X (binned)')
        ax.set_ylabel('Mean Error')
        ax.set_title('Scale Dependence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_residuals_comparison(self, corrections: List, ax=None):
        """Plot original vs corrected residuals.
        
        Parameters
        ----------
        corrections : List[CorrectionSuggestion]
            Corrections to apply.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        # Original error
        ax.plot(self.analyzer.x_values, self.analyzer.error, 'o-', 
                alpha=0.4, markersize=3, label=f'Original (RMSE: {self.analyzer.rmse:.6f})')
        
        # Apply corrections
        current_analyzer = self.analyzer
        for i, corr in enumerate(corrections[:3]):
            current_analyzer = current_analyzer.apply_correction(corr)
            ax.plot(current_analyzer.x_values, current_analyzer.error, 'o-',
                    alpha=0.6, markersize=3, 
                    label=f'+ {corr.pattern_type} (RMSE: {current_analyzer.rmse:.6f})')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Residual Error')
        ax.set_title('Error Reduction with Corrections')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_full_dashboard(self, figsize=(16, 12)):
        """Create comprehensive dashboard with all visualizations.
        
        Layout:
        - Top left: Error signal
        - Top right: Spectral analysis
        - Middle left: Autocorrelation
        - Middle right: Scale dependence
        - Bottom: Correction suggestions (text)
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Error signal
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_error_signal(ax=ax1)
        
        # Spectral analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_spectral_analysis(ax=ax2)
        
        # Autocorrelation
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_autocorrelation(ax=ax3)
        
        # Scale dependence
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_scale_dependence(ax=ax4)
        
        # Correction suggestions
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Get suggestions
        report = self.analyzer.analyze_all()
        
        # Format suggestions as text
        text_lines = [f"TOP CORRECTION SUGGESTIONS FOR: {self.analyzer.name}\n"]
        text_lines.append(f"Initial RMSE: {report.initial_rmse:.6f}\n\n")
        
        for i, sug in enumerate(report.suggestions[:5], 1):
            text_lines.append(f"{i}. {sug.description}")
            text_lines.append(f"   Estimated improvement: {sug.improvement_estimate:.1%}")
            text_lines.append(f"   Priority: {sug.priority}\n")
        
        ax5.text(
            0.05, 0.95,
            '\n'.join(text_lines),
            transform=ax5.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
        )
        
        plt.suptitle(f'Error Pattern Analysis Dashboard: {self.analyzer.name}', 
                     fontsize=14, fontweight='bold')
        plt.show()
    
    def plot_recursive_refinement(self, history, figsize=(12, 8)):
        """Visualize recursive refinement process.
        
        Shows:
        - RMSE convergence
        - Error distribution at each stage
        - Applied corrections
        
        Parameters
        ----------
        history : RefinementHistory
            Refinement history to visualize.
        figsize : tuple
            Figure size.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # RMSE convergence
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(range(len(history.rmse_history)), history.rmse_history, 'o-', linewidth=2)
        ax1.set_xlabel('Refinement Step')
        ax1.set_ylabel('RMSE')
        ax1.set_title(f'Convergence: {history.improvement:.1%} improvement')
        ax1.grid(True, alpha=0.3)
        
        # Annotate corrections
        for i, corr in enumerate(history.corrections_applied):
            ax1.annotate(
                corr.pattern_type,
                xy=(i+1, history.rmse_history[i+1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
        
        # Correction summary
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        text_lines = ["APPLIED CORRECTIONS:\n"]
        text_lines.append(f"Initial RMSE: {history.initial_rmse:.6f}")
        text_lines.append(f"Final RMSE:   {history.final_rmse:.6f}")
        text_lines.append(f"Improvement:  {history.improvement:.1%}\n")
        
        for i, corr in enumerate(history.corrections_applied, 1):
            text_lines.append(f"{i}. {corr.description}")
            text_lines.append(f"   RMSE after: {history.rmse_history[i]:.6f}\n")
        
        ax2.text(
            0.05, 0.95,
            '\n'.join(text_lines),
            transform=ax2.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
        )
        
        plt.suptitle('Recursive Refinement Analysis', fontsize=14, fontweight='bold')
        plt.show()
