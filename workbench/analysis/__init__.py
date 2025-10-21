"""
workbench.analysis
~~~~~~~~~~~~~~~~~~

Layer 3: Read-only analyzers for performance profiling, error pattern detection,
convergence analysis, and time affinity optimization.

This layer contains diagnostic and analysis tools that read data without modifying it.
"""

from .performance import (
    PerformanceProfiler,
    ProfileResult,
    IterationProfile,
    BatchProfile,
    BottleneckReport,
    profile,
    ProfileContext,
    compare_profiles,
    estimate_complexity,
    format_time,
    format_memory,
)

from .errors import (
    ErrorPatternAnalyzer,
    ErrorVisualizer,
    SpectralPattern,
    PolynomialPattern,
    AutocorrPattern,
    ScalePattern,
    CorrectionSuggestion,
    ErrorAnalysisReport,
    RefinementHistory,
)

from .convergence import (
    ConvergenceAnalyzer,
    ConvergenceVisualizer,
    ConvergenceModelFitter,
    ConvergenceRate,
    DiminishingReturnsPoint,
    StoppingRecommendation,
    OscillationPattern,
    ConvergenceReport,
)

from .affinity import (
    TimeAffinityOptimizer,
    GridSearchTimeAffinity,
    TimeAffinityResult,
    quick_calibrate,
)

__all__ = [
    # Performance profiling
    'PerformanceProfiler',
    'ProfileResult',
    'IterationProfile',
    'BatchProfile',
    'BottleneckReport',
    'profile',
    'ProfileContext',
    'compare_profiles',
    'estimate_complexity',
    'format_time',
    'format_memory',
    
    # Error pattern analysis
    'ErrorPatternAnalyzer',
    'ErrorVisualizer',
    'SpectralPattern',
    'PolynomialPattern',
    'AutocorrPattern',
    'ScalePattern',
    'CorrectionSuggestion',
    'ErrorAnalysisReport',
    'RefinementHistory',
    
    # Convergence analysis
    'ConvergenceAnalyzer',
    'ConvergenceVisualizer',
    'ConvergenceModelFitter',
    'ConvergenceRate',
    'DiminishingReturnsPoint',
    'StoppingRecommendation',
    'OscillationPattern',
    'ConvergenceReport',
    
    # Time affinity optimization
    'TimeAffinityOptimizer',
    'GridSearchTimeAffinity',
    'TimeAffinityResult',
    'quick_calibrate',
]
