"""
Holographer's Workbench
=======================

A unified toolkit for holographic signal processing, spectral optimization,
and sublinear algorithms.

Core Modules:
-------------
- spectral: Frequency-domain analysis and zeta-based scoring
- holographic: Phase retrieval, interference patterns, and signal refinement
- optimization: Sublinear algorithms and parameter calibration
- fractal_peeling: Recursive fractal peeling for lossless compression
- holographic_compression: Holographic image compression via harmonic encoding
- fast_zetas: High-performance Riemann zeta zero computation (26× faster)
- time_affinity: Walltime-based parameter optimization
- performance_profiler: Performance profiling and bottleneck identification
- error_pattern_visualizer: Automatic error pattern discovery and correction
- formula_code_generator: Production code generation from formula discoveries
- convergence_analyzer: Convergence analysis and optimal stopping decisions
- utils: Common utilities and helper functions

Philosophy:
-----------
The workbench unifies techniques from:
1. Holography (interference, phase retrieval)
2. Spectral theory (Fourier, zeta zeros, eigenvalues)
3. Optimization (sublinear search, SRT calibration)
4. Signal processing (envelope detection, filtering)

All tools share common patterns:
- Spectral decomposition
- Phase/amplitude separation
- Interference-based enhancement
- Adaptive parameter tuning
"""

__version__ = "0.1.0"

# Core imports
from .spectral import (
    SpectralScorer,
    ZetaFiducials,
    compute_spectral_scores,
)

from .holographic import (
    PhaseRetrieval,
    phase_retrieve_hilbert,
    phase_retrieve_gs,
    align_phase,
    holographic_refinement,
)

from .optimization import (
    SublinearOptimizer,
    SRTCalibrator,
    optimize_sublinear,
)

from .utils import (
    compute_envelope,
    normalize_signal,
    adaptive_blend,
    compute_psnr,
    detect_peaks,
    smooth_signal,
)

from .fractal_peeling import (
    FractalPeeler,
    resfrac_score,
    extract_pattern,
    compress,
    decompress,
    compression_ratio,
    tree_statistics,
    visualize_tree,
    CompressionTree,
    CompressionNode,
    CompressionLeaf,
    PredictorModel,
)

from .holographic_compression import (
    HolographicCompressor,
    compress_image,
    decompress_image,
    CompressionStats,
)

from .fast_zetas import (
    zetazero,
    zetazero_batch,
    zetazero_range,
    ZetaZeroParameters,
)

from .time_affinity import (
    TimeAffinityOptimizer,
    GridSearchTimeAffinity,
    TimeAffinityResult,
    quick_calibrate,
)

from .performance_profiler import (
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

from .error_pattern_visualizer import (
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

from .formula_code_generator import (
    FormulaCodeGenerator,
    CodeValidator,
    CodeOptimizer,
    TestGenerator,
    BenchmarkGenerator,
    ValidationReport,
    ValidationIssue,
)

from .convergence_analyzer import (
    ConvergenceAnalyzer,
    ConvergenceVisualizer,
    ConvergenceModelFitter,
    ConvergenceRate,
    DiminishingReturnsPoint,
    StoppingRecommendation,
    OscillationPattern,
    ConvergenceReport,
)

__all__ = [
    # Spectral
    'SpectralScorer',
    'ZetaFiducials',
    'compute_spectral_scores',
    
    # Holographic
    'PhaseRetrieval',
    'phase_retrieve_hilbert',
    'phase_retrieve_gs',
    'align_phase',
    'holographic_refinement',
    
    # Optimization
    'SublinearOptimizer',
    'SRTCalibrator',
    'optimize_sublinear',
    
    # Utils
    'compute_envelope',
    'normalize_signal',
    'adaptive_blend',
    'compute_psnr',
    'detect_peaks',
    'smooth_signal',
    
    # Fractal Peeling
    'FractalPeeler',
    'resfrac_score',
    'extract_pattern',
    'compress',
    'decompress',
    'compression_ratio',
    'tree_statistics',
    'visualize_tree',
    'CompressionTree',
    'CompressionNode',
    'CompressionLeaf',
    'PredictorModel',
    
    # Holographic Compression
    'HolographicCompressor',
    'compress_image',
    'decompress_image',
    'CompressionStats',
    
    # Fast Zetas
    'zetazero',
    'zetazero_batch',
    'zetazero_range',
    'ZetaZeroParameters',
    
    # Time Affinity
    'TimeAffinityOptimizer',
    'GridSearchTimeAffinity',
    'TimeAffinityResult',
    'quick_calibrate',
    
    # Performance Profiler
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
    
    # Error Pattern Visualizer
    'ErrorPatternAnalyzer',
    'ErrorVisualizer',
    'SpectralPattern',
    'PolynomialPattern',
    'AutocorrPattern',
    'ScalePattern',
    'CorrectionSuggestion',
    'ErrorAnalysisReport',
    'RefinementHistory',
    
    # Formula Code Generator
    'FormulaCodeGenerator',
    'CodeValidator',
    'CodeOptimizer',
    'TestGenerator',
    'BenchmarkGenerator',
    'ValidationReport',
    'ValidationIssue',
    
    # Convergence Analyzer
    'ConvergenceAnalyzer',
    'ConvergenceVisualizer',
    'ConvergenceModelFitter',
    'ConvergenceRate',
    'DiminishingReturnsPoint',
    'StoppingRecommendation',
    'OscillationPattern',
    'ConvergenceReport',
]
