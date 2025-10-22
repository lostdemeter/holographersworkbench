"""
Holographer's Workbench
=======================

A unified toolkit for holographic signal processing, spectral optimization,
and sublinear algorithms.

Architecture:
-------------
The workbench is organized into 5 layers with unidirectional dependencies:

Layer 5: generation/     (Generates artifacts: code, files)
    ↓
Layer 4: processors/     (Stateful transformers: scorers, optimizers, compressors)
    ↓
Layer 3: analysis/       (Read-only analyzers: profilers, pattern detectors)
    ↓
Layer 2: core/           (Domain primitives: zeta zeros, quantum modes)
    ↓
Layer 1: primitives/     (Pure utility functions: signal processing, FFT, phase math)

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

Usage:
------
    # High-level convenience imports (recommended for most users)
    from workbench import SpectralScorer, SublinearOptimizer
    
    # Explicit layer imports (recommended for library developers)
    from workbench.processors.spectral import SpectralScorer
    from workbench.primitives.signal import normalize
"""

__version__ = "0.1.0"

# Layer 1: Primitives (pure functions)
from .primitives import signal, frequency, phase, kernels

# Layer 2: Core (domain primitives)
from .core import (
    zetazero,
    zetazero_batch,
    zetazero_range,
    ZetaZeroParameters,
    ZetaFiducials,
    GushurstCrystal,
)

# Layer 3: Analysis (read-only analyzers)
from .analysis import (
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
    ErrorPatternAnalyzer,
    ErrorVisualizer,
    SpectralPattern,
    PolynomialPattern,
    AutocorrPattern,
    ScalePattern,
    CorrectionSuggestion,
    ErrorAnalysisReport,
    RefinementHistory,
    ConvergenceAnalyzer,
    ConvergenceVisualizer,
    ConvergenceModelFitter,
    ConvergenceRate,
    DiminishingReturnsPoint,
    StoppingRecommendation,
    OscillationPattern,
    ConvergenceReport,
    TimeAffinityOptimizer,
    GridSearchTimeAffinity,
    TimeAffinityResult,
    quick_calibrate,
)

# Layer 4: Processors (stateful transformers)
from .processors import (
    SpectralScorer,
    DiracOperator,
    compute_spectral_scores,
    SpectralConfig,
    PhaseRetrieval,
    FourPhaseShifting,
    phase_retrieve_hilbert,
    phase_retrieve_gs,
    align_phase,
    holographic_refinement,
    PhaseRetrievalConfig,
    SublinearOptimizer,
    SRTCalibrator,
    optimize_sublinear,
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
    HolographicCompressor,
    compress_image,
    decompress_image,
    CompressionStats,
    HolographicEncoder,
    ErgodicJump,
)

# Layer 5: Generation (artifact generators)
from .generation import (
    FormulaCodeGenerator,
    CodeValidator,
    CodeOptimizer,
    TestGenerator,
    BenchmarkGenerator,
    ValidationReport,
    ValidationIssue,
)

__all__ = [
    # Version
    '__version__',
    
    # Layer 1: Primitives (modules)
    'signal',
    'frequency',
    'phase',
    'kernels',
    
    # Layer 2: Core
    'zetazero',
    'zetazero_batch',
    'zetazero_range',
    'ZetaZeroParameters',
    'ZetaFiducials',
    'GushurstCrystal',
    
    # Layer 3: Analysis
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
    'ErrorPatternAnalyzer',
    'ErrorVisualizer',
    'SpectralPattern',
    'PolynomialPattern',
    'AutocorrPattern',
    'ScalePattern',
    'CorrectionSuggestion',
    'ErrorAnalysisReport',
    'RefinementHistory',
    'ConvergenceAnalyzer',
    'ConvergenceVisualizer',
    'ConvergenceModelFitter',
    'ConvergenceRate',
    'DiminishingReturnsPoint',
    'StoppingRecommendation',
    'OscillationPattern',
    'ConvergenceReport',
    'TimeAffinityOptimizer',
    'GridSearchTimeAffinity',
    'TimeAffinityResult',
    'quick_calibrate',
    
    # Layer 4: Processors
    'SpectralScorer',
    'DiracOperator',
    'compute_spectral_scores',
    'SpectralConfig',
    'PhaseRetrieval',
    'FourPhaseShifting',
    'phase_retrieve_hilbert',
    'phase_retrieve_gs',
    'align_phase',
    'holographic_refinement',
    'PhaseRetrievalConfig',
    'SublinearOptimizer',
    'SRTCalibrator',
    'optimize_sublinear',
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
    'HolographicCompressor',
    'compress_image',
    'decompress_image',
    'CompressionStats',
    'HolographicEncoder',
    'ErgodicJump',
    
    # Layer 5: Generation
    'FormulaCodeGenerator',
    'CodeValidator',
    'CodeOptimizer',
    'TestGenerator',
    'BenchmarkGenerator',
    'ValidationReport',
    'ValidationIssue',
]
