"""
workbench.processors
~~~~~~~~~~~~~~~~~~~~

Layer 4: Stateful transformers for spectral scoring, holographic processing,
optimization, compression, encoding, and ergodic analysis.

This layer contains classes that transform data with internal state.
"""

from .spectral import (
    SpectralScorer,
    DiracOperator,
    compute_spectral_scores,
    SpectralConfig,
)

from .holographic import (
    PhaseRetrieval,
    FourPhaseShifting,
    phase_retrieve_hilbert,
    phase_retrieve_gs,
    align_phase,
    holographic_refinement,
    PhaseRetrievalConfig,
)

from .holographic_depth import (
    HolographicDepthExtractor,
    DepthMapStats,
)

from .optimization import (
    SublinearOptimizer,
    SRTCalibrator,
    optimize_sublinear,
)

from .compression import (
    # Fractal peeling
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
    # Holographic compression
    HolographicCompressor,
    compress_image,
    decompress_image,
    CompressionStats,
)

from .encoding import (
    HolographicEncoder,
)

from .ergodic import (
    ErgodicJump,
)

from .adaptive_nonlocality import (
    AdaptiveNonlocalityOptimizer,
    AffinityMetrics,
    DimensionalTrajectory,
)

from .sublinear_qik import (
    SublinearQIK,
    SublinearQIKStats,
)

from .quantum_autoencoder import (
    QuantumAutoencoder,
    QuantumAutoencoderStats,
    HolographicProfile,
)

from .additive_error_stereo import (
    AdditiveErrorStereo,
    StereoStats,
)

__all__ = [
    # Spectral processing
    'SpectralScorer',
    'DiracOperator',
    'compute_spectral_scores',
    'SpectralConfig',
    
    # Holographic processing
    'PhaseRetrieval',
    'FourPhaseShifting',
    'phase_retrieve_hilbert',
    'phase_retrieve_gs',
    'align_phase',
    'holographic_refinement',
    'PhaseRetrievalConfig',
    
    # Holographic depth extraction
    'HolographicDepthExtractor',
    'DepthMapStats',
    
    # Optimization
    'SublinearOptimizer',
    'SRTCalibrator',
    'optimize_sublinear',
    
    # Compression (fractal + holographic)
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
    
    # Encoding
    'HolographicEncoder',
    
    # Ergodic analysis
    'ErgodicJump',
    
    # Adaptive nonlocality
    'AdaptiveNonlocalityOptimizer',
    'AffinityMetrics',
    'DimensionalTrajectory',
    
    # Sublinear QIK
    'SublinearQIK',
    'SublinearQIKStats',
    
    # Quantum autoencoder
    'QuantumAutoencoder',
    'QuantumAutoencoderStats',
    'HolographicProfile',
    
    # Additive error stereo
    'AdditiveErrorStereo',
    'StereoStats',
]
