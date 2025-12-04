"""
Truth Structure Discovery Module

Discovers mathematical rules governing truth space geometry.

Inspired by Ribbon LCM v4's "Error as Signal" paradigm:
- Deviations from simple patterns contain hidden structure
- φ^(-k) patterns appear in coordinate deviations
- Fibonacci recurrence in sorted coordinates
- Self-similarity across scales

Usage:
    python run_discovery.py --constraint golden --samples 5000
    python run_discovery.py --constraint balanced --symmetries
    python run_discovery.py --visualize --save output.png
"""

from .truth_structure_discovery import (
    TruthStructureDiscovery,
    DiscoveryConfig,
    DiscoveredRelation,
    ErrorAsSignalAnalyzer,
    demo_discovery,
)

from .ribbon_constraint_learner import (
    RibbonConstraintLearner,
    SolverObservation,
    TransitionRule,
)

from .structural_compression import (
    FibonacciRecurrenceEncoder,
    GoldenQuantizer,
    StructuralCompressor,
    CompressionStats,
)

__all__ = [
    # Core discovery
    'TruthStructureDiscovery',
    'DiscoveryConfig', 
    'DiscoveredRelation',
    'demo_discovery',
    
    # Error-as-signal analysis
    'ErrorAsSignalAnalyzer',
    
    # Ribbon solver integration
    'RibbonConstraintLearner',
    'SolverObservation',
    'TransitionRule',
    
    # Structural compression
    'FibonacciRecurrenceEncoder',
    'GoldenQuantizer',
    'StructuralCompressor',
    'CompressionStats',
]
