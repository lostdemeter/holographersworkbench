"""
Ribbon Solver - Group Theory Edition
=====================================

A group-theoretic approach to automated mathematical discovery.

This module reimagines ribbon_solver3 through the lens of group theory,
treating the 6D truth space as a Lie group where:

- **Group Elements**: Points in truth space (6D anchor coordinates)
- **Group Operation**: Composition of mathematical transformations
- **Identity**: The origin (all anchors = 0)
- **Inverses**: Inverse transformations (arctan ↔ tan, exp ↔ log)
- **Subgroups**: Categories of truths (trigonometric, exponential, etc.)

Key Insight from Truth Space Papers:
    The 6 anchors (identity, stability, inverse, unity, pattern, growth)
    form a basis for a 6-dimensional Lie algebra. Mathematical truths
    are locations where different group elements map to the same point.

Group Structure:
    - The group is non-abelian (order of operations matters)
    - It has a natural metric (geodesic distance = proof length)
    - Subgroups correspond to mathematical categories
    - Cosets represent equivalence classes of truths

Philosophy:
    - SYMMETRY IS TRUTH: Identities are fixed points under group action
    - PROOF IS PATH: Geodesics through the group manifold
    - ERROR IS SIGNAL: Deviations reveal hidden symmetries

Quick Start:
    from ribbon_solver_group_theory import TruthGroup, discover
    
    # Create the truth group
    group = TruthGroup()
    
    # Find symmetries (identities)
    symmetries = group.find_symmetries("sin²(x) + cos²(x)")
    
    # Discover new truths via group action
    result = discover.find_orbit("exp(x)")

Author: Holographer's Workbench
Version: 1.0.0
Date: December 2025
"""

# Core - Group structure
from .core import (
    TruthGroup,
    GroupElement,
    AnchorVector,
    Subgroup,
    Anchor,
    ANCHOR_NAMES,
    ANCHOR_VALUES,
)

# Processors - Action modules
from .processors import (
    BaseProcessor,
    ProcessorResult,
    CodeOptimizer,
    OptimizationResult,
    IdentityMiner,
    Identity,
    FormulaDiscovery,
    FormulaResult,
    ErrorAnalyzer,
    ErrorPattern,
    SymmetryFinder,
    Symmetry,
)

# Legacy discovery interface (uses processors internally)
from .discover import (
    GroupDiscovery,
    DiscoveryResult,
    discover_symmetries,
    find_orbit,
    optimize_via_conjugation,
    find_formula,
    analyze,
)

# AI Agents for automated truth space operations
from .agents import (
    BaseAgent,
    AgentResult,
    TaskRouter,
    TaskType,
    FormulaParser,
    ParsedFormula,
    RibbonSpeechTranslator,
    RibbonSpeech,
    TruthSpaceNavigator,
    NavigationResult,
    TruthSpaceOrchestrator,
    OrchestratorResult,
)

# Visualizations for truth space navigation
from .visualizations import (
    TruthSpaceVisualizer,
    ProjectionMode,
    CameraView,
    PathAnimator,
    AnimationConfig,
)

__version__ = "1.0.0"
__author__ = "Holographer's Workbench"

__all__ = [
    # Core group structure
    'TruthGroup',
    'GroupElement',
    'AnchorVector',
    'Subgroup',
    'Anchor',
    'ANCHOR_NAMES',
    'ANCHOR_VALUES',
    
    # Processors
    'BaseProcessor',
    'ProcessorResult',
    'CodeOptimizer',
    'OptimizationResult',
    'IdentityMiner',
    'Identity',
    'FormulaDiscovery',
    'FormulaResult',
    'ErrorAnalyzer',
    'ErrorPattern',
    'SymmetryFinder',
    'Symmetry',
    
    # Discovery interface
    'GroupDiscovery',
    'DiscoveryResult',
    'discover_symmetries',
    'find_orbit',
    'optimize_via_conjugation',
    'find_formula',
    'analyze',
    
    # AI Agents
    'BaseAgent',
    'AgentResult',
    'TaskRouter',
    'TaskType',
    'FormulaParser',
    'ParsedFormula',
    'RibbonSpeechTranslator',
    'RibbonSpeech',
    'TruthSpaceNavigator',
    'NavigationResult',
    'TruthSpaceOrchestrator',
    'OrchestratorResult',
    
    # Visualizations
    'TruthSpaceVisualizer',
    'ProjectionMode',
    'CameraView',
    'PathAnimator',
    'AnimationConfig',
]
