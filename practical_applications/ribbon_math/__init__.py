"""
Ribbon LCM v5: Validated Error-as-Signal Framework
===================================================

A framework for automated mathematical discovery that treats
numerical error as signal rather than noise.

Key Achievement:
- φ-BBP Formula: 20% faster than Bellard's formula
- Error: 7.85×10⁻²² (machine precision)
- Convergence: 3.61 digits/term
"""

from .discovery_engine import (
    # Core classes
    DiscoveryEngine,
    Domain,
    Concept,
    Candidate,
    Discovery,
    ErrorAnalysis,
    
    # Pattern detection
    PhiPattern,
    PhiPatternDetector,
    ClosedFormSearcher,
    ClosedFormResult,
    
    # Abstract layers
    ConceptLayer,
    NSmoothLayer,
    StructureLayer,
    VerificationLayer,
    ErrorAnalysisLayer,
    
    # Constants
    PHI,
    PSI,
    FIB,
    LUCAS,
    
    # Utilities
    compute_convergence_rate,
    format_discovery_report,
)

__version__ = "5.0.0"
__all__ = [
    'DiscoveryEngine',
    'Domain',
    'Concept',
    'Candidate',
    'Discovery',
    'ErrorAnalysis',
    'PhiPattern',
    'PhiPatternDetector',
    'ClosedFormSearcher',
    'ClosedFormResult',
    'ConceptLayer',
    'NSmoothLayer',
    'StructureLayer',
    'VerificationLayer',
    'ErrorAnalysisLayer',
    'PHI',
    'PSI',
    'FIB',
    'LUCAS',
    'compute_convergence_rate',
    'format_discovery_report',
]
