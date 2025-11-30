"""
Ribbon LCM v5 Domains
=====================

Domain implementations for specific problem areas.
"""

from .bbp_domain import (
    BBPDomain,
    UnifiedSeries,
    BBPConceptLayer,
    BBPNSmoothLayer,
    BBPStructureLayer,
    BBPVerificationLayer,
)

from .quadratic_field_domain import (
    QuadraticFieldDomain,
    QuadraticFieldSeries,
    QuadraticFieldConceptLayer,
    QuadraticFieldNSmoothLayer,
    QuadraticFieldStructureLayer,
    QuadraticFieldVerificationLayer,
    # Constants
    D, EPSILON, EPSILON_CONJ, EPSILON_A, EPSILON_B, SQRT_D,
    LOG10_EPSILON, REGULATOR, DIGITS_PER_TERM,
    # Utilities
    print_field_info,
    translate_to_ribbon_speech,
    RIBBON_VOCABULARY,
)

__all__ = [
    # BBP Domain
    'BBPDomain',
    'UnifiedSeries',
    'BBPConceptLayer',
    'BBPNSmoothLayer',
    'BBPStructureLayer',
    'BBPVerificationLayer',
    # Quadratic Field Domain
    'QuadraticFieldDomain',
    'QuadraticFieldSeries',
    'QuadraticFieldConceptLayer',
    'QuadraticFieldNSmoothLayer',
    'QuadraticFieldStructureLayer',
    'QuadraticFieldVerificationLayer',
    'D', 'EPSILON', 'EPSILON_CONJ', 'EPSILON_A', 'EPSILON_B', 'SQRT_D',
    'LOG10_EPSILON', 'REGULATOR', 'DIGITS_PER_TERM',
    'print_field_info',
    'translate_to_ribbon_speech',
    'RIBBON_VOCABULARY',
]
