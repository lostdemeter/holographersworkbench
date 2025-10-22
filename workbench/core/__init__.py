"""
workbench.core
~~~~~~~~~~~~~~

Layer 2: Domain-specific primitives for zeta zeros and crystalline structures.

This layer contains domain primitives that build on Layer 1 functions.
Includes the Gushurst Crystal - a unified number-theoretic crystalline structure.
"""

from .zeta import (
    zetazero,
    zetazero_batch,
    zetazero_range,
    ZetaZeroParameters,
    ZetaFiducials,
)

from .gushurst_crystal import (
    GushurstCrystal,
)

__all__ = [
    # Zeta zero computation
    'zetazero',
    'zetazero_batch',
    'zetazero_range',
    'ZetaZeroParameters',
    'ZetaFiducials',
    
    # Gushurst crystal (unified framework)
    'GushurstCrystal',
]
