"""
workbench.core
~~~~~~~~~~~~~~

Layer 2: Domain-specific primitives for zeta zeros and quantum modes.

This layer contains domain primitives that build on Layer 1 functions.
"""

from .zeta import (
    zetazero,
    zetazero_batch,
    zetazero_range,
    ZetaZeroParameters,
    ZetaFiducials,
)

from .quantum import (
    QuantumClock,
)

__all__ = [
    # Zeta zero computation
    'zetazero',
    'zetazero_batch',
    'zetazero_range',
    'ZetaZeroParameters',
    'ZetaFiducials',
    
    # Quantum clock
    'QuantumClock',
]
