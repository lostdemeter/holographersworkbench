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

from .clock_compiler import (
    ClockResonanceCompiler,
    ClockOracleMixin,
    CompilerAnalysis,
    make_clock_resonant,
)

# Dimensional Bridge (DD-Workbench integration)
try:
    from .dimensional_bridge import (
        ZetaDowncaster,
        ClockSeededPredictor,
        DowncastTSP,
        GushurstDD,
        zetazero_dd,
        zetazero_batch_dd,
        solve_tsp_downcast,
        is_dd_available,
    )
    _DD_EXPORTS = [
        'ZetaDowncaster',
        'ClockSeededPredictor', 
        'DowncastTSP',
        'GushurstDD',
        'zetazero_dd',
        'zetazero_batch_dd',
        'solve_tsp_downcast',
        'is_dd_available',
    ]
except ImportError:
    _DD_EXPORTS = []

__all__ = [
    # Zeta zero computation
    'zetazero',
    'zetazero_batch',
    'zetazero_range',
    'ZetaZeroParameters',
    'ZetaFiducials',
    
    # Gushurst crystal (unified framework)
    'GushurstCrystal',
    
    # Clock Resonance Compiler
    'ClockResonanceCompiler',
    'ClockOracleMixin',
    'CompilerAnalysis',
    'make_clock_resonant',
] + _DD_EXPORTS
