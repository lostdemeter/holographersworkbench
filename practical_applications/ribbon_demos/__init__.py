"""
Clock Dimensional Downcasting
=============================

Machine-precision spectral oracle for quantum clock states.

The breakthrough: Instead of diagonalizing 2^n × 2^n matrices,
we compute exact eigenphases in O(log n) time using dimensional
downcasting.

Usage:
    from clock_downcaster import solve_clock_phase, ClockDimensionalDowncaster
    
    # Quick: get the 100th eigenphase
    theta = solve_clock_phase(100)
    
    # Full control
    solver = ClockDimensionalDowncaster()
    result = solver.verify(100)

For O(1) lookup with memoization:
    from clock_downcaster import LazyClockOracle
    
    oracle = LazyClockOracle(max_depth=20)
    theta = oracle.get_phase(1000)

Author: Lesley Gushurst
License: GPLv3
URL: https://github.com/lostdemeter/clock_downcaster
Year: 2025
"""

__version__ = "1.0.0"
__author__ = "Lesley Gushurst"
__license__ = "GPLv3"
__url__ = "https://github.com/lostdemeter/clock_downcaster"

from .clock_solver import ClockDimensionalDowncaster, solve_clock_phase
from .fast_clock_predictor import (
    LazyClockOracle,
    CLOCK_RATIOS_6D,
    CLOCK_RATIOS_12D,
    PHI,
    recursive_theta,
)

__all__ = [
    'ClockDimensionalDowncaster',
    'solve_clock_phase',
    'LazyClockOracle',
    'CLOCK_RATIOS_6D',
    'CLOCK_RATIOS_12D',
    'PHI',
    'recursive_theta',
]
