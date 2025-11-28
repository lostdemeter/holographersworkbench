"""
Dimensional Downcasting for Riemann Zeta Zeros
===============================================

Machine-precision computation of Riemann zeta zeros using pure mathematics.

Key Result: <10^-14 accuracy with no training required.

Main Classes:
    DimensionalDowncaster: The main solver achieving machine precision
    RamanujanPredictor: Fast O(1) initial guess
    
Example:
    >>> from src.solver import DimensionalDowncaster
    >>> solver = DimensionalDowncaster()
    >>> t_100 = solver.solve(100)
    >>> print(f"{t_100:.15f}")
    236.524229665816193
"""

from .solver import DimensionalDowncaster
from .predictors import RamanujanPredictor, GeometricPredictor, ClockSeededPredictor

__version__ = "1.1.0"
__author__ = "Dimensional Downcasting Research + Holographer's Workbench"
__license__ = "GPL-3.0"

__all__ = [
    "DimensionalDowncaster",
    "RamanujanPredictor", 
    "GeometricPredictor",
    "ClockSeededPredictor",
]
