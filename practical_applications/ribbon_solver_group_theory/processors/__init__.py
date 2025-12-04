"""
Processors - Action Modules for Truth Space
=============================================

Processors are specialized modules that use the core group structure
to perform specific mathematical tasks:

1. **CodeOptimizer** - Find and apply code optimizations
2. **IdentityMiner** - Discover new mathematical identities  
3. **FormulaDiscovery** - Find formulas for constants (BBP, arctan, etc.)
4. **ErrorAnalyzer** - Use error-as-signal to find corrections
5. **SymmetryFinder** - Find symmetries (fixed points) of expressions

Each processor:
- Takes the TruthGroup as its foundation
- Operates on GroupElements
- Returns structured results
- Can be used independently or composed

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  User Interface                  │
    │         (discover.py, CLI, notebooks)           │
    └─────────────────────────────────────────────────┘
                          │
    ┌─────────────────────────────────────────────────┐
    │                   Processors                     │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
    │  │ Code     │ │ Identity │ │ Formula          │ │
    │  │ Optimizer│ │ Miner    │ │ Discovery        │ │
    │  └──────────┘ └──────────┘ └──────────────────┘ │
    │  ┌──────────┐ ┌──────────┐                      │
    │  │ Error    │ │ Symmetry │                      │
    │  │ Analyzer │ │ Finder   │                      │
    │  └──────────┘ └──────────┘                      │
    └─────────────────────────────────────────────────┘
                          │
    ┌─────────────────────────────────────────────────┐
    │                     Core                         │
    │  TruthGroup, GroupElement, AnchorVector, etc.   │
    └─────────────────────────────────────────────────┘

Usage:
    from ribbon_solver_group_theory.core import TruthGroup
    from ribbon_solver_group_theory.processors import CodeOptimizer
    
    group = TruthGroup()
    optimizer = CodeOptimizer(group)
    result = optimizer.optimize_file("path/to/code.py")
"""

from .base import BaseProcessor, ProcessorResult
from .code_optimizer import CodeOptimizer, OptimizationResult
from .identity_miner import IdentityMiner, Identity
from .formula_discovery import FormulaDiscovery, FormulaResult
from .error_analyzer import ErrorAnalyzer, ErrorPattern
from .symmetry_finder import SymmetryFinder, Symmetry

__all__ = [
    # Base classes
    'BaseProcessor',
    'ProcessorResult',
    
    # Code optimization
    'CodeOptimizer',
    'OptimizationResult',
    
    # Identity mining
    'IdentityMiner', 
    'Identity',
    
    # Formula discovery
    'FormulaDiscovery',
    'FormulaResult',
    
    # Error analysis
    'ErrorAnalyzer',
    'ErrorPattern',
    
    # Symmetry finding
    'SymmetryFinder',
    'Symmetry',
]
