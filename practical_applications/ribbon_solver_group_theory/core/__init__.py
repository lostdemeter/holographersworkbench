"""
Core - Group-Theoretic Truth Space Structure
=============================================

The core module provides the fundamental mathematical structure:

- **TruthGroup**: The 6D Lie group of truth space
- **GroupElement**: Elements representing mathematical expressions
- **AnchorVector**: 6D coordinates in truth space
- **Subgroup**: Mathematical categories (trigonometric, exponential, algebraic)

This is the foundation that all processors build upon. The core handles:
1. Representation of mathematical truths as group elements
2. Group operations (composition, inverse, conjugation)
3. Distance metrics (geodesics = proof length)
4. Subgroup classification

Processors use the core to:
- Navigate truth space
- Find symmetries (identities)
- Optimize expressions
- Discover formulas

Philosophy:
    The core is STRUCTURE, processors are ACTION.
    Core defines WHERE truths live, processors find WHAT to do with them.
"""

from .group_structure import (
    TruthGroup,
    GroupElement,
    AnchorVector,
    Subgroup,
    Anchor,
    ANCHOR_NAMES,
    ANCHOR_VALUES,
)

__all__ = [
    'TruthGroup',
    'GroupElement', 
    'AnchorVector',
    'Subgroup',
    'Anchor',
    'ANCHOR_NAMES',
    'ANCHOR_VALUES',
]
