"""
Visualizations - Truth Space Path Visualization
================================================

Visualize navigation paths through the 6D truth space hyperbigasket.

The truth space has perfect self-similarity, meaning it can be projected to:
- 6D → Full hyperbigasket (abstract representation)
- 5D → Hyperbigasket slice
- 4D → Tetrix (4D Sierpiński)
- 3D → Sierpiński pyramid (tetrahedron)
- 2D → Sierpiński triangle
- 1D → Cantor-like line

Each projection preserves the fractal structure at its dimension level.

Usage:
    from ribbon_solver_group_theory.visualizations import TruthSpaceVisualizer
    
    viz = TruthSpaceVisualizer()
    viz.plot_path(navigation_result)
    viz.show()
"""

from .truth_space_viz import (
    TruthSpaceVisualizer,
    ProjectionMode,
    CameraView,
)

from .path_animator import (
    PathAnimator,
    AnimationConfig,
)

from .interactive_viewer import (
    InteractiveTruthSpaceViewer,
)

__all__ = [
    'TruthSpaceVisualizer',
    'ProjectionMode',
    'CameraView',
    'PathAnimator',
    'AnimationConfig',
    'InteractiveTruthSpaceViewer',
]
