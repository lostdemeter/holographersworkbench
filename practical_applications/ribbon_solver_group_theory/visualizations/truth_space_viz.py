"""
Truth Space Visualizer
======================

Visualize paths through the 6D truth space hyperbigasket.

The hyperbigasket has perfect symmetry, allowing projection to any lower dimension
while preserving fractal structure.

Projection Hierarchy:
    6D Hyperbigasket
         ↓ (project out one dimension)
    5D Hyperbigasket
         ↓
    4D Tetrix (4-simplex Sierpiński)
         ↓
    3D Sierpiński Pyramid
         ↓
    2D Sierpiński Triangle
         ↓
    1D Cantor-like Line
"""

import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors


class ProjectionMode(Enum):
    """Projection modes for truth space visualization."""
    FULL_6D = auto()      # Abstract 6D representation (parallel coords or radial)
    SLICE_5D = auto()     # 5D slice (multiple 3D views)
    TETRIX_4D = auto()    # 4D tetrix projection
    PYRAMID_3D = auto()   # 3D Sierpiński pyramid
    TRIANGLE_2D = auto()  # 2D Sierpiński triangle
    LINE_1D = auto()      # 1D Cantor-like projection


class CameraView(Enum):
    """Preset camera views for 3D visualization."""
    FRONT = (0, 0)
    TOP = (90, 0)
    SIDE = (0, 90)
    ISOMETRIC = (35.264, 45)
    GOLDEN = (31.7, 58.3)  # Golden angle view
    CUSTOM = None


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    figsize: Tuple[int, int] = (12, 10)
    dpi: int = 100
    background_color: str = '#0a0a0a'
    path_color: str = '#00ff88'
    point_color: str = '#ff6600'
    anchor_colors: Dict[str, str] = None
    show_grid: bool = True
    show_anchors: bool = True
    show_labels: bool = True
    alpha: float = 0.8
    
    # Geometry overlay settings
    geometry_alpha: float = 0.15  # Opacity of Sierpiński structure (0-1)
    geometry_color: str = '#4488ff'  # Color of geometry overlay
    geometry_linewidth: float = 0.5  # Line width for geometry
    geometry_depth: int = 3  # Recursion depth for fractal
    show_geometry: bool = True  # Whether to show geometry overlay
    geometry_fill: bool = True  # Whether to fill faces (3D only)
    geometry_fill_alpha: float = 0.05  # Fill opacity (very transparent)
    
    def __post_init__(self):
        if self.anchor_colors is None:
            self.anchor_colors = {
                'IDENTITY': '#ffffff',
                'STABILITY': '#00ffff',
                'INVERSE': '#ff00ff',
                'UNITY': '#ffff00',
                'PATTERN': '#ff0088',
                'GROWTH': '#00ff00',
            }


class TruthSpaceVisualizer:
    """
    Visualize paths through 6D truth space.
    
    The truth space is a 6D hyperbigasket with perfect self-similarity.
    This visualizer projects paths to viewable dimensions.
    """
    
    # 6D anchor positions (vertices of 6-simplex)
    ANCHOR_POSITIONS_6D = {
        'IDENTITY': np.array([1, 0, 0, 0, 0, 0]),
        'STABILITY': np.array([0, 1, 0, 0, 0, 0]),
        'INVERSE': np.array([0, 0, 1, 0, 0, 0]),
        'UNITY': np.array([0, 0, 0, 1, 0, 0]),
        'PATTERN': np.array([0, 0, 0, 0, 1, 0]),
        'GROWTH': np.array([0, 0, 0, 0, 0, 1]),
    }
    
    # 3D projection of 6-simplex (for pyramid view)
    ANCHOR_POSITIONS_3D = {
        'IDENTITY': np.array([0, 0, 1]),
        'STABILITY': np.array([0.943, 0, -0.333]),
        'INVERSE': np.array([-0.471, 0.816, -0.333]),
        'UNITY': np.array([-0.471, -0.816, -0.333]),
        'PATTERN': np.array([0.471, 0.816, -0.333]),
        'GROWTH': np.array([0.471, -0.816, -0.333]),
    }
    
    # 2D projection (for triangle view)
    ANCHOR_POSITIONS_2D = {
        'IDENTITY': np.array([0.5, 0.866]),
        'STABILITY': np.array([0, 0]),
        'INVERSE': np.array([1, 0]),
        'UNITY': np.array([0.25, 0.433]),
        'PATTERN': np.array([0.75, 0.433]),
        'GROWTH': np.array([0.5, 0]),
    }
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.fig = None
        self.ax = None
        self.mode = ProjectionMode.PYRAMID_3D
        self.camera_view = CameraView.GOLDEN
        self._paths = []
        self._points = []
        self._zoom_level = 1.0
        self._base_limit = 1.5  # Base axis limit
    
    def set_mode(self, mode: ProjectionMode):
        """Set the projection mode."""
        self.mode = mode
        return self
    
    def set_camera(self, view: CameraView, elev: float = None, azim: float = None):
        """Set camera view."""
        self.camera_view = view
        if view == CameraView.CUSTOM and elev is not None and azim is not None:
            self._custom_view = (elev, azim)
        return self
    
    def set_geometry_opacity(self, edge_alpha: float = None, fill_alpha: float = None):
        """
        Set the opacity of the geometry overlay.
        
        Args:
            edge_alpha: Opacity of edges/wireframe (0-1), default 0.15
            fill_alpha: Opacity of filled faces (0-1), default 0.05
        
        Example:
            viz.set_geometry_opacity(0.3, 0.1)  # More visible geometry
            viz.set_geometry_opacity(0.05, 0.02)  # Very subtle
        """
        if edge_alpha is not None:
            self.config.geometry_alpha = max(0, min(1, edge_alpha))
        if fill_alpha is not None:
            self.config.geometry_fill_alpha = max(0, min(1, fill_alpha))
        return self
    
    def set_geometry_style(self, color: str = None, linewidth: float = None, 
                           depth: int = None, show_fill: bool = None):
        """
        Set the style of the geometry overlay.
        
        Args:
            color: Color of geometry (hex string)
            linewidth: Line width for edges
            depth: Recursion depth for fractal (higher = more detail)
            show_fill: Whether to fill faces
        """
        if color is not None:
            self.config.geometry_color = color
        if linewidth is not None:
            self.config.geometry_linewidth = linewidth
        if depth is not None:
            self.config.geometry_depth = depth
        if show_fill is not None:
            self.config.geometry_fill = show_fill
        return self
    
    def hide_geometry(self):
        """Hide the geometry overlay."""
        self.config.show_geometry = False
        return self
    
    def show_geometry(self):
        """Show the geometry overlay."""
        self.config.show_geometry = True
        return self
    
    def _setup_scroll_zoom(self):
        """Setup scroll wheel zoom for 3D view."""
        if self.fig is None:
            return
        
        def on_scroll(event):
            if event.inaxes != self.ax:
                return
            
            # Zoom factor
            if event.button == 'up':
                self._zoom_level *= 0.9  # Zoom in
            elif event.button == 'down':
                self._zoom_level *= 1.1  # Zoom out
            
            # Clamp zoom level
            self._zoom_level = max(0.2, min(5.0, self._zoom_level))
            
            # Apply zoom
            self._apply_zoom()
            self.fig.canvas.draw_idle()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    def _apply_zoom(self):
        """Apply current zoom level to axes."""
        if self.ax is None:
            return
        
        limit = self._base_limit * self._zoom_level
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
    
    def _setup_fixed_axes(self):
        """Setup fixed axes that don't clip with viewing angle."""
        if self.ax is None:
            return
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Apply initial zoom
        self._apply_zoom()
        
        # Remove the default panes (they cause clipping issues)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Make pane edges invisible
        self.ax.xaxis.pane.set_edgecolor('none')
        self.ax.yaxis.pane.set_edgecolor('none')
        self.ax.zaxis.pane.set_edgecolor('none')
        
        # Draw custom grid that stays consistent
        self._draw_fixed_grid()
    
    def _draw_fixed_grid(self):
        """Draw a fixed grid that doesn't change with viewing angle."""
        limit = self._base_limit * self._zoom_level
        
        # Draw axis lines through origin
        axis_color = '#444444'
        axis_alpha = 0.6
        
        # X axis
        self.ax.plot([-limit, limit], [0, 0], [0, 0], 
                    c=axis_color, linewidth=1, alpha=axis_alpha)
        # Y axis
        self.ax.plot([0, 0], [-limit, limit], [0, 0], 
                    c=axis_color, linewidth=1, alpha=axis_alpha)
        # Z axis
        self.ax.plot([0, 0], [0, 0], [-limit, limit], 
                    c=axis_color, linewidth=1, alpha=axis_alpha)
        
        # Draw grid lines on each plane (subtle)
        grid_color = '#333333'
        grid_alpha = 0.3
        n_lines = 5
        
        for i in np.linspace(-limit, limit, n_lines):
            if abs(i) < 0.01:  # Skip center line (already drawn)
                continue
            # XY plane (z=0)
            self.ax.plot([i, i], [-limit, limit], [0, 0], 
                        c=grid_color, linewidth=0.3, alpha=grid_alpha)
            self.ax.plot([-limit, limit], [i, i], [0, 0], 
                        c=grid_color, linewidth=0.3, alpha=grid_alpha)
    
    def plot_path(self, navigation_result: Dict, label: str = None):
        """
        Plot a navigation path through truth space.
        
        Args:
            navigation_result: Result from TruthSpaceNavigator
            label: Optional label for the path
        """
        # Extract positions from navigation result
        positions = []
        
        # Start position
        start = navigation_result.get('start', {})
        if 'position' in start:
            positions.append(self._dict_to_vector(start['position']))
        
        # Visited positions (if available)
        for find in navigation_result.get('interesting_finds', []):
            if 'location' in find and hasattr(find['location'], 'position'):
                positions.append(self._dict_to_vector(find['location'].position))
        
        if positions:
            self._paths.append({
                'positions': positions,
                'label': label or f"Path {len(self._paths) + 1}",
            })
        
        return self
    
    def add_point(self, position: Dict[str, float], label: str = None, 
                  color: str = None):
        """Add a single point to the visualization."""
        vec = self._dict_to_vector(position)
        self._points.append({
            'position': vec,
            'label': label,
            'color': color or self.config.point_color,
        })
        return self
    
    def _dict_to_vector(self, pos_dict: Dict[str, float]) -> np.ndarray:
        """Convert position dict to 6D vector."""
        anchors = ['IDENTITY', 'STABILITY', 'INVERSE', 'UNITY', 'PATTERN', 'GROWTH']
        return np.array([pos_dict.get(a, 0.0) for a in anchors])
    
    def _project_to_3d(self, pos_6d: np.ndarray) -> np.ndarray:
        """Project 6D position to 3D using anchor positions."""
        result = np.zeros(3)
        for i, anchor in enumerate(['IDENTITY', 'STABILITY', 'INVERSE', 
                                    'UNITY', 'PATTERN', 'GROWTH']):
            result += pos_6d[i] * self.ANCHOR_POSITIONS_3D[anchor]
        return result
    
    def _project_to_2d(self, pos_6d: np.ndarray) -> np.ndarray:
        """Project 6D position to 2D using anchor positions."""
        result = np.zeros(2)
        for i, anchor in enumerate(['IDENTITY', 'STABILITY', 'INVERSE', 
                                    'UNITY', 'PATTERN', 'GROWTH']):
            result += pos_6d[i] * self.ANCHOR_POSITIONS_2D[anchor]
        return result
    
    def _project_to_1d(self, pos_6d: np.ndarray) -> float:
        """Project 6D position to 1D (weighted sum)."""
        weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return np.dot(pos_6d, weights)
    
    def render(self):
        """Render the visualization."""
        if self.mode == ProjectionMode.PYRAMID_3D:
            self._render_3d()
        elif self.mode == ProjectionMode.TRIANGLE_2D:
            self._render_2d()
        elif self.mode == ProjectionMode.LINE_1D:
            self._render_1d()
        elif self.mode == ProjectionMode.FULL_6D:
            self._render_6d_parallel()
        elif self.mode == ProjectionMode.TETRIX_4D:
            self._render_4d_tetrix()
        else:
            self._render_3d()  # Default
        
        return self
    
    def _render_3d(self):
        """Render 3D Sierpiński pyramid view."""
        self.fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set background
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Draw Sierpiński pyramid structure
        if self.config.show_grid and self.config.show_geometry:
            self._draw_sierpinski_pyramid()
        
        # Draw anchor points
        if self.config.show_anchors:
            for anchor, pos in self.ANCHOR_POSITIONS_3D.items():
                color = self.config.anchor_colors.get(anchor, '#ffffff')
                self.ax.scatter(*pos, c=color, s=100, marker='o', alpha=0.9)
                if self.config.show_labels:
                    self.ax.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, 
                               anchor[:3], color=color, fontsize=8)
        
        # Draw paths
        for path_data in self._paths:
            positions_3d = [self._project_to_3d(p) for p in path_data['positions']]
            if len(positions_3d) > 1:
                xs, ys, zs = zip(*positions_3d)
                self.ax.plot(xs, ys, zs, c=self.config.path_color, 
                           linewidth=2, alpha=self.config.alpha,
                           label=path_data['label'])
                # Mark points
                self.ax.scatter(xs, ys, zs, c=self.config.path_color, 
                              s=50, marker='o')
        
        # Draw individual points
        for point in self._points:
            pos_3d = self._project_to_3d(point['position'])
            self.ax.scatter(*pos_3d, c=point['color'], s=80, marker='*')
            if point['label'] and self.config.show_labels:
                self.ax.text(*pos_3d, f"  {point['label']}", 
                           color=point['color'], fontsize=8)
        
        # Set camera view
        self._apply_camera_view()
        
        # Setup fixed axes (consistent regardless of viewing angle)
        self._setup_fixed_axes()
        
        # Setup scroll zoom
        self._setup_scroll_zoom()
        
        # Labels
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        self.ax.set_title('Truth Space - 3D Sierpiński Pyramid Projection', 
                         color='white', fontsize=12)
        
        # Style
        self.ax.tick_params(colors='white')
        
        if self._paths:
            self.ax.legend(facecolor='#222222', edgecolor='white', 
                          labelcolor='white')
    
    def _draw_sierpinski_pyramid(self, depth: int = None):
        """Draw Sierpiński pyramid with configurable opacity."""
        if depth is None:
            depth = self.config.geometry_depth
        
        # Use first 4 anchors for tetrahedron base
        vertices = np.array(list(self.ANCHOR_POSITIONS_3D.values()))[:4]
        self._draw_sierpinski_recursive(vertices, depth)
    
    def _draw_sierpinski_recursive(self, vertices: np.ndarray, depth: int):
        """Recursively draw Sierpiński structure with filled faces."""
        if depth == 0:
            # Draw tetrahedron edges
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            for i, j in edges:
                self.ax.plot([vertices[i,0], vertices[j,0]],
                           [vertices[i,1], vertices[j,1]],
                           [vertices[i,2], vertices[j,2]],
                           c=self.config.geometry_color, 
                           linewidth=self.config.geometry_linewidth, 
                           alpha=self.config.geometry_alpha)
            
            # Draw filled faces if enabled
            if self.config.geometry_fill:
                faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                for face in faces:
                    verts = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                    poly = Poly3DCollection([verts], alpha=self.config.geometry_fill_alpha)
                    poly.set_facecolor(self.config.geometry_color)
                    poly.set_edgecolor('none')
                    self.ax.add_collection3d(poly)
            return
        
        # Compute midpoints
        midpoints = []
        for i in range(4):
            for j in range(i+1, 4):
                midpoints.append((vertices[i] + vertices[j]) / 2)
        
        # Create 4 sub-tetrahedra (corner tetrahedra)
        for i in range(4):
            sub_verts = [vertices[i]]
            for j in range(4):
                if j != i:
                    sub_verts.append((vertices[i] + vertices[j]) / 2)
            self._draw_sierpinski_recursive(np.array(sub_verts), depth - 1)
    
    def _render_2d(self):
        """Render 2D Sierpiński triangle view."""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize, 
                                         dpi=self.config.dpi)
        
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Draw Sierpiński triangle
        if self.config.show_grid:
            self._draw_sierpinski_triangle(depth=5)
        
        # Draw anchor points
        if self.config.show_anchors:
            for anchor, pos in self.ANCHOR_POSITIONS_2D.items():
                color = self.config.anchor_colors.get(anchor, '#ffffff')
                self.ax.scatter(*pos, c=color, s=100, marker='o', alpha=0.9)
                if self.config.show_labels:
                    self.ax.annotate(anchor[:3], pos, color=color, fontsize=8,
                                   xytext=(5, 5), textcoords='offset points')
        
        # Draw paths
        for path_data in self._paths:
            positions_2d = [self._project_to_2d(p) for p in path_data['positions']]
            if len(positions_2d) > 1:
                xs, ys = zip(*positions_2d)
                self.ax.plot(xs, ys, c=self.config.path_color, 
                           linewidth=2, alpha=self.config.alpha,
                           label=path_data['label'])
                self.ax.scatter(xs, ys, c=self.config.path_color, s=50)
        
        # Draw points
        for point in self._points:
            pos_2d = self._project_to_2d(point['position'])
            self.ax.scatter(*pos_2d, c=point['color'], s=80, marker='*')
            if point['label'] and self.config.show_labels:
                self.ax.annotate(point['label'], pos_2d, color=point['color'])
        
        self.ax.set_aspect('equal')
        self.ax.set_title('Truth Space - 2D Sierpiński Triangle Projection',
                         color='white', fontsize=12)
        self.ax.tick_params(colors='white')
        
        if self._paths:
            self.ax.legend(facecolor='#222222', edgecolor='white',
                          labelcolor='white')
    
    def _draw_sierpinski_triangle(self, depth: int = None):
        """Draw Sierpiński triangle with configurable opacity."""
        if depth is None:
            depth = self.config.geometry_depth + 2  # 2D needs more depth to look good
        vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]])
        self._draw_sierpinski_triangle_recursive(vertices, depth)
    
    def _draw_sierpinski_triangle_recursive(self, vertices: np.ndarray, depth: int):
        """Recursively draw Sierpiński triangle with configurable opacity."""
        if depth == 0:
            # Draw filled triangle if enabled
            if self.config.geometry_fill:
                triangle = plt.Polygon(vertices, fill=True, 
                                      facecolor=self.config.geometry_color,
                                      edgecolor=self.config.geometry_color,
                                      linewidth=self.config.geometry_linewidth,
                                      alpha=self.config.geometry_fill_alpha)
            else:
                triangle = plt.Polygon(vertices, fill=False, 
                                      edgecolor=self.config.geometry_color, 
                                      linewidth=self.config.geometry_linewidth, 
                                      alpha=self.config.geometry_alpha)
            self.ax.add_patch(triangle)
            return
        
        # Midpoints
        mid01 = (vertices[0] + vertices[1]) / 2
        mid12 = (vertices[1] + vertices[2]) / 2
        mid02 = (vertices[0] + vertices[2]) / 2
        
        # Three sub-triangles (skip middle)
        self._draw_sierpinski_triangle_recursive(
            np.array([vertices[0], mid01, mid02]), depth - 1)
        self._draw_sierpinski_triangle_recursive(
            np.array([mid01, vertices[1], mid12]), depth - 1)
        self._draw_sierpinski_triangle_recursive(
            np.array([mid02, mid12, vertices[2]]), depth - 1)
    
    def _render_1d(self):
        """Render 1D Cantor-like projection."""
        self.fig, self.ax = plt.subplots(figsize=(self.config.figsize[0], 3),
                                         dpi=self.config.dpi)
        
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Draw Cantor set
        if self.config.show_grid:
            self._draw_cantor_set(depth=6)
        
        # Draw paths as points on line
        y_offset = 0.5
        for path_data in self._paths:
            positions_1d = [self._project_to_1d(p) for p in path_data['positions']]
            ys = [y_offset] * len(positions_1d)
            self.ax.scatter(positions_1d, ys, c=self.config.path_color, 
                          s=100, marker='|', label=path_data['label'])
            # Connect with line
            if len(positions_1d) > 1:
                self.ax.plot(positions_1d, ys, c=self.config.path_color,
                           linewidth=2, alpha=0.5)
        
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Truth Space - 1D Cantor Projection', color='white')
        self.ax.tick_params(colors='white')
        self.ax.set_yticks([])
    
    def _draw_cantor_set(self, depth: int = 6):
        """Draw Cantor set."""
        self._draw_cantor_recursive(0, 1, depth, 0.1)
    
    def _draw_cantor_recursive(self, left: float, right: float, 
                                depth: int, y: float):
        """Recursively draw Cantor set."""
        if depth == 0:
            self.ax.plot([left, right], [y, y], c='#444444', linewidth=2)
            return
        
        third = (right - left) / 3
        self._draw_cantor_recursive(left, left + third, depth - 1, y + 0.05)
        self._draw_cantor_recursive(right - third, right, depth - 1, y + 0.05)
    
    def _render_6d_parallel(self):
        """Render 6D using parallel coordinates."""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize,
                                         dpi=self.config.dpi)
        
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        anchors = ['IDENTITY', 'STABILITY', 'INVERSE', 'UNITY', 'PATTERN', 'GROWTH']
        x_positions = range(6)
        
        # Draw anchor axes
        for i, anchor in enumerate(anchors):
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            self.ax.axvline(x=i, color=color, alpha=0.3, linewidth=2)
            self.ax.text(i, -0.15, anchor[:3], color=color, 
                        ha='center', fontsize=10)
        
        # Draw paths
        for path_data in self._paths:
            for pos in path_data['positions']:
                self.ax.plot(x_positions, pos, c=self.config.path_color,
                           linewidth=2, alpha=self.config.alpha, marker='o')
        
        self.ax.set_xlim(-0.5, 5.5)
        self.ax.set_ylim(-0.2, 1.2)
        self.ax.set_title('Truth Space - 6D Parallel Coordinates', color='white')
        self.ax.tick_params(colors='white')
    
    def _render_4d_tetrix(self):
        """Render 4D tetrix using stereographic projection."""
        # Use 3D view with color as 4th dimension
        self._render_3d()
        self.ax.set_title('Truth Space - 4D Tetrix (color = 4th dim)', color='white')
    
    def _apply_camera_view(self):
        """Apply camera view to 3D axes."""
        if self.camera_view == CameraView.CUSTOM:
            elev, azim = getattr(self, '_custom_view', (30, 45))
        else:
            elev, azim = self.camera_view.value or (30, 45)
        
        self.ax.view_init(elev=elev, azim=azim)
    
    def show(self):
        """Display the visualization."""
        if self.fig is None:
            self.render()
        plt.tight_layout()
        plt.show()
    
    def save(self, filename: str):
        """Save visualization to file."""
        if self.fig is None:
            self.render()
        self.fig.savefig(filename, facecolor=self.config.background_color,
                        edgecolor='none', bbox_inches='tight')
        print(f"Saved to {filename}")
    
    def clear(self):
        """Clear all paths and points."""
        self._paths = []
        self._points = []
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        return self
    
    def multi_view(self):
        """Create a multi-view figure showing multiple projections."""
        fig = plt.figure(figsize=(16, 12), dpi=self.config.dpi)
        fig.patch.set_facecolor(self.config.background_color)
        
        # 3D view (main)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_facecolor(self.config.background_color)
        self.ax = ax1
        self._render_3d_on_ax(ax1)
        ax1.set_title('3D Pyramid', color='white')
        
        # 2D view
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(self.config.background_color)
        self._render_2d_on_ax(ax2)
        ax2.set_title('2D Triangle', color='white')
        
        # 6D parallel
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor(self.config.background_color)
        self._render_parallel_on_ax(ax3)
        ax3.set_title('6D Parallel Coords', color='white')
        
        # 1D Cantor
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor(self.config.background_color)
        self._render_1d_on_ax(ax4)
        ax4.set_title('1D Cantor', color='white')
        
        plt.tight_layout()
        self.fig = fig
        return self
    
    def _render_3d_on_ax(self, ax):
        """Render 3D on specific axes."""
        # Draw paths
        for path_data in self._paths:
            positions_3d = [self._project_to_3d(p) for p in path_data['positions']]
            if len(positions_3d) > 1:
                xs, ys, zs = zip(*positions_3d)
                ax.plot(xs, ys, zs, c=self.config.path_color, linewidth=2)
                ax.scatter(xs, ys, zs, c=self.config.path_color, s=50)
        
        # Anchors
        for anchor, pos in self.ANCHOR_POSITIONS_3D.items():
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            ax.scatter(*pos, c=color, s=60, alpha=0.7)
        
        ax.view_init(elev=35, azim=45)
    
    def _render_2d_on_ax(self, ax):
        """Render 2D on specific axes."""
        for path_data in self._paths:
            positions_2d = [self._project_to_2d(p) for p in path_data['positions']]
            if len(positions_2d) > 1:
                xs, ys = zip(*positions_2d)
                ax.plot(xs, ys, c=self.config.path_color, linewidth=2)
                ax.scatter(xs, ys, c=self.config.path_color, s=50)
        
        for anchor, pos in self.ANCHOR_POSITIONS_2D.items():
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            ax.scatter(*pos, c=color, s=60, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
    
    def _render_parallel_on_ax(self, ax):
        """Render parallel coordinates on specific axes."""
        anchors = ['ID', 'ST', 'IN', 'UN', 'PA', 'GR']
        
        for i in range(6):
            ax.axvline(x=i, color='#444444', alpha=0.5)
        
        for path_data in self._paths:
            for pos in path_data['positions']:
                ax.plot(range(6), pos, c=self.config.path_color, 
                       linewidth=2, alpha=0.7, marker='o', markersize=4)
        
        ax.set_xticks(range(6))
        ax.set_xticklabels(anchors, color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(-0.1, 1.1)
    
    def _render_1d_on_ax(self, ax):
        """Render 1D on specific axes."""
        for path_data in self._paths:
            positions_1d = [self._project_to_1d(p) for p in path_data['positions']]
            ax.scatter(positions_1d, [0.5]*len(positions_1d), 
                      c=self.config.path_color, s=100, marker='|')
            if len(positions_1d) > 1:
                ax.plot(positions_1d, [0.5]*len(positions_1d),
                       c=self.config.path_color, linewidth=2, alpha=0.5)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(colors='white')


def demo():
    """Demonstrate the visualizer."""
    print("Truth Space Visualizer Demo")
    print("=" * 50)
    
    # Create sample path data
    sample_path = {
        'start': {
            'position': {
                'IDENTITY': 0.1, 'STABILITY': 0.1, 'INVERSE': 0.1,
                'UNITY': 0.1, 'PATTERN': 0.2, 'GROWTH': 0.4
            }
        },
        'interesting_finds': []
    }
    
    # Create visualizer
    viz = TruthSpaceVisualizer()
    
    # Add path
    viz.plot_path(sample_path, "φ × φ exploration")
    
    # Add some points
    viz.add_point({
        'IDENTITY': 0.0, 'STABILITY': 0.0, 'INVERSE': 0.0,
        'UNITY': 0.0, 'PATTERN': 0.0, 'GROWTH': 1.0
    }, "Pure Growth", "#00ff00")
    
    viz.add_point({
        'IDENTITY': 1.0, 'STABILITY': 0.0, 'INVERSE': 0.0,
        'UNITY': 0.0, 'PATTERN': 0.0, 'GROWTH': 0.0
    }, "Pure Identity", "#ffffff")
    
    # Show multi-view
    viz.multi_view()
    viz.show()


if __name__ == "__main__":
    demo()
