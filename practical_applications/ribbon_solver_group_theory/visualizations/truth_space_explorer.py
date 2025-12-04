"""
Truth Space Explorer - Empirical Discovery of Valid Truth Space Structure

Instead of assuming the structure (Sierpiński, etc.), this module explores
truth space empirically by:
1. Starting from a known valid point
2. Taking small steps in random directions
3. Testing if each new point is "valid" (satisfies constraints)
4. Building a map of discovered valid regions

This allows us to discover the ACTUAL structure of truth space rather
than overlaying an assumed geometry.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.group_structure import TruthAnchor
except ImportError:
    # Fallback anchor definitions
    class TruthAnchor(Enum):
        IDENTITY = "identity"
        STABILITY = "stability"
        INVERSE = "inverse"
        UNITY = "unity"
        PATTERN = "pattern"
        GROWTH = "growth"


@dataclass
class ExplorationConfig:
    """Configuration for truth space exploration."""
    # Exploration parameters
    step_size: float = 0.02          # Size of exploration steps
    max_points: int = 5000           # Maximum points to discover
    branch_factor: int = 6           # Number of directions to try per point
    min_distance: float = 0.01       # Minimum distance between stored points
    
    # Validity constraints
    sum_tolerance: float = 0.05      # How close to 1.0 the sum must be
    min_weight: float = 0.0          # Minimum weight for any anchor
    max_weight: float = 1.0          # Maximum weight for any anchor
    
    # Visualization
    figsize: Tuple[int, int] = (14, 10)
    point_size: float = 5.0
    point_alpha: float = 0.6
    valid_color: str = '#00ff88'
    invalid_color: str = '#ff4444'
    frontier_color: str = '#ffaa00'
    background_color: str = '#0a0a0a'


@dataclass
class ExploredPoint:
    """A point discovered during exploration."""
    position: np.ndarray
    valid: bool
    depth: int  # How many steps from start
    parent_idx: Optional[int] = None
    
    def __hash__(self):
        return hash(tuple(self.position.round(4)))


class TruthSpaceExplorer:
    """
    Explores truth space empirically to discover its actual structure.
    
    Rather than assuming a geometric structure, this class:
    1. Starts from a valid seed point
    2. Explores outward in all directions
    3. Tests validity at each point
    4. Builds a map of the discovered structure
    """
    
    ANCHORS = ['IDENTITY', 'STABILITY', 'INVERSE', 'UNITY', 'PATTERN', 'GROWTH']
    
    def __init__(self, config: ExplorationConfig = None, 
                 validity_fn: Callable[[np.ndarray], bool] = None):
        self.config = config or ExplorationConfig()
        self.validity_fn = validity_fn or self._default_validity
        
        # Discovered points
        self.valid_points: List[np.ndarray] = []
        self.invalid_points: List[np.ndarray] = []
        self.frontier: deque = deque()  # Points to explore from
        self.explored_set: Set[Tuple] = set()  # For fast lookup
        
        # Statistics
        self.stats = {
            'total_tested': 0,
            'valid_found': 0,
            'invalid_found': 0,
            'max_depth': 0,
        }
        
        # Visualization
        self.fig = None
        self.ax = None
    
    def _default_validity(self, point: np.ndarray) -> bool:
        """
        Default validity check: weights sum to ~1 and are in valid range.
        
        This is the MINIMAL constraint. You can provide custom validity
        functions that encode additional mathematical constraints.
        """
        # Check bounds
        if np.any(point < self.config.min_weight) or np.any(point > self.config.max_weight):
            return False
        
        # Check sum constraint
        total = np.sum(point)
        if abs(total - 1.0) > self.config.sum_tolerance:
            return False
        
        return True
    
    def _point_key(self, point: np.ndarray) -> Tuple:
        """Convert point to hashable key for deduplication."""
        return tuple((point / self.config.min_distance).astype(int))
    
    def _is_explored(self, point: np.ndarray) -> bool:
        """Check if we've already explored near this point."""
        return self._point_key(point) in self.explored_set
    
    def _mark_explored(self, point: np.ndarray):
        """Mark a point as explored."""
        self.explored_set.add(self._point_key(point))
    
    def _generate_directions(self) -> List[np.ndarray]:
        """
        Generate exploration directions in 6D space.
        
        We generate directions that:
        1. Stay on the simplex (sum to 0 for the delta)
        2. Cover different "directions" in truth space
        """
        directions = []
        n = len(self.ANCHORS)
        
        # Axis-aligned directions (increase one, decrease another)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = np.zeros(n)
                    d[i] = 1
                    d[j] = -1
                    d = d / np.linalg.norm(d)
                    directions.append(d)
        
        # Random directions on the simplex tangent space
        for _ in range(self.config.branch_factor * 2):
            d = np.random.randn(n)
            d = d - np.mean(d)  # Project to sum-zero subspace
            if np.linalg.norm(d) > 0.01:
                d = d / np.linalg.norm(d)
                directions.append(d)
        
        return directions
    
    def _normalize_to_simplex(self, point: np.ndarray) -> np.ndarray:
        """Project point onto the probability simplex."""
        # Clip to valid range
        point = np.clip(point, self.config.min_weight, self.config.max_weight)
        # Normalize to sum to 1
        total = np.sum(point)
        if total > 0:
            point = point / total
        return point
    
    def explore_from(self, start: np.ndarray, max_iterations: int = None) -> Dict:
        """
        Explore truth space starting from a seed point.
        
        Args:
            start: Starting point (6D vector, should sum to 1)
            max_iterations: Override max points to discover
            
        Returns:
            Dictionary with exploration results
        """
        max_iter = max_iterations or self.config.max_points
        
        # Normalize start point
        start = self._normalize_to_simplex(start)
        
        # Initialize
        self.valid_points = [start]
        self.invalid_points = []
        self.frontier = deque([(start, 0)])  # (point, depth)
        self.explored_set = set()
        self._mark_explored(start)
        
        self.stats = {
            'total_tested': 1,
            'valid_found': 1,
            'invalid_found': 0,
            'max_depth': 0,
        }
        
        directions = self._generate_directions()
        
        print(f"Starting exploration from {start.round(3)}")
        print(f"Generated {len(directions)} exploration directions")
        
        iteration = 0
        while self.frontier and len(self.valid_points) < max_iter:
            # Get next point to explore from
            current, depth = self.frontier.popleft()
            
            # Try each direction
            for direction in directions:
                # Take a step
                new_point = current + direction * self.config.step_size
                
                # Skip if already explored
                if self._is_explored(new_point):
                    continue
                
                self._mark_explored(new_point)
                self.stats['total_tested'] += 1
                
                # Normalize to stay on simplex
                new_point = self._normalize_to_simplex(new_point)
                
                # Test validity
                if self.validity_fn(new_point):
                    self.valid_points.append(new_point)
                    self.frontier.append((new_point, depth + 1))
                    self.stats['valid_found'] += 1
                    self.stats['max_depth'] = max(self.stats['max_depth'], depth + 1)
                else:
                    self.invalid_points.append(new_point)
                    self.stats['invalid_found'] += 1
            
            iteration += 1
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: {len(self.valid_points)} valid, "
                      f"{len(self.invalid_points)} invalid, "
                      f"frontier size: {len(self.frontier)}")
        
        print(f"\nExploration complete!")
        print(f"  Valid points: {len(self.valid_points)}")
        print(f"  Invalid points: {len(self.invalid_points)}")
        print(f"  Max depth: {self.stats['max_depth']}")
        
        return {
            'valid_points': np.array(self.valid_points),
            'invalid_points': np.array(self.invalid_points) if self.invalid_points else np.array([]),
            'stats': self.stats,
        }
    
    def explore_grid(self, resolution: int = 20) -> Dict:
        """
        Systematically explore truth space on a grid.
        
        This provides complete coverage but is slower than frontier-based exploration.
        """
        print(f"Grid exploration with resolution {resolution}")
        
        self.valid_points = []
        self.invalid_points = []
        
        # Generate grid points on the 5-simplex
        # We use barycentric coordinates
        n = len(self.ANCHORS)
        total_points = 0
        
        # Generate points where all coordinates sum to 1
        for idx in np.ndindex(*([resolution + 1] * (n - 1))):
            coords = np.array(idx) / resolution
            if np.sum(coords) <= 1.0:
                last_coord = 1.0 - np.sum(coords)
                point = np.append(coords, last_coord)
                
                total_points += 1
                if self.validity_fn(point):
                    self.valid_points.append(point)
                else:
                    self.invalid_points.append(point)
        
        print(f"Tested {total_points} grid points")
        print(f"  Valid: {len(self.valid_points)}")
        print(f"  Invalid: {len(self.invalid_points)}")
        
        return {
            'valid_points': np.array(self.valid_points),
            'invalid_points': np.array(self.invalid_points) if self.invalid_points else np.array([]),
            'stats': {
                'total_tested': total_points,
                'valid_found': len(self.valid_points),
                'invalid_found': len(self.invalid_points),
            }
        }
    
    def _project_to_3d(self, point: np.ndarray) -> np.ndarray:
        """Project 6D point to 3D for visualization."""
        # Use first 3 principal components or anchor-based projection
        # This maps to a tetrahedron-like structure
        
        # Anchor positions in 3D (tetrahedron vertices + 2 extra)
        anchors_3d = np.array([
            [0, 0, 1],           # IDENTITY (top)
            [-0.5, -0.5, -0.5],  # STABILITY
            [0.5, -0.5, -0.5],   # INVERSE
            [0, 0.7, -0.3],      # UNITY
            [-0.7, 0.3, 0],      # PATTERN
            [0.7, 0.3, 0],       # GROWTH
        ])
        
        # Weighted sum of anchor positions
        return np.dot(point, anchors_3d)
    
    def _project_to_2d(self, point: np.ndarray) -> np.ndarray:
        """Project 6D point to 2D for visualization."""
        # Use barycentric-like projection
        # Map to triangle with 3 main anchors
        
        # Simplified: use first 3 weights for triangle position
        w1, w2, w3 = point[0], point[3], point[5]  # IDENTITY, UNITY, GROWTH
        total = w1 + w2 + w3
        if total > 0:
            w1, w2, w3 = w1/total, w2/total, w3/total
        
        # Triangle vertices
        v1 = np.array([0.5, 0.866])  # top
        v2 = np.array([0, 0])         # bottom left
        v3 = np.array([1, 0])         # bottom right
        
        return w1 * v1 + w2 * v2 + w3 * v3
    
    def visualize_3d(self, show_invalid: bool = False, 
                     title: str = "Discovered Truth Space Structure",
                     zoom: float = 1.0, center: np.ndarray = None,
                     color_by_depth: bool = True, max_points: int = 50000):
        """
        Visualize discovered structure in 3D.
        
        Args:
            show_invalid: Show invalid points in red
            title: Plot title
            zoom: Zoom level (higher = more zoomed in)
            center: Center point for zoom (3D coordinates)
            color_by_depth: Color points by exploration depth
            max_points: Maximum points to render (subsample if more)
        """
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Plot valid points
        if self.valid_points:
            points_3d = np.array([self._project_to_3d(p) for p in self.valid_points])
            
            # Subsample if too many points
            if len(points_3d) > max_points:
                indices = np.random.choice(len(points_3d), max_points, replace=False)
                points_3d = points_3d[indices]
            
            # Apply zoom - filter to region around center
            if zoom > 1.0:
                if center is None:
                    center = np.mean(points_3d, axis=0)
                
                # Calculate distance from center
                distances = np.linalg.norm(points_3d - center, axis=1)
                radius = 1.0 / zoom
                mask = distances < radius
                points_3d = points_3d[mask]
                
                if len(points_3d) == 0:
                    print(f"Warning: No points within zoom radius {radius:.3f}")
                    points_3d = np.array([center])
            
            # Color by distance from center for depth perception
            if color_by_depth and len(points_3d) > 1:
                center_pt = np.mean(points_3d, axis=0)
                distances = np.linalg.norm(points_3d - center_pt, axis=1)
                # Normalize distances for coloring
                if distances.max() > distances.min():
                    colors = (distances - distances.min()) / (distances.max() - distances.min())
                else:
                    colors = np.zeros(len(distances))
                
                self.ax.scatter(
                    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c=colors, cmap='viridis', s=self.config.point_size,
                    alpha=self.config.point_alpha, label=f'Valid ({len(self.valid_points)})'
                )
            else:
                self.ax.scatter(
                    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c=self.config.valid_color, s=self.config.point_size,
                    alpha=self.config.point_alpha, label=f'Valid ({len(self.valid_points)})'
                )
        
        # Plot invalid points (optional)
        if show_invalid and self.invalid_points:
            points_3d = np.array([self._project_to_3d(p) for p in self.invalid_points])
            self.ax.scatter(
                points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=self.config.invalid_color, s=self.config.point_size * 0.5,
                alpha=self.config.point_alpha * 0.3, label=f'Invalid ({len(self.invalid_points)})'
            )
        
        # Plot anchor positions
        anchor_colors = ['#ffffff', '#00ffff', '#ff00ff', '#ffff00', '#ff8800', '#00ff00']
        for i, (anchor, color) in enumerate(zip(self.ANCHORS, anchor_colors)):
            pos = np.zeros(6)
            pos[i] = 1.0
            pos_3d = self._project_to_3d(pos)
            self.ax.scatter(*pos_3d, c=color, s=200, marker='o', edgecolors='white')
            self.ax.text(pos_3d[0]*1.1, pos_3d[1]*1.1, pos_3d[2]*1.1, 
                        anchor[:3], color=color, fontsize=10)
        
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        self.ax.set_title(title, color='white', fontsize=14)
        self.ax.tick_params(colors='white')
        # Set axis limits based on zoom
        if zoom > 1.0:
            if len(points_3d) > 0:
                center = np.mean(points_3d, axis=0)
                # Calculate actual data range
                data_range = np.max(np.abs(points_3d - center))
                radius = max(data_range * 1.2, 0.01)  # Ensure minimum radius
                self.ax.set_xlim(center[0] - radius, center[0] + radius)
                self.ax.set_ylim(center[1] - radius, center[1] + radius)
                self.ax.set_zlim(center[2] - radius, center[2] + radius)
        
        return self.fig, self.ax
    
    def visualize_3d_interactive(self, max_points: int = 50000):
        """
        Create an interactive 3D visualization with zoom slider.
        """
        from matplotlib.widgets import Slider
        
        if not self.valid_points:
            print("No points to visualize")
            return None, None
        
        # Project all points
        all_points_3d = np.array([self._project_to_3d(p) for p in self.valid_points])
        
        # Subsample if needed
        if len(all_points_3d) > max_points:
            indices = np.random.choice(len(all_points_3d), max_points, replace=False)
            all_points_3d = all_points_3d[indices]
        
        center = np.mean(all_points_3d, axis=0)
        max_dist = np.max(np.linalg.norm(all_points_3d - center, axis=1))
        
        # Create figure with slider
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Main 3D axes
        self.ax = self.fig.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
        self.ax.set_facecolor(self.config.background_color)
        
        # Slider axes
        ax_zoom = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
        
        def update_view(zoom_val):
            self.ax.clear()
            self.ax.set_facecolor(self.config.background_color)
            
            # Filter points by zoom
            radius = max_dist / zoom_val
            distances = np.linalg.norm(all_points_3d - center, axis=1)
            mask = distances < radius
            visible_points = all_points_3d[mask]
            
            if len(visible_points) > 0:
                # Color by distance
                vis_distances = np.linalg.norm(visible_points - center, axis=1)
                if vis_distances.max() > vis_distances.min():
                    colors = (vis_distances - vis_distances.min()) / (vis_distances.max() - vis_distances.min())
                else:
                    colors = np.zeros(len(vis_distances))
                
                self.ax.scatter(
                    visible_points[:, 0], visible_points[:, 1], visible_points[:, 2],
                    c=colors, cmap='viridis', s=3, alpha=0.6
                )
            
            # Set limits
            self.ax.set_xlim(center[0] - radius, center[0] + radius)
            self.ax.set_ylim(center[1] - radius, center[1] + radius)
            self.ax.set_zlim(center[2] - radius, center[2] + radius)
            
            self.ax.set_title(f'Truth Space (zoom={zoom_val:.1f}x, {len(visible_points)} visible)', 
                            color='white')
            self.ax.tick_params(colors='white')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            
            self.fig.canvas.draw_idle()
        
        # Create slider
        zoom_slider = Slider(ax_zoom, 'Zoom', 1, 100, valinit=1, valstep=0.5)
        zoom_slider.label.set_color('white')
        zoom_slider.valtext.set_color('white')
        zoom_slider.on_changed(update_view)
        
        # Initial view
        update_view(1)
        
        return self.fig, self.ax
    
    def visualize_2d(self, show_invalid: bool = False,
                     title: str = "Discovered Truth Space (2D Projection)"):
        """Visualize discovered structure in 2D."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Plot valid points
        if self.valid_points:
            points_2d = np.array([self._project_to_2d(p) for p in self.valid_points])
            self.ax.scatter(
                points_2d[:, 0], points_2d[:, 1],
                c=self.config.valid_color, s=self.config.point_size,
                alpha=self.config.point_alpha, label=f'Valid ({len(self.valid_points)})'
            )
        
        # Plot invalid points
        if show_invalid and self.invalid_points:
            points_2d = np.array([self._project_to_2d(p) for p in self.invalid_points])
            self.ax.scatter(
                points_2d[:, 0], points_2d[:, 1],
                c=self.config.invalid_color, s=self.config.point_size * 0.5,
                alpha=self.config.point_alpha * 0.3, label=f'Invalid ({len(self.invalid_points)})'
            )
        
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.0)
        self.ax.set_aspect('equal')
        self.ax.set_title(title, color='white', fontsize=14)
        self.ax.tick_params(colors='white')
        self.ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
        
        return self.fig, self.ax
    
    def visualize_slices(self, n_slices: int = 6, axis: int = 2,
                         title: str = "Truth Space Cross-Sections"):
        """
        Visualize cross-sectional slices through truth space.
        This helps reveal internal structure that's hidden in 3D views.
        """
        if not self.valid_points:
            print("No points to visualize")
            return None, None
        
        points_3d = np.array([self._project_to_3d(p) for p in self.valid_points])
        
        # Determine slice positions
        axis_vals = points_3d[:, axis]
        slice_positions = np.linspace(axis_vals.min(), axis_vals.max(), n_slices + 2)[1:-1]
        slice_thickness = (axis_vals.max() - axis_vals.min()) / (n_slices * 2)
        
        # Create subplot grid
        cols = min(3, n_slices)
        rows = (n_slices + cols - 1) // cols
        
        self.fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        self.fig.patch.set_facecolor(self.config.background_color)
        
        if n_slices == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        axis_names = ['X', 'Y', 'Z']
        other_axes = [i for i in range(3) if i != axis]
        
        for idx, (ax, pos) in enumerate(zip(axes[:n_slices], slice_positions)):
            ax.set_facecolor(self.config.background_color)
            
            # Filter points in this slice
            mask = np.abs(points_3d[:, axis] - pos) < slice_thickness
            slice_points = points_3d[mask]
            
            if len(slice_points) > 0:
                ax.scatter(
                    slice_points[:, other_axes[0]], 
                    slice_points[:, other_axes[1]],
                    c='#00ff88', s=2, alpha=0.5
                )
            
            ax.set_title(f'{axis_names[axis]}={pos:.2f} ({len(slice_points)} pts)', 
                        color='white', fontsize=10)
            ax.set_xlabel(axis_names[other_axes[0]], color='white')
            ax.set_ylabel(axis_names[other_axes[1]], color='white')
            ax.tick_params(colors='white')
            ax.set_aspect('equal')
        
        for ax in axes[n_slices:]:
            ax.set_visible(False)
        
        plt.suptitle(title, color='white', fontsize=14)
        plt.tight_layout()
        
        return self.fig, axes
    
    def visualize_density(self, resolution: int = 50,
                          title: str = "Truth Space Density"):
        """Visualize point density as a heatmap."""
        if not self.valid_points:
            print("No points to visualize")
            return None, None
        
        points_3d = np.array([self._project_to_3d(p) for p in self.valid_points])
        
        self.fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.patch.set_facecolor(self.config.background_color)
        
        projections = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
        
        for ax, (i, j, name) in zip(axes, projections):
            ax.set_facecolor(self.config.background_color)
            
            h, xedges, yedges = np.histogram2d(
                points_3d[:, i], points_3d[:, j], bins=resolution
            )
            h = np.log1p(h)  # Log scale
            
            im = ax.imshow(h.T, origin='lower', cmap='hot',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          aspect='auto')
            
            ax.set_title(f'{name} Projection', color='white')
            ax.set_xlabel(name[0], color='white')
            ax.set_ylabel(name[1], color='white')
            ax.tick_params(colors='white')
        
        plt.suptitle(title, color='white', fontsize=14)
        plt.tight_layout()
        
        return self.fig, axes
    
    def save(self, filepath: str):
        """Save discovered points to file."""
        np.savez(filepath,
                 valid_points=np.array(self.valid_points),
                 invalid_points=np.array(self.invalid_points) if self.invalid_points else np.array([]),
                 stats=self.stats)
        print(f"Saved exploration results to {filepath}")
    
    def load(self, filepath: str):
        """Load discovered points from file."""
        data = np.load(filepath, allow_pickle=True)
        self.valid_points = list(data['valid_points'])
        self.invalid_points = list(data['invalid_points'])
        self.stats = data['stats'].item()
        print(f"Loaded {len(self.valid_points)} valid points from {filepath}")


def create_ribbon_validity_fn():
    """
    Create a validity function based on the actual ribbon solver constraints.
    
    This attempts to integrate with the ribbon solver to test if a point
    represents a valid mathematical operation.
    """
    try:
        from ..core.group_structure import TruthSpaceNavigator
        navigator = TruthSpaceNavigator()
        
        def validity_fn(point: np.ndarray) -> bool:
            # Basic simplex check
            if np.any(point < 0) or np.any(point > 1):
                return False
            if abs(np.sum(point) - 1.0) > 0.05:
                return False
            
            # Try to use the navigator to validate
            try:
                # Check if this point can be reached via valid operations
                # This is a placeholder - actual implementation would
                # test group closure properties
                return True
            except:
                return False
        
        return validity_fn
    except ImportError:
        # Fallback to simplex if navigator not available
        return create_mathematical_validity_fn("simplex")


def create_mathematical_validity_fn(constraint_type: str = "simplex"):
    """
    Create a validity function with mathematical constraints.
    
    Constraint types:
    - "simplex": Just the probability simplex (all valid)
    - "balanced": Requires some balance between anchors
    - "group_closure": Enforces group-theoretic closure
    - "golden": Enforces golden ratio relationships
    - "sierpinski": Enforces Sierpiński-like fractal structure
    """
    
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    def simplex_validity(point: np.ndarray) -> bool:
        if np.any(point < 0) or np.any(point > 1):
            return False
        if abs(np.sum(point) - 1.0) > 0.05:
            return False
        return True
    
    def balanced_validity(point: np.ndarray) -> bool:
        if not simplex_validity(point):
            return False
        # No single anchor can dominate too much
        if np.max(point) > 0.6:
            return False
        # At least 3 anchors must be active
        if np.sum(point > 0.05) < 3:
            return False
        return True
    
    def group_closure_validity(point: np.ndarray) -> bool:
        """
        Enforce group-theoretic constraints:
        - IDENTITY + INVERSE should balance
        - STABILITY should be related to IDENTITY
        - GROWTH and PATTERN should be complementary
        """
        if not simplex_validity(point):
            return False
        
        # Unpack weights
        identity, stability, inverse, unity, pattern, growth = point
        
        # Constraint 1: Identity and Inverse should be related
        # In a group, a * a^-1 = e, so they should be balanced
        if abs(identity - inverse) > 0.3:
            return False
        
        # Constraint 2: Stability relates to Identity
        # Stable elements are close to identity
        if stability > 0.1 and identity < stability * 0.5:
            return False
        
        # Constraint 3: Growth and Pattern are complementary
        # Growth creates new patterns, patterns constrain growth
        if abs(growth + pattern - 0.4) > 0.3:
            return False
        
        return True
    
    def golden_validity(point: np.ndarray) -> bool:
        """
        Enforce golden ratio relationships between anchors.
        This creates a more structured, self-similar space.
        
        More permissive version: checks if the distribution of weights
        follows a golden-ratio-like decay pattern.
        """
        if not simplex_validity(point):
            return False
        
        sorted_weights = np.sort(point)[::-1]
        
        # Check multiple conditions (any one is sufficient)
        
        # 1. Any adjacent pair has golden ratio (relaxed tolerance)
        for i in range(len(sorted_weights) - 1):
            if sorted_weights[i+1] > 0.005:
                ratio = sorted_weights[i] / sorted_weights[i+1]
                if abs(ratio - PHI) < 0.5 or abs(ratio - 1/PHI) < 0.4:
                    return True
        
        # 2. Sum of smaller weights approximates largest weight (Fibonacci-like)
        if len(sorted_weights) >= 3:
            if abs(sorted_weights[0] - sorted_weights[1] - sorted_weights[2]) < 0.1:
                return True
        
        # 3. Total deviation from uniform follows φ^(-k) pattern
        uniform = 1/6
        mean_dev = np.mean(np.abs(sorted_weights - uniform))
        for k in range(2, 8):
            if abs(mean_dev - PHI**(-k)) < 0.03:
                return True
        
        # 4. Relaxed: at least some structure (not perfectly uniform)
        if np.std(sorted_weights) > 0.02:
            return True
        
        return False
    
    def sierpinski_validity(point: np.ndarray) -> bool:
        """
        Enforce Sierpiński-like structure.
        Points are valid if they're "close" to one of the recursive
        sub-simplices of the Sierpiński gasket.
        
        More permissive version that allows exploration of the fractal.
        """
        if not simplex_validity(point):
            return False
        
        # Sort weights to find dominant anchors
        sorted_weights = np.sort(point)[::-1]
        
        # Sierpiński removes the "middle" - points where weights are too uniform
        # But we need to be permissive enough to allow exploration
        
        # Condition 1: Not perfectly uniform (middle region)
        if np.std(sorted_weights) < 0.01:
            return False
        
        # Condition 2: At least one anchor should be somewhat dominant
        # (we're in a "corner" of the simplex)
        if sorted_weights[0] < 0.2:
            return False
        
        # Condition 3: The bottom weights shouldn't all be equal
        # (that would be the center of a sub-simplex)
        bottom3 = sorted_weights[3:]
        if len(bottom3) > 0 and np.std(bottom3) < 0.005 and np.mean(bottom3) > 0.1:
            return False
        
        # Condition 4: Self-similarity check - ratios should follow a pattern
        # In Sierpiński, there's a recursive structure
        ratios = []
        for i in range(len(sorted_weights) - 1):
            if sorted_weights[i+1] > 0.01:
                ratios.append(sorted_weights[i] / sorted_weights[i+1])
        
        if ratios:
            # Ratios should not all be 1 (uniform)
            if np.std(ratios) < 0.1:
                return False
        
        return True
    
    # Return the appropriate function
    validators = {
        "simplex": simplex_validity,
        "balanced": balanced_validity,
        "group_closure": group_closure_validity,
        "golden": golden_validity,
        "sierpinski": sierpinski_validity,
    }
    
    return validators.get(constraint_type, simplex_validity)


def demo(constraint_type: str = "golden"):
    """Demonstrate truth space exploration with different constraints."""
    print("=" * 70)
    print("TRUTH SPACE EXPLORER")
    print("Empirical Discovery of Valid Truth Space Structure")
    print(f"Constraint type: {constraint_type}")
    print("=" * 70)
    
    # Create explorer with config
    config = ExplorationConfig(
        step_size=0.02,
        max_points=5000,
        branch_factor=12,
        min_distance=0.01,
    )
    
    # Get validity function for chosen constraint type
    validity_fn = create_mathematical_validity_fn(constraint_type)
    
    explorer = TruthSpaceExplorer(config, validity_fn=validity_fn)
    
    # Start from a point that satisfies the constraints
    # For golden ratio, start with a golden-ratio weighted point
    PHI = (1 + np.sqrt(5)) / 2
    if constraint_type == "golden":
        # Create a point with golden ratio relationships
        start = np.array([0.38, 0.24, 0.15, 0.09, 0.08, 0.06])
    elif constraint_type == "group_closure":
        # Start with balanced identity/inverse
        start = np.array([0.2, 0.1, 0.2, 0.2, 0.15, 0.15])
    elif constraint_type == "sierpinski":
        # Start near a corner
        start = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    else:
        start = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
    
    # Normalize
    start = start / np.sum(start)
    
    # Explore!
    results = explorer.explore_from(start, max_iterations=5000)
    
    # Visualize
    print("\nGenerating visualizations...")
    
    # 3D view
    fig3d, ax3d = explorer.visualize_3d(show_invalid=True,
        title=f"Truth Space Structure ({constraint_type} constraints)")
    fig3d.savefig(f'/tmp/truth_space_{constraint_type}_3d.png', dpi=150, 
                  facecolor=config.background_color)
    print(f"Saved 3D view to /tmp/truth_space_{constraint_type}_3d.png")
    
    # 2D view
    fig2d, ax2d = explorer.visualize_2d(show_invalid=True,
        title=f"Truth Space 2D ({constraint_type} constraints)")
    fig2d.savefig(f'/tmp/truth_space_{constraint_type}_2d.png', dpi=150,
                  facecolor=config.background_color)
    print(f"Saved 2D view to /tmp/truth_space_{constraint_type}_2d.png")
    
    return explorer


def compare_constraints():
    """Compare different constraint types."""
    print("=" * 70)
    print("COMPARING CONSTRAINT TYPES")
    print("=" * 70)
    
    constraint_types = ["simplex", "balanced", "group_closure", "golden", "sierpinski"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0a')
    
    for idx, ctype in enumerate(constraint_types):
        print(f"\n--- Exploring with {ctype} constraints ---")
        
        config = ExplorationConfig(
            step_size=0.025,
            max_points=2000,
            branch_factor=10,
            min_distance=0.012,
        )
        
        validity_fn = create_mathematical_validity_fn(ctype)
        explorer = TruthSpaceExplorer(config, validity_fn=validity_fn)
        
        # Start point
        start = np.array([0.25, 0.2, 0.15, 0.15, 0.13, 0.12])
        start = start / np.sum(start)
        
        results = explorer.explore_from(start, max_iterations=2000)
        
        # Plot in 2D
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('#0a0a0a')
        
        if explorer.valid_points:
            points_2d = np.array([explorer._project_to_2d(p) for p in explorer.valid_points])
            ax.scatter(points_2d[:, 0], points_2d[:, 1],
                      c='#00ff88', s=3, alpha=0.5)
        
        if explorer.invalid_points:
            points_2d = np.array([explorer._project_to_2d(p) for p in explorer.invalid_points])
            ax.scatter(points_2d[:, 0], points_2d[:, 1],
                      c='#ff4444', s=1, alpha=0.2)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.set_title(f'{ctype}\nValid: {len(explorer.valid_points)}', 
                    color='white', fontsize=12)
        ax.tick_params(colors='white')
    
    # Empty last subplot
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/tmp/truth_space_comparison.png', dpi=150, facecolor='#0a0a0a')
    print("\nSaved comparison to /tmp/truth_space_comparison.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explore Truth Space Structure")
    parser.add_argument('--constraint', '-c', type=str, default='golden',
                       choices=['simplex', 'balanced', 'group_closure', 'golden', 'sierpinski'],
                       help='Constraint type to use')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all constraint types')
    parser.add_argument('--points', '-n', type=int, default=5000,
                       help='Maximum points to discover')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_constraints()
    else:
        demo(args.constraint)
