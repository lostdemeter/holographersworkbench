"""
Path Animator
=============

Animate navigation paths through truth space with smooth transitions
and camera rotation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

try:
    from .truth_space_viz import TruthSpaceVisualizer, ProjectionMode, VisualizationConfig
except ImportError:
    from truth_space_viz import TruthSpaceVisualizer, ProjectionMode, VisualizationConfig


@dataclass
class AnimationConfig:
    """Configuration for path animation."""
    fps: int = 30
    duration_seconds: float = 10.0
    rotate_camera: bool = True
    rotation_speed: float = 1.0  # degrees per frame
    trail_length: int = 20  # Number of trailing points
    trail_fade: bool = True
    pause_at_points: float = 0.5  # seconds to pause at each point
    interpolation_steps: int = 30  # steps between points
    save_format: str = 'mp4'  # or 'gif'


class PathAnimator:
    """
    Animate paths through truth space.
    
    Features:
    - Smooth interpolation between points
    - Camera rotation
    - Trail effects
    - Multiple projection modes
    """
    
    def __init__(self, config: AnimationConfig = None, 
                 viz_config: VisualizationConfig = None):
        self.config = config or AnimationConfig()
        self.viz_config = viz_config or VisualizationConfig()
        self.visualizer = TruthSpaceVisualizer(self.viz_config)
        
        self._paths = []
        self._animation = None
        self._fig = None
        self._ax = None
    
    def add_path(self, positions: List[np.ndarray], label: str = None):
        """
        Add a path to animate.
        
        Args:
            positions: List of 6D position vectors
            label: Optional label for the path
        """
        self._paths.append({
            'positions': positions,
            'label': label or f"Path {len(self._paths) + 1}",
        })
        return self
    
    def add_navigation_result(self, nav_result: Dict, label: str = None):
        """Add path from navigation result."""
        positions = []
        
        # Extract start position
        start = nav_result.get('start', {})
        if 'position' in start:
            positions.append(self._dict_to_vector(start['position']))
        
        # Extract visited positions
        for find in nav_result.get('interesting_finds', []):
            if 'location' in find and hasattr(find['location'], 'position'):
                positions.append(self._dict_to_vector(find['location'].position))
        
        if positions:
            self.add_path(positions, label)
        
        return self
    
    def _dict_to_vector(self, pos_dict: Dict[str, float]) -> np.ndarray:
        """Convert position dict to 6D vector."""
        anchors = ['IDENTITY', 'STABILITY', 'INVERSE', 'UNITY', 'PATTERN', 'GROWTH']
        return np.array([pos_dict.get(a, 0.0) for a in anchors])
    
    def _interpolate_path(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """Interpolate between positions for smooth animation."""
        if len(positions) < 2:
            return positions
        
        interpolated = []
        steps = self.config.interpolation_steps
        
        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]
            
            for t in range(steps):
                alpha = t / steps
                # Use smooth interpolation (ease in-out)
                alpha = 0.5 * (1 - np.cos(alpha * np.pi))
                point = start * (1 - alpha) + end * alpha
                interpolated.append(point)
        
        interpolated.append(positions[-1])
        return interpolated
    
    def _project_to_3d(self, pos_6d: np.ndarray) -> np.ndarray:
        """Project 6D to 3D."""
        return self.visualizer._project_to_3d(pos_6d)
    
    def animate_3d(self, save_path: str = None):
        """
        Create 3D animation with camera rotation.
        
        Args:
            save_path: Optional path to save animation
        """
        self._fig = plt.figure(figsize=self.viz_config.figsize, 
                               dpi=self.viz_config.dpi)
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        # Set style
        self._ax.set_facecolor(self.viz_config.background_color)
        self._fig.patch.set_facecolor(self.viz_config.background_color)
        
        # Prepare interpolated paths
        all_frames = []
        for path_data in self._paths:
            interpolated = self._interpolate_path(path_data['positions'])
            frames_3d = [self._project_to_3d(p) for p in interpolated]
            all_frames.append(frames_3d)
        
        if not all_frames:
            print("No paths to animate")
            return
        
        max_frames = max(len(f) for f in all_frames)
        total_frames = max_frames + int(self.config.fps * 2)  # Extra for rotation
        
        # Draw static elements
        self._draw_static_elements()
        
        # Create line and point objects for animation
        lines = []
        points = []
        trails = []
        
        colors = ['#00ff88', '#ff6600', '#00ffff', '#ff00ff', '#ffff00']
        
        for i, frames in enumerate(all_frames):
            color = colors[i % len(colors)]
            line, = self._ax.plot([], [], [], c=color, linewidth=2, alpha=0.8)
            point, = self._ax.plot([], [], [], c=color, marker='o', markersize=10)
            trail, = self._ax.plot([], [], [], c=color, linewidth=1, alpha=0.3)
            lines.append(line)
            points.append(point)
            trails.append(trail)
        
        # Animation function
        def update(frame):
            # Update camera if rotating
            if self.config.rotate_camera:
                azim = frame * self.config.rotation_speed
                self._ax.view_init(elev=30, azim=azim)
            
            # Update each path
            for i, frames in enumerate(all_frames):
                if frame < len(frames):
                    # Current position
                    pos = frames[frame]
                    points[i].set_data([pos[0]], [pos[1]])
                    points[i].set_3d_properties([pos[2]])
                    
                    # Path so far
                    path_so_far = frames[:frame+1]
                    xs, ys, zs = zip(*path_so_far)
                    lines[i].set_data(xs, ys)
                    lines[i].set_3d_properties(zs)
                    
                    # Trail (last N points with fade)
                    trail_start = max(0, frame - self.config.trail_length)
                    trail_points = frames[trail_start:frame+1]
                    if trail_points:
                        txs, tys, tzs = zip(*trail_points)
                        trails[i].set_data(txs, tys)
                        trails[i].set_3d_properties(tzs)
            
            return lines + points + trails
        
        # Create animation
        self._animation = FuncAnimation(
            self._fig, update, frames=total_frames,
            interval=1000/self.config.fps, blit=False
        )
        
        if save_path:
            self._save_animation(save_path)
        
        return self
    
    def _draw_static_elements(self):
        """Draw static background elements."""
        # Draw anchor points
        for anchor, pos in self.visualizer.ANCHOR_POSITIONS_3D.items():
            color = self.viz_config.anchor_colors.get(anchor, '#ffffff')
            self._ax.scatter(*pos, c=color, s=80, marker='o', alpha=0.6)
            self._ax.text(pos[0]*1.15, pos[1]*1.15, pos[2]*1.15,
                         anchor[:3], color=color, fontsize=8)
        
        # Draw light wireframe
        self._draw_wireframe()
        
        # Labels
        self._ax.set_xlabel('X', color='white')
        self._ax.set_ylabel('Y', color='white')
        self._ax.set_zlabel('Z', color='white')
        self._ax.tick_params(colors='white')
        
        # Set limits
        self._ax.set_xlim(-1.2, 1.2)
        self._ax.set_ylim(-1.2, 1.2)
        self._ax.set_zlim(-1.2, 1.2)
    
    def _draw_wireframe(self):
        """Draw light wireframe of truth space structure."""
        positions = list(self.visualizer.ANCHOR_POSITIONS_3D.values())
        
        # Connect all pairs with faint lines
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                self._ax.plot(
                    [positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    [positions[i][2], positions[j][2]],
                    c='#333333', linewidth=0.5, alpha=0.3
                )
    
    def _save_animation(self, path: str):
        """Save animation to file."""
        if self._animation is None:
            print("No animation to save")
            return
        
        print(f"Saving animation to {path}...")
        
        if path.endswith('.gif'):
            self._animation.save(path, writer='pillow', fps=self.config.fps)
        else:
            try:
                self._animation.save(path, writer='ffmpeg', fps=self.config.fps)
            except Exception as e:
                print(f"Could not save as video: {e}")
                gif_path = path.rsplit('.', 1)[0] + '.gif'
                print(f"Trying to save as GIF: {gif_path}")
                self._animation.save(gif_path, writer='pillow', fps=self.config.fps)
        
        print("Saved!")
    
    def show(self):
        """Display the animation."""
        if self._animation is None:
            self.animate_3d()
        plt.show()
    
    def animate_multi_view(self, save_path: str = None):
        """
        Create animation with multiple synchronized views.
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.viz_config.dpi)
        fig.patch.set_facecolor(self.viz_config.background_color)
        
        # Create subplots
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_2d = fig.add_subplot(2, 2, 2)
        ax_parallel = fig.add_subplot(2, 2, 3)
        ax_1d = fig.add_subplot(2, 2, 4)
        
        for ax in [ax_3d, ax_2d, ax_parallel, ax_1d]:
            if hasattr(ax, 'set_facecolor'):
                ax.set_facecolor(self.viz_config.background_color)
        
        # Prepare paths
        all_frames = []
        for path_data in self._paths:
            interpolated = self._interpolate_path(path_data['positions'])
            all_frames.append(interpolated)
        
        if not all_frames:
            print("No paths to animate")
            return
        
        max_frames = max(len(f) for f in all_frames)
        
        # Initialize plot elements
        self._setup_multi_view_axes(ax_3d, ax_2d, ax_parallel, ax_1d)
        
        # Create animated elements
        elements_3d = []
        elements_2d = []
        elements_parallel = []
        elements_1d = []
        
        color = self.viz_config.path_color
        
        for frames in all_frames:
            # 3D
            line_3d, = ax_3d.plot([], [], [], c=color, linewidth=2)
            point_3d, = ax_3d.plot([], [], [], c=color, marker='o', markersize=8)
            elements_3d.append((line_3d, point_3d))
            
            # 2D
            line_2d, = ax_2d.plot([], [], c=color, linewidth=2)
            point_2d, = ax_2d.plot([], [], c=color, marker='o', markersize=8)
            elements_2d.append((line_2d, point_2d))
            
            # Parallel
            line_p, = ax_parallel.plot([], [], c=color, linewidth=2)
            elements_parallel.append(line_p)
            
            # 1D
            point_1d, = ax_1d.plot([], [], c=color, marker='|', markersize=20)
            elements_1d.append(point_1d)
        
        def update(frame):
            for i, frames in enumerate(all_frames):
                if frame < len(frames):
                    pos_6d = frames[frame]
                    
                    # 3D
                    pos_3d = self._project_to_3d(pos_6d)
                    path_3d = [self._project_to_3d(p) for p in frames[:frame+1]]
                    if path_3d:
                        xs, ys, zs = zip(*path_3d)
                        elements_3d[i][0].set_data(xs, ys)
                        elements_3d[i][0].set_3d_properties(zs)
                    elements_3d[i][1].set_data([pos_3d[0]], [pos_3d[1]])
                    elements_3d[i][1].set_3d_properties([pos_3d[2]])
                    
                    # 2D
                    pos_2d = self.visualizer._project_to_2d(pos_6d)
                    path_2d = [self.visualizer._project_to_2d(p) for p in frames[:frame+1]]
                    if path_2d:
                        xs, ys = zip(*path_2d)
                        elements_2d[i][0].set_data(xs, ys)
                    elements_2d[i][1].set_data([pos_2d[0]], [pos_2d[1]])
                    
                    # Parallel
                    elements_parallel[i].set_data(range(6), pos_6d)
                    
                    # 1D
                    pos_1d = self.visualizer._project_to_1d(pos_6d)
                    elements_1d[i].set_data([pos_1d], [0.5])
            
            # Rotate 3D camera
            if self.config.rotate_camera:
                ax_3d.view_init(elev=30, azim=frame * self.config.rotation_speed)
            
            return ([e for pair in elements_3d for e in pair] + 
                    [e for pair in elements_2d for e in pair] +
                    elements_parallel + elements_1d)
        
        anim = FuncAnimation(fig, update, frames=max_frames,
                            interval=1000/self.config.fps, blit=False)
        
        if save_path:
            print(f"Saving multi-view animation to {save_path}...")
            try:
                anim.save(save_path, writer='pillow', fps=self.config.fps)
            except Exception as e:
                print(f"Save error: {e}")
        
        self._fig = fig
        self._animation = anim
        return self
    
    def _setup_multi_view_axes(self, ax_3d, ax_2d, ax_parallel, ax_1d):
        """Setup static elements for multi-view."""
        # 3D
        for anchor, pos in self.visualizer.ANCHOR_POSITIONS_3D.items():
            color = self.viz_config.anchor_colors.get(anchor, '#ffffff')
            ax_3d.scatter(*pos, c=color, s=50, alpha=0.5)
        ax_3d.set_xlim(-1.2, 1.2)
        ax_3d.set_ylim(-1.2, 1.2)
        ax_3d.set_zlim(-1.2, 1.2)
        ax_3d.set_title('3D Pyramid', color='white')
        
        # 2D
        for anchor, pos in self.visualizer.ANCHOR_POSITIONS_2D.items():
            color = self.viz_config.anchor_colors.get(anchor, '#ffffff')
            ax_2d.scatter(*pos, c=color, s=50, alpha=0.5)
        ax_2d.set_xlim(-0.1, 1.1)
        ax_2d.set_ylim(-0.1, 1.0)
        ax_2d.set_aspect('equal')
        ax_2d.set_title('2D Triangle', color='white')
        ax_2d.tick_params(colors='white')
        
        # Parallel
        for i in range(6):
            ax_parallel.axvline(x=i, color='#444444', alpha=0.5)
        ax_parallel.set_xlim(-0.5, 5.5)
        ax_parallel.set_ylim(-0.1, 1.1)
        ax_parallel.set_xticks(range(6))
        ax_parallel.set_xticklabels(['ID', 'ST', 'IN', 'UN', 'PA', 'GR'], color='white')
        ax_parallel.set_title('6D Parallel', color='white')
        ax_parallel.tick_params(colors='white')
        
        # 1D
        ax_1d.set_xlim(-0.1, 1.1)
        ax_1d.set_ylim(0, 1)
        ax_1d.set_yticks([])
        ax_1d.axhline(y=0.5, color='#444444', alpha=0.5)
        ax_1d.set_title('1D Cantor', color='white')
        ax_1d.tick_params(colors='white')


def demo():
    """Demonstrate the path animator."""
    print("Path Animator Demo")
    print("=" * 50)
    
    # Create sample path
    positions = [
        np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.4]),  # Start (growth dominant)
        np.array([0.2, 0.1, 0.1, 0.1, 0.3, 0.2]),  # Move toward pattern
        np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.1]),  # Move toward identity
        np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05]), # Near identity
    ]
    
    animator = PathAnimator()
    animator.add_path(positions, "φ² exploration")
    animator.animate_3d()
    animator.show()


if __name__ == "__main__":
    demo()
