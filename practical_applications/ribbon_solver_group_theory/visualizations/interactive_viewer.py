#!/usr/bin/env python3
"""
Interactive Truth Space Viewer
==============================

An interactive viewer for truth space navigation with:
- Keyboard controls for switching views
- Mouse rotation in 3D
- Real-time projection switching

Controls:
    1-6: Switch projection modes
    r: Toggle camera rotation
    g: Reset to golden angle view
    +/-: Zoom in/out
    q: Quit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truth_space_viz import TruthSpaceVisualizer, ProjectionMode, VisualizationConfig
from truth_space_explorer import TruthSpaceExplorer, ExplorationConfig, create_mathematical_validity_fn


class InteractiveTruthSpaceViewer:
    """
    Interactive viewer for truth space with real-time controls.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig(figsize=(14, 10))
        self.visualizer = TruthSpaceVisualizer(self.config)
        
        self._paths = []
        self._points = []
        self._current_mode = ProjectionMode.PYRAMID_3D
        self._rotating = False
        self._rotation_angle = 0
        self._zoom_level = 1.0
        self._base_limit = 1.5
        
        # Geometry controls
        self._show_geometry = True
        self._geometry_alpha = 0.15
        self._geometry_fill_alpha = 0.05
        self._geometry_depth = 3
        self._auto_depth = True  # Auto-adjust depth based on zoom
        
        # Exploration mode
        self._explore_mode = False
        self._explored_points = []
        self._explorer = None
        self._exploration_constraint = "simplex"
        
        self.fig = None
        self.ax_main = None
        self.ax_controls = None
        self.sliders = {}
    
    def add_path(self, positions, label=None):
        """Add a path to visualize."""
        if isinstance(positions, dict):
            # Navigation result format
            self.visualizer.plot_path(positions, label)
        else:
            # List of position vectors
            self._paths.append({
                'positions': positions,
                'label': label or f"Path {len(self._paths) + 1}"
            })
        return self
    
    def add_point(self, position, label=None, color=None):
        """Add a point to visualize."""
        self.visualizer.add_point(position, label, color)
        return self
    
    def show(self):
        """Display the interactive viewer."""
        # Create figure with control panel
        self.fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        # Main visualization area
        self.ax_main = self.fig.add_axes([0.1, 0.15, 0.75, 0.8], projection='3d')
        self.ax_main.set_facecolor(self.config.background_color)
        
        # Control panel
        self._setup_controls()
        
        # Initial render
        self._render_current_mode()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Connect scroll wheel for zoom
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # Title
        self.fig.suptitle('Interactive Truth Space Viewer (scroll to zoom)', 
                         color='white', fontsize=14, y=0.98)
        
        plt.show()
    
    def _on_scroll(self, event):
        """Handle scroll wheel for zoom."""
        if self._current_mode != ProjectionMode.PYRAMID_3D:
            return
        
        if event.button == 'up':
            self._zoom_level *= 0.9  # Zoom in
        elif event.button == 'down':
            self._zoom_level *= 1.1  # Zoom out
        
        # Clamp zoom level
        self._zoom_level = max(0.2, min(5.0, self._zoom_level))
        
        # Apply zoom and redraw
        self._apply_zoom()
        self.fig.canvas.draw_idle()
    
    def _apply_zoom(self):
        """Apply current zoom level to 3D axes."""
        if self.ax_main is None:
            return
        
        limit = self._base_limit * self._zoom_level
        self.ax_main.set_xlim(-limit, limit)
        self.ax_main.set_ylim(-limit, limit)
        try:
            self.ax_main.set_zlim(-limit, limit)
        except:
            pass  # 2D mode doesn't have zlim
    
    def _setup_controls(self):
        """Setup control buttons and sliders."""
        # Mode selection buttons (bottom row)
        button_specs = [
            ('3D', 0.02, ProjectionMode.PYRAMID_3D),
            ('2D', 0.08, ProjectionMode.TRIANGLE_2D),
            ('1D', 0.14, ProjectionMode.LINE_1D),
            ('6D', 0.20, ProjectionMode.FULL_6D),
        ]
        
        self.buttons = []
        for label, x, mode in button_specs:
            ax_btn = self.fig.add_axes([x, 0.02, 0.05, 0.04])
            btn = Button(ax_btn, label, color='#333333', hovercolor='#555555')
            btn.label.set_color('white')
            btn.label.set_fontsize(9)
            btn.on_clicked(lambda event, m=mode: self._set_mode(m))
            self.buttons.append(btn)
        
        # Rotation toggle
        ax_rot = self.fig.add_axes([0.27, 0.02, 0.06, 0.04])
        self.btn_rotate = Button(ax_rot, 'Rotate', color='#333333', hovercolor='#555555')
        self.btn_rotate.label.set_color('white')
        self.btn_rotate.label.set_fontsize(9)
        self.btn_rotate.on_clicked(self._toggle_rotation)
        
        # Geometry toggle button
        ax_geom = self.fig.add_axes([0.35, 0.02, 0.07, 0.04])
        self.btn_geometry = Button(ax_geom, 'Geometry', color='#336633', hovercolor='#448844')
        self.btn_geometry.label.set_color('white')
        self.btn_geometry.label.set_fontsize(9)
        self.btn_geometry.on_clicked(self._toggle_geometry)
        
        # Auto-depth toggle
        ax_auto = self.fig.add_axes([0.44, 0.02, 0.07, 0.04])
        self.btn_auto_depth = Button(ax_auto, 'AutoDep', color='#336633', hovercolor='#448844')
        self.btn_auto_depth.label.set_color('white')
        self.btn_auto_depth.label.set_fontsize(9)
        self.btn_auto_depth.on_clicked(self._toggle_auto_depth)
        
        # --- Sliders on the right side ---
        slider_color = '#444444'
        
        # Edge opacity slider
        ax_edge = self.fig.add_axes([0.88, 0.70, 0.03, 0.20])
        self.sliders['edge_alpha'] = Slider(
            ax_edge, 'Edge', 0.0, 1.0, 
            valinit=self._geometry_alpha,
            orientation='vertical',
            color='#4488ff'
        )
        self.sliders['edge_alpha'].label.set_color('white')
        self.sliders['edge_alpha'].valtext.set_color('white')
        self.sliders['edge_alpha'].on_changed(self._on_edge_alpha_change)
        
        # Fill opacity slider
        ax_fill = self.fig.add_axes([0.93, 0.70, 0.03, 0.20])
        self.sliders['fill_alpha'] = Slider(
            ax_fill, 'Fill', 0.0, 0.5, 
            valinit=self._geometry_fill_alpha,
            orientation='vertical',
            color='#4488ff'
        )
        self.sliders['fill_alpha'].label.set_color('white')
        self.sliders['fill_alpha'].valtext.set_color('white')
        self.sliders['fill_alpha'].on_changed(self._on_fill_alpha_change)
        
        # Depth slider
        ax_depth = self.fig.add_axes([0.88, 0.35, 0.03, 0.25])
        self.sliders['depth'] = Slider(
            ax_depth, 'Depth', 1, 6, 
            valinit=self._geometry_depth,
            valstep=1,
            orientation='vertical',
            color='#ff8844'
        )
        self.sliders['depth'].label.set_color('white')
        self.sliders['depth'].valtext.set_color('white')
        self.sliders['depth'].on_changed(self._on_depth_change)
        
        # Zoom slider
        ax_zoom = self.fig.add_axes([0.93, 0.35, 0.03, 0.25])
        self.sliders['zoom'] = Slider(
            ax_zoom, 'Zoom', 0.2, 3.0, 
            valinit=self._zoom_level,
            orientation='vertical',
            color='#88ff44'
        )
        self.sliders['zoom'].label.set_color('white')
        self.sliders['zoom'].valtext.set_color('white')
        self.sliders['zoom'].on_changed(self._on_zoom_change)
        
        # Explore button
        ax_explore = self.fig.add_axes([0.53, 0.02, 0.07, 0.04])
        self.btn_explore = Button(ax_explore, 'Explore', color='#333366', hovercolor='#444488')
        self.btn_explore.label.set_color('white')
        self.btn_explore.label.set_fontsize(9)
        self.btn_explore.on_clicked(self._toggle_explore_mode)
        
        # Help text
        help_text = "Keys: 1-4 modes | r rotate | g geometry | e explore | d depth+/- | +/- zoom | q quit"
        self.fig.text(0.62, 0.025, help_text, color='#888888', fontsize=7)
    
    def _toggle_geometry(self, event=None):
        """Toggle geometry overlay on/off."""
        self._show_geometry = not self._show_geometry
        # Update button color
        if self._show_geometry:
            self.btn_geometry.color = '#336633'
            self.btn_geometry.hovercolor = '#448844'
        else:
            self.btn_geometry.color = '#663333'
            self.btn_geometry.hovercolor = '#884444'
        self._render_current_mode()
    
    def _toggle_auto_depth(self, event=None):
        """Toggle auto-depth adjustment based on zoom."""
        self._auto_depth = not self._auto_depth
        if self._auto_depth:
            self.btn_auto_depth.color = '#336633'
            self.btn_auto_depth.hovercolor = '#448844'
            self._update_auto_depth()
        else:
            self.btn_auto_depth.color = '#663333'
            self.btn_auto_depth.hovercolor = '#884444'
        self._render_current_mode()
    
    def _update_auto_depth(self):
        """Update depth based on zoom level."""
        if not self._auto_depth:
            return
        # More zoom = more detail needed
        # zoom 0.5 -> depth 4-5, zoom 1.0 -> depth 3, zoom 2.0 -> depth 2
        auto_depth = max(1, min(6, int(4 - self._zoom_level)))
        if auto_depth != self._geometry_depth:
            self._geometry_depth = auto_depth
            if 'depth' in self.sliders:
                self.sliders['depth'].set_val(auto_depth)
    
    def _on_edge_alpha_change(self, val):
        """Handle edge opacity slider change."""
        self._geometry_alpha = val
        self._render_current_mode()
    
    def _on_fill_alpha_change(self, val):
        """Handle fill opacity slider change."""
        self._geometry_fill_alpha = val
        self._render_current_mode()
    
    def _on_depth_change(self, val):
        """Handle depth slider change."""
        self._geometry_depth = int(val)
        self._auto_depth = False  # Disable auto when manually changed
        self.btn_auto_depth.color = '#663333'
        self._render_current_mode()
    
    def _on_zoom_change(self, val):
        """Handle zoom slider change."""
        self._zoom_level = val
        if self._auto_depth:
            self._update_auto_depth()
        self._apply_zoom()
        self.fig.canvas.draw_idle()
    
    def _toggle_explore_mode(self, event=None):
        """Toggle exploration mode - discover valid truth space empirically."""
        self._explore_mode = not self._explore_mode
        
        if self._explore_mode:
            self.btn_explore.color = '#336699'
            self.btn_explore.hovercolor = '#4477aa'
            print("Exploration mode ON - discovering valid truth space...")
            self._run_exploration()
        else:
            self.btn_explore.color = '#333366'
            self.btn_explore.hovercolor = '#444488'
            self._explored_points = []
            print("Exploration mode OFF")
        
        self._render_current_mode()
    
    def _run_exploration(self, max_points: int = 3000):
        """Run truth space exploration from current path position."""
        # Create explorer
        exp_config = ExplorationConfig(
            step_size=0.015,
            max_points=max_points,
            branch_factor=12,
            min_distance=0.008,
        )
        
        validity_fn = create_mathematical_validity_fn(self._exploration_constraint)
        self._explorer = TruthSpaceExplorer(exp_config, validity_fn=validity_fn)
        
        # Start from center of current path or default center
        if self.visualizer._paths and self.visualizer._paths[0]['positions']:
            start = np.mean(self.visualizer._paths[0]['positions'], axis=0)
        else:
            start = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
        
        # Normalize
        start = start / np.sum(start)
        
        # Explore
        results = self._explorer.explore_from(start, max_iterations=max_points)
        self._explored_points = results['valid_points']
        
        print(f"Discovered {len(self._explored_points)} valid points")
    
    def _set_mode(self, mode):
        """Set visualization mode."""
        self._current_mode = mode
        self._render_current_mode()
    
    def _render_current_mode(self):
        """Render the current mode."""
        self.ax_main.clear()
        
        if self._current_mode == ProjectionMode.PYRAMID_3D:
            self._render_3d()
        elif self._current_mode == ProjectionMode.TRIANGLE_2D:
            self._render_2d()
        elif self._current_mode == ProjectionMode.LINE_1D:
            self._render_1d()
        elif self._current_mode == ProjectionMode.FULL_6D:
            self._render_6d()
        
        self.fig.canvas.draw_idle()
    
    def _render_3d(self):
        """Render 3D view."""
        # Need to recreate 3D axes - safely remove old one
        try:
            self.ax_main.remove()
        except:
            pass
        
        self.ax_main = self.fig.add_axes([0.05, 0.12, 0.78, 0.83], projection='3d')
        self.ax_main.set_facecolor(self.config.background_color)
        
        # Draw explored points if in explore mode
        if self._explore_mode and len(self._explored_points) > 0:
            self._draw_explored_points_3d()
        # Draw Sierpiński pyramid geometry if enabled (and not in explore mode)
        elif self._show_geometry:
            self._draw_sierpinski_3d(self._geometry_depth)
        
        # Draw anchor points
        for anchor, pos in self.visualizer.ANCHOR_POSITIONS_3D.items():
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            self.ax_main.scatter(*pos, c=color, s=100, marker='o', alpha=0.8, zorder=10)
            self.ax_main.text(pos[0]*1.15, pos[1]*1.15, pos[2]*1.15,
                            anchor[:3], color=color, fontsize=9, zorder=11)
        
        # Draw paths (on top of geometry)
        for path_data in self.visualizer._paths:
            positions_3d = [self.visualizer._project_to_3d(p) 
                          for p in path_data['positions']]
            if len(positions_3d) > 1:
                xs, ys, zs = zip(*positions_3d)
                self.ax_main.plot(xs, ys, zs, c=self.config.path_color,
                                linewidth=3, alpha=0.9, label=path_data['label'], zorder=20)
                self.ax_main.scatter(xs, ys, zs, c=self.config.path_color, s=60, zorder=21)
        
        # Draw points (on top)
        for point in self.visualizer._points:
            pos_3d = self.visualizer._project_to_3d(point['position'])
            self.ax_main.scatter(*pos_3d, c=point['color'], s=120, marker='*', zorder=22)
            if point['label']:
                self.ax_main.text(*pos_3d, f"  {point['label']}", 
                                color=point['color'], fontsize=9, zorder=23)
        
        self.ax_main.set_xlabel('X', color='white')
        self.ax_main.set_ylabel('Y', color='white')
        self.ax_main.set_zlabel('Z', color='white')
        self.ax_main.tick_params(colors='white')
        self.ax_main.view_init(elev=31.7, azim=58.3 + self._rotation_angle)
        
        # Setup fixed axes
        self.ax_main.set_box_aspect([1, 1, 1])
        self._apply_zoom()
        
        # Remove pane edges for cleaner look
        self.ax_main.xaxis.pane.fill = False
        self.ax_main.yaxis.pane.fill = False
        self.ax_main.zaxis.pane.fill = False
        self.ax_main.xaxis.pane.set_edgecolor('none')
        self.ax_main.yaxis.pane.set_edgecolor('none')
        self.ax_main.zaxis.pane.set_edgecolor('none')
        
        if self.visualizer._paths:
            self.ax_main.legend(facecolor='#222222', edgecolor='white',
                              labelcolor='white', loc='upper left')
    
    def _draw_sierpinski_3d(self, depth: int):
        """Draw Sierpiński pyramid with current settings."""
        vertices = np.array(list(self.visualizer.ANCHOR_POSITIONS_3D.values()))[:4]
        self._draw_sierpinski_recursive_3d(vertices, depth)
    
    def _draw_sierpinski_recursive_3d(self, vertices: np.ndarray, depth: int):
        """Recursively draw Sierpiński tetrahedron."""
        if depth == 0:
            # Draw edges
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            for i, j in edges:
                self.ax_main.plot(
                    [vertices[i,0], vertices[j,0]],
                    [vertices[i,1], vertices[j,1]],
                    [vertices[i,2], vertices[j,2]],
                    c=self.config.geometry_color,
                    linewidth=self.config.geometry_linewidth,
                    alpha=self._geometry_alpha,
                    zorder=1
                )
            
            # Draw filled faces
            if self._geometry_fill_alpha > 0.001:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
                for face in faces:
                    verts = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                    poly = Poly3DCollection([verts], alpha=self._geometry_fill_alpha, zorder=0)
                    poly.set_facecolor(self.config.geometry_color)
                    poly.set_edgecolor('none')
                    self.ax_main.add_collection3d(poly)
            return
        
        # Recurse into 4 sub-tetrahedra
        for i in range(4):
            sub_verts = [vertices[i]]
            for j in range(4):
                if j != i:
                    sub_verts.append((vertices[i] + vertices[j]) / 2)
            self._draw_sierpinski_recursive_3d(np.array(sub_verts), depth - 1)
    
    def _draw_explored_points_3d(self):
        """Draw empirically discovered valid points in 3D."""
        if not self._explored_points.any():
            return
        
        # Project all points to 3D
        points_3d = np.array([self.visualizer._project_to_3d(p) for p in self._explored_points])
        
        # Draw as point cloud
        self.ax_main.scatter(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            c='#00ff88', s=2, alpha=0.4, zorder=1,
            label=f'Discovered ({len(self._explored_points)})'
        )
    
    def _draw_explored_points_2d(self):
        """Draw empirically discovered valid points in 2D."""
        if not self._explored_points.any():
            return
        
        # Project all points to 2D
        points_2d = np.array([self.visualizer._project_to_2d(p) for p in self._explored_points])
        
        # Draw as point cloud
        self.ax_main.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c='#00ff88', s=2, alpha=0.4,
            label=f'Discovered ({len(self._explored_points)})'
        )
    
    def _render_2d(self):
        """Render 2D view."""
        try:
            self.ax_main.remove()
        except:
            pass
        
        self.ax_main = self.fig.add_axes([0.05, 0.12, 0.78, 0.83])
        self.ax_main.set_facecolor(self.config.background_color)
        
        # Draw explored points if in explore mode
        if self._explore_mode and len(self._explored_points) > 0:
            self._draw_explored_points_2d()
        # Draw Sierpiński triangle geometry if enabled
        elif self._show_geometry:
            self._draw_sierpinski_2d(self._geometry_depth + 2)  # 2D needs more depth
        
        # Draw anchor points
        for anchor, pos in self.visualizer.ANCHOR_POSITIONS_2D.items():
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            self.ax_main.scatter(*pos, c=color, s=100, marker='o', alpha=0.8)
            self.ax_main.annotate(anchor[:3], pos, color=color, fontsize=9,
                                xytext=(5, 5), textcoords='offset points')
        
        # Draw paths
        for path_data in self.visualizer._paths:
            positions_2d = [self.visualizer._project_to_2d(p) 
                          for p in path_data['positions']]
            if len(positions_2d) > 1:
                xs, ys = zip(*positions_2d)
                self.ax_main.plot(xs, ys, c=self.config.path_color,
                                linewidth=2, alpha=0.8, label=path_data['label'])
                self.ax_main.scatter(xs, ys, c=self.config.path_color, s=50)
        
        # Draw points
        for point in self.visualizer._points:
            pos_2d = self.visualizer._project_to_2d(point['position'])
            self.ax_main.scatter(*pos_2d, c=point['color'], s=100, marker='*')
            if point['label']:
                self.ax_main.annotate(point['label'], pos_2d, color=point['color'])
        
        self.ax_main.set_xlim(-0.1, 1.1)
        self.ax_main.set_ylim(-0.1, 1.0)
        self.ax_main.set_aspect('equal')
        self.ax_main.tick_params(colors='white')
        
        if self.visualizer._paths:
            self.ax_main.legend(facecolor='#222222', edgecolor='white',
                              labelcolor='white', loc='upper left')
    
    def _draw_sierpinski_2d(self, depth: int):
        """Draw Sierpiński triangle with current settings."""
        vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]])
        self._draw_sierpinski_recursive_2d(vertices, depth)
    
    def _draw_sierpinski_recursive_2d(self, vertices: np.ndarray, depth: int):
        """Recursively draw Sierpiński triangle."""
        if depth == 0:
            if self._geometry_fill_alpha > 0.001:
                triangle = plt.Polygon(vertices, fill=True,
                                      facecolor=self.config.geometry_color,
                                      edgecolor=self.config.geometry_color,
                                      linewidth=self.config.geometry_linewidth,
                                      alpha=self._geometry_fill_alpha)
            else:
                triangle = plt.Polygon(vertices, fill=False,
                                      edgecolor=self.config.geometry_color,
                                      linewidth=self.config.geometry_linewidth,
                                      alpha=self._geometry_alpha)
            self.ax_main.add_patch(triangle)
            return
        
        # Midpoints
        mid01 = (vertices[0] + vertices[1]) / 2
        mid12 = (vertices[1] + vertices[2]) / 2
        mid02 = (vertices[0] + vertices[2]) / 2
        
        # Three sub-triangles
        self._draw_sierpinski_recursive_2d(np.array([vertices[0], mid01, mid02]), depth - 1)
        self._draw_sierpinski_recursive_2d(np.array([mid01, vertices[1], mid12]), depth - 1)
        self._draw_sierpinski_recursive_2d(np.array([mid02, mid12, vertices[2]]), depth - 1)
    
    def _render_1d(self):
        """Render 1D view."""
        try:
            self.ax_main.remove()
        except:
            pass
        
        self.ax_main = self.fig.add_axes([0.05, 0.12, 0.78, 0.83])
        self.ax_main.set_facecolor(self.config.background_color)
        
        # Draw Cantor-like structure
        self.ax_main.axhline(y=0.5, color='#333333', linewidth=2, alpha=0.5)
        
        # Draw anchor positions on line
        anchor_1d = {
            'IDENTITY': 0.0, 'STABILITY': 0.2, 'INVERSE': 0.4,
            'UNITY': 0.6, 'PATTERN': 0.8, 'GROWTH': 1.0
        }
        for anchor, x in anchor_1d.items():
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            self.ax_main.scatter(x, 0.5, c=color, s=100, marker='|')
            self.ax_main.text(x, 0.55, anchor[:3], color=color, 
                            ha='center', fontsize=9)
        
        # Draw paths
        for path_data in self.visualizer._paths:
            positions_1d = [self.visualizer._project_to_1d(p) 
                          for p in path_data['positions']]
            self.ax_main.scatter(positions_1d, [0.5]*len(positions_1d),
                               c=self.config.path_color, s=150, marker='|')
            if len(positions_1d) > 1:
                self.ax_main.plot(positions_1d, [0.5]*len(positions_1d),
                                c=self.config.path_color, linewidth=2, alpha=0.5)
        
        self.ax_main.set_xlim(-0.1, 1.1)
        self.ax_main.set_ylim(0, 1)
        self.ax_main.set_yticks([])
        self.ax_main.tick_params(colors='white')
    
    def _render_6d(self):
        """Render 6D parallel coordinates."""
        try:
            self.ax_main.remove()
        except:
            pass
        
        self.ax_main = self.fig.add_axes([0.05, 0.12, 0.78, 0.83])
        self.ax_main.set_facecolor(self.config.background_color)
        
        anchors = ['IDENTITY', 'STABILITY', 'INVERSE', 'UNITY', 'PATTERN', 'GROWTH']
        
        # Draw axes
        for i, anchor in enumerate(anchors):
            color = self.config.anchor_colors.get(anchor, '#ffffff')
            self.ax_main.axvline(x=i, color=color, alpha=0.4, linewidth=2)
            self.ax_main.text(i, -0.12, anchor[:3], color=color, 
                            ha='center', fontsize=10)
        
        # Draw paths
        for path_data in self.visualizer._paths:
            for pos in path_data['positions']:
                self.ax_main.plot(range(6), pos, c=self.config.path_color,
                                linewidth=2, alpha=0.7, marker='o', markersize=6)
        
        self.ax_main.set_xlim(-0.5, 5.5)
        self.ax_main.set_ylim(-0.15, 1.1)
        self.ax_main.tick_params(colors='white')
        self.ax_main.set_ylabel('Weight', color='white')
    
    def _toggle_rotation(self, event=None):
        """Toggle camera rotation."""
        self._rotating = not self._rotating
        if self._rotating:
            self._animate_rotation()
    
    def _animate_rotation(self):
        """Animate camera rotation."""
        if self._rotating and self._current_mode == ProjectionMode.PYRAMID_3D:
            self._rotation_angle += 2
            self._render_current_mode()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.05)
            if self._rotating:
                self.fig.canvas.manager.window.after(50, self._animate_rotation)
    
    def _show_multi_view(self, event=None):
        """Show multi-view window."""
        self.visualizer.multi_view()
        plt.show()
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == '1':
            self._set_mode(ProjectionMode.PYRAMID_3D)
        elif event.key == '2':
            self._set_mode(ProjectionMode.TRIANGLE_2D)
        elif event.key == '3':
            self._set_mode(ProjectionMode.LINE_1D)
        elif event.key == '4':
            self._set_mode(ProjectionMode.FULL_6D)
        elif event.key == 'r':
            self._toggle_rotation()
        elif event.key == 'g':
            # Toggle geometry
            self._toggle_geometry()
        elif event.key == 'e':
            # Toggle explore mode
            self._toggle_explore_mode()
        elif event.key == 'a':
            # Toggle auto-depth
            self._toggle_auto_depth()
        elif event.key == 'd':
            # Increase depth
            self._geometry_depth = min(6, self._geometry_depth + 1)
            self._auto_depth = False
            if 'depth' in self.sliders:
                self.sliders['depth'].set_val(self._geometry_depth)
            self._render_current_mode()
        elif event.key == 'D':
            # Decrease depth (shift+d)
            self._geometry_depth = max(1, self._geometry_depth - 1)
            self._auto_depth = False
            if 'depth' in self.sliders:
                self.sliders['depth'].set_val(self._geometry_depth)
            self._render_current_mode()
        elif event.key in ['+', '=']:
            self._zoom_level *= 0.8  # Zoom in
            self._zoom_level = max(0.2, self._zoom_level)
            if self._auto_depth:
                self._update_auto_depth()
            if 'zoom' in self.sliders:
                self.sliders['zoom'].set_val(self._zoom_level)
            self._apply_zoom()
            self.fig.canvas.draw_idle()
        elif event.key in ['-', '_']:
            self._zoom_level *= 1.25  # Zoom out
            self._zoom_level = min(5.0, self._zoom_level)
            if self._auto_depth:
                self._update_auto_depth()
            if 'zoom' in self.sliders:
                self.sliders['zoom'].set_val(self._zoom_level)
            self._apply_zoom()
            self.fig.canvas.draw_idle()
        elif event.key == '0':
            self._zoom_level = 1.0  # Reset zoom
            if 'zoom' in self.sliders:
                self.sliders['zoom'].set_val(self._zoom_level)
            self._apply_zoom()
            self.fig.canvas.draw_idle()
        elif event.key == 'v':
            # Reset view angle
            self._rotation_angle = 0
            self._render_current_mode()
        elif event.key == 'q':
            plt.close('all')


def demo():
    """Demonstrate the interactive viewer."""
    print("Interactive Truth Space Viewer Demo")
    print("=" * 50)
    print("Controls:")
    print("  1-4: Switch projection modes")
    print("  r: Toggle rotation")
    print("  g: Toggle geometry overlay")
    print("  e: Toggle EXPLORE mode (discover valid space)")
    print("  a: Toggle auto-depth (zoom-based)")
    print("  d/D: Increase/decrease fractal depth")
    print("  +/-: Zoom in/out")
    print("  0: Reset zoom")
    print("  v: Reset view angle")
    print("  Scroll wheel: Zoom")
    print("  Sliders: Edge/Fill opacity, Depth, Zoom")
    print("  q: Quit")
    print()
    print("EXPLORE MODE: Discovers the actual valid truth space")
    print("              by testing points empirically!")
    print()
    
    viewer = InteractiveTruthSpaceViewer()
    
    # Add sample path
    path = [
        {'IDENTITY': 0.05, 'STABILITY': 0.05, 'INVERSE': 0.05, 
         'UNITY': 0.05, 'PATTERN': 0.2, 'GROWTH': 0.6},
        {'IDENTITY': 0.2, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.2, 'PATTERN': 0.2, 'GROWTH': 0.2},
        {'IDENTITY': 0.1, 'STABILITY': 0.2, 'INVERSE': 0.1, 
         'UNITY': 0.35, 'PATTERN': 0.15, 'GROWTH': 0.1},
    ]
    
    import numpy as np
    positions = [np.array(list(p.values())) for p in path]
    
    viewer.visualizer._paths.append({
        'positions': positions,
        'label': 'φ × φ → φ + 1'
    })
    
    # Add key points
    viewer.add_point(path[0], "Start: φ×φ", "#ff6600")
    viewer.add_point(path[-1], "End: φ+1", "#00ff00")
    
    viewer.show()


if __name__ == "__main__":
    demo()
