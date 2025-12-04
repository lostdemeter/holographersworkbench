#!/usr/bin/env python3
"""
Truth Space Visualization Runner
================================

Run truth space visualizations from the command line.

Usage:
    python run_viz.py                     # Interactive multi-view
    python run_viz.py --mode 3d           # 3D Sierpiński pyramid
    python run_viz.py --mode 2d           # 2D Sierpiński triangle
    python run_viz.py --mode 1d           # 1D Cantor projection
    python run_viz.py --mode 6d           # 6D parallel coordinates
    python run_viz.py --mode multi        # All views at once
    python run_viz.py --interactive       # Interactive viewer with controls
    python run_viz.py --animate           # Animated path
    python run_viz.py --query "phi * phi" # Visualize a specific query
    python run_viz.py --save output.png   # Save to file

Examples:
    python run_viz.py --query "sin(x)**2 + cos(x)**2" --mode 3d
    python run_viz.py --animate --save animation.gif
    python run_viz.py --interactive
"""

import sys
import os
import argparse
import numpy as np

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from truth_space_viz import TruthSpaceVisualizer, ProjectionMode, VisualizationConfig
from path_animator import PathAnimator, AnimationConfig
from interactive_viewer import InteractiveTruthSpaceViewer
from truth_space_explorer import TruthSpaceExplorer, ExplorationConfig, create_mathematical_validity_fn


def create_sample_path():
    """Create a sample exploration path for φ × φ."""
    return [
        {'IDENTITY': 0.05, 'STABILITY': 0.05, 'INVERSE': 0.05, 
         'UNITY': 0.05, 'PATTERN': 0.2, 'GROWTH': 0.6},
        {'IDENTITY': 0.1, 'STABILITY': 0.05, 'INVERSE': 0.05, 
         'UNITY': 0.1, 'PATTERN': 0.4, 'GROWTH': 0.3},
        {'IDENTITY': 0.2, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.2, 'PATTERN': 0.2, 'GROWTH': 0.2},
        {'IDENTITY': 0.15, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.4, 'PATTERN': 0.15, 'GROWTH': 0.1},
        {'IDENTITY': 0.1, 'STABILITY': 0.2, 'INVERSE': 0.1, 
         'UNITY': 0.35, 'PATTERN': 0.15, 'GROWTH': 0.1},
    ]


def run_navigation(query: str):
    """Run navigation for a query and return result."""
    try:
        from agents.navigator import TruthSpaceNavigator
        navigator = TruthSpaceNavigator(use_llm=False, verbose=True)
        result = navigator.process(query)
        return result.data
    except ImportError:
        print("Could not import navigator, using sample path")
        return None


def get_mode(mode_str: str) -> ProjectionMode:
    """Convert string to ProjectionMode."""
    modes = {
        '3d': ProjectionMode.PYRAMID_3D,
        '2d': ProjectionMode.TRIANGLE_2D,
        '1d': ProjectionMode.LINE_1D,
        '6d': ProjectionMode.FULL_6D,
        '4d': ProjectionMode.TETRIX_4D,
        '5d': ProjectionMode.SLICE_5D,
    }
    return modes.get(mode_str.lower(), ProjectionMode.PYRAMID_3D)


def main():
    parser = argparse.ArgumentParser(
        description="Truth Space Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_viz.py                          # Default multi-view
  python run_viz.py --mode 3d                # 3D pyramid view
  python run_viz.py --query "phi * phi"      # Visualize specific query
  python run_viz.py --interactive            # Interactive controls
  python run_viz.py --animate --save out.gif # Save animation
        """
    )
    
    parser.add_argument('--mode', '-m', 
                       choices=['3d', '2d', '1d', '6d', '4d', '5d', 'multi'],
                       default='multi',
                       help='Visualization mode (default: multi)')
    
    parser.add_argument('--query', '-q', type=str,
                       help='Mathematical query to visualize')
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Launch interactive viewer')
    
    parser.add_argument('--animate', '-a', action='store_true',
                       help='Create animated visualization')
    
    parser.add_argument('--save', '-s', type=str,
                       help='Save visualization to file')
    
    parser.add_argument('--rotate', '-r', action='store_true',
                       help='Enable camera rotation (for animation)')
    
    parser.add_argument('--fps', type=int, default=20,
                       help='Frames per second for animation (default: 20)')
    
    parser.add_argument('--dark', action='store_true', default=True,
                       help='Use dark theme (default: True)')
    
    # Geometry overlay options
    parser.add_argument('--geometry-alpha', '-ga', type=float, default=0.15,
                       help='Opacity of geometry edges (0-1, default: 0.15)')
    
    parser.add_argument('--geometry-fill', '-gf', type=float, default=0.05,
                       help='Opacity of geometry fill (0-1, default: 0.05)')
    
    parser.add_argument('--geometry-depth', '-gd', type=int, default=3,
                       help='Fractal recursion depth (default: 3)')
    
    parser.add_argument('--geometry-color', '-gc', type=str, default='#4488ff',
                       help='Color of geometry overlay (default: #4488ff)')
    
    parser.add_argument('--no-geometry', action='store_true',
                       help='Hide geometry overlay')
    
    parser.add_argument('--no-fill', action='store_true',
                       help='Disable geometry fill (wireframe only)')
    
    parser.add_argument('--explore', '-e', action='store_true',
                       help='Run in exploration mode to discover valid truth space')
    
    parser.add_argument('--explore-points', type=int, default=5000,
                       help='Number of points to discover in exploration mode')
    
    parser.add_argument('--explore-zoom', type=float, default=1.0,
                       help='Zoom level for exploration view (higher = more zoomed in)')
    
    parser.add_argument('--explore-view', type=str, default='3d',
                       choices=['3d', 'density', 'slices', 'interactive'],
                       help='Visualization mode for exploration (interactive has zoom slider)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRUTH SPACE VISUALIZATION")
    print("Projecting 6D Hyperbigasket to viewable dimensions")
    print("=" * 60)
    
    # Get path data
    if args.query:
        print(f"\nProcessing query: {args.query}")
        nav_result = run_navigation(args.query)
        if nav_result:
            path_label = args.query
        else:
            # Fallback to sample
            nav_result = None
            path_label = "φ × φ (sample)"
    else:
        nav_result = None
        path_label = "φ × φ → φ + 1 (sample)"
    
    # Create sample path if no navigation result
    sample_path = create_sample_path()
    positions = [np.array(list(p.values())) for p in sample_path]
    
    # Configuration
    config = VisualizationConfig(
        figsize=(12, 10),
        background_color='#0a0a0a' if args.dark else '#ffffff',
        path_color='#00ff88',
        point_color='#ff6600',
        # Geometry overlay settings
        geometry_alpha=args.geometry_alpha,
        geometry_fill_alpha=args.geometry_fill,
        geometry_depth=args.geometry_depth,
        geometry_color=args.geometry_color,
        show_geometry=not args.no_geometry,
        geometry_fill=not args.no_fill,
    )
    
    print(f"\nGeometry overlay: alpha={args.geometry_alpha}, fill={args.geometry_fill}, depth={args.geometry_depth}")
    
    # Exploration mode - discover valid truth space empirically
    if args.explore:
        print("\n" + "=" * 60)
        print("EXPLORATION MODE")
        print("Discovering valid truth space structure empirically...")
        print("=" * 60)
        
        exp_config = ExplorationConfig(
            step_size=0.015,
            max_points=args.explore_points,
            branch_factor=15,
            min_distance=0.008,
        )
        
        validity_fn = create_mathematical_validity_fn("simplex")
        explorer = TruthSpaceExplorer(exp_config, validity_fn=validity_fn)
        
        # Start from path center or default
        start = np.mean(positions, axis=0) if positions else np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
        start = start / np.sum(start)
        
        results = explorer.explore_from(start, max_iterations=args.explore_points)
        
        # Visualize based on view mode
        title = f"Discovered Truth Space ({len(explorer.valid_points)} valid points)"
        
        if args.explore_view == 'density':
            print("\nGenerating density heatmap...")
            fig, ax = explorer.visualize_density(resolution=80, title=title)
        elif args.explore_view == 'slices':
            print("\nGenerating cross-section slices...")
            fig, ax = explorer.visualize_slices(n_slices=9, title=title)
        elif args.explore_view == 'interactive':
            print("\nLaunching interactive 3D viewer with zoom slider...")
            print("Use the slider at bottom to zoom in/out (1-100x)")
            fig, ax = explorer.visualize_3d_interactive(max_points=100000)
        else:
            print(f"\nGenerating 3D view (zoom={args.explore_zoom}x)...")
            fig, ax = explorer.visualize_3d(
                show_invalid=True,
                title=title,
                zoom=args.explore_zoom,
                color_by_depth=True,
                max_points=100000
            )
        
        if args.save:
            fig.savefig(args.save, dpi=150, facecolor=config.background_color)
            print(f"Saved to {args.save}")
        
        import matplotlib.pyplot as plt
        plt.show()
        return
    
    # Interactive mode
    if args.interactive:
        print("\nLaunching interactive viewer...")
        print("Controls:")
        print("  1-4: Switch modes | r: rotate | g: toggle geometry")
        print("  a: auto-depth | d/D: depth +/- | +/-: zoom | v: reset view")
        print("  Sliders on right: Edge/Fill opacity, Depth, Zoom")
        
        viewer = InteractiveTruthSpaceViewer(config)
        
        if nav_result:
            viewer.visualizer.plot_path(nav_result, path_label)
        else:
            viewer.visualizer._paths.append({
                'positions': positions,
                'label': path_label
            })
        
        # Add key points
        viewer.add_point(sample_path[0], "Start", "#ff6600")
        viewer.add_point(sample_path[-1], "End", "#00ff00")
        
        viewer.show()
        return
    
    # Animation mode
    if args.animate:
        print("\nCreating animation...")
        
        anim_config = AnimationConfig(
            fps=args.fps,
            rotate_camera=args.rotate or True,
            rotation_speed=2.0,
            interpolation_steps=15,
        )
        
        animator = PathAnimator(anim_config, config)
        animator.add_path(positions, path_label)
        
        if args.save:
            print(f"Saving to {args.save}...")
            animator.animate_3d(save_path=args.save)
        else:
            animator.animate_3d()
            animator.show()
        return
    
    # Static visualization
    viz = TruthSpaceVisualizer(config)
    
    if nav_result:
        viz.plot_path(nav_result, path_label)
    else:
        viz._paths.append({
            'positions': positions,
            'label': path_label
        })
    
    # Add key points
    viz.add_point(sample_path[0], "Start: φ×φ", "#ff6600")
    viz.add_point(sample_path[2], "Found: φ+1", "#00ff00")
    viz.add_point(sample_path[-1], "Found: 2.618", "#00ffff")
    
    if args.mode == 'multi':
        print("\nRendering multi-view...")
        viz.multi_view()
    else:
        mode = get_mode(args.mode)
        print(f"\nRendering {mode.name} view...")
        viz.set_mode(mode)
        viz.render()
    
    if args.save:
        viz.save(args.save)
        print(f"Saved to {args.save}")
    
    viz.show()


if __name__ == "__main__":
    main()
