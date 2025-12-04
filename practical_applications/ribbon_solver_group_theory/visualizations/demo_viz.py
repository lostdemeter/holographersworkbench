#!/usr/bin/env python3
"""
Truth Space Visualization Demo
==============================

Demonstrates visualizing navigation paths through the 6D truth space
hyperbigasket using various projection modes.

The truth space has perfect self-similarity:
    6D Hyperbigasket
         ↓
    5D Hyperbigasket slice
         ↓
    4D Tetrix
         ↓
    3D Sierpiński Pyramid
         ↓
    2D Sierpiński Triangle
         ↓
    1D Cantor Set

Usage:
    python demo_viz.py [--mode MODE] [--animate] [--save PATH]
    
    Modes: 3d, 2d, 1d, 6d, multi
"""

import sys
import os
import argparse
import numpy as np

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truth_space_viz import TruthSpaceVisualizer, ProjectionMode, VisualizationConfig
from path_animator import PathAnimator, AnimationConfig


def create_sample_exploration():
    """Create a sample exploration path for φ × φ."""
    # Simulated path through truth space
    # Starting from growth-dominant (phi * phi)
    # Moving through various regions
    
    path = [
        # Start: φ × φ (growth dominant)
        {'IDENTITY': 0.05, 'STABILITY': 0.05, 'INVERSE': 0.05, 
         'UNITY': 0.05, 'PATTERN': 0.2, 'GROWTH': 0.6},
        
        # Compose with PATTERN generator
        {'IDENTITY': 0.1, 'STABILITY': 0.05, 'INVERSE': 0.05, 
         'UNITY': 0.1, 'PATTERN': 0.4, 'GROWTH': 0.3},
        
        # Found: φ + 1 (more balanced)
        {'IDENTITY': 0.2, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.2, 'PATTERN': 0.2, 'GROWTH': 0.2},
        
        # Compose with UNITY
        {'IDENTITY': 0.15, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.4, 'PATTERN': 0.15, 'GROWTH': 0.1},
        
        # Found: ≈ 2.618 (numerical, near unity)
        {'IDENTITY': 0.1, 'STABILITY': 0.2, 'INVERSE': 0.1, 
         'UNITY': 0.35, 'PATTERN': 0.15, 'GROWTH': 0.1},
    ]
    
    return path


def create_pythagorean_exploration():
    """Create exploration path for sin²(x) + cos²(x)."""
    path = [
        # Start: sin²(x) + cos²(x) (pattern dominant)
        {'IDENTITY': 0.1, 'STABILITY': 0.1, 'INVERSE': 0.1, 
         'UNITY': 0.2, 'PATTERN': 0.4, 'GROWTH': 0.1},
        
        # Recognize cyclic pattern
        {'IDENTITY': 0.15, 'STABILITY': 0.15, 'INVERSE': 0.1, 
         'UNITY': 0.25, 'PATTERN': 0.3, 'GROWTH': 0.05},
        
        # Simplify toward unity
        {'IDENTITY': 0.2, 'STABILITY': 0.2, 'INVERSE': 0.1, 
         'UNITY': 0.35, 'PATTERN': 0.1, 'GROWTH': 0.05},
        
        # Found: 1 (pure unity)
        {'IDENTITY': 0.1, 'STABILITY': 0.3, 'INVERSE': 0.05, 
         'UNITY': 0.5, 'PATTERN': 0.03, 'GROWTH': 0.02},
    ]
    
    return path


def demo_static_views():
    """Demonstrate static visualization views."""
    print("\n" + "=" * 60)
    print("STATIC VISUALIZATION DEMO")
    print("=" * 60)
    
    # Create visualizer
    config = VisualizationConfig(
        figsize=(10, 8),
        background_color='#0a0a0a',
        path_color='#00ff88',
    )
    viz = TruthSpaceVisualizer(config)
    
    # Add exploration paths
    phi_path = create_sample_exploration()
    pyth_path = create_pythagorean_exploration()
    
    # Convert to navigation result format
    viz.plot_path({
        'start': {'position': phi_path[0]},
        'interesting_finds': []
    }, "φ × φ → φ + 1")
    
    # Add key points
    viz.add_point(phi_path[0], "Start: φ×φ", "#ff6600")
    viz.add_point(phi_path[2], "Found: φ+1", "#00ff00")
    viz.add_point(phi_path[-1], "Found: 2.618", "#00ffff")
    
    print("\n1. 3D Sierpiński Pyramid View")
    viz.set_mode(ProjectionMode.PYRAMID_3D)
    viz.render()
    viz.show()
    
    print("\n2. 2D Sierpiński Triangle View")
    viz.clear()
    viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ × φ")
    viz.set_mode(ProjectionMode.TRIANGLE_2D)
    viz.render()
    viz.show()
    
    print("\n3. 6D Parallel Coordinates View")
    viz.clear()
    viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ × φ")
    viz.set_mode(ProjectionMode.FULL_6D)
    viz.render()
    viz.show()


def demo_multi_view():
    """Demonstrate multi-view visualization."""
    print("\n" + "=" * 60)
    print("MULTI-VIEW DEMO")
    print("=" * 60)
    
    viz = TruthSpaceVisualizer()
    
    # Add both paths
    phi_path = create_sample_exploration()
    pyth_path = create_pythagorean_exploration()
    
    for pos in phi_path:
        viz._paths.append({
            'positions': [np.array(list(pos.values()))],
            'label': 'φ²'
        })
    
    # Create multi-view
    viz.multi_view()
    viz.show()


def demo_animation():
    """Demonstrate animated visualization."""
    print("\n" + "=" * 60)
    print("ANIMATION DEMO")
    print("=" * 60)
    
    config = AnimationConfig(
        fps=20,
        rotate_camera=True,
        rotation_speed=2.0,
        interpolation_steps=20,
    )
    
    animator = PathAnimator(config)
    
    # Create path as numpy arrays
    phi_path = create_sample_exploration()
    positions = [np.array(list(p.values())) for p in phi_path]
    
    animator.add_path(positions, "φ × φ exploration")
    animator.animate_3d()
    animator.show()


def demo_with_agents():
    """Demonstrate visualization integrated with agents."""
    print("\n" + "=" * 60)
    print("AGENT INTEGRATION DEMO")
    print("=" * 60)
    
    try:
        from agents.orchestrator import TruthSpaceOrchestrator
        from agents.navigator import TruthSpaceNavigator
        
        # Run navigation
        print("\nRunning truth space navigation...")
        navigator = TruthSpaceNavigator(use_llm=False, verbose=True)
        result = navigator.process("phi * phi")
        
        # Visualize the result
        print("\nVisualizing navigation path...")
        viz = TruthSpaceVisualizer()
        viz.plot_path(result.data, "φ × φ navigation")
        
        # Add interesting finds as points
        for find in result.data.get('interesting_finds', []):
            if 'location' in find:
                loc = find['location']
                viz.add_point(loc.position, find.get('expression', ''), '#00ffff')
        
        viz.set_mode(ProjectionMode.PYRAMID_3D)
        viz.render()
        viz.show()
        
    except ImportError as e:
        print(f"Could not import agents: {e}")
        print("Running standalone demo instead...")
        demo_static_views()


def main():
    parser = argparse.ArgumentParser(description="Truth Space Visualization Demo")
    parser.add_argument('--mode', choices=['3d', '2d', '1d', '6d', 'multi', 'animate', 'agents'],
                       default='multi', help='Visualization mode')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRUTH SPACE VISUALIZATION")
    print("Projecting 6D Hyperbigasket to viewable dimensions")
    print("=" * 60)
    
    if args.mode == 'multi':
        demo_multi_view()
    elif args.mode == 'animate':
        demo_animation()
    elif args.mode == 'agents':
        demo_with_agents()
    elif args.mode == '3d':
        viz = TruthSpaceVisualizer()
        phi_path = create_sample_exploration()
        viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ²")
        viz.set_mode(ProjectionMode.PYRAMID_3D)
        viz.render()
        if args.save:
            viz.save(args.save)
        viz.show()
    elif args.mode == '2d':
        viz = TruthSpaceVisualizer()
        phi_path = create_sample_exploration()
        viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ²")
        viz.set_mode(ProjectionMode.TRIANGLE_2D)
        viz.render()
        if args.save:
            viz.save(args.save)
        viz.show()
    elif args.mode == '1d':
        viz = TruthSpaceVisualizer()
        phi_path = create_sample_exploration()
        viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ²")
        viz.set_mode(ProjectionMode.LINE_1D)
        viz.render()
        if args.save:
            viz.save(args.save)
        viz.show()
    elif args.mode == '6d':
        viz = TruthSpaceVisualizer()
        phi_path = create_sample_exploration()
        viz.plot_path({'start': {'position': phi_path[0]}, 'interesting_finds': []}, "φ²")
        viz.set_mode(ProjectionMode.FULL_6D)
        viz.render()
        if args.save:
            viz.save(args.save)
        viz.show()


if __name__ == "__main__":
    main()
