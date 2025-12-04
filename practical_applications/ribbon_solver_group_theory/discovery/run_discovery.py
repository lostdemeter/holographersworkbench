#!/usr/bin/env python3
"""
Run Truth Structure Discovery

Discovers mathematical rules governing truth space by analyzing
explored regions and finding patterns.

Usage:
    python run_discovery.py                    # Basic discovery
    python run_discovery.py --samples 10000    # More samples
    python run_discovery.py --constraint balanced  # Use specific constraint
    python run_discovery.py --visualize        # Show discovered structure
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.truth_structure_discovery import (
    TruthStructureDiscovery,
    DiscoveryConfig,
    ErrorAsSignalAnalyzer,
)
from visualizations.truth_space_explorer import (
    TruthSpaceExplorer,
    ExplorationConfig,
    create_mathematical_validity_fn,
)


def run_discovery(args):
    """Run the discovery process."""
    print("=" * 60)
    print("TRUTH STRUCTURE DISCOVERY ENGINE")
    print("=" * 60)
    
    # Create validity function based on constraint type
    print(f"\nUsing constraint: {args.constraint}")
    validity_fn = create_mathematical_validity_fn(args.constraint)
    
    # Configure discovery
    config = DiscoveryConfig(
        n_boundary_samples=args.samples,
        n_interior_samples=args.samples // 2,
        residual_threshold=args.threshold,
        min_confidence=args.confidence,
        test_golden_ratios=True,
        test_symmetries=True,
    )
    
    # Run discovery
    discovery = TruthStructureDiscovery(config)
    relations = discovery.discover(validity_fn, verbose=True)
    
    # Print full report
    print("\n" + discovery.report())
    
    # Analyze coordinate distributions
    print("\n" + "=" * 60)
    print("COORDINATE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    all_points = discovery.boundary_points + discovery.interior_points
    if all_points:
        points_array = np.array(all_points)
        
        print("\nAnchor Statistics:")
        print("-" * 50)
        print(f"{'Anchor':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("-" * 50)
        
        for i, anchor in enumerate(discovery.ANCHORS):
            vals = points_array[:, i]
            print(f"{anchor:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")
        
        # Check correlations
        print("\nCorrelation Matrix:")
        print("-" * 50)
        corr = np.corrcoef(points_array.T)
        
        # Find strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        for i in range(6):
            for j in range(i + 1, 6):
                if abs(corr[i, j]) > 0.5:
                    print(f"  {discovery.ANCHORS[i]} ↔ {discovery.ANCHORS[j]}: "
                          f"r = {corr[i, j]:.3f}")
    
    # Error-as-Signal Analysis (inspired by Ribbon LCM v4)
    print("\n" + "=" * 60)
    print("ERROR-AS-SIGNAL ANALYSIS")
    print("(Inspired by Ribbon LCM v4 φ-BBP Discovery)")
    print("=" * 60)
    
    error_analyzer = ErrorAsSignalAnalyzer()
    print(error_analyzer.report(all_points))
    
    # Find symmetries
    if args.symmetries:
        print("\n" + "=" * 60)
        print("SYMMETRY ANALYSIS")
        print("=" * 60)
        
        symmetries = discovery.find_symmetry_groups(validity_fn, n_tests=500)
        if symmetries:
            print(f"\nFound {len(symmetries)} symmetry permutations:")
            for perm in symmetries[:10]:
                # Describe the permutation
                swaps = []
                for i, j in enumerate(perm):
                    if i != j:
                        swaps.append(f"{discovery.ANCHORS[i]}↔{discovery.ANCHORS[j]}")
                if swaps:
                    print(f"  {', '.join(swaps[:3])}")
        else:
            print("\nNo symmetries found (space may be asymmetric)")
    
    # Visualize if requested
    if args.visualize:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION")
        print("=" * 60)
        
        import matplotlib.pyplot as plt
        
        # Use the already-discovered boundary and interior points!
        # No need to re-explore - we already have the data
        all_points = discovery.boundary_points + discovery.interior_points
        
        if len(all_points) < 10:
            print("Not enough points for visualization. Try increasing --samples")
        else:
            print(f"\nVisualizing {len(all_points)} discovered points...")
            
            # Create explorer and inject the discovered points
            exp_config = ExplorationConfig()
            explorer = TruthSpaceExplorer(exp_config, validity_fn=validity_fn)
            explorer.valid_points = all_points
            
            # Create visualization
            fig, ax = explorer.visualize_density(
                resolution=60,
                title=f"Truth Space Structure ({args.constraint} constraint, {len(all_points)} points)"
            )
        
        if args.save:
            fig.savefig(args.save, dpi=150, facecolor='#0a0a0a')
            print(f"\nSaved visualization to {args.save}")
        
        plt.show()
    
    # Save results if requested
    if args.output:
        import json
        
        results = {
            'constraint': args.constraint,
            'n_boundary_points': len(discovery.boundary_points),
            'n_interior_points': len(discovery.interior_points),
            'relations': [
                {
                    'name': r.name,
                    'description': r.description,
                    'confidence': r.confidence,
                }
                for r in relations
            ],
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to {args.output}")
    
    return discovery, relations


def main():
    parser = argparse.ArgumentParser(
        description="Discover mathematical structure in truth space"
    )
    
    parser.add_argument('--constraint', '-c', type=str, default='simplex',
                       choices=['simplex', 'balanced', 'group_closure', 'golden', 'sierpinski'],
                       help='Base constraint type to analyze')
    
    parser.add_argument('--samples', '-n', type=int, default=5000,
                       help='Number of boundary samples')
    
    parser.add_argument('--threshold', '-t', type=float, default=0.02,
                       help='Residual threshold for relation detection')
    
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Minimum confidence to report a relation')
    
    parser.add_argument('--symmetries', '-s', action='store_true',
                       help='Analyze symmetry groups')
    
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization')
    
    parser.add_argument('--viz-points', type=int, default=10000,
                       help='Points for visualization')
    
    parser.add_argument('--save', type=str,
                       help='Save visualization to file')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    run_discovery(args)


if __name__ == "__main__":
    main()
