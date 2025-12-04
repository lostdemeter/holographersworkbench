#!/usr/bin/env python3
"""
Demonstration of Ribbon Solver - Group Theory Edition

This script demonstrates the group-theoretic approach to mathematical discovery,
showing how the concepts from ribbon_solver3 map to group theory.

Usage:
    python demo.py
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ribbon_solver_group_theory import (
    # Core
    TruthGroup, GroupElement, AnchorVector, Anchor,
    # Processors
    CodeOptimizer, IdentityMiner, FormulaDiscovery, ErrorAnalyzer, SymmetryFinder,
    # Discovery interface
    GroupDiscovery, discover_symmetries, find_orbit, analyze
)


def demo_group_structure():
    """Demonstrate the basic group structure."""
    print("=" * 60)
    print("PART 1: GROUP STRUCTURE")
    print("=" * 60)
    
    # Create the truth group
    group = TruthGroup()
    print(f"\nTruth Group: {group}")
    
    # Show generators
    print("\nLie Algebra Generators:")
    for anchor in Anchor:
        gen = group.generator(anchor)
        print(f"  {anchor.name}: {gen.position}")
    
    # Create elements from expressions
    print("\nCreating group elements from expressions:")
    
    expressions = [
        "sin²(x) + cos²(x)",
        "arctan(tan(x))",
        "exp(log(x))",
        "φ² - φ - 1",
    ]
    
    for expr in expressions:
        g = group.element(expr)
        print(f"\n  '{expr}'")
        print(f"    Position: {g.position}")
        print(f"    Category: {group.classify(g)}")
        print(f"    Dominant anchor: {g.position.dominant_anchor()[0].name}")


def demo_group_operations():
    """Demonstrate group operations."""
    print("\n" + "=" * 60)
    print("PART 2: GROUP OPERATIONS")
    print("=" * 60)
    
    group = TruthGroup()
    
    # Get some generators
    pattern = group.generator(Anchor.PATTERN)
    growth = group.generator(Anchor.GROWTH)
    inverse = group.generator(Anchor.INVERSE)
    
    print("\nComposition (⊕):")
    composed = pattern @ growth
    print(f"  pattern ⊕ growth = {composed.position}")
    
    print("\nInverse:")
    inv = pattern.inverse()
    print(f"  pattern⁻¹ = {inv.position}")
    
    print("\nConjugation (h ⊕ g ⊕ h⁻¹):")
    conjugated = pattern.conjugate(growth)
    print(f"  growth ⊕ pattern ⊕ growth⁻¹ = {conjugated.position}")
    
    print("\nCommutator [g, h]:")
    comm = pattern.commutator(growth)
    print(f"  [pattern, growth] = {comm.position}")
    print(f"  (Non-zero commutator shows non-commutativity)")
    
    # Check identity
    print("\nIdentity check:")
    e = group.identity()
    g_inv = pattern @ pattern.inverse()
    print(f"  pattern ⊕ pattern⁻¹ ≈ identity: {g_inv.is_identity(tolerance=0.1)}")


def demo_symmetry_finding():
    """Demonstrate finding symmetries (identities)."""
    print("\n" + "=" * 60)
    print("PART 3: FINDING SYMMETRIES (IDENTITIES)")
    print("=" * 60)
    
    discovery = GroupDiscovery(verbose=False)
    
    expressions = [
        "sin²(x) + cos²(x)",
        "arctan(tan(x))",
        "exp(x)·exp(-x)",
        "cosh²(x) - sinh²(x)",
    ]
    
    for expr in expressions:
        print(f"\nExpression: {expr}")
        result = discovery.find_symmetries(expr)
        
        for finding in result.findings:
            if 'identity' in finding:
                status = "✓" if finding.get('verified', False) else "?"
                print(f"  {status} {finding['identity']}")
            elif 'description' in finding:
                print(f"  • {finding['description']}")


def demo_orbit_exploration():
    """Demonstrate orbit exploration."""
    print("\n" + "=" * 60)
    print("PART 4: ORBIT EXPLORATION")
    print("=" * 60)
    
    group = TruthGroup()
    
    # Start from exp(x)
    g = group.element("exp(x)")
    print(f"\nStarting point: exp(x)")
    print(f"  Position: {g.position}")
    
    # Compute orbit
    orbit = group.orbit(g, max_depth=2)
    print(f"\nOrbit contains {len(orbit)} elements:")
    
    for i, elem in enumerate(orbit[:5]):
        dist = elem.distance_to(g)
        print(f"  [{i}] distance={dist:.3f}, dominant={elem.position.dominant_anchor()[0].name}")
    
    if len(orbit) > 5:
        print(f"  ... and {len(orbit) - 5} more")


def demo_optimization():
    """Demonstrate optimization via conjugation."""
    print("\n" + "=" * 60)
    print("PART 5: OPTIMIZATION VIA CONJUGATION")
    print("=" * 60)
    
    discovery = GroupDiscovery(verbose=False)
    
    expressions = [
        "arctan(tan(x))",
        "exp(log(x))",
        "sin²(x) + cos²(x)",
    ]
    
    for expr in expressions:
        print(f"\nOptimizing: {expr}")
        result = discovery.optimize_via_conjugation(expr)
        
        for finding in result.findings[:2]:
            if finding['type'] == 'direct_simplification':
                print(f"  → {finding['simplified']} (verified: {finding['verified']})")
            else:
                print(f"  Conjugate by {finding['conjugate_by']}: improvement={finding['improvement']:.3f}")


def demo_subgroups():
    """Demonstrate subgroup structure."""
    print("\n" + "=" * 60)
    print("PART 6: SUBGROUP STRUCTURE")
    print("=" * 60)
    
    group = TruthGroup()
    
    print("\nSubgroups of the Truth Group:")
    for name, subgroup in group.subgroups.items():
        print(f"\n  {name.upper()} subgroup:")
        print(f"    Generators: {[g.name for g in subgroup.generators]}")
        
        # Generate some elements
        elements = subgroup.generate(max_depth=2)
        print(f"    Elements (depth 2): {len(elements)}")
    
    # Classify some expressions
    print("\nClassifying expressions:")
    expressions = [
        ("sin(2x)", "trigonometric"),
        ("exp(x)", "exponential"),
        ("x + 1", "algebraic"),
    ]
    
    for expr, expected in expressions:
        g = group.element(expr)
        actual = group.classify(g)
        match = "✓" if actual == expected else "✗"
        print(f"  {match} '{expr}' → {actual} (expected: {expected})")


def demo_quick_analysis():
    """Demonstrate quick analysis function."""
    print("\n" + "=" * 60)
    print("PART 7: QUICK ANALYSIS")
    print("=" * 60)
    
    expressions = [
        "sin²(x) + cos²(x)",
        "exp(log(x))",
        "φ² - φ - 1",
    ]
    
    for expr in expressions:
        print(f"\nAnalyzing: {expr}")
        result = analyze(expr)
        
        print(f"  Category: {result['category']}")
        print(f"  Dominant anchor: {result['dominant_anchor']}")
        print(f"  Distance from origin: {result['distance_from_origin']:.3f}")
        
        if result['optimizations']:
            for opt in result['optimizations']:
                print(f"  → Simplifies to: {opt['simplified']}")


def demo_processors():
    """Demonstrate the modular processor architecture."""
    print("\n" + "=" * 60)
    print("PART 8: MODULAR PROCESSORS")
    print("=" * 60)
    
    group = TruthGroup()
    
    print("\nAvailable Processors:")
    print("  1. CodeOptimizer - Find code optimizations")
    print("  2. IdentityMiner - Discover mathematical identities")
    print("  3. FormulaDiscovery - Find formulas for constants")
    print("  4. ErrorAnalyzer - Use error-as-signal")
    print("  5. SymmetryFinder - Find symmetries of expressions")
    
    # Demo SymmetryFinder
    print("\n--- SymmetryFinder Demo ---")
    finder = SymmetryFinder(group, verbose=False)
    result = finder.find_symmetries("sin²(x) + cos²(x)")
    print(f"  Expression: sin²(x) + cos²(x)")
    print(f"  Symmetries found: {result.symmetries_found}")
    for s in result.simplifications:
        print(f"    → {s['simplified']} (verified: {s['verified']})")
    
    # Demo CodeOptimizer
    print("\n--- CodeOptimizer Demo ---")
    optimizer = CodeOptimizer(group, verbose=False)
    result = optimizer.optimize_expressions(["arctan(tan(x))", "exp(log(x))"])
    print(f"  Expressions analyzed: {result.total_expressions}")
    print(f"  Optimizations found: {len(result.optimizations)}")
    for opt in result.optimizations:
        print(f"    {opt['original']} → {opt['optimized']}")
    
    # Demo IdentityMiner
    print("\n--- IdentityMiner Demo ---")
    miner = IdentityMiner(group, verbose=False)
    result = miner.mine_pythagorean()
    print(f"  Pythagorean identities found: {result.identities_found}")
    for identity in result.all_identities[:3]:
        print(f"    {identity['lhs']} = {identity['rhs']}")


def demo_comparison_with_ribbon_solver3():
    """Show the mapping from ribbon_solver3 to group theory."""
    print("\n" + "=" * 60)
    print("PART 9: COMPARISON WITH RIBBON_SOLVER3")
    print("=" * 60)
    
    print("""
    ribbon_solver3              →  Group Theory Edition
    ─────────────────────────────────────────────────────
    optimize_file()             →  find_symmetries()
      "Find patterns in code"      "Find invariants under group action"
    
    find_formula()              →  find_orbit()
      "Search for BBP formulas"    "Explore orbit in truth space"
    
    find_identities()           →  find_fixed_points()
      "Mine for identities"        "Find elements g where g = e"
    
    analyze()                   →  classify()
      "Analyze expression"         "Determine subgroup membership"
    
    Pattern matching            →  Symmetry detection
      "regex on expressions"       "invariance under generators"
    
    Error-as-signal             →  Lie bracket structure
      "φ-corrections"              "non-commutativity encodes info"
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("RIBBON SOLVER - GROUP THEORY EDITION")
    print("Demonstration of Group-Theoretic Mathematical Discovery")
    print("=" * 60)
    
    demo_group_structure()
    demo_group_operations()
    demo_symmetry_finding()
    demo_orbit_exploration()
    demo_optimization()
    demo_subgroups()
    demo_quick_analysis()
    demo_processors()
    demo_comparison_with_ribbon_solver3()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Truth space is a Lie group with 6 generators (anchors)")
    print("  2. Mathematical identities are symmetries (fixed points)")
    print("  3. Proof = geodesic path through the group manifold")
    print("  4. Categories = subgroups (trigonometric, exponential, etc.)")
    print("  5. Optimization = conjugation to simpler representatives")
    print()


if __name__ == "__main__":
    main()
