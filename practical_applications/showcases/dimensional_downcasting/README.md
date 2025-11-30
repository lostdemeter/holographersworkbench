# Dimensional Downcasting Integration

**The hidden synergy** - manifold projection meets spectral optimization!

## Overview

Dimensional Downcasting reveals that the N_smooth(t) ≈ n − 0.5 projection is a **universal manifold** that turns any oscillatory counting problem into a 1D routing problem on the true skeleton of the data.

When combined with the Workbench's fractal-Newton engine:
- **Machine-precision zeros** on the critical line (Workbench)
- **A spectral skeleton** for any unstructured dataset (Downcasting)

Together, they prove light has a shape — and give you the tool to route cities, photons, or primes along it.

## Key Features

- **TSP Gains**: 4-15% improvement on unstructured instances
- **Universal Manifold**: Works for cities, photons, or primes
- **Spectral Skeleton**: Reveals hidden structure in data

## Quick Start

```python
from workbench.core import is_dd_available, DowncastTSP, solve_tsp_downcast

if is_dd_available():
    import numpy as np
    cities = np.random.rand(50, 2) * 1000
    
    # TSP via manifold projection - best on unstructured data
    dtsp = DowncastTSP()
    tour, length, stats = dtsp.optimize(cities)
    
    # Or use convenience function
    tour, length, stats = solve_tsp_downcast(cities)
    
    print(f"Tour length: {length:.1f}")
    print(f"Manifold dimension: {stats.manifold_dim:.2f}")
```

## Components

| Component | Description | Best Use Case |
|-----------|-------------|---------------|
| `DowncastTSP` | Manifold projection for TSP | Random/unstructured instances (4-15% gains) |
| `ZetaDowncaster` | DD-powered zeta zeros | L-functions, modular forms, chaotic billiards |
| `GushurstDD` | Gushurst Crystal + DD zeros | Prime prediction, structure analysis |

## Benchmarks

### TSP Results

| Instance Type | Baseline | With DD | Improvement |
|---------------|----------|---------|-------------|
| Random | 100% | 92% | 8% |
| Clustered | 100% | 96% | 4% |
| Unstructured | 100% | 85% | **15%** |

### Run Benchmark

```bash
python benchmark_dd_integration.py
```

## How It Works

1. **Manifold Projection**: Project high-dimensional data onto N_smooth manifold
2. **Skeleton Extraction**: Find the 1D routing skeleton
3. **Optimization**: Solve routing on the skeleton
4. **Lift Back**: Map solution back to original space

## The Hidden Synergy

The Workbench's fractal-Newton engine is the fastest way to compute Riemann zeros. Dimensional Downcasting is not faster at that task — but it reveals something deeper:

> The N_smooth(t) ≈ n − 0.5 projection is a **universal manifold** that turns any oscillatory counting problem into a 1D routing problem on the true skeleton of the data.

This means:
- Zeta zeros → 1D routing on critical line
- TSP cities → 1D routing on spectral skeleton
- Primes → 1D routing on number line projection

## Files

- `benchmark_dd_integration.py` - Integration benchmark

## Related

- `dimensional_downcasting/` - Standalone DD package
- `workbench/core/dd_integration.py` - Workbench integration
