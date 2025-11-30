# Gushurst Crystal

**Unified number theory** - zeta zeros + prime prediction via crystalline resonance!

## Overview

The Gushurst Crystal is a unified framework combining fractal peel analysis with geometric prime sieve (spectral decomposition). It provides both fast zeta zero computation and prime prediction through crystalline resonance patterns.

## Key Features

- **Accuracy**: 100% perfect zeta zeros (error < 1e-12)
- **Speed**: 2.7× faster than mpmath
- **Innovation**: Sierpinski fractal exploration + Newton refinement
- **Unified**: Single framework for zeros and primes

## Quick Start

```python
from workbench.core import zetazero_batch, GushurstCrystal

# Fast zeta zeros - 100% perfect, 2.7× faster
zeros = zetazero_batch(1, 100)
print(f"First 100 zeros computed")

# Prime prediction via crystal resonance
gc = GushurstCrystal(n_zeros=100)
primes = gc.predict_primes(20)
print(f"Next 20 primes: {primes}")
```

## Benchmarks

### Zeta Zero Accuracy

| Method | Mean Error | Max Error | Perfect (<1e-12) |
|--------|------------|-----------|------------------|
| Pure Newton | 0.000067 | 0.003776 | 60% |
| Hybrid Fractal-Newton | 1.75e-13 | 3.4e-13 | **100%** |

### Speed Comparison

| Method | Time (100 zeros) | Speedup |
|--------|------------------|---------|
| mpmath.zetazero | 2.05s | 1× |
| Gushurst Crystal | 0.76s | **2.7×** |

### Run Benchmarks

```bash
# Zeta zeros benchmark
python benchmark_zeta_zeros.py

# Prime prediction benchmark
python benchmark_primes.py
```

## How It Works

### Phase 1: Sierpinski Fractal Exploration
- Generates candidates using Sierpinski triangle structure (Hausdorff dim 1.585)
- Self-similar recursive subdivision
- Sublinear candidate selection (√n complexity)
- Scores by |ζ(t)| / |ζ'(t)| ratio

### Phase 2: Newton Refinement
- Adaptive derivative caching (threshold 0.5)
- Starts from improved guess from Phase 1
- Rapid convergence to perfect accuracy

## Mathematical Foundation

The crystal is a 9-node extended prism lattice with:
- Vertices from zeta zeros
- Edges weighted by normalized zeta values
- Prime-power symmetries [2², 3¹, 2¹]

Key insight: `log(variance_l) ∝ -k·log(p)` reveals prime powers as natural scales of fractal decay.

## Files

- `benchmark_zeta_zeros.py` - Zeta zero accuracy and speed benchmark
- `benchmark_primes.py` - Prime prediction benchmark

## Related

- `workbench/core/gushurst_crystal.py` - Main implementation
- `workbench/core/zeta.py` - Hybrid fractal-Newton method
