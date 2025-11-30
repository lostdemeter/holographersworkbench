# Quick Start for AI

**Author: Lesley Gushurst** | **License: GPL-3.0** | **2025**

Fastest path for AI agents to use the Holographer's Workbench.

---

## Repository Structure

```
holographersworkbench/
├── workbench/                    # Core library (import from here)
│   ├── primitives/               # Pure functions (signal, frequency, phase)
│   ├── core/                     # Domain primitives (zeta, gushurst_crystal, clock_compiler)
│   ├── analysis/                 # Analyzers (performance, errors, convergence)
│   ├── processors/               # Transformers (spectral, holographic, clock optimizers)
│   └── generation/               # Code generation
├── practical_applications/       # Demos and applications
│   ├── showcases/                # Feature demonstrations
│   │   ├── clock_resonant_tsp/   # TSP optimizer benchmarks
│   │   ├── gushurst_crystal/     # Zeta + prime prediction
│   │   ├── clock_compiler/       # Auto clock-phase upgrades
│   │   └── dimensional_downcasting/
│   ├── ribbon_demos/             # Clock-phase diffusion demos
│   ├── ribbon_math/              # φ-BBP formula discovery
│   └── clock_downcaster/         # Clock state solver
├── tests/                        # 81 tests (smoke, workbench, practical)
└── protocols/                    # Research protocols
```

---

## Common Tasks → Imports

| Task | Import |
|------|--------|
| **Solve TSP (best)** | `from workbench.processors import solve_tsp_clock_v2` |
| **Compute zeta zeros** | `from workbench import zetazero, zetazero_batch` |
| **Predict primes** | `from workbench import GushurstCrystal` |
| **Score with zeta** | `from workbench import SpectralScorer, ZetaFiducials` |
| **Phase retrieval** | `from workbench import phase_retrieve_hilbert` |
| **Profile code** | `from workbench import PerformanceProfiler` |
| **Analyze errors** | `from workbench import ErrorPatternAnalyzer` |
| **Generate code** | `from workbench import FormulaCodeGenerator` |
| **Compress data** | `from workbench import FractalPeeler, HolographicCompressor` |
| **Extract depth** | `from workbench import HolographicDepthExtractor` |

---

## Quick Examples

### Clock-Resonant TSP (Best TSP Solver)
```python
from workbench.processors import solve_tsp_clock_v2
import numpy as np

cities = np.random.rand(50, 2) * 100
tour, length, stats = solve_tsp_clock_v2(cities)
print(f"Tour length: {length:.2f}, Resonance: {stats.resonance_strength:.3f}")
```

### Gushurst Crystal (Primes + Zeta)
```python
from workbench import GushurstCrystal

gc = GushurstCrystal(n_zeros=100, max_prime=1000)
primes = gc.predict_primes(n_primes=10)  # 100% accurate
zeros = gc.predict_zeta_zeros(n_zeros=5)
```

### Zeta Zeros (100% Perfect)
```python
from workbench import zetazero, zetazero_batch

z100 = zetazero(100)           # Single zero
zeros = zetazero_batch(1, 50)  # Batch (2.7× faster than mpmath)
```

### Spectral Scoring
```python
from workbench import SpectralScorer, ZetaFiducials

zeros = ZetaFiducials.get_standard(20)
scorer = SpectralScorer(frequencies=zeros)
scores = scorer.compute_scores(candidates)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `workbench/processors/sublinear_clock_v2.py` | Clock-Resonant TSP v2 (5.7% gap on TSPLIB) |
| `workbench/core/zeta.py` | Hybrid fractal-Newton zeta zeros |
| `workbench/core/gushurst_crystal.py` | Unified number-theoretic framework |
| `workbench/core/clock_compiler.py` | Auto-upgrade processors to clock phases |
| `workbench/analysis/performance.py` | Performance profiling |
| `workbench/analysis/errors.py` | Error pattern detection |

---

## Run Tests

```bash
python tests/smoke_test.py                    # 28 tests, ~10s
python tests/test_workbench.py                # 25 tests
python tests/test_practical_applications.py   # 28 tests
```

---

## Run Showcases

```bash
python practical_applications/showcases/clock_resonant_tsp/benchmark_v2.py
python practical_applications/showcases/gushurst_crystal/benchmark_zeta_zeros.py
python practical_applications/showcases/gushurst_crystal/benchmark_primes.py
python practical_applications/showcases/clock_compiler/demo_clock_compiler.py
```

---

## Import Pattern

```python
# Simple (recommended)
from workbench import SpectralScorer, zetazero, GushurstCrystal

# Explicit (for developers)
from workbench.processors.spectral import SpectralScorer
from workbench.core.zeta import zetazero
from workbench.core.gushurst_crystal import GushurstCrystal
```

---

## Documentation

- **README.md** - Full documentation for humans
- **AI_README.md** - Detailed API tables and math
- **ARCHITECTURE.md** - 5-layer design with mermaid diagrams

---

**© 2025 Lesley Gushurst. GPL-3.0-or-later.**
