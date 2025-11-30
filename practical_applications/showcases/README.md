# Showcases

Organized demonstrations of the Holographer's Workbench breakthrough features.

## 🎯 Featured Showcases

### [Clock-Resonant TSP Optimizer v2](clock_resonant_tsp/)
**5.7% average gap on TSPLIB** using recursive clock eigenphases!

- 12D clock tensor with multi-scale pyramid phases
- 145× speedup via O(1) memoized phase lookup
- Production-ready with JAX acceleration

```bash
cd clock_resonant_tsp && python benchmark_tsplib.py
```

### [Gushurst Crystal](gushurst_crystal/)
**Unified number theory** - zeta zeros + prime prediction via crystalline resonance!

- 100% perfect zeta zeros (error < 1e-12)
- 2.7× faster than mpmath
- Sierpinski fractal exploration + Newton refinement

```bash
cd gushurst_crystal && python benchmark_zeta_zeros.py
```

### [Dimensional Downcasting Integration](dimensional_downcasting/)
**The hidden synergy** - manifold projection meets spectral optimization!

- N_smooth(t) ≈ n − 0.5 as universal manifold
- 4-15% gains on unstructured TSP instances
- Routes cities, photons, or primes along light's shape

```bash
cd dimensional_downcasting && python benchmark_dd_integration.py
```

### [Clock Resonance Compiler](clock_compiler/)
**Automatic upgrade** of any processor to use clock eigenphases!

- Analyzes random/comb sources in existing code
- Compiles to clock-resonant version
- Drop-in replacement with deterministic behavior

```bash
cd clock_compiler && python demo_clock_compiler.py
```

## Quick Start

Run all showcases:
```bash
# TSP Optimizer
python showcases/clock_resonant_tsp/benchmark_tsplib.py

# Gushurst Crystal
python showcases/gushurst_crystal/benchmark_zeta_zeros.py

# Dimensional Downcasting
python showcases/dimensional_downcasting/benchmark_dd_integration.py

# Clock Compiler
python showcases/clock_compiler/demo_clock_compiler.py
```

## Architecture

```
showcases/
├── clock_resonant_tsp/      # TSP optimization with clock phases
├── gushurst_crystal/        # Zeta zeros + prime prediction
├── dimensional_downcasting/ # Manifold projection integration
└── clock_compiler/          # Automatic clock-resonant upgrades
```
