# Tests

Automated test suite for the Holographer's Workbench.

## Test Organization

Three test files:

### 1. `smoke_test.py` ⚡ **Fast CI/CD Smoke Test**
Quick validation that all major components work (28 tests, ~10 seconds).

```bash
python tests/smoke_test.py
```

**Coverage**: Primitives, Core, Analysis, Processors (including Clock v1/v2), Generation

### 2. `test_workbench.py` 🔬 **Comprehensive Unit Tests**
Detailed unit and integration tests for all workbench modules (25 tests).

```bash
python tests/test_workbench.py
```

**Tests**: Imports, Spectral, Holographic, Fractal Peeling, Compression, Fast Zetas, Gushurst Crystal, Zeta Benchmark, Prime Benchmark, Holographic Encoder, Ergodic Jump, Time Affinity, Optimization, Utils, Performance Profiler, Error Pattern Analyzer, Formula Code Generator, Convergence Analyzer, Quantum Folding, Chaos Seeding, Adaptive Nonlocality, Sublinear QIK, Holographic Depth, Quantum Autoencoder, Additive Error Stereo

### 3. `test_practical_applications.py` 🎯 **Comprehensive Showcase & Processor Tests**
Tests all practical applications, showcases, CLI commands, and processors (28 tests).

```bash
python tests/test_practical_applications.py
```

**Tests**:
- Core imports (workbench.core, processors, primitives, analysis)
- Clock optimizers (SublinearClockOptimizer, SublinearClockOptimizerV2, ClockResonanceCompiler, LazyClockOracle)
- All processors (SpectralScorer, HolographicEncoder, FractalPeeler, HolographicCompressor, ErgodicJump, SublinearOptimizer, AdaptiveNonlocalityOptimizer, SublinearQIK, HolographicDepthExtractor, QuantumAutoencoder, AdditiveErrorStereo)
- CLI commands (bench tsp, bench zeta)
- All showcases (TSP benchmarks, Gushurst Crystal, Dimensional Downcasting, Clock Compiler)

## Recommended Usage

### Quick Verification
```bash
python tests/smoke_test.py
```

### Full Test Suite
```bash
python tests/smoke_test.py && python tests/test_workbench.py && python tests/test_practical_applications.py
```

**Total: 81 tests across all suites**

## Requirements

- numpy
- scipy
- mpmath
- jax (optional, for clock optimizer)
