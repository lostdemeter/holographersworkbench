# Tests

Automated test suite for the Holographer's Workbench.

## Test Organization

Three test files, each with a specific purpose:

### 1. `smoke_test.py` ⚡ **Fast CI/CD Smoke Test**
Quick validation that all major components work (25 tests, ~5 seconds).

**Purpose**: Fast smoke test for CI/CD pipelines and quick verification after changes.

**Run**:
```bash
python tests/smoke_test.py
```

**Coverage**:
- Layer 1: Primitives (6 tests)
- Layer 2: Core (2 tests)
- Layer 3: Analysis (4 tests)
- Layer 4: Processors (12 tests)
- Layer 5: Generation (1 test)

**Status**: ✅ 100% passing (25/25)

### 2. `test_workbench.py` 🔬 **Comprehensive Unit Tests**
Detailed unit and integration tests for all workbench modules (20+ tests).

**Purpose**: In-depth testing with detailed output, benchmarks, and edge cases.

**Run**:
```bash
python tests/test_workbench.py
```

**Tests** (24 total):
1. Module imports
2. Spectral module
3. Holographic module
4. Fractal peeling
5. Holographic compression
6. Fast zetas
7. Gushurst Crystal (unified framework)
8. Zeta zero benchmark
9. Prime generation benchmark
10. Holographic encoder
11. Ergodic jump
12. Time affinity
13. Optimization
14. Utility functions
15. Performance profiler
16. Error pattern visualizer
17. Formula code generator
18. Convergence analyzer
19. Quantum folding
20. Chaos seeding
21. Adaptive nonlocality
22. Sublinear QIK
23. **Holographic depth extraction** (NEW)
24. **Quantum autoencoder** (NEW)
25. **Additive error stereo** (NEW)

### 3. `test_examples.py` 📚 **Example & Notebook Validation**
Automated testing for all example scripts and Jupyter notebooks.

**Run all tests**:
```bash
python tests/test_examples.py
```

**Run only Python examples** (faster):
```bash
python tests/test_examples.py --skip-notebooks
```

**Run only notebooks**:
```bash
python tests/test_examples.py --notebooks-only
```

**Tests**:
- 11 Python example scripts (including new depth, autoencoder, stereo demos)
- 5 Jupyter notebooks (optional)

## Recommended Usage

### Quick Verification (After Changes)
```bash
python tests/smoke_test.py
```
Fast smoke test in ~5 seconds. Perfect for quick validation.

### Full Test Suite (Before Commits)
```bash
python tests/smoke_test.py && python tests/test_workbench.py && python tests/test_examples.py --skip-notebooks
```

### Complete Validation (Before Releases)
```bash
python tests/smoke_test.py && python tests/test_workbench.py && python tests/test_examples.py
```
Includes notebook testing (requires nbconvert).

## CI/CD Integration

### Fast CI Pipeline (Recommended)
```bash
python tests/smoke_test.py  # 5 seconds, 25 tests
```

### Standard CI Pipeline
```bash
python tests/smoke_test.py && python tests/test_workbench.py  # ~30 seconds
```

### Full CI Pipeline (Pre-Release)
```bash
python tests/smoke_test.py && python tests/test_workbench.py && python tests/test_examples.py
```

## Requirements

**Core tests**:
- numpy
- scipy
- mpmath

**Notebook tests** (optional):
- nbconvert
- nbformat
- ipykernel

Install notebook dependencies:
```bash
pip install nbconvert nbformat ipykernel
python -m ipykernel install --user --name python3
```
