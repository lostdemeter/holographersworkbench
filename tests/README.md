# Tests

Automated test suite for the Holographer's Workbench.

## Test Files

### `test_workbench.py`
Comprehensive unit tests for all workbench modules.

**Run**:
```bash
python tests/test_workbench.py
```

**Tests** (16 total):
1. Module imports
2. Spectral module
3. Holographic module
4. Fractal peeling
5. Holographic compression
6. Fast zetas
7. Gushurst Crystal (unified framework)
8. Holographic encoder
9. Ergodic jump
10. Time affinity
11. Optimization
12. Utility functions
13. Performance profiler
14. Error pattern visualizer
15. Formula code generator
16. Convergence analyzer

### `test_examples.py`
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
- 6 Python example scripts
- 5 Jupyter notebooks (optional)

## Quick Test

Run both test suites:
```bash
python tests/test_workbench.py && python tests/test_examples.py --skip-notebooks
```

## CI/CD Integration

For continuous integration, use:
```bash
# Fast tests (Python only)
python tests/test_workbench.py
python tests/test_examples.py --skip-notebooks

# Full tests (including notebooks)
python tests/test_workbench.py
python tests/test_examples.py
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
