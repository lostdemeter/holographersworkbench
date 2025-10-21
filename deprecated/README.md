# Deprecated Files

This folder contains the original flat-structure files from before the 5-layer reorganization.

## Files Moved Here (October 21, 2025)

All functionality from these files has been migrated to the new `workbench/` structure:

### Original Root Files
- `__init__.py` - Old root package init (replaced by `workbench/__init__.py`)
- `utils.py` - Utilities (→ `workbench/primitives/signal.py`, `frequency.py`, `phase.py`, `kernels.py`)
- `spectral.py` - Spectral analysis (→ `workbench/processors/spectral.py`)
- `holographic.py` - Holographic processing (→ `workbench/processors/holographic.py`)
- `optimization.py` - Optimization algorithms (→ `workbench/processors/optimization.py`)
- `fast_zetas.py` - Zeta zeros (→ `workbench/core/zeta.py`)
- `quantum_clock.py` - Quantum clock (→ `workbench/core/quantum.py`)
- `fractal_peeling.py` - Fractal compression (→ `workbench/processors/compression.py`)
- `holographic_compression.py` - Holographic compression (→ `workbench/processors/compression.py`)
- `holographic_encoder.py` - Holographic encoding (→ `workbench/processors/encoding.py`)
- `ergodic_jump.py` - Ergodic analysis (→ `workbench/processors/ergodic.py`)
- `performance_profiler.py` - Performance profiling (→ `workbench/analysis/performance.py`)
- `error_pattern_visualizer.py` - Error analysis (→ `workbench/analysis/errors.py`)
- `convergence_analyzer.py` - Convergence analysis (→ `workbench/analysis/convergence.py`)
- `time_affinity.py` - Time affinity optimization (→ `workbench/analysis/affinity.py`)
- `formula_code_generator.py` - Code generation (→ `workbench/generation/code.py`)

## Purpose

These files are kept for reference and can be safely deleted after verifying the new structure works correctly.

## Migration Status

✓ All functionality has been migrated to the new structure
✓ All imports have been updated in tests and examples
✓ All verification checks pass

## Safe to Delete?

**Yes**, after you've:
1. Verified the new structure works: `python3 verify_structure.py`
2. Run tests successfully: `pytest tests/`
3. Backed up this folder outside the project (as planned)

---

**Note**: Do not import from these files. Use the new `workbench/` structure instead.
