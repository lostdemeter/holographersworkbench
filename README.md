# Holographer's Workbench

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A unified toolkit for holographic signal processing, spectral optimization, and sublinear algorithms.

## Overview

The workbench consolidates techniques from multiple specialized modules into a coherent library with shared patterns and reusable components. AIs should start at AI_README.md for the quickest overview.

### Unified From

- **`sublinear_optimizer.py`** → `spectral.py`, `holographic.py`, `optimization.py`
- **`srt_auto_calibrator.py`** → `spectral.py`, `optimization.py`
- **`holo_lossless.py`** → `holographic.py`
- **`fast_zetas.py`** → `spectral.py` (ZetaFiducials)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/lostdemeter/holographerworkbench.git
   cd holographerworkbench
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Includes NumPy, SciPy, mpmath for core functionality; fast_zetas optional for faster zeta computation.)

3. For development (editable install):
   ```
   pip install -e .
   ```

4. Verify:
   ```python
   from workbench import ZetaFiducials
   print(ZetaFiducials.get_standard(5))  # Should print first 5 zeta zeros
   ```

Run demos in Jupyter/Colab: `jupyter notebook demos/` or open in [Colab](https://colab.research.google.com/github/lostdemeter/holographerworkbench/blob/main/demos/demo_5_complete_workflow.ipynb) (example for full workflow).

## Architecture

```
workbench/
├── __init__.py          # Main exports
├── spectral.py          # Frequency-domain analysis
├── holographic.py       # Phase retrieval & interference
├── optimization.py      # Sublinear algorithms & calibration
└── utils.py             # Common utilities

demos/                   # Interactive Jupyter notebooks
├── demo_1_spectral_scoring.ipynb
├── demo_2_phase_retrieval.ipynb
├── demo_3_holographic_refinement.ipynb
├── demo_4_sublinear_optimization.ipynb
├── demo_5_complete_workflow.ipynb
└── demo_6_srt_calibration.ipynb
```

## Core Modules

### 1. Spectral Module (`spectral.py`)

**Purpose**: Frequency-domain analysis and zeta-based scoring

**Key Classes**:
- `ZetaFiducials` - Unified zeta zero computation
- `SpectralScorer` - Oscillatory pattern scoring
- `DiracOperator` - SRT-style operator construction

**Example**:
```python
from workbench import SpectralScorer, ZetaFiducials

# Get zeta zeros
zeros = ZetaFiducials.get_standard(20)

# Score candidates
scorer = SpectralScorer(frequencies=zeros, damping=0.05)
scores = scorer.compute_scores(candidates, shift=0.05)
```

### 2. Holographic Module (`holographic.py`)

**Purpose**: Phase retrieval, interference patterns, signal refinement

**Key Classes**:
- `PhaseRetrieval` - Unified phase retrieval interface
- `FourPhaseShifting` - Lossless holographic encoding

**Key Functions**:
- `phase_retrieve_hilbert()` - Fast envelope extraction
- `phase_retrieve_gs()` - Gerchberg-Saxton algorithm
- `align_phase()` - Phase alignment
- `holographic_refinement()` - Signal enhancement

**Example**:
```python
from workbench import holographic_refinement, phase_retrieve_hilbert

# Extract envelope
envelope, phase_var = phase_retrieve_hilbert(signal)

# Refine scores
refined = holographic_refinement(scores, reference, method="hilbert")
```

### 3. Optimization Module (`optimization.py`)

**Purpose**: Sublinear algorithms and parameter calibration

**Key Classes**:
- `SublinearOptimizer` - O(n) → O(√n) optimization
- `SRTCalibrator` - Automated parameter tuning

**Example**:
```python
from workbench import SublinearOptimizer

optimizer = SublinearOptimizer(use_holographic=True)
top_candidates, stats = optimizer.optimize(
    candidates, score_fn, top_k=100
)

print(f"Reduced {stats.n_original} → {stats.n_final}")
print(f"Complexity: {stats.complexity_estimate}")
```

### 4. Utils Module (`utils.py`)

**Purpose**: Common utilities

**Key Functions**:
- `compute_envelope()` - Signal envelope
- `normalize_signal()` - Normalization
- `adaptive_blend()` - Adaptive signal blending
- `compute_psnr()` - Quality metrics
- `detect_peaks()` - Peak detection
- `smooth_signal()` - Signal smoothing

## Common Patterns

### Pattern 1: Spectral Scoring

Used across all modules for frequency-domain analysis:

```python
from workbench import SpectralScorer, ZetaFiducials

# Setup
frequencies = ZetaFiducials.get_standard(20)
scorer = SpectralScorer(frequencies=frequencies, damping=0.05)

# Score
scores = scorer.compute_scores(candidates, shift=0.0, mode="real")
```

### Pattern 2: Holographic Refinement

Used to extract signal from noise:

```python
from workbench import holographic_refinement

# Define object and reference
object_signal = noisy_scores
reference_signal = smooth_baseline

# Refine
refined = holographic_refinement(
    object_signal, 
    reference_signal,
    method="hilbert",
    blend_ratio=0.6
)
```

### Pattern 3: Phase Retrieval

Extract amplitude/phase information:

```python
from workbench import PhaseRetrieval

# Method 1: Hilbert (fast)
retriever = PhaseRetrieval(method="hilbert")
envelope, phase_var = retriever.retrieve(signal)

# Method 2: Gerchberg-Saxton (accurate)
retriever = PhaseRetrieval(
    method="gs",
    intensity=fourier_magnitude,
    target_amp=time_domain_amplitude
)
envelope, phase_var = retriever.retrieve(signal)
```

### Pattern 4: Sublinear Optimization

Convert O(n) to O(√n):

```python
from workbench import SublinearOptimizer

def expensive_score(candidates):
    # Your O(n) computation
    return scores

optimizer = SublinearOptimizer(use_holographic=True)
top_k, stats = optimizer.optimize(
    all_candidates,
    expensive_score,
    top_k=100
)
```

## Reusable Components

### Zeta Fiducials

All modules use the same zeta zero source:

```python
from workbench.spectral import ZetaFiducials

# Compute once, cache automatically
zeros = ZetaFiducials.get_standard(20)

# Use in multiple places
scorer = SpectralScorer(frequencies=zeros)
dirac = DiracOperator(zeta_fiducials=zeros)
```

### Phase Retrieval

Unified interface across all phase retrieval needs:

```python
from workbench.holographic import phase_retrieve_hilbert, phase_retrieve_gs

# Fast method
env1, pv1 = phase_retrieve_hilbert(signal)

# Accurate method
refined = phase_retrieve_gs(fourier_mag, target_amp, n_iter=30)
env2, pv2 = phase_retrieve_hilbert(np.real(refined))
```

### Normalization

Consistent normalization across modules:

```python
from workbench.utils import normalize_signal

# Min-max to [0, 1]
norm1 = normalize_signal(signal, method="minmax")

# Z-score standardization
norm2 = normalize_signal(signal, method="zscore")

# Max normalization
norm3 = normalize_signal(signal, method="max")
```

## Complete Example

```python
from workbench import (
    SpectralScorer,
    ZetaFiducials,
    SublinearOptimizer,
    holographic_refinement,
    normalize_signal,
)
import numpy as np

# Problem: Find top 100 candidates from 10,000
n = 10000
candidates = np.arange(n)

# Step 1: Spectral scoring
zeros = ZetaFiducials.get_standard(20)
scorer = SpectralScorer(frequencies=zeros, damping=0.05)
scores = scorer.compute_scores(candidates, shift=0.05, mode="real")

# Step 2: Define reference
reference = 1.0 / (np.log(candidates + 1) + 1e-12)

# Step 3: Holographic refinement
refined_scores = holographic_refinement(
    scores, reference,
    method="hilbert",
    blend_ratio=0.6
)

# Step 4: Sublinear optimization
optimizer = SublinearOptimizer(use_holographic=False)  # Already refined
top_100, stats = optimizer.optimize(
    candidates,
    lambda c: refined_scores[c],
    top_k=100
)

print(f"Reduced {stats.n_original} → {stats.n_final}")
print(f"Complexity: {stats.complexity_estimate}")
print(f"Top 10: {top_100[:10]}")
```

## Demos

Interactive Jupyter notebooks in `demos/` for quick prototyping. Each is self-contained, NumPy-based, and runs in <1 min. Open in Jupyter/Colab for hands-on exploration:

- **`demo_1_spectral_scoring.ipynb`**: Zeta fiducials + oscillatory scoring on candidates (top-10 output).
- **`demo_2_phase_retrieval.ipynb`**: Hilbert envelope extraction on noisy sines (phase variance checks).
- **`demo_3_holographic_refinement.ipynb`**: Refine noisy peaks; PSNR gains + accuracy.
- **`demo_4_sublinear_optimization.ipynb`**: O(√n) on 10k candidates w/ holography (stats + top-k).
- **`demo_5_complete_workflow.ipynb`**: Full pipeline: 5k → 50 candidates (end-to-end results).
- **`demo_6_srt_calibration.ipynb`**: SRT param tuning w/ mock GT (best params + metrics).

These align with common patterns—start with `demo_5_complete_workflow.ipynb` for overview.

## Migration Guide

### From `sublinear_optimizer.py`

**Before**:
```python
from sublinear_optimizer import SublinearOptimizer, SpectralScorer

optimizer = SublinearOptimizer(use_holographic=True)
scorer = SpectralScorer(frequencies, damping=0.05)
```

**After**:
```python
from workbench import SublinearOptimizer, SpectralScorer

optimizer = SublinearOptimizer(use_holographic=True)
scorer = SpectralScorer(frequencies, damping=0.05)
```

### From `srt_auto_calibrator.py`

**Before**:
```python
from srt_auto_calibrator import SRTAutoCalibrator, SRTParams

calibrator = SRTAutoCalibrator(config)
result = calibrator.calibrate()
```

**After**:
```python
from workbench import SRTCalibrator, SRTParams

calibrator = SRTCalibrator(
    ground_truth_indices,
    ground_truth_values,
    affinity_functions
)
params, metrics = calibrator.calibrate()
```

### From `holo_lossless.py`

**Before**:
```python
from holo_lossless import encode_lossless, decode_lossless

encode_lossless(img, "out.holo", quantize=False)
recon = decode_lossless("out.holo")
```

**After**:
```python
from workbench.holographic import FourPhaseShifting

encoder = FourPhaseShifting(kx=0.3, ky=0.3)
holograms = encoder.encode(img)
recon = encoder.decode(holograms)
```

## Design Principles

1. **Unified Interfaces**: Common patterns across all modules
2. **Reusable Components**: Shared utilities and base classes
3. **Clear Separation**: Spectral, holographic, optimization concerns
4. **Backward Compatible**: Easy migration from original modules
5. **Well Documented**: Examples and docstrings throughout

## Benefits

✓ **No Code Duplication**: Phase retrieval, zeta zeros, normalization unified  
✓ **Consistent API**: Same patterns everywhere  
✓ **Easy to Extend**: Add new methods to existing classes  
✓ **Better Testing**: Test once, use everywhere  
✓ **Clear Dependencies**: Know what depends on what  

## Future Extensions

- GPU acceleration for FFT operations
- Parallel batch processing
- Additional phase retrieval methods
- More spectral scoring variants
- Integration with ML frameworks

## Contributing

To add new functionality:

1. Identify which module it belongs to (spectral/holographic/optimization)
2. Check if similar functionality exists
3. Add to appropriate module with consistent API
4. Update `__init__.py` exports
5. Add examples to this README or `demos/`

---

